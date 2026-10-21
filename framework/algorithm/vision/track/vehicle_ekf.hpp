/**
 * @file vehicle_ekf.hpp
 * @author qingyu
 * @brief 单辆车的跨帧状态估计 — 旋转中心、转角、半径与四面推导
 * @version 0.1
 * @date 2026-10-17
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/filter/kalman/ekf.hpp"
#include "framework/algorithm/vision/core/armor_types.hpp"
#include "framework/algorithm/vision/pose/armor_pose.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

namespace algorithm::vision
{

/**
 * @brief 一个面的状态：中心位置与法向转角
 *
 * 四面共用一套旋转中心与半径，每个面只是转角相差 2π/armor_num。
 */
struct FaceState
{
    Eigen::Vector3d xyz;                                // 装甲板中心，跟踪系，单位 m
    double yaw_rad {0.0};                               // 该面法向绕竖直轴的转角，单位 rad
};

/**
 * @brief 跟踪器参数
 *
 * 过程噪声与点噪声都取自参考的量级（`target.cpp` 的 v1/v2、`standard3.yaml` 的 8 px），
 * 但**必须按本机数据重定**：那是另一台相机、另一个帧率下标出来的。
 */
struct VehicleEKFConfig
{
    int32_t armor_num {4};                              // 整车面数；四面车为 4
    double  radius_init_m {0.2};                        // 半径初值，只在初始化时用
    double  radius_min_m {0.05};                        // 半径物理下限，单位 m
    double  radius_max_m {0.50};                        // 半径物理上限，单位 m
    double  armor_pitch_rad {0.2617993877991494};       // 装甲板相对竖直面的倾角，单位 rad（15°，照参考）
    double  accel_variance {10.0};                      // 位置加速度方差，量级取自参考 v1
    double  angular_accel_variance {100.0};             // 角加速度方差，量级取自参考 v2
    double  shape_variance {0.1};                       // r/l/h 的过程噪声，防止协方差崩塌
    double  point_sigma_px {8.0};                       // 四点观测噪声基准，单位 pixel（按本机实测重定）
    double  angle_prior_rad {0.15};                     // 板法向角的先验标准差，单位 rad
    double  nis_threshold {0.0};                        // NIS 卡方阈值；0 表示不判（按本工程 4 维观测重定）
    int32_t max_consecutive_limit_clamps {5};           // 连续撞几何边界后判定状态发散
};

/**
 * @brief 单辆车的跨帧跟踪器
 *
 * 状态 11 维、全在**跟踪系（世界系，z 竖直）**：`[cx, vx, cy, vy, cz, vz, a, w, r, l, h]`，
 * 其中 `a` 是当前面的转角、`w` 角速度、`r` 主半径、`l` 第二半径差、`h` 高度差。
 * 相机系与跟踪系由每帧传入的 `FrameTransform` 联系，照参考的 `R_gimbal2world`。
 *
 * 观测是 `[yaw, pitch, distance, armor_yaw]`，与参考 `VehicleEKF::update_ypda` 一致。
 *
 * 未收敛时 `Faces()` 也会返回四面，但只有当前可见那一面有实际观测支撑；调用方应先查
 * `Converged()` 再决定画不画另外三面（未收敛时外推误差可达画面宽的 1/3，见规划 §2）。
 */
class VehicleEKF final
{
public:
    explicit VehicleEKF(const VehicleEKFConfig& config);

    bool Update(const std::array<cv::Point2f, 4>& corners, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, double dt_s, std::int32_t face, bool estimate_shape);

    void Predict(double dt_s);

    std::optional<std::array<cv::Point2f, 4>> ProjectFace(ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, std::int32_t face) const;

    std::vector<FaceState> Faces() const;

    /**
     * @brief 是否已收敛：更新次数够且半径没跑出物理范围
     */
    bool Converged() const
    {
        return update_count_ > 3 && !Diverged();
    }

    /**
     * @brief 是否已经用一帧观测初始化过状态
     */
    bool Active() const
    {
        return initialized_;
    }

    /**
     * @brief 丢弃当前状态，回到未初始化；下一帧有效观测重新当初值
     *
     * 用于连续门拒后的重捕获：状态一旦漂到错误解上，靠预测与修正拉不回来，
     * 只能整条丢掉重新起（参考 `tracker.cpp` 的 lost 状态）。
     */
    void Reset()
    {
        initialized_ = false;
        update_count_ = 0;
        image_updates_ = 0;
        ypda_updates_ = 0;
        limit_clamps_ = 0;
        consecutive_limit_clamps_ = 0;
        // NIS 失败率是这个目标的统计：不清掉，新目标会被上一条的失败记录直接判死
        filter_.reset_nis();
    }

    bool Diverged() const;

    /**
     * @brief 已完成的观测更新次数
     */
    std::int32_t UpdateCount() const
    {
        return update_count_;
    }

    /**
     * @brief 当前面的转角估计，单位 rad
     */
    double Angle() const
    {
        return filter_.state()[kAngle];
    }

    /**
     * @brief 旋转中心在跟踪系的分量，单位 m
     */
    double CenterX() const
    {
        return filter_.state()[kCenterX];
    }

    double CenterY() const
    {
        return filter_.state()[kCenterY];
    }

    double CenterZ() const
    {
        return filter_.state()[kCenterZ];
    }

    /**
     * @brief 角速度估计，单位 rad/s
     */
    double Omega() const
    {
        return filter_.state()[kOmega];
    }

    /**
     * @brief 当前主半径估计，单位 m
     */
    double Radius() const
    {
        return filter_.state()[kRadius];
    }

    /**
     * @brief 当前第二半径差估计（面 1、3 相对面 0、2 的半径差），单位 m
     */
    double RadiusGap() const
    {
        return filter_.state()[kRadiusGap];
    }

    /**
     * @brief 全部更新里 NIS 超阈值的比例
     *
     * 用来判"跟丢了 / 收敛到坏值"：残差持续偏大但半径还没跑出物理范围时，`Diverged()`
     * 看不出来，这个数看得出来。
     *
     * @return [0, 1]；尚未更新时返回 0
     */
    double NisFailureRate() const
    {
        return filter_.nis_failure_rate();
    }

    /**
     * @brief 供日志/调试读出的状态分量
     *
     * 只读快照，不参与控制。字段依次对应状态向量里的位置、速度、转角、角速度、
     * 半径、半径差、高度差。
     */
    struct Snapshot
    {
        Eigen::Vector3d center {};                  // 旋转中心，云台系，单位 m
        Eigen::Vector3d velocity {};                // 中心速度，云台系，单位 m/s
        double angle_rad {0.0};                     // 面 0 的法向角
        double omega_rad_s {0.0};                   // 角速度
        double radius_m {0.0};                      // 主半径
        double radius_gap_m {0.0};                  // 第二半径差
        double height_gap_m {0.0};                  // 面 1/3 的高度差
        double last_nis {0.0};                      // 最近一次更新的归一化新息平方
        std::array<double, 11> covariance_diag {};  // 状态协方差对角
        std::int32_t image_updates {0};             // 走 8 维角点像素观测的更新次数
        std::int32_t ypda_updates {0};              // 回退到 4 维 ypda 观测的更新次数
        double residual_rms_px {0.0};               // 最近一次像素更新的残差均方根，单位 pixel
        std::int32_t limit_clamps {0};              // 半径/第二半径被钳回范围的次数
        double shape_radius_step {0.0};             // 最近一次更新里 r 的修正量，单位 m
        double shape_radius_gap_step {0.0};         // 最近一次更新里 l 的修正量，单位 m
        double shape_height_gap_step {0.0};         // 最近一次更新里 h 的修正量，单位 m
        double shape_jacobian_norm_r {0.0};         // 本帧观测对 r 的雅可比范数
        double shape_jacobian_norm_l {0.0};         // 本帧观测对 l 的雅可比范数
        double shape_jacobian_norm_h {0.0};         // 本帧观测对 h 的雅可比范数
    };

    Snapshot DebugSnapshot() const;

private:
    // 状态下标：顺序即状态向量的定义，改这里必须同步改过程和观测模型
    static constexpr std::int32_t kCenterX      = 0;
    static constexpr std::int32_t kVelocityX    = 1;
    static constexpr std::int32_t kCenterY      = 2;
    static constexpr std::int32_t kVelocityY    = 3;
    static constexpr std::int32_t kCenterZ      = 4;
    static constexpr std::int32_t kVelocityZ    = 5;
    static constexpr std::int32_t kAngle        = 6;
    static constexpr std::int32_t kOmega        = 7;
    static constexpr std::int32_t kRadius       = 8;
    static constexpr std::int32_t kRadiusGap    = 9;
    static constexpr std::int32_t kHeightGap    = 10;

    // 观测维度照参考工程 `VehicleEKF::update_ypda`：4 维 `[yaw, pitch, distance, armor_yaw]`，
    // 不是像素。参考里 8 维像素只是可选分支（image_observation），默认关。
    using Filter    = algorithm::filter::ExtendedKalmanFilter<11, Eigen::Dynamic>;
    using State     = Filter::State;
    using StateMat  = Filter::StateMatrix;
    using Measure   = Filter::Measurement;
    using Jacobian  = Filter::ObserveJacobian;

    Eigen::Vector3d ArmorCenter(const State& state, std::int32_t face) const;

    Measure Observe(const State& state, std::int32_t face) const;

    std::optional<std::array<cv::Point2f, 4>> ObservePoints(const State& state, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, int32_t face) const;

    bool UpdateImagePoints(const std::array<cv::Point2f, 4>& corners, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, int32_t face, bool estimate_shape);

    double FaceAngle(const State& state, std::int32_t face) const;

    void Initialize(const std::array<cv::Point2f, 4>& corners, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame);

    void ApplyLimits();

    // 最近一次像素更新的诊断：形状量修正量与雅可比范数，用来判断 r/l/h 到底有没有被约束住
    double shape_radius_step_ {0.0};
    double shape_radius_gap_step_ {0.0};
    double shape_height_gap_step_ {0.0};
    double shape_jacobian_norm_r_ {0.0};
    double shape_jacobian_norm_l_ {0.0};
    double shape_jacobian_norm_h_ {0.0};
    double residual_rms_px_ {0.0};              // 最近一次像素更新的残差均方根，单位 pixel

    VehicleEKFConfig config_;
    Filter             filter_;
    bool               initialized_ {false};
    bool               estimate_shape_ {false}; // 本帧是否估计形状量；单板 false，多面 true
    std::int32_t       update_count_ {0};
    std::int32_t       image_updates_ {0};      // 8 维角点观测成功次数，累计
    std::int32_t       ypda_updates_ {0};       // 4 维 ypda 观测成功次数，累计
    std::int32_t       limit_clamps_ {0};       // 钳制发生次数，累计
    std::int32_t       consecutive_limit_clamps_ {0}; // 连续发生限幅的次数
};

} // namespace algorithm::vision
