/**
 * @file tracker.hpp
 * @author qingyu
 * @brief 单目标跟踪 — 选板、像素四门关联与生命周期状态机
 * @version 0.2
 * @date 2026-10-17
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/detections.hpp"
#include "framework/algorithm/vision/track/association.hpp"
#include "framework/algorithm/vision/track/vehicle_ekf.hpp"

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

namespace algorithm::vision
{

/**
 * @brief 跟踪器的生命周期状态
 *
 * 照参考 `tracker.hpp`：`lost → detecting → tracking → temp_lost → lost`。
 * 只有 `tracking` 对外可用；`lost` 下每帧都允许用当帧观测重新建目标。
 */
enum class TrackerState : std::uint8_t
{
    Lost,                                           // 无目标，允许当帧重新初始化
    Detecting,                                      // 已建目标，连续确认中
    Tracking,                                       // 稳定跟踪
    TempLost                                        // 暂时丢失，靠预测维持
};

/**
 * @brief 判丢重建的原因，供日志定位
 */
enum class ResetReason : std::uint8_t
{
    None,                                           // 本周期没有发生重建
    Nis,                                            // NIS 失败率超过阈值
    Diverged,                                       // 半径或第二半径跑出物理范围
    InitFailed,                                     // lost 下当帧观测建不起目标
    Manual                                          // 外部要求强制重置（松开自瞄键）
};

/**
 * @brief 跟踪器参数
 */
struct TrackerConfig
{
    VehicleEKFConfig vehicle {};                    // 单个目标的滤波器参数
    AssociationConfig  association {};              // 四门关联参数
    int32_t min_detect_count {5};                   // detecting 升 tracking 的连续确认帧数
    int32_t max_temp_lost_count {25};               // temp_lost 回 lost 的连续丢帧数
    double nis_failure_ratio {0.8};                 // NIS 失败率超过它就判丢（参考 100 帧里 80 次）

    // 选面打分：score = w_center·(d/fx)² + w_cost·代价² + w_angle·夹角² +（非当前面时）换面损失
    double aim_weight_center {1.0};                 // 检测框中心到画面中心的归一化距离
    double aim_weight_cost {1.0};                   // 这对匹配的四门代价
    double aim_weight_angle {1.0};                  // 板法向与视线的夹角
    double aim_switch_cost {1.0};                   // 换面损失，量纲与关联代价相同（1.0 ≈ 一次刚好卡在门限上的关联）
};

/**
 * @brief 跟踪器运行统计，仅供日志与调试
 */
struct TrackerReport
{
    bool active {false};                                // 是否已建过目标
    bool converged {false};                             // 是否处于 tracking 且滤波器收敛
    int32_t updates {0};                                // 累计观测更新次数
    int32_t losses  {0};                                // 累计未匹配帧数
    int32_t gate_rejects {0};                           // 累计全部候选被门拒的帧数
    int32_t resets  {0};                                // 累计判丢重建次数
    ResetReason  last_reset_reason {ResetReason::None}; // 最近一次重建的原因
    int32_t last_face {-1};                             // 最近一次匹配到的面序，-1 表示本帧没更新
    double  last_cost {0.0};                            // 最近一次匹配的最小代价
    std::vector<AssociationDiagnostic> diagnostics;     // 最近一帧每次配对尝试的分项误差，按归一化代价升序
    std::size_t candidate_count {0};                    // 最近一帧同编号同尺寸的候选板数
    std::size_t match_count {0};                        // 最近一帧实际配对成功的数量
    TrackerState state {TrackerState::Lost};            // 生命周期状态
    int32_t detect_count {0};                           // detecting 阶段的连续确认帧数
    int32_t temp_lost_count {0};                        // temp_lost 阶段的连续丢失帧数
};

/**
 * @brief 单辆车的跟踪器
 *
 * 结构照参考 `Tracker`：只持有一个 `VehicleEKF`（目标），没有槽位表。每帧先按状态决定
 * 建目标还是更新目标；更新时按像素四门把检测板与预测面做一对一最优配对，每个配对各更新一次 EKF。
 *
 * 该类只做调度与关联，不做几何计算：观测模型、状态约束都在 `VehicleEKF` 里。
 */
class Tracker final
{
public:
    /**
     * @brief 建立跟踪器，初始处于 lost
     */
    explicit Tracker(const TrackerConfig& config) : config_(config), vehicle_(config.vehicle)
    {
    }

    void ProcessFrame(const std::vector<DetectedArmor>& armors, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, double dt_s);

    /**
     * @brief 丢弃当前目标并回到 lost
     *
     * 下一帧处于 lost，会重新按 `FindLockArmor` 选板锁定，可以换成另一辆车。
     * 本来就是 lost 时不再重复计数，松键期间每帧调用也只算一次重建。
     *
     * @param reason 记录到 `last_reset_reason` 的原因
     */
    void Reset(ResetReason reason)
    {
        if (report_.state == TrackerState::Lost)
        {
            return;
        }
        ResetVehicle(reason);
    }

    /**
     * @brief 只按 dt 推进状态，不修正
     */
    void Predict(double dt_s)
    {
        vehicle_.Predict(dt_s);
    }

    /**
     * @brief 是否处于 tracking 且滤波器收敛
     */
    bool Converged() const
    {
        return report_.state == TrackerState::Tracking && vehicle_.Converged();
    }

    /**
     * @brief 是否已经建过目标
     */
    bool Active() const
    {
        return report_.state != TrackerState::Lost && vehicle_.Active();
    }

    /**
     * @brief 目标四面的中心与转角；未建目标时返回空
     */
    std::vector<FaceState> Faces() const
    {
        return vehicle_.Faces();
    }

    /**
     * @brief 目标主半径估计，单位 m；未建目标时为 0
     */
    double Radius() const
    {
        return vehicle_.Radius();
    }

    /**
     * @brief 目标最近一次观测到的装甲板大小
     */
    ArmorSize Size() const
    {
        return size_;
    }

    /**
     * @brief 锁定时的机器人编号，用于核对候选筛选
     */
    ArmorLabel Label() const
    {
        return label_;
    }

    /**
     * @brief 当前 NIS 失败率，供日志判断离判丢还有多远
     */
    double NisFailureRate() const
    {
        return vehicle_.NisFailureRate();
    }

    /**
     * @brief 滤波器状态快照，供日志与调试
     */
    VehicleEKF::Snapshot DebugSnapshot() const
    {
        return vehicle_.DebugSnapshot();
    }

    /**
     * @brief 当前运行统计
     */
    const TrackerReport& Report() const
    {
        return report_;
    }

private:
    const DetectedArmor* FindLockArmor(const std::vector<DetectedArmor>& armors, const CameraModel& camera) const;

    bool LockVehicle(const DetectedArmor& armor, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame);

    bool UpdateVehicle(const std::vector<DetectedArmor>& armors, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, double dt_s);

    void AdvanceState(bool found);

    void ResetVehicle(ResetReason reason);

    std::int32_t SelectAimMatch(const std::vector<ArmorMatch>& matches, const ArmorDimensions& dimensions, const CameraModel& camera) const;

    double AimMatchScore(const ArmorMatch& match, const ArmorDimensions& dimensions, const CameraModel& camera) const;

    TrackerConfig config_;
    VehicleEKF  vehicle_;
    TrackerReport report_;

    ArmorLabel label_ {ArmorLabel::NotArmor};       // 目标的机器人编号
    ArmorSize  size_ {ArmorSize::Unknown};          // 目标的装甲板大小
};

} // namespace algorithm::vision
