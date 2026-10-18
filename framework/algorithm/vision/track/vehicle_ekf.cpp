/**
 * @file vehicle_ekf.cpp
 * @author qingyu
 * @brief 单辆车跨帧状态估计实现 — 过程模型、像素四点观测与四面推导
 * @version 0.1
 * @date 2026-10-17
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/track/vehicle_ekf.hpp"

#include "framework/algorithm/math/angle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <opencv2/calib3d.hpp>

namespace algorithm::vision
{

namespace
{

/**
 * @brief 数值雅可比的扰动步长
 *
 * 位置类状态用 1e-4 m、角度用 1e-4 rad：与参考 `target.cpp` 的中心差分同量级，
 * 小到不引入可观的非线性误差，大到不会让差分被双精度舍入吃掉。
 */
constexpr double kPositionStepM = 1.0e-4;
constexpr double kAngleStepRad  = 1.0e-4;

} // namespace

/**
 * @brief 建立跟踪器，状态清零直到第一次 Update
 */
VehicleEKF::VehicleEKF(const VehicleEKFConfig& config) : config_(config)
{
    State state = State::Zero();
    state[kRadius] = config_.radius_init_m;

    // 角度分量在每步加法后包回 (−π, π]：状态里有转角，平移与修正都可能让它越过 ±π。
    // 这个钩子挂在滤波器内部后，预测与修正两条路径都不会漏掉归一化。
    filter_.set_add([](const State& first, const State& second)
    {
        State sum = first + second;
        sum[kAngle] = algorithm::math::normalize_radian_pm_pi(sum[kAngle]);
        return sum;
    });

    // 观测里的 yaw / pitch / armor_yaw 都要按最短角差相减，否则 ±π 附近会差一整圈
    filter_.set_subtract([](const Measure& first, const Measure& second)
    {
        Measure difference = first - second;
        difference[0] = algorithm::math::normalize_radian_pm_pi(difference[0]);
        difference[1] = algorithm::math::normalize_radian_pm_pi(difference[1]);
        difference[3] = algorithm::math::normalize_radian_pm_pi(difference[3]);
        return difference;
    });

    filter_.set_nis_threshold(config_.nis_threshold);
    filter_.set_state(state, StateMat::Identity());
}

/**
 * @brief 用一帧的四点观测推进并修正状态
 *
 * 首次调用会把状态初始化到这一帧的位姿上（半径取配置初值），之后的调用都是
 * 先按 dt 预测、再修正。
 *
 * @param corners 图像四点，顺序 LT、RT、RB、LB
 * @param size 装甲板大小，决定物点宽度
 * @param dimensions 实物尺寸，单位 m
 * @param camera 内参与畸变
 * @param dt_s 距上一次更新的时间，单位 s；首次调用忽略
 * @param face 这一帧看到的是第几面；四面车的面 1、3 用第二半径与高度差
 * @param estimate_shape 本帧是否把 r/l/h 放进观测；单板必须为 false，多面才为 true
 * @return true 状态可用；false 观测无效、面序越界或修正后状态发散
 */
bool VehicleEKF::Update(const std::array<cv::Point2f, 4>& corners, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, double dt_s, std::int32_t face, bool estimate_shape)
{
    if (face < 0 || face >= config_.armor_num) {
        return false;
    }

    estimate_shape_ = estimate_shape;

    const std::optional<std::array<cv::Point3f, 4>> object_points = MakeObjectPoints(size, dimensions);
    if (!object_points) {
        return false;
    }

    for (const cv::Point2f& point : corners)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
    }

    if (!initialized_)
    {
        // 初始化内部会解一次 PnP；解不出来说明这帧观测本身不可用
        Initialize(corners, size, dimensions, camera, frame);
        if (!initialized_) {
            return false;
        }
        ++update_count_;

        return true;
    }

    Predict(dt_s);

    // 照参考：优先用 8 维角点像素观测，投影或雅可比失败再回退 PnP 的 ypda
    if (UpdateImagePoints(corners, size, dimensions, camera, frame, face, estimate_shape))
    {
        if (estimate_shape)
        {
            ApplyLimits();
        }

        if (!filter_.state().allFinite() || !filter_.covariance().allFinite())
        {
            initialized_ = false;
            return false;
        }

        ++image_updates_;
        ++update_count_;
        return true;
    }

    // 观测量照参考：先用 PnP 解出这块板的位姿，取板心坐标与该面法向角
    const std::optional<ArmorPose> pose = SolvePose(corners, *MakeObjectPoints(size, dimensions), camera);
    if (!pose)
    {
        return false;
    }

    cv::Matx33d rotation;
    cv::Rodrigues(cv::Mat(pose->rvec), rotation);

    const Eigen::Vector3d camera_normal(rotation(0, 0), rotation(1, 0), rotation(2, 0));
    const Eigen::Vector3d world_normal    = frame.rotation * camera_normal;
    const Eigen::Vector3d observed_center = CameraToFrame(Eigen::Vector3d(pose->tvec[0], pose->tvec[1], pose->tvec[2]), frame);
    
    const double observed_yaw = OptimizeYaw(corners, observed_center, size, dimensions, frame, camera, std::atan2(world_normal.y(), world_normal.x()));
    const Eigen::Vector3d ypd = XyzToYpd(observed_center);

    Measure measured(4);
    measured << ypd.x(), ypd.y(), ypd.z(), observed_yaw;

    if (!measured.allFinite())
    {
        return false;
    }

    // 观测雅可比用中心差分数值求（照参考 `target.cpp` 的做法），省去手推解析式。
    const State x = filter_.state();
    Jacobian jacobian = Jacobian::Zero(4, 11);
    std::vector<std::int32_t> observed_states = {kCenterX, kCenterY, kCenterZ, kAngle};
    if (estimate_shape)
    {
        observed_states.insert(observed_states.end(), {kRadius, kRadiusGap, kHeightGap});
    }

    for (const std::int32_t index : observed_states)
    {
        const double step = (index == kAngle) ? kAngleStepRad : kPositionStepM;
        State plus    = x;
        State minus   = x;
        plus[index]  += step;
        minus[index] -= step;

        if (index == kAngle)
        {
            plus[index]  = algorithm::math::normalize_radian_pm_pi(plus[index]);
            minus[index] = algorithm::math::normalize_radian_pm_pi(minus[index]);
        }

        const Measure forward  = Observe(plus,  face);
        const Measure backward = Observe(minus, face);
        if (!forward.allFinite() || !backward.allFinite())
        {
            return false;
        }
        jacobian.col(index) = (forward - backward) / (2.0 * step);
    }

    // 测量噪声照参考 `update_ypda` 的 R_dig：
    // {4e-3, 4e-3, log(|delta_angle|+1)+1, log(|pitch|+1)/200 + 1e-2}
    // 第三项给距离、第四项给角度；方位角偏差越大越不信这帧的角度观测。
    const double center_yaw = std::atan2(observed_center.y(), observed_center.x());
    const double delta_angle = algorithm::math::normalize_radian_pm_pi(observed_yaw - center_yaw);

    // 测量噪声优先用像素噪声一阶传播；传播失败时回退参考的内置对角公式
    Filter::MeasureMatrix measurement_noise = Filter::MeasureMatrix::Zero(4, 4);
    Eigen::Matrix3d armor_to_camera;
    for (std::int32_t row = 0; row < 3; ++row)
    {
        for (std::int32_t col = 0; col < 3; ++col)
        {
            armor_to_camera(row, col) = rotation(row, col);
        }
    }
    const std::optional<Eigen::Matrix4d> propagated = YpdaMeasurementCovariance(armor_to_camera, Eigen::Vector3d(pose->tvec[0], pose->tvec[1], pose->tvec[2]), size, dimensions, frame, camera, config_.point_sigma_px, config_.angle_prior_rad);
    if (propagated)
    {
        measurement_noise = *propagated;
    }
    else
    {
        measurement_noise(0, 0) = 4e-3;
        measurement_noise(1, 1) = 4e-3;
        measurement_noise(2, 2) = std::log(std::abs(delta_angle) + 1.0) + 1.0;
        measurement_noise(3, 3) = std::log(std::abs(ypd.z()) + 1.0) / 200.0 + 1e-2;
    }
    filter_.set_measurement_noise(measurement_noise);

    (void)filter_.correct(measured, [this, face](const State& state)
    {
        return Observe(state, face);
    }, jacobian);

    if (estimate_shape)
    {
        ApplyLimits();
    }

    if (!filter_.state().allFinite() || !filter_.covariance().allFinite())
    {
        initialized_ = false;
        return false;
    }

    ++ypda_updates_;
    ++update_count_;
    return true;
}

/**
 * @brief 只按 dt 推进状态，不修正
 *
 * 过程模型是位置匀加速 + 转角匀速；`r/l/h` 只加一点噪声防止协方差长期崩塌（照参考）。
 * 角度在传播后包回 (−π, π]：本工程的 EKF 没有角度归一化，只能在这里手动补。
 */
void VehicleEKF::Predict(double dt_s)
{
    if (!initialized_)
    {
        return;
    }

    const double dt = dt_s;

    StateMat transition = StateMat::Identity();
    transition(kCenterX, kVelocityX) = dt;
    transition(kCenterY, kVelocityY) = dt;
    transition(kCenterZ, kVelocityZ) = dt;
    transition(kAngle,   kOmega)     = dt;

    // 逐段白噪声：位置与角度的匀加速模型
    const double a = dt * dt * dt * dt / 4.0;
    const double b = dt * dt * dt / 2.0;
    const double c = dt * dt;

    StateMat noise = StateMat::Zero();
    const double v1 = config_.accel_variance;
    const double v2 = config_.angular_accel_variance;
    
    noise(kCenterX,   kCenterX) = a * v1; noise(kCenterX,   kVelocityX) = b * v1;
    noise(kVelocityX, kCenterX) = b * v1; noise(kVelocityX, kVelocityX) = c * v1;
    noise(kCenterY,   kCenterY) = a * v1; noise(kCenterY,   kVelocityY) = b * v1;
    noise(kVelocityY, kCenterY) = b * v1; noise(kVelocityY, kVelocityY) = c * v1;
    noise(kCenterZ,   kCenterZ) = a * v1; noise(kCenterZ,   kVelocityZ) = b * v1;
    noise(kVelocityZ, kCenterZ) = b * v1; noise(kVelocityZ, kVelocityZ) = c * v1;
    noise(kAngle,     kAngle)   = a * v2; noise(kAngle,     kOmega)     = b * v2;
    noise(kOmega,     kAngle)   = b * v2; noise(kOmega,     kOmega)     = c * v2;

    // 形状量只给很小的过程噪声，否则半径会被噪声推着乱走
    const double shape_variance = estimate_shape_ ? config_.shape_variance : 0.0;
    noise(kRadius,    kRadius)    = shape_variance * c;
    noise(kRadiusGap, kRadiusGap) = shape_variance * c;
    noise(kHeightGap, kHeightGap) = shape_variance * c;

    // Q 必须在 predict 之前写入：predict 内部会读 q_
    filter_.set_process_noise(noise);

    // 角度归一化由滤波器的加法钩子负责，这里不再手动包回 ±π
    (void)filter_.predict([dt](const State& x)
    {
        State next = x;
        next[kCenterX] += x[kVelocityX] * dt;
        next[kCenterY] += x[kVelocityY] * dt;
        next[kCenterZ] += x[kVelocityZ] * dt;
        next[kAngle]   += x[kOmega]     * dt;
        return next;
    }, transition);
}

/**
 * @brief 由状态推出某个面的转角
 */
double VehicleEKF::FaceAngle(const State& state, std::int32_t face) const
{
    const double step = 2.0 * M_PI / static_cast<double>(config_.armor_num);
    return algorithm::math::normalize_radian_pm_pi(state[kAngle] + static_cast<double>(face) * step);
}

/**
 * @brief 某个面的中心位置，在云台系下
 *
 * 车的旋转绕云台系竖直轴（y 向下，水平面是 x-z）：面心 = 旋转中心 − r·法向，法向
 * `n = (cos a, 0, sin a)`（与 ArmorRotation 同一约定）。高度差沿 y。
 * 只有四面车的面 1、3 用第二半径与高度差（照参考 `target.cpp` 的 `h_armor_xyz`）。
 */
Eigen::Vector3d VehicleEKF::ArmorCenter(const State& state, std::int32_t face) const
{
    const double angle = FaceAngle(state, face);
    const bool use_second = config_.armor_num == 4 && (face == 1 || face == 3);

    const double radius = use_second ? state[kRadius] + state[kRadiusGap] : state[kRadius];

    return Eigen::Vector3d(state[kCenterX] - radius * std::cos(angle), state[kCenterY] - radius * std::sin(angle), state[kCenterZ] + (use_second ? state[kHeightGap] : 0.0));
}

/**
 * @brief 状态到像素四点的观测模型
 *
 * 用位姿把板系物点反投回像素。物点由板尺寸决定，所以观测函数需要 size 与 dimensions；
 * 二者是常量参数，不参与状态，也不进雅可比。状态在云台系，投影要先把面心换到相机系；
 * 外参默认单位阵时这一步是恒等。
 *
 * `face` 决定的是"这一帧看到的是车上哪一面"：四面车的面 1、3 用第二半径与高度差，
 * 所以同一个状态按不同的面投出来的四点不同——上层正是靠这一点分辨面序。
 */
VehicleEKF::Measure VehicleEKF::Observe(const State& state, std::int32_t face) const
{
    // 照参考 `VehicleEKF::update_ypda`：观测量是 `[yaw, pitch, distance, armor_yaw]`。
    Measure observed(4);

    const Eigen::Vector3d ypd = XyzToYpd(ArmorCenter(state, face));
    observed[0] = ypd.x();
    observed[1] = ypd.y();
    observed[2] = ypd.z();
    observed[3] = FaceAngle(state, face);
    return observed;
}

/**
 * @brief 状态到像素四点的观测模型（8 维）
 */
std::optional<std::array<cv::Point2f, 4>> VehicleEKF::ObservePoints(const State& state, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, std::int32_t face) const
{
    return ReprojectArmor(ArmorCenter(state, face), FaceAngle(state, face), size, dimensions, config_.armor_pitch_rad, frame, camera);
}

/**
 * @brief 用角点像素观测更新
 *
 * 照参考 `VehicleEKF::update_image_points`：观测是 8 维像素，雅可比只对位置、角度与形状量
 * 做中心差分；噪声是各向同性的 `σ_px²`。
 */
bool VehicleEKF::UpdateImagePoints(const std::array<cv::Point2f, 4>& corners, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, std::int32_t face, bool estimate_shape)
{
    constexpr double kPositionStepM = 1e-3;
    constexpr double kAngleStepRad = 1e-3;

    const std::optional<std::array<cv::Point2f, 4>> predicted = ObservePoints(filter_.state(), size, dimensions, camera, frame, face);
    if (!predicted)
    {
        return false;
    }

    Measure measured(8);
    for (std::size_t index = 0; index < corners.size(); ++index)
    {
        if (!std::isfinite(corners[index].x) || !std::isfinite(corners[index].y))
        {
            return false;
        }
        measured[2 * index] = corners[index].x;
        measured[2 * index + 1] = corners[index].y;
    }

    const State x = filter_.state();
    Jacobian jacobian = Jacobian::Zero(8, 11);
    std::vector<std::int32_t> observed_states = {kCenterX, kCenterY, kCenterZ, kAngle};
    if (estimate_shape)
    {
        observed_states.insert(observed_states.end(), {kRadius, kRadiusGap, kHeightGap});
    }

    for (const int32_t state_index : observed_states)
    {
        const double step = (state_index == kAngle) ? kAngleStepRad : kPositionStepM;
        State plus  = x;
        State minus = x;
        plus[state_index]  += step;
        minus[state_index] -= step;

        if (state_index == kAngle)
        {
            plus[state_index]  = algorithm::math::normalize_radian_pm_pi(plus[state_index]);
            minus[state_index] = algorithm::math::normalize_radian_pm_pi(minus[state_index]);
        }

        const std::optional<std::array<cv::Point2f, 4>> forward  = ObservePoints(plus,  size, dimensions, camera, frame, face);
        const std::optional<std::array<cv::Point2f, 4>> backward = ObservePoints(minus, size, dimensions, camera, frame, face);
        if (!forward || !backward)
        {
            return false;
        }

        for (std::size_t point = 0; point < 4; ++point)
        {
            jacobian(2 * point,     state_index) = ((*forward)[point].x - (*backward)[point].x) / (2.0 * step);
            jacobian(2 * point + 1, state_index) = ((*forward)[point].y - (*backward)[point].y) / (2.0 * step);
        }
    }

    // 形状量的雅可比范数：恒为 0 说明本帧观测根本没约束到这个量
    shape_jacobian_norm_r_ = jacobian.col(kRadius).norm();
    shape_jacobian_norm_l_ = jacobian.col(kRadiusGap).norm();
    shape_jacobian_norm_h_ = jacobian.col(kHeightGap).norm();

    const State before_correct = filter_.state();

    Filter::MeasureMatrix measurement_noise = Filter::MeasureMatrix::Identity(8, 8) * config_.point_sigma_px * config_.point_sigma_px;
    filter_.set_measurement_noise(measurement_noise);

    (void)filter_.correct(measured, [this, size, &dimensions, &camera, &frame, face](const State& state)
    {
        Measure projected(8);
        const std::optional<std::array<cv::Point2f, 4>> points = ObservePoints(state, size, dimensions, camera, frame, face);
        if (!points)
        {
            projected.setConstant(std::numeric_limits<double>::quiet_NaN());
            return projected;
        }
        for (std::size_t index = 0; index < points->size(); ++index)
        {
            projected[2 * index]     = (*points)[index].x;
            projected[2 * index + 1] = (*points)[index].y;
        }
        return projected;
    }, jacobian);

    // 修正前后各形状量的差：单帧看修正方向与幅度，连续看是在收敛还是在被观测来回拖
    const State& after_correct = filter_.state();
    shape_radius_step_      = after_correct[kRadius] - before_correct[kRadius];
    shape_radius_gap_step_  = after_correct[kRadiusGap] - before_correct[kRadiusGap];
    shape_height_gap_step_  = after_correct[kHeightGap] - before_correct[kHeightGap];

    return filter_.state().allFinite() && filter_.covariance().allFinite();
}

/**
 * @brief 把状态按指定面投成像素四点，不修改状态
 */
std::optional<std::array<cv::Point2f, 4>> VehicleEKF::ProjectFace(ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, std::int32_t face) const
{
    if (!initialized_ || face < 0 || face >= config_.armor_num)
    {
        return std::nullopt;
    }

    const State& state = filter_.state();
    return ReprojectArmor(ArmorCenter(state, face), FaceAngle(state, face), size, dimensions, config_.armor_pitch_rad, frame, camera);
}

/**
 * @brief 用一帧位姿把状态初始化到该帧
 *
 * PnP 解出的是相机系位姿，先按外参换到云台系（单位阵时是恒等）；旋转中心由面心减去
 * r·法向得到，与观测模型里 `面心 = 旋转中心 + r·法向` 互为逆运算，所以初始帧残差为 0。
 *
 * @note 云台系下法向由相机系法向经方向变换得到，再取方位角。外参为单位阵时与直接用
 *       相机系法向等价。
 */
void VehicleEKF::Initialize(const std::array<cv::Point2f, 4>& corners, ArmorSize size, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame)
{
    const std::optional<std::array<cv::Point3f, 4>> object_points = MakeObjectPoints(size, dimensions);
    if (!object_points) {
        return;
    }

    const std::optional<ArmorPose> pose = SolvePose(corners, *object_points, camera);
    if (!pose) {
        return;
    }

    cv::Matx33d rotation;
    cv::Rodrigues(cv::Mat(pose->rvec), rotation);
    // 相机系下法向是旋转矩阵第一列，换到跟踪系后取水平面（x-y）内的方位角
    const Eigen::Vector3d camera_normal(rotation(0, 0), rotation(1, 0), rotation(2, 0));
    const Eigen::Vector3d world_normal = frame.rotation * camera_normal;
    const double raw_yaw = std::atan2(world_normal.y(), world_normal.x());

    const Eigen::Vector3d pose_center = CameraToFrame(Eigen::Vector3d(pose->tvec[0], pose->tvec[1], pose->tvec[2]), frame);
    const double yaw = OptimizeYaw(corners, pose_center, size, dimensions, frame, camera, raw_yaw);

    const double radius = config_.radius_init_m;
    const Eigen::Vector3d normal(-std::cos(yaw), -std::sin(yaw), 0.0);
    const Eigen::Vector3d face_center = pose_center;

    State state = State::Zero();
    const Eigen::Vector3d center = face_center - radius * normal;
    state[kCenterX] = center.x();
    state[kCenterY] = center.y();
    state[kCenterZ] = center.z();
    state[kAngle]   = yaw;
    state[kRadius]  = radius;

    // 初值不确定度：位置 ±0.1 m、角度 ±0.3 rad、半径 ±0.2 m；速度未知但量级小，
    // 给 0.01（std 0.1 m/s）——给太大（如 1.0）会让滤波器把弧线运动解释成平移而不修半径。
    // 与参考一致只给对角阵：中心与半径的相关性由后续观测学，不预先写入
    StateMat covariance = StateMat::Zero();
    covariance.diagonal() << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.09, 0.25, 0.04, 0.01, 0.01;

    filter_.set_state(state, covariance);

    initialized_ = true;
    update_count_ = 0;
}

/**
 * @brief 把半径与第二半径钳回物理范围
 *
 * 钳制改变了状态却没有对应的观测，协方差必须跟着回到钳制后的实际不确定度；
 * 否则会被压到很小而状态仍贴着边界跳，后续 NIS 与收敛判据都被这个虚假的置信度带偏。
 */
void VehicleEKF::ApplyLimits()
{
    if (config_.armor_num != 4)
    {
        return;
    }

    State state = filter_.state();
    StateMat covariance = filter_.covariance();

    const double raw_radius = state[kRadius];
    const double raw_second = state[kRadius] + state[kRadiusGap];
    state[kRadius]      = std::clamp(raw_radius, config_.radius_min_m, config_.radius_max_m);
    const double second = std::clamp(raw_second, config_.radius_min_m, config_.radius_max_m);
    state[kRadiusGap]   = second - state[kRadius];

    // 钳掉了多少，就把多少不确定度还给对应的对角元
    const double radius_shift = state[kRadius] - raw_radius;
    const double second_shift = second - raw_second;
    if (radius_shift != 0.0 || second_shift != 0.0)
    {
        ++limit_clamps_;
        ++consecutive_limit_clamps_;
    }
    else
    {
        consecutive_limit_clamps_ = 0;
    }
    if (radius_shift != 0.0)
    {
        // 钳制后 r 与其它分量不再有观测推导出的相关性
        const double variance = std::max(covariance(kRadius, kRadius) + radius_shift * radius_shift, radius_shift * radius_shift);
        covariance.row(kRadius).setZero();
        covariance.col(kRadius).setZero();
        covariance(kRadius, kRadius) = variance;
    }
    if (second_shift != 0.0)
    {
        covariance(kRadiusGap, kRadiusGap) += second_shift * second_shift;
    }

    filter_.set_state(state, covariance);
}

/**
 * @brief 四面的中心与转角
 */
std::vector<FaceState> VehicleEKF::Faces() const
{
    std::vector<FaceState> armors;
    armors.reserve(static_cast<std::size_t>(config_.armor_num));

    const State& state = filter_.state();
    for (std::int32_t face = 0; face < config_.armor_num; ++face)
    {
        FaceState armor;
        armor.xyz     = ArmorCenter(state, face);
        armor.yaw_rad = FaceAngle(state, face);
        armors.push_back(armor);
    }
    return armors;
}

/**
 * @brief 半径或第二半径是否已跑出物理范围
 */
bool VehicleEKF::Diverged() const
{
    if (config_.max_consecutive_limit_clamps > 0 && consecutive_limit_clamps_ >= config_.max_consecutive_limit_clamps)
    {
        return true;
    }

    const State& state = filter_.state();
    const double radius = state[kRadius];
    const double second = state[kRadius] + state[kRadiusGap];

    const bool radius_ok = radius > 0.0 && radius < 0.8;
    const bool second_ok = second > 0.0 && second < 0.8;
    return !radius_ok || !second_ok;
}

/**
 * @brief 供日志/调试读出的状态分量
 */
VehicleEKF::Snapshot VehicleEKF::DebugSnapshot() const
{
    Snapshot snapshot;
    if (!initialized_)
    {
        return snapshot;
    }

    const State& state = filter_.state();
    snapshot.center        = Eigen::Vector3d(state[kCenterX], state[kCenterY], state[kCenterZ]);
    snapshot.velocity      = Eigen::Vector3d(state[kVelocityX], state[kVelocityY], state[kVelocityZ]);
    snapshot.angle_rad     = state[kAngle];
    snapshot.omega_rad_s   = state[kOmega];
    snapshot.radius_m      = state[kRadius];
    snapshot.radius_gap_m  = state[kRadiusGap];
    snapshot.height_gap_m  = state[kHeightGap];
    snapshot.last_nis      = filter_.last_nis();

    const StateMat& covariance = filter_.covariance();
    for (std::size_t index = 0; index < snapshot.covariance_diag.size(); ++index)
    {
        snapshot.covariance_diag[index] = covariance(static_cast<Eigen::Index>(index), static_cast<Eigen::Index>(index));
    }
    snapshot.image_updates = image_updates_;
    snapshot.ypda_updates  = ypda_updates_;
    snapshot.limit_clamps  = limit_clamps_;
    snapshot.shape_radius_step      = shape_radius_step_;
    snapshot.shape_radius_gap_step  = shape_radius_gap_step_;
    snapshot.shape_height_gap_step  = shape_height_gap_step_;
    snapshot.shape_jacobian_norm_r  = shape_jacobian_norm_r_;
    snapshot.shape_jacobian_norm_l  = shape_jacobian_norm_l_;
    snapshot.shape_jacobian_norm_h  = shape_jacobian_norm_h_;
    return snapshot;
}

} // namespace algorithm::vision
