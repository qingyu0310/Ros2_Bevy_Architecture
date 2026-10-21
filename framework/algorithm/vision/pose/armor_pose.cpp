/**
 * @file armor_pose.cpp
 * @author qingyu
 * @brief 单块装甲板位姿实现 — 物点生成、IPPE 求解与反投校验
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/pose/armor_pose.hpp"

#include "framework/algorithm/math/angle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <vector>

namespace algorithm::vision
{

namespace
{

/**
 * @brief 检查内参与畸变是否可参与投影
 *
 * fx、fy 必须为正，全部参数须为有限值；否则投影会产出 NaN 而不会被发现。
 *
 * @param camera 内参与畸变
 * @return true 可用于 solvePnP
 */
bool IsCameraUsable(const CameraModel& camera)
{
    return std::isfinite(camera.fx) && std::isfinite(camera.fy) && std::isfinite(camera.cx) && std::isfinite(camera.cy) &&
           std::isfinite(camera.k1) && std::isfinite(camera.k2) && std::isfinite(camera.p1) && std::isfinite(camera.p2) &&
           std::isfinite(camera.k3) && camera.fx > 0.0 && camera.fy > 0.0;
}

/**
 * @brief 拼出 3x3 内参矩阵，行优先
 */
cv::Mat CameraMatrix(const CameraModel& camera)
{
    return (cv::Mat_<double>(3, 3) << camera.fx, 0.0,       camera.cx,
                                                  0.0,      camera.fy, camera.cy,
                                                  0.0,      0.0,        1.0);
}

/**
 * @brief 拼出 1x5 畸变系数，顺序 k1、k2、p1、p2、k3
 */
cv::Mat DistortCoeffs(const CameraModel& camera)
{
    return (cv::Mat_<double>(1, 5) << camera.k1, camera.k2, camera.p1, camera.p2, camera.k3);
}

/**
 * @brief 检查四个点是否都是有限值
 */
bool AllFinite(const std::array<cv::Point2f, 4>& points)
{
    return std::all_of(points.begin(), points.end(), [](const cv::Point2f& point)
    {
        return std::isfinite(point.x) && std::isfinite(point.y);
    });
}

/**
 * @brief 四点平均误差，反投与实测逐点配对
 */
double MeanReprojectionError(const std::array<cv::Point2f, 4>& measured, const std::vector<cv::Point2f>& projected)
{
    double total = 0.0;
    for (std::size_t index = 0; index < measured.size(); ++index)
    {
        const cv::Point2f delta = measured[index] - projected[index];
        total += std::sqrt(delta.dot(delta));
    }

    return total / static_cast<double>(measured.size());
}

/**
 * @brief Rodrigues 旋转向量转 Eigen 3x3 矩阵
 */
Eigen::Matrix3d ToEigenRotation(const cv::Vec3d& rvec)
{
    cv::Matx33d rotation;
    cv::Rodrigues(cv::Mat(rvec), rotation);

    Eigen::Matrix3d result;
    for (std::int32_t row = 0; row < 3; ++row)
    {
        for (std::int32_t col = 0; col < 3; ++col)
        {
            result(row, col) = rotation(row, col);
        }
    }
    return result;
}

/**
 * @brief Eigen 3x3 矩阵转 Rodrigues 旋转向量
 */
cv::Vec3d ToRotationVector(const Eigen::Matrix3d& rotation)
{
    cv::Matx33d matrix;
    for (std::int32_t row = 0; row < 3; ++row)
    {
        for (std::int32_t col = 0; col < 3; ++col)
        {
            matrix(row, col) = rotation(row, col);
        }
    }

    cv::Vec3d rvec;
    cv::Rodrigues(cv::Mat(matrix), rvec);
    return rvec;
}

/**
 * @brief 外参旋转向量转 3x3 旋转矩阵
 */
Eigen::Matrix3d ExtrinsicRotation(const GimbalExtrinsics& extrinsics)
{
    return ToEigenRotation(extrinsics.rotation);
}

} // namespace

/**
 * @brief 相机系点 → 云台系点
 */
Eigen::Vector3d CameraToGimbal(const Eigen::Vector3d& point, const GimbalExtrinsics& extrinsics)
{
    const Eigen::Vector3d translation(extrinsics.translation[0], extrinsics.translation[1], extrinsics.translation[2]);
    return ExtrinsicRotation(extrinsics) * point + translation;
}

/**
 * @brief 相机系方向 → 云台系方向，只转不平移
 */
Eigen::Vector3d CameraToGimbalDirection(const Eigen::Vector3d& direction, const GimbalExtrinsics& extrinsics)
{
    return ExtrinsicRotation(extrinsics) * direction;
}

/**
 * @brief 云台系点 → 相机系点
 */
Eigen::Vector3d GimbalToCamera(const Eigen::Vector3d& point, const GimbalExtrinsics& extrinsics)
{
    const Eigen::Vector3d translation(extrinsics.translation[0], extrinsics.translation[1], extrinsics.translation[2]);
    return ExtrinsicRotation(extrinsics).transpose() * (point - translation);
}

/**
 * @brief 云台系方向 → 相机系方向，只转不平移
 */
Eigen::Vector3d GimbalToCameraDirection(const Eigen::Vector3d& direction, const GimbalExtrinsics& extrinsics)
{
    return ExtrinsicRotation(extrinsics).transpose() * direction;
}

/**
 * @brief 把一块装甲板的姿态从云台系表达换到相机系表达
 */
cv::Vec3d GimbalToCameraRotation(const cv::Vec3d& gimbal_rvec, const GimbalExtrinsics& extrinsics)
{
    const Eigen::Matrix3d camera_from_armor = ExtrinsicRotation(extrinsics).transpose() * ToEigenRotation(gimbal_rvec);
    return ToRotationVector(camera_from_armor);
}

/**
 * @brief 由板法向转角与板面倾角拼出板系到跟踪系的旋转向量
 *
 * 照参考 `solver.cpp` 的 `R_armor2world`：跟踪系 z 竖直，水平面为 x-y，转角从 +x 向 +y 量。
 * 三个列向量依次是板系 x（法向 n）、y（宽度 w）、z（灯条 l）。
 *
 * @param yaw_rad 板法向在水平面内的方位角，单位 rad
 * @param pitch_rad 板面绕宽度轴的倾角，单位 rad
 * @return 板系到跟踪系的 Rodrigues 旋转向量
 */
cv::Vec3d ArmorRotation(double yaw_rad, double pitch_rad)
{
    const double cos_yaw   = std::cos(yaw_rad);
    const double sin_yaw   = std::sin(yaw_rad);
    const double cos_pitch = std::cos(pitch_rad);
    const double sin_pitch = std::sin(pitch_rad);

    const cv::Matx33d rotation(cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
                               sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch,
                               -sin_pitch,           0.0,     cos_pitch);

    cv::Vec3d rvec;
    cv::Rodrigues(cv::Mat(rotation), rvec);
    return rvec;
}

/**
 * @brief 相机系点 → 跟踪系点
 */
Eigen::Vector3d CameraToFrame(const Eigen::Vector3d& point, const FrameTransform& frame)
{
    return frame.rotation * point + frame.translation;
}

/**
 * @brief 跟踪系点 → 相机系点
 */
Eigen::Vector3d FrameToCamera(const Eigen::Vector3d& point, const FrameTransform& frame)
{
    return frame.rotation.transpose() * (point - frame.translation);
}

/**
 * @brief 跟踪系方向 → 相机系方向，只转不平移
 */
Eigen::Vector3d FrameToCameraDirection(const Eigen::Vector3d& direction, const FrameTransform& frame)
{
    return frame.rotation.transpose() * direction;
}

/**
 * @brief 把一块装甲板在跟踪系的姿态换到相机系表达
 */
cv::Vec3d FrameToCameraRotation(const cv::Vec3d& world_rvec, const FrameTransform& frame)
{
    const Eigen::Matrix3d camera_from_armor = frame.rotation.transpose() * ToEigenRotation(world_rvec);
    return ToRotationVector(camera_from_armor);
}

/**
 * @brief 跟踪系位置转球坐标，照参考 `xyz2ypd`
 */
Eigen::Vector3d XyzToYpd(const Eigen::Vector3d& xyz)
{
    const double x = xyz.x();
    const double y = xyz.y();
    const double z = xyz.z();

    return Eigen::Vector3d(std::atan2(y, x), std::atan2(z, std::hypot(x, y)), xyz.norm());
}

/**
 * @brief `XyzToYpd` 对位置的雅可比，照参考 `xyz2ypd_jacobian`
 */
Eigen::Matrix3d YpdJacobian(const Eigen::Vector3d& xyz)
{
    const double x = xyz.x();
    const double y = xyz.y();
    const double z = xyz.z();
    const double xy2 = std::max(x * x + y * y, 1e-12);
    const double norm2 = std::max(xy2 + z * z, 1e-12);
    const double pitch_scale = (z * z / xy2 + 1.0) * std::sqrt(xy2);

    Eigen::Matrix3d jacobian;
    jacobian << -y / xy2, x / xy2, 0.0,
                -(x * z) / (pitch_scale * xy2), -(y * z) / (pitch_scale * xy2), 1.0 / pitch_scale,
                x / std::sqrt(norm2), y / std::sqrt(norm2), z / std::sqrt(norm2);
    return jacobian;
}

/**
 * @brief 按装甲板尺寸生成板系物点，顺序固定 LT、RT、RB、LB
 *
 * @param size 装甲板大小；Unknown 表示尺寸不明，返回 std::nullopt 而不是猜一个
 * @param dimensions 实物尺寸，单位 m
 * @return 四个物点，顺序与检测结果的角点顺序一致
 */
std::optional<std::array<cv::Point3f, 4>> MakeObjectPoints(ArmorSize size, const ArmorDimensions& dimensions)
{
    double width = 0.0;
    switch (size)
    {
        case ArmorSize::Small:
            width = dimensions.small_width_m;
            break;
        case ArmorSize::Large:
            width = dimensions.big_width_m;
            break;
        default:
            return std::nullopt; // 尺寸不明就不解：拿错宽度解出的尺度会系统性偏大或偏小
    }

    const double length = dimensions.lightbar_length_m;
    if (!std::isfinite(width) || !std::isfinite(length) || width <= 0.0 || length <= 0.0)
    {
        return std::nullopt;
    }

    const float half_width  = static_cast<float>(width * 0.5);
    const float half_length = static_cast<float>(length * 0.5);

    // 板系 x 为法向，四点的 x 都是 0；y 沿宽度、z 沿灯条方向
    return std::array<cv::Point3f, 4> {{
        cv::Point3f(0.0f, +half_width, +half_length), // LT
        cv::Point3f(0.0f, -half_width, +half_length), // RT
        cv::Point3f(0.0f, -half_width, -half_length), // RB
        cv::Point3f(0.0f, +half_width, -half_length), // LB
    }};
}

/**
 * @brief 由像素四点解出相机系位姿，并把结果反投回像素计算误差
 *
 * @param corners 图像四点，顺序必须与 object_points 的 LT、RT、RB、LB 一致
 * @param object_points 板系物点，单位 m
 * @param camera 内参与畸变
 * @return 位姿与反投结果；角点非有限、内参非法或求解失败时返回 std::nullopt
 */
std::optional<ArmorPose> SolvePose(const std::array<cv::Point2f, 4>& corners, const std::array<cv::Point3f, 4>& object_points, const CameraModel& camera)
{
    if (!IsCameraUsable(camera) || !AllFinite(corners))
    {
        return std::nullopt;
    }

    const cv::Mat camera_matrix = CameraMatrix(camera);
    const cv::Mat distort       = DistortCoeffs(camera);

    const std::vector<cv::Point3f> object_list(object_points.begin(), object_points.end());
    const std::vector<cv::Point2f> image_list(corners.begin(), corners.end());

    // 共面四点用 IPPE。IPPE 对这个几何有两个解（板正面朝相机 / 背面朝相机），
    // solvePnP 只返回其中一个，选哪个取决于板尺寸与姿态，没有保证。
    // 这里把两个解都解出来，按四角重投影误差取小的那个；背面解会让 z 变负、误差大一个量级。
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    if (!cv::solvePnPGeneric(object_list, image_list, camera_matrix, distort, rvecs, tvecs, false, cv::SOLVEPNP_IPPE) || rvecs.empty())
    {
        return std::nullopt;
    }

    // 优先取"板在相机前方"的解；两个解都不在前方时仍按重投影误差取一个，
    // 返回空会让调用方整帧丢掉观测，比给一个可疑解更糟（跟踪器还会做门限与 NIS 判断）。
    std::optional<ArmorPose> best;
    bool best_in_front = false;
    for (std::size_t index = 0; index < rvecs.size(); ++index)
    {
        const cv::Vec3d rvec(rvecs[index]);
        const cv::Vec3d tvec(tvecs[index]);

        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_list, rvec, tvec, camera_matrix, distort, projected);
        if (projected.size() != object_points.size())
        {
            continue;
        }

        ArmorPose pose;
        pose.rvec = rvec;
        pose.tvec = tvec;
        pose.reprojection_error_px = MeanReprojectionError(corners, projected);
        std::copy(projected.begin(), projected.end(), pose.projected.begin());

        const bool in_front = tvec[2] > 0.0;
        const bool better = !best || (in_front && !best_in_front) ||
                            (in_front == best_in_front && pose.reprojection_error_px < best->reprojection_error_px);
        if (better)
        {
            best = pose;
            best_in_front = in_front;
        }
    }

    return best;
}

/**
 * @brief 把板系位姿反投回像素四点
 *
 * @param object_points 板系物点，单位 m
 * @param rvec 板系到相机系的旋转向量
 * @param tvec 板心在相机系的位置，单位 m
 * @param camera 内参与畸变
 * @return 反投的四点；内参或位姿含非有限值时返回 std::nullopt
 */
std::optional<std::array<cv::Point2f, 4>> ProjectPose(const std::array<cv::Point3f, 4>& object_points, const cv::Vec3d& rvec, const cv::Vec3d& tvec, const CameraModel& camera)
{
    if (!IsCameraUsable(camera) || !std::isfinite(tvec[0]) || !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]))
    {
        return std::nullopt;
    }

    const std::vector<cv::Point3f> object_list(object_points.begin(), object_points.end());
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_list, rvec, tvec, CameraMatrix(camera), DistortCoeffs(camera), projected);
    if (projected.size() != object_points.size())
    {
        return std::nullopt;
    }

    std::array<cv::Point2f, 4> result {};
    std::copy(projected.begin(), projected.end(), result.begin());

    return result;
}

/**
 * @brief 把一个跟踪系的面按 `(world_xyz, yaw) + 固定 pitch` 反投回像素
 *
 * 照参考 `Solver::reproject_armor`：先按跟踪系位姿拼出板系到跟踪系的旋转，再用帧变换
 * 换到相机系，最后用板尺寸的物点投回像素。
 */
std::optional<std::array<cv::Point2f, 4>> ReprojectArmor(const Eigen::Vector3d& world_xyz, double yaw_rad, ArmorSize size, const ArmorDimensions& dimensions, double pitch_rad, const FrameTransform& frame, const CameraModel& camera)
{
    const std::optional<std::array<cv::Point3f, 4>> object_points = MakeObjectPoints(size, dimensions);
    if (!object_points)
    {
        return std::nullopt;
    }

    const cv::Vec3d rvec = FrameToCameraRotation(ArmorRotation(yaw_rad, pitch_rad), frame);
    const Eigen::Vector3d camera_center = FrameToCamera(world_xyz, frame);

    return ProjectPose(*object_points, rvec, cv::Vec3d(camera_center.x(), camera_center.y(), camera_center.z()), camera);
}

/**
 * @brief 由 PnP 位姿一阶传播出 ypda 观测的 4x4 测量协方差
 */
std::optional<Eigen::Matrix4d> YpdaMeasurementCovariance(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation, ArmorSize size, const ArmorDimensions& dimensions, const FrameTransform& frame, const CameraModel& camera, double point_sigma_px, double angle_prior_rad)
{
    constexpr std::int32_t kPoseDim = 6;
    constexpr double kTranslationStep = 1e-4;
    constexpr double kRotationStep = 1e-5;
    constexpr double kTranslationPrior = 10.0;

    if (!(point_sigma_px > 0.0) || !(angle_prior_rad > 0.0) || !std::isfinite(point_sigma_px) || !std::isfinite(angle_prior_rad))
    {
        return std::nullopt;
    }

    const std::optional<std::array<cv::Point3f, 4>> object_points = MakeObjectPoints(size, dimensions);
    if (!object_points)
    {
        return std::nullopt;
    }

    // 相机系位姿 → 世界系观测 z = [yaw, pitch, distance, armor_yaw]
    const auto observation_of = [&frame](const Eigen::Matrix3d& armor_to_camera, const Eigen::Vector3d& center_camera)
    {
        const Eigen::Vector3d center_world = CameraToFrame(center_camera, frame);
        const Eigen::Vector3d ypd = XyzToYpd(center_world);
        const Eigen::Matrix3d armor_to_world = frame.rotation * armor_to_camera;
        return Eigen::Vector4d(ypd.x(), ypd.y(), ypd.z(), std::atan2(armor_to_world(1, 0), armor_to_world(0, 0)));
    };

    Eigen::Matrix<double, 8, kPoseDim> pixel_jacobian = Eigen::Matrix<double, 8, kPoseDim>::Zero();
    Eigen::Matrix<double, 4, kPoseDim> measurement_jacobian = Eigen::Matrix<double, 4, kPoseDim>::Zero();

    for (std::int32_t index = 0; index < kPoseDim; ++index)
    {
        Eigen::Matrix3d rotation_plus = rotation;
        Eigen::Matrix3d rotation_minus = rotation;
        Eigen::Vector3d translation_plus = translation;
        Eigen::Vector3d translation_minus = translation;
        double step = kTranslationStep;

        if (index < 3)
        {
            translation_plus[index] += kTranslationStep;
            translation_minus[index] -= kTranslationStep;
        }
        else
        {
            Eigen::Vector3d axis = Eigen::Vector3d::Zero();
            axis[index - 3] = 1.0;
            rotation_plus = rotation * Eigen::AngleAxisd(kRotationStep, axis).toRotationMatrix();
            rotation_minus = rotation * Eigen::AngleAxisd(-kRotationStep, axis).toRotationMatrix();
            step = kRotationStep;
        }

        const cv::Vec3d rvec_plus = ToRotationVector(rotation_plus);
        const cv::Vec3d rvec_minus = ToRotationVector(rotation_minus);
        const std::optional<std::array<cv::Point2f, 4>> points_plus = ProjectPose(*object_points, rvec_plus, cv::Vec3d(translation_plus.x(), translation_plus.y(), translation_plus.z()), camera);
        const std::optional<std::array<cv::Point2f, 4>> points_minus = ProjectPose(*object_points, rvec_minus, cv::Vec3d(translation_minus.x(), translation_minus.y(), translation_minus.z()), camera);
        if (!points_plus || !points_minus)
        {
            return std::nullopt;
        }

        const Eigen::Vector4d z_plus = observation_of(rotation_plus, translation_plus);
        const Eigen::Vector4d z_minus = observation_of(rotation_minus, translation_minus);

        for (std::size_t point = 0; point < 4; ++point)
        {
            pixel_jacobian(2 * point, index) = ((*points_plus)[point].x - (*points_minus)[point].x) / (2.0 * step);
            pixel_jacobian(2 * point + 1, index) = ((*points_plus)[point].y - (*points_minus)[point].y) / (2.0 * step);
        }
        for (std::int32_t row = 0; row < 3; ++row)
        {
            measurement_jacobian(row, index) = (z_plus[row] - z_minus[row]) / (2.0 * step);
        }
        measurement_jacobian(3, index) = std::remainder(z_plus[3] - z_minus[3], 2.0 * M_PI) / (2.0 * step);
    }

    const Eigen::Matrix<double, kPoseDim, kPoseDim> information = pixel_jacobian.transpose() * pixel_jacobian / (point_sigma_px * point_sigma_px);
    Eigen::Matrix<double, kPoseDim, kPoseDim> prior = Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
    prior.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() / (kTranslationPrior * kTranslationPrior);
    prior.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() / (angle_prior_rad * angle_prior_rad);

    const Eigen::LDLT<Eigen::Matrix<double, kPoseDim, kPoseDim>> solver(information + prior);
    if (solver.info() != Eigen::Success)
    {
        return std::nullopt;
    }

    const Eigen::Matrix<double, kPoseDim, kPoseDim> pose_covariance = solver.solve(Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity());
    if (!pose_covariance.allFinite())
    {
        return std::nullopt;
    }

    Eigen::Matrix4d covariance = measurement_jacobian * pose_covariance * measurement_jacobian.transpose();
    covariance = 0.5 * (covariance + covariance.transpose());
    if (!covariance.allFinite())
    {
        return std::nullopt;
    }
    return covariance;
}

/**
 * @brief 按四角重投影误差最优化板法向 yaw
 *
 * 照参考 `Solver::optimize_yaw`：搜索范围 140°、步长 1°，误差是四角像素距离之和；
 * 板面倾角用普通装甲的 15°。
 */
double OptimizeYaw(const std::array<cv::Point2f, 4>& corners, const Eigen::Vector3d& world_xyz, ArmorSize size, const ArmorDimensions& dimensions, const FrameTransform& frame, const CameraModel& camera, double initial_yaw)
{
    constexpr std::int32_t kSearchRangeDeg = 140;
    constexpr double kDegToRad = M_PI / 180.0;

    // 相机光轴（相机系 +z）在跟踪系水平面内的方位角
    const Eigen::Vector3d camera_forward = frame.rotation * Eigen::Vector3d::UnitZ();
    const double camera_yaw = std::atan2(camera_forward.y(), camera_forward.x());
    const double yaw0 = algorithm::math::normalize_radian_pm_pi(camera_yaw - kSearchRangeDeg / 2.0 * kDegToRad);

    double best_yaw = initial_yaw;
    double min_error = std::numeric_limits<double>::max();

    for (std::int32_t index = 0; index < kSearchRangeDeg; ++index)
    {
        const double yaw = algorithm::math::normalize_radian_pm_pi(yaw0 + static_cast<double>(index) * kDegToRad);
        const std::optional<std::array<cv::Point2f, 4>> projected = ReprojectArmor(world_xyz, yaw, size, dimensions, kArmorPitchRad, frame, camera);
        if (!projected)
        {
            continue;
        }

        double error = 0.0;
        for (std::size_t point = 0; point < corners.size(); ++point)
        {
            error += cv::norm(corners[point] - (*projected)[point]);
        }

        if (error < min_error)
        {
            min_error = error;
            best_yaw = yaw;
        }
    }

    return best_yaw;
}

} // namespace algorithm::vision
