/**
 * @file aim_solver.cpp
 * @author qingyu
 * @brief 期望角度解算实现
 * @version 0.1
 * @date 2026-10-19
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/aim/aim_solver.hpp"

#include <cmath>

namespace algorithm::vision
{

/**
 * @brief 由瞄准点解出云台期望角
 *
 * 角度与距离的定义照旧节点实现，逐项保持不变：yaw 取世界系 atan2(y, x)，pitch 从竖直轴
 * 量起（水平为 90°），distance 取相机到瞄准点的模长而不是世界系坐标的模长。
 */
std::optional<AimSolution> SolveAim(const AimSolverInput& input)
{
    if (!input.aim_camera.allFinite() || !input.map_from_camera.allFinite() || !input.camera_in_map.allFinite() || !input.gimbal_in_map.allFinite())
    {
        return std::nullopt;
    }

    // 瞄准点必须在相机前方且距离合理；否则 atan2 会给出朝天或朝后的角，云台会直接甩飞
    const double distance_m = input.aim_camera.norm();
    if (input.aim_camera.z() <= input.min_forward_m || distance_m > input.distance_max_m)
    {
        return std::nullopt;
    }

    const Eigen::Vector3d point_map = input.map_from_camera * input.aim_camera + input.camera_in_map;
    // 方向从云台指向目标，不是从世界原点；仿真的 GimbalCmd 把 yaw/pitch 当枪口的期望朝向
    const Eigen::Vector3d direction_map = point_map - input.gimbal_in_map;
    const double horizontal = std::hypot(direction_map.x(), direction_map.y());
    if (horizontal < 1e-6 || !direction_map.allFinite())
    {
        return std::nullopt;
    }

    AimSolution solution;
    solution.yaw_deg = std::atan2(direction_map.y(), direction_map.x()) * 180.0 / M_PI;
    solution.pitch_deg = 90.0 - std::atan2(direction_map.z(), horizontal) * 180.0 / M_PI;
    solution.distance_m = distance_m;
    solution.fire_advice = distance_m <= input.distance_max_m && std::abs(solution.pitch_deg - 90.0) <= input.fire_angle_deg;
    return solution;
}

} // namespace algorithm::vision
