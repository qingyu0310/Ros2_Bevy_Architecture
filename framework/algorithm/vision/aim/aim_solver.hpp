/**
 * @file aim_solver.hpp
 * @author qingyu
 * @brief 期望角度解算 — 瞄准点 → 云台 yaw/pitch/距离/开火判定
 * @version 0.1
 * @date 2026-10-19
 *
 * @copyright Copyright (c) 2026
 *
 * @note 纯计算：不依赖 ROS、tf2，也不读时间。时间对齐由调用方完成，这里只吃已对齐的量。
 */

#pragma once

#include <Eigen/Core>

#include <optional>

namespace algorithm::vision
{

/**
 * @brief 一次解算的输入
 *
 * 瞄准点与相机原点都用相机系；map 系的相机到云台关系由另外两个向量给出。
 */
struct AimSolverInput
{
    Eigen::Vector3d aim_camera {};                                      // 瞄准点，相机系，单位 m
    Eigen::Matrix3d map_from_camera {Eigen::Matrix3d::Identity()};   // 相机系 → map
    Eigen::Vector3d camera_in_map {};                                   // 相机原点在 map，单位 m
    Eigen::Vector3d gimbal_in_map {};                                   // 云台原点在 map，单位 m
    double fire_angle_deg {1.5};                                        // 开火允许的 pitch 偏差，单位度
    double distance_max_m {8.0};                                        // 可视与开火距离上限，单位 m
    double min_forward_m {0.1};                                         // 瞄准点在相机系的最小前向距离，单位 m
};

/**
 * @brief 一次解算的结果
 */
struct AimSolution
{
    double yaw_deg {0.0};                                               // 云台期望绝对 yaw，从 map x 轴量起
    double pitch_deg {90.0};                                            // 云台期望 pitch，从 map 竖直轴量起
    double distance_m {0.0};                                            // 相机到瞄准点的直线距离
    bool   fire_advice {false};                                         // 是否满足开火判定
};

/**
 * @brief 由瞄准点解出云台期望角
 *
 * @param input 解算输入
 * @return 解算结果；输入含非有限值、瞄准点不在相机前方或超出距离上限时为空
 */
std::optional<AimSolution> SolveAim(const AimSolverInput& input);

} // namespace algorithm::vision
