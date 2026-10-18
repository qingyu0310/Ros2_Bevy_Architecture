/**
 * @file aim_publisher.hpp
 * @author qingyu
 * @brief 自瞄发布 — 取跟踪器选定面、对齐位姿、解出云台指令并发布
 * @version 0.1
 * @date 2026-10-19
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <rm_interfaces/msg/gimbal_cmd.hpp>
#include <tf2_ros/buffer.h>

#include "framework/algorithm/vision/aim/aim_solver.hpp"
#include "framework/algorithm/vision/pose/armor_pose.hpp"
#include "framework/algorithm/vision/track/tracker.hpp"

namespace algorithm::vision
{

/**
 * @brief 自瞄参数
 */
struct AimPublisherConfig
{
    std::string topic {"/rm_gimbal/cmd"};           // 云台指令话题
    std::string map_frame {"map"};                  // 世界系
    std::string camera_frame {"camera_optical_frame"}; // 图像所在坐标系
    double fire_angle_deg {1.5};                    // 开火允许的 pitch 偏差，单位度
    double distance_max_m {8.0};                    // 可视与开火距离上限，单位 m
    std::string fixed_yaw_pitch;                    // 非空时忽略解算，恒发这组绝对角
    double fixed_yaw_deg {0.0};                     // 上面解析出的 yaw
    double fixed_pitch_deg {0.0};                   // 上面解析出的 pitch
};

/**
 * @brief 自瞄发布器
 *
 * 每次 Publish 从跟踪器取当前选定面，用当帧的相机位姿换到 map，再交给 `SolveAim` 解角。
 * 没有可瞄面、查不到 TF、解算失败时发一条 `distance = -1` 的空指令。
 */
class AimPublisher final
{
public:
    /**
     * @brief 建立发布器
     *
     * @param node 宿主节点，用于建发布器与日志
     * @param buffer 宿主节点持有的 tf2 buffer
     * @param config 自瞄参数
     */
    AimPublisher(rclcpp::Node& node, const tf2_ros::Buffer& buffer, const AimPublisherConfig& config);

    /**
     * @brief 用当帧跟踪结果发一条云台指令
     *
     * @param header 图像 header，时间戳用于查 TF
     * @param tracker 跟踪器
     * @param frame 当帧相机 → map 的位姿；为空时按无目标处理
     * @param gimbal_in_map 云台原点在 map 中的位置
     */
    void Publish(const std_msgs::msg::Header& header, const Tracker& tracker, const std::optional<FrameTransform>& frame, const Eigen::Vector3d& gimbal_in_map);

private:
    std::optional<Eigen::Vector3d> SelectAimPointCamera(const Tracker& tracker, const std::optional<FrameTransform>& frame) const;

    static Eigen::Quaterniond ToEigen(const geometry_msgs::msg::Quaternion& value);

    rclcpp::Node& node_;
    const tf2_ros::Buffer& buffer_;
    AimPublisherConfig config_;
    rclcpp::Publisher<rm_interfaces::msg::GimbalCmd>::SharedPtr publisher_;
};

} // namespace algorithm::vision
