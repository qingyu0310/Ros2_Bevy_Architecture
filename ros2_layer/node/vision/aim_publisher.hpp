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

    // 瞄准面选择，照参考 `Planner::choose_aim_point` 的三个角
    double aim_sector_deg {60.0};                   // 慢转时算"可射击"的扇区半角，单位度
    double coming_angle_deg {55.0};                 // 小陀螺时开始考虑的角，单位度（参考 55）
    double leaving_angle_deg {20.0};                // 小陀螺时"将要离开"的界，单位度（参考 20）
    double spin_omega_threshold {2.0};              // |ω| 低于它按慢转处理，单位 rad/s
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

    /**
     * @brief 最近一次选中的瞄准面序，-1 表示没有可选面
     */
    std::int32_t aim_face() const
    {
        return aim_face_;
    }

private:
    /**
     * @brief 从整车四个预测面里选一个可射击的面
     *
     * 照参考 `Planner::choose_aim_point`：按 `delta = 该面法向角 − 车心相对云台的方位角`
     * 判断哪个面朝向自己。慢转时在扇区内选并加锁防止两块板之间来回跳；小陀螺时选
     * "将要转出视野"的那一块，提前量才有意义。
     *
     * @param tracker 跟踪器，提供四面状态与角速度
     * @param gimbal_in_map 云台原点在 map 中的位置，车心方位角从这里量
     * @return 面序；无目标或四面都不可射击时返回 -1
     */
    std::int32_t SelectAimFace(const Tracker& tracker, const Eigen::Vector3d& gimbal_in_map);

    std::optional<Eigen::Vector3d> SelectAimPointCamera(const Tracker& tracker, const std::optional<FrameTransform>& frame) const;

    std::int32_t aim_face_ {-1};                    // 本帧选中的瞄准面序
    std::int32_t aim_lock_id_ {-1};                 // 慢转时的锁定面序，-1 表示未锁定

    static Eigen::Quaterniond ToEigen(const geometry_msgs::msg::Quaternion& value);

    rclcpp::Node& node_;
    const tf2_ros::Buffer& buffer_;
    AimPublisherConfig config_;
    rclcpp::Publisher<rm_interfaces::msg::GimbalCmd>::SharedPtr publisher_;
};

} // namespace algorithm::vision
