/**
 * @file aim_publisher.cpp
 * @author qingyu
 * @brief 自瞄发布实现
 * @version 0.1
 * @date 2026-10-19
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "ros2_layer/node/vision/aim_publisher.hpp"

#include <cmath>

#include <tf2/exceptions.h>

namespace algorithm::vision
{

/**
 * @brief 建立发布器
 *
 * @param node 宿主节点，用于建发布器与日志
 * @param buffer 宿主节点持有的 tf2 buffer
 * @param config 自瞄参数
 */
AimPublisher::AimPublisher(rclcpp::Node& node, const tf2_ros::Buffer& buffer, const AimPublisherConfig& config)
    : node_(node), buffer_(buffer), config_(config)
{
    publisher_ = node_.create_publisher<rm_interfaces::msg::GimbalCmd>(config_.topic, rclcpp::SensorDataQoS());
}

/**
 * @brief 用当帧跟踪结果发一条云台指令
 *
 * 固定角标定模式优先于解算；没有可瞄面时发 -1，让仿真端撤掉 PID 目标。
 * TF 先按图像时间戳查，超窗时退到最新一帧——仿真的 TF 频率低于图像帧率。
 *
 * @param header 图像 header
 * @param tracker 跟踪器
 * @param frame 当帧相机 → map 的位姿
 * @param gimbal_in_map 云台原点在 map 中的位置
 */
void AimPublisher::Publish(const std_msgs::msg::Header& header, const Tracker& tracker, const std::optional<FrameTransform>& frame, const Eigen::Vector3d& gimbal_in_map)
{
    rm_interfaces::msg::GimbalCmd cmd;
    cmd.header = header;
    cmd.distance = -1.0;
    cmd.fire_advice = false;

    // 标定模式：忽略解算，恒发固定绝对角，用来对齐仿真端的角度约定
    if (!config_.fixed_yaw_pitch.empty())
    {
        cmd.yaw = config_.fixed_yaw_deg;
        cmd.pitch = config_.fixed_pitch_deg;
        cmd.distance = 2.0;
        publisher_->publish(cmd);
        return;
    }

    const std::optional<Eigen::Vector3d> aim_point = SelectAimPointCamera(tracker, frame);
    if (!aim_point)
    {
        publisher_->publish(cmd);
        return;
    }

    geometry_msgs::msg::TransformStamped camera_to_map;
    try
    {
        // 先用图像时间戳：TF 与图像同源，最准确
        camera_to_map = buffer_.lookupTransform(config_.map_frame, header.frame_id, header.stamp, tf2::durationFromSec(0.0));
    }
    catch (const tf2::ExtrapolationException&)
    {
        // 仿真的 TF 发布频率低于图像帧率，图像时间戳偶尔会晚于最新 TF。
        // 退一步用最新可用的变换：误差只在一帧内，比直接丢目标好
        try
        {
            camera_to_map = buffer_.lookupTransform(config_.map_frame, header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.0));
        }
        catch (const tf2::TransformException& exception)
        {
            RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000, "自瞄：查 TF 失败（%s）", exception.what());
            publisher_->publish(cmd);
            return;
        }
    }
    catch (const tf2::TransformException& exception)
    {
        RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000, "自瞄：查 TF 失败（%s）", exception.what());
        publisher_->publish(cmd);
        return;
    }

    AimSolverInput input;
    input.aim_camera = *aim_point;
    input.map_from_camera = ToEigen(camera_to_map.transform.rotation).toRotationMatrix();
    input.camera_in_map = Eigen::Vector3d(camera_to_map.transform.translation.x, camera_to_map.transform.translation.y, camera_to_map.transform.translation.z);
    input.gimbal_in_map = gimbal_in_map;
    input.fire_angle_deg = config_.fire_angle_deg;
    input.distance_max_m = config_.distance_max_m;

    const std::optional<AimSolution> solution = SolveAim(input);
    if (!solution)
    {
        publisher_->publish(cmd);
        return;
    }

    const Eigen::Vector3d target_in_map = input.map_from_camera * input.aim_camera + input.camera_in_map;
    const Eigen::Vector3d direction_in_map = target_in_map - input.gimbal_in_map;

    cmd.yaw = solution->yaw_deg;
    cmd.pitch = solution->pitch_deg;
    cmd.distance = solution->distance_m;
    cmd.fire_advice = solution->fire_advice;

    RCLCPP_INFO(node_.get_logger(), "aim cmd yaw=%.2f pitch=%.2f dist=%.3f fire=%d | target_map=(%.3f, %.3f, %.3f) gimbal_world=(%.3f, %.3f, %.3f) dir_map=(%.3f, %.3f, %.3f)",
                cmd.yaw, cmd.pitch, cmd.distance, cmd.fire_advice ? 1 : 0,
                target_in_map.x(), target_in_map.y(), target_in_map.z(),
                input.gimbal_in_map.x(), input.gimbal_in_map.y(), input.gimbal_in_map.z(),
                direction_in_map.x(), direction_in_map.y(), direction_in_map.z());

    publisher_->publish(cmd);
}

/**
 * @brief 选瞄准点，返回相机系坐标；没有收敛目标时为空
 *
 * 面序由跟踪器按参考 `choose_aim_point` 选好（锁定 + 角度判据），这里只取那个面的位置。
 * 该面本帧可能没有观测，位置由整车状态外推。
 *
 * @param tracker 跟踪器
 * @param frame 当帧相机 → map 的位姿
 * @return 瞄准点；无收敛目标、面序无效或点在相机后方时为空
 */
std::optional<Eigen::Vector3d> AimPublisher::SelectAimPointCamera(const Tracker& tracker, const std::optional<FrameTransform>& frame) const
{
    if (!tracker.Converged())
    {
        return std::nullopt;
    }

    // last_face 是选面结果：跟踪器见过多面之后，它可能是本帧没有观测、由状态外推的面
    const std::int32_t aim_face = tracker.Report().last_face;
    const std::vector<FaceState> faces = tracker.Faces();
    if (!frame || aim_face < 0 || static_cast<std::size_t>(aim_face) >= faces.size())
    {
        return std::nullopt;
    }

    const Eigen::Vector3d aim = FrameToCamera(faces[static_cast<std::size_t>(aim_face)].xyz, *frame);

    // 必须在相机前方且距离合理；否则 atan2 会给出朝天或朝后的角，云台会直接甩飞
    if (!aim.allFinite() || aim.z() <= 0.1 || aim.norm() > config_.distance_max_m)
    {
        return std::nullopt;
    }
    return aim;
}

/**
 * @brief ROS 四元数 → Eigen
 *
 * @param value ROS 四元数
 * @return 归一化后的 Eigen 四元数
 */
Eigen::Quaterniond AimPublisher::ToEigen(const geometry_msgs::msg::Quaternion& value)
{
    return Eigen::Quaterniond(value.w, value.x, value.y, value.z).normalized();
}

} // namespace algorithm::vision
