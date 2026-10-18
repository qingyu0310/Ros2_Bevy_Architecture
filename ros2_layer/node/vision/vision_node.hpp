/**
 * @file vision_node.hpp
 * @author qingyu
 * @brief 视觉节点：检测 → 位姿 → 多目标跟踪 → 发布结果并可视化
 * @version 0.2
 * @date 2026-10-19
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rm_interfaces/msg/gimbal_cmd.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include "framework/algorithm/vision/detect/armor_detector.hpp"
#include "framework/algorithm/vision/pose/armor_pose.hpp"
#include "framework/algorithm/vision/track/tracker.hpp"
#include "ros2_layer/node/vision/aim_publisher.hpp"
#include "ros2_layer/node/vision/pose_buffer.hpp"

/**
 * @brief 视觉节点
 *
 * 订阅相机节点转好的 BGR 图像与内参，同步跑完检测、位姿与跟踪，发布带框图像与跟踪结果。
 * 内参只从 `/camera_info` 来（仿真与实机的焦距不同，配置里那份标定不能直接用在仿真上）。
 *
 * @note 处理放在图像回调里同步完成：单线程 executor 下不会堆积，且保持"看到的帧就在处理这一帧"
 *       的因果顺序。慢于图像到达速率时由 QoS 的 `SensorDataQoS` 丢弃旧帧。
 */
class VisionNode : public rclcpp::Node
{
public:
    explicit VisionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    /**
     * @brief 视觉节点参数
     */
    struct Params
    {
        std::string image_topic {"/vision/image"};
        std::string info_topic  {"/vision/camera_info"};
        std::string output_image_topic {"/vision/debug_image"};
        std::string aim_topic {"/rm_gimbal/cmd"};

        double score_threshold {0.7};
        double confidence_threshold {0.8};
        double nms_threshold {0.3};
        std::string engine_path;
        std::int32_t cuda_device {0};

        bool   show_window {false};
        std::string window_name {"Vision"};

        // 自瞄：瞄准点转 map 系后按"云台指令"协议发出去，仿真端用 PID 跟踪
        bool   aim_enabled {true};
        std::string aim_frame {"map"};
        std::string tf_topic {"/tf"};                        // TF 消息，相机位姿缓冲的来源
        std::string camera_frame {"camera_optical_frame"};   // 图像所在的坐标系
        double pose_hold_s {0.03};                              // 图像晚于最新姿态样本时允许用它的时间窗，单位 s
        std::string enemy_color {"red"};                     // 敌方颜色：red / blue；只把这种颜色的装甲板送进跟踪器
        double aim_fire_angle_deg {1.5};                        // 目标偏离光轴小于它才给 fire_advice
        double aim_distance_max_m {8.0};                        // 超过这个距离不给 fire_advice
        std::string aim_fixed_yaw_pitch;                        // 形如 "30,80" 时忽略解算，恒发这组绝对角；空表示正常自瞄

        // 跟踪参数，与旧工程 YAML 的同名项一一对应
        std::int32_t armor_num {4};
        double radius_init_m {0.2};
        double point_sigma_px {8.0};
        double accel_variance {10.0};
        double angular_accel_variance {100.0};
        double shape_variance {0.1};
    };

    Params params_ {};
    algorithm::vision::ArmorDetector::Config detector_config_ {};
    algorithm::vision::TrackerConfig          tracker_config_ {};

    std::unique_ptr<algorithm::vision::ArmorDetector> detector_;
    std::unique_ptr<algorithm::vision::Tracker>       tracker_;

    std::unique_ptr<tf2_ros::Buffer>            tf_buffer_;      // 相机位姿查 TF 用
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<algorithm::vision::AimPublisher> aim_publisher_;

    algorithm::vision::PoseBuffer camera_pose_;                 // 相机在 map 中的位姿缓冲，按图像时间戳插值
    Eigen::Vector3d    gimbal_world_ {};                        // 云台原点在 map 中的位置
    double fixed_yaw_deg_ {0.0};                                // 固定自瞄角，aim_fixed_yaw_pitch 解析结果
    double fixed_pitch_deg_ {0.0};

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr      image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr     tf_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         image_pub_;

    std::optional<algorithm::vision::CameraModel> camera_;      // 内参只从话题来，收到之前不处理图像
    std::optional<algorithm::vision::FrameTransform> frame_;    // 当帧相机→map 的位姿，跟踪与绘制共用
    std::optional<rclcpp::Time> last_stamp_;                    // 上一帧的 ROS 时间，用于跟踪 dt
    std::vector<algorithm::vision::DetectedArmor> last_armors_; // 上一帧检测结果，日志对照用

    void read_parameters();

    void on_camera_info(const sensor_msgs::msg::CameraInfo& msg);

    void on_image(const sensor_msgs::msg::Image& msg);

    void on_tf(const tf2_msgs::msg::TFMessage& msg);

    static algorithm::vision::FrameTransform MakeFrameTransform(const algorithm::vision::PoseSample& sample);

    static Eigen::Quaterniond ToEigen(const geometry_msgs::msg::Quaternion& value);

    bool show_window_() const;

    void draw_armors(cv::Mat& drawn, const std::vector<algorithm::vision::DetectedArmor>& armors) const;

    void draw_targets(cv::Mat& drawn) const;

    void draw_face(cv::Mat& drawn, const algorithm::vision::FaceState& face, const cv::Scalar& color, std::int32_t thickness) const;

    void publish_image(const cv::Mat& image, const std_msgs::msg::Header& header);

    static const char* StateName(algorithm::vision::TrackerState state);

    static const char* ResetReasonName(algorithm::vision::ResetReason reason);

    void log_window(const algorithm::vision::ArmorDetectionResult& result, double processing_ms, double age_ms);

    uint64_t window_frames_ {0};
    uint64_t window_accepted_ {0};
    uint64_t window_pose_missing_ {0};
    double   window_pose_offset_sum_ {0.0};                // 每秒累计的图像时刻 − 最新姿态时刻
    double   window_pose_offset_min_ {0.0};                // 本秒最小偏移，单位 s
    double   window_pose_offset_max_ {0.0};                // 本秒最大偏移，单位 s
    double   window_processing_ms_ {0.0};
    double   window_age_ms_ {0.0};
    rclcpp::Time  window_start_ {0, 0, RCL_ROS_TIME};
};
