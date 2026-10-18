/**
 * @file vision_node.cpp
 * @author qingyu
 * @brief 视觉节点实现：检测 → 跟踪 → 绘制 → 自瞄发布
 * @version 0.2
 * @date 2026-10-19
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "ros2_layer/node/vision/vision_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tf2/exceptions.h>

VisionNode::VisionNode(const rclcpp::NodeOptions& options) : rclcpp::Node("vision", options)
{
    read_parameters();

    detector_ = std::make_unique<algorithm::vision::ArmorDetector>(detector_config_);
    tracker_  = std::make_unique<algorithm::vision::Tracker>(tracker_config_);
    window_start_ = now();

    // 自瞄发指令前要把瞄准点从相机系换到 map 系，这里仍用 tf2 的 buffer
    tf_buffer_     = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_   = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, false);

    // 相机位姿按消息时间戳入缓冲，图像来时按图像时间戳插值取：tf2 的 buffer 只提供
    // "离目标时刻最近"的查询，不插值，查不到时会退回最新一帧；这里自己存一份可插值的
    tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        params_.tf_topic, rclcpp::QoS(100),
        [this](tf2_msgs::msg::TFMessage::ConstSharedPtr msg) { on_tf(*msg); });

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        params_.image_topic, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_image(*msg); });

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        params_.info_topic, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { on_camera_info(*msg); });

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(params_.output_image_topic, rclcpp::SensorDataQoS());

    if (params_.aim_enabled)
    {
        algorithm::vision::AimPublisherConfig aim_config;
        aim_config.topic = params_.aim_topic;
        aim_config.map_frame = params_.aim_frame;
        aim_config.camera_frame = params_.camera_frame;
        aim_config.fire_angle_deg = params_.aim_fire_angle_deg;
        aim_config.distance_max_m = params_.aim_distance_max_m;
        aim_config.fixed_yaw_pitch = params_.aim_fixed_yaw_pitch;
        aim_config.fixed_yaw_deg = fixed_yaw_deg_;
        aim_config.fixed_pitch_deg = fixed_pitch_deg_;
        aim_publisher_ = std::make_unique<algorithm::vision::AimPublisher>(*this, *tf_buffer_, aim_config);
    }

    RCLCPP_INFO(get_logger(), "视觉节点：%s -> %s（显示=%s，自瞄=%s）", params_.image_topic.c_str(), params_.output_image_topic.c_str(), show_window_() ? "on" : "off", params_.aim_enabled ? "on" : "off");
}

/**
 * @brief 读取并校验参数
 */
void VisionNode::read_parameters()
{
    params_.image_topic        = declare_parameter("image_topic", params_.image_topic);
    params_.info_topic         = declare_parameter("camera_info_topic", params_.info_topic);
    params_.output_image_topic = declare_parameter("output_image_topic", params_.output_image_topic);
    params_.show_window        = declare_parameter("show_window", params_.show_window);
    params_.window_name        = declare_parameter("window_name", params_.window_name);

    params_.aim_topic          = declare_parameter("aim_topic", params_.aim_topic);
    params_.aim_frame          = declare_parameter("aim_frame", params_.aim_frame);
    params_.tf_topic           = declare_parameter("tf_topic", params_.tf_topic);
    params_.camera_frame       = declare_parameter("camera_frame", params_.camera_frame);
    params_.pose_hold_s        = declare_parameter("pose_hold_s", params_.pose_hold_s);
    params_.aim_enabled        = declare_parameter("aim_enabled", params_.aim_enabled);
    params_.enemy_color        = declare_parameter("enemy_color", params_.enemy_color);
    params_.aim_fire_angle_deg = declare_parameter("aim_fire_angle_deg", params_.aim_fire_angle_deg);
    params_.aim_distance_max_m = declare_parameter("aim_distance_max_m", params_.aim_distance_max_m);
    params_.aim_fixed_yaw_pitch = declare_parameter("aim_fixed_yaw_pitch", params_.aim_fixed_yaw_pitch);
    if (!params_.aim_fixed_yaw_pitch.empty())
    {
        double fixed_yaw = 0.0;
        double fixed_pitch = 0.0;
        if (std::sscanf(params_.aim_fixed_yaw_pitch.c_str(), "%lf,%lf", &fixed_yaw, &fixed_pitch) != 2)
        {
            throw std::runtime_error("vision: aim_fixed_yaw_pitch 需要形如 \"30,80\"");
        }
        // 解析结果留在成员里，构造 AimPublisher 时取走
        fixed_yaw_deg_ = fixed_yaw;
        fixed_pitch_deg_ = fixed_pitch;
    }

    if (params_.enemy_color != "red" && params_.enemy_color != "blue")
    {
        throw std::runtime_error("vision: enemy_color 只能是 red 或 blue，收到：" + params_.enemy_color);
    }

    detector_config_.engine_path = declare_parameter("engine_path", std::string {});
    if (detector_config_.engine_path.empty())
    {
        throw std::runtime_error("vision: engine_path 不能为空");
    }
    detector_config_.cuda_device          = declare_parameter("cuda_device", params_.cuda_device);
    detector_config_.score_threshold      = static_cast<float>(declare_parameter("score_threshold", params_.score_threshold));
    detector_config_.confidence_threshold = static_cast<float>(declare_parameter("confidence_threshold", params_.confidence_threshold));
    detector_config_.nms_threshold        = static_cast<float>(declare_parameter("nms_threshold", params_.nms_threshold));

    tracker_config_.vehicle.armor_num              = declare_parameter("armor_num",               params_.armor_num);
    tracker_config_.vehicle.radius_init_m          = declare_parameter("radius_init_m",           params_.radius_init_m);
    tracker_config_.vehicle.point_sigma_px         = declare_parameter("point_sigma_px",          params_.point_sigma_px);
    tracker_config_.vehicle.angle_prior_rad        = declare_parameter("angle_prior_rad",         tracker_config_.vehicle.angle_prior_rad);
    tracker_config_.vehicle.accel_variance         = declare_parameter("accel_variance",          params_.accel_variance);
    tracker_config_.vehicle.angular_accel_variance = declare_parameter("angular_accel_variance",  params_.angular_accel_variance);
    tracker_config_.vehicle.shape_variance         = declare_parameter("shape_variance",          params_.shape_variance);
    tracker_config_.min_detect_count               = declare_parameter("min_detect_count",       tracker_config_.min_detect_count);
    tracker_config_.max_temp_lost_count            = declare_parameter("max_temp_lost_count",    tracker_config_.max_temp_lost_count);
    tracker_config_.association.center_gate_px     = declare_parameter("center_gate_px",         tracker_config_.association.center_gate_px);
    tracker_config_.association.corner_gate_px     = declare_parameter("corner_gate_px",         tracker_config_.association.corner_gate_px);
    tracker_config_.association.angle_gate_rad     = declare_parameter("angle_gate_rad",         tracker_config_.association.angle_gate_rad);
    tracker_config_.association.perimeter_ratio_gate = declare_parameter("perimeter_ratio_gate",   tracker_config_.association.perimeter_ratio_gate);
    tracker_config_.nis_failure_ratio                = declare_parameter("nis_failure_ratio",      tracker_config_.nis_failure_ratio);
    tracker_config_.aim_weight_center                = declare_parameter("aim_weight_center",      tracker_config_.aim_weight_center);
    tracker_config_.aim_weight_cost                  = declare_parameter("aim_weight_cost",        tracker_config_.aim_weight_cost);
    tracker_config_.aim_weight_angle                 = declare_parameter("aim_weight_angle",       tracker_config_.aim_weight_angle);
}

/**
 * @brief 缓存内参；畸变顺序按 `plumb_bob`：k1、k2、p1、p2、k3
 */
void VisionNode::on_camera_info(const sensor_msgs::msg::CameraInfo& msg)
{
    algorithm::vision::CameraModel model;
    model.fx = msg.k[0];
    model.fy = msg.k[4];
    model.cx = msg.k[2];
    model.cy = msg.k[5];

    // d 为空表示无畸变（仿真是五项 0），保持默认的 0
    if (msg.d.size() >= 5)
    {
        model.k1 = msg.d[0];
        model.k2 = msg.d[1];
        model.p1 = msg.d[2];
        model.p2 = msg.d[3];
        model.k3 = msg.d[4];
    }

    if (!camera_ || model.fx != camera_->fx || model.fy != camera_->fy || model.cx != camera_->cx || model.cy != camera_->cy)
    {
        RCLCPP_INFO(get_logger(), "内参：%dx%d fx=%.2f fy=%.2f cx=%.2f cy=%.2f", msg.width, msg.height, model.fx, model.fy, model.cx, model.cy);
    }
    camera_ = model;
}

/**
 * @brief 处理一帧：检测 → 跟踪 → 绘制 → 发布
 */
void VisionNode::on_image(const sensor_msgs::msg::Image& msg)
{
    if (!camera_)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "还没收到内参，跳过本帧");
        return;
    }

    const cv::Mat image(static_cast<std::int32_t>(msg.height), static_cast<std::int32_t>(msg.width), CV_8UC3, const_cast<std::uint8_t*>(msg.data.data()), msg.step);

    const auto begin = std::chrono::steady_clock::now();
    algorithm::vision::ArmorDetectionResult result = detector_->Detect(image);

    // dt 用 ROS 时间算：仿真下 ROS 时钟与墙钟不同源，混用会让角速度估计全错
    const rclcpp::Time stamp(msg.header.stamp);
    double dt_s = 1.0 / 30.0;
    if (last_stamp_)
    {
        const double measured = (stamp - *last_stamp_).seconds();
        if (measured > 0.0 && measured < 1.0)
        {
            dt_s = measured;
        }
    }
    last_stamp_ = stamp;

    // 只把敌方颜色的装甲板送进跟踪器：同一编号的红蓝两辆车否则会抢同一个槽，
    // 面序匹配挑错对象后状态被拽飞、门全拒，跟踪再也拉不回来
    const algorithm::vision::TeamColor enemy = params_.enemy_color == "red" ? algorithm::vision::TeamColor::Red : algorithm::vision::TeamColor::Blue;
    std::vector<algorithm::vision::DetectedArmor> enemy_armors;
    enemy_armors.reserve(result.armors.size());
    for (const algorithm::vision::DetectedArmor& armor : result.armors)
    {
        if (armor.GetColor() == enemy)
        {
            enemy_armors.push_back(armor);
        }
    }

    // 每帧取一次相机到 map 的位姿，跟踪器据此工作在稳定的世界系
    const double image_stamp = stamp.seconds();
    const double pose_latest = camera_pose_.LatestStamp();
    const std::optional<algorithm::vision::PoseSample> pose_at_image = camera_pose_.At(image_stamp, params_.pose_hold_s);
    frame_.reset();
    if (pose_at_image)
    {
        frame_ = MakeFrameTransform(*pose_at_image);
    }
    // 图像时刻与缓冲里最近一条姿态的差：正数表示图像比姿态晚（用旧姿态），负数表示超窗
    window_pose_offset_sum_ += image_stamp - pose_latest;
    window_pose_offset_min_ = std::min(window_pose_offset_min_, image_stamp - pose_latest);
    window_pose_offset_max_ = std::max(window_pose_offset_max_, image_stamp - pose_latest);
    if (frame_)
    {
        tracker_->ProcessFrame(enemy_armors, algorithm::vision::ArmorDimensions {}, *camera_, *frame_, dt_s);
    }
    else
    {
        // 这一帧的姿态还没到，整帧丢掉：不检测、不跟踪、不发布，也不推进预测。
        // 用旧姿态换算观测会把云台转动误当成目标在动，比丢一帧更糟。
        ++window_pose_missing_;
        last_armors_ = std::move(enemy_armors);
        const double dropped_age_ms = (now() - stamp).seconds() * 1000.0;
        log_window(result, 0.0, dropped_age_ms);
        return;
    }
    last_armors_ = std::move(enemy_armors);

    cv::Mat drawn = image.clone();
    draw_armors(drawn, result.armors);
    draw_targets(drawn);

    const auto end = std::chrono::steady_clock::now();
    const double processing_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const double age_ms = (now() - stamp).seconds() * 1000.0;

    if (show_window_())
    {
        cv::imshow(params_.window_name, drawn);
        cv::waitKey(1);
    }

    publish_image(drawn, msg.header);
    if (aim_publisher_)
    {
        aim_publisher_->Publish(msg.header, *tracker_, frame_, gimbal_world_);
    }
    log_window(result, processing_ms, age_ms);
}

/**
 * @brief 缓存 `map ← 相机` 的位姿
 *
 * `/tf` 每条消息只带一条父子边，`map → 相机` 不是一条边而是若干段拼起来的。这里在同一
 * 批 TF 里按子坐标系建索引，从相机逐级走到目标坐标系，把各段变换乘起来。
 */
void VisionNode::on_tf(const tf2_msgs::msg::TFMessage& msg)
{
    std::unordered_map<std::string, const geometry_msgs::msg::TransformStamped*> by_child;
    for (const geometry_msgs::msg::TransformStamped& transform : msg.transforms)
    {
        by_child[transform.child_frame_id] = &transform;
    }

    // 云台在世界系中的位姿单独从根往下走一遍：它在链的中间，不能拿"相机相对它的位姿"顶替
    {
        Eigen::Matrix3d gimbal_rotation = Eigen::Matrix3d::Identity();
        Eigen::Vector3d gimbal_translation = Eigen::Vector3d::Zero();
        std::string frame = "gimbal_link";
        for (std::int32_t depth = 0; depth < 16 && frame != params_.aim_frame; ++depth)
        {
            const auto found = by_child.find(frame);
            if (found == by_child.end())
            {
                break;
            }
            const geometry_msgs::msg::TransformStamped& transform = *found->second;
            const Eigen::Matrix3d parent_to_child = ToEigen(transform.transform.rotation).toRotationMatrix();
            const Eigen::Vector3d parent_to_child_t(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);
            gimbal_translation = parent_to_child * gimbal_translation + parent_to_child_t;
            gimbal_rotation = parent_to_child * gimbal_rotation;
            frame = transform.header.frame_id;
        }
        if (frame == params_.aim_frame && gimbal_translation.allFinite())
        {
            gimbal_world_ = gimbal_translation;
        }
    }

    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double stamp = 0.0;
    std::string current = params_.camera_frame;

    // 从相机往上走到目标坐标系；超过树深度就放弃，避免异常 TF 造成死循环
    for (std::int32_t depth = 0; depth < 16; ++depth)
    {
        const auto found = by_child.find(current);
        if (found == by_child.end())
        {
            return;
        }

        const geometry_msgs::msg::TransformStamped& transform = *found->second;
        const Eigen::Matrix3d parent_to_child = ToEigen(transform.transform.rotation).toRotationMatrix();
        const Eigen::Vector3d parent_to_child_t(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);

        // p_parent = R_parent_child·p_child + t_parent_child，逐级往根方向乘
        translation = parent_to_child * translation + parent_to_child_t;
        rotation = parent_to_child * rotation;
        if (depth == 0)
        {
            stamp = rclcpp::Time(transform.header.stamp).seconds();
        }
        current = transform.header.frame_id;

        if (current == params_.aim_frame)
        {
            algorithm::vision::PoseSample sample;
            sample.rotation = Eigen::Quaterniond(rotation);
            sample.translation = translation;
            camera_pose_.Push(stamp, sample);
            return;
        }
    }
}

/**
 * @brief 位姿样本 → 跟踪器用的帧变换
 */
algorithm::vision::FrameTransform VisionNode::MakeFrameTransform(const algorithm::vision::PoseSample& sample)
{
    algorithm::vision::FrameTransform frame;
    frame.rotation = sample.rotation.toRotationMatrix();
    frame.translation = sample.translation;
    return frame;
}

/**
 * @brief ROS 四元数 → Eigen
 */
Eigen::Quaterniond VisionNode::ToEigen(const geometry_msgs::msg::Quaternion& value)
{
    return Eigen::Quaterniond(value.w, value.x, value.y, value.z).normalized();
}

/**
 * @brief 显示窗口是否打开；由参数决定，运行期不变
 */
bool VisionNode::show_window_() const
{
    return params_.show_window;
}

/**
 * @brief 描本帧实测到的每一块板：解出的四点轮廓与置信度
 */
void VisionNode::draw_armors(cv::Mat& drawn, const std::vector<algorithm::vision::DetectedArmor>& armors) const
{
    for (const auto& armor : armors)
    {
        const std::optional<std::array<cv::Point3f, 4>> object_points = algorithm::vision::MakeObjectPoints(armor.GetSize(), algorithm::vision::ArmorDimensions {});
        const std::optional<algorithm::vision::ArmorPose> pose = object_points ? algorithm::vision::SolvePose(armor.GetCorners(), *object_points, *camera_) : std::nullopt;
        if (!pose)
        {
            continue;
        }

        std::vector<cv::Point> contour;
        contour.reserve(pose->projected.size());
        for (const cv::Point2f& point : pose->projected)
        {
            contour.emplace_back(cv::Point(static_cast<std::int32_t>(std::lround(point.x)), static_cast<std::int32_t>(std::lround(point.y))));
        }
        cv::polylines(drawn, contour, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}

/**
 * @brief 描目标的四面：状态推出来的轮廓
 *
 * 照参考工程的叠加层（`visualization/vision_overlay.cpp`）：四面全画（同一套状态，
 * 只是转角相差 2π/armor_num），当前匹配到的那面用绿色，其余三面用黄色；
 * 再有自瞄目标时画红色瞄准点，画面中心画白色准星。参考只要 target 存在就画，
 * 不要求 tracking——否则 detecting/temp_lost 那几帧整组框会消失。
 */
void VisionNode::draw_targets(cv::Mat& drawn) const
{
    if (tracker_->Active() && frame_)
    {
        // 用实测的装甲尺寸：拿小板的物点去投大板，框会明显偏小
        const std::int32_t current_face = tracker_->Report().last_face;
        const std::vector<algorithm::vision::FaceState> faces = tracker_->Faces();
        for (std::size_t index = 0; index < faces.size(); ++index)
        {
            // 当前面绿、其余面黄；四面共用一套状态，只是转角相差一个面序步长
            const cv::Scalar color = (static_cast<std::int32_t>(index) == current_face) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            draw_face(drawn, faces[index], color, 2);
        }
    }

    // 画面中心准星
    const cv::Point center(static_cast<std::int32_t>(std::lround(camera_->cx)), static_cast<std::int32_t>(std::lround(camera_->cy)));
    cv::line(drawn, cv::Point(center.x - 20, center.y), cv::Point(center.x + 20, center.y), cv::Scalar(255, 255, 255), 1);
    cv::line(drawn, cv::Point(center.x, center.y - 20), cv::Point(center.x, center.y + 20), cv::Scalar(255, 255, 255), 1);
}

/**
 * @brief 把一个面按状态投成像素轮廓画出来
 */
void VisionNode::draw_face(cv::Mat& drawn, const algorithm::vision::FaceState& face, const cv::Scalar& color, std::int32_t thickness) const
{
    const std::optional<std::array<cv::Point2f, 4>> points = algorithm::vision::ReprojectArmor(face.xyz, face.yaw_rad, tracker_->Size(), algorithm::vision::ArmorDimensions {}, tracker_config_.vehicle.armor_pitch_rad, *frame_, *camera_);
    if (!points)
    {
        return;
    }

    std::vector<cv::Point> contour;
    contour.reserve(points->size());
    for (const cv::Point2f& point : *points)
    {
        contour.emplace_back(cv::Point(static_cast<std::int32_t>(std::lround(point.x)), static_cast<std::int32_t>(std::lround(point.y))));
    }
    cv::polylines(drawn, contour, true, color, thickness, cv::LINE_AA);
}

/**
 * @brief 把带框图像发布出去，保留原帧的时间戳
 */
void VisionNode::publish_image(const cv::Mat& image, const std_msgs::msg::Header& header)
{
    sensor_msgs::msg::Image out;
    out.header = header;
    out.height = static_cast<std::uint32_t>(image.rows);
    out.width  = static_cast<std::uint32_t>(image.cols);
    out.encoding = "bgr8";
    out.is_bigendian = 0;
    out.step = static_cast<sensor_msgs::msg::Image::_step_type>(image.step);
    out.data.assign(image.data, image.data + image.step * image.rows);

    image_pub_->publish(out);
}

/**
 * @brief 状态机状态的可读名字，供日志用
 */
const char* VisionNode::StateName(algorithm::vision::TrackerState state)
{
    switch (state)
    {
        case algorithm::vision::TrackerState::Lost:      return "lost";
        case algorithm::vision::TrackerState::Detecting: return "detecting";
        case algorithm::vision::TrackerState::Tracking:  return "tracking";
        case algorithm::vision::TrackerState::TempLost:  return "temp_lost";
    }
    return "?";
}

/**
 * @brief 判丢原因的可读名字，供日志用
 */
const char* VisionNode::ResetReasonName(algorithm::vision::ResetReason reason)
{
    switch (reason)
    {
        case algorithm::vision::ResetReason::None:       return "none";
        case algorithm::vision::ResetReason::Nis:        return "nis";
        case algorithm::vision::ResetReason::Diverged:   return "diverged";
        case algorithm::vision::ResetReason::InitFailed: return "init_failed";
    }
    return "?";
}

/**
 * @brief 每秒汇总一次运行状态，不逐帧打
 */
void VisionNode::log_window(const algorithm::vision::ArmorDetectionResult& result, double processing_ms, double age_ms)
{
    ++window_frames_;
    window_processing_ms_ += processing_ms;
    window_age_ms_ += age_ms;
    window_accepted_ += result.stats.accepted;

    const double elapsed = (now() - window_start_).seconds();
    if (elapsed < 1.0)
    {
        return;
    }

    const double denominator = window_frames_ == 0 ? 1.0 : static_cast<double>(window_frames_);
    RCLCPP_INFO(get_logger(), "fps=%.1f accepted=%llu ms=%.2f age=%.1f",
                window_frames_ / elapsed,
                static_cast<unsigned long long>(window_accepted_),
                window_processing_ms_ / denominator,
                window_age_ms_ / denominator);

    RCLCPP_INFO(get_logger(), "  detections=%zu（敌方颜色检测）", last_armors_.size());
    for (const algorithm::vision::DetectedArmor& armor : last_armors_)
    {
        RCLCPP_INFO(get_logger(), "  det color=%s label=%d size=%d center=(%.0f, %.0f) conf=%.2f",
                    armor.GetColor() == algorithm::vision::TeamColor::Red ? "red" : "blue",
                    static_cast<std::int32_t>(armor.GetLabel()),
                    static_cast<std::int32_t>(armor.GetSize()),
                    armor.GetCenter().x, armor.GetCenter().y,
                    static_cast<double>(armor.GetConfidence()));
    }

    const algorithm::vision::TrackerReport& report = tracker_->Report();
    const algorithm::vision::VehicleEKF::Snapshot snapshot = tracker_->DebugSnapshot();
    RCLCPP_INFO(get_logger(), "track state=%s active=%d converged=%d updates=%d losses=%d gate=%d resets=%d reset=%s face=%d cost=%.3f r=%.3fm nis=%.2f nis_rate=%.2f",
                StateName(report.state), report.active ? 1 : 0, report.converged ? 1 : 0,
                report.updates, report.losses, report.gate_rejects, report.resets, ResetReasonName(report.last_reset_reason),
                report.last_face, report.last_cost, tracker_->Radius(), snapshot.last_nis, tracker_->NisFailureRate());
    // 本帧候选、匹配数与归一化代价最低的几次尝试：一眼看出是没候选、配不上还是全被门拒
    RCLCPP_INFO(get_logger(), "  assoc candidates=%zu matches=%zu", report.candidate_count, report.match_count);
    const std::size_t diagnostic_count = std::min<std::size_t>(report.diagnostics.size(), 8);
    for (std::size_t index = 0; index < diagnostic_count; ++index)
    {
        const algorithm::vision::AssociationDiagnostic& diag = report.diagnostics[index];
        RCLCPP_INFO(get_logger(), "    try%d cand=%zu face=%d norm=%.2f center=%.1f/%d corner=%.1f/%d angle=%.2f/%d perimeter=%.2f/%d det_center=(%.0f, %.0f)",
                    static_cast<std::int32_t>(index), diag.candidate, diag.face, diag.normalized_cost,
                    diag.center_error_px, diag.center_ok ? 1 : 0,
                    diag.corner_error_px, diag.corner_ok ? 1 : 0,
                    diag.angle_error_rad, diag.angle_ok ? 1 : 0,
                    diag.perimeter_ratio, diag.perimeter_ok ? 1 : 0,
                    0.25 * (diag.corners[0].x + diag.corners[1].x + diag.corners[2].x + diag.corners[3].x),
                    0.25 * (diag.corners[0].y + diag.corners[1].y + diag.corners[2].y + diag.corners[3].y));
    }
    RCLCPP_INFO(get_logger(), "  obs image=%d ypda=%d limit_clamps=%d pose_missing=%llu",
                snapshot.image_updates, snapshot.ypda_updates, snapshot.limit_clamps,
                static_cast<unsigned long long>(window_pose_missing_));
    const double offset_denominator = window_frames_ == 0 ? 1.0 : static_cast<double>(window_frames_);
    RCLCPP_INFO(get_logger(), "  pose offset avg=%.1fms min=%.1fms max=%.1fms hold=%.0fms",
                window_pose_offset_sum_ / offset_denominator * 1000.0,
                window_pose_offset_min_ * 1000.0, window_pose_offset_max_ * 1000.0, params_.pose_hold_s * 1000.0);

    if (report.active)
    {
        // 滤波器内部状态：位置/速度/角/角速度/半径/半径差/高度差 + 关键协方差
        RCLCPP_INFO(get_logger(), "  ekf center=(%.3f, %.3f, %.3f) vel=(%.2f, %.2f, %.2f) angle=%.1f w=%.2f r=%.4f l=%.4f h=%.4f",
                    snapshot.center.x(), snapshot.center.y(), snapshot.center.z(),
                    snapshot.velocity.x(), snapshot.velocity.y(), snapshot.velocity.z(),
                    snapshot.angle_rad * 180.0 / M_PI, snapshot.omega_rad_s,
                    snapshot.radius_m, snapshot.radius_gap_m, snapshot.height_gap_m);
        RCLCPP_INFO(get_logger(), "  ekf P diag cx=%.4f cy=%.4f cz=%.4f a=%.4f w=%.4f r=%.5f l=%.5f h=%.5f",
                    snapshot.covariance_diag[0], snapshot.covariance_diag[2], snapshot.covariance_diag[4],
                    snapshot.covariance_diag[6], snapshot.covariance_diag[7],
                    snapshot.covariance_diag[8], snapshot.covariance_diag[9], snapshot.covariance_diag[10]);
        // 形状量是否被观测约束：雅可比范数为 0 表示本帧完全没约束到；修正量看它被拖向哪边
        RCLCPP_INFO(get_logger(), "  ekf shape jac=(%.3f, %.3f, %.3f) step=(%.5f, %.5f, %.5f) estimate=%d faces=%zu",
                    snapshot.shape_jacobian_norm_r, snapshot.shape_jacobian_norm_l, snapshot.shape_jacobian_norm_h,
                    snapshot.shape_radius_step, snapshot.shape_radius_gap_step, snapshot.shape_height_gap_step,
                    report.match_count >= 2 ? 1 : 0, report.match_count);

        // 四个面在跟踪系（世界系）的中心；与当帧检测的板心对照，能区分"状态偏了"和"投影错了"
        const std::vector<algorithm::vision::FaceState> faces = tracker_->Faces();
        for (std::size_t index = 0; index < faces.size(); ++index)
        {
            RCLCPP_INFO(get_logger(), "  face=%d xyz_world=(%.3f, %.3f, %.3f) yaw=%.1fdeg",
                        static_cast<std::int32_t>(index),
                        faces[index].xyz.x(), faces[index].xyz.y(), faces[index].xyz.z(), faces[index].yaw_rad * 180.0 / M_PI);
        }
    }

    // 当帧检测的板心（解 PnP 得到），与上面的预测对照
    for (const algorithm::vision::DetectedArmor& armor : last_armors_)
    {
        const std::optional<std::array<cv::Point3f, 4>> object_points = algorithm::vision::MakeObjectPoints(armor.GetSize(), algorithm::vision::ArmorDimensions {});
        const std::optional<algorithm::vision::ArmorPose> pose = object_points ? algorithm::vision::SolvePose(armor.GetCorners(), *object_points, *camera_) : std::nullopt;
        if (pose)
        {
            RCLCPP_INFO(get_logger(), "  observed xyz_cam=(%.3f, %.3f, %.3f)", pose->tvec[0], pose->tvec[1], pose->tvec[2]);
        }
    }

    window_frames_ = 0;
    window_processing_ms_ = 0.0;
    window_age_ms_ = 0.0;
    window_accepted_ = 0;
    window_pose_missing_ = 0;
    window_pose_offset_sum_ = 0.0;
    window_pose_offset_min_ = 0.0;
    window_pose_offset_max_ = 0.0;
    window_start_ = now();
}
