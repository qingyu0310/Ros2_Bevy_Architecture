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
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "framework/algorithm/math/angle.hpp"
#include <tf2/exceptions.h>

VisionNode::VisionNode(const rclcpp::NodeOptions& options) : rclcpp::Node("vision", options)
{
    read_parameters();

    detector_ = std::make_unique<algorithm::vision::ArmorDetector>(detector_config_);
    tracker_  = std::make_unique<algorithm::vision::Tracker>(tracker_config_);
    window_start_ = now();

    if (params_.drop_dump_limit > 0)
    {
        std::error_code error;
        std::filesystem::create_directories(params_.drop_dump_dir, error);
        if (error)
        {
            RCLCPP_WARN(get_logger(), "漏检帧目录建不出来：%s（%s），本次不落盘", params_.drop_dump_dir.c_str(), error.message().c_str());
        }
    }

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

    auto_aim_sub_ = create_subscription<std_msgs::msg::Bool>(
        params_.auto_aim_topic, rclcpp::QoS(10),
        [this](std_msgs::msg::Bool::ConstSharedPtr msg) { on_auto_aim(*msg); });

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(params_.output_image_topic, rclcpp::SensorDataQoS());

    if (params_.aim_enabled)
    {
        algorithm::vision::AimPublisherConfig aim_config;
        aim_config.topic = params_.aim_topic;
        aim_config.map_frame = params_.aim_frame;
        aim_config.camera_frame = params_.camera_frame;
        aim_config.fire_angle_deg = params_.aim_fire_angle_deg;
        aim_config.distance_max_m = params_.aim_distance_max_m;
        aim_config.aim_sector_deg = params_.aim_sector_deg;
        aim_config.coming_angle_deg = params_.coming_angle_deg;
        aim_config.leaving_angle_deg = params_.leaving_angle_deg;
        aim_config.spin_omega_threshold = params_.spin_omega_threshold;
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
    params_.aim_sector_deg = declare_parameter("aim_sector_deg", params_.aim_sector_deg);
    params_.coming_angle_deg = declare_parameter("coming_angle_deg", params_.coming_angle_deg);
    params_.leaving_angle_deg = declare_parameter("leaving_angle_deg", params_.leaving_angle_deg);
    params_.spin_omega_threshold = declare_parameter("spin_omega_threshold", params_.spin_omega_threshold);
    params_.aim_fixed_yaw_pitch = declare_parameter("aim_fixed_yaw_pitch", params_.aim_fixed_yaw_pitch);
    params_.auto_aim_topic     = declare_parameter("auto_aim_topic", params_.auto_aim_topic);
    params_.drop_dump_dir      = declare_parameter("drop_dump_dir", params_.drop_dump_dir);
    params_.drop_dump_limit    = declare_parameter("drop_dump_limit", params_.drop_dump_limit);
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
    tracker_config_.aim_switch_cost                  = declare_parameter("aim_switch_cost",        tracker_config_.aim_switch_cost);
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

    const rclcpp::Time stamp(msg.header.stamp);

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

    // 目标两块掉成一块时留下现场：整帧分数一起掉说明是图像本身坏了，不是某个目标卡门限
    if (last_enemy_count_ >= 2 && enemy_armors.size() < last_enemy_count_)
    {
        detect_drops_++;
        if (detect_drops_ <= params_.drop_dump_limit)
        {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::Scalar mean;
            cv::Scalar deviation;
            cv::meanStdDev(gray, mean, deviation);

            cv::imwrite(params_.drop_dump_dir + "/drop_" + std::to_string(detect_drops_) + ".png", image);
            RCLCPP_INFO(get_logger(), "det drop #%d kept=%zu -> %zu max_obj=%.2f score_pass=%zu nms=%zu rej_conf=%zu | gray mean=%.1f std=%.1f | %s/drop_%d.png",
                        detect_drops_, last_enemy_count_, enemy_armors.size(), result.stats.max_objectness,
                        result.stats.score_passed, result.stats.nms_kept, result.stats.rejected_confidence,
                        mean[0], deviation[0], params_.drop_dump_dir.c_str(), detect_drops_);
        }
        else if (detect_drops_ == params_.drop_dump_limit + 1)
        {
            RCLCPP_INFO(get_logger(), "det drop: 已抓够 %d 张，后续只计数不落盘", params_.drop_dump_limit);
        }
    }
    last_enemy_count_ = enemy_armors.size();

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
    // 插值跨度：At() 在这两条之间做 slerp，跨度越大插值越不可靠
    if (pose_at_image && pose_latest > image_stamp)
    {
        window_pose_span_sum_ += pose_latest - image_stamp;
        ++window_pose_span_count_;
    }
    window_pose_offset_min_ = std::min(window_pose_offset_min_, image_stamp - pose_latest);
    window_pose_offset_max_ = std::max(window_pose_offset_max_, image_stamp - pose_latest);
    if (frame_)
    {
        // P0：把本帧检测和仿真真值配一次，量"配对本身对不对"，不参与跟踪
        pair_armor_truth(enemy_armors, image_stamp);

        // 无论自瞄键是否按住都跑跟踪：键只决定要不要把指令发给云台。
        {
            // dt 用距上一次真正推进跟踪器的时间：姿态缺失的帧整帧跳过，不推进也不记时间戳。
            // 用"距上一张图像"的话，丢一帧后下一帧只补一个图像间隔，实际过了两帧，
            // 小陀螺下就是几度的转角预测误差，直接掉出关联门
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
            tracker_->ProcessFrame(enemy_armors, algorithm::vision::ArmorDimensions {}, *camera_, *frame_, dt_s);

            // P4：车心漂移。锁定（状态从 lost 建起）时记起点，满 1 s 后算位移。
            // 用 resets 计数判断"是不是新锁的"：重置后 tracker 回到 lost，下一次建目标就是新起点。
            const algorithm::vision::TrackerReport& drift_report = tracker_->Report();
            if (drift_report.state != algorithm::vision::TrackerState::Lost && drift_report.active)
            {
                if (!lock_center_)
                {
                    const algorithm::vision::VehicleEKF::Snapshot snapshot = tracker_->DebugSnapshot();
                    lock_center_ = snapshot.center;
                    lock_time_ = stamp;
                }
                else if (lock_time_)
                {
                    const double held = (stamp - *lock_time_).seconds();
                    if (held >= 1.0)
                    {
                        const algorithm::vision::VehicleEKF::Snapshot snapshot = tracker_->DebugSnapshot();
                        const double drift = (snapshot.center - *lock_center_).norm();
                        ++window_drift_1s_count_;
                        window_drift_1s_sum_ += drift;
                        window_drift_max_ = std::max(window_drift_max_, drift);
                        lock_center_.reset();
                        lock_time_.reset();
                    }
                }
            }
            else
            {
                lock_center_.reset();
                lock_time_.reset();
            }

            // 换面现场：记下这一帧候选/匹配数与各次配对的门限分项，判断是关联只剩一条
            // 还是双板帧里打分换了面
            const std::int32_t face = tracker_->Report().last_face;
            if (last_face_seen_ >= 0 && face != last_face_seen_)
            {
                const algorithm::vision::TrackerReport& report = tracker_->Report();
                RCLCPP_INFO(get_logger(), "face switch %d -> %d kept=%zu candidates=%zu matches=%zu locked label=%d size=%d",
                            last_face_seen_, face, enemy_armors.size(), report.candidate_count, report.match_count,
                            static_cast<std::int32_t>(tracker_->Label()), static_cast<std::int32_t>(tracker_->Size()));
            }
            last_face_seen_ = face;
        }
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
    // 只按自瞄键时发指令；松开后仿真端已经不再订阅云台，视觉这侧也不该继续发目标
    if (aim_publisher_ && auto_aim_pressed_)
    {
        aim_publisher_->Publish(msg.header, *tracker_, frame_, gimbal_world_);
    }
    log_window(result, processing_ms, age_ms);
}

/**
 * @brief 记录自瞄键状态
 *
 * 只有按住期间才跑跟踪和发指令；松开期间每帧都把车型重置，所以下次按下会重新选板。
 */
void VisionNode::on_auto_aim(const std_msgs::msg::Bool& msg)
{
    auto_aim_pressed_ = msg.data;
}

/**
 * @brief 缓存 `map ← 相机` 的位姿
 *
 * `/tf` 每条消息只带一条父子边，`map → 相机` 不是一条边而是若干段拼起来的。这里在同一
 * 批 TF 里按子坐标系建索引，从相机逐级走到目标坐标系，把各段变换乘起来。
 */
void VisionNode::on_tf(const tf2_msgs::msg::TFMessage& msg)
{
    update_armor_truth(msg);

    // TF 从产生到被处理的时间：图像链路与 TF 链路各自有延迟，两边不一致就是时间错位的来源。
    // 同时记到达间隔：间隔≈发布周期说明是单条消息在传输/排队里等，间隔本身就大说明仿真发得稀。
    if (!msg.transforms.empty())
    {
        const rclcpp::Time tf_time(msg.transforms.front().header.stamp);
        const rclcpp::Time arrival = now();
        const double tf_stamp = tf_time.seconds();
        const double delay = arrival.seconds() - tf_stamp;
        if (delay >= 0.0 && delay < 1.0)
        {
            window_tf_delay_sum_ += delay;
            ++window_tf_delay_count_;
        }

        if (last_tf_arrival_)
        {
            const double interval = (arrival - *last_tf_arrival_).seconds();
            if (interval > 0.0 && interval < 0.5)
            {
                window_tf_interval_sum_ += interval;
                ++window_tf_interval_count_;
            }
        }
        last_tf_arrival_ = arrival;

        // 发布侧自身的抖动：相邻两条消息的仿真时间戳差，与到达间隔对照
        if (last_tf_stamp_)
        {
            const double stamp_interval = tf_stamp - *last_tf_stamp_;
            if (stamp_interval > 0.0 && stamp_interval < 0.5)
            {
                window_tf_stamp_interval_sum_ += stamp_interval;
                ++window_tf_stamp_interval_count_;
            }
        }
        last_tf_stamp_ = tf_stamp;
    }

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
 * @brief 缓存仿真发的装甲板真值
 *
 * `/tf` 里每条 `armor_<N>` 的 `header.frame_id` 是 `map`、`child_frame_id` 是 `armor_N`，
 * 变换是板心在 map 中的位姿，可以直接当解算的真值。N 来自仿真的构建顺序，不保证连续，
 * 所以按解析出来的下标存，越界的丢掉。
 */
void VisionNode::update_armor_truth(const tf2_msgs::msg::TFMessage& msg)
{
    for (const geometry_msgs::msg::TransformStamped& transform : msg.transforms)
    {
        const std::string& child = transform.child_frame_id;
        if (child.compare(0, 6, "armor_") != 0 || transform.header.frame_id != params_.aim_frame)
        {
            continue;
        }

        const std::string index_text = child.substr(6);
        if (index_text.empty())
        {
            continue;
        }

        std::int32_t index = 0;
        bool valid = true;
        for (const char digit : index_text)
        {
            if (digit < '0' || digit > '9')
            {
                valid = false;
                break;
            }
            index = index * 10 + (digit - '0');
        }
        if (!valid || index < 0 || static_cast<std::size_t>(index) >= armor_truth_.size())
        {
            continue;
        }

        algorithm::vision::PoseSample truth;
        truth.rotation = ToEigen(transform.transform.rotation);
        truth.translation = Eigen::Vector3d(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);
        armor_truth_[static_cast<std::size_t>(index)].Push(rclcpp::Time(transform.header.stamp).seconds(), truth);
    }
}

/**
 * @brief 把本帧检测和真值配对，统计配对距离与 PnP 误差
 *
 * 配对用 map 系下板心位置：检测先按 PnP 解出相机系位姿、再换到 map，取距离最近的真值板。
 * 这里只用最近邻，不保证一对一——单板场景下一块检测对上一块真值，P0 要的是"配对本身对得上"。
 *
 * 配对成功后按相机视线拆横向/深度误差：横向由像素精度决定，深度由板宽与距离决定，
 * 两者量级差一个数量级，混在一起看不出 PnP 到底差在哪一项。
 */
void VisionNode::pair_armor_truth(const std::vector<algorithm::vision::DetectedArmor>& armors, double image_stamp)
{
    if (!frame_ || !camera_)
    {
        return;
    }

    std::uint64_t frame_ok = 0;
    for (const algorithm::vision::DetectedArmor& armor : armors)
    {
        const std::optional<std::array<cv::Point3f, 4>> object_points = algorithm::vision::MakeObjectPoints(armor.GetSize(), algorithm::vision::ArmorDimensions {});
        const std::optional<algorithm::vision::ArmorPose> pose = object_points ? algorithm::vision::SolvePose(armor.GetCorners(), *object_points, *camera_) : std::nullopt;
        if (!pose)
        {
            ++window_pair_fail_;
            continue;
        }

        const Eigen::Vector3d center_camera(pose->tvec[0], pose->tvec[1], pose->tvec[2]);
        const Eigen::Vector3d center_map = algorithm::vision::CameraToFrame(center_camera, *frame_);

        // 真值按图像时刻取：小陀螺下板心 1.5 m/s，用最新一条会引入几十毫米的假误差
        double best_distance = std::numeric_limits<double>::max();
        std::optional<algorithm::vision::PoseSample> best_truth;
        std::size_t best_truth_index = 0;
        for (std::size_t index = 0; index < armor_truth_.size(); ++index)
        {
            const std::optional<algorithm::vision::PoseSample> truth = armor_truth_[index].At(image_stamp, params_.pose_hold_s);
            if (!truth)
            {
                continue;
            }
            const double distance = (center_map - truth->translation).norm();
            if (distance < best_distance)
            {
                best_distance = distance;
                best_truth = truth;
                best_truth_index = index;
            }
        }

        // 超过半米就不是同一块板：宁可记失败，也不让错误配对污染后面的精度统计
        if (!best_truth || best_distance > 0.5)
        {
            ++window_pair_fail_;
            continue;
        }

        ++frame_ok;
        ++window_pair_ok_;

        // 误差按相机视线拆开：û 是检测点在相机系的方向，横向与深度用同一组真值算
        const Eigen::Vector3d error_map = center_map - best_truth->translation;
        const Eigen::Vector3d error_camera = frame_->rotation.transpose() * error_map;
        const Eigen::Vector3d view = center_camera.norm() > 0.0 ? center_camera.normalized() : Eigen::Vector3d::UnitZ();
        // 深度误差保留符号：取绝对值后它的均值恒为正，会冒充成"有偏"
        const double depth_error = error_camera.dot(view);
        const double lateral_error = (error_camera - error_camera.dot(view) * view).norm();

        ++window_pnp_count_;
        // 有偏性判据：各向同性的角点噪声只会让 |mean| << rms；两者接近说明是固定偏置
        window_error_sum_ += error_camera;
        window_error_square_sum_ += error_camera.squaredNorm();
        window_pnp_lat_sum_ += lateral_error;
        window_pnp_lat_max_ = std::max(window_pnp_lat_max_, lateral_error);
        window_pnp_depth_sum_ += depth_error;
        window_pnp_depth_abs_sum_ += std::abs(depth_error);
        window_pnp_depth_max_ = std::max(window_pnp_depth_max_, std::abs(depth_error));
        window_pnp_reproj_sum_ += pose->reprojection_error_px;
        window_pnp_reproj_max_ = std::max(window_pnp_reproj_max_, pose->reprojection_error_px);

        // 时间偏移估计：真值有速度，误差沿速度方向的分量直接给出"我们的时刻比真值早/晚多少"。
        // dt = (error · v) / |v|² 是单帧最小二乘解；窗口内取平均就是这一秒的有效时刻差。
        // 这是 P2 的核心量：横向 bias 若主要来自时刻错位，dt 乘速度应当正好等于 bias。
        {
            const algorithm::vision::PoseBuffer& buffer = armor_truth_[best_truth_index];
            const std::optional<algorithm::vision::PoseSample> later = buffer.At(image_stamp + 0.01, 0.0);
            const std::optional<algorithm::vision::PoseSample> earlier = buffer.At(image_stamp - 0.01, 0.0);
            if (later && earlier)
            {
                const Eigen::Vector3d velocity = (later->translation - earlier->translation) / 0.02;
                const double speed_squared = velocity.squaredNorm();
                if (speed_squared > 0.01)
                {
                    window_time_offset_sum_ += error_map.dot(velocity) / speed_squared;
                    ++window_time_offset_count_;
                }
            }
        }

        // 法向 yaw 误差：真值板系的 x 轴（法向）与 PnP 法向都投到 map 水平面取方位角。
        // 装甲板法向是一条轴而不是方向——朝里朝外是同一块板，真值帧的 x 轴指向与
        // `ArmorRotation` 的约定差 π，所以按轴比，取两个方向里小的那个；同时数翻转次数，
        // 翻转频繁就是 IPPE 二义没被 `OptimizeYaw` 压住。
        const Eigen::Vector3d true_normal = best_truth->rotation.toRotationMatrix().col(0);
        const double true_yaw = std::atan2(true_normal.y(), true_normal.x());
        cv::Matx33d rotation;
        cv::Rodrigues(cv::Mat(pose->rvec), rotation);
        const Eigen::Vector3d camera_normal(rotation(0, 0), rotation(1, 0), rotation(2, 0));
        const Eigen::Vector3d pnp_normal = frame_->rotation * camera_normal;
        const double pnp_yaw = std::atan2(pnp_normal.y(), pnp_normal.x());
        const double forward_error = std::abs(algorithm::math::normalize_radian_pm_pi(pnp_yaw - true_yaw));
        const double flipped_error = std::abs(algorithm::math::normalize_radian_pm_pi(pnp_yaw - true_yaw - M_PI));
        const double yaw_error = std::min(forward_error, flipped_error);
        const bool flipped = flipped_error < forward_error;
        if (yaw_branch_valid_ && flipped != yaw_branch_flipped_)
        {
            ++window_pnp_yaw_flips_;
        }
        yaw_branch_valid_ = true;
        yaw_branch_flipped_ = flipped;
        window_pnp_yaw_sum_ += yaw_error;
        window_pnp_yaw_max_ = std::max(window_pnp_yaw_max_, yaw_error);
    }

    if (frame_ok == 0)
    {
        ++window_pair_empty_;
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
 * 只是转角相差 2π/armor_num）。瞄准面红色、本帧匹配到的面绿色、其余三面黄色；
 * 画面中心画白色准星。参考只要 target 存在就画，不要求 tracking——否则
 * detecting/temp_lost 那几帧整组框会消失。
 *
 * 瞄准面与本帧匹配面不是一回事：小陀螺下匹配面每 0.21 s 换一块，瞄准面要按
 * 可射击性自己选，两个面序经常不同。
 */
void VisionNode::draw_targets(cv::Mat& drawn) const
{
    if (tracker_->Active() && frame_)
    {
        // 用实测的装甲尺寸：拿小板的物点去投大板，框会明显偏小
        const std::int32_t current_face = tracker_->Report().last_face;
        const std::int32_t aim_face = aim_publisher_ ? aim_publisher_->aim_face() : -1;
        const std::vector<algorithm::vision::FaceState> faces = tracker_->Faces();
        for (std::size_t index = 0; index < faces.size(); ++index)
        {
            // 瞄准面红、本帧匹配面绿、其余面黄；四面共用一套状态，只是转角相差一个面序步长
            const std::int32_t face_index = static_cast<std::int32_t>(index);
            cv::Scalar color(0, 255, 255);
            if (face_index == aim_face)
            {
                color = cv::Scalar(0, 0, 255);
            }
            else if (face_index == current_face)
            {
                color = cv::Scalar(0, 255, 0);
            }
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
        case algorithm::vision::ResetReason::Manual:     return "manual";
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
    window_stats_.score_passed        += result.stats.score_passed;
    window_stats_.nms_kept            += result.stats.nms_kept;
    window_stats_.rejected_confidence += result.stats.rejected_confidence;
    window_stats_.rejected_color      += result.stats.rejected_color;
    window_stats_.rejected_label_type += result.stats.rejected_label_type;
    window_stats_.rejected_geometry   += result.stats.rejected_geometry;
    window_stats_.max_objectness       = std::max(window_stats_.max_objectness, result.stats.max_objectness);
    const std::size_t enemy_count = last_armors_.size();
    window_enemy_min_ = window_enemy_min_ == 0 ? enemy_count : std::min(window_enemy_min_, enemy_count);

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

    // 逐级计数：远处目标若漏检，先看它掉在哪一级（初筛 / NMS / 终选 / 颜色 / 标签 / 几何）
    RCLCPP_INFO(get_logger(), "  det stats score_pass=%llu nms=%llu acc=%llu | rej conf=%llu color=%llu label=%llu geom=%llu | max_obj=%.2f min_enemy=%zu",
                static_cast<unsigned long long>(window_stats_.score_passed),
                static_cast<unsigned long long>(window_stats_.nms_kept),
                static_cast<unsigned long long>(window_accepted_),
                static_cast<unsigned long long>(window_stats_.rejected_confidence),
                static_cast<unsigned long long>(window_stats_.rejected_color),
                static_cast<unsigned long long>(window_stats_.rejected_label_type),
                static_cast<unsigned long long>(window_stats_.rejected_geometry),
                static_cast<double>(window_stats_.max_objectness), window_enemy_min_);

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
    RCLCPP_INFO(get_logger(), "track state=%s active=%d converged=%d updates=%d losses=%d gate=%d resets=%d reset=%s face=%d cost=%.3f r=%.3fm nis=%.2f nis_rate=%.2f resid=%.2fpx",
                StateName(report.state), report.active ? 1 : 0, report.converged ? 1 : 0,
                report.updates, report.losses, report.gate_rejects, report.resets, ResetReasonName(report.last_reset_reason),
                report.last_face, report.last_cost, tracker_->Radius(), snapshot.last_nis, tracker_->NisFailureRate(),
                snapshot.residual_rms_px);
    // 本帧候选与匹配数：一眼看出是没候选、配不上还是全被门拒
    RCLCPP_INFO(get_logger(), "  assoc candidates=%zu matches=%zu", report.candidate_count, report.match_count);
    RCLCPP_INFO(get_logger(), "  obs image=%d ypda=%d limit_clamps=%d pose_missing=%llu",
                snapshot.image_updates, snapshot.ypda_updates, snapshot.limit_clamps,
                static_cast<unsigned long long>(window_pose_missing_));
    // 真值配对成功率（P0 判据）+ PnP 误差分解（P1 判据）
    RCLCPP_INFO(get_logger(), "  truth pair ok=%llu fail=%llu empty_frames=%llu",
                static_cast<unsigned long long>(window_pair_ok_),
                static_cast<unsigned long long>(window_pair_fail_),
                static_cast<unsigned long long>(window_pair_empty_));
    if (window_pnp_count_ > 0)
    {
        const double count = static_cast<double>(window_pnp_count_);
        // mean 是误差向量的平均（有偏部分），rms 是均方根（有偏 + 随机）。
        // |mean| ≈ rms：固定偏置（真值参考点与我们的板心不是一个点）；|mean| << rms：随机噪声。
        const Eigen::Vector3d error_mean = window_error_sum_ / count;
        const double error_rms = std::sqrt(window_error_square_sum_ / count);
        RCLCPP_INFO(get_logger(), "  pnp lat=%.1f/%.1fmm depth=%.2f/%.2fm reproj=%.2f/%.2fpx yaw=%.1f/%.1fdeg flips=%llu bias=%.1fmm rms=%.1fmm n=%llu",
                    window_pnp_lat_sum_ / count * 1000.0, window_pnp_lat_max_ * 1000.0,
                    window_pnp_depth_sum_ / count, window_pnp_depth_max_,
                    window_pnp_reproj_sum_ / count, window_pnp_reproj_max_,
                    window_pnp_yaw_sum_ / count * 180.0 / M_PI, window_pnp_yaw_max_ * 180.0 / M_PI,
                    static_cast<unsigned long long>(window_pnp_yaw_flips_),
                    error_mean.norm() * 1000.0, error_rms * 1000.0,
                    static_cast<unsigned long long>(window_pnp_count_));
        // 深度误差单独给带符号与绝对值两列：符号看系统偏近还是偏远，绝对值看幅度
        RCLCPP_INFO(get_logger(), "  depth signed=%.3fm abs=%.3fm",
                    window_pnp_depth_sum_ / count,
                    window_pnp_depth_abs_sum_ / count);
        // 有效时刻差：正数表示"图像内容对应的时刻比 image_stamp 早"，是 P2 要消掉的量
        if (window_time_offset_count_ > 0)
        {
            RCLCPP_INFO(get_logger(), "  time offset=%.1fms n=%llu",
                        window_time_offset_sum_ / static_cast<double>(window_time_offset_count_) * 1000.0,
                        static_cast<unsigned long long>(window_time_offset_count_));
        }
    }
    // P4：锁定满 1 s 的车心漂移
    if (window_drift_1s_count_ > 0)
    {
        RCLCPP_INFO(get_logger(), "  drift1s avg=%.3fm max=%.3fm n=%llu",
                    window_drift_1s_sum_ / static_cast<double>(window_drift_1s_count_),
                    window_drift_max_,
                    static_cast<unsigned long long>(window_drift_1s_count_));
    }
    const double offset_denominator = window_frames_ == 0 ? 1.0 : static_cast<double>(window_frames_);
    RCLCPP_INFO(get_logger(), "  pose offset avg=%.1fms min=%.1fms max=%.1fms hold=%.0fms",
                window_pose_offset_sum_ / offset_denominator * 1000.0,
                window_pose_offset_min_ * 1000.0, window_pose_offset_max_ * 1000.0, params_.pose_hold_s * 1000.0);
    // P2：三条时间量的对照。tf_delay 是 TF 链路延迟，span 是插值跨度，
    // time offset 是真值给出的"图像内容实际时刻"相对 image_stamp 的差。
    {
        const double tf_denominator = window_tf_delay_count_ == 0 ? 1.0 : static_cast<double>(window_tf_delay_count_);
        const double span_denominator = window_pose_span_count_ == 0 ? 1.0 : static_cast<double>(window_pose_span_count_);
        const double interval_denominator = window_tf_interval_count_ == 0 ? 1.0 : static_cast<double>(window_tf_interval_count_);
        const double stamp_denominator = window_tf_stamp_interval_count_ == 0 ? 1.0 : static_cast<double>(window_tf_stamp_interval_count_);
        RCLCPP_INFO(get_logger(), "  timing tf_delay=%.1fms span=%.1fms tf_gap=%.1fms tf_stamp_gap=%.1fms",
                    window_tf_delay_sum_ / tf_denominator * 1000.0,
                    window_pose_span_sum_ / span_denominator * 1000.0,
                    window_tf_interval_sum_ / interval_denominator * 1000.0,
                    window_tf_stamp_interval_sum_ / stamp_denominator * 1000.0);
    }

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
    }

    window_frames_ = 0;
    window_processing_ms_ = 0.0;
    window_age_ms_ = 0.0;
    window_accepted_ = 0;
    window_pose_missing_ = 0;
    window_stats_ = algorithm::vision::ArmorDetectionStats {};
    window_enemy_min_ = 0;
    window_pose_offset_sum_ = 0.0;
    window_drift_1s_sum_ = 0.0;
    window_drift_1s_count_ = 0;
    window_drift_max_ = 0.0;
    window_pose_offset_min_ = 0.0;
    window_pose_offset_max_ = 0.0;
    window_pair_ok_ = 0;
    window_pair_fail_ = 0;
    window_pair_empty_ = 0;
    window_pnp_count_ = 0;
    window_error_sum_.setZero();
    window_error_square_sum_ = 0.0;
    window_time_offset_sum_ = 0.0;
    window_time_offset_count_ = 0;
    window_tf_delay_sum_ = 0.0;
    window_tf_delay_count_ = 0;
    window_tf_interval_sum_ = 0.0;
    window_tf_interval_count_ = 0;
    window_tf_stamp_interval_sum_ = 0.0;
    window_tf_stamp_interval_count_ = 0;
    window_pose_span_sum_ = 0.0;
    window_pose_span_count_ = 0;
    window_pnp_lat_sum_ = 0.0;
    window_pnp_lat_max_ = 0.0;
    window_pnp_depth_sum_ = 0.0;
    window_pnp_depth_abs_sum_ = 0.0;
    window_pnp_depth_max_ = 0.0;
    window_pnp_reproj_sum_ = 0.0;
    window_pnp_reproj_max_ = 0.0;
    window_pnp_yaw_sum_ = 0.0;
    window_pnp_yaw_max_ = 0.0;
    window_pnp_yaw_flips_ = 0;
    window_start_ = now();
}
