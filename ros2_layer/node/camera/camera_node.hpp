/**
 * @file camera_node.hpp
 * @author qingyu
 * @brief 相机适配节点：订阅仿真/实机图像，统一成 BGR 后转发给视觉链
 * @version 0.1
 * @date 2026-10-17
 *
 * @copyright Copyright (c) 2026
 *
 * @note 写成头文件而不是 .cpp：类定义连实现全 inline，project 那层直接
 *       #include "ros2_layer/node/camera/camera_node.hpp" 然后 make_shared<CameraNode>()。
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

/**
 * @brief 相机适配节点
 *
 * 职责只有两件：把任意编码的图像统一成 BGR8 转发，并把 `/camera_info` 一并转出去。
 * 算法层只认 BGR（OpenCV 默认、绘制颜色也按 BGR 写），编码判断集中在这一个节点里。
 *
 * @note 仿真发 `rgb8`、实机发 `bgr8`，所以必须按 `msg.encoding` 分支，不能假定。
 */
class CameraNode : public rclcpp::Node
{
public:
    explicit CameraNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : rclcpp::Node("camera", options)
    {
        input_topic_  = declare_parameter("input_topic", "/image_raw");
        output_topic_ = declare_parameter("output_topic", "/vision/image");
        info_input_   = declare_parameter("camera_info_topic", "/camera_info");
        info_output_  = declare_parameter("camera_info_output_topic", "/vision/camera_info");

        // 当前实现固定 BGR8 输出：预分配一份缓冲，回调里原地转换，避免每帧重新分配
        output_pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, rclcpp::SensorDataQoS());
        info_pub_   = create_publisher<sensor_msgs::msg::CameraInfo>(info_output_, rclcpp::SensorDataQoS());

        info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            info_input_, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { info_pub_->publish(*msg); });

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            input_topic_, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_image(*msg); });

        RCLCPP_INFO(get_logger(), "相机适配：%s (%s) -> %s", input_topic_.c_str(), output_topic_.c_str(), info_output_.c_str());
    }

private:
    std::string input_topic_;
    std::string output_topic_;
    std::string info_input_;
    std::string info_output_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr      image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         output_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr    info_pub_;

    cv::Mat bgr_;   // 复用的 BGR 缓冲，尺寸变化时重新分配

    /**
     * @brief 把一帧 ROS 图像转成 BGR8 并转发
     *
     * 按 `encoding` 分支：`rgb8` 用 cvtColor 转，`bgr8` 只去行尾填充。其他编码直接丢弃并
     * 限频告警——算法层假定 BGR，放错编码进去会导致红蓝互换而检测颜色分支失效。
     *
     * @param msg ROS 图像消息
     */
    void on_image(const sensor_msgs::msg::Image& msg)
    {
        const std::size_t row = static_cast<std::size_t>(msg.width) * 3;
        if (!msg.width || !msg.height || msg.step < row || msg.width  > static_cast<unsigned>(std::numeric_limits<std::int32_t>::max()) ||
             msg.height > static_cast<unsigned>(std::numeric_limits<std::int32_t>::max()) || static_cast<std::size_t>(msg.height - 1) * msg.step + row > msg.data.size())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "忽略非法图像尺寸或步长");
            return;
        }

        const auto width  = static_cast<std::int32_t>(msg.width);
        const auto height = static_cast<std::int32_t>(msg.height);

        // 编码在拷贝之前判：不支持的编码直接丢，不做无谓的 4 MB 搬运
        const bool rgb_input = (msg.encoding == "rgb8");
        if (!rgb_input && msg.encoding != "bgr8")
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "不支持的编码：%s", msg.encoding.c_str());
            return;
        }

        // 按行拷进连续缓冲，去掉 ROS 图像可能存在的行尾填充
        if (bgr_.rows != height || bgr_.cols != width || bgr_.type() != CV_8UC3)
        {
            bgr_.create(height, width, CV_8UC3);
        }

        for (std::int32_t y = 0; y < height; ++y)
        {
            std::copy_n(msg.data.data() + static_cast<std::size_t>(y) * msg.step, row, bgr_.ptr<std::uint8_t>(y));
        }

        if (rgb_input)
        {
            cv::cvtColor(bgr_, bgr_, cv::COLOR_RGB2BGR);
        }

        sensor_msgs::msg::Image out;
        out.header = msg.header;
        out.height = msg.height;
        out.width  = msg.width;
        out.encoding = "bgr8";
        out.is_bigendian = 0;
        out.step = static_cast<sensor_msgs::msg::Image::_step_type>(row);
        out.data.assign(bgr_.data, bgr_.data + static_cast<std::size_t>(row) * msg.height);

        output_pub_->publish(out);
    }
};
