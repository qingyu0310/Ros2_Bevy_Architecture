/**
 * @file light_bar.hpp
 * @author qingyu
 * @brief 规范化二维灯条 — 几何派生量与只能由工厂构造的灯条类
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/armor_types.hpp"

#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>

namespace algorithm::vision
{

/**
 * @brief 灯条的二维几何：由长轴两端点和短轴宽度确定，其余量全部派生
 */
struct LightBarGeometry
{
    cv::Point2f center;   // 长轴中点，pixel
    cv::Point2f top;      // 长轴上端点，pixel
    cv::Point2f bottom;   // 长轴下端点，pixel
    float length;         // 长轴长度，pixel
    float width;          // 短轴宽度，pixel
    float tilt_rad;       // 相对图像竖直方向倾角，rad

    /**
     * @brief 由长轴两端点和短轴宽度算出中心、长度与倾角
     *
     * @param top 长轴上端点，单位 pixel
     * @param bottom 长轴下端点，单位 pixel
     * @param width 短轴宽度，单位 pixel
     * @return 各派生量填好的几何结构
     */
    static LightBarGeometry FromEndpoints(const cv::Point2f& top, const cv::Point2f& bottom, float width)
    {
        LightBarGeometry shape {};
        const cv::Point2f delta = bottom - top;

        shape.top      = top;
        shape.bottom   = bottom;
        shape.width    = width;
        shape.center   = (top + bottom) * 0.5f;
        shape.length   = std::sqrt(delta.dot(delta));
        shape.tilt_rad = std::atan2(bottom.x - top.x, bottom.y - top.y);

        return shape;
    }
};

/**
 * @brief 规范化二维灯条
 *
 * 类只允许通过 Create() 或 CreateFromEndpoints() 构造，确保端点始终由上到下、
 * 长宽为正、中心和倾角与端点一致。图像轮廓筛选参数不属于本类。
 */
class LightBar final
{
public:
    static std::optional<LightBar> Create(const cv::RotatedRect& rect, std::uint32_t id, TeamColor color = TeamColor::Unknown, float color_score = 0.0f);
    static std::optional<LightBar> CreateFromEndpoints(std::uint32_t id, const cv::Point2f& top_or_bottom, const cv::Point2f& bottom_or_top, float width, TeamColor color, float color_score);

    /**
     * @brief 返回当前帧内灯条编号
     */
    std::uint32_t GetId() const noexcept
    {
        return id_;
    }

    /**
     * @brief 返回灯条颜色
     */
    TeamColor GetColor() const noexcept
    {
        return color_;
    }

    /**
     * @brief 返回灯条中心，单位 pixel
     */
    const cv::Point2f& GetCenter() const noexcept
    {
        return shape_.center;
    }

    /**
     * @brief 返回长轴上端点，单位 pixel
     */
    const cv::Point2f& GetTop() const noexcept
    {
        return shape_.top;
    }

    /**
     * @brief 返回长轴下端点，单位 pixel
     */
    const cv::Point2f& GetBottom() const noexcept
    {
        return shape_.bottom;
    }

    /**
     * @brief 返回长轴长度，单位 pixel
     */
    float GetLength() const noexcept
    {
        return shape_.length;
    }

    /**
     * @brief 返回短轴宽度，单位 pixel
     */
    float GetWidth() const noexcept
    {
        return shape_.width;
    }

    /**
     * @brief 返回长轴相对图像竖直方向的倾角，单位 rad，范围 [-pi/2, pi/2]
     */
    float GetTilt() const noexcept
    {
        return shape_.tilt_rad;
    }

    /**
     * @brief 返回检测器给出的颜色分数
     */
    float GetColorScore() const noexcept
    {
        return score_;
    }

private:
    /**
     * @brief 由规范化后的端点构造灯条，几何量交给 LightBarGeometry 一次算定
     */
    LightBar(std::uint32_t id, TeamColor color, const cv::Point2f& top, const cv::Point2f& bottom, float width, float color_score) : id_(id), score_(color_score), color_(color), shape_(LightBarGeometry::FromEndpoints(top, bottom, width)) {}

    uint32_t  id_;              // 当前帧内唯一编号
    float     score_;           // 检测器定义的颜色分数
    TeamColor color_;           // 红、蓝或尚未判定
    LightBarGeometry shape_;    // 端点、中心与长短轴
};

} // namespace algorithm::vision
