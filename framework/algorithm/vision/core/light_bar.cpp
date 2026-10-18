/**
 * @file light_bar.cpp
 * @author qingyu
 * @brief 规范化二维灯条的工厂实现
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/core/light_bar.hpp"

#include "framework/algorithm/vision/core/geometry_util.hpp"

#include <algorithm>
#include <array>

namespace algorithm::vision
{

namespace
{

using detail::kGeometryEpsilon;
using detail::IsFinite;
using detail::Distance;

} // namespace

/**
 * @brief 从 OpenCV 旋转矩形创建灯条
 *
 * 不使用 RotatedRect::angle，而是从四个顶点选取长轴，避免宽高交换导致角度跳变；
 * 取长轴两侧边中点作为端点，再交给端点构造。
 *
 * @param rect 轮廓的最小旋转矩形，坐标和尺寸单位为 pixel
 * @param id 当前帧内唯一灯条编号
 * @param color 灯条颜色；尚未判色时传 TeamColor::Unknown
 * @param color_score 颜色分数，必须为有限值，具体定义由检测器统一
 * @return 成功时返回满足类不变量的灯条；零尺寸或非有限输入返回 std::nullopt
 */
std::optional<LightBar> LightBar::Create(const cv::RotatedRect& rect, std::uint32_t id, TeamColor color, float color_score)
{
    if (!IsFinite(rect.center) || !IsFinite(rect.size.width)   || !IsFinite(rect.size.height) ||
        !IsFinite(color_score) || rect.size.width <= kGeometryEpsilon || rect.size.height <= kGeometryEpsilon)
    {
        return std::nullopt;
    }

    std::array<cv::Point2f, 4> points{};
    rect.points(points.data());

    if (!std::all_of(points.begin(), points.end(),  [](const cv::Point2f& point)
    {
        return IsFinite(point);
    }))
    {
        return std::nullopt;
    }

    const float edge_01 = Distance(points[0], points[1]);
    const float edge_12 = Distance(points[1], points[2]);
    if (edge_01 <= kGeometryEpsilon || edge_12 <= kGeometryEpsilon)
    {
        return std::nullopt;
    }

    cv::Point2f first_endpoint, second_endpoint;
    float width;

    if (edge_01 >= edge_12)
    {
        first_endpoint  = (points[0] + points[3]) * 0.5f;
        second_endpoint = (points[1] + points[2]) * 0.5f;
        width = edge_12;
    }
    else
    {
        first_endpoint  = (points[0] + points[1]) * 0.5f;
        second_endpoint = (points[2] + points[3]) * 0.5f;
        width = edge_01;
    }

    return CreateFromEndpoints(id, first_endpoint, second_endpoint, width, color, color_score);
}

/**
 * @brief 从长轴端点创建灯条
 *
 * 用于端点校正后的灯条，也供纯几何测试构造输入。函数会自行规范 top/bottom 顺序；
 * 宽度必须为正且不大于长轴，不满足时返回 std::nullopt，不做钳制。
 *
 * @param id 当前帧内唯一灯条编号
 * @param top_or_bottom 长轴任一端点，单位 pixel
 * @param bottom_or_top 长轴另一端点，单位 pixel
 * @param width 灯条短轴宽度，单位 pixel，必须大于 0 且不大于长轴
 * @param color 灯条颜色
 * @param color_score 颜色分数，必须为有限值
 * @return 成功时返回规范化灯条；无效输入返回 std::nullopt
 */
std::optional<LightBar> LightBar::CreateFromEndpoints(std::uint32_t id, const cv::Point2f& top_or_bottom, const cv::Point2f& bottom_or_top, float width, TeamColor color, float color_score)
{
    const float length = Distance(top_or_bottom, bottom_or_top);
    if (!IsFinite(top_or_bottom) || !IsFinite(bottom_or_top) || !IsFinite(width) || !IsFinite(color_score) ||  length <= kGeometryEpsilon || width <= kGeometryEpsilon || width > length + kGeometryEpsilon)
    {
        return std::nullopt;
    }

    const bool first_is_top   = top_or_bottom.y < bottom_or_top.y || (top_or_bottom.y == bottom_or_top.y && top_or_bottom.x <= bottom_or_top.x);
    const cv::Point2f& top    = first_is_top ? top_or_bottom : bottom_or_top;
    const cv::Point2f& bottom = first_is_top ? bottom_or_top : top_or_bottom;

    return LightBar(id, color, top, bottom, width, color_score);
}

} // namespace algorithm::vision
