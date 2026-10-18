/**
 * @file geometry_util.hpp
 * @author qingyu
 * @brief 视觉几何内部工具：有限性、叉积、距离与凸四边形判据（仅供本目录实现使用）
 * @version 0.2
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <opencv2/core.hpp>

namespace algorithm::vision::detail
{

/**
 * @brief 几何判定的零门限
 */
constexpr float kGeometryEpsilon = 1.0e-4f;

/**
 * @brief 检查浮点值是否可参与几何计算
 */
inline bool IsFinite(float value)
{
    return std::isfinite(value);
}

/**
 * @brief 检查二维点两个分量是否均为有限值
 */
inline bool IsFinite(const cv::Point2f& point)
{
    return IsFinite(point.x) && IsFinite(point.y);
}

/**
 * @brief 计算两个二维向量的叉积
 */
inline float Cross(const cv::Point2f& first, const cv::Point2f& second)
{
    return first.x * second.y - first.y * second.x;
}

/**
 * @brief 计算两点间欧氏距离，单位与输入坐标一致
 */
inline float Distance(const cv::Point2f& first, const cv::Point2f& second)
{
    const cv::Point2f delta = second - first;
    return std::sqrt(delta.dot(delta));
}

/**
 * @brief 计算四角点的有向面积，图像坐标下 TL-TR-BR-BL 为正
 */
inline float SignedArea(const std::array<cv::Point2f, 4>& corners)
{
    float twice_area = 0.0f;
    for (std::size_t index = 0; index < corners.size(); ++index)
    {
        twice_area += Cross(corners[index], corners[(index + 1) % corners.size()]);
    }
    return 0.5f * twice_area;
}

/**
 * @brief 检查角点是否按 TL-TR-BR-BL 构成严格凸四边形
 */
inline bool IsValidQuadrilateral(const std::array<cv::Point2f, 4>& corners)
{
    if (!std::all_of(corners.begin(), corners.end(), [](const cv::Point2f& point)
    {
        return IsFinite(point);
    }))
    {
        return false;
    }

    for (std::size_t index = 0; index < corners.size(); ++index)
    {
        const cv::Point2f first_edge  = corners[(index + 1) % corners.size()] - corners[index];
        const cv::Point2f second_edge = corners[(index + 2) % corners.size()] - corners[(index + 1) % corners.size()];
        if (Cross(first_edge, second_edge) <= kGeometryEpsilon) {
            return false;
        }
    }
    return SignedArea(corners) > kGeometryEpsilon;
}

} // namespace algorithm::vision::detail
