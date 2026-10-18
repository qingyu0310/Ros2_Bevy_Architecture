/**
 * @file detections.cpp
 * @author qingyu
 * @brief 模型装甲板结果的校验与构造
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/core/detections.hpp"

#include "framework/algorithm/vision/core/geometry_util.hpp"

#include <algorithm>
#include <limits>

namespace algorithm::vision
{

namespace
{

using detail::kGeometryEpsilon;
using detail::IsFinite;
using detail::IsValidQuadrilateral;

} // namespace

/**
 * @brief 校验几何、身份与置信度后构造装甲
 *
 * 除四点的有限性与凸性外，还要求外接框非退化、四角点落在图像有效像素范围内、
 * 颜色/尺寸/标签都不是 Unknown、置信度在 [0,1]。任一不满足返回 std::nullopt，
 * 由调用方计入对应的拒绝统计——这里不抛异常，脏候选是常态。
 *
 * @param geometry 解码后的四点、中心与外接框，整图坐标
 * @param identity 颜色、标签、尺寸与模型原始类别号
 * @param confidence sigmoid 后的 objectness
 * @param image_size 原图宽高，用于边界检查
 * @return 有效装甲；任一校验不过返回 std::nullopt
 */
std::optional<DetectedArmor> DetectedArmor::Create(const DetectedArmorGeometry& geometry, const DetectedArmorIdentity& identity, float confidence, const cv::Size& image_size)
{
    const bool box_is_valid = IsFinite(geometry.bounding_box.x)      && IsFinite(geometry.bounding_box.y)       && IsFinite(geometry.bounding_box.width) &&
                              IsFinite(geometry.bounding_box.height) && geometry.bounding_box.width > kGeometryEpsilon && geometry.bounding_box.height > kGeometryEpsilon;

    const bool corners_inside = image_size.width > 0 && image_size.height > 0 && std::all_of(geometry.corners.begin(), geometry.corners.end(),  [&image_size](const cv::Point2f& point)
    {
        return IsFinite(point) && point.x >= 0.0f && point.y >= 0.0f && point.x <= static_cast<float>(image_size.width - 1) &&  point.y <= static_cast<float>(image_size.height - 1);
    });

    const bool identity_is_valid = identity.color != TeamColor::Unknown && identity.size != ArmorSize::Unknown && identity.label != ArmorLabel::NotArmor;

    if (!IsValidQuadrilateral(geometry.corners) || !IsFinite(geometry.center) || !box_is_valid || !corners_inside || !identity_is_valid || !IsFinite(confidence) || confidence < 0.0f || confidence > 1.0f)
    {
        return std::nullopt;
    }

    return DetectedArmor(geometry, identity, confidence);
}

/**
 * @brief 由同一装甲板的精修四点重新构造检测结果
 *
 * 身份和置信度保持模型结果不变，几何量由新四点重算并再次走统一校验。
 */
std::optional<DetectedArmor> DetectedArmor::WithCorners(const std::array<cv::Point2f, 4>& corners, const cv::Size& image_size) const
{
    cv::Point2f center;
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& point : corners)
    {
        center += point;
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }
    center *= 0.25f;

    const DetectedArmorGeometry geometry {
        corners,
        center,
        cv::Rect2f {min_x, min_y, max_x - min_x, max_y - min_y}};
    return Create(geometry, identity_, confidence_, image_size);
}

} // namespace algorithm::vision
