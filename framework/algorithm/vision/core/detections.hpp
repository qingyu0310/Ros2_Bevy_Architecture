/**
 * @file detections.hpp
 * @author qingyu
 * @brief 检测结果契约 — 模型输出的二维几何、身份映射与装甲板结果
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/armor_types.hpp"

#include <array>
#include <opencv2/core.hpp>
#include <optional>

namespace algorithm::vision
{

/**
 * @brief 0526 模型给出的二维几何
 *
 * 角点顺序固定为左上、右上、右下、左下，全部坐标使用原始输入图像像素。
 */
struct DetectedArmorGeometry
{
    std::array<cv::Point2f, 4> corners;     // LT、RT、RB、LB
    cv::Point2f center;                     // 四角点平均位置，pixel
    cv::Rect2f  bounding_box;               // 四角点的轴对齐外接矩形，pixel
};

/**
 * @brief 0526 模型原始类别与工程枚举的显式映射结果
 */
struct DetectedArmorIdentity
{
    TeamColor  color;                       // 仅接受模型中的蓝色和红色
    ArmorLabel label;                       // 经过显式映射的工程标签
    ArmorSize  size;                        // 由模型标签分支确定的大、小装甲
    int raw_color_id;                       // 模型颜色 argmax，便于核对映射
    int raw_label_id;                       // 模型标签 argmax，便于核对映射
};

/**
 * @brief 0526 模型直接输出的装甲板结果
 *
 * 与依赖两根 LightBar 的 ArmorCandidate 分离，避免为网络四点伪造灯条。
 */
class DetectedArmor final
{
public:
    /**
     * @brief 校验并创建模型装甲板结果
     * @param geometry 原图坐标下的四点、中心和外接矩形
     * @param identity 已完成显式映射的颜色、标签和尺寸
     * @param confidence 模型 objectness 经 sigmoid 后的置信度
     * @param image_size 原始输入图像尺寸
     * @return 几何、类别和置信度都有效时返回结果，否则返回 std::nullopt
     */
    static std::optional<DetectedArmor> Create(const DetectedArmorGeometry& geometry, const DetectedArmorIdentity& identity, float confidence, const cv::Size& image_size);

    /**
     * @brief 使用同一类别和置信度重新校验一组精修后的角点
     * @param corners 原图坐标下的 TL、TR、BR、BL 四点
     * @param image_size 原始输入图像尺寸
     * @return 角点仍满足检测结果契约时返回新对象
     */
    std::optional<DetectedArmor> WithCorners(const std::array<cv::Point2f, 4>& corners, const cv::Size& image_size) const;

    /**
     * @brief 图像四角，顺序 LT、RT、RB、LB，单位 pixel
     */
    const std::array<cv::Point2f, 4>& GetCorners() const noexcept
    {
        return geometry_.corners;
    }

    /**
     * @brief 四角点平均值，单位 pixel
     */
    const cv::Point2f& GetCenter() const noexcept
    {
        return geometry_.center;
    }

    /**
     * @brief 四角点的轴对齐外接矩形，单位 pixel
     */
    const cv::Rect2f& GetBoundingBox() const noexcept
    {
        return geometry_.bounding_box;
    }

    /**
     * @brief 灯条颜色
     */
    TeamColor GetColor() const noexcept
    {
        return identity_.color;
    }

    /**
     * @brief 工程内稳定的装甲编号
     */
    ArmorLabel GetLabel() const noexcept
    {
        return identity_.label;
    }

    /**
     * @brief 装甲板大小
     */
    ArmorSize GetSize() const noexcept
    {
        return identity_.size;
    }

    /**
     * @brief 模型 objectness 经 sigmoid 后的置信度
     */
    float GetConfidence() const noexcept
    {
        return confidence_;
    }

    /**
     * @brief 模型颜色 argmax，便于核对映射
     */
    int GetRawColorId() const noexcept
    {
        return identity_.raw_color_id;
    }

    /**
     * @brief 模型标签 argmax，便于核对映射
     */
    int GetRawLabelId() const noexcept
    {
        return identity_.raw_label_id;
    }

private:
    DetectedArmor(const DetectedArmorGeometry& geometry, const DetectedArmorIdentity& identity, float confidence) : geometry_(geometry), identity_(identity), confidence_(confidence) {}

    DetectedArmorGeometry geometry_;
    DetectedArmorIdentity identity_;
    float confidence_;
};

} // namespace algorithm::vision
