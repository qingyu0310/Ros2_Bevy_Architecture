/**
 * @file model_decode.hpp
 * @author qingyu
 * @brief 0526 模型类别解码 — 颜色、标签与尺寸的显式映射
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/armor_types.hpp"

#include <optional>

namespace algorithm::vision
{

/**
 * @brief 显式映射 0526 模型颜色分支
 * @return 0=Blue、1=Red；灰色和紫色等不可发布颜色返回 std::nullopt
 */
std::optional<TeamColor> DecodeArmorModelColor(int raw_color_id) noexcept;

/**
 * @brief 显式映射 0526 模型九路标签分支
 * @return 有效的 0..8 返回工程标签，越界返回 std::nullopt
 */
std::optional<ArmorLabel> DecodeArmorModelLabel(int raw_label_id) noexcept;

/**
 * @brief 按 0526 模型标签分支映射装甲板尺寸
 * @return 标签 1、8 为 Large，其他有效标签为 Small，越界为 Unknown
 */
ArmorSize DecodeArmorModelSize(int raw_label_id) noexcept;

/**
 * @brief 复现参考实现的数字与大小装甲一致性检查
 */
bool IsArmorLabelCompatibleWithSize(ArmorLabel label, ArmorSize size) noexcept;

} // namespace algorithm::vision
