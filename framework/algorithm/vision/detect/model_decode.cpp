/**
 * @file model_decode.cpp
 * @author qingyu
 * @brief 0526 模型类别解码实现
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/detect/model_decode.hpp"

namespace algorithm::vision
{

/**
 * @brief 颜色分支的原始下标转成队伍颜色
 *
 * 0526 模型约定 0 = 蓝、1 = 红；其他值说明输出异常，返回 std::nullopt。
 */
std::optional<TeamColor> DecodeArmorModelColor(int raw_color_id) noexcept
{
    switch (raw_color_id)
    {
        case 0:
            return TeamColor::Blue;
        case 1:
            return TeamColor::Red;
        default:
            return std::nullopt;
    }
}

/**
 * @brief 标签分支的原始下标转成数字标签
 *
 * 0526 模型约定：0 = 哨兵、1~5 = 一~五号、6 = 前哨站、7 和 8 = 基地
 * （两个下标都是基地，靠尺寸规则区分）；其他值返回 std::nullopt。
 */
std::optional<ArmorLabel> DecodeArmorModelLabel(int raw_label_id) noexcept
{
    switch (raw_label_id)
    {
        case 0:
            return ArmorLabel::Sentry;
        case 1:
            return ArmorLabel::One;
        case 2:
            return ArmorLabel::Two;
        case 3:
            return ArmorLabel::Three;
        case 4:
            return ArmorLabel::Four;
        case 5:
            return ArmorLabel::Five;
        case 6:
            return ArmorLabel::Outpost;
        case 7:
        case 8:
            return ArmorLabel::Base;
        default:
            return std::nullopt;
    }
}

/**
 * @brief 由标签下标推装甲尺寸
 *
 * 0526 模型约定：一号（1）和基地（8）是大装甲，其余为小装甲；下标越界算未知。
 */
ArmorSize DecodeArmorModelSize(int raw_label_id) noexcept
{
    if (raw_label_id < 0 || raw_label_id > 8)
    {
        return ArmorSize::Unknown;
    }
    return raw_label_id == 1 || raw_label_id == 8 ? ArmorSize::Large : ArmorSize::Small;
}

/**
 * @brief 标签与尺寸是否自洽
 *
 * 小装甲不可能是 1 号或基地，大装甲不可能是 2 号、哨兵或前哨站；其余组合放行。
 * 尺寸未知时直接判否。
 */
bool IsArmorLabelCompatibleWithSize(ArmorLabel label, ArmorSize size) noexcept
{
    if (size == ArmorSize::Small)
    {
        return label != ArmorLabel::One && label != ArmorLabel::Base;
    }
    if (size == ArmorSize::Large)
    {
        return label != ArmorLabel::Two && label != ArmorLabel::Sentry && label != ArmorLabel::Outpost;
    }
    return false;
}

} // namespace algorithm::vision
