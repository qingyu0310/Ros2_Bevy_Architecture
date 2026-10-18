/**
 * @file armor_types.hpp
 * @author qingyu
 * @brief 装甲板通用枚举 — 队伍颜色、物理尺寸与工程标签
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstdint>

namespace algorithm::vision
{

/**
 * @brief 灯条颜色
 */
enum class TeamColor : std::uint8_t
{
    Red,                                    // 红方灯条
    Blue,                                   // 蓝方灯条
    Unknown                                 // 尚未判色或颜色无效
};

/**
 * @brief 装甲板物理尺寸
 */
enum class ArmorSize : std::uint8_t
{
    Small,                                  // 小装甲板
    Large,                                  // 大装甲板
    Unknown                                 // 尚未完成几何分类
};

/**
 * @brief 工程内稳定的装甲标签
 *
 * 0526 模型的九路标签顺序不同，必须经过 DecodeArmorModelLabel() 显式映射。
 */
enum class ArmorLabel : std::uint8_t
{
    One,
    Two,
    Three,
    Four,
    Five,
    Sentry,
    Outpost,
    Base,
    Any,                                    // 任意编号：只用于跟踪槽配置，不表示检测结果
    NotArmor                                // 检测无效
};

} // namespace algorithm::vision
