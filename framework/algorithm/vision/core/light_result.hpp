/**
 * @file light_result.hpp
 * @author qingyu
 * @brief 灯条提取的层间契约 — 拒绝原因、逐级统计与单帧结果
 * @version 0.1
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/light_bar.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

namespace algorithm::vision
{

/**
 * @brief 单个亮区未成为有效灯条的原因
 *
 * 枚举顺序与 LightDetectionStats::rejected 下标一致；Count 只表示数组长度。
 */
enum class LightRejectReason : std::uint8_t
{
    ContourArea,                                // 轮廓面积小于下限
    InvalidRectangle,                           // 最小旋转矩形不能构造有效灯条
    Length,                                     // 灯条长轴长度小于下限
    Ratio,                                      // 灯条长宽比不在允许范围
    Fill,                                       // 轮廓在旋转矩形内的填充率过低
    Tilt,                                       // 灯条偏离图像竖直方向过大
    Color,                                      // 颜色不确定或不是目标颜色
    Limit,                                      // 合格灯条超过本帧数量上限
    Count                                       // 拒绝原因数量，不作为实际原因使用
};

/**
 * @brief 单帧灯条检测统计
 *
 * 颜色计数发生在几何筛选之后、目标颜色筛选之前，用来区分没有亮条和颜色不匹配。
 */
struct LightDetectionStats
{
    std::size_t contour_count {0};              // 二值图外轮廓总数
    std::size_t red_count     {0};              // 几何合格的红色亮条数
    std::size_t blue_count    {0};              // 几何合格的蓝色亮条数
    std::size_t unknown_count {0};              // 几何合格但颜色落入死区的亮条数
    std::size_t light_count   {0};              // 最终保留的目标颜色灯条数

    std::array<std::size_t, static_cast<std::size_t>(LightRejectReason::Count)> rejected {}; // 各原因拒绝数

    /**
     * @brief 查询指定原因的拒绝数量
     * @param reason 拒绝原因；传入 Count 返回 0
     * @return 当前帧因该原因被拒绝的亮区数量
     */
    std::size_t GetRejected(LightRejectReason reason) const noexcept;
};

/**
 * @brief 被单灯条门限拒绝的亮区及其首个失败原因
 */
struct LightRejection
{
    cv::RotatedRect rectangle;                  // 亮区最小旋转矩形
    LightRejectReason reason;                   // 按固定筛选顺序命中的首个失败原因
};

/**
 * @brief 单帧灯条检测结果
 */
struct LightDetectionResult
{
    LightDetectionStats stats;                  // 本帧逐级筛选统计
    std::vector<LightBar> lights;               // 按中心横坐标从左到右排列的灯条
    std::vector<LightRejection> rejections;     // 被拒绝亮区，供离线可视化定位门限
};

} // namespace algorithm::vision
