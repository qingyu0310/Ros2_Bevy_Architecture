/**
 * @file armor_refiner.hpp
 * @author qingyu
 * @brief 已知装甲板粗四点约束下的灯条角点精修器
 * @version 0.2
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/light_bar.hpp"
#include "framework/algorithm/vision/core/light_result.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

namespace algorithm::vision
{

/**
 * @brief 一次角点精修的结果状态
 *
 * 非 Applied 的状态都表示"这次没改"，final_corners 与粗四点完全相同。
 */
enum class ArmorRefinementStatus : std::uint8_t
{
    Applied,          // 找到了左右灯条，final_corners 已替换
    InvalidInput,     // 图像类型不对或粗四点无效
    EmptyRoi,         // 扩展出来的搜索区域裁剪后为空
    NoLight,          // 搜索区内没有灯条
    SingleLight,      // 搜索区内只有一根灯条，无法构成装甲板
    DistanceRejected, // 灯条端点到粗四点太远，判定不是同一块装甲
    Count             // 状态数量，不作为运行结果
};

/**
 * @brief 一次角点精修结果
 */
struct ArmorRefinementResult
{
    ArmorRefinementStatus status {ArmorRefinementStatus::InvalidInput}; // 生效状态或首个失败原因
    std::array<cv::Point2f, 4> original_corners {};                     // 调用方给出的粗四点，始终原样保留
    std::array<cv::Point2f, 4> final_corners    {};                     // 生效后的四点；未生效时等于粗四点
    float left_distance_px  {0.0f};                                     // 左灯条端点到粗左边两点的距离和，pixel
    float right_distance_px {0.0f};                                     // 右灯条端点到粗右边两点的距离和，pixel
    float total_distance_px {0.0f};                                     // 左右距离之和，门控用的就是它
    cv::Rect roi;                                                       // 实际搜索的整图 ROI
    LightDetectionStats light_stats;                                    // ROI 内灯条提取统计
    std::optional<LightBar> left_light;                                 // 选中的左灯条，仅找到时存在
    std::optional<LightBar> right_light;                                // 选中的右灯条，仅找到时存在

    /**
     * @brief 本次精修是否生效
     */
    bool Applied() const noexcept
    {
        return status == ArmorRefinementStatus::Applied;
    }
};

/**
 * @brief 可选调试输出：ROI 内提取到的灯条与被拒绝的亮区
 */
struct ArmorRefinementDebug
{
    std::vector<LightBar> lights;                                       // 通过全部筛选的灯条，ROI 局部坐标
    std::vector<LightRejection> rejections;                             // 被门限拒掉的亮区及首个失败原因
};

/**
 * @brief 用 ROI 内最近的两根灯条替换已知装甲板的粗四点
 *
 * 适合作为检测器的后级：检测器给出颜色和大致四点，这里只负责把角点收到灯条端点
 * 上。本类不做全图发现、不做几何配对打分，也不修改输入。
 */
class ArmorRefiner final
{
public:
    /**
     * @brief 搜索区、灯条筛选与距离门控参数
     */
    struct Config
    {
        int   brightness_threshold       {150};                         // 灰度二值化阈值，范围 [0, 255]
        float min_light_length_px        {8.0f};                        // 灯条长轴长度下限，pixel
        float min_light_ratio            {1.5f};                        // 长宽比下限
        float max_light_ratio            {20.0f};                       // 长宽比上限
        float max_light_tilt_rad         {0.785398f};                   // 相对图像竖直方向最大倾角，rad
        float max_endpoint_distance_px   {15.0f};                       // 左右距离之和的上限，pixel

        /**
        * @brief 校验配置：亮度阈值在 [0,255]，几何门限有限且关系自洽，距离上限为正
        *
        * @throws std::invalid_argument 任一约束不满足
        */
        void Validate() const
        {
            const bool valid = brightness_threshold >= 0   && brightness_threshold <= 255             &&
                std::isfinite(min_light_length_px)      && min_light_length_px   > 0.0f            &&
                std::isfinite(min_light_ratio)          && min_light_ratio       > 0.0f            &&
                std::isfinite(max_light_ratio)          && max_light_ratio       > min_light_ratio &&
                std::isfinite(max_light_tilt_rad)       && max_light_tilt_rad    > 0.0f            && max_light_tilt_rad <= static_cast<float>(CV_PI) * 0.5f &&
                std::isfinite(max_endpoint_distance_px) && max_endpoint_distance_px > 0.0f;

            if (!valid)
            {
                throw std::invalid_argument("invalid armor refiner config");
            }
        }
    };

    explicit ArmorRefiner(const Config& config);

    ArmorRefinementResult Refine(const cv::Mat& bgr_image, const std::array<cv::Point2f, 4>& coarse_corners, ArmorRefinementDebug* debug_output = nullptr) const;

    /**
     * @brief 返回构造时校验过的配置
     */
    const Config& GetConfig() const noexcept
    {
        return config_;
    }

private:
    Config config_; // 构造后只读
};

} // namespace algorithm::vision
