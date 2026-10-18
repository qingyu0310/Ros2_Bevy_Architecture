/**
 * @file armor_refiner.cpp
 * @author qingyu
 * @brief 已知装甲板粗四点约束下的灯条角点精修器实现
 * @version 0.2
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/refine/armor_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

namespace algorithm::vision
{

namespace
{

constexpr float kGeometryEpsilon = 1.0e-4f; // 几何量比较用的最小正数

/**
 * @brief 四角点的有向面积，按 TL-TR-BR-BL 排列时为正
 */
float SignedArea(const std::array<cv::Point2f, 4>& corners)
{
    float twice_area = 0.0f;
    for (std::size_t index = 0; index < corners.size(); ++index)
    {
        const auto& first  = corners[index];
        const auto& second = corners[(index + 1) % corners.size()];
        twice_area += first.x * second.y - first.y * second.x;
    }
    return twice_area * 0.5f;
}

/**
 * @brief 四点是否全部有限、构成凸多边形且面积为正
 */
bool IsValidCorners(const std::array<cv::Point2f, 4>& corners)
{
    for (const auto& point : corners)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y))
        {
            return false;
        }
    }

    const std::vector<cv::Point2f> polygon(corners.begin(), corners.end());
    return cv::isContourConvex(polygon) && SignedArea(corners) > kGeometryEpsilon;
}

/**
 * @brief 由粗四点推出搜索区，返回整图坐标下的外接框
 *
 * 粗四点通常压不到灯条两端的端点，所以先把左右边各向外平移一个侧边长度、
 * 上下各向外扩 0.75 倍边长，再取这些点的最小旋转矩形外接框，最后裁进图像内。
 *
 * @param corners 粗四点，按 TL、TR、BR、BL 排列
 * @param image_size 整图宽高
 * @return 整图坐标搜索区；完全落在图像外时为空矩形
 */
cv::Rect MakeSearchRoi(const std::array<cv::Point2f, 4>& corners, const cv::Size& image_size)
{
    const auto& tl = corners[0];
    const auto& tr = corners[1];
    const auto& br = corners[2];
    const auto& bl = corners[3];

    // 左右边各向外平移一个侧边长：中心 ± 侧边向量
    const cv::Point2f left_side  = bl - tl;
    const cv::Point2f right_side = br - tr;
    const cv::Point2f tl1 = (tl + bl) * 0.5f - left_side;
    const cv::Point2f bl1 = (tl + bl) * 0.5f + left_side;
    const cv::Point2f tr1 = (tr + br) * 0.5f - right_side;
    const cv::Point2f br1 = (tr + br) * 0.5f + right_side;
    const cv::Point2f top_side    = tr1 - tl1;
    const cv::Point2f bottom_side = br1 - bl1;

    // 上下再各扩 0.75 倍边长，得到四个外扩角点
    const std::vector<cv::Point2f> expanded {
        (tl1 + tr) * 0.5f - top_side    * 0.75f,
        (tl1 + tr) * 0.5f + top_side    * 0.75f,
        (bl1 + br) * 0.5f + bottom_side * 0.75f,
        (bl1 + br) * 0.5f - bottom_side * 0.75f
    };

    const cv::Rect image_bounds(0, 0, image_size.width, image_size.height);
    return cv::minAreaRect(expanded).boundingRect() & image_bounds;
}

/**
 * @brief 记一次灯条拒绝：统计加一，同时留下亮区供离线可视化
 */
void Reject(LightDetectionResult& result, const cv::RotatedRect& rectangle, LightRejectReason reason)
{
    ++result.stats.rejected[static_cast<std::size_t>(reason)];
    result.rejections.push_back({rectangle, reason});
}

/**
 * @brief 按轮廓点上的平均红蓝判定灯条颜色
 *
 * 只取轮廓像素（调用方用 CHAIN_APPROX_NONE 保留全部边界点），不做内部积分：
 * 灯条整体是亮的，边界取样足够；返回的颜色由红蓝总和谁大决定，
 * 分数是 (红-蓝)/(红+蓝)，落在零附近说明接近白色或不饱和。
 *
 * @param image 原始 BGR 图像
 * @param contour 当前亮区轮廓，使用整图坐标
 * @return 颜色与归一化差值；总能量为零时分数取 0
 */
std::pair<TeamColor, float> GetColor(const cv::Mat& image, const std::vector<cv::Point>& contour)
{
    std::int64_t red_sum = 0;
    std::int64_t blue_sum = 0;
    for (const auto& point : contour)
    {
        const cv::Vec3b& pixel = image.at<cv::Vec3b>(point);
        red_sum  += pixel[2];
        blue_sum += pixel[0];
    }

    const auto  color    = blue_sum > red_sum ? TeamColor::Blue : TeamColor::Red;
    const auto  total = red_sum  + blue_sum;
    const float score = total == 0 ? 0.0f : static_cast<float>(red_sum - blue_sum) / static_cast<float>(total);

    return {color, score};
}

/**
 * @brief ROI 内的轻量灯条提取：二值化 → 外轮廓 → 几何门限 → 颜色统计
 *
 * 与 LightDetector 的区别是不做形态学闭运算、不做填充率与颜色对比度筛选：
 * ROI 已经是粗四点附近的一小块，目标是快，门限由本类 Config 直接给。
 * 帧内 ID 按当前已保留数量递增，最后按中心横坐标排序并回填统计。
 *
 * @param image ROI 图像（非空 CV_8UC3）
 * @param config 本类配置，用其中的亮度阈值与灯条几何门限
 * @return ROI 局部坐标的灯条与逐级统计
 */
LightDetectionResult DetectLights(const cv::Mat& image, const ArmorRefiner::Config& config)
{
    cv::Mat gray;
    cv::Mat binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, config.brightness_threshold, 255, cv::THRESH_BINARY);

    // CHAIN_APPROX_NONE：保留全部边界点，GetColor 要按轮廓点取色
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    LightDetectionResult result;
    result.stats.contour_count = contours.size();

    for (const auto& contour : contours)
    {
        const cv::RotatedRect rectangle = cv::minAreaRect(contour);
        const auto [color, color_score] = GetColor(image, contour);
        auto  light = LightBar::Create(rectangle, static_cast<std::uint32_t>(result.lights.size()), color, color_score);
        
        if (!light)
        {
            Reject(result, rectangle, LightRejectReason::InvalidRectangle);
            continue;
        }
        if (light->GetLength() <= config.min_light_length_px)
        {
            Reject(result, rectangle, LightRejectReason::Length);
            continue;
        }

        const float ratio = light->GetLength() / light->GetWidth();
        if (ratio <= config.min_light_ratio || ratio >= config.max_light_ratio)
        {
            Reject(result, rectangle, LightRejectReason::Ratio);
            continue;
        }
        if (std::abs(light->GetTilt()) >= config.max_light_tilt_rad)
        {
            Reject(result, rectangle, LightRejectReason::Tilt);
            continue;
        }

        if (color == TeamColor::Blue)
        {
            ++result.stats.blue_count;
        }
        else
        {
            ++result.stats.red_count;
        }
        result.lights.push_back(*light);
    }

    // 按中心横坐标排好，后面"左灯条/右灯条"的取法依赖于这个顺序的确定性
    std::sort(result.lights.begin(), result.lights.end(), [](const auto& first, const auto& second)
    {
        return first.GetCenter().x < second.GetCenter().x;
    });

    result.stats.light_count = result.lights.size();
    return result;
}

/**
 * @brief 把 ROI 局部坐标的灯条平移到整图坐标
 *
 * 用端点重新构造灯条以维持类不变量（端点顺序、长宽正）。
 */
LightBar TranslateLight(const LightBar& light, const cv::Point2f& offset)
{
    return *LightBar::CreateFromEndpoints(light.GetId(), light.GetTop() + offset, light.GetBottom() + offset, 
                                       light.GetWidth(),      light.GetColor(),          light.GetColorScore());
}

} // namespace

/**
 * @brief 构造精修器，构造期完成配置校验
 *
 * @param config 精修配置
 * @throws std::invalid_argument 配置不满足约束
 */
ArmorRefiner::ArmorRefiner(const Config& config) : config_(config)
{
    config_.Validate();
}

/**
 * @brief 在粗四点推出的搜索区内挑最近的两根灯条，替换四点
 *
 * 流程：粗四点推出搜索区 → ROI 内提灯条 → 分别求"离粗左边最近"和"离粗右边最近"
 * 的灯条（用两端点到粗四点两个角点的距离和比较）→ 总距离不超上限才生效。
 * 任一步不过都原样返回粗四点，因此调用方可以无条件采用 final_corners。
 *
 * @param bgr_image 非空 CV_8UC3 整图
 * @param coarse_corners 严格按 TL、TR、BR、BL 排列的粗四点
 * @param debug_output 可选调试输出；非空时写入 ROI 内的灯条与被拒亮区
 * @return 精修结果；未生效时 final_corners 与粗四点相同
 */
ArmorRefinementResult ArmorRefiner::Refine(const cv::Mat& bgr_image, const std::array<cv::Point2f, 4>& coarse_corners, ArmorRefinementDebug* debug_output) const
{
    ArmorRefinementResult result;
    result.original_corners = coarse_corners;
    result.final_corners    = coarse_corners;

    if (debug_output != nullptr)
    {
        debug_output->lights.clear();
        debug_output->rejections.clear();
    }
    if (bgr_image.empty() || bgr_image.type() != CV_8UC3 || !IsValidCorners(coarse_corners))
    {
        return result;
    }

    result.roi = MakeSearchRoi(coarse_corners, bgr_image.size());
    if (result.roi.empty())
    {
        result.status = ArmorRefinementStatus::EmptyRoi;
        return result;
    }

    const LightDetectionResult detection = DetectLights(bgr_image(result.roi), config_);
    result.light_stats = detection.stats;
    if (debug_output != nullptr)
    {
        debug_output->lights = detection.lights;
        debug_output->rejections = detection.rejections;
    }
    if (detection.lights.empty())
    {
        result.status = ArmorRefinementStatus::NoLight;
        return result;
    }
    if (detection.lights.size() == 1)
    {
        result.status = ArmorRefinementStatus::SingleLight;
        return result;
    }

    // 逐根平移回整图后比距离：左灯条看 TL/BL，右灯条看 TR/BR，取端点距离和最小的那根
    const cv::Point2f offset(static_cast<float>(result.roi.x), static_cast<float>(result.roi.y));

    float left_distance  = std::numeric_limits<float>::max();
    float right_distance = std::numeric_limits<float>::max();

    for (const auto& local_light : detection.lights)
    {
        const LightBar light = TranslateLight(local_light, offset);
        const float current_left = cv::norm(coarse_corners[0] - light.GetTop()) + cv::norm(coarse_corners[3] - light.GetBottom());

        if (current_left < left_distance)
        {
            left_distance = current_left;
            result.left_light = light;
        }

        const float current_right = cv::norm(coarse_corners[2] - light.GetBottom()) + cv::norm(coarse_corners[1] - light.GetTop());
        if (current_right < right_distance)
        {
            right_distance = current_right;
            result.right_light = light;
        }
    }

    result.left_distance_px  = left_distance;
    result.right_distance_px = right_distance;
    result.total_distance_px = left_distance + right_distance;

    // 距离门控：总距离超上限说明这两根灯条多半不属于这块装甲，宁可不动
    if (!result.left_light || !result.right_light || result.total_distance_px >= config_.max_endpoint_distance_px)
    {
        result.status = ArmorRefinementStatus::DistanceRejected;
        return result;
    }

    // 四点顺序恒为 TL、TR、BR、BL：上边取两根灯条的上端点，下边取下端点
    result.final_corners = {
        result.left_light ->GetTop(), 
        result.right_light->GetTop(),
        result.right_light->GetBottom(),
        result.left_light ->GetBottom()
    };
    result.status = ArmorRefinementStatus::Applied;

    return result;
}

} // namespace algorithm::vision
