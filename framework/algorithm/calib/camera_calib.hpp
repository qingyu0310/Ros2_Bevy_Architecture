/**
 * @file camera_calib.hpp
 * @author qingyu
 * @brief 圆点阵相机标定 — 板规格、样本签名、采集门限与内参求解
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <cstdint>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <optional>
#include <vector>

namespace algorithm::calib
{

/**
 * @brief 圆点阵标定板规格
 *
 * 三个字段都必须与实际板子一致：图案不符时 findCirclesGrid 只返回失败、不报错，
 * 静默失败会让整场标定白做。
 */
struct Board
{
    std::int32_t cols {9};                          // 圆点列数（横向）
    std::int32_t rows {6};                          // 圆点行数（纵向）
    double       center_distance_mm {40.0};         // 相邻圆心间距，单位 mm
};

/**
 * @brief 一帧圆点阵的几何签名，自动采集用它判断这帧算不算新姿态
 *
 * 只由当帧圆心的像素坐标算出，不需要内参，所以求解之前就能用来去重。
 * 两个比值都取长边除以短边（≥1），与 findCirclesGrid 返回的角点顺序无关。
 */
struct SampleShape
{
    cv::Point2f center {};                          // 全部圆心的均值位置，单位 pixel
    float       span_px {0.0f};                     // 四角圆心四边形四条边的平均长度，单位 pixel；与物距成反比
    float       pitch_ratio {1.0f};                 // 上下对边长度比，取长除以短
    float       yaw_ratio {1.0f};                   // 左右对边长度比，取长除以短
};

/**
 * @brief 自动采集的去重门限：与任何一个已收样本满足其一，即算新姿态
 */
struct CaptureGate
{
    float min_shift_px {80.0f};                     // 板心位移下限，单位 pixel
    float min_span_change {0.15f};                  // 尺度相对变化下限，无量纲
    float min_ratio_change {0.10f};                 // 俯仰比或偏航比的变化下限，无量纲
};

/**
 * @brief 一次标定的产出：内参、畸变与质量指标
 */
struct Intrinsics
{
    double       camera_matrix[9] {};               // 行优先 3x3，无量纲
    double       distort_coeffs[5] {};              // k1、k2、p1、p2、k3；k3 固定为 0
    double       reprojection_error_px {0.0};       // 全部样本点的平均重投影误差，单位 pixel
    std::int32_t sample_count {0};                  // 参与求解的样本数
    std::int32_t image_width {0};                   // 标定时的图像宽度，单位 pixel
    std::int32_t image_height {0};                  // 标定时的图像高度，单位 pixel
};

/**
 * @brief 在整图里识别一帧圆点阵；识别失败不修改输出
 *
 * @param bgr_image 非空 BGR 图像
 * @param board 板规格，列数与行数都至少为 2
 * @param centers 输出全部圆心，行优先，单位 pixel
 * @return true 识别到完整点阵
 */
bool DetectBoard(const cv::Mat& bgr_image, const Board& board, std::vector<cv::Point2f>& centers);

/**
 * @brief 由当帧圆心算出样本签名
 *
 * @param board 板规格
 * @param centers 一帧的全部圆心，个数须等于 cols × rows
 * @return 签名；个数不符或边长退化时返回 std::nullopt
 */
std::optional<SampleShape> MeasureShape(const Board& board, const std::vector<cv::Point2f>& centers);

/**
 * @brief 判断新样本相对已收样本是否够不同
 *
 * 位移、尺度、俯仰比、偏航比四项里，只要有任意一项超过对应门限，就认为提供了新姿态。
 *
 * @param shape 新样本签名
 * @param accepted 已接受的样本签名
 * @param gate 去重门限
 * @return true 四项门限都与每个已收样本满足其一
 */
bool IsDistinctSample(const SampleShape& shape, const std::vector<SampleShape>& accepted, const CaptureGate& gate);

/**
 * @brief 求解内参、畸变与重投影误差
 *
 * 物点用 mm，所以解出的平移量也是 mm；内参与畸变是无量纲/像素量，不做单位换算。
 * 视场窄，固定 k3 为 0。求解失败以 std::nullopt 上报，不抛异常。
 *
 * @param board 板规格，用来生成板点
 * @param image_points 每个样本的全部圆心，行优先，单位 pixel
 * @param image_size 标定图像的宽高
 * @return 求解结果；样本数不足、个数不符或 OpenCV 求解失败时返回 std::nullopt
 */
std::optional<Intrinsics> SolveIntrinsics(const Board& board, const std::vector<std::vector<cv::Point2f>>& image_points, const cv::Size& image_size);

} // namespace algorithm::calib
