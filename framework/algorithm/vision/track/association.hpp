/**
 * @file association.hpp
 * @author qingyu
 * @brief 检测板与预测面的一对一关联 — 像素四门与匹配
 * @version 0.1
 * @date 2026-10-18
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/detections.hpp"
#include "framework/algorithm/vision/track/vehicle_ekf.hpp"

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace algorithm::vision
{

/**
 * @brief 四门关联参数
 */
struct AssociationConfig
{
    double center_gate_px {100.0};                  // 关联门：预测中心与检测中心，单位 pixel
    double corner_gate_px {140.0};                  // 关联门：四角平均误差，单位 pixel
    double angle_gate_rad {0.8};                    // 关联门：四条边方向角差之和，单位 rad
    double perimeter_ratio_gate {0.6};              // 关联门：预测与检测周长相对差，无量纲
};

/**
 * @brief 一块检测板与一个预测面的配对结果
 */
struct ArmorMatch
{
    const DetectedArmor* armor {nullptr};           // 参与配对的检测板
    std::int32_t face {-1};                         // 预测面序
    double cost {0.0};                              // 四门归一化代价
};

/**
 * @brief 一次配对尝试的四门分项误差，供日志定位是哪一项在拒
 */
struct AssociationDiagnostic
{
    std::size_t  candidate {0};                     // 参与本次尝试的候选板序号
    std::int32_t face {-1};                         // 本次尝试的面序
    std::array<cv::Point2f, 4> corners {};          // 该候选的检测四角，单位 pixel
    double normalized_cost {0.0};                   // 四项归一化代价之和，上界 4.0
    double center_error_px {0.0};                   // 中心误差，单位 pixel
    double corner_error_px {0.0};                   // 四角平均误差，单位 pixel
    double angle_error_rad {0.0};                   // 四条边方向角差之和，单位 rad
    double perimeter_ratio {0.0};                   // 周长相对差，无量纲
    bool   center_ok {false};                       // 中心门是否通过
    bool   corner_ok {false};                       // 四角门是否通过
    bool   angle_ok {false};                        // 方向门是否通过
    bool   perimeter_ok {false};                    // 周长门是否通过
};

/**
 * @brief 关联结果
 */
struct AssociationResult
{
    std::vector<ArmorMatch> matches;                // 按代价升序排列的配对
    bool had_candidates {false};                    // 本帧是否有同编号、同尺寸的候选
    std::size_t candidate_count {0};                // 候选板数量
    std::size_t face_count {0};                     // 预测面数量
    std::vector<AssociationDiagnostic> diagnostics; // 每个候选、每个面的一次尝试，按代价升序
};

/**
 * @brief 按像素四门计算一个预测面与一块检测板的代价
 *
 * 四项误差照参考 `tracker.cpp` 的 `quad_cost`：中心、四角、边方向角、周长比。任一项
 * 超门即无效；有效时代价为四项分别除以各自门限后的和。
 *
 * @param predicted 预测面四点，单位 pixel
 * @param measured 检测四点，单位 pixel
 * @param config 四门门限
 * @param diagnostic 非空时写入四项误差与各门是否通过；超门时也写
 * @return 代价；无有效配对时为空
 */
std::optional<double> QuadCost(const std::array<cv::Point2f, 4>& predicted, const std::array<cv::Point2f, 4>& measured, const AssociationConfig& config, AssociationDiagnostic* diagnostic = nullptr);

/**
 * @brief 把检测板与预测面做一对一最优配对
 *
 * 候选只取与目标同编号、同尺寸的板；每个预测面按当前状态重投影成四点，与候选的四点
 * 算四门代价。之后枚举合法一对一匹配，优先最大化匹配数量、再最小化总代价。
 *
 * @param armors 当帧检测结果
 * @param label 目标的机器人编号
 * @param size 目标的装甲板大小
 * @param target 提供面投影的滤波器
 * @param dimensions 装甲板实物尺寸，单位 m
 * @param camera 内参与畸变
 * @param frame 跟踪系相对相机系的位姿
 * @param config 四门门限
 * @return 配对结果；配对按代价升序
 */
AssociationResult Associate(const std::vector<DetectedArmor>& armors, ArmorLabel label, ArmorSize size, const VehicleEKF& target, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, const AssociationConfig& config);

} // namespace algorithm::vision
