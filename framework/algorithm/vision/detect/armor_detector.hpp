/**
 * @file armor_detector.hpp
 * @author qingyu
 * @brief 基于 0526 TensorRT engine 的装甲板检测器
 * @version 0.2
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "framework/algorithm/vision/core/detections.hpp"

#include <cstddef>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace algorithm::vision
{

/**
 * @brief 单帧 0526 模型检测统计
 */
struct ArmorDetectionStats
{
    std::size_t raw_rows                   {0};         // 模型输出总行数
    std::size_t score_passed               {0};         // 通过初筛 objectness 的行数
    std::size_t nms_kept                   {0};         // NMS 后保留的候选数
    std::size_t accepted                   {0};         // 最终发布的红、蓝有效装甲数
    std::size_t rejected_confidence        {0};         // NMS 后未通过最终置信度的候选数
    std::size_t rejected_color             {0};         // 灰色、紫色或越界颜色候选数
    std::size_t rejected_label_type        {0};         // 标签越界或尺寸规则不一致候选数
    std::size_t rejected_geometry          {0};         // 四点、外接矩形或图像边界无效候选数
    float       max_objectness             {0.0f};      // 当前帧全部行的最大 sigmoid objectness
    double      preprocess_ms              {0.0};       // 补零、RGB/NCHW 和 FP16/FP32 转换耗时
    double      inference_ms               {0.0};       // H2D、enqueueV3、D2H 和同步总耗时
    double      postprocess_ms             {0.0};       // 解码、NMS、映射与几何校验耗时
};

/**
 * @brief 单帧模型直出结果
 */
struct ArmorDetectionResult
{
    std::vector<DetectedArmor> armors;                  // 按 NMS 结果顺序排列的有效装甲
    ArmorDetectionStats stats;                          // 本帧逐级计数和耗时
};

/**
 * @brief 0526 TensorRT 同步检测器
 *
 * 构造时一次加载 engine、校验张量并分配复用缓冲区。每个实例只拥有一个
 * execution context 和 CUDA stream，因此 Detect() 不允许并发调用。
 */
class ArmorDetector final
{
public:
    struct Config
    {
        std::string engine_path;                        // 本机 TensorRT engine 绝对路径
        int         cuda_device          {0};           // CUDA device 编号
        float       score_threshold      {0.7f};        // NMS 前 objectness 初筛
        float       confidence_threshold {0.8f};        // NMS 后最终置信度门限（严格大于）
        float       nms_threshold        {0.3f};        // NMS IoU 门限
    };

    /**
     * @brief 加载并校验 TensorRT engine，创建检测器资源
     *
     * @param config 检测器配置
     * @throws std::invalid_argument 配置不合法
     * @throws std::runtime_error engine、CUDA 或 TensorRT 初始化失败
     */
    explicit ArmorDetector(const Config& config);

    /**
     * @brief 释放检测器持有的 TensorRT 与 CUDA 资源
     */
    ~ArmorDetector();

    ArmorDetector(const ArmorDetector&) = delete;
    ArmorDetector& operator=(const ArmorDetector&) = delete;
    ArmorDetector(ArmorDetector&&) = delete;
    ArmorDetector& operator=(ArmorDetector&&) = delete;

    ArmorDetectionResult Detect(const cv::Mat& bgr_image);

    /**
     * @brief 返回构造时保存的配置
     */
    const Config& GetConfig() const noexcept
    {
        return config_;
    }


    const std::string& GetInputName() const noexcept;

    const std::string& GetOutputName() const noexcept;

    const std::string& GetInputTypeName() const noexcept;

    /**
     * @brief 模型固定输入宽度，单位 pixel
     */
    int GetInputWidth() const noexcept
    {
        return kModelInputWidth;
    }

    /**
     * @brief 模型固定输入高度，单位 pixel
     */
    int GetInputHeight() const noexcept
    {
        return kModelInputHeight;
    }

    /**
     * @brief 输出候选行数（锚框数）
     */
    int GetOutputRows() const noexcept
    {
        return kModelOutputRows;
    }

    /**
     * @brief 每行输出长度，字段划分见 cpp 的 Postprocess
     */
    int GetOutputColumns() const noexcept
    {
        return kModelOutputCols;
    }

private:
    struct Impl;

    // 0526 模型的固定形状：引擎校验、预处理画布和输出解码都按它来
    static constexpr int kModelInputWidth  = 640;   // 输入宽度，pixel
    static constexpr int kModelInputHeight = 640;   // 输入高度，pixel
    static constexpr int kModelOutputRows  = 25200; // 输出候选行数（锚框数）
    static constexpr int kModelOutputCols  = 22;    // 每行长度

    Config config_;
    std::unique_ptr<Impl> impl_;
};

} // namespace algorithm::vision
