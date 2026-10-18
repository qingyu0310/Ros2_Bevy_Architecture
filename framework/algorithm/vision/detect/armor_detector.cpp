/**
 * @file armor_detector.cpp
 * @author qingyu
 * @brief 0526 TensorRT 装甲板检测器实现
 * @version 0.2
 * @date 2026-10-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/detect/armor_detector.hpp"

#include "framework/algorithm/vision/core/detections.hpp"
#include "framework/algorithm/vision/detect/model_decode.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

static_assert(NV_TENSORRT_MAJOR == 10, "Dust requires TensorRT 10.x");

namespace algorithm::vision
{

namespace
{

/**
 * @brief CUDA 返回值检查：非成功就带着操作名和驱动错误串抛出
 */
void CudaCheck(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

/**
 * @brief 求张量元素总数，顺带挡住动态维度和乘法溢出
 *
 * TensorRT 可能给出 -1 表示动态维度：这里只接受全部为正的静态张量。
 */
std::size_t TensorElements(const nvinfer1::Dims& shape)
{
    if (shape.nbDims <= 0)
    {
        throw std::runtime_error("TensorRT tensor has no dimensions");
    }

    std::size_t elements = 1;
    for (int index = 0; index < shape.nbDims; ++index)
    {
        // 用除法提前判溢出：等到乘法溢出再比较就已经晚了
        if (shape.d[index] <= 0 || elements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(shape.d[index]))
        {
            throw std::runtime_error("TensorRT requires positive static tensor dimensions");
        }
        elements *= static_cast<std::size_t>(shape.d[index]);
    }

    return elements;
}

/**
 * @brief 数值稳定的 sigmoid：按符号分两段，避免 exp 溢出
 */
float Sigmoid(float value)
{
    // 分两段是为了让指数里的量恒为非正：exp 的结果落在 (0, 1]，哪个方向都不溢
    if (value >= 0.0f)
    {
        return 1.0f / (1.0f + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

/**
 * @brief 取最大值的下标，相等时取靠前的那个
 */
int ArgMax(const float* values, int count)
{
    int best = 0;
    for (int index = 1; index < count; ++index)
    {
        // 用严格大于：并列时保留靠前的分支，同一帧的结果才稳定
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return best;
}

/**
 * @brief 两个时间点之间的毫秒数，用于分段计时
 */
double Milliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

/**
 * @brief TensorRT 日志回调：只留最后一条 ERROR 及以上级别的消息
 *
 * 解序列化失败时 TensorRT 只把原因写进日志，所以这里要能把最后一条捞出来
 * 拼进异常；回调可能在库内部线程触发，因此用互斥锁保护且不抛异常。
 */
class TensorRTLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* message) noexcept override
    {
        // 级别比 ERROR 低的不参与拼异常：解序列化失败的原因一定是 ERROR
        if (severity > Severity::kERROR || message == nullptr)
        {
            return;
        }

        // 本回调可能在 TensorRT 内部线程里触发，所以锁内只做一次赋值，且不允许抛出
        try
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_error_ = message;
        }
        catch (...)
        {
        }
    }

    std::string GetLastError() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_error_;
    }

private:
    mutable std::mutex mutex_;
    std::string last_error_;
};

/**
 * @brief 解码后、尚未过颜色与标签门的候选
 *
 * 保留模型的原始类别编号，后面才用 DetectedArmor::Create 做最终校验。
 */
struct RawCandidate
{
    DetectedArmorGeometry geometry;             // 四点、中心与外接框（整图坐标）
    cv::Rect nms_box;                           // NMS 用的整数外接框
    int raw_color_id;                           // 颜色分支的原始下标
    int raw_label_id;                           // 标签分支的原始下标
    float confidence;                           // sigmoid 后的 objectness
};

} // namespace

struct ArmorDetector::Impl
{
    /**
     * @brief 校验配置后初始化引擎，任何一步失败都先释放已申请资源再抛出
     */
    explicit Impl(const Config& config) : config_(config)
    {
        // 先校验配置再碰 CUDA：配置不合法时不该留下半初始化的设备状态
        ValidateConfig();
        try
        {
            Initialize();
        }
        catch (...)
        {
            // 初始化可能是跑到一半失败的，这里要归还已经拿到的 CUDA/TensorRT 资源
            Release();
            throw;
        }
    }

    /**
     * @brief 析构即释放全部资源
     */
    ~Impl()
    {
        Release();
    }

    /**
     * @brief 校验配置：engine 路径非空、设备号非负、三个门限都是概率且终值不小于初筛值
     */
    void ValidateConfig() const
    {
        // 三个门限都是概率，统一按有限性与 [0,1] 校验
        const auto valid_probability = [](float value)
        {
            return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
        };

        if (config_.engine_path.empty())
        {
            throw std::invalid_argument("armor engine path is empty");
        }
        if (config_.cuda_device < 0)
        {
            throw std::invalid_argument("CUDA device must be non-negative");
        }
        // 终石门限低于初筛没有意义：初筛已经把更低的置信度都挡掉了
        if (!valid_probability(config_.score_threshold) ||
            !valid_probability(config_.confidence_threshold) ||
            !valid_probability(config_.nms_threshold) ||
            config_.confidence_threshold < config_.score_threshold)
        {
            throw std::invalid_argument("invalid armor detector thresholds");
        }
    }

    /**
     * @brief 加载 engine、校验张量契约并分配复用缓冲区
     *
     * 契约：恰好一个输入一个输出、都是线性设备张量、输入 [1,3,640,640] 的 FP16 或 FP32、
     * 输出 [25200,22]（或带 batch 维）。任一条不满足都在这里抛，避免拿错形状跑到后处理
     * 才出脏数据。
     *
     * @throws std::runtime_error CUDA、TensorRT 失败或引擎契约不满足
     */
    void Initialize()
    {
        // 设备号先落在本机可见范围内：后面所有 CUDA 调用都作用在它上面
        int device_count = 0;
        CudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

        if (config_.cuda_device >= device_count)
        {
            throw std::runtime_error("CUDA device is outside the available device range");
        }

        CudaCheck(cudaSetDevice(config_.cuda_device), "cudaSetDevice");

        // 一次性读进整个序列化引擎：deserialize 只接受内存里的完整字节流
        std::ifstream engine_file(config_.engine_path, std::ios::binary | std::ios::ate);
        if (!engine_file || engine_file.tellg() <= 0)
        {
            throw std::runtime_error("cannot read TensorRT engine: " + config_.engine_path);
        }

        const std::size_t engine_size = static_cast<std::size_t>(engine_file.tellg());
        std::vector<char> serialized_engine(engine_size);
        engine_file.seekg(0, std::ios::beg);

        if (!engine_file.read(serialized_engine.data(), static_cast<std::streamsize>(engine_size)))
        {
            throw std::runtime_error("cannot read the entire TensorRT engine: " + config_.engine_path);
        }

        engine_.runtime.reset(nvinfer1::createInferRuntime(engine_.logger));
        if (!engine_.runtime)
        {
            throw std::runtime_error("TensorRT createInferRuntime failed");
        }

        engine_.cuda_engine.reset(engine_.runtime->deserializeCudaEngine(serialized_engine.data(), engine_size));
        if (!engine_.cuda_engine)
        {
            const std::string detail = engine_.logger.GetLastError();
            throw std::runtime_error("TensorRT engine deserialization failed" + (detail.empty() ? std::string() : ": " + detail));
        }
        if (engine_.cuda_engine->getNbIOTensors() != 2)
        {
            throw std::runtime_error("0526 engine requires exactly one input and one output");
        }

        // TensorRT 10 用张量名字而不是绑定下标寻址，所以先把名字按输入/输出分类存好
        for (int index = 0; index < engine_.cuda_engine->getNbIOTensors(); ++index)
        {
            const char* tensor_name = engine_.cuda_engine->getIOTensorName(index);
            if (tensor_name == nullptr || tensor_name[0] == '\0')
            {
                throw std::runtime_error("TensorRT engine contains an unnamed I/O tensor");
            }
            if (engine_.cuda_engine->getTensorLocation(tensor_name) != nvinfer1::TensorLocation::kDEVICE || engine_.cuda_engine->getTensorFormat(tensor_name) != nvinfer1::TensorFormat::kLINEAR)
            {
                throw std::runtime_error("0526 engine requires linear device tensors");
            }

            // 同一侧出现第二个张量说明引擎不是 0526 那一版，宁可在这里拒绝
            std::string& destination = engine_.cuda_engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT ? contract_.input_name : contract_.output_name;
            if (!destination.empty())
            {
                throw std::runtime_error("TensorRT engine has duplicate input or output modes");
            }
            destination = tensor_name;
        }

        if (contract_.input_name.empty() || contract_.output_name.empty())
        {
            throw std::runtime_error("TensorRT engine is missing an input or output tensor");
        }

        // 形状在初始化期一次对齐：跑到后处理才发现是错形状，读出来的就是一整块脏数据
        const nvinfer1::Dims input_shape  = engine_.cuda_engine->getTensorShape(contract_.input_name.c_str());
        const nvinfer1::Dims output_shape = engine_.cuda_engine->getTensorShape(contract_.output_name.c_str());

        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 || input_shape.d[1] != 3 || input_shape.d[2] != kModelInputHeight || input_shape.d[3] != kModelInputWidth)
        {
            throw std::runtime_error("0526 input shape must be [1,3,640,640]");
        }

        // 输出带不带 batch 维都接受：只认最后两维是 25200×22
        if ((output_shape.nbDims != 2 && output_shape.nbDims != 3) || (output_shape.nbDims == 3 && output_shape.d[0] != 1) || 
             output_shape.d[output_shape.nbDims - 2] != kModelOutputRows || output_shape.d[output_shape.nbDims - 1] != kModelOutputCols)
        {
            throw std::runtime_error("0526 output shape must be [1,25200,22] or [25200,22]");
        }

        // 输入 FP16 / FP32 都支持（预处理按类型决定要不要转半精度），输出固定按 FP32 读
        contract_.input_type = engine_.cuda_engine->getTensorDataType(contract_.input_name.c_str());
        if (contract_.input_type != nvinfer1::DataType::kHALF && contract_.input_type != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error("0526 input tensor must be FP16 or FP32");
        }

        if (engine_.cuda_engine->getTensorDataType(contract_.output_name.c_str()) != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error("0526 output tensor must be FP32");
        }

        contract_.input_type_name = contract_.input_type == nvinfer1::DataType::kHALF ? "FP16" : "FP32";

        // 按输入类型算字节数：FP16 每元素 2 字节，FP32 4 字节；输出固定 FP32
        const std::size_t input_elements      = TensorElements(input_shape);
        const std::size_t output_elements     = TensorElements(output_shape);
        const std::size_t input_element_bytes = contract_.input_type == nvinfer1::DataType::kHALF ? 2 : sizeof(float);

        if (input_elements > std::numeric_limits<std::size_t>::max() / input_element_bytes || output_elements > std::numeric_limits<std::size_t>::max() / sizeof(float))
        {
            throw std::runtime_error("0526 tensor byte size overflows size_t");
        }
        contract_.input_bytes  = input_elements  * input_element_bytes;
        contract_.output_bytes = output_elements * sizeof(float);

        // 显存两份 + 锁页内存两份都在构造期一次分配：推理时只做 memcpy，不再碰 cudaMalloc
        engine_.context.reset(engine_.cuda_engine->createExecutionContext());
        if (!engine_.context)
        {
            throw std::runtime_error("TensorRT createExecutionContext failed");
        }

        CudaCheck(cudaStreamCreateWithFlags(&transfer_.stream, cudaStreamNonBlocking), "cudaStreamCreate");
        CudaCheck(cudaMalloc(&transfer_.device_input,  contract_.input_bytes),  "cudaMalloc input");
        CudaCheck(cudaMalloc(&transfer_.device_output, contract_.output_bytes), "cudaMalloc output");
        CudaCheck(cudaMallocHost(&transfer_.host_input,   contract_.input_bytes),  "cudaMallocHost input");
        CudaCheck(cudaMallocHost(&transfer_.host_output,  contract_.output_bytes), "cudaMallocHost output");

        if (!engine_.context->setTensorAddress(contract_.input_name.c_str(), transfer_.device_input) || !engine_.context->setTensorAddress(contract_.output_name.c_str(), transfer_.device_output))
        {
            throw std::runtime_error("TensorRT setTensorAddress failed");
        }

        // 预处理画布固定 640×640：每帧清零后把缩放图贴到左上角，不居中、不 letterbox
        scratch_.canvas = cv::Mat(kModelInputHeight, kModelInputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    /**
     * @brief 释放顺序：先同步流，再销毁 context、显存、锁页内存、流，最后引擎与运行时
     *
     * 可重复调用，对"只初始化了一半"的状态也是安全的（指针为空就跳过）。
     */
    void Release() noexcept
    {
        // 先切回自己那块设备：cudaFree / cudaFreeHost 只对当前设备上的指针生效
        if (config_.cuda_device >= 0)
        {
            cudaSetDevice(config_.cuda_device);
        }
        // 销毁缓冲之前先等流跑完，否则可能释放掉还在被拷贝的显存
        if (transfer_.stream != nullptr)
        {
            cudaStreamSynchronize(transfer_.stream);
        }

        engine_.context.reset();
        if (transfer_.device_input != nullptr)
        {
            cudaFree(transfer_.device_input);
            transfer_.device_input = nullptr;
        }
        if (transfer_.device_output != nullptr)
        {
            cudaFree(transfer_.device_output);
            transfer_.device_output = nullptr;
        }
        if (transfer_.host_input != nullptr)
        {
            cudaFreeHost(transfer_.host_input);
            transfer_.host_input = nullptr;
        }
        if (transfer_.host_output != nullptr)
        {
            cudaFreeHost(transfer_.host_output);
            transfer_.host_output = nullptr;
        }
        if (transfer_.stream != nullptr)
        {
            cudaStreamDestroy(transfer_.stream);
            transfer_.stream = nullptr;
        }
        engine_.cuda_engine.reset();
        engine_.runtime.reset();
    }

    /**
     * @brief 等比缩放到 640×640 画布左上角，再转成 NCHW 放进主机输入缓冲
     *
     * 取较小的那个方向的缩放系数（等比、不拉伸），空出来的部分保持 0：与 0526
     * 训练时的预处理一致。返回的缩放系数供后处理把坐标映射回原图。
     *
     * @param bgr_image 非空 CV_8UC3 原图
     * @return 缩放系数（原图 → 输入画布）
     */
    double Preprocess(const cv::Mat& bgr_image)
    {
        // 取较小的那个方向做等比缩放：长边正好贴到 640，短边空出来的地方留给 0，不拉伸、不居中
        const double scale       = std::min(static_cast<double>(kModelInputWidth) / bgr_image.cols,  static_cast<double>(kModelInputHeight) / bgr_image.rows);
        const int resized_width  = std::clamp(static_cast<int>(bgr_image.cols * scale), 1, kModelInputWidth);
        const int resized_height = std::clamp(static_cast<int>(bgr_image.rows * scale), 1, kModelInputHeight);

        // 画布先清零：上一帧缩放图之外的残留像素会跟着这一帧一起进网络
        scratch_.canvas.setTo(cv::Scalar(0, 0, 0));

        cv::resize(bgr_image, scratch_.canvas(cv::Rect(0, 0, resized_width, resized_height)), cv::Size(resized_width, resized_height));

        // 直接出 NCHW FP32：按 1/255 缩放、不减均值、不裁剪，swapRB 打开所以进网络的通道是 RGB
        cv::dnn::blobFromImage(scratch_.canvas, scratch_.chw, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

        // FP16 引擎要先把 NCHW 转半精度再写进锁页缓冲；FP32 引擎直接整块拷贝
        if (contract_.input_type == nvinfer1::DataType::kHALF)
        {
            const int shape[] = {1, 3, kModelInputHeight, kModelInputWidth};
            cv::Mat half_input(4, shape, CV_16F, transfer_.host_input);
            scratch_.chw.convertTo(half_input, CV_16F);
        }
        else
        {
            std::memcpy(transfer_.host_input, scratch_.chw.data, contract_.input_bytes);
        }
        return scale;
    }

    /**
     * @brief 同步跑一遍推理：H2D → enqueueV3 → D2H → 流同步
     *
     * 全程挂在同一个 stream 上，最后同步等待，所以返回时输出已经在主机缓冲里。
     */
    void Infer()
    {
        CudaCheck(cudaSetDevice(config_.cuda_device), "cudaSetDevice");

        // 拷贝与内核都入队到同一条流上，顺序由流保证；这里不等待，最后一次性同步
        CudaCheck(cudaMemcpyAsync(transfer_.device_input, transfer_.host_input,   contract_.input_bytes, cudaMemcpyHostToDevice, transfer_.stream), "cudaMemcpyAsync input");

        if (!engine_.context->enqueueV3(transfer_.stream))
        {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }

        CudaCheck(cudaMemcpyAsync(transfer_.host_output, transfer_.device_output, contract_.output_bytes, cudaMemcpyDeviceToHost, transfer_.stream), "cudaMemcpyAsync output");

        // 同步返回之后输出才真正落在主机缓冲里，Postprocess 依赖这一点
        CudaCheck(cudaStreamSynchronize(transfer_.stream), "cudaStreamSynchronize");
    }

    /**
     * @brief 解码 25200 行输出，逐级门控后产出有效装甲
     *
     * 每行 22 个数：0-7 是四点（TL、TR、BR、BL 各 xy，与 armor.hpp 的角点顺序一致）、
     * 8 是 objectness、9-12 是颜色分支、13-21 是标签分支。流程：objectness 初筛 →
     * 反算整图坐标 → 几何合法性 → NMS → 最终置信度 → 颜色 → 标签与尺寸一致性 →
     * DetectedArmor::Create。每级丢弃数都记进 stats，便于现场判断是哪一级滤掉了目标。
     *
     * @param scale Preprocess 返回的缩放系数
     * @param image_size 原图宽高
     * @return 有效装甲与逐级统计
     */
    ArmorDetectionResult Postprocess(double scale, const cv::Size& image_size)
    {
        ArmorDetectionResult result;
        result.stats.raw_rows = kModelOutputRows;

        const float* output = static_cast<const float*>(transfer_.host_output);
        std::vector<RawCandidate> candidates;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        candidates.reserve(256);
        boxes.reserve(256);
        confidences.reserve(256);

        // 第一遍只做初筛与反算：绝大多数行会被 objectness 挡掉，不做后面那串几何与类别判断
        for (int row_index = 0; row_index < kModelOutputRows; ++row_index)
        {
            const float* row = output + static_cast<std::size_t>(row_index) * kModelOutputCols;
            const float  confidence = Sigmoid(row[8]);
            result.stats.max_objectness = std::max(result.stats.max_objectness, confidence);

            if (confidence < config_.score_threshold)
            {
                continue;
            }
            ++result.stats.score_passed;

            // 输出坐标在 640×640 画布上，乘 1/scale 映射回原图
            const float inverse_scale = static_cast<float>(1.0 / scale);
            const std::array<cv::Point2f, 4> corners
            {
                cv::Point2f(row[0] * inverse_scale, row[1] * inverse_scale),
                cv::Point2f(row[6] * inverse_scale, row[7] * inverse_scale),
                cv::Point2f(row[4] * inverse_scale, row[5] * inverse_scale),
                cv::Point2f(row[2] * inverse_scale, row[3] * inverse_scale)
            };

            // 四点顺手算出包围盒和中心，同时判掉非有限值；NMS 只要整数框
            float min_x = corners[0].x;
            float max_x = corners[0].x;
            float min_y = corners[0].y;
            float max_y = corners[0].y;
            cv::Point2f center(0.0f, 0.0f);

            bool finite = true;
            for (const cv::Point2f& point : corners)
            {
                finite = finite && std::isfinite(point.x) && std::isfinite(point.y);
                min_x = std::min(min_x, point.x);
                max_x = std::max(max_x, point.x);
                min_y = std::min(min_y, point.y);
                max_y = std::max(max_y, point.y);
                center += point;
            }
            center *= 0.25f;

            if (!finite || max_x <= min_x || max_y <= min_y)
            {
                ++result.stats.rejected_geometry;
                continue;
            }

            const cv::Rect2f bounding_box(min_x, min_y, max_x - min_x, max_y - min_y);
            const cv::Rect   nms_box(static_cast<int>(min_x), static_cast<int>(min_y), static_cast<int>(max_x - min_x), static_cast<int>(max_y - min_y));

            const RawCandidate candidate
            {
                DetectedArmorGeometry{corners, center, bounding_box},
                nms_box,
                ArgMax(row + 9, 4),
                ArgMax(row + 13, 9),
                confidence
            };

            candidates.push_back(candidate);
            boxes.push_back(nms_box);
            confidences.push_back(confidence);
        }

        // NMS 用整数外接框（与训练时一致），之后再按最终置信度和类别逐级过滤
        std::vector<int> kept_indices;
        cv::dnn::NMSBoxes(boxes, confidences, config_.score_threshold, config_.nms_threshold, kept_indices);
        result.stats.nms_kept = kept_indices.size();
        result.armors.reserve(kept_indices.size());

        // 第二遍只走 NMS 留下来的候选：终选置信度、颜色、标签与尺寸、几何校验依次过一遍
        for (int candidate_index : kept_indices)
        {
            if (candidate_index < 0 || static_cast<std::size_t>(candidate_index) >= candidates.size())
            {
                ++result.stats.rejected_geometry;
                continue;
            }

            const RawCandidate& candidate = candidates[static_cast<std::size_t>(candidate_index)];
            if (candidate.confidence <= config_.confidence_threshold)
            {
                ++result.stats.rejected_confidence;
                continue;
            }

            const std::optional<TeamColor> color = DecodeArmorModelColor(candidate.raw_color_id);
            if (!color)
            {
                ++result.stats.rejected_color;
                continue;
            }

            const std::optional<ArmorLabel> label = DecodeArmorModelLabel(candidate.raw_label_id);
            const ArmorSize size = DecodeArmorModelSize(candidate.raw_label_id);

            // 标签与尺寸必须自洽（例如小装甲的编号不能落在前哨站上）
            if (!label || !IsArmorLabelCompatibleWithSize(*label, size))
            {
                ++result.stats.rejected_label_type;
                continue;
            }

            const DetectedArmorIdentity identity
            {
                *color,
                *label,
                size,
                candidate.raw_color_id,
                candidate.raw_label_id
            };
            std::optional<DetectedArmor> armor = DetectedArmor::Create(candidate.geometry, identity, candidate.confidence, image_size);
            
            if (!armor)
            {
                ++result.stats.rejected_geometry;
                continue;
            }
            result.armors.push_back(std::move(*armor));
        }

        result.stats.accepted = result.armors.size();
        return result;
    }

    /**
     * @brief 预处理 → 推理 → 后处理，三段分别计时并写进统计
     */
    ArmorDetectionResult Detect(const cv::Mat& bgr_image)
    {
        // 三段串行，逐段计时：现场从统计里就能看出瓶颈是在缩放、推理还是解码
        const auto preprocess_begin = std::chrono::steady_clock::now();
        const double scale = Preprocess(bgr_image);
        const auto preprocess_end   = std::chrono::steady_clock::now();

        Infer();
        const auto inference_end = std::chrono::steady_clock::now();

        ArmorDetectionResult result = Postprocess(scale, bgr_image.size());

        const auto postprocess_end = std::chrono::steady_clock::now();
        result.stats.preprocess_ms  = Milliseconds(preprocess_begin, preprocess_end);
        result.stats.inference_ms   = Milliseconds(preprocess_end,   inference_end);
        result.stats.postprocess_ms = Milliseconds(inference_end,    postprocess_end);

        return result;
    }

    /**
     * @brief TensorRT 对象链：logger 必须比 runtime 活得久，所以排在最前
     */
    struct TensorRtObjects
    {
        TensorRTLogger logger;                                      // engine 加载期的日志收集器，用于拼失败原因
        std::unique_ptr<nvinfer1::IRuntime> runtime;                // TensorRT 运行时
        std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine;         // 由引擎文件反序列化得到的引擎
        std::unique_ptr<nvinfer1::IExecutionContext> context;       // 唯一推理上下文，因此不允许并发 Detect
    };

    /**
     * @brief 引擎 I/O 契约：张量名、输入类型与缓冲字节数
     */
    struct TensorContract
    {
        std::string input_name;                                     // 引擎里的输入张量名，TensorRT 10 按名字寻址
        std::string output_name;                                    // 引擎里的输出张量名
        std::string input_type_name;                                // 输入类型的可读名（FP16/FP32），只用于回显
        std::size_t input_bytes {0};                                // 输入缓冲字节数，按输入张量类型算（FP16 为 2 字节/元素）
        std::size_t output_bytes {0};                               // 输出缓冲字节数，固定 FP32
        nvinfer1::DataType input_type {nvinfer1::DataType::kHALF};  // 输入张量类型，决定预处理是否要转 FP16
    };

    /**
     * @brief 推理数据通路：一条流 + 显存/锁页主机各两份缓冲
     */
    struct TransferState
    {
        cudaStream_t stream {nullptr};                              // H2D、推理、D2H 共用的非阻塞流
        void* device_input {nullptr};                               // 显存输入缓冲
        void* device_output {nullptr};                              // 显存输出缓冲
        void* host_input {nullptr};                                 // 锁页主机输入缓冲（H2D 源）
        void* host_output {nullptr};                                // 锁页主机输出缓冲（D2H 目标）
    };

    /**
     * @brief 预处理临时缓冲：画布与 NCHW 张量，复用避免每帧分配
     */
    struct PreprocessScratch
    {
        cv::Mat canvas;                                             // 640x640 预处理画布
        cv::Mat chw;                                                // blobFromImage 产出的 NCHW FP32，FP16 输入时再转一次
    };

    Config config_;                                                 // 构造时校验过的配置，运行期只读
    TransferState  transfer_;                                       // 推理数据通路
    TensorContract contract_;                                       // 引擎 I/O 契约
    TensorRtObjects engine_;                                        // TensorRT 对象链
    PreprocessScratch scratch_;                                     // 预处理临时缓冲
};

/**
 * @brief 加载并校验 TensorRT engine
 *
 * 重活都在 Impl 里，本层只持有配置并转发。
 *
 * @param config 检测器配置
 * @throws std::invalid_argument 配置不合法
 * @throws std::runtime_error engine、CUDA 或 TensorRT 初始化失败
 */
ArmorDetector::ArmorDetector(const Config& config) : config_(config), impl_(std::make_unique<Impl>(config_)) {}

ArmorDetector::~ArmorDetector() = default;

/**
 * @brief 对一帧 BGR 图像同步执行 0526 模型推理和后处理
 *
 * @param bgr_image 非空 CV_8UC3 图像，只读
 * @return 有效红蓝装甲及逐级统计
 * @throws std::invalid_argument 输入图像类型不符
 * @throws std::runtime_error CUDA 或 TensorRT 运行失败
 */
ArmorDetectionResult ArmorDetector::Detect(const cv::Mat& bgr_image)
{
    // 类型在对外这层挡掉：Impl 内部假定拿到的就是非空 CV_8UC3，不再重复检查
    if (bgr_image.empty() || bgr_image.type() != CV_8UC3)
    {
        throw std::invalid_argument("ArmorDetector requires a non-empty CV_8UC3 BGR image");
    }
    return impl_->Detect(bgr_image);
}

/**
 * @brief 获取 TensorRT engine 的输入张量名称
 * @return 输入张量名称的常量引用；生命周期与检测器相同
 */
const std::string& ArmorDetector::GetInputName() const noexcept
{
    return impl_->contract_.input_name;
}

/**
 * @brief 获取 TensorRT engine 的输出张量名称
 * @return 输出张量名称的常量引用；生命周期与检测器相同
 */
const std::string& ArmorDetector::GetOutputName() const noexcept
{
    return impl_->contract_.output_name;
}

/**
 * @brief 获取 TensorRT engine 输入数据类型的可读名称
 * @return 输入类型名称（例如 FP16 或 FP32）的常量引用；生命周期与检测器相同
 */
const std::string& ArmorDetector::GetInputTypeName() const noexcept
{
    return impl_->contract_.input_type_name;
}

} // namespace algorithm::vision
