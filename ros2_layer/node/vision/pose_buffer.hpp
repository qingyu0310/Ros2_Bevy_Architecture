/**
 * @file pose_buffer.hpp
 * @author qingyu
 * @brief 云台姿态缓冲 — 按图像时间戳插值出同一时刻的四元数
 * @version 0.1
 * @date 2026-10-18
 *
 * @copyright Copyright (c) 2026
 *
 * @note 照参考 `tools/quaternion_buffer.hpp` 与 `Gimbal::orientation_at`：姿态样本按接收
 *       时刻入缓冲，查询时找包住目标时刻的前后两条做 slerp；包不住就返回空，调用方跳过
 *       这一帧，而不是退到"最新一帧"。
 */

#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <optional>
#include <utility>

namespace algorithm::vision
{

/**
 * @brief 一条云台位姿样本
 */
struct PoseSample
{
    Eigen::Quaterniond rotation;                    // 云台在世界系中的姿态
    Eigen::Vector3d    translation;                 // 云台原点在世界系中的位置，单位 m
};

/**
 * @brief 云台位姿缓冲
 *
 * 样本按时间递增入队，只保留最近 `capacity` 条；查询不消费缓冲，可重复调用。
 */
class PoseBuffer final
{
public:
    /**
     * @brief 建立缓冲
     *
     * @param capacity 最多保留的样本数
     */
    explicit PoseBuffer(std::size_t capacity = 256) : capacity_(capacity) {}

    /**
     * @brief 追加一条位姿样本
     *
     * @param stamp 采样时刻，单位 s
     * @param sample 云台在世界系中的位姿
     */
    void Push(double stamp, const PoseSample& sample)
    {
        if (!samples_.empty() && stamp < samples_.back().first)
        {
            return;
        }

        PoseSample normalized = sample;
        normalized.rotation.normalize();
        samples_.emplace_back(stamp, normalized);
        while (samples_.size() > capacity_)
        {
            samples_.pop_front();
        }
    }

    /**
     * @brief 查询某一时刻的位姿
     *
     * @param stamp 目标时刻，单位 s
     * @param max_hold_s 目标时刻晚于最后一条样本时，允许用最后一条顶替的时间窗，单位 s
     * @return 插值后的位姿；样本还没包住该时刻、且超出保持窗口时为空
     *
     * @note 图像通常比同一时刻的位姿消息先到，目标时刻会略微超在最后一条之后。给一个很小
     *       的保持窗口（一个发布周期量级）比退回"最新"更可控：超窗仍返回空，由调用方跳过。
     */
    std::optional<PoseSample> At(double stamp, double max_hold_s = 0.0) const
    {
        if (samples_.empty() || stamp < samples_.front().first)
        {
            return std::nullopt;
        }

        // 找第一条不早于目标时刻的样本
        const auto upper = std::lower_bound(samples_.begin(), samples_.end(), stamp,
                                            [](const auto& sample, double value) { return sample.first < value; });
        if (upper == samples_.begin())
        {
            return samples_.front().second;
        }
        if (upper == samples_.end())
        {
            const double overshoot = stamp - samples_.back().first;
            if (overshoot <= max_hold_s)
            {
                return samples_.back().second;
            }
            return std::nullopt;
        }

        const auto& before = *std::prev(upper);
        const double span = upper->first - before.first;
        if (span <= 0.0)
        {
            return upper->second;
        }

        const double ratio = std::clamp((stamp - before.first) / span, 0.0, 1.0);
        PoseSample result;
        result.rotation = before.second.rotation.slerp(ratio, upper->second.rotation).normalized();
        result.translation = before.second.translation + ratio * (upper->second.translation - before.second.translation);
        return result;
    }

    /**
     * @brief 缓冲中的样本数
     */
    std::size_t Size() const
    {
        return samples_.size();
    }

    /**
     * @brief 最近一条样本的时刻，单位 s；空缓冲时为 0
     */
    double LatestStamp() const
    {
        return samples_.empty() ? 0.0 : samples_.back().first;
    }

    /**
     * @brief 最早一条样本的时刻，单位 s；空缓冲时为 0
     */
    double EarliestStamp() const
    {
        return samples_.empty() ? 0.0 : samples_.front().first;
    }

private:
    std::size_t capacity_;
    std::deque<std::pair<double, PoseSample>> samples_;
};

} // namespace algorithm::vision
