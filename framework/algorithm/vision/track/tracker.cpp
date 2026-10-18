/**
 * @file tracker.cpp
 * @author qingyu
 * @brief 单目标跟踪实现 — 选板、像素四门关联与状态机
 * @version 0.2
 * @date 2026-10-17
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/track/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>

namespace algorithm::vision
{

/**
 * @brief 用一帧检测推进目标
 *
 * lost 时选一块位姿最适合 PnP 的板建目标；其它状态与预测面配对更新。更新后跑状态机；
 * 发散或未初始化则回到 lost。
 */
void Tracker::ProcessFrame(const std::vector<DetectedArmor>& armors, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, double dt_s)
{
    bool found = false;

    if (report_.state == TrackerState::Lost)
    {
        const DetectedArmor* selected = FindLockArmor(armors, camera);
        if (selected != nullptr)
        {
            found = LockVehicle(*selected, dimensions, camera, frame);
            if (!found)
            {
                report_.last_reset_reason = ResetReason::InitFailed;
            }
        }
    }
    else
    {
        found = UpdateVehicle(armors, dimensions, camera, frame, dt_s);
    }

    if (!found && report_.state != TrackerState::Lost)
    {
        ++report_.losses;
    }

    AdvanceState(found);

    // 最近 100 次更新里 NIS 失败 80 次以上：模型与观测长期不符，判丢
    // 至少积累若干次实际修正后再看失败率，避免第一帧异常就以 1/1 触发重建。
    if (report_.state != TrackerState::Lost && vehicle_.UpdateCount() >= 10 && vehicle_.NisFailureRate() >= config_.nis_failure_ratio)
    {
        ResetVehicle(ResetReason::Nis);
        return;
    }

    // 半径越界或状态非有限：整条丢掉，回到 lost 等下一帧重建
    if (report_.state != TrackerState::Lost && (vehicle_.Diverged() || !vehicle_.Active()))
    {
        ResetVehicle(ResetReason::Diverged);
        return;
    }

    if (report_.state == TrackerState::Lost)
    {
        report_.active = false;
        report_.converged = false;
    }
    else
    {
        report_.active = vehicle_.Active();
        report_.converged = (report_.state == TrackerState::Tracking) && vehicle_.Converged();
    }
}

/**
 * @brief 锁定一辆车：重置滤波器，用这块板初始化，并记下编号与尺寸供后续筛板
 *
 * 照参考 `set_target`：直接取本帧选出的那一块板，用它的四点初始化滤波器；编号与尺寸
 * 一并记下来，后续更新只接受同编号、同尺寸的板。
 */
bool Tracker::LockVehicle(const DetectedArmor& armor, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame)
{
    // 参考 `set_target` 是整条 VehicleEKF 重建；不清掉旧状态的话 Update 会走"预测+修正"，
    // 半径初值根本不生效，lost 后重建仍是旧状态
    vehicle_.Reset();

    if (!vehicle_.Update(armor.GetCorners(), armor.GetSize(), dimensions, camera, frame, 0.0, 0, false))
    {
        return false;
    }

    label_ = armor.GetLabel();
    size_  = armor.GetSize();

    report_.active = true;
    report_.updates = vehicle_.UpdateCount();
    
    return true;
}

/**
 * @brief 选一块检测板用于锁定
 *
 * 照参考：取离图像中心最近的一块，不设距离门与视角门。
 */
const DetectedArmor* Tracker::FindLockArmor(const std::vector<DetectedArmor>& armors, const CameraModel& camera) const
{
    const DetectedArmor* selected = nullptr;
    double selected_center_distance = std::numeric_limits<double>::max();

    for (const DetectedArmor& armor : armors)
    {
        // 离画面中心最近的优先；同距时保留先遍历到的那块
        const double center_distance = cv::norm(armor.GetCenter() - cv::Point2f(static_cast<float>(camera.cx), static_cast<float>(camera.cy)));
        if (center_distance < selected_center_distance)
        {
            selected = &armor;
            selected_center_distance = center_distance;
        }
    }

    return selected;
}

/**
 * @brief 用关联结果逐对更新目标
 *
 * 先按 dt 预测一次，再把候选过滤、四门代价和配对交给 `Associate`；匹配到至少两个面
 * 才估计 r/l/h。所有配对共用这一次预测，逐对修正时传入 dt=0。
 */
bool Tracker::UpdateVehicle(const std::vector<DetectedArmor>& armors, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, double dt_s)
{
    // 每个活动目标每帧只推进一次；即使本帧没有同编号/同尺寸候选，也必须推进时间。
    if (!std::isfinite(dt_s) || dt_s < 0.0 || dt_s > 0.3 || !vehicle_.Active())
    {
        return false;
    }
    vehicle_.Predict(dt_s);

    const AssociationResult association = Associate(armors, label_, size_, vehicle_, dimensions, camera, frame, config_.association);
    report_.diagnostics = association.diagnostics;
    report_.match_count = association.matches.size();
    report_.candidate_count = association.candidate_count;

    // 有候选但一对都没配上：这一帧全被四门拒了，是"门限问题"还是"状态漂了"的分界
    if (association.matches.empty())
    {
        if (association.had_candidates)
        {
            ++report_.gate_rejects;
        }
        return false;
    }

    // 单板无法约束 r/l/h，只有本帧匹配到至少两个面时才把它们放进观测
    const bool estimate_shape = association.matches.size() >= 2;

    // 瞄准面按三项加权分选，与修正顺序无关；matches 的升序排列只影响 EKF 修正次序
    const std::int32_t aim_index = SelectAimMatch(association.matches, dimensions, camera, nullptr);

    std::int32_t matched_count = 0;
    for (std::size_t index = 0; index < association.matches.size(); ++index)
    {
        const ArmorMatch& match = association.matches[index];
        // 预测已在上面做过一次，后续配对不再重复推进 dt
        if (!vehicle_.Update(match.armor->GetCorners(), match.armor->GetSize(), dimensions, camera, frame, 0.0, match.face, estimate_shape))
        {
            continue;
        }

        ++matched_count;
        if (static_cast<std::int32_t>(index) == aim_index)
        {
            report_.last_face = match.face;
            report_.last_cost = match.cost;
        }
    }

    report_.updates = vehicle_.UpdateCount();
    return matched_count > 0;
}

/**
 * @brief 本帧选哪一对匹配作为瞄准面
 *
 * 三项各自量纲不同（像素距离、无量纲代价、弧度），这里按权重直接加权，靠调权重决定主导项：
 * `score = w_center·中心距离² + w_cost·代价² + w_angle·夹角²`，取最小。夹角与代价都取自这块板
 * 自己的 PnP 位姿，解不出来时该项记为 0，只用另外两项。
 */
std::int32_t Tracker::SelectAimMatch(const std::vector<ArmorMatch>& matches, const ArmorDimensions& dimensions, const CameraModel& camera, double* score) const
{
    std::int32_t selected = -1;
    double selected_score = std::numeric_limits<double>::max();

    for (std::size_t index = 0; index < matches.size(); ++index)
    {
        const ArmorMatch& match = matches[index];
        const cv::Point2f image_center(static_cast<float>(camera.cx), static_cast<float>(camera.cy));
        const double center_distance = cv::norm(match.armor->GetCenter() - image_center);

        // 夹角用的是这块板自己的 PnP 法向与视线；板越正对相机，单板解出的位姿越可信
        double view_angle_rad = 0.0;
        const std::optional<std::array<cv::Point3f, 4>> object_points = MakeObjectPoints(match.armor->GetSize(), dimensions);
        if (object_points)
        {
            const std::optional<ArmorPose> pose = SolvePose(match.armor->GetCorners(), *object_points, camera);
            if (pose)
            {
                const Eigen::Vector3d center(pose->tvec[0], pose->tvec[1], pose->tvec[2]);
                if (center.norm() > 0.0)
                {
                    cv::Matx33d rotation;
                    cv::Rodrigues(cv::Mat(pose->rvec), rotation);
                    const Eigen::Vector3d normal_camera(rotation(0, 0), rotation(1, 0), rotation(2, 0));
                    const double facing_cos = std::abs(normal_camera.dot(center.normalized()));
                    view_angle_rad = std::acos(std::clamp(facing_cos, 0.0, 1.0));
                }
            }
        }

        const double candidate_score = config_.aim_weight_center * center_distance * center_distance + config_.aim_weight_cost * match.cost * match.cost +
                                       config_.aim_weight_angle  * view_angle_rad  * view_angle_rad;
        if (candidate_score < selected_score)
        {
            selected = static_cast<std::int32_t>(index);
            selected_score = candidate_score;
        }
    }

    if (score != nullptr)
    {
        *score = selected >= 0 ? selected_score : 0.0;
    }
    return selected;
}

/**
 * @brief 状态机转移，按参考 `state_machine()`
 */
void Tracker::AdvanceState(bool found)
{
    switch (report_.state)
    {
        case TrackerState::Lost:
            if (found)
            {
                report_.state = TrackerState::Detecting;
                report_.detect_count = 1;
                report_.temp_lost_count = 0;
            }
            break;

        case TrackerState::Detecting:
            if (found)
            {
                ++report_.detect_count;
                if (report_.detect_count >= config_.min_detect_count)
                {
                    report_.state = TrackerState::Tracking;
                }
            }
            else
            {
                // 确认期内丢一帧就作废，回 lost 等下一次完整确认
                report_.detect_count = 0;
                report_.state = TrackerState::Lost;
            }
            break;

        case TrackerState::Tracking:
            if (!found)
            {
                report_.state = TrackerState::TempLost;
                report_.temp_lost_count = 1;
            }
            break;

        case TrackerState::TempLost:
            if (found)
            {
                report_.state = TrackerState::Tracking;
                report_.temp_lost_count = 0;
            }
            else
            {
                ++report_.temp_lost_count;
                if (report_.temp_lost_count > config_.max_temp_lost_count)
                {
                    report_.state = TrackerState::Lost;
                    report_.temp_lost_count = 0;
                }
            }
            break;
    }
}

/**
 * @brief 丢弃当前目标并记录原因；所有判丢重建都走这里
 */
void Tracker::ResetVehicle(ResetReason reason)
{
    vehicle_.Reset();
    report_.active = false;
    report_.converged = false;
    report_.last_face = -1;
    report_.state = TrackerState::Lost;
    report_.detect_count = 0;
    report_.temp_lost_count = 0;
    report_.last_reset_reason = reason;
    ++report_.resets;
}

} // namespace algorithm::vision
