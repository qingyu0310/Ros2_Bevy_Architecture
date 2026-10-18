/**
 * @file association.cpp
 * @author qingyu
 * @brief 检测板与预测面的一对一关联实现 — 像素四门与匹配
 * @version 0.1
 * @date 2026-10-18
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/vision/track/association.hpp"

#include "framework/algorithm/math/angle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>

namespace algorithm::vision
{

namespace
{

/**
 * @brief 无有效配对时的代价
 */
constexpr double kInvalidCost = std::numeric_limits<double>::max();

} // namespace

/**
 * @brief 按像素四门计算一个预测面与一块检测板的代价
 */
std::optional<double> QuadCost(const std::array<cv::Point2f, 4>& predicted, const std::array<cv::Point2f, 4>& measured, const AssociationConfig& config, AssociationDiagnostic* diagnostic)
{
    // 每项误差最后都要除以自己的门限做归一化；门限为 0 或非有限值时没有可比基准，直接判无效
    if (!(config.center_gate_px > 0.0) || !(config.corner_gate_px > 0.0) || !(config.angle_gate_rad > 0.0) || !(config.perimeter_ratio_gate > 0.0) || !std::isfinite(config.center_gate_px) || !std::isfinite(config.corner_gate_px) || !std::isfinite(config.angle_gate_rad) || !std::isfinite(config.perimeter_ratio_gate))
    {
        return std::nullopt;
    }

    cv::Point2f predicted_center(0.0f, 0.0f);
    cv::Point2f measured_center (0.0f, 0.0f);
    double predicted_perimeter = 0.0;
    double measured_perimeter  = 0.0;
    double corner_error = 0.0;
    double angle_error  = 0.0;

    for (std::size_t index = 0; index < predicted.size(); ++index)
    {
        const cv::Point2f& point = predicted[index];
        const cv::Point2f& observed = measured[index];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(observed.x) || !std::isfinite(observed.y))
        {
            return std::nullopt;
        }

        // 中心误差用四点均值，四角误差用同序号角点距离；两边角点顺序都是 LT、RT、RB、LB，可以逐点相减
        predicted_center += point;
        measured_center  += observed;
        corner_error     += cv::norm(point - observed);

        const std::size_t next = (index + 1) % predicted.size();
        const cv::Point2f predicted_edge = predicted[next] - point;
        const cv::Point2f measured_edge  = measured[next] - observed;
        
        // 周长逐边累加；边方向角差先归一到 ±π 再取绝对值，避免绕过 ±π 时差出一整圈
        predicted_perimeter += cv::norm(predicted_edge);
        measured_perimeter  += cv::norm(measured_edge);
        angle_error += std::abs(algorithm::math::normalize_radian_pm_pi(std::atan2(predicted_edge.y, predicted_edge.x) - std::atan2(measured_edge.y, measured_edge.x)));
    }

    if (predicted_perimeter <= std::numeric_limits<double>::epsilon() || measured_perimeter <= std::numeric_limits<double>::epsilon())
    {
        return std::nullopt;
    }

    predicted_center *= 0.25f;
    measured_center  *= 0.25f;
    corner_error     *= 0.25;

    // 周长差相对预测周长归一化：同一块板远近不同也能用同一个门限
    const double center_error = cv::norm(predicted_center - measured_center);
    const double perimeter_ratio_error = std::abs(predicted_perimeter - measured_perimeter) / predicted_perimeter;

    // 四门：任一项超限这个配对被拒，不参与后面的配对枚举
    const bool center_ok = center_error <= config.center_gate_px;
    const bool corner_ok = corner_error <= config.corner_gate_px;
    const bool angle_ok = angle_error <= config.angle_gate_rad;
    const bool perimeter_ok = perimeter_ratio_error <= config.perimeter_ratio_gate;

    if (diagnostic != nullptr)
    {
        diagnostic->center_error_px = center_error;
        diagnostic->corner_error_px = corner_error;
        diagnostic->angle_error_rad = angle_error;
        diagnostic->perimeter_ratio = perimeter_ratio_error;
        diagnostic->center_ok = center_ok;
        diagnostic->corner_ok = corner_ok;
        diagnostic->angle_ok = angle_ok;
        diagnostic->perimeter_ok = perimeter_ok;
    }

    if (!center_ok || !corner_ok || !angle_ok || !perimeter_ok)
    {
        return std::nullopt;
    }

    const double cost = center_error / config.center_gate_px + corner_error / config.corner_gate_px + angle_error / config.angle_gate_rad + perimeter_ratio_error / config.perimeter_ratio_gate;
    return std::isfinite(cost) ? std::optional<double>(cost) : std::nullopt;
}

/**
 * @brief 把检测板与预测面做一对一最优配对
 */
AssociationResult Associate(const std::vector<DetectedArmor>& armors, ArmorLabel label, ArmorSize size, const VehicleEKF& target, const ArmorDimensions& dimensions, const CameraModel& camera, const FrameTransform& frame, const AssociationConfig& config)
{
    AssociationResult result;

    // 候选只取与目标同编号、同尺寸的板：编号不同就是另一辆车，尺寸不同则物点宽度不对
    std::vector<const DetectedArmor*> candidates;
    for (const DetectedArmor& armor : armors)
    {
        if (armor.GetLabel() == label && armor.GetSize() == size)
        {
            candidates.push_back(&armor);
        }
    }

    if (candidates.empty())
    {
        return result;
    }
    result.had_candidates = true;

    // 预测面来自同一套 EKF 状态：车心、转角、半径各推一次，按 face*2π/4 偏移出四个假想面
    const std::vector<FaceState> predicted = target.Faces();
    if (predicted.empty())
    {
        return result;
    }

    const std::size_t face_count = predicted.size();
    result.candidate_count = candidates.size();
    result.face_count = face_count;
    std::vector<std::vector<double>> costs(candidates.size(), std::vector<double>(face_count, kInvalidCost));

    // 每次尝试都留一条诊断，超门时也要能看出是哪一项超了
    for (std::size_t obs = 0; obs < candidates.size(); ++obs)
    {
        for (std::size_t face = 0; face < face_count; ++face)
        {
            const std::optional<std::array<cv::Point2f, 4>> projected = target.ProjectFace(size, dimensions, camera, frame, static_cast<std::int32_t>(face));
            if (!projected)
            {
                continue;
            }

            // 诊断条目在超门时也要留下：norm 用四项归一化和，不经过门限判断
            AssociationDiagnostic diagnostic;
            diagnostic.candidate = obs;
            diagnostic.face = static_cast<std::int32_t>(face);
            diagnostic.corners = candidates[obs]->GetCorners();
            const std::optional<double> cost = QuadCost(*projected, candidates[obs]->GetCorners(), config, &diagnostic);

            const double normalized = diagnostic.center_error_px / config.center_gate_px + diagnostic.corner_error_px / config.corner_gate_px +
                                      diagnostic.angle_error_rad / config.angle_gate_rad + diagnostic.perimeter_ratio / config.perimeter_ratio_gate;
            if (std::isfinite(normalized))
            {
                diagnostic.normalized_cost = normalized;
                result.diagnostics.push_back(diagnostic);
            }

            if (cost)
            {
                costs[obs][face] = *cost;
            }
        }
    }

    // 诊断按归一化代价升序，日志里最前面几行就是最接近过门的组合
    std::sort(result.diagnostics.begin(), result.diagnostics.end(), [](const AssociationDiagnostic& lhs, const AssociationDiagnostic& rhs)
    {
        return lhs.normalized_cost < rhs.normalized_cost;
    });

    std::vector<bool>        used_face(face_count, false);
    std::vector<ArmorMatch>  current_matches;

    // 匹配的目标函数：优先最大化匹配数，其次最小化总代价；面序号必须落在环上连续的一段。
    // 四面车的面 0 与 2、1 与 3 互为对面，一个视角不可能同时看到，允许这种组合时
    // 面 0/2 只约束 r、1/3 只约束 r+l，r 与 l 会反向漂而 r+l 不动。
    const auto contiguous = [face_count](const std::vector<ArmorMatch>& matches)
    {
        if (matches.size() <= 1)
        {
            return true;
        }
        std::vector<std::int32_t> faces;
        faces.reserve(matches.size());
        for (const ArmorMatch& match : matches)
        {
            faces.push_back(match.face);
        }
        std::sort(faces.begin(), faces.end());

        // 环上连续：把所有相邻间隙（含首尾回绕）遍历一遍，空隙段最多只能有一处
        const std::int32_t count = static_cast<std::int32_t>(face_count);
        const std::int32_t used = static_cast<std::int32_t>(faces.size());
        if (used == count)
        {
            return true;
        }

        std::int32_t empty_runs = 0;
        for (std::size_t index = 0; index < faces.size(); ++index)
        {
            const std::int32_t current = faces[index];
            const std::int32_t next = faces[(index + 1) % faces.size()];
            const std::int32_t gap = (index + 1 < faces.size()) ? next - current : next + count - current;
            if (gap > 1)
            {
                ++empty_runs;
            }
        }
        return empty_runs == 1;
    };

    // 比较两份分配：先比匹配数，多者胜；同样多才比总代价
    const auto better = [](const std::vector<ArmorMatch>& lhs, double lhs_cost, const std::vector<ArmorMatch>& rhs, double rhs_cost)
    {
        if (lhs.size() != rhs.size())
        {
            return lhs.size() > rhs.size();
        }
        return lhs_cost < rhs_cost;
    };

    // 枚举所有合法的候选↔面分配。候选最多 4 块、面最多 4 个，组合数很小。
    double best_total_cost = std::numeric_limits<double>::max();
    std::function<void(std::size_t, double)> enumerate = [&](std::size_t obs, double total_cost)
    {
        if (obs == candidates.size())
        {
            if (contiguous(current_matches) && better(current_matches, total_cost, result.matches, best_total_cost))
            {
                best_total_cost = total_cost;
                result.matches = current_matches;
            }
            return;
        }

        // 当前候选不参与匹配
        enumerate(obs + 1, total_cost);
        // 或者配给某个还没用掉、且过了四门的预测面
        for (std::size_t face = 0; face < face_count; ++face)
        {
            if (used_face[face] || !std::isfinite(costs[obs][face]) || costs[obs][face] == kInvalidCost)
            {
                continue;
            }
            used_face[face] = true;
            current_matches.push_back(ArmorMatch {candidates[obs], static_cast<std::int32_t>(face), costs[obs][face]});
            enumerate(obs + 1, total_cost + costs[obs][face]);
            current_matches.pop_back();
            used_face[face] = false;
        }
    };
    enumerate(0, 0.0);

    // 配对按代价升序返回：调用方按这个顺序逐对修正 EKF，顺序本身不代表"当前面"
    std::sort(result.matches.begin(), result.matches.end(), [](const ArmorMatch& lhs, const ArmorMatch& rhs) { return lhs.cost < rhs.cost; });
    
    return result;
}

} // namespace algorithm::vision
