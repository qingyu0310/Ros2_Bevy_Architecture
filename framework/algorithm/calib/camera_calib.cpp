/**
 * @file camera_calib.cpp
 * @author qingyu
 * @brief 圆点阵相机标定实现 — 识别、样本签名、去重门禁与内参求解
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "framework/algorithm/calib/camera_calib.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <opencv2/calib3d.hpp>
#include <utility>

namespace algorithm::calib
{

namespace
{

// 平面靶标每幅对绝对二次曲线给 2 个约束，k3 固定后仍有 8 个内参待解，至少要 3 幅
constexpr std::size_t kMinSampleCount = 3;

/**
 * @brief 检查板规格能否构成点阵
 *
 * @param board 板规格
 * @return true 列、行都至少为 2，且间距为正的有限值
 */
bool IsBoardUsable(const Board& board)
{
    return board.cols >= 2 && board.rows >= 2 && std::isfinite(board.center_distance_mm) && board.center_distance_mm > 0.0;
}

/**
 * @brief 板点数，即圆心个数
 *
 * @param board 板规格，调用方保证已经过 IsBoardUsable
 * @return cols × rows
 */
std::size_t BoardPointCount(const Board& board)
{
    return static_cast<std::size_t>(board.cols) * static_cast<std::size_t>(board.rows);
}

/**
 * @brief 板点在板坐标系下的坐标，行优先
 *
 * 板平面取 z = 0，x 沿列方向、y 沿行方向，与 findCirclesGrid 返回圆心的行优先顺序一一对应。
 * 单位是 mm，大小只影响解出的平移量，不影响内参与畸变。
 *
 * @param board 板规格
 * @return cols × rows 个板点，单位 mm
 */
std::vector<cv::Point3f> MakeBoardPoints(const Board& board)
{
    std::vector<cv::Point3f> points;
    points.reserve(BoardPointCount(board));

    for (std::int32_t row = 0; row < board.rows; ++row)
    {
        for (std::int32_t col = 0; col < board.cols; ++col)
        {
            points.emplace_back(static_cast<float>(col * board.center_distance_mm), static_cast<float>(row * board.center_distance_mm), 0.0f);
        }
    }

    return points;
}

} // namespace

/**
 * @brief 用对称圆点图案识别一帧点阵
 *
 * 结果先写进局部量，识别失败时调用方的输出保持原样，避免半识别状态被当成有效样本。
 *
 * @param bgr_image 非空 BGR 图像
 * @param board 板规格
 * @param centers 输出全部圆心，行优先，单位 pixel
 * @return true 识别到完整点阵
 */
bool DetectBoard(const cv::Mat& bgr_image, const Board& board, std::vector<cv::Point2f>& centers)
{
    if (bgr_image.empty() || !IsBoardUsable(board))
    {
        return false;
    }

    std::vector<cv::Point2f> detected;
    const cv::Size pattern_size(board.cols, board.rows);
    if (!cv::findCirclesGrid(bgr_image, pattern_size, detected, cv::CALIB_CB_SYMMETRIC_GRID))
    {
        return false;
    }

    centers = std::move(detected);
    return true;
}

/**
 * @brief 由圆心算板心、尺度与两个对边比
 *
 * 四角圆心的索引按行优先取：0 是左上、cols−1 是右上、cols·(rows−1) 是左下、末位是右下。
 * 对边比取长边除以短边，因此与这两组对边谁长谁短无关，也不受识别返回顺序翻转的影响。
 *
 * @param board 板规格
 * @param centers 一帧的全部圆心
 * @return 签名；个数不符、坐标非有限或任一边退化时返回 std::nullopt
 */
std::optional<SampleShape> MeasureShape(const Board& board, const std::vector<cv::Point2f>& centers)
{
    if (!IsBoardUsable(board) || centers.size() != BoardPointCount(board))
    {
        return std::nullopt;
    }

    SampleShape shape;
    for (const auto& center : centers)
    {
        if (!std::isfinite(center.x) || !std::isfinite(center.y))
        {
            return std::nullopt;
        }
        shape.center += center;
    }
    shape.center *= 1.0f / static_cast<float>(centers.size());

    const std::size_t last_col = static_cast<std::size_t>(board.cols - 1);
    const std::size_t last_row = static_cast<std::size_t>(board.rows - 1) * static_cast<std::size_t>(board.cols);
    const std::size_t last     = centers.size() - 1;

    const float top    = cv::norm(centers[last_col] - centers[0]);
    const float right  = cv::norm(centers[last] - centers[last_col]);
    const float bottom = cv::norm(centers[last] - centers[last_row]);
    const float left   = cv::norm(centers[last_row] - centers[0]);

    if (!(top > 0.0f) || !(right > 0.0f) || !(bottom > 0.0f) || !(left > 0.0f))
    {
        return std::nullopt;
    }

    shape.span_px     = (top + right + bottom + left) * 0.25f;
    shape.pitch_ratio = std::max(top, bottom) / std::min(top, bottom);
    shape.yaw_ratio   = std::max(left, right) / std::min(left, right);

    return shape;
}

/**
 * @brief 四项门限逐个比对，任一项与已收样本区分开就算新姿态
 *
 * 板心位移、尺度、俯仰比、偏航比分别对应平移、远近、绕横轴转、绕竖轴转；
 * 与每一个已收样本都至少有一项不同，才算提供了新的约束方向。
 *
 * @param shape 新样本签名
 * @param accepted 已接受的样本签名
 * @param gate 去重门限
 * @return true 与每个已收样本都满足其一
 */
bool IsDistinctSample(const SampleShape& shape, const std::vector<SampleShape>& accepted, const CaptureGate& gate)
{
    for (const auto& existing : accepted)
    {
        if (cv::norm(shape.center - existing.center) >= gate.min_shift_px)
        {
            continue;
        }

        const float larger_span = std::max(shape.span_px, existing.span_px);
        if (larger_span > 0.0f && std::abs(shape.span_px - existing.span_px) / larger_span >= gate.min_span_change)
        {
            continue;
        }

        if (std::abs(shape.pitch_ratio - existing.pitch_ratio) >= gate.min_ratio_change)
        {
            continue;
        }
        if (std::abs(shape.yaw_ratio - existing.yaw_ratio) >= gate.min_ratio_change)
        {
            continue;
        }

        return false;
    }

    return true;
}

/**
 * @brief 用一批样本解内参、畸变，并按参考口径复算重投影误差
 *
 * 求解用 cv::calibrateCamera：视场窄所以固定 k3，迭代次数按参考从默认的 30 提到 100。
 * 误差不用返回值里的 RMS，而是逐样本 projectPoints 后取"全部点距离和 ÷ 点数"，
 * 与参考口径一致，便于两边数字对照。OpenCV 抛出的失败一律转成 std::nullopt。
 *
 * @param board 板规格
 * @param image_points 每个样本的全部圆心，行优先，单位 pixel
 * @param image_size 标定图像的宽高
 * @return 求解结果；样本数不足、个数不符或求解失败时返回 std::nullopt
 */
std::optional<Intrinsics> SolveIntrinsics(const Board& board, const std::vector<std::vector<cv::Point2f>>& image_points, const cv::Size& image_size)
{
    if (!IsBoardUsable(board) || image_size.width <= 0 || image_size.height <= 0 || image_points.size() < kMinSampleCount)
    {
        return std::nullopt;
    }

    const std::size_t point_count = BoardPointCount(board);
    for (const auto& sample : image_points)
    {
        if (sample.size() != point_count)
        {
            return std::nullopt;
        }
    }

    const std::vector<cv::Point3f> board_points = MakeBoardPoints(board);
    const std::vector<std::vector<cv::Point3f>> object_points(image_points.size(), board_points);

    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;
    std::vector<cv::Mat> rotation_vectors;
    std::vector<cv::Mat> translation_vectors;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);

    try
    {
        cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, distort_coeffs, rotation_vectors, translation_vectors, cv::CALIB_FIX_K3, criteria);
    }
    catch (const cv::Exception&)
    {
        return std::nullopt;
    }

    if (camera_matrix.rows != 3 || camera_matrix.cols != 3 || camera_matrix.type() != CV_64F ||
        rotation_vectors.size() != image_points.size() || translation_vectors.size() != image_points.size())
    {
        return std::nullopt;
    }

    // k3 固定时 OpenCV 给 5 个系数，只给 4 个时末位保持结构体里的 0
    const std::size_t coefficient_count = distort_coeffs.total();
    if ((coefficient_count != 4 && coefficient_count != 5) || distort_coeffs.type() != CV_64F || !distort_coeffs.isContinuous())
    {
        return std::nullopt;
    }
    const double* const coefficients = distort_coeffs.ptr<double>();

    Intrinsics intrinsics;
    for (std::int32_t row = 0; row < 3; ++row)
    {
        for (std::int32_t col = 0; col < 3; ++col)
        {
            const double value = camera_matrix.at<double>(row, col);
            if (!std::isfinite(value))
            {
                return std::nullopt;
            }
            intrinsics.camera_matrix[row * 3 + col] = value;
        }
    }
    for (std::size_t index = 0; index < coefficient_count; ++index)
    {
        if (!std::isfinite(coefficients[index]))
        {
            return std::nullopt;
        }
        intrinsics.distort_coeffs[index] = coefficients[index];
    }

    double error_sum    = 0.0;
    std::size_t total_points = 0;
    for (std::size_t index = 0; index < image_points.size(); ++index)
    {
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(object_points[index], rotation_vectors[index], translation_vectors[index], camera_matrix, distort_coeffs, reprojected);
        if (reprojected.size() != image_points[index].size())
        {
            return std::nullopt;
        }

        for (std::size_t point = 0; point < reprojected.size(); ++point)
        {
            error_sum += cv::norm(image_points[index][point] - reprojected[point]);
        }
        total_points += reprojected.size();
    }
    if (total_points == 0)
    {
        return std::nullopt;
    }

    intrinsics.reprojection_error_px = error_sum / static_cast<double>(total_points);
    intrinsics.sample_count          = static_cast<std::int32_t>(image_points.size());
    intrinsics.image_width           = image_size.width;
    intrinsics.image_height          = image_size.height;

    return intrinsics;
}

} // namespace algorithm::calib
