/**
 * @file ekf.hpp
 * @author qingyu
 * @brief
 * @version 0.1
 * @date 2026-09-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>

namespace algorithm::filter {

/**
* @brief 扩展卡尔曼滤波：过程与观测模型都是非线性的，各自在当前估计点用雅可比矩阵做一阶线性化，再按标准卡尔曼递推更新状态与协方差
*
* @tparam StateDim 状态向量维数
* @tparam MeasureDim 观测向量维数；可以是 Eigen::Dynamic，同一实例按帧做不同维数的更新
* @tparam Scalar 标量类型，默认 double
*/
template <std::size_t StateDim, int MeasureDim, typename Scalar = double>
class ExtendedKalmanFilter {
public:
    using State             = Eigen::Matrix<Scalar, StateDim, 1>;                       // 状态向量 x
    using Measurement       = Eigen::Matrix<Scalar, MeasureDim, 1>;                     // 观测向量 z
    using StateMatrix       = Eigen::Matrix<Scalar, StateDim, StateDim>;                // 状态方阵，用于协方差 P 与过程噪声 Q
    using MeasureMatrix     = Eigen::Matrix<Scalar, MeasureDim, MeasureDim>;            // 观测方阵，用于观测噪声 R
    using ObserveJacobian   = Eigen::Matrix<Scalar, MeasureDim, StateDim>;              // 观测雅可比 H = ∂h/∂x
    using KalmanGain        = Eigen::Matrix<Scalar, StateDim, MeasureDim>;              // 卡尔曼增益 K
    using ProcessModel      = std::function<State(const State&)>;                       // 过程模型 f(x)
    using MeasureModel      = std::function<Measurement(const State&)>;                 // 观测模型 h(x)
    using AddFunc           = std::function<State(const State&, const State&)>;         // 状态加法，默认逐元素
    using SubtractFunc      = std::function<Measurement(const Measurement&, const Measurement&)>; // 观测减法，默认逐元素

    /**
     * @brief 设置状态加法，默认逐元素相加
     *
     * 状态里含角度分量时把它换成"先相加、再把角度包回 (−π, π]"，挂在这个钩子里后，
     * 预测与修正都不必再在调用侧手动归一化。
     */
    void set_add(AddFunc add) { add_ = std::move(add); }

    /**
     * @brief 设置观测减法，默认逐元素相减
     *
     * 观测里含角度分量时换成"取最短角差"，否则 ±π 附近的残差会被算成一个整圈。
     */
    void set_subtract(SubtractFunc subtract) { subtract_ = std::move(subtract); }

    void set_state(const State& x, const StateMatrix& p) {
        x_ = x;
        p_ = p;
    }

    void set_process_noise(const StateMatrix& q)        { q_ = q; }
    void set_measurement_noise(const MeasureMatrix& r)  { r_ = r; }

    /**
     * @brief 预测步：用过程模型传播状态，用过程雅可比传播协方差
     *
     * @param f 过程模型 f(x)
     * @param process_jacobian 过程雅可比 F = ∂f/∂x，在 x_ 处取值
     * @return const State& 预测后的状态
     *
     * @note f 与 F 是分开传的，一致性只能靠调用方保证：F 必须是在当前 x_ 处取的 ∂f/∂x。
     *       若用了上一拍或标称工作点的雅可比，滤波器不会报错，只会收敛变慢甚至发散
     */
    const State& predict(const ProcessModel& f, const StateMatrix& process_jacobian) {
        x_ = f(x_);
        p_ = process_jacobian * p_ * process_jacobian.transpose() + q_;
        return x_;
    }

    /**
     * @brief 预测步（带控制输入）：x = f(x) + B u，协方差传播与无输入版本相同
     *
     * @tparam ControlDim 控制向量维数。必须是 int，不能跟类模板一样写 std::size_t：
     *         Eigen::Matrix 的行列模板参数是 int，写 size_t 会推导出 int 与声明不匹配，
     *         导致这个重载永远匹配不上
     * @param f 过程模型 f(x)
     * @param process_jacobian 过程雅可比 F = ∂f/∂x，在 x_ 处取值
     * @param b 控制输入矩阵 B
     * @param u 控制向量 u
     * @return const State& 预测后的状态
     *
     * @note F 的要求同无输入版本：必须在当前 x_ 处取值
     */
    template <int ControlDim>
    const State& predict(const ProcessModel& f, const StateMatrix& process_jacobian, const Eigen::Matrix<Scalar, StateDim, ControlDim>& b, const Eigen::Matrix<Scalar, ControlDim, 1>& u) 
    {
        x_ = f(x_) + b * u;
        p_ = process_jacobian * p_ * process_jacobian.transpose() + q_;
        return x_;
    }

    /**
     * @brief 更新步：用观测量、观测模型和观测雅可比修正状态与协方差
     *
     * @param z 观测向量
     * @param h 观测模型 h(x)
     * @param observe_jacobian 观测雅可比 H = ∂h/∂x，在 x_ 处取值
     * @return const State& 修正后的状态
     *
     * @note h 与 H 是分开传的，一致性只能靠调用方保证：H 必须是在当前 x_ 处取的 ∂h/∂x，
     *       理由同 predict
     */
    const State& correct(const Measurement& z, const MeasureModel& h, const ObserveJacobian& observe_jacobian)
    {
        const Measurement y = subtract_(z, h(x_));
        const MeasureMatrix s = observe_jacobian * p_ * observe_jacobian.transpose() + r_;
        // K = P Hᵀ S⁻¹。S 对称，转置后等价于 (S⁻¹ H P)ᵀ，用 ldlt 分解求解代替求逆
        const KalmanGain k = s.ldlt().solve(observe_jacobian * p_).transpose();
        x_ = add_(x_, k * y);

        // NIS：归一化新息平方 yᵀ S⁻¹ y。残差本身不参与递推，但它是"这帧观测像不像本状态"
        // 的唯一量化指标——滑窗统计超阈值比例，就能区分"跟丢/收敛到坏值"与正常跟踪
        last_nis_ = y.transpose() * s.ldlt().solve(y);
        ++update_count_;

        // 滑窗统计最近 window_size_ 次更新的失败数：参考 `recent_nis_failures` 的做法，
        // 避免早期异常永久污染失败率
        const Scalar threshold = nis_threshold_ > Scalar(0) ? nis_threshold_ : ChiSquare95(y.rows());
        recent_nis_failures_.push_back(last_nis_ > threshold ? 1 : 0);
        if (recent_nis_failures_.size() > window_size_)
        {
            recent_nis_failures_.pop_front();
        }
        if (recent_nis_failures_.back() == 1)
        {
            ++nis_failures_;
        }

        // Joseph 形式：比 (I - K H) P 多一层 (I - K H)ᵀ 和 K R Kᵀ，长时间递推能保住 P 的对称性与正定性
        const StateMatrix i_kh = StateMatrix::Identity() - k * observe_jacobian;
        p_ = i_kh * p_ * i_kh.transpose() + k * r_ * k.transpose();
        return x_;
    }

    const State& state() const { return x_; }
    const StateMatrix& covariance() const { return p_; }

    /**
     * @brief 最近一次修正的归一化新息平方
     */
    Scalar last_nis() const { return last_nis_; }

    /**
     * @brief 最近 `window_size` 次更新里 NIS 超阈值的比例
     *
     * @return [0, 1]；窗口为空时返回 0
     */
    Scalar nis_failure_rate() const
    {
        if (recent_nis_failures_.empty())
        {
            return Scalar(0);
        }

        Scalar failures = Scalar(0);
        for (const std::int32_t value : recent_nis_failures_)
        {
            failures += static_cast<Scalar>(value);
        }
        return failures / static_cast<Scalar>(recent_nis_failures_.size());
    }

    /**
     * @brief 设置滑窗长度，默认 100
     */
    void set_nis_window(std::size_t window_size) { window_size_ = window_size; }

    /**
     * @brief 设置 NIS 的卡方阈值
     *
     * 阈值取决于观测维数与置信水平，**必须按本工程的观测重定**：参考项目用 4 维
     * `[yaw, pitch, distance, angle]` 观测、阈值 0.711，本工程的观测是 8 维像素，
     * 自由度和量纲都不同。
     */
    void set_nis_threshold(Scalar threshold) { nis_threshold_ = threshold; }

    /**
     * @brief 清空 NIS 统计
     */
    void reset_nis()
    {
        last_nis_ = Scalar(0);
        update_count_ = 0;
        nis_failures_ = 0;
        recent_nis_failures_.clear();
    }

private:
    /**
     * @brief 卡方分布 95% 分位点，按观测维数查表
     *
     * 观测维数在运行期才确定（4 维 ypda 或 8 维角点），阈值必须跟着自由度走。
     */
    static Scalar ChiSquare95(Eigen::Index measurement_dim)
    {
        switch (measurement_dim)
        {
            case 1: return Scalar(3.841);
            case 2: return Scalar(5.991);
            case 3: return Scalar(7.815);
            case 4: return Scalar(9.488);
            case 5: return Scalar(11.070);
            case 6: return Scalar(12.592);
            case 7: return Scalar(14.067);
            case 8: return Scalar(15.507);
            default: return Scalar(15.507);
        }
    }

    State           x_{State::Zero()};                          // 状态估计
    StateMatrix     p_{StateMatrix::Identity()};                // 状态估计协方差
    StateMatrix     q_{StateMatrix::Identity()};                // 过程噪声协方差
    MeasureMatrix   r_{};                                       // 观测噪声协方差，维数由 set_measurement_noise 决定（动态尺寸不能用 Identity()）

    AddFunc         add_ = [](const State& a, const State& b) { return State(a + b); };            // 状态加法，默认逐元素
    SubtractFunc    subtract_ = [](const Measurement& a, const Measurement& b) { return Measurement(a - b); }; // 观测减法，默认逐元素

    Scalar          last_nis_ {0};                              // 最近一次修正的 NIS
    Scalar          nis_threshold_ {0};                         // NIS 卡方阈值；0 表示按观测维数自动取 95% 分位
    std::size_t     window_size_ {100};                         // NIS 失败率滑窗长度
    std::uint64_t   update_count_ {0};                          // 修正次数
    std::uint64_t   nis_failures_ {0};                          // NIS 超阈值的累计次数
    std::deque<std::int32_t> recent_nis_failures_;              // 最近 window_size_ 次是否失败
};

} // namespace algorithm::filter
