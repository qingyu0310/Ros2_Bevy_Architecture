# 跟踪器数据流与 EKF

日期：2026-10-18  
对象：`framework/algorithm/vision/track/`、`ros2_layer/node/vision/vision_node.hpp`  
说明：本文按当前源码写，路径与符号以工作区为准；不含代码修改。

---

## 0. 一句话概括

每一帧，检测器给出若干块装甲板；`Tracker` 决定这一帧是“建目标”还是“更新目标”；`Association` 把检测板配到车的四个预测面上；`Target` 用配上的板做一次 EKF 预测与修正；最后 `Tracker` 根据更新结果推进生命周期状态，并把状态、统计和四面输出给节点。

数据流可以写成：

```text
图像
  → ArmorDetector（网络 + NMS + 几何校验）
  → DetectedArmor 列表
  → [节点] 敌方颜色过滤
  → Tracker::Update
      → Lost   : SetTarget → Target::Update → Initialize（PnP 起状态）
      → 其它态 : UpdateTarget
          → Target::Predict（每帧一次）
          → Associate（候选过滤 + 四门代价 + 一对一匹配）
          → 对每个 match：Target::Update(dt=0)
              → UpdateImagePoints（主路径）
              → 或 PnP ypda（回退）
      → AdvanceState（状态机）
      → NIS / Diverged 判据（判丢重建）
      → report_（active / converged / 统计）
  → [节点] 取 Faces() / Radius() / DebugSnapshot() 做绘制、自瞄、日志
```

---

## 1. 输入、输出与坐标系

### 1.1 输入

| 数据 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| 检测板列表 | `std::vector<DetectedArmor>` | `ArmorDetector::Detect()` | 每块板含四角、中心、外接框、颜色、编号、大小、置信度 |
| 相机内参 | `CameraModel` | `/vision/camera_info` | `fx, fy, cx, cy, k1, k2, p1, p2, k3` |
| 相机→跟踪系位姿 | `FrameTransform` | TF `map ← camera_optical_frame` | `p_world = R·p_camera + t` |
| 时间间隔 | `double dt_s` | 图像时间戳差 | 节点里默认 `1/30 s`，测到 `0 < dt < 1` 时用实测 |
| 实物尺寸 | `ArmorDimensions` | 代码默认值 | 节点当前传 `ArmorDimensions{}`，即小 0.135 m、大 0.230 m、灯条 0.056 m |

节点层在送进跟踪器之前先做两件事：

1. 只保留 `enemy_color` 对应的颜色，避免同号红蓝两辆车抢同一个目标槽；
2. 查询本帧的 TF；查不到时不调用 `Tracker::Update`，只调用 `Tracker::Predict(dt_s)`，因为相机系观测不能直接修正世界系状态。

### 1.2 输出

| 输出 | 位置 | 用途 |
| --- | --- | --- |
| `TrackerReport` | `Tracker::Report()` | 状态、更新/丢失/门拒/重建计数、最后面序与代价 |
| `std::vector<FaceState>` | `Tracker::Faces()` | 四个面的中心与法向 yaw，跟踪系 |
| `double Radius()` | `Tracker::Radius()` | 主半径估计，日志与调试 |
| `ArmorSize Size()` | `Tracker::Size()` | 当前目标的板大小 |
| `Target::Snapshot` | `Tracker::DebugSnapshot()` | 11 维状态、协方差对角、NIS、三条计数器 |

节点再把它们用于：

- 绘制：四个面按状态重投影成像素轮廓；当前面绿色，其余黄色；画自瞄点与画面中心准星；
- 自瞄：取离相机最近的面心，换回相机系，经 TF 转到 `map` 后算 yaw/pitch/distance；
- 日志：每秒汇总一次。

### 1.3 坐标系

| 坐标系 | 约定 | 备注 |
| --- | --- | --- |
| 相机系 | OpenCV 约定，x 向右、y 向下、z 向前 | PnP 与像素投影都在这里 |
| 跟踪系 | x-y 水平、z 竖直向上 | 对应 TF 的 `map`；EKF 状态全部在这里 |
| 板系 | x 为板法向、y 沿板宽、z 沿灯条方向 | 原点在板心，板面为 x=0 |
| 像素 | u 向右、v 向下 | 四角顺序固定 LT、RT、RB、LB |

两个方向的变换：

```text
CameraToFrame(p) = R · p + t
FrameToCamera(p) = Rᵀ · (p − t)
CameraToFrameDirection(v) = R · v
FrameToCameraDirection(v) = Rᵀ · v
```

`FrameTransform::rotation` 是相机系到跟踪系的旋转，`translation` 是相机原点在跟踪系中的位置。

---

## 2. 每帧的数据流（详细）

### 2.1 节点层：`VisionNode::on_image`

1. 没有收到内参就直接跳过本帧（`camera_` 为空）。
2. 用 `msg.data` 和 `msg.step` 构造 `cv::Mat`，交给 `ArmorDetector::Detect`。
3. 计算 `dt_s`：
   - 默认 `1/30 s`；
   - 有上一帧时间戳时，用 `(stamp − last_stamp)`，只接受 `0 < dt < 1`；
   - 时间戳全部用 ROS 时间，不能和墙钟混用。
4. 只保留敌方颜色的板，得到 `enemy_armors`。
5. 查 `map ← camera_optical_frame` 的 TF：
   - 先按图像时间戳查；
   - `ExtrapolationException` 时退到最新一帧（TF 频率可能低于图像帧率）；
   - 仍失败则 `frame_` 为空。
6. `frame_` 存在 → `tracker_->Update(enemy_armors, ArmorDimensions{}, *camera_, *frame_, dt_s)`；
   不存在 → `tracker_->Predict(dt_s)`，只推进预测，不做修正。
7. 绘制、发布 `/vision/debug_image`、发布 `/rm_gimbal/cmd`、每秒日志。

### 2.2 `Tracker::Update`

**作用**：一帧的总调度与生命周期判断。

1. 扫一遍所有检测板，找离图像中心 `(camera.cx, camera.cy)` 最近的一块，记为 `nearest`。
   - 这只是在 `Lost` 时用来起目标；跟踪中不用它换板。
2. 按状态分派：
   - `Lost`：`nearest` 存在就 `SetTarget(*nearest, ...)`；失败则记录 `last_reset_reason = InitFailed`。
   - 其它状态：`UpdateTarget(armors, ...)`。
3. `found` 表示这一帧有没有拿到有效更新：
   - `Lost` 下建目标成功为 true；
   - 其它状态下至少一对匹配更新成功为 true。
4. `!found && state != Lost` → `report_.losses++`。
5. `AdvanceState(found)`：跑四状态机。
6. NIS 判据：`state != Lost && target_.UpdateCount() >= 10 && NisFailureRate() >= nis_failure_ratio` → `ResetTarget(ResetReason::Nis)`。
   - `>= 10` 是防止第一帧异常就以 `1/1 = 100%` 触发重建。
7. 发散判据：`state != Lost && (target_.Diverged() || !target_.Active())` → `ResetTarget(ResetReason::Diverged)`。
8. 写 `report_.active`：
   - `Lost`：false；
   - 其它：`target_.Active()`；
   - `report_.converged = state == Tracking && target_.Converged()`。

跟踪效果：

- 第 2 步决定“这一帧是起跟还是续跟”；
- 第 4 步的 `losses` 是“有目标但没匹配上”的帧数，和“完全没有目标”不是一回事；
- 第 6、7 步是唯一的判丢入口，二者都会清掉 EKF 状态；
- 第 8 步决定下游能不能用这个目标。

### 2.3 `Tracker::SetTarget`

**作用**：从一块板建立一辆车。

1. `target_.Reset()`：清 `initialized_`、更新计数、三条诊断计数、连续钳制计数，并清 EKF 的 NIS 滑窗。
2. `target_.Update(corners, size, dimensions, camera, frame, dt=0, face=0, estimate_shape=false)`：
   - 因为 `initialized_ == false`，内部直接走 `Initialize`，不做预测、不做观测修正；
   - `face=0` 表示把这块板当作车的第 0 面；
   - `estimate_shape=false` 在这条路径上不参与计算，因为初始化不走 H。
3. 成功后记录 `label_`、`size_`，`report_.active = true`，`report_.updates = target_.UpdateCount()`。

### 2.4 `Target::Initialize`

**作用**：用第一帧的 PnP 位姿把 11 维状态放到一个具体位置。

1. `MakeObjectPoints(size, dimensions)`：按板大小生成板系四个物点。
2. `SolvePose`：`cv::solvePnP`（IPPE）解出板系到相机系的 `rvec/tvec`。
3. `Rodrigues` 转成旋转矩阵，取第一列（板法向），用 `frame.rotation` 转到跟踪系，`atan2` 得 `raw_yaw`。
4. `pose_center = CameraToFrame(tvec)`：板心换到跟踪系。
5. `OptimizeYaw`：在 ±70° 内按四角重投影误差搜索，解决 IPPE 对共面四点的二义解。
6. 旋转中心：

```text
n = (−cos yaw, −sin yaw, 0)
center = face_center − r_init · n
```

`r_init = radius_init_m = 0.2 m`，`l = h = 0`，速度与角速度为 0。

7. 初值协方差（对角）：

```text
cx,cy,cz : 0.01
vx,vy,vz : 0.01
a        : 0.09
w        : 0.25
r        : 0.04
l,h      : 0.01
```

8. `initialized_ = true`，`update_count_ = 0`。

跟踪效果：

- 初始化后 face 0 的投影与第一帧检测四点重合，残差为 0；
- 另外三个面是按 `r=0.2, l=h=0` 外推的，只有当前面有观测支撑；
- 此时只是“有状态”，状态机还停在 `Detecting`，不能自瞄。

### 2.5 `Tracker::UpdateTarget`

**作用**：一帧的预测、关联和逐块更新。

1. 参数检查：`dt_s` 必须有限、非负、不超过 `0.3 s`，且 `target_.Active()`。
2. `target_.Predict(dt_s)`：**每帧只预测一次**，即使本帧没有候选也要推进时间。
3. `Associate(...)`：候选过滤、四门代价、一对一匹配，输出 `matches` 和 `had_candidates`。
4. `matches` 为空：
   - 有候选 → `gate_rejects++`（这一帧全被四门拒了）；
   - 没候选 → 不计门拒。
   - 返回 false。
5. `estimate_shape = matches.size() >= 2`：只有本帧匹配到至少两个面，才允许估计 `r/l/h`。
6. 按代价升序遍历 `matches`，每个调用 `target_.Update(..., dt=0, face, estimate_shape)`：
   - `dt=0` 是因为第 2 步已经预测过，同帧多次更新不再推进时间；
   - 每个配对更新一次 EKF；失败的配对跳过。
7. `report_.updates = target_.UpdateCount()`，返回是否有任何一次成功。

跟踪效果：

- 第 2 步保证“没有检测的帧”状态也会按运动模型前进，不会停在旧时间；
- 第 5 步是单板/多板的分界：单板只修正位置和角度，多板才修正形状；
- 第 6 步的更新顺序按代价从低到高，代价最低的配对先修正状态，后续配对在修正后的状态上继续。

### 2.6 `Association::Associate`

**作用**：回答“这一帧的哪块板是车的哪个面”。

1. 候选过滤：只保留 `label == target label` 且 `size == target size` 的板。
2. 没有候选 → 返回 `had_candidates = false`。
3. 取 `target.Faces()`：当前状态推出四个面的中心与 yaw。
4. 对每个候选 × 每个面：
   - `target.ProjectFace(size, ...)` 把该面投成四个像素点；
   - `QuadCost` 算四项误差并做门控；
   - 通过门控的填进 `costs[obs][face]`，否则保持无效值。
5. 枚举所有合法一对一匹配：
   - 每个候选可以“不参与匹配”，也可以配一个尚未使用的面；
   - 评价顺序：**先最大化匹配数量，再最小化总代价**；
   - 这样不会为了低代价而丢掉本来能配上的第二块板。
6. 结果按代价升序排序返回。

### 2.7 `Association::QuadCost`

四项误差，全部转成“占门限的比例”：

| 项 | 计算 | 门限 |
| --- | --- | --- |
| 中心误差 | 预测四点均值与检测四点均值的欧氏距离 | `center_gate_px = 100` |
| 角点误差 | 四个对应角点距离的平均值 | `corner_gate_px = 140` |
| 方向误差 | 四条边方向角差的绝对值之和 | `angle_gate_rad = 0.8` |
| 周长误差 | `|预测周长 − 检测周长| / 预测周长` | `perimeter_ratio_gate = 0.6` |

任一项超门 → 该配对无效。全部通过时：

```text
cost = center/100 + corner/140 + angle/0.8 + perimeter/0.6
```

四项各占 1 的上界，所以总代价上界是 **4.0**。日志里 `cost = 2.351` 表示四项平均用掉了各自门限的 58.8%，仍然算有效配对。

### 2.8 `Target::Update`（每个配对）

**作用**：用一块板推进一次 EKF。

1. 检查 `face` 范围和角点有限性，记下 `estimate_shape_`。
2. 未初始化 → `Initialize`，`update_count_++`，返回。
3. `Predict(dt_s)`；这里多数情况 `dt_s = 0`（同帧后续配对）。
4. 主路径 `UpdateImagePoints`：
   - 成功 → `estimate_shape` 时 `ApplyLimits()`，检查状态/协方差有限，`image_updates_++`、`update_count_++`；
   - 失败 → 回退。
5. 回退路径 PnP ypda：
   - `SolvePose` → 板心、法向、`OptimizeYaw`；
   - 观测 `[yaw, pitch, distance, armor_yaw]`；
   - 数值雅可比 + `YpdaMeasurementCovariance` 的 R；
   - `correct`，`estimate_shape` 时 `ApplyLimits()`；
   - `ypda_updates_++`、`update_count_++`。

### 2.9 `Target::Predict`

**作用**：按时间推进状态和协方差。

状态转移矩阵 F：

```text
cx += vx·dt
cy += vy·dt
cz += vz·dt
a  += w·dt
```

其余状态（速度、角速度、r、l、h）在 F 中是常量项。

过程噪声 Q 采用分段白噪声：

```text
a = dt⁴/4
b = dt³/2
c = dt²

位置/速度块：a·v1, b·v1, c·v1
角度/角速度块：a·v2, b·v2, c·v2
v1 = accel_variance = 10
v2 = angular_accel_variance = 100
```

形状量：

```text
r, l, h 的过程噪声 = shape_variance · c
shape_variance = 0.1
```

但只有在 `estimate_shape_ == true` 时才加；单板帧形状噪声为 0，避免协方差被噪声推大。

角度归一化由 EKF 的加法钩子完成：每次状态相加后把 `a` 包回 `(−π, π]`。

### 2.10 `Target::UpdateImagePoints`

**作用**：8 维像素观测的 EKF 测量更新，主路径。

1. 预测像素：`h(x) = ObservePoints(state, face)`，四个角点共 8 个数。
2. 观测 `z`：检测四角 `(u,v)`。
3. 数值雅可比：
   - 对 `cx, cy, cz, a`（`estimate_shape` 时再加 `r, l, h`）；
   - 每个量分别加/减 `1e-3`（角度为 `1e-3 rad`），投影两次；
   - `(forward − backward) / (2·step)` 填进 H 的对应列；
   - `H` 是 8×11，速度与角速度列保持 0。
4. 观测噪声：

```text
R = σ² · I₈
σ = point_sigma_px = 8 px
方差 = 64 px²
```

5. `filter_.correct(z, h, H)`：
   - 新息 `y = z − h(x)`；
   - `S = H·P·Hᵀ + R`；
   - `K = P·Hᵀ·S⁻¹`；
   - `x += K·y`；
   - 协方差用 Joseph 形式：`P = (I−KH)P(I−KH)ᵀ + K R Kᵀ`；
   - 计算 NIS：`yᵀ·S⁻¹·y`；
   - 把“是否超阈”压进最近 100 次滑窗。
6. 返回状态和协方差是否有限。

### 2.11 回退路径：PnP 的 ypda 观测

**作用**：像素观测不可用时，用 PnP 解出的 `[yaw, pitch, distance, armor_yaw]` 继续修正。

1. `SolvePose` 解出板位姿。
2. 取旋转矩阵第一列作为法向，换到跟踪系，`OptimizeYaw` 解二义。
3. 观测量：

```text
z = [yaw, pitch, distance, armor_yaw]
```

4. 数值雅可比 4×11，状态列与主路径相同。
5. R 优先用 `YpdaMeasurementCovariance`：
   - 对位姿扰动 `[dt(3), dθ(3)]` 做像素四点的一阶传播；
   - 组合信息矩阵后投影回 ypda 空间；
   - 失败时回退对角公式：

```text
R[0,0] = R[1,1] = 4e-3
R[2,2] = log(|Δangle| + 1) + 1
R[3,3] = log(|distance| + 1)/200 + 1e-2
```

6. `correct` 后按 `estimate_shape` 决定是否 `ApplyLimits`。

---

## 3. EKF 本身

### 3.1 状态

11 维，顺序固定：

| 下标 | 符号 | 含义 | 单位 |
| ---: | --- | --- | --- |
| 0 | cx | 旋转中心 x | m |
| 1 | vx | x 方向速度 | m/s |
| 2 | cy | 旋转中心 y | m |
| 3 | vy | y 方向速度 | m/s |
| 4 | cz | 旋转中心 z | m |
| 5 | vz | z 方向速度 | m/s |
| 6 | a | 当前面法向角 | rad |
| 7 | w | 角速度 | rad/s |
| 8 | r | 主半径 | m |
| 9 | l | 第二半径差（r2 − r1） | m |
| 10 | h | 面 1/3 的高度差 | m |

四个面的几何：

```text
angle(face) = wrap(a + face · 2π/armor_num)
use_second  = armor_num == 4 && (face == 1 || face == 3)
radius      = use_second ? r + l : r
center(face) = (cx − radius·cos(angle),
                cy − radius·sin(angle),
                cz + (use_second ? h : 0))
```

### 3.2 预测

```text
x⁻ = F·x
P⁻ = F·P·Fᵀ + Q
```

`F` 只把速度积到位置、角速度积到角度。`Q` 是分段白噪声，位置与角度用不同的加速度方差。

### 3.3 观测

| 路径 | 维数 | 观测量 | 噪声 |
| --- | ---: | --- | --- |
| 主路径 | 8 | 四角像素 `u0,v0,…,u3,v3` | 各向同性 `σ²`，σ=8 px |
| 回退 | 4 | `yaw, pitch, distance, armor_yaw` | PnP 一阶传播或对角公式 |

两条路径共用同一个 `ArmorCenter` / `FaceAngle` 几何；区别只是把面投成像素还是球坐标。

### 3.4 更新

```text
y = z − h(x⁻)
S = H·P⁻·Hᵀ + R
K = P⁻·Hᵀ·S⁻¹
x = x⁻ + K·y
P = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ
```

Joseph 形式是为了长时间递推后 `P` 仍然对称、正定。

### 3.5 NIS

```text
NIS = yᵀ · S⁻¹ · y
```

- 单次 NIS 大只说明这一帧观测和状态不一致，不直接判丢；
- EKF 维护最近 100 次“是否超阈”的滑窗；
- 阈值：`nis_threshold > 0` 时用它，否则按观测维数取 χ² 95% 分位（8 维 15.507，4 维 9.488）；
- `NisFailureRate()` = 滑窗内失败次数 / 滑窗长度；
- `Tracker` 在 `UpdateCount() >= 10` 后检查失败率，`>= 0.8` 才判丢。

### 3.6 角度处理

两个钩子：

- `set_add`：状态相加后把 `a` 包回 `(−π, π]`；
- `set_subtract`：观测残差的 `yaw, pitch, armor_yaw` 取最短角差，避免 ±π 附近差一整圈。

没有这两个钩子，滤波器在角度越过 ±π 时会看到“突变一圈”，直接判发散。

### 3.7 可观性

单块板只能观测到：

```text
面心 = 圆心 − r·法向
```

圆心沿法向平移 ε、同时 r 加 ε，面心完全不变，八个像素也完全不变。所以单板观测的雅可比至少有一个零空间方向；面 1/3 更直接：r 和 l 只以 `r+l` 出现，两列雅可比相同，只能估出 `r+l`。

这就是 `estimate_shape` 存在的原因：

- 单板帧：H 不含 r/l/h 三列，这三列全 0 → 卡尔曼增益对应行也是 0 → 状态和协方差都不动；
- 多板帧：至少两个面有观测，弦长和相对转角提供额外约束，才把 r/l/h 放进 H。

对称四面车的弦长关系：

```text
chord² = r² + (r + l)²
l = 0 时：r = chord / √2
```

实测两板帧：观测板心 `(0.065,-0.058,2.009)` 与 `(0.341,-0.033,1.934)`，弦长 `0.287 m`，反算 `r = 0.203 m`。同一时刻滤波器给 `0.291 m`，误差 0.088 m。

### 3.8 形状约束

`ApplyLimits()` 只在 `estimate_shape == true` 时调用：

```text
r   ∈ [0.05, 0.50] m
r+l ∈ [0.05, 0.50] m
```

钳制后：

- `r` 的行/列协方差清零，方差加上钳制位移的平方；
- `l` 的方差加上第二半径位移的平方。

这样不会出现“状态被硬拉回边界，但协方差还假装很准”的情况。连续钳制次数超过 `max_consecutive_limit_clamps = 5` 时，`Diverged()` 返回 true。

---

## 4. 生命周期与诊断

### 4.1 四状态机

```text
lost ──found──> detecting ──连续 min_detect_count 帧──> tracking
                    │                                      │
                    └──一帧没找到──> lost                    └──一帧没找到──> temp_lost
                                                                │
                                                    连续 max_temp_lost_count 帧
                                                                │
                                                                v
                                                              lost
```

| 状态 | 含义 | 对外 |
| --- | --- | --- |
| `Lost` | 没有有效目标，允许当帧重建 | `active = 0` |
| `Detecting` | 已建目标，连续确认中 | `active = 1`，`converged = 0` |
| `Tracking` | 稳定跟踪 | `active = 1`，`converged` 看 Target |
| `TempLost` | 暂时丢失，靠预测维持 | `active = 1`，`converged = 0` |

参数：`min_detect_count = 5`，`max_temp_lost_count = 25`。

### 4.2 判丢与重建

统一入口 `ResetTarget(reason)`：

```text
target_.Reset()
report_.active = false
report_.converged = false
report_.last_face = -1
report_.state = Lost
report_.detect_count = 0
report_.temp_lost_count = 0
report_.last_reset_reason = reason
report_.resets++
```

`Target::Reset()` 清掉：

- `initialized_`、`update_count_`；
- `image_updates_`、`ypda_updates_`、`limit_clamps_`；
- `consecutive_limit_clamps_`；
- **EKF 的 NIS 滑窗**（`filter_.reset_nis()`）。

最后一条是关键：不清 NIS 历史，新目标会被上一条目标的失败记录直接判死。

三种原因：

| 原因 | 触发条件 |
| --- | --- |
| `Nis` | `UpdateCount >= 10` 且 `NisFailureRate >= 0.8` |
| `Diverged` | `r` 或 `r+l` 越界，或连续撞边界 ≥5 次，或滤波器不再 active |
| `InitFailed` | `Lost` 下当帧建目标失败（只记录原因，不计 `resets`） |

### 4.3 报告字段

| 字段 | 含义 | 跟踪效果 |
| --- | --- | --- |
| `active` | 滤波器是否持有目标 | 下游是否画框/自瞄 |
| `converged` | `Tracking` 且 `Target::Converged()` | 是否允许输出稳定目标 |
| `updates` | 累计观测更新次数 | 判断是否长期无更新 |
| `losses` | 有目标但未匹配的帧数 | 关联失败频率 |
| `gate_rejects` | 有候选但四门全拒的帧数 | 门限过紧或状态漂移 |
| `resets` | 判丢重建次数 | 稳定性 |
| `last_reset_reason` | 最近一次重建原因 | `nis` / `diverged` / `init_failed` |
| `last_face` | 最近一次匹配到的面 | 面序切换是否频繁 |
| `last_cost` | 最近一次匹配代价 | 关联质量，上界 4.0 |

### 4.4 三条观测路径计数

| 计数 | 位置 | 用途 |
| --- | --- | --- |
| `image_updates` | `Target` | 8 维角点更新成功次数 |
| `ypda_updates` | `Target` | PnP 回退更新成功次数 |
| `limit_clamps` | `Target` | r/r+l 被钳制的次数 |

日志里每秒的增量能直接看出这一秒走的是哪条路径、形状量有没有被硬拉。

### 4.5 节点日志字段

每秒一条：

```text
state / active / converged / updates / losses / gate / resets / reset / face / cost /
r / nis / nis_rate
obs image / ypda / limit_clamps / tf_missing
ekf center / vel / angle / w / r / l / h
ekf P diag
face=0..3 xyz_world / yaw
observed xyz_cam（检测板 PnP 解出的板心）
```

这些字段的用途：

- `updates` 反复归零 → 重建；
- `gate` 增长 → 关联被拒；
- `resets` + `reset` → 判丢原因；
- `nis_rate` → 离判丢还有多远；
- `limit_clamps` 增长 → 形状量在撞边界；
- `tf_missing` → 是否只在预测；
- `observed xyz_cam` 与 `face xyz_world` 不能直接相减（前者相机系、后者跟踪系），要看 `observed` 经 TF 转过去后的对照。

---

## 5. 跟踪效果：每一环影响什么

### 5.1 检测与颜色过滤

- 编号/尺寸是关联的第一层筛选；编号错或尺寸错，板根本进不了候选。
- 敌方颜色过滤在节点层完成；同号红蓝车不会同时进入跟踪器。
- 检测框抖动直接进像素观测，是 EKF 的原始噪声来源。

### 5.2 选板（Lost 时）

- 只按“离画面中心最近”选。画面中心附近恰好是另一辆同编号车时，会建错目标；之后靠编号/尺寸和四门维持。
- 参考项目还有兵种优先级排序；本工程没有优先级来源。

### 5.3 关联

- 门限太松：错误配对进 EKF，NIS 升高，r/l/h 被拉坏；
- 门限太紧：正确板被拒，`losses`/`gate_rejects` 增长，进入 `temp_lost`；
- 匹配采用“先最大化数量、再最小化代价”，两块板同时可见时不会为了低代价丢掉第二块；第二块对 r/l/h 至关重要。

### 5.4 EKF 修正

- 主路径 8 维像素更新：直接消像素残差，跟踪框贴得紧；
- 回退 4 维 ypda：PnP 深度噪声大，但像素投影失败时仍能维持；
- NIS 大：观测与模型不一致；连续大则判丢重建，防止坏状态继续输出。

### 5.5 形状量

- 单板帧 r/l/h 冻结：圆心和 r 不可分，强行更新会把 r 推到边界；
- 多板帧才估计：两板弦长提供 `r`（`l≈0` 时 `r = chord/√2`），三/四板更强；
- `ApplyLimits` 是最后一道物理约束，但钳制只能在多板帧发生；连续钳制会触发 `Diverged`，避免“贴着边界跳但一直判为收敛”。

### 5.6 状态机

- `min_detect_count=5` 抑制单帧误检；
- `max_temp_lost_count=25` 在约 70 fps 下约 0.36 s，短暂遮挡能靠预测接上；
- `Diverged/NIS` 是强制重建，避免带着错误状态继续自瞄。

---

## 6. 一次完整跟踪的时序（示例）

以一次稳定的四面车跟踪为例：

1. 第 1 帧：检测到一块 Three + Small → `Tracker::Update` 处于 `Lost` → `SetTarget` → PnP 起状态，`state = Detecting`，`updates = 1`。
2. 第 2–5 帧：每帧 `UpdateTarget` → `Predict` → `Associate` 把板配到对应面 → `UpdateImagePoints` 修正。`detect_count` 累加到 5 → `state = Tracking`。
3. 第 6 帧起：`converged = 1`。单板帧 `estimate_shape=false`，r 保持 0.2，l=h=0；位置和角度每帧被修正。
4. 目标转向，出现第二块板：本帧 `matches.size() = 2` → `estimate_shape=true`，两块的弦长把 r 拉向真值；`ApplyLimits` 允许钳制。
5. 短暂遮挡：某帧没有候选 → `UpdateTarget` 仍 `Predict` → `found=false` → `Tracking → TempLost`，`losses++`；后续帧靠预测维持。
6. 25 帧仍无检测 → `TempLost → Lost`；下一帧有检测就 `SetTarget` 重建，`resets` 不增加（这条路径不计 `resets`），`last_reset_reason` 保持上一次的值。
7. NIS 失败率超过 0.8 → `ResetTarget(Nis)`，`resets++`，`last_reset_reason=nis`，状态回 `Lost`。

---

## 7. 已实测的异常链（作为诊断参照）

以下数字来自本工程的实机/仿真日志，用来说明诊断字段怎么读：

```text
cost = 2.351（占上界 4.0 的 58.8%）
nis  = 350.12 / 257.83 / 77.00
r    = 0.191 → 0.500
limit_clamps: 7 → 324（约 24 s）
```

对应链条：

```text
关联允许“每项都没超门”的高代价配对
  → 单板/错误面序下 r 与圆心互相补偿
  → r 被推到 0.500
  → ApplyLimits 每帧钳回
  → Diverged() 的范围判据看不到（钳后仍在合法区间）
  → 连续钳制计数继续涨
  → NIS 失败率超过 0.8
  → ResetTarget(nis)
```

单板冻结 r/l/h 后，这条链的前半段被切断：单板不再更新 r/l/h，`limit_clamps` 也不会在单板帧增长。多板帧才允许形状更新和钳制。

---

## 8. 文件与符号索引

| 文件 | 主要符号 | 职责 |
| --- | --- | --- |
| `framework/algorithm/vision/track/tracker.hpp/cpp` | `Tracker`、`TrackerConfig`、`TrackerReport`、`TrackerState`、`ResetReason` | 每帧调度、状态机、判丢、报告 |
| `framework/algorithm/vision/track/association.hpp/cpp` | `Associate`、`QuadCost`、`AssociationConfig`、`ArmorMatch`、`AssociationResult` | 候选过滤、四门代价、一对一匹配 |
| `framework/algorithm/vision/track/target.hpp/cpp` | `Target`、`TargetConfig`、`FaceState`、`Snapshot` | 11 维 EKF、观测模型、初始化、钳制、四面输出 |
| `framework/algorithm/vision/pose/armor_pose.cpp` | `SolvePose`、`ReprojectArmor`、`OptimizeYaw`、`YpdaMeasurementCovariance` | PnP、重投影、yaw 优化、噪声传播 |
| `framework/algorithm/filter/kalman/ekf.hpp` | `ExtendedKalmanFilter` | 通用 EKF：predict / correct / NIS |
| `ros2_layer/node/vision/vision_node.hpp` | `VisionNode` | 话题收发、TF、颜色过滤、绘制、自瞄、日志 |

---

## 9. 参数与调参入口

| 参数 | 默认 | 作用 | 调大的效果 |
| --- | ---: | --- | --- |
| `center_gate_px` | 100 | 关联中心门 | 更容易匹配，也更容易配错 |
| `corner_gate_px` | 140 | 关联四角门 | 同上 |
| `angle_gate_rad` | 0.8 | 关联方向门 | 容忍更大姿态差 |
| `perimeter_ratio_gate` | 0.6 | 关联周长门 | 容忍更大尺度差 |
| `min_detect_count` | 5 | detecting 升 tracking 的帧数 | 起跟更慢，更抗误检 |
| `max_temp_lost_count` | 25 | temp_lost 回 lost 的帧数 | 遮挡后更容易接回 |
| `nis_failure_ratio` | 0.8 | NIS 判丢比例 | 更容忍模型误差 |
| `point_sigma_px` | 8.0 | 像素观测噪声 | 更不信观测，更平滑 |
| `radius_init_m` | 0.2 | 初始化半径 | 只影响第一帧的外推 |
| `accel_variance` | 10.0 | 位置加速度方差 | 机动跟得上，静止更抖 |
| `angular_accel_variance` | 100.0 | 角加速度方差 | 换面更快，转速更抖 |
| `shape_variance` | 0.1 | r/l/h 过程噪声 | 形状收敛快，但可能被噪声推走 |
| `max_consecutive_limit_clamps` | 5 | 连续钳制判发散 | 更快丢弃贴边状态 |

---

## 10. 边界与已知取舍

1. **只跟踪一辆车**：`Tracker` 只持有一个 `Target`，没有多槽。当前场景下只允许一个编号进入跟踪。
2. **初始化只用一块板**：第一帧即使有多块板，也只取 `nearest` 建目标；其余板从下一帧开始参与关联。
3. **匹配是全枚举**：最多 `5^N` 个叶子（N=4 时 625），当前帧率下开销可接受；参考项目用的是贪心。
4. **钳制只在多板帧发生**：单板帧 r/l/h 冻结，`Diverged()` 的半径范围判据基本不会触发，实际靠连续钳制计数和 NIS 判丢。
5. **TF 缺失时只预测**：不更新状态机，`temp_lost` 计数只在有 TF 的帧推进。
6. **`observed xyz_cam` 与 `face xyz_world` 不可直接相减**：前者是相机系，后者是跟踪系；要比较必须先做 `CameraToFrame`。
