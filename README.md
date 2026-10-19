# ROS 2 视觉节点

相机图像 → TensorRT 装甲板检测 → 整车 EKF 跟踪 → 云台期望角，发布带框图像与 `GimbalCmd`。

- ROS 2 **jazzy**（`/opt/ros/jazzy`）
- 152 行起的 TensorRT 引擎（`0526.engine`），CUDA 11+
- OpenCV 4、Eigen 3

---

## 1. 分层

```text
third_party/               第三方与外部工程（仿真器等），不参与本工程构建
framework/                 纯算法，不链接 ROS
  algorithm/vision/
    detect/                0526 TensorRT 检测器
    pose/                  PnP、坐标变换、板尺寸
    track/                 vehicle_ekf / association / tracker
    aim/                   aim_solver：瞄准点 → 期望角（纯函数）
    refine/ core/          细化与数据类型
  algorithm/filter/         EKF 模板
  algorithm/math/           角度工具

ros2_layer/                ROS 节点类（静态库）
  node/camera/             编码转换与内参转发
  node/vision/
    vision_node            订阅、参数、每帧调度
    aim_publisher          取跟踪结果、查 TF、发布 GimbalCmd
    pose_buffer            按图像时间戳插值的位姿缓冲

project/                   可执行入口
  node/{camera,vision}/    main + spin
  params/                  camera.yaml、vision.yaml
  launch/vision.launch.py
```

依赖方向单向：`vision_node → aim_publisher → aim_solver`，算法层不引 ROS。

---

## 2. 每帧数据流

```text
/image_raw ──> camera 节点 ──> /vision/image (bgr8) ──┐
                              /vision/camera_info ───┤
                                                      ▼
                                              ArmorDetector::Detect
                                                      │  result.armors
                                                      ▼
                                  按 enemy_color 过滤 ──> enemy_armors
                                                      │
/tf ──> pose_buffer（按图像时间戳插值）──> frame_ ────┼──> Tracker::ProcessFrame
                                                      │
                                                      ▼
                                          Faces() / Report().last_face
                                                      │
                                                      ▼
                                        AimPublisher → AimSolver
                                                      │
                              /vision/debug_image ◀──┴──▶ /rm_gimbal/cmd
```

`Tracker::ProcessFrame` 按状态分两路：

- `Lost`：`FindLockArmor` 取离图像中心最近的板 → `LockVehicle` 初始化 EKF，记下 `label_/size_` 锁定这辆车
- 其它状态：`UpdateVehicle` → `Predict(dt)` → `Associate` 四门配对 → 逐对修正 EKF → `SelectAimMatch` 选瞄准面

只在自瞄键按住时跑跟踪与发布：`/simulator/auto_aim`（`std_msgs/Bool`，true = 按住）为假时每帧只调
`Tracker::Reset(Manual)`，不选板、不锁车、不推进 EKF，下次按下重新选板。

`dt` 按“距上一次真正调用 `ProcessFrame`”计算，姿态缺失而整帧跳过的那些帧不计入——
用图像间隔会让丢帧后的预测欠一步，小陀螺下直接掉出关联门。

状态机：`lost → detecting → tracking → temp_lost → lost`，`detecting` 丢一帧即回 `lost`，
`temp_lost` 超过 `max_temp_lost_count` 帧回 `lost`。NIS 失败率超限或半径越界也会判丢重建。

---

## 3. 关键机制

**关联（`Associate`）**：只看同 `label`、同 `size` 的候选板；每块板 × 每个预测面算四门代价——
中心、四角、边方向角、周长比，任一项超门作废；再枚举一对一分配，按“匹配数最多、总代价最小”取。
门限在 `vision.yaml` 里。

**瞄准面选择（`SelectAimMatch`）**：对每对匹配算

```text
score = aim_weight_center · (d/fx)² + aim_weight_cost · cost² + aim_weight_angle · θ²
      + (face != last_face ? aim_switch_cost : 0)
```

`d` 是检测框中心到画面中心的像素距离，除以 `fx` 后与另外两项同量级；`cost` 是四门代价值，
`θ` 是这块板 PnP 法向与视线的夹角。取 `score` 最小的一对写 `last_face`。四项权重都在 YAML 里可调。

`aim_switch_cost` 是换面损失：当前瞄准面仍在本帧匹配里时，别的面要好出一整个罚值才抢得走。
**它只在同一帧有多个候选时起作用**；只剩一对匹配时，该对无条件成为瞄准面并改写 `last_face`。

> 曾经的 bug：`last_face` 在修正循环里被逐对覆写，最终留下的是代价最大的那对；两块板代价接近时
> 绿框和自瞄点会在两块板之间来回跳。现在先选好 `aim_index`，只在遍历到那一对时写入。

**期望角（`SolveAim`）**：

```text
p_map = R_map_camera · p_camera + t_map_camera
d     = p_map − gimbal_in_map
yaw   = atan2(d.y, d.x)
pitch = 90° − atan2(d.z, hypot(d.x, d.y))     # 从竖直轴量起，水平 90°
distance = |p_camera|
```

`distance ≤ aim_distance_max_m` 且 `|pitch − 90°| ≤ aim_fire_angle_deg` 才给 `fire_advice`。

---

## 4. 构建与运行

```bash
# 构建（rm_interfaces 由仿真工程提供，需先 source 它的 install）
cd ~/projects/vscode/ros2/opencv
source /opt/ros/jazzy/setup.bash
source third_party/bevy_robomaster_simulator-master/rm_interfaces/install/setup.bash
colcon build --packages-select framework ros2_layer project --symlink-install

# 仿真（另开终端；默认 feature 是 talos，ros2 要显式开）
# 仿真会读 rm_interfaces，必须先 source 它的 install
cd third_party/bevy_robomaster_simulator-master
source /opt/ros/jazzy/setup.bash
source rm_interfaces/install/setup.bash
cargo run --release --features ros2

# 视觉
source /opt/ros/jazzy/setup.bash
source third_party/bevy_robomaster_simulator-master/rm_interfaces/install/setup.bash
source ~/projects/vscode/ros2/opencv/install/setup.bash
ros2 launch project vision.launch.py
```

`vision.launch.py` 起两个节点（camera 适配 + vision），`use_sim_time` 默认 `false`——仿真器不发 `/clock`。

接实机时 `use_sim_time:=false` 保持；内参必须由相机的 `/camera_info` 提供，否则 `on_image` 每帧跳过
（见 `doc/跟踪与自瞄链路问题总结.md` §9）。

---

## 5. 参数

全部在 `project/params/vision.yaml`，按节点名 `/vision` 分段。分组：

| 组 | 项 |
| --- | --- |
| 话题 | `image_topic`、`camera_info_topic`、`output_image_topic` |
| 检测 | `engine_path`、`cuda_device`、`score_threshold`、`confidence_threshold`、`nms_threshold` |
| 自瞄 | `aim_enabled`、`aim_topic`、`aim_frame`、`camera_frame`、`enemy_color`、`aim_fire_angle_deg`、`aim_distance_max_m`、`aim_fixed_yaw_pitch`、`auto_aim_topic` |
| 位姿 | `tf_topic`、`pose_hold_s` |
| 诊断 | `drop_dump_dir`、`drop_dump_limit` |
| 跟踪 | `armor_num`、`radius_init_m`、`point_sigma_px`、`angle_prior_rad`、`accel_variance`、`angular_accel_variance`、`shape_variance` |
| 状态机 | `min_detect_count`、`max_temp_lost_count`、`nis_failure_ratio` |
| 关联门 | `center_gate_px`、`corner_gate_px`、`angle_gate_rad`、`perimeter_ratio_gate` |
| 瞄准面权重 | `aim_weight_center`、`aim_weight_cost`、`aim_weight_angle`、`aim_switch_cost` |

调参影响的说明写在 YAML 每项上方。

---

## 6. 话题

| 方向 | 话题 | 类型 | 说明 |
| --- | --- | --- | --- |
| 订阅 | `/image_raw` | `sensor_msgs/Image` | 相机/仿真原始图 |
| 订阅 | `/vision/image` | `sensor_msgs/Image` | camera 节点转换后的 bgr8 |
| 订阅 | `/vision/camera_info` | `sensor_msgs/CameraInfo` | 内参来源 |
| 订阅 | `/tf` | `tf2_msgs/TFMessage` | 相机位姿与云台位置 |
| 订阅 | `/simulator/auto_aim` | `std_msgs/Bool` | 仿真发来的自瞄键状态，true = 按住 |
| 发布 | `/vision/debug_image` | `sensor_msgs/Image` | 带实测框与预测四面 |
| 发布 | `/rm_gimbal/cmd` | `rm_interfaces/GimbalCmd` | `yaw/pitch` 世界系绝对角，`distance = -1` 表示无目标 |

---

## 7. 日志与诊断

`log_window` 每秒一组：

- `fps=… accepted=… ms=… age=…`：吞吐、每帧处理耗时、图像到处理的延迟
- `det stats score_pass=… nms=… acc=… | rej conf=… color=… label=… geom=… | max_obj=… min_enemy=…`
  ——逐级检测计数，直接看目标掉在哪一级；`min_enemy` 是本秒敌方检测数的最小值
- 每块检测的 `color/label/size/center/conf`，跟踪状态机，关联候选/匹配数与前 8 条四门分项，
  观测计数、姿态时序偏移、EKF 状态与协方差对角、形状量雅可比

按帧打的两类现场：

- `det drop …`：敌方检测数变少时打逐级计数与灰度 mean/std，并把原图存到 `drop_dump_dir`
  （最多 `drop_dump_limit` 张）
- `face switch A -> B kept=… candidates=… matches=… locked label=… size=…`：瞄准面变化时打，
  随后列出每块检测的 `label/raw_label/size/center/conf` 与前 4 条四门分项。
  据此区分：`kept=1` 是检测少了一块，`candidates=1`（而 `kept` 更多）是编号/尺寸对不上被筛掉，
  `matches=1` 是关联只剩一对。

---

## 8. 相关文档

| 文件 | 内容 |
| --- | --- |
| `doc/远距离漏检排查.md` | 从采集分辨率到选面的逐级核算、判据与验证步骤 |
| `doc/跟踪与自瞄链路问题总结.md` | 链路、已修问题、部署到实车要改的地方 |
| `doc/瞄准解算拆分与期望角度规划.md` | 瞄准解算分层的规划与验收数字 |
| `doc/跟踪器问题与改进规划.md` | EKF / 关联 / 形状估计的问题清单 |
| `doc/跟踪器数据流与EKF.md` | 状态量、观测模型、坐标约定 |
