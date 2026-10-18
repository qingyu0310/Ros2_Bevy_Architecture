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

状态机：`lost → detecting → tracking → temp_lost → lost`，`detecting` 丢一帧即回 `lost`，
`temp_lost` 超过 `max_temp_lost_count` 帧回 `lost`。NIS 失败率超限或半径越界也会判丢重建。

---

## 3. 关键机制

**关联（`Associate`）**：只看同 `label`、同 `size` 的候选板；每块板 × 每个预测面算四门代价——
中心、四角、边方向角、周长比，任一项超门作废；再枚举一对一分配，按“匹配数最多、总代价最小”取。
门限在 `vision.yaml` 里。

**瞄准面选择（`SelectAimMatch`）**：对每对匹配算

```text
score = aim_weight_center · d² + aim_weight_cost · cost² + aim_weight_angle · θ²
```

`d` 是检测框中心到画面中心的像素距离，`cost` 是四门代价值，`θ` 是这块板 PnP 法向与视线的夹角，
取 `score` 最小的一对写 `last_face`。三项权重都在 YAML 里可调。

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

# 仿真（另开终端；仿真默认 feature 是 talos，ros2 要显式开）
cd third_party/bevy_robomaster_simulator-master
cargo run --release --no-default-features --features ros2

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
| 自瞄 | `aim_enabled`、`aim_topic`、`aim_frame`、`camera_frame`、`enemy_color`、`aim_fire_angle_deg`、`aim_distance_max_m`、`aim_fixed_yaw_pitch` |
| 位姿 | `tf_topic`、`pose_hold_s` |
| 跟踪 | `armor_num`、`radius_init_m`、`point_sigma_px`、`angle_prior_rad`、`accel_variance`、`angular_accel_variance`、`shape_variance` |
| 状态机 | `min_detect_count`、`max_temp_lost_count`、`nis_failure_ratio` |
| 关联门 | `center_gate_px`、`corner_gate_px`、`angle_gate_rad`、`perimeter_ratio_gate` |
| 瞄准面权重 | `aim_weight_center`、`aim_weight_cost`、`aim_weight_angle` |

调参影响的说明写在 YAML 每项上方。

---

## 6. 话题

| 方向 | 话题 | 类型 | 说明 |
| --- | --- | --- | --- |
| 订阅 | `/image_raw` | `sensor_msgs/Image` | 相机/仿真原始图 |
| 订阅 | `/vision/image` | `sensor_msgs/Image` | camera 节点转换后的 bgr8 |
| 订阅 | `/vision/camera_info` | `sensor_msgs/CameraInfo` | 内参来源 |
| 订阅 | `/tf` | `tf2_msgs/TFMessage` | 相机位姿与云台位置 |
| 发布 | `/vision/debug_image` | `sensor_msgs/Image` | 带实测框与预测四面 |
| 发布 | `/rm_gimbal/cmd` | `rm_interfaces/GimbalCmd` | `yaw/pitch` 世界系绝对角，`distance = -1` 表示无目标 |

---

## 7. 日志

`log_window` 每秒一组：FPS、检测数、状态机、每帧候选/匹配数、四门分项、观测计数、
姿态时序偏移、EKF 状态与协方差对角、四面世界坐标。

---

## 8. 相关文档

| 文件 | 内容 |
| --- | --- |
| `doc/跟踪与自瞄链路问题总结.md` | 链路、已修问题、部署到实车要改的地方 |
| `doc/瞄准解算拆分与期望角度规划.md` | 瞄准解算分层的规划与验收数字 |
| `doc/跟踪器问题与改进规划.md` | EKF / 关联 / 形状估计的问题清单 |
| `doc/跟踪器数据流与EKF.md` | 状态量、观测模型、坐标约定 |
| `STATUS.md` | 迁移现状与交接说明 |
