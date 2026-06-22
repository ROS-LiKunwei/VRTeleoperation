你现在是一个资深 ROS2 C++ 机器人控制工程师，请帮我把现有遥操作工程中的 FA 上肢 minimum snap 七次多项式轨迹优化逻辑，单独摘出来做成一个独立的 C++ ROS2 功能包。

最终功能包生成路径：

```bash
/home/likunwei/humanoid_ws/src/min_snap
```

功能包名：

```bash
min_snap
```

要求使用 ROS2 Humble，主要实现语言为 C++，绘图脚本可以使用 Python。

---

# 一、任务背景

当前 beavr-bot 遥操作工程中已经有 FA 上肢轨迹优化代码框架，核心逻辑是：

1. 左右臂各 7 个关节分别进行 minimum snap 七次多项式轨迹规划。
2. 轨迹优化发生在目标关节角度输入之后、FA 16 维上肢位置命令发布之前。
3. 输出最终发布到 FA 机器人原生上肢位置控制 topic：

```bash
/upper_position_controller/commands
```

消息类型：

```bash
std_msgs/msg/Float64MultiArray
```

消息长度固定为 16。

我现在希望把 minimum snap 轨迹优化从遥操作系统中解耦出来，做成一个独立 ROS2 C++ 功能包，作为通用的上肢关节轨迹优化节点使用。

---

# 二、FA 上肢 16 维控制命令 ABI

发布 topic：

```bash
/upper_position_controller/commands
```

消息类型：

```cpp
std_msgs::msg::Float64MultiArray
```

长度固定为 16。

关节顺序必须严格如下：

```text
data[0]  = left_shoulder_pitch_joint
data[1]  = left_shoulder_roll_joint
data[2]  = left_shoulder_yaw_joint
data[3]  = left_elbow_joint
data[4]  = left_wrist_yaw_joint
data[5]  = left_wrist_pitch_joint
data[6]  = left_wrist_roll_joint

data[7]  = right_shoulder_pitch_joint
data[8]  = right_shoulder_roll_joint
data[9]  = right_shoulder_yaw_joint
data[10] = right_elbow_joint
data[11] = right_wrist_yaw_joint
data[12] = right_wrist_pitch_joint
data[13] = right_wrist_roll_joint

data[14] = neck_yaw_joint
data[15] = neck_pitch_joint
```

本功能包只对左右臂 14 个关节做 minimum snap 轨迹优化。

neck 两个关节不参与轨迹优化，默认保持当前 `/joint_states` 中的 neck 位置；如果 `/joint_states` 中没有 neck 状态，则 neck 默认保持 0.0 或配置文件中的默认值。

---

# 三、功能包目标

请实现一个独立 C++ ROS2 package：

```bash
min_snap
```

包内至少包含：

```text
min_snap/
├── CMakeLists.txt
├── package.xml
├── include/min_snap/
│   ├── min_snap_trajectory.hpp
│   ├── min_snap_planner.hpp
│   ├── fa_joint_mapping.hpp
│   └── trajectory_recorder.hpp
├── src/
│   ├── min_snap_trajectory.cpp
│   ├── min_snap_planner.cpp
│   ├── min_snap_node.cpp
│   └── trajectory_recorder.cpp
├── config/
│   └── min_snap.yaml
├── scripts/
│   └── plot_min_snap_tracking.py
└── README.md
```

可以根据工程需要增减文件，但整体结构要清晰。

---

# 四、节点设计

请实现 ROS2 节点：

```bash
min_snap_node
```

节点职责：

1. 订阅双臂目标关节角度。
2. 接收期望轨迹执行时间。
3. 接收电机最大速度限制。
4. 接收电机最大加速度限制。
5. 生成满足速度、加速度约束的七次 minimum snap 轨迹。
6. 按固定控制频率采样轨迹。
7. 发布 16 维 FA 上肢控制命令到：

```bash
/upper_position_controller/commands
```

8. 订阅 `/joint_states`，记录实际关节位置、速度和期望轨迹。
9. 提供 CSV 记录和 Python 绘图脚本。

---

# 五、输入接口设计

请为轨迹目标设计一个清晰的 ROS2 输入接口。

优先建议使用自定义 message，例如：

```text
MinSnapTarget.msg
```

字段建议：

```text
float64[] left_arm_target_rad
float64[] right_arm_target_rad
float64 expected_duration_s
float64 max_velocity_rad_s
float64 max_acceleration_rad_s2
```

要求：

```text
left_arm_target_rad  长度必须为 7
right_arm_target_rad 长度必须为 7
expected_duration_s > 0
max_velocity_rad_s > 0
max_acceleration_rad_s2 > 0
```

如果你认为使用 `std_msgs/msg/Float64MultiArray` 更简单，也可以实现，但必须在 README 中明确输入数组格式。

推荐输入 topic：

```bash
/min_snap/target
```

推荐消息格式：

```text
left_arm_target_rad[0:7]
right_arm_target_rad[0:7]
expected_duration_s
max_velocity_rad_s
max_acceleration_rad_s2
```

---

# 六、输出接口设计

轨迹采样后发布到：

```bash
/upper_position_controller/commands
```

类型：

```cpp
std_msgs::msg::Float64MultiArray
```

长度固定为 16。

发布频率通过参数配置：

```yaml
publish_hz: 200.0
```

注意：

1. 输出必须始终是 16 维。
2. 左臂填充 data[0] ~ data[6]。
3. 右臂填充 data[7] ~ data[13]。
4. neck 填充 data[14] ~ data[15]。
5. 不允许发布长度错误的数组。
6. 如果还没收到 `/joint_states`，允许等待实际关节状态后再开始规划，避免起点不可信。

---

# 七、minimum snap 七次多项式轨迹要求

每个关节轨迹形式为：

```text
q(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5 + a6 t^6 + a7 t^7
```

需要支持直接计算：

```text
position     q(t)
velocity     q_dot(t)
acceleration q_ddot(t)
jerk         q_dddot(t)
```

边界条件：

轨迹起点 t = 0：

```text
q(0)       = 当前起点位置
q_dot(0)   = 当前起点速度
q_ddot(0)  = 当前起点加速度
q_jerk(0)  = 0
```

轨迹终点 t = T：

```text
q(T)       = 目标关节位置
q_dot(T)   = 目标速度，默认为 0
q_ddot(T)  = 目标加速度，默认为 0
q_jerk(T)  = 0
```

实现细节：

```text
a0 = q0
a1 = v0
a2 = 0.5 * acc0
a3 = 0
```

`a4 ~ a7` 通过 4x4 线性方程求解，使终点位置、速度、加速度、jerk 满足约束。

要求：

1. 每个关节独立求解七次多项式。
2. 左右臂可以共用一套 planner 类，但状态必须分开管理。
3. planner 必须能保存当前轨迹段的起点状态、终点状态、开始时间、持续时间和多项式系数。
4. planner 必须支持任意时刻采样 position、velocity、acceleration、jerk。

---

# 八、速度和加速度硬约束逻辑

这是最核心的逻辑，必须严格实现。

输入包括：

```text
expected_duration_s
max_velocity_rad_s
max_acceleration_rad_s2
```

但是不能盲目使用 expected_duration_s。

必须检查规划出来的轨迹是否满足：

```text
max(|q_dot(t)|)  <= max_velocity_rad_s
max(|q_ddot(t)|) <= max_acceleration_rad_s2
```

如果不能满足，则必须自动增大轨迹执行时间 T，直到满足最大速度和最大加速度约束。

初始 T：

```text
T = max(expected_duration_s, min_duration_s)
```

建议保留原有经验计算逻辑：

```text
T >= abs(delta) * 2.1875 / max_velocity_rad_s
T >= sqrt(abs(delta) * 7.513188404399293 / max_acceleration_rad_s2)
```

其中：

```text
delta = goal_position - start_position
```

对 14 个关节全部计算，最终 T 取所有关节约束中的最大值。

也可以在生成轨迹后通过密集采样校验峰值速度和峰值加速度，例如每段采样 200 ~ 1000 个点。如果采样发现超过限制，则继续增大 T，例如：

```text
T = T * 1.1
```

直到满足约束，或者达到最大迭代次数。

必须打印 warning，例如：

```text
[WARN] Requested duration 0.300 s violates velocity/acceleration limits. Extended trajectory duration to 0.842 s.
```

要求：

1. 速度和加速度限制必须作用在最终生成的轨迹上。
2. 不能只依赖发布端限幅器来兜底。
3. 如果 expected_duration_s 太短，必须自动延长。
4. 如果用户给的 max_velocity_rad_s 或 max_acceleration_rad_s2 非法，必须拒绝该目标并打印错误。
5. README 中必须明确说明：最终执行时间可能大于用户输入的 expected_duration_s，因为需要满足电机速度和加速度约束。

---

# 九、在线重规划逻辑

这是第二个核心逻辑，必须严格实现。

当轨迹正在执行过程中，如果 `/min_snap/target` 收到新的目标关节角度，不能等当前轨迹执行完再规划。

必须立即进行重规划。

重规划起点必须使用当前轨迹正在执行的状态，而不是简单使用最新 `/joint_states`：

```text
start_position     = 当前轨迹采样位置
start_velocity     = 当前轨迹采样速度
start_acceleration = 当前轨迹采样加速度
start_jerk         = 0
```

重规划终点：

```text
goal_position     = 新收到的目标关节角度
goal_velocity     = 0
goal_acceleration = 0
goal_jerk         = 0
```

重规划后要求：

1. 位置连续。
2. 速度连续。
3. 加速度连续。
4. jerk 可以不跨段连续，起点 jerk 固定为 0 即可。
5. 重新计算满足速度和加速度约束的轨迹时长。
6. 打印重规划日志，例如：

```text
[INFO] Replanned min-snap trajectory from active trajectory state.
```

如果当前没有正在执行的轨迹，则以 `/joint_states` 中当前关节位置作为起点，速度优先使用 `/joint_states.velocity`，如果没有速度则默认为 0，加速度默认为 0。

---

# 十、/joint_states 订阅和状态管理

节点需要订阅：

```bash
/joint_states
```

类型：

```cpp
sensor_msgs::msg::JointState
```

要求：

1. 根据 joint name 映射实际关节位置和速度。
2. 支持 14 个手臂关节。
3. 支持 neck 两个关节。
4. 如果 `/joint_states` 缺失某个关节，需要打印 warning，但不要崩溃。
5. 如果长期没有收到 `/joint_states`，不要发布危险命令。
6. 提供参数：

```yaml
joint_state_timeout_s: 0.5
require_joint_state_before_start: true
```

如果 `require_joint_state_before_start=true`，在收到有效 joint_states 前不能开始轨迹规划。

---

# 十一、记录功能

请移植并保留现有的 `/joint_states` 和优化轨迹记录功能。

记录开关通过参数控制：

```yaml
record_tracking: true
tracking_output_dir: "/home/likunwei/humanoid_ws/src/min_snap/logs"
```

CSV 文件名建议：

```text
min_snap_tracking_YYYYMMDD_HHMMSS.csv
```

CSV 字段至少包括：

```text
time_s

joint_states.position.<joint_name>
joint_states.velocity.<joint_name>

desired.position.<joint_name>
desired.velocity.<joint_name>
desired.acceleration.<joint_name>
desired.jerk.<joint_name>

error.position.<joint_name>
```

其中：

```text
error.position.<joint_name> = joint_states.position.<joint_name> - desired.position.<joint_name>
```

要求：

1. 每次发布命令时记录一行。
2. desired 数据来自 minimum snap planner 当前采样值。
3. actual 数据来自 `/joint_states`。
4. 如果 `/joint_states` 中某个关节不存在，对应字段留空或写 NaN。
5. neck 的 desired velocity、acceleration、jerk 可以记录为 0。
6. 程序退出时正常关闭 CSV 文件。

---

# 十二、Python 绘图脚本

请实现：

```bash
scripts/plot_min_snap_tracking.py
```

功能：

```bash
python3 scripts/plot_min_snap_tracking.py <csv_file>
```

默认输出：

```text
*_left_tracking.png
*_right_tracking.png
*_neck_tracking.png
```

每张图包含 4 行：

1. `joint_states.position` 和 `desired.position`
2. `desired.velocity`
3. `desired.acceleration`
4. `desired.jerk`

要求支持参数：

```bash
--groups left
--groups right
--groups neck
--groups left right
```

绘图建议使用：

```python
matplotlib
pandas
```

要求：

1. 自动识别 CSV 字段。
2. 如果某些字段不存在，跳过并打印 warning。
3. 图片保存到 CSV 同目录。
4. README 中给出绘图命令示例。

---

# 十三、参数文件

请提供：

```bash
config/min_snap.yaml
```

参数至少包括：

```yaml
min_snap_node:
  ros__parameters:
    publish_hz: 200.0

    target_topic: "/min_snap/target"
    command_topic: "/upper_position_controller/commands"
    joint_states_topic: "/joint_states"

    min_duration_s: 0.01
    default_expected_duration_s: 0.5
    default_max_velocity_rad_s: 0.25
    default_max_acceleration_rad_s2: 0.25

    duration_scale_on_violation: 1.1
    max_duration_search_iterations: 50
    constraint_sample_count: 500

    replan_threshold_rad: 0.0005

    require_joint_state_before_start: true
    joint_state_timeout_s: 0.5

    record_tracking: true
    tracking_output_dir: "/home/likunwei/humanoid_ws/src/min_snap/logs"

    neck_default_position:
      - 0.0
      - 0.0
```

---

# 十四、类设计建议

请尽量按下面结构实现。

## 1. MinSnapTrajectory

职责：

1. 保存单个关节七次多项式系数。
2. 根据时间 t 采样 position、velocity、acceleration、jerk。
3. 提供峰值速度和峰值加速度采样检查。

建议接口：

```cpp
struct TrajectoryState {
  double position;
  double velocity;
  double acceleration;
  double jerk;
};

class MinSnapTrajectory {
public:
  bool solve(
    double q0,
    double v0,
    double a0,
    double q1,
    double v1,
    double a1,
    double duration);

  TrajectoryState sample(double t) const;

  double duration() const;
};
```

## 2. MinSnapArmPlanner

职责：

1. 管理 7 个关节的 trajectory。
2. 接收 start state 和 goal state。
3. 自动扩展 duration 以满足速度和加速度限制。
4. 支持在线重规划。

建议接口：

```cpp
class MinSnapArmPlanner {
public:
  bool plan(
    const std::array<double, 7>& start_pos,
    const std::array<double, 7>& start_vel,
    const std::array<double, 7>& start_acc,
    const std::array<double, 7>& goal_pos,
    double expected_duration,
    double max_velocity,
    double max_acceleration);

  ArmTrajectorySample sample(double elapsed_time) const;

  bool active() const;
  bool finished(double elapsed_time) const;
};
```

## 3. MinSnapNode

职责：

1. ROS2 参数读取。
2. 订阅 target。
3. 订阅 joint_states。
4. 管理左右臂 planner。
5. 定时器采样并发布 16 维命令。
6. 在线重规划。
7. 调用 recorder 记录 CSV。

## 4. TrajectoryRecorder

职责：

1. 创建 CSV。
2. 写 header。
3. 每周期写入 actual、desired、error。
4. 节点退出时关闭文件。

---

# 十五、安全和鲁棒性要求

必须处理以下情况：

1. 输入目标数组长度不为 7：拒绝并打印错误。
2. 输入目标中包含 NaN 或 inf：拒绝并打印错误。
3. expected_duration_s <= 0：使用默认值或拒绝，并打印 warning。
4. max_velocity_rad_s <= 0：拒绝。
5. max_acceleration_rad_s2 <= 0：拒绝。
6. 没有 `/joint_states`：不发布新轨迹命令。
7. `/joint_states` 超时：暂停发布或保持最后安全命令，并打印 warning。
8. 轨迹执行中收到新目标：立即重规划。
9. 目标变化小于 `replan_threshold_rad`：可以忽略，避免高频微小目标导致频繁重规划。
10. 输出到 `/upper_position_controller/commands` 的数组长度必须始终为 16。

---

# 十六、构建和运行方式

请确保可以在 ROS2 Humble 下构建：

```bash
cd /home/likunwei/humanoid_ws
colcon build --packages-select min_snap --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

运行：

```bash
ros2 run min_snap min_snap_node --ros-args --params-file /home/likunwei/humanoid_ws/src/min_snap/config/min_snap.yaml
```

发送测试目标：

如果使用自定义 msg，请在 README 中提供示例命令。

如果使用 Float64MultiArray 输入，请提供类似：

```bash
ros2 topic pub /min_snap/target std_msgs/msg/Float64MultiArray "data: [...]"
```

但更推荐自定义 msg，因为需要同时传目标角度、期望执行时间、最大速度、最大加速度。

---

# 十七、测试要求

请至少提供以下验证方式：

1. 编译通过：

```bash
colcon build --packages-select min_snap --cmake-args -DCMAKE_BUILD_TYPE=Release
```

2. 节点能正常启动。

3. 输入目标后，节点能发布 16 维命令：

```bash
ros2 topic echo /upper_position_controller/commands
```

4. 轨迹速度不超过输入的 `max_velocity_rad_s`。

5. 轨迹加速度不超过输入的 `max_acceleration_rad_s2`。

6. 如果 expected_duration_s 太短，程序会自动延长 duration 并打印 warning。

7. 轨迹执行过程中再次发送目标，程序会从当前轨迹 position、velocity、acceleration 进行重规划。

8. CSV 文件可以正常生成。

9. Python 脚本可以读取 CSV 并输出 tracking 图。

---

# 十八、README 内容要求

请在 README.md 中写清楚：

1. 功能包用途。
2. 订阅 topic。
3. 发布 topic。
4. FA 16 维命令顺序。
5. 输入目标格式。
6. minimum snap 七次多项式边界条件。
7. 为什么 expected_duration_s 可能会被自动增大。
8. 在线重规划逻辑。
9. `/joint_states` 记录字段说明。
10. 绘图脚本使用方法。
11. 编译和运行命令。
12. 常见问题排查。

---

# 十九、重要实现原则

请严格遵守：

1. 不要再依赖 beavr-bot 的 Python 代码。
2. 不要把 IK、VR、遥操作逻辑带进这个功能包。
3. 这个功能包只处理“关节目标角度 → min-snap 轨迹 → 16 维 FA 上肢命令”。
4. 速度、加速度约束必须在轨迹规划阶段满足，而不是靠发布端限幅器兜底。
5. 在线重规划必须从当前轨迹状态出发，不能简单从新 `/joint_states` 位置重启，否则会造成速度、加速度不连续。
6. C++ 代码要有清晰类结构，避免所有逻辑堆在一个 node 文件里。
7. 所有 topic、频率、限制参数都要可配置。
8. 代码生成完成后，请给出最终文件列表、关键实现说明和运行命令。
