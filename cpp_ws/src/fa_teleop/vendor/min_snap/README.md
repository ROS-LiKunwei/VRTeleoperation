# min_snap

`min_snap` 是一个独立 ROS2 Humble C++ 功能包，用于把 FA 上肢左右臂关节目标转换成满足速度、加速度约束的七次 minimum snap 轨迹，并发布 FA 原生 16 维上肢位置命令。

本包不包含 IK、VR 手势解析或 beavr-bot 内部逻辑。它只处理：

```text
左臂 / 右臂 / 头部关节目标 -> minimum snap 轨迹 -> 16 维 FA 上肢位置命令
```

## Topic

订阅：

- `/min_snap/target`
  - 类型：`min_snap/msg/MinSnapTarget`
  - 遥操作或测试程序发布左臂、右臂、头部目标关节角。
- `/joint_states`
  - 类型：`sensor_msgs/msg/JointState`
  - 用于获取起点状态、neck 位置和 CSV 中的实际轨迹。

发布：

- `/upper_position_controller/commands`
  - 类型：`std_msgs/msg/Float64MultiArray`
  - 长度固定 16，直接对接 FA 上肢位置控制器。
- `/min_snap/desired_joint_states`
  - 类型：`sensor_msgs/msg/JointState`
  - RViz 可订阅该 topic 查看 minimum snap 期望姿态。

## FA 16 维命令顺序

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

minimum snap 作用于左右臂 14 个关节，以及可选的 2 个 neck 关节。左臂、右臂或 neck 任一目标数组为空时，该组关节保持 `/joint_states` 中的当前值；如果 `/joint_states` 不包含 neck，则 neck 使用 `neck_default_position`。

## 输入消息

`msg/MinSnapTarget.msg`：

```text
float64[] left_arm_target_rad
float64[] right_arm_target_rad
float64[] neck_target_rad
float64 expected_duration_s
float64 max_velocity_rad_s
float64 max_acceleration_rad_s2
```

要求：

- `left_arm_target_rad` 可以为空，或提供 7 个目标值。
- `right_arm_target_rad` 可以为空，或提供 7 个目标值。
- `neck_target_rad` 可以为空，或按 `[neck_yaw_joint, neck_pitch_joint]` 提供 2 个目标值。
- 目标数组不能包含 NaN 或 Inf。
- `max_velocity_rad_s > 0`。
- `max_acceleration_rad_s2 > 0`。
- `expected_duration_s <= 0` 时使用参数 `default_expected_duration_s`，并打印 warning。

## Minimum Snap 边界条件

每个关节独立求解七次多项式：

```text
q(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5 + a6 t^6 + a7 t^7
```

起点：

```text
q(0)       = 当前轨迹位置或 /joint_states 位置
q_dot(0)   = 当前轨迹速度或 /joint_states.velocity
q_ddot(0)  = 当前轨迹加速度，新轨迹首次规划时为 0
q_jerk(0)  = 0
```

终点：

```text
q(T)       = 目标关节角
q_dot(T)   = 0
q_ddot(T)  = 0
q_jerk(T)  = 0
```

实现中固定：

```text
a0 = q0
a1 = v0
a2 = 0.5 * a0_acc
a3 = 0
```

然后求解 `a4..a7`。

## 执行时间为什么会变长

输入的 `expected_duration_s` 是期望值，不是硬约束。节点会保证最终轨迹满足：

```text
max(|q_dot(t)|)  <= max_velocity_rad_s
max(|q_ddot(t)|) <= max_acceleration_rad_s2
```

初始 duration 会先套用经验约束：

```text
T >= abs(delta) * 2.1875 / max_velocity_rad_s
T >= sqrt(abs(delta) * 7.513188404399293 / max_acceleration_rad_s2)
```

之后再对轨迹做密集采样检查。如果超限，会按 `duration_scale_on_violation` 继续放大 duration，直到满足约束或达到最大迭代次数。

因此最终执行时间可能大于用户输入的 `expected_duration_s`。这是为了避免把过短、过激的轨迹直接下发给电机。

## 在线重规划

轨迹执行过程中收到新的 `/min_snap/target` 时，节点会立即重规划。

重规划起点来自当前正在执行的轨迹采样状态：

```text
start_position     = 当前轨迹 position
start_velocity     = 当前轨迹 velocity
start_acceleration = 当前轨迹 acceleration
start_jerk         = 0
```

这样可以保持位置、速度、加速度连续。jerk 不保证跨段连续。

如果当前没有正在执行的轨迹，则从 `/joint_states` 读取起点位置和速度；如果 velocity 字段缺失，则速度按 0 处理。

## 最小使用 Demo

这个 demo 使用 `fa_rviz_command_bridge` 把 `/upper_position_controller/commands` 转成 `/joint_states` 并在 RViz 中显示；`min_snap_node` 负责接收 `/min_snap/target` 并发布 16 维上肢命令。

### 1. 编译

`min_snap`：

```bash
cd /home/likunwei/humanoid_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select min_snap --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

RViz command bridge：

```bash
cd /home/likunwei/dataCollection/beavr-bot
source /opt/ros/humble/setup.bash
colcon build \
  --base-paths robots/fa_description robots/fa_rviz_command_bridge \
  --packages-select fa_description fa_rviz_command_bridge
source install/setup.bash
```

### 2. 启动 RViz Bridge

开一个终端：

```bash
cd /home/likunwei/dataCollection/beavr-bot
source install/setup.bash
ros2 launch fa_rviz_command_bridge fa_command_rviz.launch.py
```

该 bridge 默认发布 `/joint_states` 到 1 kHz，包括静止时的 idle 状态。确认频率：

```bash
ros2 topic hz /joint_states
```

### 3. 启动 min_snap

再开一个终端：

```bash
cd /home/likunwei/humanoid_ws
source install/setup.bash
ros2 launch min_snap min_snap.launch.py
```

确认接口存在：

```bash
ros2 topic info /min_snap/target
ros2 service list | grep min_snap
```

### 4. 发布最小目标

只发左臂目标，右臂和头部保持当前 `/joint_states`：

```bash
ros2 topic pub --once /min_snap/target min_snap/msg/MinSnapTarget "{
  left_arm_target_rad: [0.30, -0.20, 0.15, -0.40, 0.12, -0.08, 0.10],
  right_arm_target_rad: [],
  neck_target_rad: [],
  expected_duration_s: 1.0,
  max_velocity_rad_s: 0.8,
  max_acceleration_rad_s2: 4.0
}"
```

只发右臂目标，左臂和头部保持当前 `/joint_states`：

```bash
ros2 topic pub --once /min_snap/target min_snap/msg/MinSnapTarget "{
  left_arm_target_rad: [],
  right_arm_target_rad: [-0.30, 0.20, -0.15, -0.40, -0.12, 0.08, -0.10],
  neck_target_rad: [],
  expected_duration_s: 1.0,
  max_velocity_rad_s: 0.8,
  max_acceleration_rad_s2: 4.0
}"
```

只发头部目标，左右臂保持当前 `/joint_states`：

```bash
ros2 topic pub --once /min_snap/target min_snap/msg/MinSnapTarget "{
  left_arm_target_rad: [],
  right_arm_target_rad: [],
  neck_target_rad: [0.20, -0.10],
  expected_duration_s: 1.0,
  max_velocity_rad_s: 0.8,
  max_acceleration_rad_s2: 4.0
}"
```

同时发左臂、右臂和头部目标：

```bash
ros2 topic pub --once /min_snap/target min_snap/msg/MinSnapTarget "{
  left_arm_target_rad: [0.30, -0.20, 0.15, -0.40, 0.12, -0.08, 0.10],
  right_arm_target_rad: [-0.30, 0.20, -0.15, -0.40, -0.12, 0.08, -0.10],
  neck_target_rad: [0.20, -0.10],
  expected_duration_s: 1.0,
  max_velocity_rad_s: 0.8,
  max_acceleration_rad_s2: 4.0
}"
```

检查输出：

```bash
ros2 topic echo /upper_position_controller/commands
ros2 topic echo /min_snap/desired_joint_states
```

### 5. 暂停和恢复

暂停会停用当前 active trajectory，并停止继续发布轨迹点：

```bash
ros2 service call /min_snap/pause_trajectory_publish std_srvs/srv/Trigger "{}"
```

恢复只解除“暂停发布”状态，不会继续执行已经被 pause 清掉的旧轨迹。恢复后机器人应保持不动，直到收到新的 `/min_snap/target`：

```bash
ros2 service call /min_snap/resume_trajectory_publish std_srvs/srv/Trigger "{}"
```

### 6. 在线自动测试

在 RViz bridge 和 `min_snap_node` 都已启动后，可以运行 9 个组合场景测试：

```bash
cd /home/likunwei/humanoid_ws
source install/setup.bash
/usr/bin/python3 install/min_snap/lib/min_snap/test_min_snap_online_scenarios.py --no-start-min-snap --manual
```

如果没有单独启动 `min_snap_node`，让测试脚本自己启动：

```bash
/usr/bin/python3 install/min_snap/lib/min_snap/test_min_snap_online_scenarios.py --manual
```

目标幅度可用 `--amplitude-scale` 调整：

```bash
/usr/bin/python3 install/min_snap/lib/min_snap/test_min_snap_online_scenarios.py \
  --no-start-min-snap --manual --amplitude-scale 1.5
```

测试结果会保存到：

```bash
/home/likunwei/humanoid_ws/src/min_snap/logs/online_scenarios
```

## 参数

默认参数文件：

```bash
/home/likunwei/humanoid_ws/src/min_snap/config/min_snap.yaml
```

关键参数：

```yaml
publish_hz: 1000.0
target_topic: "/min_snap/target"
command_topic: "/upper_position_controller/commands"
desired_joint_states_topic: "/min_snap/desired_joint_states"
joint_states_topic: "/joint_states"
pause_publish_service: "/min_snap/pause_trajectory_publish"
resume_publish_service: "/min_snap/resume_trajectory_publish"

min_duration_s: 0.01
default_expected_duration_s: 0.1
default_max_velocity_rad_s: 1.2
default_max_acceleration_rad_s2: 8.0

duration_scale_on_violation: 1.12
max_duration_search_iterations: 10
constraint_sample_count: 200

replan_threshold_rad: 0.001
require_joint_state_before_start: true
joint_state_timeout_s: 0.1

record_tracking: false
tracking_output_dir: "/home/likunwei/humanoid_ws/src/min_snap/logs"
record_run_log: false
run_log_output_dir: "/home/likunwei/humanoid_ws/src/min_snap/Log"
neck_default_position: [0.0, 0.0]
```

默认 launch 会把该参数文件传给 `min_snap_node`。如需换配置：

```bash
ros2 launch min_snap min_snap.launch.py \
  params_file:=/path/to/min_snap.yaml
```

## C++ 使用示例

其他 ROS2 C++ 节点可以直接发布 `min_snap/msg/MinSnapTarget` 到 `/min_snap/target`。下面是一个最小 publisher 示例：

```cpp
#include <chrono>
#include <memory>

#include "min_snap/msg/min_snap_target.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class MinSnapTargetPublisher : public rclcpp::Node
{
public:
  MinSnapTargetPublisher()
  : Node("min_snap_target_publisher")
  {
    publisher_ = create_publisher<min_snap::msg::MinSnapTarget>("/min_snap/target", 10);
    timer_ = create_wall_timer(500ms, [this]() { publish_once(); });
  }

private:
  void publish_once()
  {
    min_snap::msg::MinSnapTarget msg;
    msg.left_arm_target_rad = {0.30, -0.20, 0.15, -0.40, 0.12, -0.08, 0.10};
    msg.right_arm_target_rad = {};
    msg.neck_target_rad = {};
    msg.expected_duration_s = 1.0;
    msg.max_velocity_rad_s = 0.8;
    msg.max_acceleration_rad_s2 = 4.0;

    publisher_->publish(msg);
    RCLCPP_INFO(get_logger(), "Published one min_snap target");
    timer_->cancel();
  }

  rclcpp::Publisher<min_snap::msg::MinSnapTarget>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinSnapTargetPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

如果这个示例放在另一个 ROS2 包中，`package.xml` 至少需要依赖：

```xml
<depend>rclcpp</depend>
<depend>min_snap</depend>
```

对应的 `CMakeLists.txt` 可写成：

```cmake
find_package(rclcpp REQUIRED)
find_package(min_snap REQUIRED)

add_executable(min_snap_target_publisher src/min_snap_target_publisher.cpp)
ament_target_dependencies(min_snap_target_publisher rclcpp min_snap)

install(TARGETS min_snap_target_publisher
  DESTINATION lib/${PROJECT_NAME})
```

运行前先启动 `min_snap_node`，并确保 `/joint_states` 中存在目标机器人所需的手臂关节状态。

## 与 VR 遥操作通信

VR 遥操作侧应继续负责手势输入、坐标变换和 IK。IK 产生左右臂 7 维关节目标后，发布到：

```bash
/min_snap/target
```

本节点接收该关节目标并输出 `/upper_position_controller/commands`。这样轨迹优化从遥操作主进程中解耦，VR 侧只关心目标关节角，轨迹节点负责 minimum snap、速度/加速度约束和实际下发。

## RViz

节点发布：

```bash
/min_snap/desired_joint_states
```

如果只想查看期望姿态，可以让 RViz/robot_state_publisher 使用该 topic，或用 topic remap 把它映射为 `/joint_states`。真实硬件运行时不要把期望状态覆盖真实 `/joint_states`。

## CSV 记录

开启参数：

```yaml
record_tracking: true
tracking_output_dir: "/home/likunwei/humanoid_ws/src/min_snap/logs"
```

CSV 文件名：

```text
min_snap_tracking_YYYYMMDD_HHMMSS.csv
```

字段：

```text
time_s
joint_states.position.<joint>
joint_states.velocity.<joint>
desired.position.<joint>
desired.velocity.<joint>
desired.acceleration.<joint>
desired.jerk.<joint>
error.position.<joint>
```

`error.position = joint_states.position - desired.position`。

## 绘图

```bash
python3 /home/likunwei/humanoid_ws/src/min_snap/scripts/plot_min_snap_tracking.py \
  /home/likunwei/humanoid_ws/src/min_snap/logs/min_snap_tracking_YYYYMMDD_HHMMSS.csv
```

只画左臂：

```bash
python3 scripts/plot_min_snap_tracking.py <csv_file> --groups left
```

只画右臂：

```bash
python3 scripts/plot_min_snap_tracking.py <csv_file> --groups right
```

输出：

```text
*_left_tracking.png
*_right_tracking.png
*_neck_tracking.png
```

## 常见问题

节点收到目标但不发布命令：

- 检查 `/joint_states` 是否存在 14 个 FA 手臂关节。
- 检查 `require_joint_state_before_start` 是否为 `true`。
- 检查 `/joint_states` 是否超过 `joint_state_timeout_s`。

目标被拒绝：

- 检查左右目标数组长度是否都是 7。
- 检查目标、速度、加速度中是否有 NaN/Inf。
- `max_velocity_rad_s` 和 `max_acceleration_rad_s2` 必须大于 0。

duration 比输入更长：

- 这是预期行为，说明输入 duration 太短，无法满足速度/加速度限制。

RViz 中看不到期望姿态：

- 检查 `/min_snap/desired_joint_states` 是否有数据。
- 确认 robot_state_publisher/RViz 使用同一套 URDF 关节名。
