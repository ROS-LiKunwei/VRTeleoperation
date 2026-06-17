你是一个机器人遥操作系统开发 Agent。请基于当前 BeaVR / beavr-bot 工程，把现有 SYSMO-32 双臂遥操作架构适配到 FA 机器人。

## 一、当前 BeaVR 遥操作架构背景

当前工程中已经存在 SYSMO-32 的实机遥操作链路，整体数据流为：

```text
PICO4 Unity
  -> PICO4 detector
  -> keypoint_transform left/right
  -> Sysmo32Operator left/right
  -> CartesianTarget via ZMQ endeff_coords
  -> one bimanual Sysmo32RealControl
  -> /joint_states feedback
  -> IK + smoothing + safety limiting
  -> arm command
  -> ROS2 position command topic
  -> optional MuJoCo mirror
```

当前 SYSMO-32 实机控制层只启动一个双臂控制器，不是左右臂各一个控制器。它同时订阅左右手的 CartesianTarget、reset、pause/resume、transformed hand coords，并从 `/joint_states` 获取当前双臂关节反馈。

当前默认 arm smoother 是：

```text
jerk_limited_servo
```

用于连续追踪 VR 流式目标；也保留：

```text
min_snap
none
```

## 二、本次目标

新增 FA 机器人遥操作适配，使 FA 可以复用现有 VR 遥操作链路：

```text
PICO4 Unity
  -> PICO4 detector
  -> keypoint_transform left/right
  -> FaOperator left/right
  -> CartesianTarget via ZMQ endeff_coords
  -> one bimanual FaRealControl
  -> /joint_states feedback
  -> FA C++ IK
  -> jerk-limited smoothing
  -> command limiting
  -> 16D Float64MultiArray
  -> /upper_position_controller/commands
```

FA 机器人模型文件路径为：

```text
/home/likunwei/dataCollection/beavr-bot/robots/fa_description
```

FA 上肢原生位置控制 topic 为：

```text
/upper_position_controller/commands
```

消息类型：

```text
std_msgs/msg/Float64MultiArray
```

含义：

```text
一次发送 16 个上肢目标关节位置，单位 rad
```

16 维关节顺序来自 `controllers.yaml`：

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

注意：本阶段不要再复用 SYSMO-32 的 18 维 `/sysmo_left_arm_controller/commands` 接口。FA 应该直接发布到 `/upper_position_controller/commands`。

## 三、FA C++ 逆解优先级要求

FA 已经有可用的 C++ 逆解功能包 `ik_7dof`，请优先复用它，不要重新从零实现 Python IK。

README 中给出的核心接口如下：

```cpp
#include "ik_7dof/fa_ik_solver.hpp"

using namespace fa_arm_kinematic;

IKSolver solver(urdf_file, srdf_file);

ArmKinematicsOptions options;
options.reference_frame = ArmReferenceFrame::PELVIS;

Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
PoseSE3 fk = solver.computeArmFK_SE3(q, ArmSide::LEFT, options);

pinocchio::SE3 target;
target.translation(fk.p);
target.rotation(fk.R);

IKResult result = solver.solveArmIK(
    target, ArmSide::LEFT, Eigen::VectorXd(), options, 1000, 1e-3);
```

`fa_ik_solver` 支持两种参考坐标系：

```text
reference_frame:=pelvis
reference_frame:=arm_base
```

默认使用：

```text
pelvis
```

适配时优先使用：

```text
ArmReferenceFrame::PELVIS
```

因为 BeaVR 中 `CartesianTarget(frame_id="base")` 更接近全身/躯干统一坐标系。若 FA 的 `base` 与 `pelvis` 不一致，请增加明确的外参变换配置，而不要在代码里硬编码猜测。

## 四、实现方案要求

### 1. 新增 FA robot config

请参考现有 `sysmo32_config.py`、`sysmo_mujoco_config.py` 或类似配置文件，新增：

```text
beavr/teleop/configs/robots/fa_config.py
```

配置内容至少包括：

```text
robot_name = "fa"
laterality = left | right | bimanual
control_backend = mujoco | real | real_with_mujoco

model_root = /home/likunwei/dataCollection/beavr-bot/robots/fa_description
urdf_file = 待从 fa_description 中确认
srdf_file = 待从 fa_description 或 moveit config 中确认

left_arm_joint_names:
  - left_shoulder_pitch_joint
  - left_shoulder_roll_joint
  - left_shoulder_yaw_joint
  - left_elbow_joint
  - left_wrist_yaw_joint
  - left_wrist_pitch_joint
  - left_wrist_roll_joint

right_arm_joint_names:
  - right_shoulder_pitch_joint
  - right_shoulder_roll_joint
  - right_shoulder_yaw_joint
  - right_elbow_joint
  - right_wrist_yaw_joint
  - right_wrist_pitch_joint
  - right_wrist_roll_joint

neck_joint_names:
  - neck_yaw_joint
  - neck_pitch_joint

upper_position_command_topic = /upper_position_controller/commands
upper_position_command_type = std_msgs/msg/Float64MultiArray
upper_position_command_size = 16

left_ee_frame = 待确认
right_ee_frame = 待确认

H_R_V_FA = FA 专用 VR 到 robot base/pelvis 的坐标变换矩阵

joint_limits
velocity_limits
acceleration_limits
jerk_limits
home_joint_positions
ready_joint_positions
neck_default_positions
```

如果 URDF、SRDF、末端 frame、joint limit 无法自动确定，请不要瞎填。代码中使用 TODO、配置项或显式报错，并在最终报告中列出需要人工确认的字段。

### 2. 新增 FA Operator

请参考 `Sysmo32Operator`，新增：

```text
FaOperator
```

目标：

```text
继承现有 XArmOperator 或通用 Operator 重定向逻辑
只替换 FA 专用的 H_R_V_FA 坐标变换矩阵
reset 时记录机器人当前末端位姿和当前 VR 手部基准位姿
正常运行时计算手相对初始帧的运动
映射到 FA pelvis/base frame
生成 CartesianTarget(frame_id="base")
通过 ZMQ endeff_coords 发布给下游控制层
```

要求：

```text
不要修改 PICO4 Unity
不要修改 PICO4 detector
不要修改 keypoint_transform 输出协议
不要修改 CartesianTarget 数据结构
不要破坏 SYSMO-32 Operator
FA 和 SYSMO-32 通过 robot_name 或配置文件选择
```

### 3. 新增 FA C++ IK 适配层

请新增一个 FA IK 封装层，例如：

```text
FaArmIkSolver
```

推荐实现方式之一：

```text
方案 A：新增 ROS2 C++ IK service/action 节点
  Python FaRealControl 调用该 service/action 获取 IK 结果

方案 B：使用 pybind11 封装 ik_7dof/fa_ik_solver.hpp
  Python 侧直接调用 C++ IKSolver

方案 C：FaRealControl 改为 C++ 节点
  直接在 C++ 中订阅 ZMQ/ROS2 输入并调用 IKSolver
```

优先推荐：

```text
方案 A 或方案 B
```

原因：低侵入，方便保留现有 BeaVR Python 遥操作链路。

IK 适配层输入：

```text
arm_side: left | right
target_pose: CartesianTarget 对应的 SE3
current_arm_q: 当前对应手臂 7 维关节角
reference_frame: pelvis | arm_base
max_iters
eps
```

IK 适配层输出：

```text
success: bool
q_target: 7 维目标关节角
position_error
orientation_error
iterations
solve_time_ms
message
```

要求：

```text
IK 初值必须优先使用 /joint_states 中对应手臂当前 7 维关节角
只对有新 CartesianTarget 的手臂调用 IK
没有新 target 的手臂沿用上一帧安全目标
IK max_iters 必须配置化
IK eps 必须配置化
reference_frame 必须配置化，默认 pelvis
IK 失败时保持上一帧安全目标，不要输出 NaN
IK 输出必须检查维度为 7
IK 输出必须检查有限值 np.isfinite / std::isfinite
IK 输出必须经过 joint limit 检查或裁剪
```

### 4. 新增 FA RealControl

请新增：

```text
FaRealControl
```

它应该是一个双臂统一控制器，类似 `Sysmo32RealControl`，不要左右臂各启动一个控制器。

职责：

```text
同时订阅 left/right CartesianTarget
同时处理 left/right reset
同时处理 pause/resume
订阅 /joint_states
提取 FA 左右上肢 7+7 关节反馈
调用 FA C++ IK
经过 jerk_limited_servo smoother
经过 command limiter
组包成 16 维 Float64MultiArray
发布到 /upper_position_controller/commands
```

控制链路：

```text
CartesianTarget
  -> FaArmIkSolver / ik_7dof::IKSolver
  -> FaJerkLimitedServoSmoother 或通用 smoother
  -> FaCommandLimiter
  -> FaUpperPositionCommandBuilder
  -> std_msgs/msg/Float64MultiArray[16]
  -> /upper_position_controller/commands
```

### 5. FA 16 维 command builder

请新增：

```text
FaUpperPositionCommandBuilder
```

输入：

```text
left_arm_7
right_arm_7
neck_yaw
neck_pitch
```

输出：

```text
std_msgs/msg/Float64MultiArray
长度固定为 16
单位 rad
```

映射关系必须严格为：

```text
data[0]  = left_arm_7[0]  # left_shoulder_pitch_joint
data[1]  = left_arm_7[1]  # left_shoulder_roll_joint
data[2]  = left_arm_7[2]  # left_shoulder_yaw_joint
data[3]  = left_arm_7[3]  # left_elbow_joint
data[4]  = left_arm_7[4]  # left_wrist_yaw_joint
data[5]  = left_arm_7[5]  # left_wrist_pitch_joint
data[6]  = left_arm_7[6]  # left_wrist_roll_joint

data[7]  = right_arm_7[0] # right_shoulder_pitch_joint
data[8]  = right_arm_7[1] # right_shoulder_roll_joint
data[9]  = right_arm_7[2] # right_shoulder_yaw_joint
data[10] = right_arm_7[3] # right_elbow_joint
data[11] = right_arm_7[4] # right_wrist_yaw_joint
data[12] = right_arm_7[5] # right_wrist_pitch_joint
data[13] = right_arm_7[6] # right_wrist_roll_joint

data[14] = neck_yaw_joint
data[15] = neck_pitch_joint
```

要求：

```text
发布前 assert len(data) == 16
任何 NaN/Inf 直接拒绝发布
任何关节超限必须限幅或拒绝发布
日志中打印 left/right/neck 的最大变化量
pause 时保持上一帧安全目标或进入 hold
```

### 6. joint_states 解析

`FaRealControl` 必须从 `/joint_states` 中按关节名解析，而不是按数组下标假设。

必须解析：

```text
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_yaw_joint
left_wrist_pitch_joint
left_wrist_roll_joint

right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_yaw_joint
right_wrist_pitch_joint
right_wrist_roll_joint

neck_yaw_joint
neck_pitch_joint
```

要求：

```text
如果 /joint_states 缺少任一关节，real 模式不得发布命令
如果 /joint_states 超时，real 模式不得发布命令
超时时间配置化，例如 joint_state_timeout_sec
```

### 7. smoother 和 limiter

保留当前架构中的：

```text
jerk_limited_servo
min_snap
none
```

FA 默认使用：

```text
jerk_limited_servo
```

原因：VR 遥操作目标是连续流式变化目标，不是离散点到点轨迹。`jerk_limited_servo` 更适合跟踪 VR 目标。

新增或复用：

```text
FaJerkLimitedServoSmoother
FaCommandLimiter
```

要求：

```text
支持 7 维左臂
支持 7 维右臂
支持 neck_yaw / neck_pitch hold 默认值
速度限制配置化
加速度限制配置化
jerk 限制配置化
target jump threshold 配置化
resync threshold 配置化
max_dt / min_dt 配置化
pause/resume/reset 行为保留
```

### 8. control_backend

FA 支持：

```text
mujoco
real
real_with_mujoco
```

要求：

```text
real:
  必须读取新鲜 /joint_states
  必须调用 FA C++ IK
  必须发布 /upper_position_controller/commands

mujoco:
  可只做 dry-run / mirror / IK 验证
  不发布真实 /upper_position_controller/commands

real_with_mujoco:
  真实命令路径与 real 相同
  同时启动 MuJoCo mirror 观察同一条 command
```

如果当前 FA MuJoCo mirror 暂时不可用，需要明确输出 warning，不要影响 real 链路实现。

### 9. 配置入口

请修改机器人加载入口，例如：

```text
load_robot_config(robot_name="fa", ...)
```

使其支持：

```text
robot_name = "fa"
control_backend = "mujoco" | "real" | "real_with_mujoco"
laterality = "left" | "right" | "bimanual"
```

同时确保：

```text
robot_name="sysmo32" 保持原行为
robot_name="fa" 走新增 FA 配置
```

### 10. 启动方式

请补充 FA 启动说明。命令以工程实际入口为准，不要编造不存在的入口。

示例形式：

```bash
# FA dry-run / MuJoCo
python -m beavr.teleop.main --robot fa --control-backend mujoco

# FA real
python -m beavr.teleop.main --robot fa --control-backend real

# FA real + mujoco mirror
python -m beavr.teleop.main --robot fa --control-backend real_with_mujoco
```

如果工程实际入口不是 `beavr.teleop.main`，请根据实际代码补充正确启动命令。

## 五、建议文件结构

请优先按下面结构实现，具体路径以工程实际结构为准：

```text
beavr/teleop/configs/robots/fa_config.py
beavr/teleop/operators/fa_operator.py
beavr/teleop/components/real/fa_real_control.py
beavr/teleop/components/real/fa_command_limiter.py
beavr/teleop/components/real/fa_upper_position_command_builder.py
beavr/teleop/components/real/fa_arm_ik_client.py
beavr/teleop/components/real/fa_arm_ik_service_node.cpp
```

如果采用 pybind11，则可以替换为：

```text
beavr/teleop/kinematics/fa_ik_pybind.cpp
beavr/teleop/kinematics/fa_arm_ik_solver.py
```

## 六、测试要求

### 1. 配置加载测试

```text
load_robot_config("fa", laterality=..., control_backend=...)
```

要求能正常返回 FA 配置。

### 2. /joint_states 解析测试

输入模拟 JointState，检查能按名称解析：

```text
left_arm_7
right_arm_7
neck_2
```

禁止按固定下标解析 `/joint_states`。

### 3. FA C++ IK 测试

调用 `ik_7dof` 提供的测试方式：

```bash
ros2 run ik_7dof fa_arm_kinematic_node --ros-args \
  -p urdf_file:=<FA_URDF_PATH> \
  -p srdf_file:=<FA_SRDF_PATH> \
  -p arm_side:=left \
  -p reference_frame:=pelvis \
  -p num_tests:=100 \
  -p max_iters:=500 \
  -p eps:=1e-3
```

右臂也要测：

```bash
ros2 run ik_7dof fa_arm_kinematic_node --ros-args \
  -p urdf_file:=<FA_URDF_PATH> \
  -p srdf_file:=<FA_SRDF_PATH> \
  -p arm_side:=right \
  -p reference_frame:=pelvis \
  -p num_tests:=100 \
  -p max_iters:=500 \
  -p eps:=1e-3
```

要求输出：

```text
关节名称和限位
正逆解位置/姿态误差
平均耗时
平均迭代步数
成功率
```

如果失败，检查生成的失败样本：

```text
fa_left_arm_ik_failed_cases.log
fa_right_arm_ik_failed_cases.log
```

### 4. FaRealControl IK 集成测试

给定小范围 CartesianTarget：

```text
当前位置附近 +x / +y / +z 小位移
```

要求：

```text
IK 不输出 NaN
输出 7 维 arm q_target
关节值在 limit 内
失败时保持上一帧目标
```

### 5. FA 16 维 command builder 测试

输入：

```text
left_arm_7
right_arm_7
neck_yaw
neck_pitch
```

检查输出：

```text
Float64MultiArray.data 长度 = 16
单位 = rad
顺序严格符合 controllers.yaml
```

映射检查：

```text
data[0:7]   = left_arm_7
data[7:14]  = right_arm_7
data[14:16] = neck_2
```

### 6. ROS2 topic 测试

在 real 模式中检查：

```bash
ros2 topic echo /upper_position_controller/commands
ros2 topic info /upper_position_controller/commands
```

要求：

```text
topic = /upper_position_controller/commands
type = std_msgs/msg/Float64MultiArray
data 长度 = 16
```

### 7. pause/resume/reset 测试

要求：

```text
pause 时停止更新真实命令或保持上一帧安全目标
resume 后恢复追踪
reset 后重新记录 VR 手部基准位姿和机器人当前末端位姿
```

### 8. SYSMO-32 回归测试

必须确认：

```text
robot_name="sysmo32" 原功能不受影响
/sysmo_left_arm_controller/commands 原 SYSMO-32 路径不受影响
```

## 七、代码约束

1. 不要删除或破坏 SYSMO-32 相关代码。
2. 不要修改 PICO4 Unity。
3. 不要修改 PICO4 detector。
4. 不要修改 keypoint_transform 输出协议。
5. 不要修改 CartesianTarget 数据结构。
6. FA 不再发布 `/sysmo_left_arm_controller/commands`。
7. FA 必须发布 `/upper_position_controller/commands`。
8. FA command 长度必须固定为 16。
9. FA arm IK 必须优先使用 `ik_7dof/fa_ik_solver.hpp`。
10. IK 初值必须来自 `/joint_states` 当前关节角。
11. `/joint_states` 必须按关节名解析，不要按下标假设。
12. 所有关键参数必须配置化。
13. 所有不确定字段必须在代码 TODO 和最终报告里明确列出。
14. 不允许静默截断 7 维手臂关节。
15. 不允许发布 NaN/Inf。
16. 不允许在 joint_states 不新鲜时继续发布 real command。

## 八、最终输出

完成后请给出：

1. 修改文件列表。
2. 新增类和职责说明。
3. FA 适配后的数据流。
4. FA 16 维 `/upper_position_controller/commands` command 映射说明。
5. FA C++ IK 的调用方式说明。
6. `pelvis` / `arm_base` 参考坐标系选择说明。
7. 当前仍需人工确认的字段：

   * FA URDF 路径
   * FA SRDF 路径
   * left ee frame
   * right ee frame
   * H_R_V_FA 坐标变换矩阵
   * joint limits 是否直接来自 URDF
   * home/ready pose
   * neck 默认角度
8. 测试命令和测试结果。
9. 如果有失败项，明确说明失败原因，不要假装完成。

## 九、验收标准

满足以下条件才算完成：

```text
robot_name="sysmo32" 原功能不受影响
robot_name="fa" 能加载配置
FaOperator 能生成 CartesianTarget
FaRealControl 能订阅 /joint_states
FaRealControl 能按关节名解析 FA 上肢 7+7+2 关节
FA IK 通过 ik_7dof/fa_ik_solver.hpp 调用
IK 输出左/右臂 7 维目标关节角
jerk_limited_servo smoother 保留
FaUpperPositionCommandBuilder 输出 16 维 Float64MultiArray
FA real 模式发布 /upper_position_controller/commands
发布消息类型为 std_msgs/msg/Float64MultiArray
发布 data 长度固定为 16
pause/resume/reset 行为保留
joint_states 超时或缺关节时不发布真实命令
```
