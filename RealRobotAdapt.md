你现在是一个机器人遥操作系统代码 Agent。请基于当前 BeaVR VR 遥操作框架，只针对 `sysmo32` 实现真实机器人适配，并新增一个独立 MuJoCo 仿真层，用于接收真实机械臂控制接口数据，验证机械臂关节命令链路。

本任务只关心 `sysmo32`，不要适配 `leap`、`xarm7`、`xarm7_sim`，也不要为了兼容其它机器人引入复杂通用抽象。

## 一、背景

当前项目包含两部分：

1. `BeaVR-app/BeaVR-Unity`

   * 运行在 PICO4 / Quest 类头显上。
   * 采集 XR Hands 左右手 26 个关键点。
   * 通过 NetMQ / ZeroMQ 发送右手、左手、分辨率、暂停/恢复信号。

2. `beavr-bot`

   * Python 后端。
   * 当前 `sysmo32` 分支链路大致为：

```text
PICO4VRHandDetector
  -> TransformHandPositionCoords
  -> Sysmo32Operator
  -> Sysmo32Robot
  -> MockSysmo32Control
  -> MuJoCoSysmoSimulator
```

当前问题：

* `sysmo32` 当前接口层仍然是 Mock，没有真正调用真实机器人的 ROS2 控制接口。
* 当前 MuJoCo 仿真主要走高层 CartesianTarget / URDF 仿真，不是严格接收真实机器人底层接口格式。
* 暂停/恢复链路存在端口对齐问题：PICO4 detector 发布 `pause` 在 `8088`，部分 operator / robot interface 订阅 `pause` 在 `8089`。
* 暂停时 VR 前端不再继续下发手势数据，因此后端不能依赖新的手势帧来处理 pause 后的安全逻辑。
* 真实机器人状态接口为 `/joint_states`。
* 本工程只需要完成 `sysmo32` 真实机器人适配，不需要处理其它机器人类型。

## 二、目标数据流

### 1. 改造前

```text
PICO4 / Unity
  -> PICO4VRHandDetector
  -> TransformHandPositionCoords
  -> Sysmo32Operator
  -> CartesianTarget
  -> MockSysmo32Control / MuJoCoSysmoSimulator
```

### 2. 改造后

```text
PICO4 / Unity
  -> PICO4VRHandDetector
  -> TransformHandPositionCoords
  -> Sysmo32Operator
  -> 左右臂 CartesianTarget
  -> Sysmo32 IK / 关节目标生成
  -> Sysmo32CommandBuilder
  -> Sysmo32CommandLimiter
  -> 18维真实机械臂命令
  -> Sysmo32RealControl
  -> /sysmo_left_arm_controller/commands
```

同时：

```text
PICO4 / Unity 手势 / 手部关键点
  -> HandGestureMapper
  -> left_action / right_action
  -> /left_topic_to_hand
  -> /right_topic_to_hand
```

MuJoCo 仿真层：

```text
/sysmo_left_arm_controller/commands
  -> MuJoCo 订阅
  -> 解析18维机械臂命令
  -> 左右臂关节位置跟踪
```

MuJoCo 手部：

```text
/left_topic_to_hand
/right_topic_to_hand
  -> MuJoCo 订阅
  -> 只打印动作编号
  -> 不执行固定动作
```

注意：MuJoCo 手部不需要、也不应该执行固定动作。原因是当前 MuJoCo 手部模型不具备对应真实灵巧手固定动作执行能力，本次只做话题订阅和日志打印，验证数据是否正确下发。

## 三、真实机器人接口要求

### 1. 机械臂启动方式

真实机器人底层启动命令：

```bash
cd /home/jialimeng/monkey_king
ros2 launch sysmo_bringup sysmo_upbody_start.launch.py > debug/limeng.txt
```

该命令只作为使用说明，不要求代码自动执行，除非项目已有统一 launch 管理机制。

### 2. 机械臂控制 topic

真实机械臂通过 ROS2 topic 接收关节位置命令：

```text
/sysmo_left_arm_controller/commands
```

消息类型：

```cpp
std_msgs::msg::Float64MultiArray
```

发布队列：

```cpp
60
```

数据格式必须严格保持如下顺序：

```cpp
position_command_msg.data = {
    left_arm_j1, left_arm_j2, left_arm_j3, left_arm_j4, left_arm_j5, left_arm_j6,
    right_arm_j1, right_arm_j2, right_arm_j3, right_arm_j4, right_arm_j5, right_arm_j6,
    speed_mode,
    reserved_1, reserved_2, reserved_3, reserved_4,
    neck_joint
};
```

字段语义：

```text
index 0~5   : 左臂6个关节角，单位 rad
index 6~11  : 右臂6个关节角，单位 rad
index 12    : speed_mode
index 13~16 : reserved，默认 0.0
index 17    : neck_joint，默认 0.0
```

约束：

* `speed_mode = 0.0` 表示底层 5 次多项式插值，速度适中，默认使用该模式。
* `speed_mode = 4.0` 表示上层自行插值，底层速度很快。本任务默认不要使用 `4.0`。
* `reserved_1 ~ reserved_4` 默认全部填 `0.0`。
* `neck_joint` 默认填 `0.0`。
* 发布前必须做关节限幅、速度限幅、NaN / Inf 检查。
* 消息长度必须固定为 18。
* 任意异常时不要发布突变命令。

### 3. 真实机器人状态 topic

真实机器人状态接口是：

```text
/joint_states
```

消息类型：

```cpp
sensor_msgs::msg::JointState
```

要求：

* `Sysmo32RealControl` 或状态管理模块必须订阅 `/joint_states`。
* 从 `/joint_states` 中解析 sysmo32 左右臂关节角。
* 根据当前关节状态计算 FK，用于 operator reset 时返回当前 `endeff_homo`。
* 如果 FK 接口暂时缺失，必须至少缓存当前左右臂关节角，并明确 TODO：接入 FK 后返回真实末端位姿。
* 不允许继续依赖 Mock 状态作为真实机器人状态。
* 如果 `/joint_states` 超时未更新，必须认为真实机器人状态不可用，禁止进入真实机器人控制或禁止 reset 成功。

### 4. 灵巧手启动方式

真实灵巧手启动方式：

```bash
cd /home/jialimeng/merger_over/linkerhand-python-sdk-main
cd example/O6/gesture/
python3 linker_hand_loop_O6_0427.py
```

该命令只作为使用说明，不要求代码自动执行，除非项目已有统一 launch 管理机制。

### 5. 灵巧手控制 topic

左手 topic：

```text
/left_topic_to_hand
```

右手 topic：

```text
/right_topic_to_hand
```

消息类型：

```cpp
std_msgs::msg::Int32
```

动作编号：

```text
1 = 松开
2 = 抓瓶子
```

要求：

* 本次不要实现连续手指 IK。
* 本次只做 VR 手势到固定动作编号的转换。
* 左右手分别独立判断，分别发布。
* 暂停、手势超时、手部丢失、输入异常时，必须发布或保持 `1`，表示松开。
* 建议只在动作变化时发布；也可以低频心跳重复发布，例如 2~5 Hz。
* 不要高频重复刷同一个动作编号。

## 四、暂停 / 恢复的重要约束

这是本任务的关键点。

### 1. 暂停时 VR 不再下发手势数据

当前系统中，进入暂停状态后，VR 端不再继续下发新的手势数据。因此后端不能写成：

```text
等待下一帧手势数据
  -> 根据新手势数据判断暂停
  -> 再松开手 / 停止机械臂
```

这是错误设计，因为 pause 后可能根本没有下一帧手势数据。

正确设计：

```text
pause 信号到达
  -> 立即切换 teleop_active = false
  -> 机械臂停止更新目标 / 保持当前位置
  -> 灵巧手立即切换到松开动作 1
  -> 清空或冻结当前手势状态
  -> 等待 resume 信号
```

### 2. 必须增加手势数据超时检测

因为暂停或手部丢失时可能收不到新 frame，所以必须增加输入超时保护。

建议逻辑：

```text
last_right_hand_frame_time
last_left_hand_frame_time

if now - last_hand_frame_time > hand_frame_timeout:
    hand_valid = false
    arm_target_update_allowed = false
    hand_action = 1
```

建议配置：

```yaml
hand_frame_timeout_s: 0.3
```

含义：

* 超过 0.3 秒没有收到某侧手部数据，该侧手部视为无效。
* 对应机械臂停止更新。
* 对应灵巧手动作强制为松开 `1`。
* 不允许继续使用过期手势生成新的抓取动作或机械臂目标。

### 3. pause 和 frame timeout 都要能触发安全状态

安全状态触发条件：

```text
收到 pause
手部数据超时
手部关键点无效
IK 失败
/joint_states 超时
目标跳变过大
ROS2 publisher 异常
```

安全状态动作：

```text
机械臂：停止更新新目标，保持上一条安全目标或当前位置
灵巧手：发布 1，松开
MuJoCo：机械臂停止更新或保持上一目标，手部只打印 action，不执行动作
```

### 4. resume 时必须重新初始化

从 pause 恢复后，必须重新记录：

```text
当前 VR 手部初始位姿
当前真实机器人末端初始位姿
当前 /joint_states 对应的左右臂关节角
```

然后再开始计算相对位移。

禁止在 resume 后沿用 pause 前的 `hand_init_homo`，否则会产生跳变。

## 五、需要实现的功能

### 功能 1：只针对 sysmo32 新增真实机器人控制层

请在 `beavr-bot` 中为 `sysmo32` 新增真实机器人接口层，替换或并存当前 `MockSysmo32Control`。

建议新增类名：

```text
Sysmo32RealControl
Sysmo32Ros2Bridge
Sysmo32JointStateSubscriber
Sysmo32ArmCommandPublisher
Sysmo32HandActionPublisher
```

具体命名以当前项目风格为准。

真实机器人控制层职责：

1. 只处理 `sysmo32`。
2. 接收 `Sysmo32Operator` 输出的左右臂末端目标。
3. 根据 `/joint_states` 当前状态和机器人模型计算 IK。
4. 将 IK 结果组装成 18 维 Float64MultiArray。
5. 发布到 `/sysmo_left_arm_controller/commands`。
6. 订阅 `/joint_states`，维护当前真实机器人关节状态。
7. 在 reset 时基于 `/joint_states` 和 FK 返回当前 `endeff_homo`。
8. 接收 pause / resume 状态。
9. 管理灵巧手固定动作发布。
10. 输出必要日志：

* 当前控制模式。
* 当前 `/joint_states` 更新时间。
* 左右臂目标末端位姿。
* IK 是否成功。
* 发布的 18 维 Float64MultiArray。
* 左右手动作编号。
* pause / resume 状态。
* 安全拦截原因。

不要改造其它 robot_name 的控制链路。

### 功能 2：机械臂位姿跟踪

请沿用 BeaVR 当前机械臂重定向逻辑：

```text
VR 手部初始位姿
VR 手部当前位姿
真实机器人末端初始位姿
=> 计算机器人 base 下目标末端位姿
=> IK
=> 关节位置命令
```

要求：

1. 支持双臂。
2. 左手控制左臂，右手控制右臂。
3. 首次启动时：

   * 等待 `/joint_states` 有效。
   * 根据 `/joint_states` 计算当前左右臂末端位姿。
   * 记录为 `robot_init_homo`。
   * 等待有效 VR 手部 frame。
   * 记录为 `hand_init_homo`。
4. 从 pause 恢复时：

   * 必须重新执行 reset。
   * 重新记录当前 VR 手部位姿。
   * 重新读取 `/joint_states` 并计算当前机器人末端位姿。
5. 分辨率缩放：

   * `High -> 1.0`
   * `Low -> 0.6`
   * 没有按钮事件时保持上一次缩放值。
6. 目标位姿必须限幅：

   * 最大平移增量限幅。
   * 最大姿态增量限幅。
   * 工作空间限幅。
   * 关节角限幅。
   * 关节速度限幅。
7. 输入异常时不能继续运动：

   * VR 手丢失。
   * VR 手势数据超时。
   * frame 退化。
   * `/joint_states` 超时。
   * IK 失败。
   * 目标跳变过大。
   * ROS2 发布器不可用。

### 功能 3：灵巧手固定动作转换

请新增 `sysmo32` 专用手势到固定动作转换模块。

建议模块名：

```text
Sysmo32HandGestureMapper
Sysmo32FixedHandActionController
```

输入：

* PICO4 detector 发布的左右手原始关键点，或 Transform 后的左右手关键点。
* pause / resume 状态。
* 手部 frame 更新时间。
* 可选 button event。

输出：

```text
/left_topic_to_hand  std_msgs/Int32
/right_topic_to_hand std_msgs/Int32
```

动作：

```text
1 = 松开
2 = 抓瓶子
```

手势判断建议：

优先复用当前 Unity / detector 已经存在的手势语义。如果当前后端已有 pinch / button / pause 事件，则不要重复造复杂逻辑。

如果必须从 26 个手部关键点判断，先实现简单稳定版本：

```text
食指指尖与拇指指尖距离小于 grasp_enter_threshold
  -> 连续 confirm_frames 帧满足
  -> 进入抓取动作 2

食指指尖与拇指指尖距离大于 grasp_exit_threshold
  -> 退出抓取动作
  -> 松开动作 1
```

要求：

* 使用迟滞阈值，避免抖动。
* 左右手独立判断。
* pause 时立即切换到 `1`。
* 手势数据超时时立即切换到 `1`。
* 手部关键点无效时立即切换到 `1`。
* 只在状态变化时发布，或以低频心跳重复发布。
* 不实现连续手指关节控制。
* 不接入 LEAP / LinkerHand IK。

### 功能 4：新增独立 MuJoCo 机械臂命令仿真层

请新增一个独立 MuJoCo 仿真层，让 MuJoCo 接收真实机械臂控制接口格式的数据。

目标：

```text
真实机器人控制层生成的18维 Float64MultiArray
  -> 真实机器人 /sysmo_left_arm_controller/commands
  -> MuJoCo 仿真层订阅同一 topic 或同一份内部 command
  -> MuJoCo 左右臂执行关节位置跟踪
```

要求：

1. 单独增加，不要破坏现有 `MuJoCoSysmoSimulator`。
2. 支持配置开关：

   * `control_backend: real`
   * `control_backend: mujoco`
   * `control_backend: real_with_mujoco`
3. 当 `control_backend=mujoco`：

   * 不连接真实机器人。
   * 可以不发布真实 ROS2 arm command topic。
   * 但内部命令格式仍必须是 18 维真实接口格式。
   * MuJoCo 按 18 维命令中的左右臂关节角执行。
4. 当 `control_backend=real`：

   * 发布 `/sysmo_left_arm_controller/commands`。
   * 不启动 MuJoCo 镜像。
5. 当 `control_backend=real_with_mujoco`：

   * 发布 `/sysmo_left_arm_controller/commands`。
   * MuJoCo 同时订阅或接收同一份 18 维命令。
   * 真实机器人和 MuJoCo 的机械臂目标必须来自同一份 command。
6. MuJoCo 必须校验 18 维命令：

   * 长度必须为 18。
   * 0~5 是左臂关节角。
   * 6~11 是右臂关节角。
   * 12 是 speed_mode。
   * 13~16 是 reserved。
   * 17 是 neck_joint。
7. MuJoCo 只做机械臂关节位置跟踪。
8. MuJoCo 手部不执行固定动作。

### 功能 5：MuJoCo 手部只订阅打印，不执行动作

请在 MuJoCo 仿真层增加手部 action 订阅，但不要执行动作。

订阅：

```text
/left_topic_to_hand
/right_topic_to_hand
```

消息类型：

```text
std_msgs/Int32
```

行为：

```text
收到 /left_topic_to_hand:
  打印 [MuJoCo][HandAction] left action = {1 or 2}

收到 /right_topic_to_hand:
  打印 [MuJoCo][HandAction] right action = {1 or 2}
```

要求：

* 不要调用 MuJoCo hand actuator。
* 不要设置 MuJoCo 手部关节。
* 不要模拟抓瓶子动作。
* 如果 action 不是 1 或 2，打印 warning。
* 如果长时间没有收到 hand action，可以打印 throttled debug，不要刷屏。
* 保留函数接口但内部只打印，例如：

```python
def on_left_hand_action(self, action_id: int):
    logger.info("[MuJoCo][HandAction] left action=%d, print only, no execution", action_id)

def on_right_hand_action(self, action_id: int):
    logger.info("[MuJoCo][HandAction] right action=%d, print only, no execution", action_id)
```

### 功能 6：配置文件

请新增或修改 `sysmo32` 配置文件，至少支持：

```yaml
sysmo32_robot:
  control_backend: real        # real / mujoco / real_with_mujoco
  laterality: bimanual

  ros2:
    joint_state_topic: /joint_states
    arm_command_topic: /sysmo_left_arm_controller/commands
    left_hand_topic: /left_topic_to_hand
    right_hand_topic: /right_topic_to_hand
    arm_command_queue_size: 60
    hand_command_queue_size: 10
    joint_state_timeout_s: 0.5

  vr:
    hand_frame_timeout_s: 0.3
    pause_port_policy: bridge_or_unify_8088_8089

  arm:
    speed_mode: 0.0
    reserved: [0.0, 0.0, 0.0, 0.0]
    neck_joint: 0.0
    max_joint_velocity_rad_s: [...]
    joint_lower_limits_rad: [...]
    joint_upper_limits_rad: [...]
    workspace_limits:
      x: [...]
      y: [...]
      z: [...]
    max_translation_step_m: 0.02
    max_rotation_step_rad: 0.08

  hand:
    default_action: 1
    grasp_action: 2
    publish_on_change_only: true
    heartbeat_hz: 3.0
    grasp_enter_threshold_m: 0.035
    grasp_exit_threshold_m: 0.055
    confirm_frames: 3
    force_release_on_pause: true
    force_release_on_timeout: true

  safety:
    hold_arm_on_pause: true
    reject_nan_inf: true
    reject_large_jump: true
    reject_stale_joint_state: true
    reject_stale_hand_frame: true
    emergency_stop_on_ik_fail: false

  mujoco:
    enabled: true
    mirror_real_command_to_sim: true
    model_path: ...
    control_dt: 0.01
    subscribe_arm_command_topic: true
    subscribe_hand_action_topics: true
    execute_hand_action: false
    print_hand_action_only: true
```

实际字段可根据项目 dataclass / config 风格调整，但语义必须完整。

## 六、启动入口

请提供清晰启动方式。

至少支持以下模式：

```bash
# 只跑 MuJoCo 仿真，不连接真实机器人
python -m beavr.teleop.main \
  --robot_name=sysmo32 \
  --laterality=bimanual \
  --teleop.flags.sim_env=True \
  --teleop.flags.robot_interface=True \
  --control_backend=mujoco

# 只跑真实机器人
python -m beavr.teleop.main \
  --robot_name=sysmo32 \
  --laterality=bimanual \
  --teleop.flags.robot_interface=True \
  --control_backend=real

# 真实机器人 + MuJoCo 镜像仿真
python -m beavr.teleop.main \
  --robot_name=sysmo32 \
  --laterality=bimanual \
  --teleop.flags.robot_interface=True \
  --teleop.flags.sim_env=True \
  --control_backend=real_with_mujoco
```

如果当前 CLI 不支持 `--control_backend`，请按项目风格补充参数。

## 七、推荐代码结构

只围绕 `sysmo32` 修改。

推荐结构：

```text
beavr/teleop/robot/sysmo32/
  sysmo32_real_control.py
  sysmo32_ros2_bridge.py
  sysmo32_joint_state_subscriber.py
  sysmo32_arm_command_publisher.py
  sysmo32_hand_action_publisher.py
  sysmo32_command.py
  sysmo32_command_builder.py
  sysmo32_command_limiter.py
  sysmo32_hand_gesture_mapper.py
  sysmo32_mujoco_command_receiver.py
  sysmo32_mujoco_command_applier.py
```

实际路径以当前仓库结构为准。

职责划分：

```text
Sysmo32Operator
  - 只负责 VR 手部位姿到左右臂末端 CartesianTarget

Sysmo32HandGestureMapper
  - 只负责 VR 手势到 left_action / right_action

Sysmo32CommandBuilder
  - 负责把左右臂关节角、speed_mode、reserved、neck_joint 组装成18维命令

Sysmo32CommandLimiter
  - 负责关节限幅、速度限幅、目标跳变检查

Sysmo32RealControl
  - 负责 /joint_states 订阅
  - 负责真实机器人 FK / IK 状态管理
  - 负责发布 /sysmo_left_arm_controller/commands
  - 负责发布 /left_topic_to_hand 和 /right_topic_to_hand

Sysmo32MujocoCommandReceiver
  - 负责接收18维机械臂命令
  - 负责订阅 hand action topic

Sysmo32MujocoCommandApplier
  - 负责 MuJoCo 左右臂关节位置跟踪
  - 手部 action 只打印，不执行
```

## 八、ROS2 集成要求

如果当前 `beavr-bot` 是 Python 后端，请优先使用 `rclpy`。

注意：

* 每个进程只初始化一次 `rclpy.init()`。
* 避免 multiprocessing 下重复初始化 ROS2。
* 退出时 destroy node 并 shutdown。
* `/joint_states` 必须持续订阅。
* 发布 arm command 前必须检查消息长度等于 18。
* 发布 hand action 前必须检查 action 是 1 或 2。
* pause 后即使没有新手势帧，也必须能发布 hand action = 1。
* topic 名称必须来自配置，不要散落硬编码。

## 九、暂停/恢复端口修复

当前 pause 发布和订阅端口可能不一致。

请二选一修复：

### 方案 A：统一订阅 8088

将 `sysmo32` 相关 operator / robot interface 的 pause 订阅改为 `8088` 的 `pause` topic。

### 方案 B：增加 bridge

增加 bridge：

```text
8088 pause -> 8089 pause
```

要求：

* 只影响 `sysmo32`。
* 不破坏其它机器人。
* pause 到达后，不等待下一帧 VR 手势数据，立即进入安全状态。
* resume 到达后，重新 reset。

## 十、安全要求

真实机器人控制必须优先保证安全：

1. 默认 `speed_mode = 0.0`。
2. 禁止默认使用 `speed_mode = 4.0`。
3. 所有关节角发布前必须限幅。
4. 所有关节速度必须限幅。
5. IK 失败不发布新机械臂命令。
6. 输入 NaN / Inf 不发布新机械臂命令。
7. 目标跳变过大不发布新机械臂命令。
8. `/joint_states` 超时不发布新机械臂命令。
9. VR 手势数据超时不更新对应机械臂目标。
10. pause 时不等待下一帧手势数据，立即停止机械臂目标更新。
11. pause 时立即发布左右手松开动作 `1`。
12. 手部数据超时时立即发布对应手松开动作 `1`。
13. MuJoCo 手部 action 只打印，不执行。
14. 所有安全拦截都要打印原因，但要做日志节流，避免刷屏。

## 十一、测试与验收

请完成后提供以下测试结果或测试脚本。

### 1. 单元测试

覆盖：

* 18 维机械臂命令构造。
* 18 维命令长度校验。
* speed_mode 默认是 0.0。
* reserved 默认是 `[0.0, 0.0, 0.0, 0.0]`。
* neck_joint 默认是 `0.0`。
* 左右手 action 编号构造。
* 手势到固定动作转换。
* pause 时不依赖新手势帧也能强制发布 hand action = 1。
* hand frame timeout 时强制 hand action = 1。
* `/joint_states` timeout 时拒绝机械臂控制。
* NaN / Inf 拦截。
* 关节限幅。
* 关节速度限幅。
* MuJoCo hand action callback 只打印，不执行。

### 2. 无真实机器人 dry-run 测试

要求可以在没有真实机器人时运行：

```bash
control_backend=mujoco
```

检查：

* PICO4 手腕运动能生成 18 维机械臂命令。
* MuJoCo 能接收 18 维命令并驱动左右臂关节位置跟踪。
* 抓取手势能发布 `/left_topic_to_hand` 或 `/right_topic_to_hand` 的 `2`。
* 松开手势能发布 `1`。
* MuJoCo 收到 hand action 后只打印，不执行动作。
* pause 后即使没有新的 VR 手势帧，也能立即让 hand action 切到 `1`。
* pause 后机械臂停止更新。

### 3. ROS2 topic 测试

提供命令说明：

```bash
ros2 topic echo /joint_states
ros2 topic echo /sysmo_left_arm_controller/commands
ros2 topic echo /left_topic_to_hand
ros2 topic echo /right_topic_to_hand
```

期望：

```text
/joint_states
  - 能持续收到真实机器人关节状态

/sysmo_left_arm_controller/commands
  - 每条消息长度为18
  - 第0~5位是左臂关节角
  - 第6~11位是右臂关节角
  - 第12位是 speed_mode，默认0.0
  - 第13~16位是 reserved，默认0.0
  - 第17位是 neck_joint，默认0.0

/left_topic_to_hand
/right_topic_to_hand
  - 只发布1或2
  - pause / timeout 时发布1
```

### 4. 真实机器人小步测试

真实机器人测试必须先使用极小位移和低速：

* 平移缩放先设置为很小，例如 0.1。
* 最大单步平移不超过 1~2 cm。
* 最大单步姿态不超过 3~5 度。
* 先测试 `/joint_states` 解析。
* 再测试 18 维命令 dry-run 打印。
* 再测试单臂。
* 再测试双臂。
* 先测试手部松开 / 抓取 topic。
* 最后联动机械臂和手部。
* 确认 pause 后即使 VR 不再发手势帧，也能停止机械臂更新并发布 hand action = 1。

## 十二、交付内容

请最终交付：

1. 修改文件列表。
2. 新增类和职责说明。
3. 新增配置字段说明。
4. 启动命令。
5. ROS2 topic 列表。
6. `/joint_states` 解析逻辑说明。
7. 18 维机械臂命令格式说明。
8. 灵巧手 action 规则说明。
9. MuJoCo 仿真层数据流说明。
10. MuJoCo 手部只打印、不执行的说明。
11. pause / timeout 安全策略说明。
12. 测试方法。
13. 关键代码注释。
14. 若某些接口因仓库缺失无法完全实现，请明确标记 TODO，并提供最小可运行 dry-run 版本。

## 十三、重要限制

* 只做 `sysmo32`。
* 不适配 `leap`。
* 不适配 `xarm7`。
* 不适配 `xarm7_sim`。
* 不修改 Unity 前端，除非必须修复 pause 信号发送问题。
* 不要让真实机器人控制逻辑依赖下一帧 VR 手势数据来处理 pause。
* 不要在 MuJoCo 中执行手部固定动作。
* 不要让 MuJoCo 手部 action 控制任何 actuator 或 hand joint。
* 不要把 Mock 状态当作真实机器人状态。
* 真实状态必须来自 `/joint_states`。
* 不要发布未经限幅和安全检查的机械臂命令。
* 不要默认使用高速 `speed_mode = 4.0`。
* 不要实现复杂连续手指 IK。
* 不要引入影响其它 robot_name 的大范围重构。
