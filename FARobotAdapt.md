你是一个机器人遥操作系统开发 Agent。请基于当前 BeaVR / beavr-bot 工程，把现有 SYSMO-32 双臂遥操作架构适配到 FA 机器人。

## 一、背景

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
  -> 18D Sysmo32ArmCommand
  -> /sysmo_left_arm_controller/commands
  -> optional MuJoCo mirror
```

当前 SYSMO-32 实机控制层只启动一个双臂控制器，不是左右臂各一个控制器。它同时订阅左右手的 CartesianTarget、reset、pause/resume、transformed hand coords，并从 `/joint_states` 获取当前 12 关节反馈。

当前默认 arm smoother 是 `jerk_limited_servo`，用于连续追踪 VR 流式目标；也保留 `min_snap` 和 `none` 模式。

当前真实机器人 arm command 只有一个 ROS2 topic：

```text
/sysmo_left_arm_controller/commands
std_msgs/msg/Float64MultiArray
```

18 维 payload 格式固定为：

```text
data[0:6]    = left_arm_6
data[6:12]   = right_arm_6
data[12]     = speed_mode
data[13:17]  = reserved
data[17]     = neck_joint
```

注意：不要新增 `/sysmo_right_arm_controller/commands`，不要改变 `/sysmo_left_arm_controller/commands` 的消息类型。手部动作走独立路径，不要合并进 18 维 arm command。

## 二、目标

请在现有架构基础上新增 FA 机器人适配层，使 FA 机器人可以复用现有 VR 遥操作链路。

FA 机器人模型文件路径为：

```text
/home/likunwei/dataCollection/beavr-bot/robots/fa_description
```

本阶段要求：

1. 新增 FA 机器人配置、运动学、Operator、RealControl 或兼容控制封装。
2. FA 机器人先复用 SYSMO-32 的 ROS2 控制接口。
3. 后续再将底层 ROS2 控制接口替换为 FA 机器人的真实 ROS2 控制接口。
4. 当前阶段重点保证：VR 输入、坐标变换、IK、平滑器、命令限幅、18 维命令发布链路先跑通。
5. 不要破坏现有 SYSMO-32 功能。
6. 不要大规模重构工程，只做清晰、可回滚、低侵入式适配。

## 三、实现要求

### 1. 新增 FA robot config

请参考现有 `sysmo32_config.py`、`sysmo_mujoco_config.py` 或类似机器人配置文件，新增 FA 机器人的配置文件，例如：

```text
beavr/teleop/configs/robots/fa_config.py
```

配置内容至少包括：

```text
robot_name = "fa"
laterality 支持 left / right / bimanual
control_backend 支持 mujoco / real / real_with_mujoco
模型路径指向 /home/likunwei/dataCollection/beavr-bot/robots/fa_description
左右臂关节名
左右臂关节数量
左右末端 frame/site 名称
左右臂 joint limits
速度限制
加速度限制
jerk 限制
默认 home / ready 关节位姿
VR 到 robot base 的坐标变换矩阵 H_R_V_FA
```

如果 FA 机器人关节名、末端 frame 名、MJCF/URDF 文件名无法确定，请不要硬编码猜测。请在代码里用清晰的 TODO、配置项或自动解析逻辑处理，并在最终总结中列出需要人工确认的字段。

### 2. 新增 FA Operator

请参考 `Sysmo32Operator` 的实现，新增：

```text
FaOperator
```

目标：

```text
继承现有 XArmOperator 或通用 Operator 重定向逻辑
只替换 FA 专用的 H_R_V_FA 坐标变换矩阵
reset 时记录机器人当前末端位姿和当前 VR 手部基准位姿
正常运行时计算手相对初始帧的运动
映射到 FA robot base frame
生成 CartesianTarget(frame_id="base")
通过 ZMQ endeff_coords 发布给下游控制层
```

要求：

```text
不要修改 keypoint_transform 的输出协议
不要修改 CartesianTarget 的数据结构
不要破坏 SYSMO-32 Operator
FA 和 SYSMO-32 通过 robot_name 或配置文件选择
```

### 3. 新增 FA MuJoCo / Kinematics 适配

请参考 `Sysmo32MujocoKinematics.solve_ik`，新增 FA 版本，例如：

```text
FaMujocoKinematics
```

目标：

```text
加载 /home/likunwei/dataCollection/beavr-bot/robots/fa_description 下的 FA 模型
支持左臂 IK
支持右臂 IK
输入 CartesianTarget
输入当前全身/双臂 joint state
输出对应 arm joint target
```

要求：

```text
IK 初值优先使用当前 /joint_states 对应臂关节值
只对有新 target 的手臂跑 IK
没有新 target 的手臂沿用上一帧目标
IK max_iter 必须配置化
阻尼 DLS 参数必须配置化
失败时保持上一帧安全目标，不要输出 NaN
对 joint limit 做裁剪或安全检查
```

如果 FA 模型是 URDF 而不是 MJCF，请优先检查工程现有的模型加载方式。可以新增 URDF->MuJoCo 的适配说明或转换脚本，但不要引入破坏性依赖。

### 4. 新增 FA RealControl

请新增或封装一个 FA 控制层，例如：

```text
FaRealControl
```

本阶段要求先复用 SYSMO-32 的 ROS2 控制接口：

```text
订阅：/joint_states
发布：/sysmo_left_arm_controller/commands
消息：std_msgs/msg/Float64MultiArray
payload：18 维
```

暂时沿用当前 SYSMO-32 18 维 payload 格式：

```text
data[0:6]    = left_arm_6
data[6:12]   = right_arm_6
data[12]     = speed_mode
data[13:17]  = reserved
data[17]     = neck_joint
```

如果 FA 机器人左右臂不是 6+6 关节，请先做兼容层：

```text
FA 内部 IK 输出真实关节维度
临时命令发布层适配到 18 维 sysmo32 payload
多余维度需要显式处理
不足维度需要显式报错或配置映射
不要静默截断关键关节
```

要求：

```text
保留 jerk_limited_servo smoother
保留 min_snap / none 模式
保留 Sysmo32CommandLimiter 类似的命令限幅逻辑，可新增 FaCommandLimiter
保留 pause/resume/reset 行为
保留 real / mujoco / real_with_mujoco 三种 backend
real 模式必须依赖新鲜 /joint_states
real_with_mujoco 模式既发布真实命令，也启动 MuJoCo mirror 观察同一条命令
```

### 5. 后续 FA ROS2 控制接口预留

请在代码结构中预留 FA 原生 ROS2 控制接口替换点，但当前不要强行实现未知接口。

建议新增抽象层：

```text
ArmCommandPublisherBase
Sysmo32CompatibleCommandPublisher
FaNativeCommandPublisher  # 先留 TODO 或 skeleton
```

当前 FA 使用：

```text
Sysmo32CompatibleCommandPublisher
```

后续只需要把 publisher 替换为：

```text
FaNativeCommandPublisher
```

不要让 IK、smoother、operator、VR 输入链路依赖具体 ROS2 topic。

### 6. 配置入口

请修改机器人加载入口，例如：

```text
load_robot_config(robot_name="fa", ...)
```

使其支持：

```text
robot_name = "fa"
control_backend = "mujoco" | "real" | "real_with_mujoco"
laterality = left | right | bimanual
```

同时确保：

```text
robot_name="sysmo32" 仍然保持原行为
robot_name="fa" 走新增 FA 配置
```

### 7. 启动方式

请补充或新增启动说明，例如：

```bash
# dry-run / MuJoCo
python -m beavr.teleop.main --robot fa --control-backend mujoco

# real, but using sysmo32-compatible ROS2 command interface
python -m beavr.teleop.main --robot fa --control-backend real

# real + mujoco mirror
python -m beavr.teleop.main --robot fa --control-backend real_with_mujoco
```

如果工程当前不是这个入口，请根据实际入口补充正确命令，不要编造不存在的命令。

## 四、约束

1. 不要删除或破坏 SYSMO-32 相关代码。
2. 不要修改 PICO4 Unity、PICO4 detector、keypoint_transform 的通信协议。
3. 不要改变 CartesianTarget 数据结构。
4. 不要新增 `/sysmo_right_arm_controller/commands`。
5. 不要改变 `/sysmo_left_arm_controller/commands` 消息类型。
6. 当前阶段 FA 真机发布仍然走 SYSMO-32 兼容 18 维 command。
7. FA 原生 ROS2 控制接口只做预留，不要在信息不足时乱写。
8. 所有新增代码要尽量 class 化、模块化、低侵入。
9. 对所有关键参数使用配置文件，不要散落硬编码。
10. 任何不确定字段必须在代码和最终报告中明确列出。

## 五、建议文件结构

请优先按下面结构实现，具体路径以工程实际结构为准：

```text
beavr/teleop/configs/robots/fa_config.py
beavr/teleop/operators/fa_operator.py
beavr/teleop/kinematics/fa_mujoco_kinematics.py
beavr/teleop/components/real/fa_real_control.py
beavr/teleop/components/real/fa_command_limiter.py
beavr/teleop/components/real/fa_command_builder.py
beavr/teleop/components/real/arm_command_publisher.py
```

其中：

```text
arm_command_publisher.py
  - ArmCommandPublisherBase
  - Sysmo32CompatibleCommandPublisher
  - FaNativeCommandPublisher skeleton
```

## 六、测试要求

请至少完成以下测试：

### 1. 配置加载测试

```text
load_robot_config("fa", laterality=..., control_backend=...)
```

要求能正常返回 FA 配置。

### 2. 模型加载测试

```text
FaMujocoKinematics 能加载 fa_description 下的模型
能识别左右臂关节
能识别左右末端 frame/site
```

如果失败，需要输出明确错误信息，告诉用户缺少哪个字段或模型文件。

### 3. IK 测试

给定一个小范围 CartesianTarget：

```text
当前位置附近 +x / +y / +z 小位移
```

要求：

```text
IK 不输出 NaN
输出关节数正确
关节值在 limit 内
失败时保持上一帧目标
```

### 4. ROS2 command 测试

在 real 模式中：

```text
读取 /joint_states
发布 /sysmo_left_arm_controller/commands
消息类型 std_msgs/msg/Float64MultiArray
data 长度为 18
```

### 5. 双臂链路测试

输入左右手 CartesianTarget，检查：

```text
左臂 target 写入 data[0:6]
右臂 target 写入 data[6:12]
speed_mode 写入 data[12]
reserved 写入 data[13:17]
neck_joint 写入 data[17]
```

### 6. pause/resume/reset 测试

要求：

```text
pause 时停止更新真实命令或保持安全目标
resume 后恢复追踪
reset 后重新记录 VR 手部基准位姿和机器人当前末端位姿
```

## 七、最终输出

完成后请给出：

1. 修改文件列表。
2. 新增类和职责说明。
3. FA 适配后的数据流。
4. 当前仍复用 SYSMO-32 ROS2 控制接口的位置。
5. 后续替换为 FA 原生 ROS2 控制接口时需要改哪些文件。
6. 需要人工确认的 FA 模型字段，例如：

   * 模型文件名
   * 左右臂 joint names
   * 左右末端 frame/site names
   * joint limits
   * home/ready pose
   * FA 原生 ROS2 command topic
   * FA 原生 ROS2 command message type
7. 测试命令和测试结果。
8. 如果有失败项，明确说明失败原因，不要假装完成。

## 八、验收标准

满足以下条件才算完成：

```text
robot_name="sysmo32" 原功能不受影响
robot_name="fa" 能加载配置
FA 模型能被加载或给出明确缺失信息
FA IK 接口存在并可被 FaRealControl 调用
FA real 模式能复用 /joint_states 和 /sysmo_left_arm_controller/commands
发布的 Float64MultiArray 长度固定为 18
pause/resume/reset 行为保留
jerk_limited_servo smoother 保留
FA 原生 ROS2 控制接口有清晰替换点
```
