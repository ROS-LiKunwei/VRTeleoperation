你是一个机器人遥操作工程代码 Agent。请基于当前 BeaVR-bot 的 SYSMO-32 实机遥操作架构，实现一个 **jerk-limited online servo smoother**，用于替代或并行支持当前七次 minimum-snap 点到点轨迹优化，从而提升 VR 遥操作的动作跟手效果。

# 1. 当前架构背景

当前 SYSMO-32 实机遥操作链路为：

```text
PICO 4 App
    -> ZMQ 网络层
    -> VR detector / keypoint transform / operator
    -> 左右手 CartesianTarget
    -> Sysmo32RealControl 实机主控制循环
    -> /joint_states 当前 12 关节反馈
    -> Sysmo32MujocoKinematics.solve_ik 左右臂 IK
    -> Sysmo32ArmTrajectorySmoother 七次 minimum-snap 关节轨迹
    -> Sysmo32CommandLimiter joint jump / velocity / joint limit 限幅
    -> Sysmo32CommandBuilder 18 维 arm command
    -> /sysmo_left_arm_controller/commands
```

当前实机 arm command 只有一个话题：

```text
/sysmo_left_arm_controller/commands
```

消息类型：

```text
std_msgs/msg/Float64MultiArray
```

18 维 payload 格式为：

```text
data[0:6]    = left_arm_6
data[6:12]   = right_arm_6
data[12]     = speed_mode
data[13:17]  = reserved / default 0.0
data[17]     = neck_joint
```

注意：

```text
不要创建 /sysmo_right_arm_controller/commands
不要改变 /sysmo_left_arm_controller/commands 的消息类型
不要改变上游 VR / Transform / Operator 的数据协议
```

当前实机配置里 `speed_mode=4.0`，表示底层按较快速度执行，上层必须负责轨迹平滑和安全限幅。

# 2. 为什么要改成 jerk-limited online servo

当前七次 minimum-snap 是点到点轨迹：

```text
start -> goal -> duration -> sample trajectory
```

它适合固定目标点运动，但 VR 遥操作是连续流式目标：

```text
target_0, target_1, target_2, target_3, ...
```

如果每 16ms~20ms 都有新目标，而 smoother 不断对新目标做点到点更新，会出现：

```text
频繁滚动更新
轨迹相位滞后
小幅连续目标跟随变肉
快速目标下 remaining / duration 被不断刷新
```

当前日志结论说明：

```text
VR target 接收约 50Hz
arm command 发布约 60Hz
IK 左臂约 3.1ms，右臂约 2.4ms
publish 约 0.2ms
loop 约 7.3ms
source_to_publish_ms 平均约 99ms
REAL_ARM_COMMAND_DELTA 平均约 0.15rad，p95 约 0.22rad
```

结论：

```text
当前不是没有发布 command；
IK 和 ROS2 publish 不是主要瓶颈；
机械臂能跟随，但快速目标下存在明显相位滞后；
主要瓶颈是流式目标轨迹滞后、上游排队延迟、后级限幅/底层跟踪误差叠加。
```

因此请实现一个在线 servo smoother：

```text
每个控制周期读取最新 IK/limit 后的 target_joints
根据当前 command / velocity / acceleration 状态
在线生成下一帧 command
限制 velocity / acceleration / jerk
永远追最新目标，不再规划完整点到点 duration
```

# 3. 新增 smoother 类型

请新增配置项，允许选择 smoother 类型：

```yaml
arm_trajectory_smoother: "jerk_limited_servo"
```

可选值建议：

```text
"min_snap"              保留现有七次 minimum-snap 行为
"jerk_limited_servo"    新增 jerk-limited online servo
"none"                  不做上层 smoothing，仅保留原有限幅
```

默认建议：

```yaml
arm_trajectory_smoother: "jerk_limited_servo"
```

但如果担心兼容性，可以先默认 `"min_snap"`，同时支持通过配置切到 `"jerk_limited_servo"`。

# 4. 推荐新增文件和类

当前实现位置：

```text
src/beavr/teleop/components/interface/robots/sysmo32_trajectory.py
```

新增类：

```python
class Sysmo32JerkLimitedServoSmoother:
    ...
```

左右臂各一个实例：

```python
self._left_arm_smoother = Sysmo32JerkLimitedServoSmoother(name="left", ...)
self._right_arm_smoother = Sysmo32JerkLimitedServoSmoother(name="right", ...)
```

不要左右臂共用一个 smoother 状态。

# 5. 在线 servo 状态变量

每个 servo 内部维护：

```python
self._q_cmd       # 当前实际下发 command，shape=(6,)
self._q_vel       # 当前命令速度，shape=(6,)
self._q_acc       # 当前命令加速度，shape=(6,)
self._last_time   # 上一次 update 时间
self._initialized # 是否已初始化
```

初始化优先级：

```text
优先使用 /joint_states feedback
其次使用 target_joints
最后使用 last command
绝对不要首帧默认从全 0 跳过去
```

# 6. jerk-limited online servo 核心算法

每一帧输入：

```python
target_joints          # IK + safety pre-limit 后的目标关节，shape=(6,)
feedback_joints        # 当前 /joint_states 对应手臂反馈，shape=(6,)
now                    # time.monotonic()
```

每一帧输出：

```python
q_cmd                  # 本周期要下发的平滑关节 command，shape=(6,)
```

算法逻辑：

```python
error = target_joints - q_cmd

desired_acc = omega**2 * error - 2.0 * damping_ratio * omega * q_vel

desired_acc = clip(desired_acc, -amax, amax)

delta_acc = desired_acc - q_acc
delta_acc = clip(delta_acc, -jmax * dt, jmax * dt)

q_acc = q_acc + delta_acc
q_acc = clip(q_acc, -amax, amax)

q_vel = q_vel + q_acc * dt
q_vel = clip(q_vel, -vmax, vmax)

q_cmd = q_cmd + q_vel * dt
q_cmd = clamp_to_joint_limits(q_cmd)
```

默认使用临界阻尼：

```yaml
arm_servo_damping_ratio: 1.0
```

`omega` 控制响应速度：

```text
omega 越大，跟手越快，但更容易抖或触发限幅
omega 越小，更稳但更慢
```

# 7. 推荐配置项

请在 SYSMO-32 config / YAML 中新增：

```yaml
arm_trajectory_smoother: "jerk_limited_servo"

arm_servo_omega: 35.0
arm_servo_damping_ratio: 1.0

arm_servo_max_velocity_rad_s: 3.0
arm_servo_max_acceleration_rad_s2: 10.0
arm_servo_max_jerk_rad_s3: 120.0

arm_servo_target_deadband_rad: 0.0005
arm_servo_max_dt_s: 0.05
```

保守真机首测建议：

```yaml
arm_trajectory_smoother: "jerk_limited_servo"

arm_servo_omega: 25.0
arm_servo_damping_ratio: 1.0

arm_servo_max_velocity_rad_s: 2.0
arm_servo_max_acceleration_rad_s2: 10.0
arm_servo_max_jerk_rad_s3: 100.0

arm_command_speed_mode: 0.0
```

稳定后再逐步提高：

```yaml
arm_servo_omega: 35.0
arm_servo_max_velocity_rad_s: 3.0
arm_servo_max_acceleration_rad_s2: 20.0
arm_servo_max_jerk_rad_s3: 200.0
arm_command_speed_mode: 4.0
```

更激进但需要谨慎：

```yaml
arm_servo_omega: 45.0
arm_servo_max_velocity_rad_s: 3.0
arm_servo_max_acceleration_rad_s2: 30.0
arm_servo_max_jerk_rad_s3: 300.0
arm_command_speed_mode: 4.0
```

不要默认使用激进配置。

# 8. dt 处理要求

控制循环理论接近 60Hz：

```text
dt ≈ 0.0167s
```

但实机可能有调度抖动。必须处理 dt 异常：

```python
dt = now - self._last_time
dt = np.clip(dt, min_dt, max_dt)
```

如果 `dt > max_dt`，说明控制循环断帧或 pause/resume 后恢复。此时不要直接积分很大一步，应该：

```text
clamp dt 到 max_dt
必要时 reset velocity / acceleration
低频 warning
```

推荐：

```yaml
arm_servo_max_dt_s: 0.05
```

# 9. feedback resync 要求

如果命令状态和真实 `/joint_states` 差距过大，说明底层跟不上、暂停恢复、或者发生了控制断层。

请实现 resync：

```python
if feedback_joints is valid:
    err_feedback = max(abs(feedback_joints - q_cmd))
    if err_feedback > resync_threshold_rad:
        q_cmd = feedback_joints.copy()
        q_vel[:] = 0.0
        q_acc[:] = 0.0
```

后续可选配置：

```yaml
arm_servo_resync_threshold_rad: 0.15
```

当前代码未默认启用 feedback resync。原因是最近日志里 `REAL_ARM_COMMAND_DELTA`
本身会达到 0.15rad 左右，如果每帧按 feedback 重同步，可能把 command 状态拖住。

注意：

```text
resync 不能每帧频繁触发，否则 command 会被 feedback 拖住，跟手性变差。
需要低频日志记录 resync 次数。
```

# 10. target jump 处理要求

如果 IK 目标突然大跳，比如断帧、重置、VR tracking 丢失恢复：

```python
target_jump = max(abs(target_joints - previous_target_joints))
```

如果超过：

```yaml
arm_servo_target_jump_threshold_rad: 0.6
```

当前代码仍由 `Sysmo32CommandLimiter.max_joint_jump_rad` 负责最终跳变保护。

不要用激进速度追过去。应选择以下策略之一：

```text
策略 A：保持当前 command，并等待下一帧稳定 target
策略 B：reset servo，以当前 feedback 为 q_cmd，速度/加速度清零
策略 C：临时降低 omega / vmax / amax / jmax 一段时间
```

首版建议策略 B：

```python
reset(feedback_joints or q_cmd)
```

并低频 warning。

# 11. pause/resume 行为

pause 时：

```text
reset online servo
q_cmd 对齐当前 /joint_states 或当前 hold command
q_vel = 0
q_acc = 0
继续发布 hold command，不追旧 target
```

resume 时：

```text
从当前 /joint_states 重新初始化
不要继续追 pause 前旧 target
清空 pending target 或忽略 pause 前 target
```

这点必须和现有 `safety_hold_arm_on_pause=True`、`pause_hold_heartbeat_hz=20.0` 兼容。

# 12. target missing 行为

如果某一侧手臂本周期没有新 target：

```text
不要外推旧目标
不要继续用旧速度漂移
应保持当前 command 或继续缓慢收敛到 last target
```

建议首版：

```python
if missing_target:
    q_vel *= damping_decay
    q_acc[:] = 0.0
    q_cmd = q_cmd
```

或者：

```text
继续以 last target 更新，但必须有 target timeout，例如 0.1s。
超过 timeout 后 hold 当前 command。
```

后续可选配置：

```yaml
arm_servo_target_timeout_s: 0.10
```

当前代码仍复用 `CartesianTarget` stale 检查和 pause/reset 时的 smoother reset。

超过 timeout：

```text
hold current command
q_vel -> 0
q_acc -> 0
```

# 13. joint limit 与后级 limiter 顺序

当前架构中原有后级 limiter 是：

```text
Sysmo32CommandLimiter
    max_joint_velocity_rad_s = 3.0
    max_joint_jump_rad = 0.5
```

请保持后级 limiter，不要删除。

新顺序建议为：

```text
IK target
    -> optional pre-clamp joint limit
    -> online jerk-limited servo
    -> Sysmo32CommandLimiter 后级 safety limit
    -> Sysmo32CommandBuilder 18 维 command
    -> /sysmo_left_arm_controller/commands
```

servo 内部也可以做 joint limit clamp，但它是防御性保护。最终安全仍由 `Sysmo32CommandLimiter` 兜底。

# 14. RealControl 接入要求

当前 `Sysmo32RealControl` 中应该已有类似逻辑：

```text
left IK joints
right IK joints
-> smoother
-> limiter
-> command builder
-> publish
```

请改造成可配置：

```python
if self.arm_trajectory_smoother == "min_snap":
    left_smooth = self.left_arm_min_snap.update_and_sample(...)
    right_smooth = self.right_arm_min_snap.update_and_sample(...)

elif self.arm_trajectory_smoother == "jerk_limited_servo":
    left_smooth = self.left_arm_servo.update(
        target_joints=left_limited_or_ik_target,
        feedback_joints=current_left_feedback,
        now=now,
        target_timestamp=left_target_timestamp,
        target_valid=left_target_valid,
        paused=not self._teleop_active,
    )
    right_smooth = self.right_arm_servo.update(...)

elif self.arm_trajectory_smoother == "none":
    left_smooth = left_limited_or_ik_target
    right_smooth = right_limited_or_ik_target
```

然后统一进入：

```python
limited_command = self.command_limiter.limit(
    left_smooth,
    right_smooth,
    current_joint_states,
    ...
)

cmd = self.command_builder.build(
    left_arm=limited_command.left,
    right_arm=limited_command.right,
    speed_mode=self.arm_command_speed_mode,
    ...
)

self.arm_pub.publish(cmd)
```

最终发布仍然只能是：

```text
/sysmo_left_arm_controller/commands
```

不要新增 `/sysmo_right_arm_controller/commands`。

# 15. command cache / LeRobot action cache

LeRobot action cache 应记录最终实际下发的平滑 command，而不是 IK raw target。

也就是：

```text
action = after online servo + after final command limiter 的 left/right arm command
```

如果当前 action cache 语义已经记录最终 arm command，请保持该语义。

建议额外 debug 字段记录：

```text
raw_ik_target
online_servo_output
final_limited_command
joint_states_feedback
```

但主 action 应该是最终实际下发 command。

# 16. 日志和诊断要求

新增启动配置日志：

```text
SYSMO-32 online jerk-limited servo config:
enabled/type=...
omega=...
damping_ratio=...
vmax=...
amax=...
jmax=...
resync_threshold=...
target_timeout=...
speed_mode=...
```

新增低频诊断：

```text
[Diag][ONLINE_SERVO]
side=left/right
target_error_max_rad=...
cmd_vel_max_rad_s=...
cmd_acc_max_rad_s2=...
cmd_jerk_max_rad_s3=...
feedback_error_max_rad=...
resync_count=...
target_timeout_count=...
target_jump_count=...
dt_ms=...
```

新增与原有诊断联动观察：

```text
[Diag][REAL_ARM_COMMAND_DELTA]
[Diag][TIMING_REAL]
[Diag][REAL_ARM_COMMAND_RATE]
limit=joint velocity limited / joint jump limited
```

目标：

```text
REAL_ARM_COMMAND_DELTA 不应持续增大
joint velocity limited 次数不应显著增加
joint jump limited 应减少或不出现
source_to_publish_ms 不一定因 smoother 直接下降，但动作相位滞后应降低
```

# 17. 测试要求

请增加单元测试或离线脚本：

```text
tests/interface/test_sysmo32_trajectory.py
```

或：

```text
scripts/offline_test_sysmo32_trajectory.py
```

至少验证：

1. 首帧从 feedback 初始化，不从 0 跳变。
2. target 固定时，q_cmd 单调接近 target。
3. velocity 不超过 vmax。
4. acceleration 不超过 amax。
5. jerk 不超过 jmax。
6. dt 异常大时不会积分出大跳变。
7. feedback 和 q_cmd 偏差超过 resync_threshold 时会 reset 到 feedback。
8. target jump 超阈值时会 reset 或 hold。
9. pause 时 reset，resume 从当前 feedback 重新初始化。
10. target timeout 后 hold 当前 command，不继续漂移。
11. 左右臂状态独立。
12. 输出 shape 必须是 6。
13. 最终 command builder 输出 18 维。
14. 不创建 `/sysmo_right_arm_controller/commands`。

# 18. 仿真验证流程

请提供仿真验证方法：

1. 使用 `backend=mujoco` 或 `real_with_mujoco`。
2. 设置：

```yaml
arm_trajectory_smoother: "jerk_limited_servo"
arm_servo_omega: 25.0
arm_servo_max_velocity_rad_s: 2.0
arm_servo_max_acceleration_rad_s2: 10.0
arm_servo_max_jerk_rad_s3: 100.0
```

3. 记录：

```text
raw IK target
online servo output
final limited command
joint_states feedback
```

4. 绘制每个关节曲线，确认：

```text
online servo output 比 raw IK target 平滑
online servo output 比 min-snap 响应更快
velocity / acceleration / jerk 没超限
```

# 19. 真机保守测试流程

真机首测使用保守配置：

```yaml
arm_trajectory_smoother: "jerk_limited_servo"
arm_servo_omega: 25.0
arm_servo_damping_ratio: 1.0
arm_servo_max_velocity_rad_s: 2.0
arm_servo_max_acceleration_rad_s2: 10.0
arm_servo_max_jerk_rad_s3: 100.0
arm_command_speed_mode: 0.0
```

真机验证步骤：

```text
1. 小幅慢速移动 VR 手
2. 确认 arm command 发布约 60Hz
3. 确认没有 /sysmo_right_arm_controller/commands
4. 确认 pause/resume 正常
5. 检查 joint velocity limited / joint jump limited
6. 检查 REAL_ARM_COMMAND_DELTA
7. 检查是否有明显抖动或冲击
```

如果稳定但慢，再调：

```yaml
arm_servo_omega: 35.0
arm_servo_max_velocity_rad_s: 3.0
arm_servo_max_acceleration_rad_s2: 20.0
arm_servo_max_jerk_rad_s3: 200.0
arm_command_speed_mode: 0.0
```

如果仍稳定，再测试：

```yaml
arm_command_speed_mode: 4.0
```

如果 `speed_mode=4.0` 后抖动或限幅变多，回退到：

```yaml
arm_command_speed_mode: 0.0
```

或者降低：

```yaml
arm_servo_omega
arm_servo_max_acceleration_rad_s2
arm_servo_max_jerk_rad_s3
```

# 20. 与 min-snap 的兼容和回退

必须保留原有 min-snap 实现，不要直接删掉。

配置回退：

```yaml
arm_trajectory_smoother: "min_snap"
```

完全关闭：

```yaml
arm_trajectory_smoother: "none"
```

如果线上测试发现 online servo 有问题，可以立即切回 min-snap 或 none。

# 21. 实现注意事项

1. 不要在 smoother 内部做 ROS2 publish。
2. 不要在 smoother 内部订阅 ZMQ。
3. smoother 只处理关节数组，不处理 CartesianTarget。
4. IK 仍由 `Sysmo32MujocoKinematics.solve_ik()` 完成。
5. `/joint_states` 仍由 `Sysmo32RealControl` 读取后传给 smoother。
6. command builder 仍负责 18 维格式。
7. command limiter 仍作为最终安全兜底。
8. pause/resume 状态由 `Sysmo32RealControl` 传入 smoother。
9. 不要为左右臂各自发布 ROS command；最终必须统一发布一个 18 维 command。
10. 不要改变手部 `/left_topic_to_hand`、`/right_topic_to_hand` 的逻辑。

# 22. 交付内容

完成后请输出：

1. 修改文件列表。
2. 新增 `Sysmo32JerkLimitedServoSmoother` 类说明。
3. 新增配置项说明。
4. RealControl 接入位置说明。
5. 与原 min-snap 的切换方式。
6. 18 维 command 发布格式是否保持不变。
7. pause/resume 行为说明。
8. target timeout / target jump / feedback resync 行为说明。
9. 仿真测试方法。
10. 真机保守测试方法。
11. 如何回退到 min-snap。
12. 是否新增或误用了 `/sysmo_right_arm_controller/commands`，答案必须是没有。

# 23. 验收标准

最终数据流应为：

```text
IK/limit 后的左臂目标 + IK/limit 后的右臂目标
    -> left/right jerk-limited online servo
    -> final command limiter
    -> 18 维 Float64MultiArray
    -> /sysmo_left_arm_controller/commands
    -> MuJoCo mirror
    -> LeRobot action cache
```

验收时必须确认：

```text
左右臂 servo 状态独立
输出 velocity / acceleration / jerk 不超配置限制
pause 时不追旧目标
resume 后从当前 joint_states 初始化
target timeout 后不继续漂移
target jump 后不产生大跳变
feedback error 过大时可 resync
arm command 发布频率仍接近 60Hz
Float64MultiArray.data 长度仍为 18
data[0:6] 是左臂最终 command
data[6:12] 是右臂最终 command
data[12] 是 arm_command_speed_mode
data[13:17] 是 0.0
data[17] 是 neck_joint
没有创建 /sysmo_right_arm_controller/commands
```
