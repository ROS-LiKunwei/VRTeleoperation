你是一个机器人遥操作工程代码 Agent。请基于当前 BeaVR-bot 工程，为 SYSMO32 真机和仿真机械臂命令下发增加七次多项式 minimum snap 轨迹优化。

# 1. 背景

当前 BeaVR-bot 的 SYSMO32 遥操作链路大致是：

```text
PICO4 / Transform
    -> Sysmo32Operator
    -> Sysmo32RealControl
    -> Sysmo32MujocoKinematics IK
    -> safety limiter / command builder
    -> 机械臂真机控制话题
    -> MuJoCo command mirror
    -> LeRobot state/action cache
```

当前真机机械臂接口只有一个 ROS2 控制话题：

```text
/sysmo_left_arm_controller/commands
```

注意：**没有 `/sysmo_right_arm_controller/commands`。不要创建、发布或订阅这个不存在的话题。**

真机机械臂控制话题消息类型为：

```text
std_msgs/msg/Float64MultiArray
```

该话题一次性下发左臂、右臂、速度模式位、保留位和脖子关节，总长度为 18。

`data` 字段定义如下：

```text
data[0:6]    = 左臂 6 个关节弧度
data[6:12]   = 右臂 6 个关节弧度
data[12]     = 速度模式位
               0.0 表示底层 5 次多项式插值，速度适中
               4.0 表示上层插值，底层速度很快；如果使用 4.0，上层必须做好轨迹平滑
data[13:17]  = 4 个保留位，默认 0.0
data[17]     = 脖子关节弧度
```

因此，本次七次多项式轨迹优化的目标不是分别发布左右臂两个话题，而是：

```text
左臂 IK/limit target + 右臂 IK/limit target
    -> 左右臂分别进行七次 minimum snap 平滑
    -> 组装 18 维 Float64MultiArray
    -> 统一发布到 /sysmo_left_arm_controller/commands
```

# 2. 真机测试日志结论

当前真机测试日志说明：

```text
日志范围：15:00:58.710 - 15:03:54.346
有效遥操作段：约 136s，分两段：19.4s 和 116.6s

PICO 原始接收频率：
RightHand avg 39.3Hz，p50 39.7Hz，p95 57.4Hz，max 60.0Hz
LeftHand  avg 38.8Hz，p50 38.2Hz，p95 58.0Hz，max 59.0Hz

Operator 目标点输出：
right/left 基本 60Hz，avg 59.55Hz

Real target 接收：
right/left avg 约 50.6Hz，p50 51.6Hz

真机 arm command 下发：
avg 59.2Hz，p50 59.6Hz，p95 59.9Hz，max 60.3Hz

RealControl：
source_to_publish avg 99.3ms，p50 96.8ms，p95 142.2ms
real loop avg 6.18ms，p95 9.15ms
publish avg 0.21ms
build/limit avg 0.19ms

IK：
right avg 2.36ms，p95 3.8ms
left  avg 2.90ms，p95 4.0ms

异常/限幅：
PICO_RX_GAP 5 次，其中有效段内主要有一次约 5s 断帧和一次 200ms 右手断帧
OP_HAND_FRAME_GAP 3 次
REAL_TARGET_GAP 2 次
joint velocity limited 17 次
joint jump limited 1 次
左手姿态 clamp 1 次
```

关键结论：

```text
bot 下发已经基本达到 60Hz；
IK、build、publish 已经不是主要瓶颈；
PICO 原始手帧没有稳定到 60Hz，平均约 39Hz；
端到端排队延迟仍偏高；
RealControl source_to_publish avg≈99ms，p95≈142ms；
因此七次多项式轨迹段时间不能按 60Hz 单帧周期设置。
```

所以本次实现时，真机默认轨迹时间不要使用 `0.08s`。
真机默认值请使用：

```yaml
arm_min_snap_segment_time: 0.18
arm_min_snap_min_duration: 0.06
arm_min_snap_replan_threshold_rad: 0.0005
```

如果区分仿真和真机：

```yaml
# 仿真
arm_min_snap_segment_time: 0.10
arm_min_snap_min_duration: 0.04

# 真机默认
arm_min_snap_segment_time: 0.18
arm_min_snap_min_duration: 0.06

# 真机保守模式
arm_min_snap_segment_time: 0.22
arm_min_snap_min_duration: 0.08
```

# 3. 任务目标

请在给真机和仿真的机械臂命令下发前，增加七次多项式 minimum snap 轨迹优化。

核心目标：

1. 每次 IK 得到新的关节目标后，不要直接跳变发布目标关节角。
2. 使用当前实际关节反馈作为轨迹起点。
3. 使用 IK + safety limiter 后的安全目标关节角作为轨迹终点。
4. 左臂和右臂分别使用独立的七次多项式 smoother。
5. 轨迹时间根据关节最大速度、最大加速度自动拉长。
6. 真机 ROS2 publish 和 MuJoCo command mirror 必须使用同一份平滑后的 arm command。
7. LeRobot action cache 优先记录最终实际下发的平滑命令。
8. 不要破坏现有 pause/resume、安全限幅、IK、日志统计、录制逻辑。
9. 不要修改上游 VR、Transform、Operator 的数据协议。
10. 不要修改 `/sysmo_left_arm_controller/commands` 的消息类型。
11. 不要为了平滑降低主控制循环频率。
12. 不得创建或使用 `/sysmo_right_arm_controller/commands`。

# 4. 参考实现

请参考我提供的 `sysmo_hand` 包中七次多项式轨迹优化实现，尤其是以下思想：

```text
startMinSnapTrajectory()
computeConstrainedDuration()
sampleMinSnapTrajectory()
minSnapPositionBlend()
clampToJointLimits()
```

七次 minimum snap 位置插值基函数为：

```python
s(tau) = 35*tau**4 - 84*tau**5 + 70*tau**6 - 20*tau**7
```

满足：

```text
s(0)=0
s(1)=1
两端速度为 0
两端加速度为 0
两端 jerk 为 0
```

一阶导数：

```python
s_dot(tau) = 140*tau**3 - 420*tau**4 + 420*tau**5 - 140*tau**6
```

二阶导数：

```python
s_ddot(tau) = 420*tau**2 - 1680*tau**3 + 2100*tau**4 - 840*tau**5
```

# 5. 实现位置要求

优先在 RealControl 下发前实现，不要在 Operator 层做。

建议新增文件：

```text
src/beavr/teleop/components/interface/robots/sysmo32_trajectory.py
```

新增类：

```python
class Sysmo32ArmTrajectorySmoother:
    ...
```

然后在：

```text
src/beavr/teleop/components/interface/robots/sysmo32_real_control.py
```

中左右臂各创建一个 smoother：

```python
self.left_arm_smoother = Sysmo32ArmTrajectorySmoother(...)
self.right_arm_smoother = Sysmo32ArmTrajectorySmoother(...)
```

如果当前工程已有更合适的 command builder / command limiter 类，也可以集成到现有结构里，但必须保证轨迹优化发生在：

```text
IK + safety limit 之后
18 维 Float64MultiArray 组装之前
ROS2 arm command publish 之前
MuJoCo command mirror 之前
LeRobot action cache 更新之前
```

正确顺序应该是：

```text
CartesianTarget
    -> IK
    -> joint/cartesian safety limiter
    -> left/right seventh-order minimum snap smoother
    -> build 18-dim Float64MultiArray
    -> publish /sysmo_left_arm_controller/commands
    -> publish MuJoCo command mirror
    -> update LeRobot action cache
```

# 6. 真机命令组装要求

最终真机命令只能发布到：

```text
/sysmo_left_arm_controller/commands
```

消息类型：

```text
std_msgs/msg/Float64MultiArray
```

发布内容必须是 18 维：

```python
cmd.data = [
    # data[0:6] 左臂 6 个关节弧度
    left_smooth_target[0],
    left_smooth_target[1],
    left_smooth_target[2],
    left_smooth_target[3],
    left_smooth_target[4],
    left_smooth_target[5],

    # data[6:12] 右臂 6 个关节弧度
    right_smooth_target[0],
    right_smooth_target[1],
    right_smooth_target[2],
    right_smooth_target[3],
    right_smooth_target[4],
    right_smooth_target[5],

    # data[12] 速度模式位
    arm_command_speed_mode,

    # data[13:17] 保留位，默认 0.0
    0.0,
    0.0,
    0.0,
    0.0,

    # data[17] 脖子关节弧度
    neck_joint_target,
]
```

其中：

```text
arm_command_speed_mode = 0.0 或 4.0
```

含义：

```text
0.0 表示底层 5 次多项式插值，速度适中
4.0 表示上层插值，底层速度很快；如果使用 4.0，上层必须做好轨迹平滑
```

本次做的是上层七次多项式插值，理论上可以使用 `4.0`。但是必须提供配置项控制，不要写死。

建议新增配置：

```yaml
arm_command_speed_mode: 4.0
```

真机首测如果担心风险，可以临时设为：

```yaml
arm_command_speed_mode: 0.0
```

但需要在说明中明确：

```text
如果 arm_command_speed_mode=0.0，则会形成“上层七次插值 + 底层五次插值”的双重插值，运动会更柔和但响应更慢。
如果 arm_command_speed_mode=4.0，则主要依赖上层七次插值，响应更快，但必须确认上层轨迹足够平滑。
```

# 7. 配置项要求

请为 SYSMO32 增加以下配置项。优先放在已有的 sysmo32 config dataclass / YAML 中。

真机默认值：

```yaml
arm_min_snap_enabled: true

# 根据真机日志设置：
# PICO 原始帧约 39Hz，RealControl source_to_publish avg≈99ms，p95≈142ms
# 所以默认不要低于 0.14s，首版使用 0.18s
arm_min_snap_segment_time: 0.18
arm_min_snap_min_duration: 0.06
arm_min_snap_replan_threshold_rad: 0.0005

# 真机速度模式位
# 0.0 = 底层 5 次插值
# 4.0 = 上层插值，底层速度快
arm_command_speed_mode: 4.0

# 初始保守速度/加速度限制，后续根据 joint velocity limited 次数调参
arm_joint_velocity_max: [0.8, 0.8, 0.8, 0.8, 1.0, 1.0]
arm_joint_acceleration_max: [2.5, 2.5, 2.5, 2.5, 3.0, 3.0]
```

如果已有硬件速度/加速度/关节位置限制，请优先复用已有配置，不要重复定义冲突参数。

如果支持仿真和真机分别配置，建议：

```yaml
# simulation
arm_min_snap_enabled: true
arm_min_snap_segment_time: 0.10
arm_min_snap_min_duration: 0.04
arm_min_snap_replan_threshold_rad: 0.0003
arm_command_speed_mode: 4.0

# real robot
arm_min_snap_enabled: true
arm_min_snap_segment_time: 0.18
arm_min_snap_min_duration: 0.06
arm_min_snap_replan_threshold_rad: 0.0005
arm_command_speed_mode: 4.0

# real robot conservative
arm_min_snap_enabled: true
arm_min_snap_segment_time: 0.22
arm_min_snap_min_duration: 0.08
arm_min_snap_replan_threshold_rad: 0.0005
arm_command_speed_mode: 0.0
```

# 8. 轨迹平滑类设计要求

请实现一个独立、可测试的 Python 类，例如：

```python
class Sysmo32ArmTrajectorySmoother:
    def __init__(
        self,
        enabled: bool,
        segment_time: float,
        min_duration: float,
        replan_threshold_rad: float,
        velocity_limits: Sequence[float],
        acceleration_limits: Sequence[float],
        joint_lower_limits: Optional[Sequence[float]] = None,
        joint_upper_limits: Optional[Sequence[float]] = None,
        logger: Optional[logging.Logger] = None,
        name: str = "arm",
    ):
        ...
```

至少支持以下方法：

```python
def reset(self, hold_joints: Optional[np.ndarray] = None) -> None:
    ...

def update_and_sample(
    self,
    target_joints: np.ndarray,
    current_feedback_joints: Optional[np.ndarray],
    now: float,
    force_replan: bool = False,
) -> np.ndarray:
    ...

def sample(self, now: float) -> np.ndarray:
    ...

def has_active_trajectory(self, now: float) -> bool:
    ...
```

内部状态建议包含：

```python
self._start_joints
self._goal_joints
self._last_command_joints
self._start_time
self._duration
self._active
```

# 9. 七次多项式实现要求

实现：

```python
def _blend(self, tau: float) -> float:
    tau = float(np.clip(tau, 0.0, 1.0))
    return (
        35.0 * tau**4
        - 84.0 * tau**5
        + 70.0 * tau**6
        - 20.0 * tau**7
    )
```

实现一阶、二阶导数：

```python
def _blend_dot(self, tau: float) -> float:
    tau = float(np.clip(tau, 0.0, 1.0))
    return (
        140.0 * tau**3
        - 420.0 * tau**4
        + 420.0 * tau**5
        - 140.0 * tau**6
    )

def _blend_ddot(self, tau: float) -> float:
    tau = float(np.clip(tau, 0.0, 1.0))
    return (
        420.0 * tau**2
        - 1680.0 * tau**3
        + 2100.0 * tau**4
        - 840.0 * tau**5
    )
```

最大导数可以在初始化时密集采样计算：

```python
taus = np.linspace(0.0, 1.0, 1001)
self._max_blend_dot = max(abs(self._blend_dot(t)) for t in taus)
self._max_blend_ddot = max(abs(self._blend_ddot(t)) for t in taus)
```

# 10. 轨迹时间约束要求

根据七次基函数：

```text
q(t) = q0 + dq * s(tau)
tau = t / T
```

速度和加速度为：

```text
q_dot = dq * s_dot(tau) / T
q_ddot = dq * s_ddot(tau) / T^2
```

所以要满足：

```text
T >= abs(dq) * max_s_dot / vmax
T >= sqrt(abs(dq) * max_s_ddot / amax)
```

实现：

```python
def _compute_constrained_duration(
    self,
    start_joints: np.ndarray,
    goal_joints: np.ndarray,
) -> float:
    dq = np.abs(goal_joints - start_joints)

    duration = max(self.segment_time, self.min_duration)

    for i in range(num_joints):
        if self.velocity_limits[i] > 1e-6:
            duration = max(
                duration,
                dq[i] * self._max_blend_dot / self.velocity_limits[i],
            )

        if self.acceleration_limits[i] > 1e-6:
            duration = max(
                duration,
                math.sqrt(dq[i] * self._max_blend_ddot / self.acceleration_limits[i]),
            )

    return duration
```

如果 duration 被速度/加速度约束拉长，需要低频 warning，不要每帧刷屏。

# 11. 重规划逻辑要求

每次收到新的 `target_joints` 时：

1. 如果 smoother disabled，直接返回 `target_joints`。
2. 如果 `target_joints` 非法，直接保持上一帧命令或反馈值，不要产生 NaN。
3. 如果没有有效 `/joint_states` 反馈：

   * 首帧不要默认用全 0 作为起点。
   * 可退化为直接返回 target，或使用上一帧 command 作为起点。
   * 必须打 warning。
4. 如果当前没有 active trajectory：

   * 起点优先用 `current_feedback_joints`。
   * 没有反馈时用 `last_command_joints`。
   * 终点用 `target_joints`。
   * 开始新轨迹。
5. 如果当前有 active trajectory，但新目标和当前 goal 差异小于 `replan_threshold_rad`：

   * 不重规划。
   * 继续 sample 当前轨迹。
6. 如果当前有 active trajectory，新目标和当前 goal 差异较大：

   * 采用滚动重规划。
   * 起点优先用当前轨迹 sample 值或最新反馈值。
   * 终点为新目标。
   * 重新计算 duration。
7. 轨迹完成后保持 goal，不要回弹。

目标变化判断建议：

```python
delta_norm = np.linalg.norm(target_joints - self._goal_joints, ord=np.inf)
if delta_norm < self.replan_threshold_rad:
    return self.sample(now)
```

注意：`arm_min_snap_replan_threshold_rad` 真机默认用 `0.0005`，避免 PICO 抖动导致每帧重规划。

# 12. pause/resume 行为要求

pause 状态下不要继续追未完成轨迹。

当检测到 pause：

```text
冻结当前下发值，或者保持当前反馈值；
reset smoother；
不要继续推进上一段轨迹。
```

resume 后：

```text
从当前 /joint_states 反馈重新规划；
不要从 pause 前的旧目标继续追。
```

如果现有代码有 pause/resume 命令端口或状态变量，请复用现有状态，不要新增并行状态机。

# 13. 左右臂独立要求

左右臂必须独立规划：

```python
self.left_arm_smoother
self.right_arm_smoother
```

不要左右臂共用一个 smoother 状态。

如果只有左臂有新目标：

```text
左臂按新目标重规划；
右臂保持原逻辑，可以继续采样未完成轨迹，完成后保持最后目标。
```

如果只有右臂有新目标，同理。

# 14. RealControl 接入伪代码

请找到当前实际发布 arm command 的代码，逻辑可能类似：

```python
left_limited_target = ...
right_limited_target = ...

cmd = Float64MultiArray()
cmd.data = [
    *left_limited_target,
    *right_limited_target,
    speed_mode,
    0.0, 0.0, 0.0, 0.0,
    neck_joint_target,
]
arm_pub.publish(cmd)

publish_mujoco_command_mirror(...)
update_lerobot_action_cache(...)
```

请改为：

```python
now = time.monotonic()

left_smooth_target = self.left_arm_smoother.update_and_sample(
    target_joints=left_limited_target,
    current_feedback_joints=current_left_feedback_joints,
    now=now,
)

right_smooth_target = self.right_arm_smoother.update_and_sample(
    target_joints=right_limited_target,
    current_feedback_joints=current_right_feedback_joints,
    now=now,
)

cmd = Float64MultiArray()
cmd.data = [
    float(left_smooth_target[0]),
    float(left_smooth_target[1]),
    float(left_smooth_target[2]),
    float(left_smooth_target[3]),
    float(left_smooth_target[4]),
    float(left_smooth_target[5]),

    float(right_smooth_target[0]),
    float(right_smooth_target[1]),
    float(right_smooth_target[2]),
    float(right_smooth_target[3]),
    float(right_smooth_target[4]),
    float(right_smooth_target[5]),

    float(self.arm_command_speed_mode),

    0.0,
    0.0,
    0.0,
    0.0,

    float(neck_joint_target),
]

self.arm_command_pub.publish(cmd)

publish_mujoco_command_mirror(
    left_arm_joints=left_smooth_target,
    right_arm_joints=right_smooth_target,
    ...
)

update_lerobot_action_cache(
    left_arm_command=left_smooth_target,
    right_arm_command=right_smooth_target,
    ...
)
```

如果当前工程中 arm command 是 12 维双臂数组或 18-field command，请保持原格式，只替换其中 arm joints 部分为平滑后的 joints。

# 15. joint limit 处理要求

轨迹 smoother 内部可以再次 clamp 到 joint limits，但不要替代原有 safety limiter。

顺序应为：

```text
原有 safety limiter：负责安全边界
smoother 内部 clamp：防御性保护，避免数值误差越界
```

如果目标已经被原有限幅器处理，smoother 不应该改变目标语义。

# 16. 日志要求

启动时打印一次配置：

```text
Arm minimum snap config: enabled=..., segment_time=..., min_duration=..., replan_threshold=..., speed_mode=..., vmax=..., amax=...
```

轨迹 duration 被拉长时，低频 warning：

```text
left arm minimum snap duration stretched: requested=0.180s, constrained=0.240s, max_delta=...
```

建议增加低频统计字段：

```text
left_arm_min_snap_ms
right_arm_min_snap_ms
left_trajectory_duration
right_trajectory_duration
left_target_delta_norm
right_target_delta_norm
left_replan_count
right_replan_count
arm_command_speed_mode
```

不要在 60Hz 控制循环里每帧打印完整关节数组。完整数组只允许 debug 开关开启时打印。

# 17. 测试要求

请增加最小单元测试或离线脚本，验证以下场景：

1. `tau=0` 时输出起点。
2. `tau=1` 时输出终点。
3. 中间点连续、平滑、无跳变。
4. 起点和终点相同，不产生 NaN。
5. 速度限制很小时，duration 自动变长。
6. 加速度限制很小时，duration 自动变长。
7. 连续新目标到来时，轨迹从当前采样点或当前反馈点滚动重规划，不跳变。
8. `arm_min_snap_enabled=false` 时，行为与原始逻辑一致，直接返回限幅后的 IK 目标。
9. pause 时 smoother reset，resume 后从当前反馈重新规划。
10. MuJoCo command mirror 和 ROS2 publish 使用的是同一份平滑后关节命令。
11. LeRobot action cache 记录最终实际下发的平滑 command。
12. 最终真机发布的 `Float64MultiArray.data` 长度必须为 18。
13. `data[0:6]` 必须是平滑后的左臂关节。
14. `data[6:12]` 必须是平滑后的右臂关节。
15. `data[12]` 必须来自配置 `arm_command_speed_mode`。
16. `data[13:17]` 必须保持 0.0。
17. 不得出现 `/sysmo_right_arm_controller/commands` 相关发布器。

可以新增类似脚本：

```text
scripts/offline_test_sysmo32_min_snap.py
```

或 pytest：

```text
tests/interface/test_sysmo32_min_snap.py
```

# 18. 仿真验证要求

请提供仿真测试方法：

1. 启动 sysmo32 MuJoCo 仿真。
2. 启动 teleop 或构造离线 target 输入。
3. 观察 `/sysmo_left_arm_controller/commands` 是否为 18 维连续命令。
4. 确认没有 `/sysmo_right_arm_controller/commands`。
5. 观察 MuJoCo mirror 是否使用平滑后的 left/right arm command。
6. 对比开启/关闭 `arm_min_snap_enabled` 的关节曲线差异。

建议增加简单绘图脚本，记录并绘制：

```text
raw IK target joints
limited target joints
smoothed command joints
joint_states feedback
```

# 19. 真机保守测试要求

真机首测不要激进。请给出如下测试流程：

首先设置保守配置：

```yaml
arm_min_snap_enabled: true
arm_min_snap_segment_time: 0.22
arm_min_snap_min_duration: 0.08
arm_min_snap_replan_threshold_rad: 0.0005
arm_command_speed_mode: 0.0
```

低速、小幅度遥操作，检查：

```text
/sysmo_left_arm_controller/commands 是否稳定发布 18 维 Float64MultiArray
data[0:6] 是否为左臂平滑命令
data[6:12] 是否为右臂平滑命令
data[12] 是否等于 arm_command_speed_mode
是否还有明显跳变
joint velocity limited 次数是否下降
joint jump limited 是否消失
```

如果保守模式稳定，再尝试：

```yaml
arm_min_snap_segment_time: 0.18
arm_min_snap_min_duration: 0.06
arm_command_speed_mode: 0.0
```

如果仍然稳定但响应偏慢，再尝试：

```yaml
arm_command_speed_mode: 4.0
```

使用 `4.0` 前必须确认：

```text
上层七次多项式轨迹已经生效；
轨迹输出没有跳变；
pause/resume 不追旧轨迹；
断帧后不会产生大跳变；
joint velocity limited 次数可接受。
```

如果运动过慢但限幅很少，可以尝试：

```yaml
arm_min_snap_segment_time: 0.15
arm_min_snap_min_duration: 0.05
```

如果 `joint velocity limited` 仍然很多，不要继续减小时间，应该增大到：

```yaml
arm_min_snap_segment_time: 0.22
```

甚至：

```yaml
arm_min_snap_segment_time: 0.25
```

# 20. 调参判断标准

请在实现说明里写清楚：

```text
如果 joint velocity limited 次数仍然很多：
    segment_time 从 0.18 -> 0.22 或 0.25

如果 joint jump limited 仍然出现：
    segment_time 增大，或者检查 target 是否有断帧后突跳

如果末端运动太肉、明显跟手慢，但限幅很少：
    segment_time 从 0.18 -> 0.15 或 0.14

如果 real source_to_publish p95 仍然 > 140ms：
    不建议 segment_time 低于 0.16

如果 PICO 原始手帧仍然只有 39Hz：
    不建议 segment_time 低于 0.14

如果 PICO 稳定到 60Hz，且 source_to_publish p95 降到 80ms 以下：
    可以尝试 0.10~0.12

如果 arm_command_speed_mode=0.0：
    运动更柔和但响应更慢，因为底层也在做 5 次插值

如果 arm_command_speed_mode=4.0：
    响应更快，但必须确保上层七次轨迹足够平滑
```

# 21. 回退要求

必须支持一键关闭：

```yaml
arm_min_snap_enabled: false
```

关闭后行为必须与原先一致：

```text
IK target
    -> safety limiter
    -> 组装 18 维 Float64MultiArray
    -> 发布 /sysmo_left_arm_controller/commands
    -> MuJoCo command mirror
    -> LeRobot action cache
```

不能因为新增 smoother 影响原始下发逻辑。

# 22. 最终交付内容

完成后请输出：

1. 修改文件列表。
2. 新增类和函数说明。
3. 新增配置项说明。
4. 数据流变化说明。
5. 真机 18 维命令格式说明。
6. 七次多项式轨迹公式说明。
7. 轨迹时间约束公式说明。
8. 仿真测试方法。
9. 真机保守测试方法。
10. 如何关闭功能回退。
11. 本次实现是否影响上游 VR、Transform、Operator 协议。
12. 明确说明没有使用 `/sysmo_right_arm_controller/commands`。

# 23. 验收标准

最终数据流应为：

```text
IK/limit 后的左臂目标 + IK/limit 后的右臂目标
    -> 左右臂各自七次 minimum snap 轨迹优化
    -> left_smooth_target + right_smooth_target
    -> 组装 18 维 Float64MultiArray
    -> 发布到 /sysmo_left_arm_controller/commands
    -> MuJoCo command mirror
    -> LeRobot action cache
```

验收时必须确认：

```text
真机 ROS2 arm command 使用平滑后 joints
只发布 /sysmo_left_arm_controller/commands
不创建 /sysmo_right_arm_controller/commands
Float64MultiArray.data 长度为 18
data[0:6] 是左臂平滑 joints
data[6:12] 是右臂平滑 joints
data[12] 是 arm_command_speed_mode
data[13:17] 是 0.0
data[17] 是 neck_joint_target
MuJoCo mirror 使用平滑后 joints
LeRobot action cache 使用平滑后 joints
pause 时不会继续追旧轨迹
resume 后从当前反馈重新规划
arm_min_snap_enabled=false 可以完全回退
```

不要修改 `/sysmo_left_arm_controller/commands` 消息类型。
不要新增 `/sysmo_right_arm_controller/commands`。
不要改变上游 VR、Transform、Operator 的数据协议。
不要降低 RealControl 主循环频率。
不要只做仿真，不接真机发布路径。
不要只做真机，不接 MuJoCo mirror。
