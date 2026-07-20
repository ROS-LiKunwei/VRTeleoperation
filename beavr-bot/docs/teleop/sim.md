你正在修改项目：

`/home/likunwei/dataCollection/beavr-bot`

请先完整阅读以下文档和相关代码，再开始修改：

* `docs/teleop/fa_coordinate_transform.md`
* `docs/teleop/operators.md`
* `docs/teleop/index.md`
* `src/beavr/teleop/components/detector/vr/keypoint_transform.py`
* `src/beavr/teleop/components/operator/robots/fa_operator.py`
* `src/beavr/teleop/components/operator/robots/xarm7_operator.py`
* `src/beavr/teleop/configs/robots/fa_config.py`
* FA reset、resume、控制状态、输入消息和状态发布相关代码
* `fa_real_control.py` 及 FA IK/min-snap 后端代码，仅用于确认接口边界，不要无故修改其控制算法

# 一、任务背景

目前 FA 双臂遥操作的坐标链路为：

```text
PICO/Unity InputFrame
  -> keypoint_transform.py
  -> fa_operator.py / xarm7_operator.py
  -> fa_real_control.py
  -> FA IK / min-snap / upper-body command
```

Unity/PICO 输入先经过固定的左手系到内部右手系转换：

```text
internal_xyz = unity_xyz * [1, 1, -1]
```

FA Operator 当前使用固定的 `H_R_V_FA`，内部实际使用：

```python
r_vr_to_robot_nominal = inverse(H_R_V_FA[:3, :3])
```

当前固定方向关系为：

```text
robot_x = -vr_z
robot_y = -vr_x
robot_z =  vr_y
```

reset 或 resume 后，现有代码记录：

```text
robot_init_h
hand_init_h
```

之后使用手腕相对初始位姿增量控制机器人：

```python
hand_translation_delta_vr =
    current_wrist_vr - init_wrist_vr

robot_translation_delta =
    r_vr_to_robot @ hand_translation_delta_vr * resolution_scale
```

当前旋转默认配置为：

```python
rotation_delta_frame = "base"
```

并使用：

```python
robot_rotation_delta = (
    r_vr_to_robot
    @ hand_rotation_delta_vr
    @ r_robot_to_vr
)
```

实际测试中，PICO4 挂在脖子上而不是正常佩戴时，PICO tracking/world 坐标系可能相对操作者人体方向发生 pitch、roll 或 yaw 偏转。此时操作者双手平行向左移动，机器人可能出现向前上、前下等耦合运动。

这不是单纯的位置零点偏差，而是 VR 世界坐标系与操作者人体坐标系之间存在未补偿的旋转偏差。

# 二、总体目标

为 FA 双臂遥操作增加一套“每次开启遥操作必须执行”的双手运动方向标定机制。

采用以下方案：

1. 完全不使用实时头显姿态作为遥操作控制基准。
2. 每次开启或重新使能遥操作时，通过操作者双手的“向前”和“向左”平行动作估计当前人体坐标系。
3. 使用 Gram-Schmidt、叉乘和 SVD 将测量坐标轴投影为合法的右手正交坐标系。
4. 生成并冻结本次遥操作会话使用的 `r_vr_to_robot_calibrated`。
5. 左右手共用同一套标定旋转矩阵。
6. 后续仍使用现有 reset/resume 相对位姿控制，不改成绝对位姿控制。
7. 检测 PICO tracking origin 跳变；检测到跳变后立即暂停遥操作、废弃当前标定并要求重新标定。
8. 标定过程中机器人不得跟随操作者的标定动作运动。
9. 不修改 FA IK、min-snap、奇异区滤波、fallback 和关节跳变限制等后端控制逻辑。

# 三、架构约束

## 3.1 保留 keypoint_transform.py 的职责

`keypoint_transform.py` 只负责：

* Unity/PICO 左手系到内部右手系转换；
* 手部关键点合法性检查；
* wrist 和手掌方向帧生成；
* 现有 Gram-Schmidt 正交化；
* 必要的输入平滑。

不要在该文件中硬编码 FA 机器人坐标系，也不要在该文件中实现 `H_R_V_FA` 或 FA 专用标定状态机。

FA 专用标定应位于 Operator 层。

## 3.2 尽量避免复制 XArmOperator 的整套 retargeting

`FaOperator` 当前继承 `XArmOperator`。请采用最小侵入式设计，例如：

* 在 `XArmOperator` 中增加受保护的旋转映射获取接口；
* 或支持可选的运行时映射矩阵 override；
* 或将“VR 到机器人旋转矩阵”封装为 getter。

示意：

```python
def _get_r_vr_to_robot(self) -> np.ndarray:
    return self._r_vr_to_robot
```

FA 可以覆盖该接口，返回标定成功后被冻结的矩阵。

必须保证：

* 非 FA 机器人行为不变；
* 关闭标定功能时 FA 行为与当前版本一致；
* 不复制粘贴整段平移、旋转 retargeting 代码到 `FaOperator`。

## 3.3 标定矩阵必须同时作用于平移和旋转

标定成功后，当前代码所有使用固定 `r_vr_to_robot` 的位置，均应通过同一个被冻结的：

```python
r_vr_to_robot_calibrated
```

完成映射。

平移：

```python
robot_translation_delta = (
    r_vr_to_robot_calibrated
    @ hand_translation_delta_vr
    * resolution_scale
)
```

旋转：

```python
robot_rotation_delta = (
    r_vr_to_robot_calibrated
    @ hand_rotation_delta_vr
    @ r_vr_to_robot_calibrated.T
)
```

继续保留当前：

```python
target_position = robot_init_position + robot_translation_delta
target_rotation = robot_rotation_delta @ robot_init_rotation
```

继续保留 `rotation_delta_frame = "base"` 的现有默认行为，同时保证 `"body"` 模式不被破坏。

# 四、标定数学定义

## 4.1 双手中心位置

对于同一个有效输入帧，取左右手腕在内部右手 VR 坐标系中的位置：

```python
p_left_vr
p_right_vr
```

计算双手中心：

```python
p_center_vr = 0.5 * (p_left_vr + p_right_vr)
```

标定方向主要由双手中心轨迹估计，同时保留左右手单独位移用于一致性校验。

## 4.2 向前动作

记录向前动作起点和终点采样窗口的均值：

```python
d_forward_vr = p_forward_end_mean - p_forward_start_mean
```

要求：

```python
norm(d_forward_vr) >= calibration_min_motion_distance_m
```

归一化：

```python
f_raw = normalize(d_forward_vr)
```

## 4.3 向左动作

同理：

```python
d_left_vr = p_left_end_mean - p_left_start_mean
l_raw = normalize(d_left_vr)
```

## 4.4 正交化

操作者动作不会严格正交，不要直接把两个原始向量拼成旋转矩阵。

先处理前向轴：

```python
f = normalize(f_raw)
```

从左向轴中移除前向分量：

```python
l_projected = l_raw - dot(l_raw, f) * f
l = normalize(l_projected)
```

根据右手系构造向上轴：

```python
u = normalize(cross(f, l))
```

重新计算左向轴以提高正交性：

```python
l = normalize(cross(u, f))
```

构造测量到的人体坐标轴在 VR 坐标系中的表达：

```python
E_measured_vr = column_stack([f, l, u])
```

列含义固定为：

```text
第 0 列：人体向前
第 1 列：人体向左
第 2 列：人体向上
```

必须检查：

```python
det(E_measured_vr) > 0
```

## 4.5 使用现有固定映射作为名义基准

不要在多个文件重复硬编码：

```text
forward = -vr_z
left    = -vr_x
up      =  vr_y
```

优先从当前代码实际使用的：

```python
r_vr_to_robot_nominal
```

推导名义人体坐标轴：

```python
E_nominal_vr = r_vr_to_robot_nominal.T
```

这里默认机器人 base 的：

```text
+x = 前
+y = 左
+z = 上
```

如果实际代码中的 FA pelvis/base 定义与此不同，以现有 `H_R_V_FA` 和文档为准，不要凭空改变坐标约定。

计算 VR 内部的标定修正：

```python
R_correction_vr = E_nominal_vr @ E_measured_vr.T
```

计算最终映射：

```python
R_calibrated_raw = (
    r_vr_to_robot_nominal
    @ R_correction_vr
)
```

当测量方向等于名义方向时，结果必须退化为：

```python
R_calibrated_raw == r_vr_to_robot_nominal
```

## 4.6 SVD 投影到 SO(3)

对最终矩阵做 SVD：

```python
U, _, Vt = np.linalg.svd(R_calibrated_raw)

D = np.eye(3)
D[2, 2] = np.linalg.det(U @ Vt)

r_vr_to_robot_calibrated = U @ D @ Vt
```

要求：

```python
R.T @ R ≈ I
det(R) ≈ +1
```

若正交误差、行列式或输入动作质量不满足要求，标定失败，不允许进入遥操作 READY 状态。

# 五、标定状态机

实现明确、可测试的状态机。名称可根据项目风格调整，但语义至少包括：

```text
DISABLED
CALIBRATION_REQUIRED
WAITING_STABLE_ORIGIN
CAPTURING_ORIGIN
CAPTURING_FORWARD
WAITING_RETURN_AFTER_FORWARD
CAPTURING_LEFT
VALIDATING
WAITING_RETURN_BEFORE_READY
READY
INVALIDATED
FAILED
```

建议流程：

```text
遥操作启动或重新使能
    ↓
CALIBRATION_REQUIRED
    ↓
双手稳定保持，采集原点窗口
    ↓
双手平行向前移动
    ↓
回到原点附近并稳定
    ↓
双手平行向左移动
    ↓
计算方向、正交化、SVD 和质量检查
    ↓
标定成功
    ↓
双手回到起点附近并短暂保持
    ↓
冻结 r_vr_to_robot_calibrated
    ↓
重新建立 hand_init_h 和 robot_init_h
    ↓
READY，开始发布机器人目标
```

重要要求：

1. 标定动作期间不得将手部动作映射为机器人运动。
2. 标定阶段保持机器人当前目标，或保持 Operator 暂停状态。
3. 标定成功后，不能沿用标定动作开始前的 `hand_init_h`。
4. 必须在标定成功后重新执行一次现有 reset/resume 基准建立流程。
5. 矩阵更新和状态切换必须原子化，避免控制线程读到一半更新的数据。
6. 标定失败后继续保持暂停，不得回退到未经用户确认的旧标定。
7. 每次重新开启遥操作均重新标定，不要默认从磁盘加载上一次标定结果。

# 六、标定触发和交互

先检查当前工程已有的：

* reset/resume 消息；
* VR 按钮事件；
* Operator 控制消息；
* 状态发布 topic；
* 日志或界面提示机制。

优先复用现有控制通道，不要随意增加一套不兼容的网络协议。

如果当前没有合适的标定控制事件，则增加清晰的命令接口，至少支持：

```text
start
capture_origin
capture_forward
capture_left
cancel
restart
```

也可以在保证可靠性的情况下实现半自动状态推进，但每个阶段必须有明确的状态输出，使用户知道下一步应该：

```text
保持双手不动
双手向前
回到中心
双手向左
标定成功或失败
```

不要依赖实时头显朝向来判断“前”和“左”。

# 七、采样和质量检查

不要只取单帧。

每个静止点和动作终点应采集一个短时间窗口并取均值或稳健统计量，例如中位数。所有阈值进入配置，不要散落魔法数字。

建议增加以下配置，字段名可按项目风格调整：

```python
enable_vr_axis_calibration: bool = True
require_calibration_each_enable: bool = True

calibration_sample_duration_s: float = 0.4
calibration_stable_dwell_s: float = 0.3
calibration_stable_position_epsilon_m: float = 0.02
calibration_ready_return_position_epsilon_m: float = 0.04
calibration_ready_return_dwell_s: float = 0.20

calibration_min_motion_distance_m: float = 0.12
calibration_max_motion_distance_m: float = 0.60

calibration_max_axis_abs_dot: float = 0.35
calibration_max_left_right_direction_error_deg: float = 20.0
calibration_max_left_right_distance_ratio_error: float = 0.35

calibration_rotation_orthogonality_tolerance: float = 1e-4
calibration_rotation_determinant_tolerance: float = 1e-4

tracking_origin_jump_detection_enabled: bool = True
tracking_origin_jump_translation_m: float = 0.15
tracking_origin_jump_rotation_deg: float = 15.0
tracking_origin_jump_confirm_frames: int = 2
tracking_origin_jump_interhand_change_m: float = 0.04
```

实际默认值可以根据现有帧率和数据噪声微调，但必须有配置说明。

标定有效性至少检查：

1. 左右控制器追踪数据有效，无 NaN/Inf。
2. 左右手时间戳满足同步要求。
3. 向前和向左的位移长度均超过最小阈值。
4. 位移不能异常大。
5. 向前和向左不能近似平行。
6. 正交化前：

```python
abs(dot(f_raw, l_raw)) <= calibration_max_axis_abs_dot
```

7. 左右手各自测得的运动方向应基本一致。
8. 左右手运动距离不能严重不一致。
9. 最终旋转矩阵必须属于 SO(3)。

对于失败情况，输出明确原因，例如：

```text
FORWARD_MOTION_TOO_SHORT
LEFT_MOTION_TOO_SHORT
AXES_NEARLY_COLLINEAR
FORWARD_HANDS_MISMATCH
LEFT_HANDS_MISMATCH
TRACKING_INVALID
SVD_PROJECTION_FAILED
TRACKING_ORIGIN_JUMP
```

# 八、PICO tracking origin 跳变检测

标定成功后，不允许持续使用头显姿态更新映射矩阵。

`r_vr_to_robot_calibrated` 必须保持冻结，直到：

* 用户主动重新标定；
* 遥操作关闭；
* tracking origin 跳变；
* 输入追踪长时间丢失；
* 其他现有安全状态要求 reset。

先检查 PICO/InputFrame 是否已经提供：

* tracking origin ID；
* recenter 事件；
* tracking session ID；
* head pose；
* SLAM relocalization 状态；
* tracking validity 标志。

检测优先级：

1. 有明确的 origin/recenter/session 事件时，直接使用。
2. 如果只有头显位姿，可仅将头显用于“坐标原点突变检测”，不能用作实时控制映射基准。
3. 如果没有上述信息，实现双手共同跳变检测作为 fallback。

双手共同跳变 fallback 的思路：

```python
delta_left = current_left - previous_left
delta_right = current_right - previous_right
common_delta = 0.5 * (delta_left + delta_right)
```

当以下条件同时满足时，可判定疑似 tracking origin 跳变：

* 左右手在极短时间内出现同方向的大位移；
* `norm(common_delta)` 超过阈值；
* 左右手相对距离变化很小；
* 连续若干帧确认，或伴随共同姿态突变；
* 变化速度明显超过合理人体运动范围。

必须避免因为操作者正常双手同步运动而频繁误触发，因此：

* 使用单帧大跳变阈值，而不是普通速度变化；
* 结合左右手相对几何关系；
* 使用可配置的确认帧数；
* 输出检测原因和测量值。

一旦确认 tracking origin 跳变：

```text
立即停止发布新的运动目标
保持机器人当前状态
清除 READY
废弃 r_vr_to_robot_calibrated
进入 INVALIDATED / CALIBRATION_REQUIRED
要求重新标定
```

不要在机器人运动过程中自动计算新矩阵并无缝切换，否则会造成末端目标突跳。

# 九、与 reset/resume 的关系

必须明确区分两个概念：

## 坐标轴标定

解决：

```text
操作者的前/左/上
        ↓
机器人 base 的 x/y/z
```

输出：

```python
r_vr_to_robot_calibrated
```

## 相对位姿基准

解决：

```text
当前手腕位置和姿态
        ↓
本次遥操作的零点
```

输出：

```text
hand_init_h
robot_init_h
```

推荐顺序：

```text
先完成坐标轴标定
    ↓
冻结 r_vr_to_robot_calibrated
    ↓
再执行 reset/resume 基准采集
    ↓
开始相对位姿控制
```

不要尝试仅通过重新设置 `hand_init_h` 解决坐标轴倾斜问题。位置零点只能消除平移偏置，不能消除“向左映射成前上/前下”的旋转耦合。

# 十、日志和可观测性

增加结构化日志或现有项目风格下的状态输出，至少包括：

```text
calibration_state
calibration_valid
calibration_failure_reason

forward_vector_vr
left_vector_vr
up_vector_vr

forward_motion_distance_left
forward_motion_distance_right
left_motion_distance_left
left_motion_distance_right

forward_left_dot
rotation_determinant
rotation_orthogonality_error

r_vr_to_robot_nominal
r_vr_to_robot_calibrated

tracking_origin_jump_detected
tracking_origin_jump_translation
```

不要每个控制周期刷屏。状态变化、失败和标定完成时打印详细信息即可。

# 十一、测试要求

为标定算法和 Operator 集成补充自动化测试。不要只依赖真机测试。

至少覆盖：

## 11.1 名义坐标测试

输入名义方向：

```python
forward_vr = [0, 0, -1]
left_vr = [-1, 0, 0]
up_vr = [0, 1, 0]
```

结果应与当前 `r_vr_to_robot_nominal` 一致。

## 11.2 已知旋转偏差测试

对名义人体坐标系施加已知 yaw、pitch、roll，例如：

```text
yaw = 25°
pitch = -20°
roll = 15°
```

使用旋转后的前、左方向作为标定输入。

验证：

```python
R_calibrated @ forward_measured ≈ robot_forward
R_calibrated @ left_measured ≈ robot_left
R_calibrated @ up_measured ≈ robot_up
```

## 11.3 噪声和非正交输入

在前、左轨迹中加入合理噪声，并让夹角不是严格 90°。

验证输出仍为合法旋转矩阵：

```python
R.T @ R ≈ I
det(R) ≈ 1
```

## 11.4 退化输入

覆盖：

* 位移太短；
* 前向和左向近似平行；
* 左右手方向不一致；
* 包含 NaN/Inf；
* 追踪超时。

这些情况必须失败，不能生成可用标定矩阵。

## 11.5 冻结行为

标定成功后继续输入变化的头显姿态或普通手部运动。

验证：

```python
r_vr_to_robot_calibrated
```

不会被逐帧更新。

## 11.6 平移映射

验证左右手都使用同一个标定矩阵，且：

```text
人体向前 -> 机器人 +x
人体向左 -> 机器人 +y
人体向上 -> 机器人 +z
```

## 11.7 旋转映射

在当前：

```python
rotation_delta_frame = "base"
```

模式下，验证旋转共轭映射使用标定矩阵，并且不会破坏原有 `target_rotation` 组合顺序。

同时保证 `"body"` 模式现有行为不回归。

## 11.8 reset/resume 顺序

验证：

* 未标定时不能进入正常遥操作；
* 标定动作不会驱动机器人；
* 标定完成后重新采集 `hand_init_h` 和 `robot_init_h`；
* 进入 READY 后第一帧目标不产生明显跳变。

## 11.9 tracking origin 跳变

构造左右手同时发生大幅共同跳变、但相对手间距离基本不变的数据。

验证：

* 当前标定失效；
* Operator 暂停；
* 不继续发布运动目标；
* 必须重新标定。

## 11.10 功能关闭兼容性

当：

```python
enable_vr_axis_calibration = False
```

时，验证 FA 使用现有固定 `H_R_V_FA`，行为与修改前一致。

# 十二、真机验收标准

实现后给出明确的真机测试步骤。

至少包含：

1. PICO4 正常佩戴标定。
2. PICO4 挂在脖子上，并人为制造明显 pitch/roll。
3. 双手平行向前运动。
4. 双手平行向左运动。
5. 双手平行向上运动。
6. 左右手分别运动。
7. 手腕进行绕 base 轴旋转。
8. 标定完成后晃动头显。
9. 主动触发 PICO recenter 或 tracking origin 变化。
10. reset/resume 后检查第一帧是否跳变。

建议验收指标：

```text
前、左、上三个方向映射角误差 <= 5°，最差不超过 10°
非目标轴位移耦合显著降低
左右手方向映射一致
标定完成后头显普通晃动不改变映射
tracking origin 跳变后机器人立即停止继续跟随
reset/resume 第一帧无明显目标跳变
```

还要区分：

* Operator 发布目标是否正确；
* FA control/IK/min-snap 是否因为可达性、奇异区、关节限位或安全策略修改了实际动作。

测试时同时记录：

```text
Operator 原始 CartesianTarget
FA control 处理后的目标或关节命令
机器人实际末端反馈
```

避免把后端 IK hold 或 min-snap 限制误判成标定失败。

# 十三、文档更新

完成代码后更新：

```text
docs/teleop/fa_coordinate_transform.md
```

新增至少以下内容：

* 为什么仅 reset/resume 零点不能解决坐标轴倾斜；
* 标定状态机；
* 双手向前和向左的标定流程；
* 正交化和 SVD 公式；
* 名义矩阵、修正矩阵、最终矩阵的关系；
* 标定矩阵如何作用于平移和旋转；
* 标定与 reset/resume 的执行顺序；
* tracking origin 跳变处理；
* 配置项说明；
* 排障方法。

检查：

```text
docs/teleop/operators.md
docs/teleop/index.md
```

现有入口链接是否仍然有效。没有必要时不要重复增加入口。

# 十四、实现原则

1. 先阅读实际代码，不要仅根据文档猜测类名、topic 或消息字段。
2. 遵循现有项目编码风格、类型标注、日志风格和配置方式。
3. 标定算法与 ROS/ZMQ/Operator I/O 解耦，便于单元测试。
4. 不在控制循环中反复分配大量对象。
5. 不在每帧执行不必要的 SVD；SVD 只在标定完成时执行。
6. 所有外部输入先检查长度、有限性、时间戳和追踪有效性。
7. 不改变当前 Unity 左手系到内部右手系转换。
8. 不改变当前手腕方向帧定义。
9. 不改变 FA IK、min-snap 和安全后端策略。
10. 不使用实时头显姿态驱动控制方向。
11. 不允许未标定状态静默退化为错误的固定映射，除非配置明确关闭标定。
12. 不允许 tracking origin 跳变后自动热切换新矩阵。

# 十五、最终输出要求

完成后请输出：

1. 当前架构理解和问题根因。
2. 修改过的文件列表。
3. 每个文件的主要修改。
4. 标定矩阵的数学定义。
5. 状态机说明。
6. 新增配置项及默认值。
7. 自动化测试列表和测试结果。
8. 真机测试步骤。
9. 兼容性和潜在风险。
10. 可以直接执行的测试、启动和格式检查命令。
11. 关键代码 diff 或完整实现。
12. 明确说明哪些内容未能在当前环境验证。

请直接实施，不要只给设计建议。不要在未检查现有代码的情况下重新实现一套平行的遥操作链路。
