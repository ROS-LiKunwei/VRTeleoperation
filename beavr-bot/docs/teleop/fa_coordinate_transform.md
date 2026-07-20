# FA 遥操作坐标变换逻辑

本文按当前代码路径总结 FA 双臂遥操作中的坐标变换逻辑。核心路径是：

```text
PICO/Unity InputFrame
  -> keypoint_transform.py
  -> fa_operator.py / xarm7_operator.py
  -> fa_real_control.py
  -> FA IK / min_snap / upper-body command
```

## 1. PICO/Unity 原始手部坐标

Unity/PICO 端发来的 `InputFrame.keypoints` 是 26 个手部关节点，每个关节点 3 个坐标，共 78 个数。`keypoint_transform.py` 先把它 reshape 成 `(26, 3)`，并检查数量、NaN/Inf、关键点是否退化。

Unity 使用左手系：`+X` 向右，`+Y` 向上，`+Z` 向前。BeaVR 后端内部使用右手 VR 坐标系：`+X` 向右，`+Y` 向上，`+Z` 向后。因此当前代码会做一次固定翻转：

```text
internal_xyz = unity_xyz * [1, 1, -1]
```

这一步只发生在 `TransformHandPositionCoords.transform_keypoints()` 中，属于 VR 输入标准化，不是 FA 机器人坐标系映射。

## 2. 手腕方向帧生成

`keypoint_transform.py` 对每只手发布两类数据：

- `*_transformed_hand_coords`：以手腕为原点的相对关键点。
- `*_transformed_hand_frame`：下游 Operator 真正用于手臂 retarget 的方向帧。

FA 手臂位姿主要使用 `frame_vectors`，格式是：

```text
[wrist, x_vec, y_vec, z_vec]
```

其中 `wrist` 是手腕在内部 VR 坐标系中的绝对位置，三个方向向量由手腕、食指掌指、中指掌指、小指掌指关节计算得到。左手会翻转 palm normal，使左右手最终都遵守同一套右手方向帧约定。

方向帧随后经过滑动平均和 Gram-Schmidt 正交化，避免手掌关键点抖动导致基向量不正交。当前 FA 配置里 transform 的 `moving_average_limit=1`，也就是默认不做多帧平滑。

## 3. FA Operator 继承 XArm retargeting

FA 没有单独实现一套完整的坐标变换算法。`FaOperator` 继承 `XArmOperator`，主要差异是传入 FA 自己的 VR 到机器人基座映射矩阵 `H_R_V_FA`：

```text
H_R_V_FA =
  [[ 0, -1,  0, 0],
   [ 0,  0,  1, 0],
   [-1,  0,  0, 0],
   [ 0,  0,  0, 1]]
```

代码注释里说明 FA 的 IK 参考基座是 `pelvis`，方向约定与 SYSMO-32 base frame 一致，因此复用这套 VR-to-robot 映射。

只看旋转部分，当前映射关系可以理解为：

```text
robot_x = -vr_z
robot_y = -vr_x
robot_z =  vr_y
```

Operator 内部实际用的是 `r_vr_to_robot = inverse(H_R_V_FA[:3, :3])`，把 VR 中的手部位移和旋转增量转换到机器人 base/pelvis 坐标系。

## 4. Reset / Resume 建立相对运动基准

FA 遥操作不是把手腕绝对位置直接发给机器人，而是做相对 retarget。

Operator 在 reset 或 resume 后会建立两组基准：

- `robot_init_h`：当时机器人末端的 4x4 齐次矩阵，由下游 FA control 通过 `endeff_homo` 返回。
- `hand_init_h`：当时操作者手腕方向帧转换成的 4x4 齐次矩阵。

正常运行时，每一帧会取当前手部方向帧生成：

```text
hand_moving_h
```

然后计算从初始手部姿态到当前手部姿态的相对变化：

```text
h_ht_hi = inverse(hand_init_h) @ hand_moving_h
hand_translation_delta_vr = hand_moving_h.translation - hand_init_h.translation
```

因此操作者在 resume 那一刻的手姿态就是新的零点。之后手腕移动多少、手掌旋转多少，才会映射成机器人末端相对 `robot_init_h` 的目标变化。

## 5. 平移映射

平移增量始终用 VR 坐标系中的手腕位置差计算：

```text
hand_translation_delta_vr = current_wrist_vr - init_wrist_vr
robot_translation_delta =
    r_vr_to_robot @ hand_translation_delta_vr * resolution_scale
```

FA 当前配置中：

```text
translation_scale = 1.0
high_resolution_translation_scale = 1.0
low_resolution_translation_scale = 1.0
```

`FaOperator` 会把这些值写入 `resolution_scale`。如果后续接入按钮切换高/低精度，平移比例也是通过 `resolution_scale` 生效。

最终目标位置不是用矩阵整体左乘去转动初始点，而是显式解耦：

```text
target_position = robot_init_position + robot_translation_delta
```

这样手腕姿态变化不会把机器人初始末端位置绕 base 旋转一圈。

## 6. 旋转映射

旋转增量由 `rotation_delta_frame` 控制。当前 `FaOperatorCfg` 默认值是：

```text
rotation_delta_frame = "base"
```

两种模式含义不同：

- `base`：使用 `hand_moving_R @ hand_init_R.T`，表示在 VR 世界/base 坐标系下看的姿态变化。
- `body`：使用 `inverse(hand_init_h) @ hand_moving_h` 的旋转部分，表示在手自身初始坐标系下看的姿态变化。

当前 FA 默认走 `base`。随后旋转增量被映射到机器人坐标系：

```text
robot_rotation_delta =
    r_vr_to_robot @ hand_rotation_delta_vr @ r_robot_to_vr
```

最终目标姿态同样与平移解耦：

```text
target_rotation = robot_rotation_delta @ robot_init_rotation
```

输出前会用 SVD 投影回合法旋转矩阵，避免数值漂移造成非法 SO(3) 矩阵。

## 7. 发布给 FA 控制层

Operator 把 4x4 目标矩阵转成 7D 笛卡尔目标：

```text
[x, y, z, qx, qy, qz, qw]
```

并发布 `CartesianTarget`：

```text
topic = "endeff_coords"
frame_id = "base"
hand_side = "left" 或 "right"
position_m = 目标位置
orientation_xyzw = 目标四元数
hand_command = 抓取/松开命令
```

`fa_real_control.py` 订阅左右手目标后，做 FA IK、可达性 fallback、奇异区滤波、关节跳变限制等安全处理，最后按配置发布到 `/min_snap/target` 或原生上半身控制 topic。

## 8. 当前关键配置入口

主要代码入口：

- `src/beavr/teleop/components/detector/vr/keypoint_transform.py`
- `src/beavr/teleop/components/operator/robots/fa_operator.py`
- `src/beavr/teleop/components/operator/robots/xarm7_operator.py`
- `src/beavr/teleop/configs/robots/fa_config.py`

当前 FA 坐标变换相关默认值在 `FaOperatorCfg`：

```text
moving_average_limit = 1
hand_frame_timeout_s = 1.0
rotation_delta_frame = "base"
translation_scale = 1.0
high_resolution_translation_scale = 1.0
low_resolution_translation_scale = 1.0
post_resume_stable_position_epsilon_m = 0.03
post_resume_stable_orientation_epsilon_rad = 0.20
post_resume_stable_dwell_s = 0.2
use_filter = False
```

## 9. 排查方向问题时看哪里

如果出现“平移方向反了”“手腕旋转方向不符合直觉”“向某个方向动不了”，优先按这个顺序查：

1. `keypoint_transform.py` 是否收到了正确的手侧、关键点数量和 Unity 左手系到内部右手系转换后的 wrist 坐标。
2. `fa_operator.py` 的 `H_R_V_FA` 是否符合当前 FA base/pelvis 坐标定义。
3. `FaOperatorCfg.rotation_delta_frame` 是 `base` 还是 `body`，因为这会改变旋转增量的参考坐标系。
4. `translation_scale` / `resolution_scale` 是否被运行时按钮消息改写。
5. `fa_real_control.py` 后端 IK、fallback、奇异区滤波、关节跳变限制是否改变或 hold 了 Operator 发出的笛卡尔目标。

Operator 只负责发布笛卡尔目标；真实机器人最终是否到达该目标，还会受到 IK 可达性、关节限位、min-snap 频率和安全滤波影响。

## 10. 双手方向标定流程

当前代码使用固定的 `H_R_V_FA` 做 VR 到 FA base/pelvis 的方向映射。如果 PICO4 没有正常佩戴，而是挂在脖子上或以其他姿态固定，PICO tracking/world 坐标系可能相对操作者人体方向发生 pitch、roll 或 yaw 偏转。这时仅靠 reset/resume 重置 `hand_init_h` 只能消除位置零点偏差，不能消除“操作者向左移动，机器人却耦合出前上/前下运动”的坐标轴倾斜问题。

为解决这个问题，FA 遥操作应在每次启动或重新使能时执行一次双手方向标定。标定只估计“操作者人体前/左/上”到“FA 机器人 base/pelvis x/y/z”的旋转关系，不改变后续相对位姿控制方式。

当前实现位置：

- `fa_axis_calibration.py`：标定数学、状态机、质量检查和 tracking origin 跳变检测。
- `fa_operator.py`：订阅左右手 `*_transformed_hand_frame`，在 Operator 层运行标定状态机，READY 前阻断 `endeff_coords` 发布。
- `xarm7_operator.py`：提供 `_get_r_vr_to_robot()` 和 `_before_retargeting_cycle()` 受保护接口，非 FA 机器人默认行为不变。
- `fa_config.py`：暴露标定配置项，FA 双臂启用，FA 单臂关闭。

### 10.1 标定交互步骤

标定阶段机器人不得跟随手部动作运动。推荐流程如下：

```text
启动或重新使能 FA 遥操作
  -> 暂停发布 endeff_coords
  -> 双手自然放在身体前方并保持不动，采集原点窗口
  -> 双手平行向前移动并保持，采集 forward 终点窗口
  -> 双手回到原点附近并保持
  -> 双手平行向左移动并保持，采集 left 终点窗口
  -> 计算并验证 r_vr_to_robot_calibrated
  -> 标定成功后冻结矩阵
  -> 双手回到起点附近并短暂保持
  -> 重新执行现有 reset/resume，建立 hand_init_h 和 robot_init_h
  -> READY，开始发布机器人目标
```

每个采样点都应采集一个短时间窗口，例如 `0.4 s`，取均值或中位数，不使用单帧数据。左右手必须同时有效，时间戳需要满足同步要求。

### 10.2 双手中心轨迹

同一有效输入帧中取左右手 wrist 位置：

```text
p_left_vr
p_right_vr
```

双手中心为：

```text
p_center_vr = 0.5 * (p_left_vr + p_right_vr)
```

向前动作的方向由起点窗口和终点窗口中心均值计算：

```text
d_forward_vr = p_forward_end_mean - p_forward_start_mean
f_raw = normalize(d_forward_vr)
```

向左动作同理：

```text
d_left_vr = p_left_end_mean - p_left_start_mean
l_raw = normalize(d_left_vr)
```

左右手各自的位移也需要保留，用于检查两只手是否真的做了同方向、相近距离的平移动作。

### 10.3 正交化和 SVD

操作者动作不会严格正交，不能直接把 `f_raw` 和 `l_raw` 拼成旋转矩阵。先用 Gram-Schmidt 从左向轴中去掉前向分量：

```text
f = normalize(f_raw)
l_projected = l_raw - dot(l_raw, f) * f
l = normalize(l_projected)
u = normalize(cross(f, l))
l = normalize(cross(u, f))
```

测量得到的人体坐标轴在 VR 坐标系中的表达为：

```text
E_measured_vr = column_stack([f, l, u])
```

列含义固定为：

```text
第 0 列：人体向前
第 1 列：人体向左
第 2 列：人体向上
```

名义坐标轴从当前固定映射推导，不要在多个文件重复手写轴关系：

```text
r_vr_to_robot_nominal = inverse(H_R_V_FA[:3, :3])
E_nominal_vr = r_vr_to_robot_nominal.T
```

VR 内部修正矩阵和最终映射为：

```text
R_correction_vr = E_nominal_vr @ E_measured_vr.T
R_calibrated_raw = r_vr_to_robot_nominal @ R_correction_vr
```

当测量方向等于名义方向时，`R_calibrated_raw` 应退化为当前固定的 `r_vr_to_robot_nominal`。

最终矩阵需要投影到合法 SO(3)：

```text
U, _, Vt = svd(R_calibrated_raw)
D = diag(1, 1, det(U @ Vt))
r_vr_to_robot_calibrated = U @ D @ Vt
```

要求：

```text
R.T @ R ~= I
det(R) ~= +1
```

标定成功后，`r_vr_to_robot_calibrated` 在本次遥操作会话内冻结，左右手共用同一个矩阵。

### 10.4 标定矩阵如何进入现有 retargeting

标定矩阵必须同时作用于平移和旋转。原来的固定 `r_vr_to_robot` 应替换为冻结后的：

```text
r_vr_to_robot_calibrated
```

平移：

```text
robot_translation_delta =
    r_vr_to_robot_calibrated @ hand_translation_delta_vr * resolution_scale
```

旋转：

```text
robot_rotation_delta =
    r_vr_to_robot_calibrated
    @ hand_rotation_delta_vr
    @ r_vr_to_robot_calibrated.T
```

后续仍保留当前解耦逻辑：

```text
target_position = robot_init_position + robot_translation_delta
target_rotation = robot_rotation_delta @ robot_init_rotation
```

`rotation_delta_frame = "base"` 的默认行为继续保留；如果配置为 `"body"`，只改变 `hand_rotation_delta_vr` 的来源，不改变标定矩阵的使用方式。

### 10.5 与 reset/resume 的顺序

坐标轴标定和相对位姿基准是两件事：

```text
坐标轴标定：
    操作者前/左/上 -> 机器人 base x/y/z
    输出 r_vr_to_robot_calibrated

相对位姿基准：
    当前手腕位置/姿态 -> 本次遥操作零点
    输出 hand_init_h 和 robot_init_h
```

正确顺序是：

```text
先完成坐标轴标定
  -> 冻结 r_vr_to_robot_calibrated
  -> 再执行 reset/resume 基准采集
  -> 开始相对位姿控制
```

标定成功后不能沿用标定动作开始前的 `hand_init_h`，否则标定动作本身可能残留成机器人目标偏移。

### 10.6 状态机

当前实现使用以下状态机语义：

```text
DISABLED
CALIBRATION_REQUIRED
WAITING_STABLE_ORIGIN
CAPTURING_ORIGIN
CAPTURING_FORWARD
WAITING_RETURN_AFTER_FORWARD
CAPTURING_LEFT
VALIDATING
READY
INVALIDATED
FAILED
```

状态切换要求：

- 标定动作期间不发布新的机器人运动目标。
- 标定失败后保持暂停，不静默回退到旧标定。
- 每次重新开启遥操作默认重新标定。
- 矩阵更新和 `READY` 切换必须原子化，避免控制循环读到半更新状态。
- 标定成功后触发现有 reset/resume 流程，让当前手姿态成为新的 `hand_init_h`。

### 10.7 质量检查和失败原因

当前实现配置项：

```text
enable_vr_axis_calibration = True
require_calibration_each_enable = True
calibration_sample_duration_s = 0.4
calibration_stable_dwell_s = 0.3
calibration_stable_position_epsilon_m = 0.02
calibration_ready_return_position_epsilon_m = 0.04
calibration_ready_return_dwell_s = 0.20
calibration_min_motion_distance_m = 0.12
calibration_max_motion_distance_m = 0.60
calibration_max_axis_abs_dot = 0.35
calibration_max_left_right_direction_error_deg = 20.0
calibration_max_left_right_distance_ratio_error = 0.35
calibration_rotation_orthogonality_tolerance = 1e-4
calibration_rotation_determinant_tolerance = 1e-4
calibration_max_timestamp_skew_s = 0.15
tracking_origin_jump_detection_enabled = True
tracking_origin_jump_translation_m = 0.15
tracking_origin_jump_rotation_deg = 15.0
tracking_origin_jump_confirm_frames = 2
tracking_origin_jump_interhand_change_m = 0.04
```

至少检查：

- 左右手数据有效，无 NaN/Inf。
- 左右手时间戳同步。
- forward 和 left 位移均超过最小距离，且不超过最大距离。
- `abs(dot(f_raw, l_raw))` 不超过阈值，避免两轴近似平行。
- 左右手各自测得的运动方向基本一致。
- 左右手运动距离不能严重不一致。
- 最终旋转矩阵正交，且行列式接近 `+1`。

失败原因应写入日志或状态输出，例如：

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

### 10.8 Tracking Origin 跳变处理

标定成功后不要用头显姿态实时更新控制方向。`r_vr_to_robot_calibrated` 必须保持冻结，直到用户重新标定、遥操作关闭、追踪丢失或检测到 PICO tracking origin 跳变。

优先使用 PICO/InputFrame 中明确的 origin、recenter、session 或 relocalization 事件。如果没有这些字段，可以用双手共同跳变作为 fallback：

```text
delta_left = current_left - previous_left
delta_right = current_right - previous_right
common_delta = 0.5 * (delta_left + delta_right)
```

疑似跳变需要满足：

- 左右手在极短时间内出现同方向大位移。
- `norm(common_delta)` 超过阈值。
- 左右手相对距离变化很小。
- 连续若干帧确认。
- 变化速度明显超过合理人体运动范围。

确认跳变后：

```text
立即停止发布新的运动目标
保持机器人当前状态
清除 READY
废弃 r_vr_to_robot_calibrated
进入 INVALIDATED / CALIBRATION_REQUIRED
要求重新标定
```

不要在机器人运动中自动计算新矩阵并热切换，否则末端目标会突跳。
