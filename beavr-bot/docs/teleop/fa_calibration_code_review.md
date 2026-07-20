# FA 双臂标定代码梳理

当前标定不是独立流程，而是作为 `FaOperator` 每个遥操作控制周期前的一道门控。标定未完成时，原有 `XArmOperator` 重定向和机器人目标发布都不会执行。

整体调用链如下：

```
PICO 双手追踪
  ↓
Unity 发送左右手关键点
  ↓
pico4.py 接收原始关键点
  ↓
keypoint_transform.py
  ├─ 右手 transformed hand frame
  └─ 左手 transformed hand frame
  ↓
FaOperator._get_fa_calibration_sample()
  ↓
FaAxisCalibrationSession.update()
  ↓
起点 → 向前 → 回起点 → 向左 → 校验 → 宽松回起点
  ↓
compute_fa_axis_calibration()
  ↓
生成 r_vr_to_robot
  ↓
FaOperator 放行 XArmOperator 原有遥操作链路
  ↓
使用标定矩阵进行手腕位姿重定向
  ↓
发布机器人末端目标
```
## 1. 创建标定组件

FA 双臂配置会创建两个独立的 `FaOperator`：

```
fa_right_operator
fa_left_operator
```

配置入口在 [fa_config.py (line 422)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/configs/robots/fa_config.py:422)。

只有双臂模式启用坐标轴标定：

```python
enable_vr_axis_calibration=self.laterality == Laterality.BIMANUAL
```

两个 operator 都会创建自己的：

```python
FaAxisCalibrationSession(...)
```

并且都订阅左右手 `transformed hand frame`，见 [fa_operator.py (line 154)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/fa_operator.py:154)。

需要注意：左右臂各自拥有独立的标定状态机，并不是共享同一个 session。它们使用相同的双手输入，正常情况下会得到相同结果；只有右臂 operator 开启语音，避免两边重复播放。
## 2. 双手数据输入

PICO 原始关键点经过 [keypoint_transform.py (line 550)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/keypoint_transform.py:550)：

```
原始手部关键点
  → 左手坐标系转换
  → 计算手腕位置和手掌方向
  → 滑动平均
  → 正交化
  → InputFrame
```

其中：

```python
frame_vectors = [wrist, x_vec, y_vec, z_vec]
```

然后分别发布：

```
right_transformed_hand_frame
left_transformed_hand_frame
```

发布位置在 [keypoint_transform.py (line 631)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/keypoint_transform.py:631)。

标定只使用 `frame_vectors[0]`，也就是手腕绝对位置，不使用手掌朝向向量。
## 3. 遥操作启动触发标定

PICO 开启遥操作时发送：

```
Pause / High
```

`pico4.py` 将其转换为：

```python
SessionCommand(command="resume")
```

对应代码在 [pico4.py (line 167)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/pico4.py:167) 和 [pico4.py (line 736)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/pico4.py:736)。

`XArmOperator._get_arm_teleop_state()` 收到 `resume` 后返回 `ARM_TELEOP_CONT`，见 [xarm7_operator.py (line 750)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/xarm7_operator.py:750)。

随后每个控制周期进入：

```
XArmOperator._apply_retargeted_angles()
    ↓
FaOperator._before_retargeting_cycle()
```

调用点在 [xarm7_operator.py (line 976)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/xarm7_operator.py:976)。

因为配置了：

```python
require_calibration_each_enable = True
```

每次 Resume 都会执行：

```python
self._fa_axis_calibration.require_recalibration(...)
```

清除旧标定结果，重新开始。
## 4. 构造双手标定样本

`_before_retargeting_cycle()` 每周期调用：

```python
_get_fa_calibration_sample()
```

见 [fa_operator.py (line 300)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/fa_operator.py:300)。

处理顺序为：

```
非阻塞读取右手最新帧。
非阻塞读取左手最新帧。
检查左右帧是否都存在。
检查帧龄是否超过 1.0s。
检查时间戳偏差。
```

时间戳偏差超过 0.15s 时目前只记录日志，不阻断标定。最终构造：

```python
BimanualWristSample(
    timestamp_s=max(left_timestamp, right_timestamp),
    left=left_wrist,
    right=right_wrist,
)
```
## 5. 状态机推进

样本进入 [FaAxisCalibrationSession.update() (line 364)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/fa_axis_calibration.py:364)。

正常状态顺序是：

```
CALIBRATION_REQUIRED
  ↓
CAPTURING_ORIGIN
  ↓
CAPTURING_FORWARD
  ↓
WAITING_RETURN_AFTER_FORWARD
  ↓
CAPTURING_LEFT
  ↓
VALIDATING
  ↓
WAITING_RETURN_BEFORE_READY
  ↓
READY
```

具体动作是：

- 双手在起点保持，采集 0.4s，中点波动不能超过 2cm。
- 双手向前移动至少 6cm，采集 0.4s。
- 双手回到起点 2cm 范围内并保持 0.3s。
- 双手向左移动至少 6cm，采集 0.4s。
- 解算并验证坐标轴。
- 左右手腕分别回到各自起点 4cm 范围内，并保持 0.2s。
- 进入 READY。

移动太短、向前和向左过于接近时，会进入对应的“先回起点，再重试”状态。
## 6. 解算标定矩阵

完成左移动作后调用：

```python
compute_fa_axis_calibration(...)
```

入口在 [fa_axis_calibration.py (line 173)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/fa_axis_calibration.py:173)。

计算过程是：

```
起点双手中点
向前双手中点
向左双手中点
  ↓
forward_delta / left_delta
  ↓
距离和双手一致性检查
  ↓
Gram-Schmidt 正交化
  ↓
构造 forward / left / up 三轴
  ↓
结合 nominal 轴映射计算校正
  ↓
SVD 投影到合法 SO(3) 旋转矩阵
  ↓
r_vr_to_robot
```

当前关键限制：

- 前向和左向距离：6cm～60cm
- 前向与左向夹角：约 69.5°～110.5°
- 单手相对整体方向偏差：不超过 20°
- 较短一侧移动距离：至少为较长一侧的 65%
## 7. 标定结果开始生效

第一次进入 READY 时，[fa_operator.py (line 289)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/fa_operator.py:289) 会：

```python
self.is_first_frame = True
self._ignore_hand_frames_before_s = time.time()
self._clear_hand_tracking_cache()
return False
```

也就是说，READY 的当前周期仍不启动机器人。它先清除标定期间积累的手部控制帧。

下一个周期才返回 `True`，父类重新建立：

```
当前机器人末端位姿
+
标定完成后的新鲜手腕基准
```

之后父类调用 [fa_operator.py (line 223)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/fa_operator.py:223) 获取标定矩阵：

```python
r_vr_to_robot = calibration.result.r_vr_to_robot
```

并在 [xarm7_operator.py (line 1097)](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/xarm7_operator.py:1097) 中用于：

```python
robot_translation = r_vr_to_robot @ hand_translation_delta_vr
robot_rotation = r_vr_to_robot @ hand_rotation_delta_vr @ r_robot_to_vr
```

随后继续走原来的滤波、末端目标计算和发布链路。
## 8. PICO 语音链路

状态变化后：

```
FaAxisCalibrationState
  ↓
_fa_calibration_audio_key_for_state()
  ↓
_play_fa_calibration_audio()
  ↓
ZMQStringPublisher
  ↓
tcp://*:8112
topic = fa_calibration_prompt
payload = fa_calib_xxx
```

PICO 的 [CalibrationPromptReceiver.cs (line 92)](/home/likunwei/dataCollection/BeaVR-app/BeaVR-Unity/Assets/Scripts/Audio/CalibrationPromptReceiver.cs:92) 接收后，在 Unity 主线程加载：

```
Resources/Audio/FaCalibration/<prompt_key>.wav
```

并调用 `AndroidTextToSpeech.PlayVoiceClipResource()` 播放。
## 9. READY 后的持续监控

标定完成后并不是不再读取标定帧。每周期仍会把双手样本送入状态机，用来检测 PICO 追踪原点整体跳变。

检测到左右手连续同步瞬移后：

```
READY → INVALIDATED
```

此时立即：

- 清除标定矩阵
- 清除遥操作基准
- 停止发布新目标
- 等待下一次 Resume 重新标定

因此当前完整关系可以概括为：**Resume 触发标定，标定状态机控制父类遥操作是否放行，标定矩阵只替换原有重定向中的 VR 到机器人旋转映射。**
