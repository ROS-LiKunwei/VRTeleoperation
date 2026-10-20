# AHA-WAM：FA 人工接管与实机数据连续采集

本文描述 FA 侧已经接入的两项能力：模型失败后的人工接管，以及成功/失败实机
episode 的连续保存。SO101 的对应实现与运行参数在
`HARP-3D-WAM/docs/SO101_REAL_TEST_DATA_AND_INTERVENTION_ZH.md`。

> 当前只完成了静态检查、mock 闭环和编码/解码测试。大脑、小脑和实机未启动，
> 因而 MinSnap 覆盖、PICO 接管、ROS 发现拓扑及长时间磁盘吞吐仍需后续实机验收。

## 控制权设计

FA 的模型客户端和原 VR 遥操作都会发布 `/min_snap/target`。为避免两个发布者争抢
机器人，接管模式不允许 VR 遥操作直接写实机话题：

```text
模型 action --------------------------\
                                        AHA-WAM 唯一命令网关
PICO -> FA IK/限幅 -> human candidate --+--> /min_snap/target
                                        +--> /left_topic_to_hand
新鲜 /joint_states -> one-shot hold ----/    /right_topic_to_hand
```

VR 候选话题固定为：

- `/ahawam/fa/human/min_snap_target`
- `/ahawam/fa/human/left_hand`
- `/ahawam/fa/human/right_hand`

`FaRealControl` 必须用 `--ahawam_intervention_mode=true` 启动。该模式会关闭其启动
初始姿态和初始手爪命令，但保留现有 IK、奇异区处理、fallback 和限幅器。AHA 客户端
在允许运动前以及每次采集 observation 时都会检查：三个实机话题各只有它自己一个
发布者，而且启用人工介入时三条候选话题均恰有一个发布者。启动后出现旁路写入、候选
源重复或消失也会立即 fail-closed。

控制状态只有四种：

- `AUTO`：只允许模型命令。
- `SAFE_HOLD`：取消剩余模型 chunk，以最新完整 14 关节反馈发布一次保持目标；夹爪
  保持当前命令 latch。
- `HUMAN`：只转发接管边界之后的新鲜 PICO 候选，接管之前积压的候选会被丢弃。
- `FINALIZED`：拒绝后续模型/人工命令，只允许退出时保持。

模型 socket/返回格式/NaN/首目标跳变，以及服务端 active recovery 的
`operator_intervention_required=true`，可以进入 `SAFE_HOLD -> HUMAN`。相机、完整
关节反馈、ROS 发布、录制器或磁盘故障会 fail closed，不能继续人工运动。

## 操作键

在 AHA 客户端所在的交互终端：

- `i`：人工请求接管。
- `s`：episode 标注为 success。
- `f`：episode 标注为 failure。
- `q` 或 `Esc`：中止 episode。

模型自动失败也会进入人工模式。人工修复后按 `s`，保存为
`outcome=success, assisted=true`；介入与任务结果是两个独立字段。

## 后续实机运行命令

以下命令留待大脑、小脑和实机全部打开后人工执行。先启动 PICO/VR 候选源，再启动
AHA 客户端；不要同时启动普通 FA 实机遥操作。

终端 A（现有 FA PICO/VR 链，只输出候选）：

```bash
cd /home/likunwei/dataCollection/beavr-bot
source /opt/ros/humble/setup.bash
source /home/likunwei/humanoid_ws/install/setup.bash
source .venv/bin/activate
PYTHONPATH=src python -m beavr.teleop.main \
  --robot_name=fa \
  --laterality=bimanual \
  --control_backend=real \
  --ahawam_intervention_mode=true
```

终端 B（一个进程对应一个 episode）：

```bash
cd /home/likunwei/dataCollection/beavr-bot
source /opt/ros/humble/setup.bash
source /home/likunwei/humanoid_ws/install/setup.bash
source .venv/bin/activate
read -r -p "请输入 AHA 大脑 IP: " AHA_BRAIN_IP
test -n "$AHA_BRAIN_IP" || { echo "AHA 大脑 IP 不能为空" >&2; exit 1; }
PYTHONPATH=src python -m beavr.scripts.ahawam_fa_async_client \
  --host "$AHA_BRAIN_IP" \
  --port 10000 \
  --camera-settings configs/cameras/sysmo32_rgbd_aligned_native.json \
  --enable-motion \
  --confirm-motion RUN \
  --enable-operator-intervention \
  --require-outcome-label \
  --record-dir /home/likunwei/real_rollouts/fa
```

默认连续录制为 10 Hz，因为两路 FA wrist 相机的原生 profile 是 10 Hz。不要为了得到
30 Hz 文件而把重复图像伪装成新采样；每路相机的 sequence、采集时间和与关节状态的
skew 都保存在逐帧记录中。

## 数据结构与语义

每次运行先写：

```text
<record-dir>/.in_progress/<uuid>/
```

三路 H.264、逐帧 JSONL、事件、metadata 和 manifest 全部关闭并校验后，目录才会原子
移动到：

```text
<record-dir>/episodes/<uuid>/
<record-dir>/episodes.jsonl
```

只有带 `COMMITTED` 的目录可进入下游。编码、队列、磁盘、shape/dtype、时间因果或
manifest 校验失败时不会提交，staging 会保留用于排障。

逐帧原始记录包含：

- 三路原始 RGB 与每路时间戳/sequence；
- 16D state：左臂 7 + 右臂 7 + 左右夹爪命令 latch；
- 最后一个在该图像采集时刻之前成功 publish 的 `sent_command`、来源和回执；
- `command_publication`：完整/部分 ROS 写入、来源以及“输出可能已改变”的保守证据；
- 控制状态、介入事件以及最终 success/failure/aborted/error/unlabeled。

`publish()` 返回只证明命令进入 ROS 发布路径，不能证明机械臂已执行。因此当前 FA
没有硬件执行 ACK 时，发生任务命令发布后 `motion_executed=null`，并保留
`physical_execution_verified=false`；未启用运动或没有任何任务输出证据时才写
`motion_executed=false`。不得把 `command_publication` 当作实机轨迹执行确认。即使三话题
中途部分写入，数据也会记为 error，并保守记录“可能已产生运动”。

注意 FA 旧训练数据的 action 契约不是发送目标，而是下一帧实测状态：

```text
observation.state[t] = state[t]
action[t]            = state[t + 1]
```

因此 `sent_command` 只用于控制审计；后续导出 LeRobot 训练集时必须用相邻 measured
state 构造 action，不能直接把 `sent_command` 混入旧数据。state 最后两维目前也是
客户端命令 latch，不是手爪传感器反馈，metadata 已显式记录这一点。

汇总与严格校验：

```bash
cd /home/likunwei/dataCollection/beavr-bot
PYTHONPATH=src python -m beavr.scripts.summarize_fa_real_rollouts \
  /home/likunwei/real_rollouts/fa --strict
```

## 实机验收顺序

1. 只读确认三相机 profile、完整 14 关节 freshness/时钟/skew 和 ROS 发布者拓扑。
2. 不发模型动作，确认连续录制、三视频解码、JSONL/manifest 和磁盘吞吐。
3. 低风险姿态下验证一次 measured hold，确认旧 MinSnap 轨迹被覆盖。
4. 验证 `i` 后只有接管后的 PICO 候选能到实机，并核对夹爪 `20=关、21=开`。
5. 分别完成 autonomous success、assisted success、failure、abort、故障 staging。
6. 长时间运行检查有界队列、剩余磁盘、视频帧数和退出清理。
