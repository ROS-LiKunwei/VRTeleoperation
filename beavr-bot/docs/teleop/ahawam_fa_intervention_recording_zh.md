# AHA-WAM：FA 人工接管与实机数据连续采集

本文描述 FA 侧已经接入的两项能力：模型失败后的人工接管，以及成功/失败实机
episode 的连续保存。SO101 的对应实现与运行参数在
`HARP-3D-WAM/docs/SO101_REAL_TEST_DATA_AND_INTERVENTION_ZH.md`。

> 2026-10-23 已在 FA 实机端完成禁用运动的 RGB-D 严格预检：5 Hz 连续采集 163 帧，
> 三路 RGB H.264 与三路无损深度均通过完整解码校验，episode 正常提交。该预检不包含
> MinSnap 运动、PICO 接管或任务成功/失败标注，仍需按下文顺序继续验收。

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
- `HUMAN`：保留启动后已经完成的 VR 坐标轴标定，只让左右 Operator 分别把介入时的
  当前手姿映射到双臂当前实测位姿，完成 `Hand_Init_H` 相对零点重基准；收到双臂完成
  ACK 后，才转发接管边界之后的新鲜 PICO
  候选。标定期间以及接管之前积压的候选都会被丢弃。
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

启动遥操作后先完成一次 VR 坐标轴标定，然后等待人工介入。按 `i` 后终端会先显示
`SAFE_HOLD`；此时机械臂保持在介入瞬间的实测位置，双手短暂保持稳定即可，不会再次
要求执行 VR 坐标轴标定，也不需要把手移回模型启动前的位置。只有终端显示
`HUMAN active ... both arms now use their current poses as origins` 后，双臂运动才会作为
人工候选通过 AHA 网关下发。若 120 秒内任意一侧未完成重基准，客户端超时退出并继续
保持，不会使用旧的人手/机器人映射。

重基准会取消边界之前仍在后台计算的 IK，并给 IK 结果绑定代次，旧代次结果即使稍后
完成也会被丢弃。进入 HUMAN 后第一条新鲜手臂目标还必须与当前实测关节位置相差不超过
`0.08 rad`；超过阈值会 fail-closed 并保持当前位置，不能靠逐步限幅继续走向旧基准。

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
  --record-fps 5 \
  --max-camera-age-ms 300 \
  --max-camera-skew-ms 150 \
  --max-image-state-skew-ms 150 \
  --max-state-age-ms 200 \
  --observation-timeout-s 10 \
  --enable-motion \
  --confirm-motion RUN \
  --enable-operator-intervention \
  --require-outcome-label \
  --record-dir /home/likunwei/real_rollouts/fa
```

两路 FA wrist 相机的原生 profile 是 10 Hz，但当前实机偶发约 0.45 秒的取流停顿，
严格 10 Hz 数据连续性预检会据实失败。当前实机验收过的连续录制频率是 5 Hz；不要
为了得到更高帧率而复制图像或放宽同步安全阈值。每路相机的 sequence、采集时间和与
关节状态的 skew 都保存在逐帧记录中。短暂 stale/missing/skew 会在 10 秒观测期限内
等待整组三路新鲜帧；整组绝对年龄仍超过 300 ms 时不会进入推理或录制。

## 数据结构与语义

每次运行先写：

```text
<record-dir>/.in_progress/<uuid>/
```

三路 RGB H.264、三路无损深度 Blosc-Zstd/u16 分段流、逐帧 JSONL、事件、metadata 和
manifest 全部关闭并校验后，目录才会原子
移动到：

实时观测与编码采用进程隔离：相机/ROS 主进程只把一份同步 RGB-D 快照复制到固定
共享内存，独立的 `spawn` 子进程负责 RGB 硬编码、深度压缩、哈希、JSONL、分段、
落盘和最终校验。共享内存槽在子进程确认复制完成前不会复用；编码队列满、子进程
退出或 IPC 超时都会 fail-closed，不会静默丢帧。

```text
<record-dir>/episodes/<uuid>/
<record-dir>/episodes.jsonl
```

只有带 `COMMITTED` 的目录可进入下游。编码、队列、磁盘、shape/dtype、时间因果或
manifest 校验失败时不会提交，staging 会保留用于排障。

逐帧原始记录包含：

- 三路 RGB，以及与 RGB 对齐的三路 `uint16` 毫米深度；
- 深度以原始毫米 `uint16` 写入逐帧定界的 Blosc-Zstd 流（level 1 + bitshuffle），解码后逐帧 SHA256 校验，数值完全可逆；
  提交前会完整解码，并用逐帧 SHA-256 验证能精确恢复原始毫米值；
- 每路相机的采集时间戳、sequence、图像/关节 skew；
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
2. 不发模型动作，确认三路 RGB、三路深度连续录制、完整解码、JSONL/manifest 和磁盘吞吐。
3. 低风险姿态下验证一次 measured hold，确认旧 MinSnap 轨迹被覆盖。
4. 验证 `i` 后只有接管后的 PICO 候选能到实机，并核对夹爪 `20=关、21=开`。
5. 分别完成 autonomous success、assisted success、failure、abort、故障 staging。
6. 长时间运行检查有界队列、剩余磁盘、视频帧数和退出清理。
