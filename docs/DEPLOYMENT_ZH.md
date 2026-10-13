# FA C++ 部署与运行

## 1. 构建和版本检查

```bash
git clone --branch 'C++' https://github.com/ROS-LiKunwei/VRTeleoperation.git VRTeleoperation-cpp
cd VRTeleoperation-cpp
bash tools/build_cpp.sh
source /opt/ros/humble/setup.bash
source cpp_ws/install/setup.bash
```

只构建 `fa_teleop` 包，不需要构建原 Python BeaVR 工作区。依赖 ROS 2 Humble、Eigen、Pinocchio、yaml-cpp、OpenSSL 和 cppzmq/libzmq。构建机上的 Python 必须与 Humble 的 ROS Python ABI 对应；脚本使用 `/usr/bin/python3`。

在目标运行设备重新构建后，再执行相同离线测试和实时基准。本机 x86_64 的频率结果不能替代另一台机器的验收。

```bash
ros2 run fa_teleop fa_core_tests \
  "$(ros2 pkg prefix fa_teleop)/share/fa_teleop/bundle" --verify-only
cat provenance/COMPONENT_VERSIONS.json
```

安装目录的 `release.json`、`files.sha256`、模型、IK、MinSnap 或配置发生变化时，程序会拒绝启动。调整控制配置后应重新构建、测试并导出新的版本清单；不能直接修改安装目录 YAML 后继续沿用旧构建。

## 2. 输入、坐标与反馈条件

| 项目 | 默认设置/条件 |
|---|---|
| 右手输入 | ZMQ PULL，8087 |
| 左手输入 | ZMQ PULL，8110 |
| 暂停/使能 | ZMQ PULL，8100；`Low` 暂停，`High` 使能 |
| 分辨率事件 | 8095；保留接收，当前精度配置的平移比例固定为 1.0 |
| 标定语音 | ZMQ PUB，8112；`fa_calibration_prompt` + UTF-8 两帧协议 |
| 机器人参考坐标系 | `pelvis` |
| TCP | `left_hand_base_link` / `right_hand_base_link` |
| 姿态增量 | base/world 表达，与当前 FA Python 配置一致 |
| 反馈 | `/joint_states`；完整左右臂与颈部 16 个关节，有限值、递增源时间戳 |
| 输入失效 | 接收时间超过 100 ms、源帧过期或双手时差过大时暂停 |
| 反馈失效 | 接收时间超过 50 ms 时停止输出并锁存执行器故障 |

PICO 和机器人主机需要时间同步，PICO 原有时间协议还要求相同的本地时区。源时间戳的允许过去偏差为 150 ms，未来偏差为 50 ms。C++ 周期调度和本机新鲜度计时使用单调时钟。

当前 IK 继承原求解器的固定腰部假设。验收时腰部姿态应与模型基准一致；移动腰部需要扩展全身反馈和运动学接口后重新验证。

## 3. 默认只启动等待状态

先确保原 Python FA 控制、旧 MinSnap 或其他控制程序没有同时向同一个上身命令话题输出。新执行器会检查重复命令发布者，发现冲突后锁存故障；它不会替用户停止已有进程。

```bash
ros2 launch fa_teleop teleop.launch.py \
  profile:=precision_200hz enable_motion:=false
```

这会加载版本、模型、ZMQ 接口和 ROS 反馈，保持不发送运动命令。若提示端口占用，检查原遥操作进程的使用情况。

用户准备好实际运行后使用：

```bash
ros2 launch fa_teleop teleop.launch.py \
  profile:=precision_200hz enable_motion:=true
```

随后在 PICO 上执行使能。初始化左臂目标为 `[-1.05, 0.76, -0.62, -1.03, -0.68, 0, -0.35]` rad，右臂为 `[-1.05, -0.76, 0.62, -1.03, 0.68, 0, 0.35]` rad；颈部保持当前反馈值。默认期望时长 5 s、速度上限 0.3 rad/s，规划器可按约束延长。

初始化到位后进行双手轴标定：稳定起点 → 双手向前平移 → 回起点 → 双手向左平移 → 回起点并稳定。之后建立相对跟踪基线。

## 4. 暂停、失效与故障复位

用户暂停或 VR 失效时，控制节点提交制动目标；执行器从当前参考的位置、速度与加速度连续生成受约束的停止轨迹。控制进程断开时，执行器也会在目标心跳超时后启动制动。

反馈丢失、版本错误、非法关节命令、连续规划超时、输出越界或重复发布者会锁存故障。新程序停止发布后的底层行为还取决于驱动/控制器的超时保持机制，应在机器人验收中单独确认。

复位顺序为执行器、控制节点，然后在 PICO 上重新暂停/使能：

```bash
ros2 service call /fa_cpp/executor_reset std_srvs/srv/Trigger '{}'
ros2 service call /fa_cpp/reset std_srvs/srv/Trigger '{}'
```

执行器状态：

```bash
ros2 topic echo /fa_cpp/executor_status
ros2 topic echo /fa_cpp/teleop_status
```

`executor_status` 格式为 `部署ID|READY或FAULT|已规划数量|late=...|samples=...|reason=...`。部署 ID 为 `release SHA:所选配置 SHA`。故障原因：1=版本/参考系/模式，2=非法目标，3=规划失败，4=连续规划迟到，5=输出约束，6=重复发布者，7=实际关节跟踪差过大，8=反馈过期。控制状态同时包含 `pause_reason`、周期数量、IK 成功数量和最大计算/唤醒耗时。

## 5. 接口兼容边界

控制器输出仍为 `/upper_position_controller/commands`，16D 顺序与原 FA 一致；左右手分别发布到 `/left_topic_to_hand` 和 `/right_topic_to_hand`，默认打开 21、关闭 20。

为严格对应算法版本和配置，C++ 控制与执行器之间使用新话题 `/fa_cpp/target` 和 `fa_teleop/msg/JointTarget`。原 `/min_snap/target` 消息缺少版本、源时间戳和序号，未作为新执行器的运动入口。Python 旧链路保留原接口，不能与新链路同时控制硬件。

可选采集状态桥：

```bash
ros2 run fa_teleop beavr_state_bridge.py
```

桥在 10020/10018 发布原 `fa_left/fa_right` 状态字典，频率 30 Hz，含实际关节反馈和期望关节状态。手部状态沿用命令语义，不是物理手部反馈。桥本身不发布任何机器人运动命令。原采集工具应关闭自动启动 Python 遥操作的选项，再读取该桥。

现有 MuJoCo 工具和资源随 `beavr-bot/` 保留。本次端到端验收使用 `fa_mock_robot` 简化运动学模型；原 MuJoCo mirror 对新 `JointTarget` 的适配和 Android APK 构建不在已验证范围内。

## 6. 离线/隔离测试命令

```bash
ctest --test-dir cpp_ws/build/fa_teleop --output-on-failure

python3 cpp_ws/src/fa_teleop/tools/integration_test.py \
  --bin-dir cpp_ws/install/fa_teleop/lib/fa_teleop \
  --bundle cpp_ws/install/fa_teleop/share/fa_teleop/bundle \
  --output validation/integration

ros2 run fa_teleop fa_benchmark \
  --bundle "$(ros2 pkg prefix fa_teleop)/share/fa_teleop/bundle" \
  --steps 120000 --realtime --output validation/realtime_600s.json
```

端到端测试固定使用 localhost、ROS domain 197 和 `/fa_cpp_integration` 命名空间；ZMQ 使用 18087/18110 等测试端口。只启动并清理测试自己创建的进程，不连接物理机器人。

## 7. 真机精度验收

分别测量：PICO 新传感器帧率、200 Hz 控制循环、目标接受率、1 kHz 输出、实际反馈率。重复使用同一 VR 样本不能计为新观测。

先在低速、可达、远离奇异和限位的区域测试，记录左右臂最大位置/姿态误差、RMS、P95/P99、延迟和超限次数。以原目标、规划参考和实际 TCP 分别比较，不能用规划参考替代原目标隐藏跟踪滞后。

模型内误差使用同一 TCP/参考系及编码器 FK。物理 1 mm / 0.01 rad 验收还需要独立 TCP 标定和外部测量。未经真机测试的速度、负载和工作空间不纳入已达标条件。
