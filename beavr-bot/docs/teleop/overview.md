# Teleop 系统概览

## Teleop 架构概览

本文面向第一次阅读代码的同学，解释 BeavR teleoperation（遥操作）系统从入口到执行的端到端工作方式。文档会以一个可运行的命令为例，串起配置加载、网络通信、各类组件的职责，以及从 VR 检测器到机器人控制器的数据流。

示例运行：

```bash
python teleop.py --robot_name=xarm7,leap --laterality=right
```

- 这会启动 teleop 入口，并选择两种“机器人”：`xarm7`（机械臂）与 `leap`（手）。`--laterality=right` 表示只启用右侧链路。系统会把两者对应的组件（detector、transform、operator、robot 等）组合起来，并以进程/线程的方式启动。

### TL;DR：数据流

1) Oculus VR detector reads hand data and publishes keypoints over ZeroMQ.
2) Transform component converts raw keypoints to a stable 6DoF hand frame in the VR/world coordinates and republishes it.
3) Operator (e.g., `XArm7RightOperator`) consumes the transformed frame, computes a target end‑effector pose using calibrated transforms, optionally filters it, and publishes commands.
4) Robot interface (`XArm7Robot`) subscribes to those commands and calls the low‑level controller (`DexArmControl`) to move real hardware. It also publishes robot state for observers/recorders.
5) A lightweight handshake/ACK mechanism coordinates critical state transitions (e.g., pause/reset) between publishers and subscribers.

1) Oculus VR detector 读取手部数据，并通过 ZeroMQ 发布 keypoints。
2) Transform 组件将原始 keypoints 转换成 VR/world 坐标系下稳定的 6DoF 手部坐标系（位姿），并再次发布。
3) Operator（如 `XArm7RightOperator`）接收该“变换后的手部位姿”，结合标定得到的变换关系计算目标末端执行器（EE）位姿，可选做滤波，然后发布控制命令。
4) Robot interface（如 `XArm7Robot`）订阅命令并调用底层控制器（`DexArmControl`）驱动真实硬件运动，同时发布机器人状态供可视化/记录器订阅。
5) 轻量的 handshake/ACK 机制用于在 publisher/subscriber 之间协调关键状态切换（如 pause/reset），避免竞争条件。

---

## 1) Entry point: `src/beavr/teleop/main.py`

它做了什么：
- 定义 `MainConfig` 数据类：其中嵌套了结构化的 `TeleopConfig`（全局 teleop 配置）与机器人选择相关参数。
- 使用 `draccus` 从 dataclass 自动生成命令行参数。优先级为：CLI 参数 > YAML 覆盖 > 默认值。
- 加载 YAML 配置（默认 `config/dev.yaml`），并在保留 CLI 覆盖的前提下完成合并。
- 按机器人名称与 laterality 构建所需机器人配置，然后实例化 teleop 系统（`TeleOperator`）并启动所有进程。

`MainConfig` 的关键字段：
- `teleop: TeleopConfig`: structured teleop config (control loop rates, network ports, flags, etc.).
- `robot_name: str`: comma‑separated robot names (e.g., `xarm7,leap`).
- `laterality: str`: `right | left | bimanual`.
- `config_file: str`: path to YAML overrides.
- `robot: Any`: the composite built robot structure (set in `__post_init__`).

执行流程：
- `main()` 被 `@draccus.wrap()`包裹， 以便命令行参数能映射到 dataclass 字段.
- `load_yaml_config()` reads the YAML; `apply_yaml_preserving_cli()` merges it, preserving any CLI overrides.
- `load_robot_config(robot_name, laterality)` returns a composite configuration for the requested robots.
- `TeleOperator(config)` creates components, then `get_processes()` returns the processes to start.

你的示例命令含义：
- `--robot_name=xarm7,leap`: include both robots.
- `--laterality=right`: initialize only right‑hand/right‑arm components (you can use `bimanual` to include both sides).

---

## 2) Robot configs: `leap_config.py` and `xarm7_config.py`

两份文件遵循相同的结构模式：
- 用若干小 dataclass 描述某个机器人所需组件的配置：
  - `...OperatorCfg`: 用于将手势帧转换为命令的操作进程的配置.
  - `...RobotCfg`: 用于接收命令并与硬件/模拟器交互的机器人接口进程的配置.
  - `...Config`: 在配置注册表中注册的top-level dataclass; 它整合了每侧的组件: detector(s), transform(s), visualizer(s), operator(s), robot(s), and optional recorder settings.
- `build()` 方法会基于这些 dataclass 构造具体的组件实例。
- 所有网络常量（host、端口号、topic 名称）来自 `beavr.teleop.configs.constants`。

### 2.1) `src/beavr/teleop/configs/robots/leap_config.py`

用途：
- 配置 Leap（VR 手）“机器人”及其 operator，以支持右手、左手或双手（bimanual）模式。

关键类型：
- `LeapHandOperatorCfg`:
  - 端口：订阅转换后的关键点；发布手的关节/笛卡尔坐标指令；发布重置事件。
  - `hand_side`: `right` or `left` 选择要订阅的ZMQ话题.
  - `finger_configs` and `logging_config`: feature toggles and optional logging.
  - `build()` returns a `LeapHandOperator` instance.
- `LeapHandRobotCfg`:
  - 端口：订阅命令，发布关节状态和机器人状态以供记录，订阅重置命令.
  - `simulation_mode`: switch for sim vs. real.
  - `hand_side`: used to set `is_right_arm` when building the concrete `LeapHandRobot`.
  - `state_publish_port`: 每侧独立，以便左右两侧可以同时运行.
  - `build()` returns a `LeapHandRobot` instance.
- `LeapHandConfig`:
  - Top‑level container that, in `__post_init__`, configures detectors, transforms, visualizers, operators, and robots based on laterality.
  - For `bimanual`, 它同时依赖右侧和左侧实例；对于单侧，它仅依赖该侧的components.
  - `build()` 为每个已配置的 components 创建具体 instance.

### 2.2) `src/beavr/teleop/configs/robots/xarm7_config.py`

用途：
- 配置 XArm7 机器人（左、右或双臂）及其 operator。

关键类型：
- `XArm7RobotCfg`:
  - Network: robot IP per side (`RIGHT_XARM_IP`, `LEFT_XARM_IP`) and per‑side port mapping.
  - Publishes a rich state dictionary for recording (`state_publish_port`).
  - Validates port ranges and IP format in `__post_init__`.
  - `build()` returns an `XArm7Robot` (see interface below).
- `XArm7OperatorCfg`:
  - Subscribes to transformed keypoints and auxiliary topics (button for resolution, pause/teleop state).
  - Publishes end‑effector commands.
  - Picks the concrete class by side at build time:
    - Right: `XArm7RightOperator`
    - Left: `XArm7LeftOperator`
  - Includes optional pose logging configuration.
- `XArm7Config`:
  - Same assembly pattern as Leap: laterality decides which side(s) are appended to detectors, transforms, visualizers, operators, and robots.
  - Uses slightly offset port numbers for the left arm so both arms can run together cleanly.

---

## 3) 消息传递与网络: `src/beavr/teleop/network/*.py`

系统内部广泛使用 ZeroMQ，用于在不同进程之间进行实时、解耦的 pub/sub 消息通信。

核心组成：
- Global ZMQ context
  - `get_global_context()` and `set_global_context()` 为每个进程提供一个共享上下文(context).

- Publisher/Subscriber primitives（基元）
  - `BasePublisher`: binds a PUB socket (`tcp://*:<port>`); `pub.send_multipart([topic, payload])`.
  - `BaseSubscriber`: a thread that owns a SUB (or other) socket; it `connect()`s to `tcp://host:port`, subscribes to a topic, polls(轮询) with timeouts, and calls `process_message(data)` when messages arrive.
  -（未找到使用） `ZMQKeypointPublisher` / `ZMQKeypointSubscriber`: 用于发送/接收已封存的关键点数组的便利类.
  - `ZMQCompressedImageTransmitter` / `ZMQCompressedImageReceiver`: 压缩并流式传输摄像头图像.

- Socket（套接字） helpers
  - `create_push_socket`, `create_pull_socket`, `create_request_socket`, `create_response_socket` 函数封装了常见的 ZMQ 模式，并设置了超时和错误处理机制，以提高健壮性（包括 HWM、延迟（linger）、接收超时、合并（conflation）等）.

- Central publisher manager
  - `ZMQPublisherManager`: 一个单例对象，它为每个`(host, port)`组合拥有一个后台`PublisherThread`，并配有一个线程安全的队列。这确保了ZeroMQ套接字仅由创建它们的线程使用，从而避免了“套接字被多线程使用”的问题。
  - 调用者使用`publish(host, port, topic, data)`，管理器负责处理序列化（serialization）、高水位线（HWM）和背压（backpressure）。它还会监控发布者线程的运行状况。

- Handshake/ACK coordination 握手/确认 协调
  - `HandshakeCoordinator`: 注册订阅者（id → 主机（host），端口(port)），并在订阅者端运行REP服务器。发布者可以在执行关键操作之前/之后，向一组订阅者请求确认(`request_acknowledgments([...])`).
  - `publish_with_guaranteed_delivery(...)` 可选地将发布操作与确认的往返过程结合起来

- Cleanup
  - `cleanup_zmq_resources()` 函数会关闭发布线程、握手服务器，并终止 ZMQ 上下文.

消息格式：
- 所有消息都是 multipart：`[topic: bytes, payload: bytes]`，其中 `payload` 通常是 `pickle.dumps(data)` 的结果。
- Subscriber 会在订阅阶段按 topic 过滤消息，并通过一些轻量 accessor 暴露最近一次 payload（例如 `recv_keypoints()`）。

---

### 4.1) VR Detector: `src/beavr/teleop/components/detector/vr/oculus.py`

- `OculusVRHandDetector` (single hand) or `BimanualOculusVRHandDetector` (both hands) 通过PULL套接字从预配置的端口读取原始手部数据 (see `network.RIGHT_HAND_PORT`, `network.LEFT_HAND_PORT`, etc.).
- 它将原始流解析为一个数字列表（第一个元素表示相对坐标还是绝对坐标，后面是XYZ三元组）.
- 它通过 `ZMQPublisherManager` 在 `KEYPOINT_STREAM_PORT` 上以特定 topics 发布这些关键点:
  - Right hand: `right`
  - Left hand: `left`
  - 分辨率按钮(Resolution button): `button` (映射到高/低分辨率枚举)
  - Pause: `pause` (mapped to teleop stop/continue enum)

注意：Transform 组件（由机器人 config 创建）会订阅 `KEYPOINT_STREAM_PORT`，并将稳定化后的“transformed hand frame”发布到 `KEYPOINT_TRANSFORM_PORT`。

### 4.2) Operator base and right‑arm operator 操作员底座和右臂操作器

- `src/beavr/teleop/components/operators/robots/xarm7_operator.py` 提供基类 `XArmOperator`, 该基类具有以下功能:
  - 订阅已转换的关键点主题（每侧）。对于右手，主题通常类似于`TRANSFORMED_HAND_FRAME`（右手/左手变体由特定侧的 topics/namespace处理）
  - 可选择订阅解析（`button`）和暂停（`pause`）主题。
  - 订阅机器人当前的位姿流（`endeff_homo`），以在重置过程中捕捉初始baseline。
  - 保持校准的变换(Maintains calibrated transforms):
    - `H_R_V`: Robot base → VR base
    - `H_T_V`: Hand‑Tracking base → VR base
  - On each cycle (at `VR_FREQ`):
    1) 检查遥操作状态（暂停/继续）以及是否需要重置.
    2) 如果进行重置，则请求机器人提供姿态并捕捉初始手部帧；否则，采用最新转换后的手部帧.
    3) 计算自重置以来手的相对运动，并使用`H_R_V`和`H_T_V`将其映射到机器人基座坐标系中.
    4) 构建目标末端执行器位姿，应用可选的互补滤波器（位置线性插值（LERP）+ 方向球谐插值（SLERP）），并对四元数进行归一化处理.
    5) 将命令字典`{position: [x,y,z], orientation: [qx,qy,qz,qw], timestamp}`发布到机器人的命令端口.
  - 提供了一个小型握手服务器，以便其他组件能够确认其是否可访问.

- `src/beavr/teleop/components/operators/xarm7_right.py` 通过右臂特定的`H_R_V`和`H_T_V`矩阵来专门化`XArmOperator`，并设置`hand_side=right`。左臂类与此类似。

### 4.3) 机器人接口与控制器（Robot interface and controller）: `src/beavr/teleop/components/interface/robots/xarm7_robot.py`

- `XArm7Robot` 通过`DexArmControl`将遥控指令适配到硬件上:
  - 在命令端口上订阅`endeff_coords`（来自操作员的末端执行器命令）.
  - 订阅`reset`和`home`主题（分别对应不同的端口），以处理基线捕获(baseline capture)和归位(homing)操作.
  - 订阅`pause`（遥控状态），用于控制运动起停.
  - 当接收到命令时，它会将位置和方向进行拼接，并调用  `self._controller.move_arm_cartesian(...)`.
  - 在专用的状态端口（`state_publish_port`）上，以机器人的名称作为主题（例如，`xarm7_right`）定期发布状态字典。状态包括关节状态、笛卡尔坐标状态、指令状态等，并附带适合记录/分析的时间戳.
  - 为记录器公开实用程序访问器，如`get_cartesian_state_from_operator()`.
  - 拥有自己的握手服务器，以便控制流可以验证连接性.

- `DexArmControl` (controller)
  - 这是一个围绕XArm硬件的低级API封装。它提供了`XArm7Robot`调用的运动原语（home、move_cartesian、set modes、query states）.

---

## 5) 配置、侧向性和组成 （Configuration, laterality, and composition）

- Laterality (`right | left | bimanual`) drives which side(s) are instantiated for each robot’s detectors, transforms, visualizers, operators, and robots.
- Each side uses dedicated topics and, where needed, distinct ports so both sides can run concurrently.
- The top‑level `MainConfig` collects your `--robot_name` list, resolves each robot’s registered config class (e.g., `XArm7Config`, `LeapHandConfig`), and appends the side‑appropriate components to the launch plan.
- `build()` methods on the robot configs create concrete objects (subscribers, publishers, threads) ready to be launched by the teleop runtime.

- Laterality（`right | left | bimanual`）决定每种机器人要实例化哪些侧的 detector、transform、visualizer、operator 与 robot。
- 每一侧使用独立的 topic，并在需要时使用不同端口，以便左右两侧可以并发运行。
- 顶层 `MainConfig` 会收集 `--robot_name` 列表，解析出每种机器人的注册配置类（如 `XArm7Config`、`LeapHandConfig`），并把与 laterality 匹配的组件追加到启动计划中。
- 各机器人配置中的`build()`方法会创建具体对象（订阅者、发布者、线程），这些对象准备好由teleop运行时启动。

---

## 6) 操作控制：重置、暂停、分辨率（ Operational controls: reset, pause, resolution）

- Reset: The operator requests a robot pose sample and captures a new hand baseline; subsequent motion is interpreted relative to that baseline.
- Pause/Continue: A small integer enum (`ARM_TELEOP_STOP` / `ARM_TELEOP_CONT`) gates motion in the operator and on the robot side.
- Resolution: Button presses select “high” vs “low” resolution scaling for hand motion → end‑effector translation.
- Handshake: Critical transitions can be protected by an ACK round trip via `HandshakeCoordinator` to avoid races.

- Reset：operator 会请求一次机器人位姿采样并捕获新的手部基线（baseline）；之后的运动都会以该基线(baseline)为参照解释为相对位移/旋转。
- Pause/Continue：一个小整型枚举（`ARM_TELEOP_STOP` / `ARM_TELEOP_CONT`）用于在 operator 与 robot 两侧共同“闸门式”控制是否允许运动。
- Resolution：按键可在“高/低”分辨率缩放之间切换，用于调整手部运动到末端平移的映射尺度。
- Handshake：关键状态切换可通过 `HandshakeCoordinator` 的 ACK 往返进行保护，避免竞态。

---

## 7) Putting it together for your example

Command: `python teleop.py --robot_name=xarm7,leap --laterality=right`

会启动什么：
- `LeapHandConfig` (right):
  - VR detector (right‑hand input), transforms, visualizer, Leap operator (right), Leap robot (right).
- `XArm7Config` (right):
  - Detector can be shared if configured bimanually; right transform; right operator (`XArm7RightOperator`); right robot (`XArm7Robot` with `RIGHT_XARM_IP`).

运行时流程：
1) VR detector 发布原始右手 keypoints + button + pause。
2) Transform 发布稳定化后的右手 hand frame。
3) `XArm7RightOperator` 使用该 hand frame，应用 `H_R_V`/`H_T_V`、滤波，并发布 `endeff_coords`。
4) `XArm7Robot` 接收命令，通过 `DexArmControl` 驱动真实机器人运动，同时发布位姿与状态。
5) 可选的 recorders/visualizers 订阅状态流进行记录/显示。

---

## 8) 调试和调优技巧 (Tips for debugging and tuning)

- 端口(ports)与 topics：如果数据不流动，先核对对应侧（right/left）的端口号与 topic 是否一致。左侧端口通常会做 offset 以避免冲突。
- 滤波：如果运动抖动或延迟明显，可以调整 `CompStateFilter` 的比例参数，或临时关闭滤波做对比。
- Pause/Reset：通过 pause/reset 重新建立基线(re-baseline)；关键切换时留意日志里的 handshake/ACK 信息。
- 清理：退出时组件会调用 `cleanup_zmq_resources()` 关闭 sockets/threads；若提示端口 “in use”，通常是仍有旧进程未退出。

---

## 9) 术语表(Glossary)

- Keypoints：来自手部追踪（VR 头显）的 3D 关键点。通常会经历 raw → transformed。
- Transformed hand frame：由 keypoints 推导出的稳定 6DoF 位姿（位置 + 姿态）。
- End‑effector（EE）：机器人末端工具位姿（笛卡尔位置 + 四元数姿态）。
- Laterality：当前启用的侧（右/左/双侧）。
- ZMQ topic：PUB/SUB 中用于过滤消息的字符串标签（如 `right`、`pause`、`endeff_coords`）。
