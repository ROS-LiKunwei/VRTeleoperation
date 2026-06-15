# BeaVR-bot 思维导图

这份图按“读代码”和“理解启动过程”的顺序组织：先看模块，再看初始化/实例化，最后看各子进程 `stream()` 之后在做什么。

## 模块总览

```mermaid
mindmap
  root((BeaVR-bot))
    入口与配置
      teleop.py
        转发到 beavr.teleop.main.main
      MainConfig
        TeleopConfig
        robot_name
        laterality
        config_file
      YAML 与 CLI
        draccus 解析 CLI
        load_yaml_config
        apply_yaml_preserving_cli
      机器人配置注册
        TeleopRobotConfig
        leap_config.py
        xarm7_config.py
        sysmo32_config.py
        sysmo_mujoco_config.py
    Teleop 运行框架
      TeleOperator
        根据配置创建子进程
        detector
        transform
        visualizer
        robot
        operator
        environment
      ProcessInstantiator
        _start_component
        instantiate_from_target
        component.stream
      Component
        统一 stream 接口
    遥操作数据链路
      Detector
        PICO4VRHandDetector
        OculusVRHandDetector
        原始关键点接收
      Transform
        TransformHandPositionCoords
        坐标平移
        方向帧计算
        滑动平均
        正交化
      Operator
        XArmOperator
        Sysmo32Operator
        LeapHandOperator
        手部运动重定向
      Robot Interface
        XArm7Robot
        Sysmo32RealControl
        Sysmo32CommandBuilder
        Sysmo32CommandLimiter
        Sysmo32MujocoKinematics
        LeapHandRobot
        接收命令并发布状态
      Simulation
        MuJoCoSysmoSimulator
        Sysmo32MujocoCommandMirror
        URDF 加载
        endeff site
        Jacobian IK
    网络与协议
      ZMQ
        PUB SUB
        PUSH PULL
        multipart topic payload
      common network
        publisher.py
        subscriber.py
        serialization.py
        handshake.py
        utils.py
      端口
        8087 right PICO4 raw
        8110 left PICO4 raw
        8088 keypoint stream
        8092 right transform
        8093 left transform
        10011 right command
        10013 left command
    数据类型
      detector_types.py
        InputFrame
        ButtonEvent
        SessionCommand
      operator_types.py
        CartesianTarget
        JointTarget
      interface_types.py
        CartesianState
        robot state dict
    日志
      logger.py
        setup_root_logger
        BaseLogger
        PoseLogger
        HandLogger
        JSON 批量落盘
        临时文件清理
      setup_logging.py
      Log 运行日志
    LeRobot
      Dataset
        lerobot_dataset.py
        AsyncImageWriter
        deferred episode finalization
        v2.1 incremental write
        v3.0 final conversion
        compute_stats.py
        sampler.py
      Policies
        ACT
        Diffusion
        PI0
        PI0Fast
        SmolVLA
        TD-MPC
        VQ-BeT
      Train Eval
        configs train
        configs eval
    资产与脚本
      assets
        URDF
        YCB objects
        wrench
        table cube
      scripts
        run_sysmo32.sh
        run_mujoco_sim.sh
        control_robot.py
        check_cameras.sh
      tests
        detector
        operator
        interface
        keypoint transform
```

## 初始化与实例化过程

以这个启动命令为例：

```bash
python teleop.py --robot_name=sysmo32 --laterality=bimanual
```

```mermaid
sequenceDiagram
  participant CLI as teleop.py
  participant Main as main.py / MainConfig
  participant Loader as common.configs.loader
  participant Registry as TeleopRobotConfig
  participant RobotCfg as Sysmo32Config
  participant Shared as SharedComponentRegistry
  participant TeleOp as TeleOperator
  participant Proc as multiprocessing.Process
  participant Build as build()
  participant Comp as Component.stream()

  CLI->>Main: main()
  Main->>Main: draccus 解析 CLI 到 MainConfig
  Main->>Main: MainConfig.__post_init__()
  Main->>Loader: load_robot_config(robot_name, laterality, sim_env)
  Loader->>Loader: import sysmo32_config
  Loader->>Registry: get_choice_class("sysmo32")
  Registry-->>Loader: Sysmo32Config
  Loader->>RobotCfg: Sysmo32Config(laterality=bimanual)
  RobotCfg->>Shared: get_bimanual_detector_config("pico4")
  RobotCfg->>Shared: get_transform_config(right)
  RobotCfg->>Shared: get_transform_config(left)
  RobotCfg->>RobotCfg: 创建 Sysmo32RobotCfg right/left
  RobotCfg->>RobotCfg: 创建 Sysmo32OperatorCfg right/left
  RobotCfg->>RobotCfg: 创建 MuJoCoSimConfig
  Loader-->>Main: robot_config
  Main->>Main: load_yaml_config + apply_yaml_preserving_cli
  Main->>TeleOp: TeleOperator(config)
  TeleOp->>Proc: 创建 environment 进程
  TeleOp->>Proc: 创建 detector 进程
  TeleOp->>Proc: 创建 transform 进程
  TeleOp->>Proc: 创建 visualizer 进程
  TeleOp->>Proc: 创建 robot 进程
  TeleOp->>Proc: 创建 operator 进程
  Main->>Proc: process.start()
  Proc->>Build: instantiate_from_target(cfg)
  Build->>Build: cfg.build()
  Build-->>Proc: 返回真实组件实例
  Proc->>Comp: component.stream()
```

## 核心运行链路

```mermaid
flowchart LR
  PICO[PICO4 Unity<br/>GestureDetectorXR.cs<br/>60Hz hand tracking] -->|ZMQ PUSH<br/>right 8087 / left 8110| Detector[pico4.py<br/>PICO4VRHandDetector]
  Detector -->|InputFrame<br/>PUB 8088<br/>right / left / button / pause| Transform[keypoint_transform.py<br/>TransformHandPositionCoords]
  Transform -->|right transformed frame<br/>8092| OpR[sysmo32_operator.py<br/>Sysmo32Operator right]
  Transform -->|left transformed frame<br/>8093| OpL[sysmo32_operator.py<br/>Sysmo32Operator left]

  OpR -->|CartesianTarget<br/>10011| Real[sysmo32_real_control.py<br/>Sysmo32RealControl]
  OpL -->|CartesianTarget<br/>10013| Real
  Real -->|current EE/state<br/>10012 / 10014| OpR
  Real -->|current EE/state<br/>10012 / 10014| OpL

  Real -->|IK seed from /joint_states<br/>nullspace posture bias<br/>joint/cartesian limit| IK[sysmo32_kinematics.py<br/>Sysmo32MujocoKinematics]
  IK -->|12 arm joints + hands<br/>18-field command| ROS[ROS2 / hardware board<br/>/sysmo_*_arm_controller/commands]
  Real -->|command mirror<br/>arm + hand action| Mirror[sysmo32_mujoco_command_sim.py<br/>Sysmo32MujocoCommandMirror]
  Mirror -->|MuJoCo visualization / dry run| MuJoCo[MuJoCo sysmo32.urdf]

  Real -->|LeRobot state topics<br/>joint/cartesian/command state| Recorder[control_robot.py<br/>sysmo32_adapter recorder]
```

## LeRobot 录制链路

```mermaid
flowchart TB
  Teleop[外部 teleop 栈<br/>PICO4 -> Transform -> Operator -> RealControl] --> State[Sysmo32RealControl<br/>发布 state/action cache]
  Cam[OpenCV camera<br/>640x480 @ 30fps] --> Adapter[Sysmo32Adapter<br/>capture_observation]
  State --> Adapter
  Adapter --> Loop[control_robot.py record loop<br/>fps=30]

  Loop -->|add_frame| Buffer[LeRobotDataset episode_buffer]
  Buffer -->|PNG 临时帧<br/>compress_level=1| ImageWriter[AsyncImageWriter<br/>thread/process queue]
  Loop -->|save_episode| Parquet[data/chunk-xxx<br/>episode_xxxxxx.parquet]
  Loop -->|save metadata only| Meta[meta/info.json<br/>meta/episodes.jsonl]
  Loop -->|defer| Pending[deferred episode finalization queue]

  Pending -->|Stop recording 后统一执行| Finalize[wait_for_async_video_encoding]
  Finalize -->|wait image writer| ImageWriter
  Finalize -->|compute_episode_stats| Stats[meta/episodes_stats.jsonl]
  Finalize -->|encode_video_frames<br/>SVT-AV1 mp4| Videos[videos/chunk-xxx/...mp4]
  Finalize -->|update video info| Meta
  Finalize -->|dataset_format=v3.0| Convert[convert_dataset_v21_to_v30]

  Note[默认 manage_teleop_state=false<br/>record 进程不抢 8089 pause/resume] -.-> Loop
```

## 子进程启动后做什么

```mermaid
flowchart TB
  subgraph DetectorProc[Detector 进程]
    D1[创建 ZMQ PULL<br/>8087/8110]
    D2[解析 PICO4 字符串]
    D3[封装 InputFrame]
    D4[PUB 到 8088]
    D1 --> D2 --> D3 --> D4
  end

  subgraph TransformProc[Transform 进程]
    T1[SUB 8088]
    T2[按 right/left topic 过滤]
    T3[手腕平移到原点]
    T4[计算手部方向帧]
    T5[滑动平均与正交化]
    T6[PUB 到 8092/8093]
    T1 --> T2 --> T3 --> T4 --> T5 --> T6
  end

  subgraph OperatorProc[Operator 进程]
    O1[SUB transformed frame]
    O2[读取机器人当前末端状态]
    O3[建立初始 hand/robot baseline]
    O4[计算手部相对运动]
    O5[映射到机器人基坐标系]
    O6[PUB CartesianTarget]
    O1 --> O2 --> O3 --> O4 --> O5 --> O6
  end

  subgraph RobotProc[RealControl 进程]
    R1[SUB CartesianTarget]
    R2[读取 /joint_states 作为 IK seed]
    R3[Sysmo32MujocoKinematics 逆解]
    R4[零空间姿态偏置]
    R5[关节/笛卡尔安全限幅]
    R6[发布 ROS2 arm command]
    R7[PUB LeRobot state/action cache]
    R8[PUB command mirror]
    R1 --> R2 --> R3 --> R4 --> R5 --> R6
    R5 --> R7
    R5 --> R8
  end

  subgraph MirrorProc[MuJoCo Mirror 进程]
    S1[加载 sysmo32.urdf]
    S2[SUB 18维 arm command mirror]
    S3[SUB hand action mirror]
    S4[驱动 MuJoCo 关节]
    S5[渲染/干跑验证]
    S1 --> S2 --> S3 --> S4 --> S5
  end

  subgraph RecordProc[LeRobot Record 进程]
    L1[Sysmo32Adapter 订阅 state/action]
    L2[OpenCV camera 640x480@30]
    L3[episode_buffer + AsyncImageWriter]
    L4[立即写 parquet/episode metadata]
    L5[结束录制后 deferred finalization]
    L6[stats + SVT-AV1 mp4 + v3.0 转换]
    L1 --> L3
    L2 --> L3 --> L4 --> L5 --> L6
  end

  DetectorProc --> TransformProc --> OperatorProc
  OperatorProc --> RobotProc
  RobotProc --> MirrorProc
  RobotProc --> RecordProc
```

## 初始化关键点

- `MainConfig.__post_init__` 把 `robot_name/laterality/sim_env` 变成机器人配置对象。
- `TeleopRobotConfig.register_subclass("sysmo32")` 是机器人配置注册表；`--robot_name=sysmo32` 最终会查到 `Sysmo32Config`。
- `Sysmo32Config.__post_init__` 根据 `laterality` 填充 `detector/transforms/robots/operators/environment`。
- `SharedComponentRegistry` 复用共享 VR 组件配置，避免组合机器人重复创建同一只手的 detector/transform。
- `TeleOperator` 只负责把组件配置包装成多个 `multiprocessing.Process`。
- 子进程内调用 `instantiate_from_target(cfg)`；如果配置对象有 `build()`，就先构造真实组件，再调用组件的 `stream()`。

## 读代码建议顺序

1. [teleop.py](/home/likunwei/dataCollection/beavr-bot/teleop.py) 看入口如何转发到 `main()`。
2. [main.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/main.py) 看 `MainConfig`、YAML/CLI 合并和进程启动。
3. [loader.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/common/configs/loader.py) 看 `load_robot_config`、注册表查找和组合机器人合并。
4. [sysmo32_config.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/configs/robots/sysmo32_config.py) 看 SYSMO-32 如何按 `laterality` 生成组件配置。
5. [shared_components.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/configs/robots/shared_components.py) 看共享 VR 组件配置如何创建和复用。
6. [initializers.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/initializers.py) 看 `TeleOperator` 如何把配置变成多个进程。
7. [instantiator.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/common/factory/instantiator.py) 看 `build()` 和 `_target_` 的动态实例化规则。
8. [pico4.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/pico4.py)、[keypoint_transform.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/detector/vr/keypoint_transform.py)、[sysmo32_operator.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/operator/robots/sysmo32_operator.py)、[sysmo32_robot.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/interface/robots/sysmo32_robot.py)、[mujoco_sim.py](/home/likunwei/dataCollection/beavr-bot/src/beavr/teleop/components/simulation/mujoco_sim.py) 按数据流逐个读 `stream()`。
