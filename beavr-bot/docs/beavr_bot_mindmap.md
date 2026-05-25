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
        Sysmo32Robot
        LeapHandRobot
        接收命令并发布状态
      Simulation
        MuJoCoSysmoSimulator
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
  A[PICO4 Unity<br/>GestureDetectorXR.cs] -->|ZMQ PUSH<br/>8087/8110| B[pico4.py<br/>PICO4VRHandDetector]
  B -->|InputFrame<br/>PUB 8088| C[keypoint_transform.py<br/>TransformHandPositionCoords]
  C -->|transformed frame<br/>8092/8093| D[sysmo32_operator.py<br/>Sysmo32Operator]
  D -->|CartesianTarget<br/>10011/10013| E[sysmo32_robot.py<br/>Sysmo32Robot]
  D -->|CartesianTarget<br/>10011/10013| F[mujoco_sim.py<br/>MuJoCoSysmoSimulator]
  E -->|state<br/>10012/10014| D
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

  subgraph RobotProc[Robot 进程]
    R1[SUB CartesianTarget]
    R2[调用控制器或 mock 控制]
    R3[维护关节和笛卡尔状态]
    R4[PUB robot state]
    R1 --> R2 --> R3 --> R4
  end

  subgraph SimProc[MuJoCo 进程]
    S1[加载 sysmo32.urdf]
    S2[添加 left/right endeff site]
    S3[SUB CartesianTarget]
    S4[Jacobian 伪逆 IK]
    S5[驱动 MuJoCo 渲染]
    S1 --> S2 --> S3 --> S4 --> S5
  end

  DetectorProc --> TransformProc --> OperatorProc
  OperatorProc --> RobotProc
  OperatorProc --> SimProc
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
