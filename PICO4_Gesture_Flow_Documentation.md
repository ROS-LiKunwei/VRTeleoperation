# PICO4 手势识别全流程文档

## 1. 流程概述

PICO4 手势识别系统包含以下几个主要环节：

1. **PICO4 获取**：通过 XR 手部追踪系统采集手部关节数据
2. **BeaVR-app 转发**：将采集到的数据通过网络转发给 BeaVR-bot
3. **BeaVR-bot 接收**：接收并处理数据，然后发布给机器人控制系统
4. **机器人控制**：根据手势数据控制机器人手臂运动

### 数据流全景图

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PICO4 VR头显 (Unity端)                             │
│                                                                      │
│  [1] GestureDetectorXR.cs                                            │
│      XRHandSubsystem → 26关节坐标采集 → 序列化为字符串               │
│      │                                                               │
│      ▼                                                               │
│  [2] NetMQController.cs                                              │
│      ZMQ PUSH套接字 → 网络转发到BeaVR-bot                            │
│      - RightHand → tcp://{IP}:8087                                   │
│      - LeftHand  → tcp://{IP}:8110                                   │
│      - Resolution → tcp://{IP}:8095                                  │
│      - Pause → tcp://{IP}:8100                                       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ ZMQ TCP
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    BeaVR-bot (Python端)                               │
│                                                                      │
│  [3] PICO4VRHandDetector (pico4.py)                                  │
│      ZMQ PULL (8087/8110) → 解析 → ZMQ PUB (8088)                   │
│      ▼                                                               │
│  [4] TransformHandPositionCoords (keypoint_transform.py)             │
│      ZMQ SUB (8088) → 坐标变换+平滑 → ZMQ PUB (8092/8093)           │
│      ▼                                                               │
│  [5] XArmOperator (xarm7_operator.py)                                │
│      ZMQ SUB (8092/8093) → 运动重定向 → ZMQ PUB (endeff_coords)     │
│      ▼                                                               │
│  [6] Sysmo32Robot (sysmo32_robot.py) / MuJoCo仿真 (mujoco_sim.py)   │
│      ZMQ SUB (endeff_coords) → 机器人控制 / 仿真渲染                 │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. 代码详细分析

### 2.1 PICO4 获取端 (`GestureDetectorXR.cs`)

**文件路径**：`BeaVR-app/BeaVR-Unity/Assets/Scripts/Gesture Detection/GestureDetectorXR.cs`

#### 核心功能
- 使用 XRHandSubsystem 采集手部关节数据
- 支持左手和右手的手势识别
- 实现手指捏合手势检测（食指、中指、无名指）
- 定期打印手腕、手掌和 26 个关节的坐标数据
- 计算并打印发送频率

#### 关键代码分析

**初始化过程**：
```csharp
void Start()
{
    // 网络配置
    GameObject netConfGameObject = GameObject.Find("NetworkConfigsLoader");
    if (netConfGameObject != null)
        netConfig = netConfGameObject.GetComponent<NetworkManager>();

    // 获取XR手部子系统
    TryResolveHandSubsystem();

    // 给OpenXR一点时间并运行NetMQController初始化
    StartCoroutine(InitializeNetMQAfterDelay());
}
```

**数据采集与发送**：
```csharp
void SendHandDataThroughController(string typeMarker)
{
    try
    {
        if (_handSubsystem == null)
            return;

        // 右手数据采集与发送
        List<Vector3> rightHandGestureData = new List<Vector3>();
        CollectHandJointPositions(_handSubsystem.rightHand, rightHandGestureData);
        string rightHandDataString = SerializeVector3List(rightHandGestureData);
        rightHandDataString = typeMarker + ":" + rightHandDataString;
        NetMQController.Instance.SendMessage("RightHand", rightHandDataString);

        // 左手数据采集与发送
        List<Vector3> leftHandGestureData = new List<Vector3>();
        CollectHandJointPositions(_handSubsystem.leftHand, leftHandGestureData);
        string leftHandDataString = SerializeVector3List(leftHandGestureData);
        leftHandDataString = typeMarker + ":" + leftHandDataString;
        NetMQController.Instance.SendMessage("LeftHand", leftHandDataString);

        // 频率统计与数据打印
        // ...（频率计算、手腕数据打印、26关节数据打印）
    }
    catch (Exception e)
    {
        Debug.LogError($"发送手部数据错误: {e.Message}");
    }
}
```

**数据格式**：
发送的字符串格式为 `<type_marker>:x1,y1,z1|x2,y2,z2|...|x26,y26,z26:`
- `type_marker`：`"relative"`（相对模式）或 `"absolute"`（绝对模式）
- 坐标部分：26个关节的xyz坐标，用 `|` 分隔，每个坐标用 `,` 分隔

**26关节顺序**（XRHandJointID）：

| 索引 | 关节名称 | 说明 | 索引 | 关节名称 | 说明 |
|------|----------|------|------|----------|------|
| 0 | Wrist | 手腕 | 13 | MiddleIntermediate | 中指中间 |
| 1 | Palm | 手掌 | 14 | MiddleDistal | 中指远端 |
| 2 | ThumbMetacarpal | 拇指掌骨 | 15 | MiddleTip | 中指尖端 |
| 3 | ThumbProximal | 拇指近端 | 16-20 | Ring系列 | 无名指 |
| 4 | ThumbDistal | 拇指远端 | 21 | LittleMetacarpal | 小指掌骨 |
| 5 | ThumbTip | 拇指尖端 | 22 | LittleProximal | 小指近端 |
| 6 | IndexMetacarpal | 食指掌骨 | 23 | LittleIntermediate | 小指中间 |
| 7 | IndexProximal | 食指近端 | 24 | LittleDistal | 小指远端 |
| 8 | IndexIntermediate | 食指中间 | 25 | LittleTip | 小指尖端 |
| 9 | IndexDistal | 食指远端 | | | |
| 10 | IndexTip | 食指尖端 | | | |
| 11 | MiddleMetacarpal | 中指掌骨 | | | |
| 12 | MiddleProximal | 中指近端 | | | |

**手势控制**：
- 左手食指捏合 → 相对数据模式（StreamRelativeData）
- 左手中指捏合 → 绝对数据模式（StreamAbsoluteData）
- 左手无名指捏合 → 停止遥操作

---

### 2.2 BeaVR-app 转发端 (`NetMQController.cs`)

**文件路径**：`BeaVR-app/BeaVR-Unity/Assets/Scripts/Network/NetMQController.cs`

#### 核心功能

NetMQController 是 BeaVR-app Unity端的网络通信核心组件，负责管理所有 ZMQ 套接字的生命周期，
将 GestureDetectorXR.cs 采集的手部数据通过网络转发给 BeaVR-bot。

主要职责：
1. **单例管理**：全局唯一的网络控制器，跨场景不销毁
2. **套接字生命周期管理**：创建、连接、重连、关闭 ZMQ PushSocket
3. **网络配置加载**：从 JSON 文件读取 IP 地址和端口号
4. **消息发送**：带超时保护的消息发送，失败自动重连
5. **诊断测试**：启动时发送测试消息验证网络连通性
6. **数据转发日志**：频率统计、手腕数据打印、26关节数据打印

#### 类结构

```
NetMQController (MonoBehaviour, 单例)
├── 套接字管理
│   ├── sockets: Dictionary<string, PushSocket>    # 套接字字典
│   ├── socketConnectionStatus: Dictionary          # 连接状态
│   └── socketFailCounts: Dictionary                # 失败计数
├── 网络配置
│   ├── ipAddress: string                           # 服务器IP
│   ├── rightKeypointPort: string                   # 右手端口
│   ├── leftKeypointPort: string                    # 左手端口
│   ├── resolutionPort: string                      # 分辨率端口
│   └── pausePort: string                           # 暂停端口
├── 日志控制
│   ├── _sendCounts / _sendFrequencies              # 频率统计
│   ├── _frameIndex: int                            # 帧索引
│   └── _lastWristLogTime / _lastFullJointLogTime   # 日志时间控制
└── 核心方法
    ├── Awake() → InitializeNetMQ() + LoadNetworkConfig()
    ├── CreateStandardSockets()                      # 创建标准套接字
    ├── SendMessage()                                # 发送消息
    ├── PerformDiagnosticTests()                     # 诊断测试
    └── CleanupNetMQ()                               # 清理资源
```

#### 初始化流程

```
Awake()
  ├── 单例检查（DontDestroyOnLoad）
  ├── InitializeNetMQ()          # 初始化NetMQ运行时
  │   └── AsyncIO.ForceDotNet.Force()
  └── LoadNetworkConfig()        # 从JSON加载网络配置
      └── Resources.Load("Configurations/Network")

GestureDetectorXR.Start()
  └── StartCoroutine(InitializeNetMQAfterDelay())  # 延迟2秒
      ├── NetMQController.Instance.CreateStandardSockets()  # 创建套接字
      └── NetMQController.Instance.PerformDiagnosticTests() # 诊断测试
```

**为什么延迟2秒初始化？**
Unity中OpenXR子系统需要时间加载，如果过早初始化网络套接字，
可能在手部数据还没准备好时就尝试发送，导致不必要的错误。
延迟2秒确保XR子系统完全就绪后再建立网络连接。

#### 套接字创建 (`CreateStandardSockets`)

```csharp
public void CreateStandardSockets()
{
    // IP地址优先级：JSON配置 > PlayerPrefs > 跳过创建
    
    // 创建4个标准PushSocket
    CreateSocket("RightHand",  $"tcp://{ipAddress}:{rightKeypointPort}");   // 8087
    CreateSocket("LeftHand",   $"tcp://{ipAddress}:{leftKeypointPort}");    // 8110
    CreateSocket("Resolution", $"tcp://{ipAddress}:{resolutionPort}");      // 8095
    CreateSocket("Pause",      $"tcp://{ipAddress}:{pausePort}");           // 8100
}
```

**套接字配置参数**：
- `SendHighWatermark = 1000`：高水位标记，缓存最多1000条未发送消息
- `Linger = 100ms`：关闭时等待100ms让消息发送完成
- 套接字类型：`PushSocket`（ZMQ PUSH模式，与BeaVR-bot的PULL模式配对）

#### 消息发送 (`SendMessage`)

```csharp
public bool SendMessage(string socketName, string message)
{
    // 1. 查找套接字
    var socket = sockets[socketName];
    
    // 2. 带超时发送（10ms超时）
    bool sent = socket.TrySendFrame(TimeSpan.FromMilliseconds(10), message);
    
    // 3. 失败处理
    if (!sent)
    {
        socketFailCounts[socketName]++;
        if (socketFailCounts[socketName] > 5)
            ReconnectSocket(socketName);  // 连续5次失败则重连
        return false;
    }
    
    // 4. 成功后：频率统计 + 数据日志打印
    // ...（详见下方"日志打印"部分）
    
    return true;
}
```

**超时保护机制**：
- 发送超时设为10ms，避免阻塞Unity主线程
- 连续失败5次后自动重连套接字
- 连续异常3次后也触发重连
- 重连时关闭旧套接字、创建新PushSocket、重新连接

#### 日志打印功能

NetMQController 在消息发送成功后，会定期打印三类日志：

**1. 转发频率统计**（每1秒）：
```
[App→Bot] RightHand 转发频率: 30.0 Hz
```

**2. 转发位姿信息**（每2秒）：
```
[App→Bot] index=123 RightHand 转发数据: relative:0.090,0.827,0.080|0.073,0.879,0.090|...
```

**3. 手腕部数据打印**（每2秒，仅Hand类型消息）：
```
[App→Bot] index=124 RightHand 手腕数据: 手腕=0.090,0.827,0.080 手掌=0.073,0.879,0.090
```
解析逻辑：从消息字符串中提取第0个关节（手腕）和第1个关节（手掌）的坐标。

**4. 26个坐标系数据打印**（每5秒，仅Hand类型消息）：
```
[App→Bot] index=125 RightHand 26关节数据: 0:0.090,0.827,0.080 1:0.073,0.879,0.090 2:... 25:...
```
解析逻辑：从消息字符串中提取所有26个关节的坐标，并添加索引标注。

**帧索引（_frameIndex）**：
每次打印日志时递增，用于在三个环节（PICO获取→App转发→Bot接收）中
匹配同一时间点的数据，避免因时间差导致的数据错位。

#### 诊断测试 (`PerformDiagnosticTests`)

```csharp
public bool PerformDiagnosticTests()
{
    // 向每个套接字发送测试消息
    foreach (var socketName in sockets.Keys)
    {
        string testMsg = $"DIAGNOSTIC_TEST_{socketName}_{DateTime.Now:HH:mm:ss.fff}";
        bool success = SendMessage(socketName, testMsg);
        // 记录测试结果
    }
    return allSuccessful;
}
```

诊断测试在套接字创建后自动执行，验证所有网络通道是否通畅。
BeaVR-bot端的PULL套接字会收到这些测试消息，但会被当作无效数据丢弃。

#### 套接字重连 (`ReconnectSocket`)

```csharp
private void ReconnectSocket(string socketName)
{
    // 1. 关闭旧套接字
    sockets[socketName].Close();
    sockets[socketName].Dispose();
    
    // 2. 根据套接字名称确定地址
    string address = socketName switch
    {
        "RightHand"  => $"tcp://{ipAddress}:{rightKeypointPort}",
        "LeftHand"   => $"tcp://{ipAddress}:{leftKeypointPort}",
        "Resolution" => $"tcp://{ipAddress}:{resolutionPort}",
        "Pause"      => $"tcp://{ipAddress}:{pausePort}",
    };
    
    // 3. 创建新套接字并连接
    var socket = new PushSocket();
    socket.Options.SendHighWatermark = 1000;
    socket.Options.Linger = TimeSpan.FromMilliseconds(100);
    socket.Connect(address);
    sockets[socketName] = socket;
}
```

#### 网络配置 (`NetworkSettings`)

配置从 `Resources/Configurations/Network.json` 加载：

```csharp
[Serializable]
public class NetworkSettings
{
    public string IPAddress;           // BeaVR-bot服务器IP地址
    public string rightkeyptPortNum;   // 右手关键点端口 (8087)
    public string leftkeyptPortNum;    // 左手关键点端口 (8110)
    public string camPortNum;          // 相机端口
    public string graphPortNum;        // 图形反馈端口
    public string resolutionPortNum;   // 分辨率控制端口 (8095)
    public string PausePortNum;        // 暂停控制端口 (8100)
    public string LeftPausePortNum;    // 左手暂停端口
    public string RightPausePortNum;   // 右手暂停端口
}
```

#### 资源清理

```csharp
private void OnApplicationQuit()
{
    CleanupNetMQ();
}

public void CleanupNetMQ()
{
    CloseAllSockets();        // 关闭所有套接字
    NetMQConfig.Cleanup(false); // 清理NetMQ运行时
}
```

Unity应用退出时自动清理所有ZMQ资源，防止资源泄漏。

---

### 2.3 BeaVR-bot 接收端 (`pico4.py`)

**文件路径**：`beavr-bot/src/beavr/teleop/components/detector/vr/pico4.py`

#### 核心功能
- 使用 ZMQ PULL 套接字接收 PICO4 Unity 应用发送的手部数据
- 解析原始字符串格式为结构化的 `InputFrame` 对象
- 通过 ZMQ PUB 套接字发布给下游的坐标变换组件
- 处理按钮事件和暂停/恢复命令

#### 端口映射

| 数据 | 接收端口 | 发布端口 | 发布Topic |
|------|----------|----------|-----------|
| 右手关键点 | 8087 | 8088 | "right" |
| 左手关键点 | 8110 | 8088 | "left" |
| 按钮事件 | 8095 | 8088 | "button" |
| 暂停/恢复 | 8100 | 8088 | "pause" |

---

### 2.4 坐标变换 (`keypoint_transform.py`)

**文件路径**：`beavr-bot/src/beavr/teleop/components/detector/vr/keypoint_transform.py`

#### 核心功能
- 以手腕为原点进行坐标平移
- 旋转矩阵变换（当前使用单位矩阵，保持原始坐标不变）
- 计算手部方向帧（基于食指/中指/小指关节）
- 滑动平均平滑（窗口大小5帧）
- Gram-Schmidt正交化

---

### 2.5 运动重定向 (`xarm7_operator.py`)

**文件路径**：`beavr-bot/src/beavr/teleop/components/operator/robots/xarm7_operator.py`

#### 核心功能
- 订阅变换后的手部方向帧
- 计算手部相对运动（当前帧相对于初始帧）
- 通过 H_R_V 和 H_T_V 变换矩阵映射到机器人坐标系
- 互补滤波器平滑
- 输出 CartesianTarget（位置+四元数姿态）

---

### 2.6 机器人接口 (`sysmo32_robot.py`)

**文件路径**：`beavr-bot/src/beavr/teleop/components/interface/robots/sysmo32_robot.py`

#### 核心功能
- 接收 CartesianTarget 命令
- 驱动 SYSMO-32 双臂机器人运动（仿真模式）
- 发布机器人当前状态（关节位置、笛卡尔位姿等）
- 每臂6个关节，共12个关节

---

### 2.7 MuJoCo仿真 (`mujoco_sim.py`)

**文件路径**：`beavr-bot/src/beavr/teleop/components/simulation/mujoco_sim.py`

#### 核心功能
- 加载 SYSMO-32 URDF 模型到 MuJoCo
- 动态添加末端执行器 site（left_endeff, right_endeff）
- 订阅 CartesianTarget 命令
- 使用 Jacobian 伪逆 IK 求解器计算关节角度
- 驱动仿真中的机器人双臂运动
- 提供可视化渲染窗口

---

## 3. 端口总览

| 端口号 | 常量名 | 用途 | 方向 |
|--------|--------|------|------|
| 8087 | RIGHT_HAND_PICO4_PORT | 右手原始数据 | Unity→Bot |
| 8110 | LEFT_HAND_PICO4_PORT | 左手原始数据 | Unity→Bot |
| 8088 | KEYPOINT_STREAM_PORT | 关键点发布 | pico4→transform |
| 8092 | KEYPOINT_TRANSFORM_PORT | 右手变换后数据 | transform→operator |
| 8093 | LEFT_KEYPOINT_TRANSFORM_PORT | 左手变换后数据 | transform→operator |
| 8095 | RESOLUTION_BUTTON_PORT | 分辨率按钮 | Unity→Bot |
| 8100 | TELEOP_RESET_PORT | 暂停/恢复 | Unity→Bot |
| 10011 | SYSMO32右臂命令 | 右臂末端命令 | operator→robot |
| 10013 | SYSMO32左臂命令 | 左臂末端命令 | operator→robot |

## 4. 数据格式汇总

### 4.1 Unity端 → BeaVR-bot（字符串格式）

```
<type_marker>:x1,y1,z1|x2,y2,z2|...|x26,y26,z26:
```

### 4.2 BeaVR-bot内部（Python对象格式）

| 数据类型 | 定义文件 | 用途 | 环节 |
|----------|----------|------|------|
| `InputFrame` | detector_types.py | 手部关键点帧 | pico4→transform→operator |
| `CartesianTarget` | operator_types.py | 笛卡尔空间目标 | operator→robot |
| `CartesianState` | interface_types.py | 笛卡尔状态反馈 | robot→recorder |
| `ButtonEvent` | detector_types.py | 按钮事件 | pico4→operator |
| `SessionCommand` | detector_types.py | 会话命令 | pico4→operator/robot |

## 5. 关键代码文件索引

| 文件路径 | 功能 | 数据流环节 |
|----------|------|-----------|
| `BeaVR-app/.../GestureDetectorXR.cs` | PICO4手部数据采集 | 环节一 |
| `BeaVR-app/.../NetMQController.cs` | ZMQ网络转发 | 环节二 |
| `beavr-bot/.../detector/vr/pico4.py` | 数据接收与解析 | 环节三 |
| `beavr-bot/.../detector/vr/keypoint_transform.py` | 坐标变换与平滑 | 环节四 |
| `beavr-bot/.../operator/robots/xarm7_operator.py` | 运动重定向 | 环节五 |
| `beavr-bot/.../interface/robots/sysmo32_robot.py` | SYSMO-32机器人接口 | 环节六 |
| `beavr-bot/.../simulation/mujoco_sim.py` | MuJoCo仿真环境 | 环节六(仿真) |
| `beavr-bot/.../configs/robots/sysmo32_config.py` | SYSMO-32配置 | 配置 |
| `beavr-bot/.../configs/constants/ports.py` | 端口常量 | 配置 |
| `beavr-bot/.../configs/constants/network.py` | 网络常量 | 配置 |
