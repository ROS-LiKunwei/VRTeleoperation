# beavr-app架构说明

## 概述

beavr-app是基于Unity开发的VR遥操作前端应用，负责采集VR设备（如PICO4）的输入数据，并通过ZMQ协议将数据发送到beavr-bot后端进行机器人控制。

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Unity应用层                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 手势检测     │  │ 网络管理     │  │ UI控制       │  │
│  │ GestureDetector│  │ NetworkManager│  │ UserInterface │  │
│  │     XR       │  │              │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │
│         │                 │                              │
│         │                 │                              │
│  ┌──────▼─────────────────▼──────────────────┐        │
│  │         ZMQ通信层                         │        │
│  │      NetMQController                      │        │
│  └────────────────────┬───────────────────────┘        │
└───────────────────────┼─────────────────────────────────┘
                        │
                        │ ZMQ协议
                        │
┌───────────────────────▼─────────────────────────────────┐
│              beavr-bot后端层                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. 手势检测层 (Gesture Detection)

**主要文件**：
- `Assets/Scripts/Gesture Detection/GestureDetectorXR.cs`

**功能**：
- 使用Unity的XR手部追踪系统采集手部关节数据
- 检测捏合手势（食指、中指、无名指）
- 将手部数据序列化为字符串格式
- 通过ZMQ发送数据到后端

**关键方法**：
- `CollectHandJointPositions()`：采集手部关节位置
- `SendHandDataThroughController()`：发送手部数据
- `StreamPauser()`：处理手势切换
- `IsPinching()`：检测手指是否捏合

**数据流程**：
```
XRHandSubsystem
    ↓ (采集26个关节点）
CollectHandJointPositions()
    ↓ (序列化为字符串）
SerializeVector3List()
    ↓ (添加类型标记）
SendHandDataThroughController()
    ↓ (通过ZMQ发送）
NetMQController.SendMessage()
```

#### 2. 网络管理层 (Network Management)

**主要文件**：
- `Assets/Scripts/NetworkManager.cs`
- `Assets/Scripts/Network/NetMQController.cs`

**功能**：
- 管理网络配置（IP地址、端口号）
- 创建和管理ZMQ套接字
- 处理网络连接和重连
- 发送和接收网络消息

**关键方法**：
- `NetworkManager`：
  - `getRightKeypointAddress()`：获取右手关键点数据地址
  - `getLeftKeypointAddress()`：获取左手关键点数据地址
  - `ConnectAllNetworkComponents()`：连接所有网络组件
  
- `NetMQController`：
  - `CreateSocket()`：创建ZMQ套接字
  - `SendMessage()`：发送消息
  - `ReconnectSocket()`：重新连接套接字
  - `CleanupNetMQ()`：清理网络资源

**套接字管理**：
```
套接字字典 (Dictionary<string, PushSocket>)
    ├── "RightHand" → 右手数据套接字
    ├── "LeftHand" → 左手数据套接字
    ├── "Resolution" → 分辨率控制套接字
    └── "Pause" → 暂停控制套接字
```

#### 3. UI控制层 (User Interface)

**主要文件**：
- `Assets/Scripts/UI/` (各种UI控制器）

**功能**：
- 显示网络连接状态
- 提供IP地址输入界面
- 显示手部追踪状态
- 提供分辨率切换按钮

**UI组件**：
- IP地址输入框
- 连接/断开按钮
- 状态指示器（红色=未连接，绿色=已连接）
- 分辨率按钮（高/低分辨率）

#### 4. ZMQ通信层 (ZMQ Communication)

**主要文件**：
- `Assets/Scripts/Network/NetMQController.cs`

**功能**：
- 实现ZMQ协议的Push-Pull模式
- 管理套接字生命周期
- 处理网络异常和重连
- 提供诊断测试功能

**通信模式**：
```
Unity (Push) ──ZMQ──> beavr-bot (Pull)
```

**消息格式**：
```
手部数据：type:x1,y1,z1|x2,y2,z2|...|x26,y26,z26:
控制数据：High/Low/None
```

## 数据流程

### 手部追踪数据流程

```
1. PICO4设备
   └─> XRHandSubsystem采集手部数据

2. GestureDetectorXR
   ├─> CollectHandJointPositions() 采集26个关节点
   ├─> SerializeVector3List() 序列化为字符串
   └─> SendHandDataThroughController() 发送数据

3. NetMQController
   └─> SendMessage() 通过ZMQ发送

4. 网络
   └─> TCP/IP传输

5. beavr-bot
   └─> PICO4VRHandDetector接收并处理数据
```

### 控制指令流程

```
1. 用户手势
   ├─> 食指捏合 → 相对数据模式
   ├─> 中指捏合 → 绝对数据模式
   └─> 无名指捏合 → 停止遥操作

2. GestureDetectorXR
   ├─> StreamPauser() 检测手势
   └─> SendPauseStatusThroughController() 发送状态

3. NetMQController
   └─> SendMessage() 发送控制指令

4. beavr-bot
   └─> 接收并执行控制指令
```

## 配置管理

### 网络配置

**配置文件**：`Assets/Resources/Configurations/Network.json`

**配置项**：
```json
{
  "IPAddress": "undefined",           // 服务器IP地址（运行时设置）
  "rightkeyptPortNum": "8088",      // 右手关键点端口
  "leftkeyptPortNum": "8111",       // 左手关键点端口
  "resolutionPortNum": "8095",      // 分辨率控制端口
  "PausePortNum": "8100"            // 暂停控制端口
}
```

### 运行时配置

**PlayerPrefs**：
- `ServerIP`：服务器IP地址（通过UI设置）

## 关键技术点

### 1. XR手部追踪

**技术栈**：
- Unity XR插件
- XR Hand子系统
- OpenXR运行时

**手部关节定义**：
- 26个关节点
- 每个关节包含位置和旋转信息
- 支持左右手独立追踪

### 2. ZMQ通信

**技术栈**：
- NetMQ库（ZeroMQ的.NET实现）
- Push-Pull模式
- TCP/IP传输

**优势**：
- 高性能、低延迟
- 自动重连机制
- 支持多种消息模式

### 3. 异步处理

**协程使用**：
IEnumerator 关键字代表这是一个协程。在 Unity 中，协程最大的本事就是可以“分步执行代码”，让你在等待的时候不会把整个程序的画面卡死。
```csharp
IEnumerator InitializeNetMQAfterDelay()
{
    yield return new WaitForSeconds(2f);
    NetMQController.Instance.CreateStandardSockets();
}
```

**优势**：
- 避免阻塞主线程
- 提高应用响应性
- 支持延迟初始化

### 4. 错误处理

**异常捕获**：
```csharp
try
{
    NetMQController.Instance.SendMessage("RightHand", data);
}
catch (Exception e)
{
    Debug.LogError("发送错误: " + e.Message);
}
```

**重连机制**：
- 自动检测连接失败
- 尝试重新连接
- 记录失败次数

## 性能优化

### 1. 数据传输优化

**策略**：
- 使用高效的序列化格式
- 减少数据包大小
- 批量发送数据

**实现**：
```csharp
// 序列化为紧凑格式
string vectorString = "";
foreach (Vector3 vec in gestureData)
    vectorString += vec.x + "," + vec.y + "," + vec.z + "|";
```

### 2. 网络优化

**策略**：
- 使用连接池
- 减少连接建立开销
- 优化消息发送频率

**实现**：
```csharp
// 套接字重用
if (!sockets.ContainsKey(socketName))
{
    sockets[socketName] = new PushSocket();
    socket.Connect(address);
}
```

### 3. 渲染优化

**策略**：
- 减少UI更新频率
- 使用对象池
- 优化渲染管线

## 扩展性设计

### 1. 支持多种VR设备

**设计模式**：
- 抽象基类
- 设备特定实现
- 工厂模式创建

**示例**：
```csharp
public abstract class VRHandDetector : MonoBehaviour
{
    public abstract void CollectHandData();
}

public class PICO4HandDetector : VRHandDetector
{
    public override void CollectHandData()
    {
        // PICO4特定实现
    }
}
```

### 2. 插件架构

**设计原则**：
- 模块化设计
- 接口定义
- 动态加载

**优势**：
- 易于扩展
- 降低耦合度
- 提高可维护性

## 调试和测试

### 1. 日志系统

**日志级别**：
- Debug：调试信息
- Warning：警告信息
- Error：错误信息

**日志输出**：
```csharp
Debug.Log("调试信息");
Debug.LogWarning("警告信息");
Debug.LogError("错误信息");
```

### 2. 诊断工具

**NetMQController诊断**：
```csharp
public bool PerformDiagnosticTests()
{
    // 测试所有套接字
    // 返回测试结果
}
```

### 3. 性能监控

**关键指标**：
- 帧率（FPS）
- 网络延迟
- 数据传输速率
- 内存使用量

## 安全考虑

### 1. 网络安全

**措施**：
- 验证IP地址格式
- 限制连接尝试次数
- 使用加密通信（可选）

### 2. 数据安全

**措施**：
- 不记录敏感数据
- 安全存储配置
- 定期清理日志

### 3. 操作安全

**措施**：
- 紧急停止功能
- 连接状态监控
- 错误恢复机制

## 总结

beavr-app是一个功能完善的VR遥操作前端应用，具有以下特点：

1. **模块化设计**：各组件职责清晰，易于维护和扩展
2. **高性能**：使用ZMQ协议，确保低延迟数据传输
3. **易用性**：提供直观的UI界面，简化操作流程
4. **可靠性**：完善的错误处理和重连机制
5. **可扩展性**：支持多种VR设备和自定义功能

通过理解beavr-app的架构，开发者可以更好地进行功能扩展和问题排查。
