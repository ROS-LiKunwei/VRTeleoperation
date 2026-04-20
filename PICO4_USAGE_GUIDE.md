# PICO4遥操作使用说明

## 概述

本文档说明如何使用PICO4 VR头显进行人形机器人双臂遥操作数据采集。

## 系统架构

### 数据流程

```
PICO4 VR头显
    ↓ (XR手部追踪)
Unity应用 (beavr-app)
    ↓ (ZMQ通信)
beavr-bot后端
    ↓ (数据转换)
机器人控制器
    ↓ (控制指令)
人形机器人双臂
```

### 关键组件

1. **PICO4 VR头显**：提供手部追踪和手柄输入
2. **beavr-app (Unity)**：采集VR数据并通过ZMQ发送到后端
3. **beavr-bot (Python)**：接收VR数据并转换为机器人控制指令
4. **机器人控制器**：执行控制指令，驱动机器人运动

## 配置说明

### 网络配置

#### beavr-bot配置

文件位置：`beavr-bot/src/beavr/teleop/configs/constants/network.py`

```python
# PICO4 VR手部追踪端口
LEFT_HAND_PICO4_PORT = 8111  # 左手数据端口
RIGHT_HAND_PICO4_PORT = 8088  # 右手数据端口
HOST_ADDRESS = "10.0.0.51"  # 服务器IP地址
```

#### beavr-app配置

文件位置：`BeaVR-app/BeaVR-Unity/Assets/Resources/Configurations/Network.json`

```json
{
  "IPAddress": "undefined",
  "rightkeyptPortNum": "8088",
  "leftkeyptPortNum": "8111",
  "resolutionPortNum": "8095",
  "PausePortNum": "8100"
}
```

**注意**：`IPAddress`字段在运行时通过Unity UI设置，不需要预先配置。

## 使用步骤

### 1. 启动beavr-bot后端

```bash
cd /home/likunwei/dataCollection/beavr-bot
source .venv/bin/activate
python -m beavr.teleop.main --robot_name=leap,xarm7 --teleop.flags.sim_env=True
```

### 2. 配置Unity应用

1. 在Unity编辑器中打开`BeaVR-app/BeaVR-Unity`项目
2. 构建并部署到PICO4设备
3. 在PICO4上启动应用

### 3. 设置网络连接

1. 在Unity应用中，点击"IP Address"输入框
2. 输入beavr-bot运行机器的IP地址（例如：`192.168.1.133`）
3. 点击"Connect"按钮建立连接

### 4. 启动手部追踪

1. 在PICO4主菜单中，点击"设置"
2. 选择"实验室" > "手部追踪"
3. 确保"手部追踪"功能已开启

### 5. 开始遥操作

1. 戴上PICO4头显，确保双手被追踪
2. 使用左手手势控制遥操作模式：
   - **食指捏合**：相对数据模式（绿色边框）
   - **中指捏合**：绝对数据模式（蓝色边框）
   - **无名指捏合**：停止遥操作（红色边框）

## 手部追踪数据格式

### 关节定义

PICO4使用26个关节点来表示手部姿态：

1. Wrist（手腕）
2. Palm（手掌）
3. ThumbMetacarpal（拇指掌骨）
4. ThumbProximal（拇指近节）
5. ThumbDistal（拇指远节）
6. ThumbTip（拇指指尖）
7. IndexMetacarpal（食指掌骨）
8. IndexProximal（食指近节）
9. IndexIntermediate（食指中节）
10. IndexDistal（食指远节）
11. IndexTip（食指指尖）
12. MiddleMetacarpal（中指掌骨）
13. MiddleProximal（中指近节）
14. MiddleIntermediate（中指中节）
15. MiddleDistal（中指远节）
16. MiddleTip（中指指尖）
17. RingMetacarpal（无名指掌骨）
18. RingProximal（无名指近节）
19. RingIntermediate（无名指中节）
20. RingDistal（无名指远节）
21. RingTip（无名指指尖）
22. LittleMetacarpal（小指掌骨）
23. LittleProximal（小指近节）
24. LittleIntermediate（小指中节）
25. LittleDistal（小指远节）
26. LittleTip（小指指尖）

### 数据格式

每个关节点包含3个坐标值（x, y, z），因此每只手的数据包含78个浮点数（26个关节 × 3个坐标）。

数据序列化格式：
```
type:x1,y1,z1|x2,y2,z2|...|x26,y26,z26:
```

其中`type`可以是：
- `relative`：相对数据模式
- `absolute`：绝对数据模式

## 故障排除

### 问题1：PICO4无法识别手势

**解决方案**：
1. 确保PICO4的"手部追踪"功能已开启
2. 检查相机和传感器权限是否已授予
3. 确保环境光线充足
4. 尝试重启PICO4设备

### 问题2：无法连接到beavr-bot

**解决方案**：
1. 检查IP地址是否正确
2. 确保beavr-bot后端正在运行
3. 检查防火墙设置，确保端口8111和8088未被阻止
4. 使用`ping`命令测试网络连通性

### 问题3：数据传输不稳定

**解决方案**：
1. 检查网络连接质量
2. 减少其他网络流量
3. 检查beavr-bot的日志，查看是否有错误信息
4. 重启beavr-bot和Unity应用

### 问题4：手部追踪精度低

**解决方案**：
1. 确保双手在PICO4摄像头的视野内
2. 避免快速移动手部
3. 调整环境光线
4. 检查PICO4的追踪设置

## 测试和验证

### 运行测试脚本

```bash
cd /home/likunwei/dataCollection/beavr-bot
python test_pico4_complete.py
```

此脚本会验证：
- 端口配置是否正确
- 探测器配置是否正确
- Network.json配置是否正确
- 数据格式是否符合预期
- 关键文件是否存在
- 文件注释是否已添加

### 检查日志

#### beavr-bot日志

```bash
# 查看实时日志
tail -f /path/to/beavr-bot.log

# 查看错误日志
grep ERROR /path/to/beavr-bot.log
```

#### Unity应用日志

使用adb工具查看Unity应用的日志：

```bash
adb logcat | grep Unity
```

## 性能优化

### 网络优化

1. 使用有线网络连接，避免WiFi干扰
2. 确保网络带宽充足（建议至少100Mbps）
3. 减少网络延迟（建议延迟<20ms）

### 数据处理优化

1. 调整数据传输频率（默认为60Hz）
2. 使用数据压缩减少带宽占用
3. 优化数据处理算法

## 安全注意事项

1. **机器人安全**：
   - 确保机器人在安全的环境中运行
   - 设置紧急停止按钮
   - 定期检查机器人状态

2. **数据安全**：
   - 不要在公共网络中传输敏感数据
   - 使用加密通信（如果需要）
   - 定期备份数据

3. **操作安全**：
   - 遵循机器人操作规程
   - 不要在疲劳时操作
   - 确保操作区域无障碍物

## 扩展功能

### 添加新的手势识别

在`GestureDetectorXR.cs`中添加新的手势检测逻辑：

```csharp
// 检测新的捏合手势
bool pinchNew = IsPinching(left, XRHandJointID.NewFingerTip);

if (pinchNew)
{
    // 执行新的操作
    // ...
}
```

### 支持更多VR设备

参考`PICO4VRHandDetector`的实现，为其他VR设备创建类似的探测器类。

## 技术支持

如有问题，请联系技术支持团队或查看项目文档：

- 项目GitHub仓库
- 技术文档
- 问题追踪系统

## 更新日志

### v1.0.0 (2025-01-XX)
- 初始版本
- 支持PICO4手部追踪
- 支持双手遥操作
- 添加中文注释和文档
