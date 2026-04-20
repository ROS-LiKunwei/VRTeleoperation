# 网络连接测试计划

## 修复内容

1. **修复了 NetworkManager 中的 IP 地址检查逻辑**
   - 更新了 `isIPAllocated()` 方法，使其同时检查 IP 地址是否为空或等于 "undefined"

2. **修复了 NetworkConnectionTest 中的错误**
   - 修复了 `UpdateStatus()` 方法，使其追加文本而不是覆盖
   - 添加了 `ClearStatus()` 方法，确保每次测试前清除之前的状态文本

3. **创建了网络诊断脚本**
   - 添加了 `NetworkDiagnosticTest.cs` 脚本，用于详细检查网络连接问题

## 测试步骤

### 1. 检查网络配置

1. 打开 `Assets/Resources/Configurations/Network.json` 文件
2. 确保 IP 地址设置正确（默认为 192.168.1.133）
3. 确保端口设置正确：
   - 右手关键点端口: 8088
   - 左手关键点端口: 8111
   - 摄像头端口: 10505
   - 图形反馈端口: 15001

### 2. 检查后端服务

1. 确保后端程序正在运行
2. 后端程序应该监听以下端口：
   - 8088 (右手关键点)
   - 8111 (左手关键点)
   - 8095 (分辨率控制)
   - 8100 (暂停控制)

### 3. 测试网络连接

1. 在 Unity 编辑器中打开 `MainScene.unity`
2. 运行游戏
3. 检查游戏界面上的 IP 地址是否正确显示
4. 点击 "Start Teleop" 按钮
5. 观察网络连接状态

### 4. 使用网络诊断工具

1. 在场景中添加一个空游戏对象
2. 将 `NetworkDiagnosticTest.cs` 脚本添加到该游戏对象
3. 在 Inspector 面板中，将一个 TextMeshProUGUI 对象分配给 `diagnosticText` 属性
4. 运行游戏
5. 查看诊断输出，了解网络连接的详细状态

## 常见问题排查

### 1. IP 地址显示为 "Unknown"
- 检查 `PlayerPrefs` 中是否保存了 IP 地址
- 检查 `Network.json` 文件中的 IP 地址设置

### 2. 网络连接失败
- 检查后端程序是否正在运行
- 检查 IP 地址是否正确
- 检查网络连接是否正常
- 检查防火墙设置是否阻止了端口访问

### 3. 套接字创建失败
- 检查端口是否被占用
- 检查后端程序是否正确监听了相应的端口

## 日志查看

在 Unity 编辑器的 Console 窗口中，可以看到详细的网络连接日志，帮助诊断问题。

## 后端服务启动命令

如果后端服务尚未启动，可以使用以下命令启动：

```bash
python -m beavr.teleop.main --teleop.network.host_address=192.168.1.133
```

确保替换为正确的 IP 地址。