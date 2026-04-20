# TeleVuer

TeleVuer 库是 [Vuer](https://github.com/vuer-ai/vuer) 库的专用版本，设计用于实现基于 XR 设备的宇树机器人遥操作。该库作为 Vuer 的包装器，提供了专门为宇树机器人定制的额外适配。通过集成 XR 设备的功能，如手部追踪和控制器追踪，TeleVuer 促进了在沉浸式环境中对机器人系统的无缝交互和控制。

目前，该模块是 [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) 库的核心组件，为遥操作任务提供高级功能。它支持各种 XR 设备，包括 Apple Vision Pro、Meta Quest 3、Pico 4 Ultra Enterprise 等，确保机器人遥操作应用的兼容性和易用性。

该库的图像输入与 [teleimager](https://github.com/silencht/teleimager) 库配合使用。我们建议同时使用这两个库。

## 0. 🔖 版本说明

### V4.0 🏷️ 更新：

1. 改进的显示模式

    移除了旧的“pass_through”模式。系统现在支持三种模式：

    - immersive: 完全沉浸式模式；VR 显示机器人的第一人称视角（必须启用 zmq 或 webrtc）。

    - pass-through: VR 通过 VR 头显相机显示真实世界；不显示来自 zmq 或 webrtc 的图像（即使已启用）。

    - ego: 中心的小窗口显示机器人的第一人称视角，而周围区域显示真实世界。

2. 增强的沉浸感

    调整了沉浸式和自我模式的图像平面高度，提供更自然、更舒适的 VR 体验

### V3.0 🏷️ 更新：
1. 添加了 `pass_through` 接口来启用/禁用透视模式。
2. 支持 `webrtc` 接口来启用/禁用 webrtc 流模式。
3. 使用 `render_to_xr` 方法（从 `set_display_image` 调整）将图像发送到 XR 设备。

### V2.0 🏷️ 更新：

1. 图像传输现在通过引用而不是外部共享内存。
2. 将获取数据的函数从 `get_motion_state_data` 重命名为 `get_tele_data`。
3. 修复了命名错误（`waist` → `wrist`）
4. 变量名与 **vuer** 约定对齐。
5. 简化了数据结构：移除了嵌套的 `TeleStateData`，并在统一的 `TeleData` 中返回所有内容。
6. 添加了新的图像传输接口，如 `set_display_image`。

## 1. 🗺️ 图表

<p align="center">
  <a href="https://oss-global-cdn.unitree.com/static/5ae3c9ee9a3d40dc9fe002281e8aeac1_2975x3000.png">
    <img src="https://oss-global-cdn.unitree.com/static/5ae3c9ee9a3d40dc9fe002281e8aeac1_2975x3000.png" alt="图表" style="width: 50%;">
  </a>
</p>

## 2. 📦 安装

### 2.1 📥 安装 televuer 仓库

```bash
git clone https://github.com/silencht/televuer
cd televuer
pip install -e . # 或 pip install .
```


### 2.2 🔑 生成证书文件

televuer 模块需要 SSL 证书，以允许 XR 设备（如 Pico / Quest / Apple Vision Pro）通过 HTTPS / WebRTC 安全连接。

1. 对于 Pico / Quest XR 设备

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
# 检查生成的文件
$ ls
build  cert.pem  key.pem  LICENSE  pyproject.toml  README.md  src  test
```

2. 对于 Apple Vision Pro

```bash
openssl genrsa -out rootCA.key 2048
openssl req -x509 -new -nodes -key rootCA.key -sha256 -days 365 -out rootCA.pem -subj "/CN=xr-teleoperate"
openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out server.csr -subj "/CN=localhost"
vim server_ext.cnf
# 添加以下内容（确保 IP.2 匹配您主机的 IP 地址，您可以使用 ifconfig 或类似命令找到）：
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 192.168.123.164
IP.2 = 192.168.123.2
# 然后签署证书：
openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out cert.pem -days 365 -sha256 -extfile server_ext.cnf
# 检查生成的文件
$ ls
build  cert.pem  key.pem  LICENSE  pyproject.toml  README.md  rootCA.key  rootCA.pem  rootCA.srl  server.csr  server_ext.cnf  src  test
# 使用 AirDrop 将 rootCA.pem 复制到您的 Apple Vision Pro 设备并手动安装为受信任的证书。
```

3. 🧱 允许防火墙访问

```bash
sudo ufw allow 8012
```

### 2.3 🔐 配置证书路径（选择一种方法）

您可以使用环境变量或用户配置目录告诉 televuer 在哪里找到证书文件。

此配置可以与 [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) 仓库中的 [teleimager](https://github.com/silencht/teleimager) 模块共享。

1. 用户配置目录（推荐）

```bash
# 此仓库属于 xr_teleoperate，因此我们使用其配置目录
mkdir -p ~/.config/xr_teleoperate/
cp cert.pem key.pem ~/.config/xr_teleoperate/
```
2. 环境变量配置（可选）

```bash
# 这使配置在未来的终端会话中保持持久。
echo 'export XR_TELEOP_CERT="your_file_path/cert.pem"' >> ~/.bashrc
echo 'export XR_TELEOP_KEY="your_file_path/key.pem"' >> ~/.bashrc
source ~/.bashrc
```

3. 默认行为

如果未使用上述任何方法，televuer 将从函数参数中查找证书文件，或回退到模块内的默认路径。

## 3. 🧐 测试

```bash
python test_televuer.py 
# 或 
python test_tv_wrapper.py

# 首先，使用 Apple Vision Pro 或 Pico 4 Ultra Enterprise 连接到与您的计算机相同的 Wi-Fi 网络。
# 接下来，打开 safari / pico 浏览器，输入 https://主机IP:8012/?ws=wss://主机IP:8012
# 例如，https://192.168.123.2:8012?ws=wss://192.168.123.2:8012
# 使用适当的方法（手势或控制器）点击屏幕左下角的 "pass-through" 按钮。

# 在终端中按 Enter 启动程序。
```

## 4. 📌 版本历史

`vuer==0.0.32rc7`

- **功能**：
  - 手部追踪工作正常。
  - 不支持控制器追踪。

---

`vuer==0.0.35`

- **功能**：
  - AVP 手部追踪工作正常。
  - PICO 手部追踪工作正常，但右眼在启动时偶尔会短暂黑屏。

---

`vuer==0.0.36rc1` 到 `vuer==0.0.42rc16`

- **功能**：
  - 手部追踪只显示平面 RGB 图像（无立体视图）。
  - PICO 手部和控制器追踪表现相同，启动时偶尔会出现右眼黑屏。
  - 手部或控制器标记显示为黑框（`vuer==0.0.36rc1`）或 RGB 三轴颜色坐标（`vuer==0.0.42rc16`）。

---

`vuer==0.0.42` 到 `vuer==0.0.45`

- **功能**：
  - 手部追踪只显示平面 RGB 图像（无立体视图）。
  - 无法检索手部追踪数据。
  - 控制器追踪只显示平面 RGB 图像（无立体视图），但可以检索控制器数据。

---

`vuer==0.0.46` 到 `vuer==0.0.56`

- **功能**：
  - AVP 手部追踪工作正常。
  - 在 PICO 手部追踪模式下：
    - 使用手势点击 "Virtual Reality" 按钮会导致屏幕保持黑屏并卡住加载。
    - 使用控制器点击按钮工作正常。
  - 在 PICO 控制器追踪模式下：
    - 使用控制器点击 "Virtual Reality" 按钮会导致屏幕保持黑屏并卡住加载。
    - 使用手势点击按钮工作正常。
  - 手部标记可视化显示为 RGB 三轴颜色坐标。

---

`vuer==0.0.60`
- **推荐版本**

- **功能**：
  - 稳定的功能，具有良好的兼容性。
  - 大多数已知问题已解决。
  - 一个小问题：启动后右眼会短暂黑屏。
- **参考**：
  - [GitHub Issue #53](https://github.com/unitreerobotics/xr_teleoperate/issues/53)
  - [GitHub Issue #45](https://github.com/vuer-ai/vuer/issues/45)
  - [GitHub Issue #65](https://github.com/vuer-ai/vuer/issues/65)

---

## 测试设备
请参考我们的维基文档 [XR 设备](https://github.com/unitreerobotics/xr_teleoperate/wiki/XR_Device)

## 注意事项
- **推荐版本**：使用 `vuer==0.0.60` 获得最佳功能和稳定性。
- **黑屏问题**：在 PICO 设备上，根据模式选择适当的交互方法（手势或控制器），以避免黑屏问题。