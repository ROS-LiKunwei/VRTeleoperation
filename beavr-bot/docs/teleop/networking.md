# 网络通信

所有运行时通信都使用 ZeroMQ。IP 地址和端口由 `beavr.teleop.configs.constants` 定义，并通过 `NetworkConfig` 和 `PortsConfig` 数据类对外暴露。

默认会话中常用的端口包括：

| 用途 | 配置属性 | 默认值 |
|------|----------|--------|
| 手部关键点流 | `keypoint_stream_port` | 8000 |
| 机器人控制指令 | `control_stream_port` | 8001 |
| 机器人状态发布 | `robot_state_port` | 8002 |

`HandshakeCoordinator` 类（见 `beavr.teleop.utils.network`）通过要求所有已注册订阅者返回确认（acknowledgement），为 stop/resume 事件提供可靠传递。

你可以通过命令行自定义任意地址或端口：

```bash
python -m beavr.teleop.main --teleop.network.host_address=192.168.1.50 --teleop.ports.control_stream_port=9001