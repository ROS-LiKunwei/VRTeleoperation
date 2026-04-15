# 遥操作栈(Teleoperation Stack)

遥操作栈负责流式传输 VR 信号、控制机器人接口并采集数据。它以 `beavr.teleop` 中的模块为核心。
该栈从 `beavr/teleop/main.py` 启动，并且可以完全通过命令行或 YAML 文件进行配置。配置内容包括：
- **网络(Networking)** —— 用于通信的 IP 地址和 ZMQ 端口
- **控制(Control)** —— 频率设置和遥操作标志
- **接口(Interfaces)** —— 将指令转换为机器人可执行动作的机器人适配器
- **操作者(Operators)** —— 将 VR 输入重定向为机器人动作的进程
使用 `python -m beavr.teleop.main --robot_name=leap,xarm7 --laterality=bimanual` 启动默认的双臂配置。