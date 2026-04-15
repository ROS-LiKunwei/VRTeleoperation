# 接口(interface)

机器人专用适配器位于 `beavr.teleop.interfaces` 下。每个适配器都实现了 `RobotWrapper` API，提供用于查询状态和发送命令的方法。

可用的接口包括：

- `LeapHandRobot` —— 与模拟或真实的 Leap 手进行通信
- `XArm7Robot` —— xArm 机械臂接口
- `RX1RightRobot` —— RX-1 人形机械臂右臂接口

接口通过 `--robot_name` 参数选择。多个机器人可以通过用逗号分隔名称来组合使用。