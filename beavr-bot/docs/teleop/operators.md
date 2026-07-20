# 操作器(Operators)

操作器将 VR 输入转换为机器人指令。它们定义在 `beavr.teleop.components.operators` 中，并继承自基础 `Operator` 类。

一个操作器会订阅手部关键点数据，应用重定向（可参考 `xarm_base.py` 中的示例），并将动作发送到机器人接口。操作器还会发布可被记录的状态信息。

FA 双臂当前复用 `XArmOperator` 的相对位姿 retargeting，并通过 `FaOperator` 注入 FA 的 `pelvis/base` 坐标映射。详细链路见 [FA 遥操作坐标变换逻辑](fa_coordinate_transform.md)。

通过 `cleanup()` 方法可实现优雅关闭，该方法会停止网络订阅并关闭套接字连接。
