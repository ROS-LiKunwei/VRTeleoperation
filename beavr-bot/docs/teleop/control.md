# Control Loop

操作员(operator)会获取 VR 关键点，并计算机器人的目标动作。`TeleopControlConfig` 数据类暴露了频率参数，例如用于 VR 输入速率的 `vr_freq`，以及用于数据集记录的 `recorder_freq`

每个操作员(operator)都会运行一个类似下面的循环：

```python
while True:
    pose = hand_subscriber.get()
    command = solver.retarget(pose)
    robot.send_action(command)
```

使用 --teleop.control.vr_freq=60 可以在命令行中修改 VR 轮询频率
