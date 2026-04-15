# 改动
1. `xarm7_robot.py`中添加了 `--simulation_mode` 参数，用于控制xarm7机器人在仿真模式下运行。`xarm7_control.py` init函数中强制为False
2. 新增xarm7_sim_config.py

运行xarm7仿真使用的命令
```bash
python -m beavr.teleop.main --teleop.network.host_address=192.168.1.50 --teleop.ports.control_stream_port=9001