# sysmo_rviz_command_bridge

This ROS2 package visualizes the SYSMO-32 arm command stream in RViz.

It subscribes to `/sysmo_left_arm_controller/commands` as `std_msgs/msg/Float64MultiArray`.
The expected command length is 18:

- `data[0:6]`: left arm joints
- `data[6:12]`: right arm joints
- `data[12:18]`: speed/reserved/neck fields, ignored for arm RViz display

Every accepted command is interpolated with a fifth-order polynomial
`10t^3 - 15t^4 + 6t^5` over five samples by default and published as
`sensor_msgs/msg/JointState`.

## Build

```bash
cd /home/likunwei/dataCollection/beavr-bot
colcon build \
  --base-paths robots/sysmo_description robots/sysmo_rviz_command_bridge \
  --packages-select sysmo_description sysmo_rviz_command_bridge
source install/setup.bash
```

## Run

```bash
ros2 launch sysmo_rviz_command_bridge sysmo32_command_rviz.launch.py
```

By default the launch file uses:

- URDF: `/home/likunwei/dataCollection/beavr-bot/robots/sysmo_description/urdf/sysmo32.urdf`
- input command topic: `/sysmo_left_arm_controller/commands`
- joint state topic: `/joint_states`

For pure simulation, publishing `/joint_states` matches the real robot feedback
interface. Do not run this bridge while the real robot driver is also publishing
`/joint_states`; in that case RViz should use the real driver feedback directly.

To only run the bridge when your own RViz and `robot_state_publisher` are already running:

```bash
ros2 launch sysmo_rviz_command_bridge sysmo32_command_rviz.launch.py \
  use_robot_state_publisher:=false \
  use_rviz:=false
```
