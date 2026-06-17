# fa_rviz_command_bridge

This ROS2 package visualizes the FA native upper-body command stream in RViz.

It subscribes to `/upper_position_controller/commands` as `std_msgs/msg/Float64MultiArray`.
The expected command length is 16:

- `data[0:7]`: left arm joints
- `data[7:14]`: right arm joints
- `data[14:16]`: neck yaw/pitch joints

Every accepted command is interpolated with the seventh-order min-snap
polynomial `35t^4 - 84t^5 + 70t^6 - 20t^7` over five samples by default and
published as `sensor_msgs/msg/JointState`.

## Build

```bash
cd /home/likunwei/dataCollection/beavr-bot
colcon build \
  --base-paths robots/fa_description robots/fa_rviz_command_bridge \
  --packages-select fa_description fa_rviz_command_bridge
source install/setup.bash
```

## Run

```bash
ros2 launch fa_rviz_command_bridge fa_command_rviz.launch.py
```

If MuJoCo is already publishing `/joint_states`, launch only RViz and
`robot_state_publisher`:

```bash
ros2 launch fa_rviz_command_bridge fa_command_rviz.launch.py use_command_bridge:=false
```

By default the launch file uses:

- URDF: `/home/likunwei/dataCollection/beavr-bot/robots/fa_description/urdf/fa_robot.urdf`
- input command topic: `/upper_position_controller/commands`
- joint state topic: `/joint_states`
- interpolation profile: `min_snap`

Do not publish `/joint_states` from both this bridge and the real robot driver at the same time.
