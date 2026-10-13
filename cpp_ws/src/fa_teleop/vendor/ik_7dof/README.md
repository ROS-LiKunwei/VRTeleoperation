# ik_7dof

`ik_7dof` 提供 FA 机器人上肢正逆运动学求解器，以及两个常用测试节点：

- `fa_arm_kinematic_node`：纯数值正逆解一致性验证
- `arm_ik_rviz_node`：结合 MoveIt/RViz 做可视化验证与执行

## 参考坐标系

新版 `fa_ik_solver` 支持通过 `ArmKinematicsOptions` 选择参考坐标系：

- `pelvis`：以 `pelvis` 为参考坐标系
- `arm_base`：以手臂基坐标系为参考坐标系

当前测试节点通过参数 `reference_frame` 选择：

```text
reference_frame:=pelvis
reference_frame:=arm_base
```

默认值为 `pelvis`。

## fa_ik_solver 用法

```cpp
#include "ik_7dof/fa_ik_solver.hpp"

using namespace fa_arm_kinematic;

IKSolver solver(urdf_file, srdf_file);

ArmKinematicsOptions options;
options.reference_frame = ArmReferenceFrame::PELVIS;

Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
PoseSE3 fk = solver.computeArmFK_SE3(q, ArmSide::LEFT, options);

pinocchio::SE3 target;
target.translation(fk.p);
target.rotation(fk.R);

IKResult result = solver.solveArmIK(
    target, ArmSide::LEFT, Eigen::VectorXd(), options, 1000, 1e-3);
```

## 纯数值测试

```bash
ros2 run ik_7dof fa_arm_kinematic_node --ros-args \
  -p urdf_file:=/home/likunwei/humanoid_ws/src/sysmo_description/urdf/fa_robot.urdf \
  -p srdf_file:=/home/likunwei/humanoid_ws/src/fa_moveit2_config/config/fa_robot.srdf \
  -p arm_side:=left \
  -p reference_frame:=pelvis \
  -p num_tests:=100 \
  -p max_iters:=500 \
  -p eps:=1e-3
```

节点输出内容包括：

- 关节名称和限位
- 每次测试的随机目标关节角
- 指定参考坐标系下的 FK 目标位姿
- IK 求解结果
- 正逆解位置/姿态误差
- 平均耗时、平均迭代步数、成功率

失败样本会写入当前工作目录下的：

```text
fa_left_arm_ik_failed_cases.log
fa_right_arm_ik_failed_cases.log
```

## 自动化测试

开启 `BUILD_TESTING` 编译 `ik_7dof`：

```bash
colcon build --packages-select ik_7dof --cmake-args -DBUILD_TESTING=ON
```

执行 `fa_ik_approx_verify` 测试并直接在终端显示日志输出：

```bash
colcon test --packages-select ik_7dof --event-handlers console_direct+ --ctest-args -R fa_ik_approx_verify
```

也可以直接运行测试可执行文件查看完整打印：

```bash
./build/ik_7dof/fa_ik_approx_verify
```

## RViz/MoveIt 测试

```bash
ros2 run ik_7dof arm_ik_rviz_node --ros-args \
  -p urdf_file:=/home/likunwei/humanoid_ws/src/sysmo_description/urdf/fa_robot.urdf \
  -p srdf_file:=/home/likunwei/humanoid_ws/src/fa_moveit2_config/config/fa_robot.srdf \
  -p arm_side:=left \
  -p reference_frame:=pelvis \
  -p num_tests:=5 \
  -p max_iters:=1000 \
  -p eps:=1e-3 \
  -p move_delay:=1.0
```

说明：

- 节点会先移动到 `leftArmHome` 或 `rightArmHome`
- 然后随机采样关节角，基于指定参考坐标系计算 FK/IK
- IK 成功后使用 MoveIt 执行到求解出的关节角
- 同时发布 `end_effector_markers` 方便在 RViz 里观察末端位姿

当 `reference_frame:=pelvis` 时，Marker 发布在 `pelvis` 坐标系下；
当 `reference_frame:=arm_base` 时，Marker 发布在 `waist_pitch_link` 坐标系下。

## 腿部 MuJoCo 验证脚本

脚本路径：

```bash
python3 scripts/leg_ik_mujoco_verify.py --pelvis_height 0.65 --knee_max 0.2 --ankle_x 0.1 --left_ankle_y 0.144 --right_ankle_y -0.144 --ankle_roll 0.0 --ankle_pitch 0.0 --ankle_yaw 0.0
```

该脚本加载 MuJoCo 模型，按指定约束求解双腿关节角，并用 FK 验证结果。当前自动 IK 使用的硬约束为：

- `left_ankle_roll_link.x - pelvis.x == --ankle_x`
- `right_ankle_roll_link.x - pelvis.x == --ankle_x`
- `left_ankle_roll_link.y - pelvis.y == --left_ankle_y`；不传时使用腿部全 0 初始角度下的左脚踝 y 偏移
- `right_ankle_roll_link.y - pelvis.y == --right_ankle_y`；不传时使用腿部全 0 初始角度下的右脚踝 y 偏移
- 脚底高度满足 `--foot_offset_z`
- 脚朝向满足 `--ankle_roll/--ankle_pitch/--ankle_yaw`

### 固定膝关节为 0

不传 `--knee_max` 时，脚本会固定 `left_knee_joint` 和 `right_knee_joint` 为 `0.0`，并尝试满足上述硬约束：

```bash
cd ~/humanoid_ws/src/ik_7dof
python3 scripts/leg_ik_mujoco_verify.py --pelvis_height 0.7798 --headless --sim_time 0
```

如果指定骨盆高度不可达，脚本会失败并打印硬约束残差。

### 指定膝关节最大角并寻找最近骨盆高度

传入 `--knee_max` 后，脚本允许膝关节在 `[0, knee_max]` 范围内弯曲。如果目标 `--pelvis_height` 不可达，会返回最接近目标的可达骨盆高度，并继续用该高度做 FK/仿真验证：

```bash
python3 scripts/leg_ik_mujoco_verify.py --pelvis_height 0.65 --knee_max 0.5 --headless --sim_time 0
```

典型输出：

```text
目标骨盆高度: 0.650000 m
采用骨盆高度: 0.755448 m
骨盆高度偏差: 0.105448 m
膝关节角度上限: left_knee_joint/right_knee_joint <= 0.500000 rad
```

如果 `--knee_max 1.2`，`--pelvis_height 0.65` 当前可直接满足，脚本会保持目标高度：

```bash
python3 scripts/leg_ik_mujoco_verify.py --pelvis_height 0.65 --knee_max 1.2 --headless --sim_time 0
```

也可以手动指定脚踝相对骨盆的目标位姿：

```bash
python3 scripts/leg_ik_mujoco_verify.py \
  --pelvis_height 0.65 \
  --knee_max 1.2 \
  --ankle_x 0.0 \
  --left_ankle_y 0.14 \
  --right_ankle_y -0.14 \
  --ankle_roll 0.0 \
  --ankle_pitch 0.0 \
  --ankle_yaw 0.0 \
  --headless \
  --sim_time 0
```

其中 `--ankle_x/--left_ankle_y/--right_ankle_y` 单位为 m，表示 `ankle_roll_link` 相对 `pelvis` 的目标位置偏移；`--ankle_roll/--ankle_pitch/--ankle_yaw` 单位为 rad，按 `Rz(yaw) @ Ry(pitch) @ Rx(roll)` 转换为目标朝向。

### 参数

| 参数 | 说明 |
|------|------|
| `--pelvis_height` | 目标骨盆高度，单位 m |
| `--knee_max` | 自动 IK 时允许的膝关节最大角，单位 rad；不传时膝关节固定为 `0.0` |
| `--foot_offset_z` | 额外脚底高度偏移，单位 m，默认 `0.0` |
| `--ankle_x` | 左右 `ankle_roll_link.x - pelvis.x` 的目标偏移，单位 m，默认 `0.0` |
| `--left_ankle_y` | `left_ankle_roll_link.y - pelvis.y` 的目标偏移，单位 m；不传时使用腿部全 0 初始偏移 |
| `--right_ankle_y` | `right_ankle_roll_link.y - pelvis.y` 的目标偏移，单位 m；不传时使用腿部全 0 初始偏移 |
| `--ankle_roll` / `--ankle_pitch` / `--ankle_yaw` | 左右脚踝目标朝向 RPY，单位 rad，默认均为 `0.0` |
| `--left_q` / `--right_q` | 手动指定左右腿 6 个关节角；不能和 `--knee_max` 同时使用 |
| `--headless` | 不启动 MuJoCo viewer，只输出验证结果 |
| `--sim_time` | headless/viewer 模式下保持姿态仿真的时间，默认 `10.0` 秒；设为 `0` 只做初始 FK 验证 |
