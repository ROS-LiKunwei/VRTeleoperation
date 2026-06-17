[![Full Tests](https://github.com/ARCLab-MIT/beavr-bot/actions/workflows/full_tests.yml/badge.svg)](https://github.com/ARCLab-MIT/beavr-bot/actions/workflows/full_tests.yml)
[![GitHub Stars](https://img.shields.io/github/stars/ARCLab-MIT/beavr-bot?style=social)](https://github.com/ARCLab-MIT/beavr-bot)

# BeaVR
**Bimanual, multi-Embodiment, Accessible VR Teleoperation for Robots**

<p align="center">
  <img src="media/BeaVR_logo.svg" alt="BeaVR-Bot Logo" width="300"/>
</p>

<p align="center">
  <strong>Alejandro Posadas-Nava</strong> ·
  <strong>Alejandro Carrasco</strong> ·
  <strong>Richard Linares</strong>
</p>

<p align="center">
  <br>
  <a href="https://arxiv.org/abs/2508.09606">
    <img src="https://yuxiaoba.github.io/assets/images/badges/Arxiv.png" alt="arXiv" width="14" style="vertical-align:middle;"/> Paper
  </a> |
  <a href="https://arclab-mit.github.io/beavr-landing/">
    <img src="https://images.icon-icons.com/3685/PNG/512/github_logo_icon_229278.png" alt="arXiv" width="14" style="vertical-align:middle;"/> Project Page
  </a> |
  <a href="https://github.com/ARCLab-MIT/BeaVR-app">
    <img src="https://images.icon-icons.com/3053/PNG/512/unity_hub_macos_bigsur_icon_189587.png" alt="arXiv" width="16" style="vertical-align:middle;"/> VR App
  </a>
</p>

---

## Overview

BeaVR is an open-source, end-to-end teleoperation pipeline that leverages affordable hardware for robotic teleoperation.

Key features:
- **VR teleoperation out-of-the-box** – Stream low-latency control and visual feedback through Meta Quest 3 (and any OculusVR supported device) while recording synchronized proprioceptive, visual, and action data.
- **Multi-embodiment support** – Ships with drivers and URDF assets for the RX-1 full-size humanoid and an xArm + LeapHand dexterous work-cell. The modular hardware abstraction layer lets you drop in new robots with a single interface file.
- **Simulation parity** – Mirror every real-world session in MuJoCo or Isaac Gym for rapid domain-randomized policy training or sim-to-real transfer.
- **Dexterous demonstration collection** – Capture single-hand, bi-manual, or whole-body demonstrations for manipulation, assembly, or locomotion tasks—no motion-capture stage required.
- **Budget-friendly extensibility** – Works on commodity PCs and laptops with accessible robotics hardware.

### Why use BeaVR?
- **Accessible** – No proprietary hardware or licenses; every component is student-budget friendly and permissively BSD-3-licensed.
- **Modular & maintainable** – Clean and performant Python and ROS modules.
- **LeRobot formatted data** – Standardizing data collection for shared robotics projects.
- **Community-driven** – Contributions already include UR-series arms, quadrupeds, and tactile grippers; PRs with new morphologies are welcome.

## Quick Start

### Prerequisites
- Linux system or containerized environment (see [Docker folder](docker))
- NVIDIA GPU with CUDA support (recommended)
- Meta Quest 3 VR headset
- [uv](https://github.com/astral-sh/uv) package manager (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Build tools: `build-essential`, `python3-dev` (for compiling C extensions)
- Rust compiler (optional, only needed if building from source): Install via [rustup](https://rustup.rs/) or `sudo apt install rustup && rustup default stable`

### Installation

#### uv Environment (Recommended)

For development and running the full system with all dependencies (including PyTorch and simulation tools), we recommend using uv:

```bash
# Install Python 3.10.13
uv python install 3.10.13

# Create virtual environment with Python 3.10.13
uv venv --python 3.10.13

# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

After setting up the environment, install BeaVR as a package using uv:

```bash
# Install all dependencies including dev extras
uv sync --extra dev

# This will create a uv.lock file and install everything
```

Alternatively, you can use pip:
```bash
pip install -e .[dev]
```

Set up pre-commit hooks:
```bash
pre-commit install
```

Verify the installation:
```bash
python -c "import sys;
try: import beavr; print('BeaVR successfully installed!')
except ImportError: print('An error occurred'); sys.exit(1)"
```

## Documentation

Full documentation lives in the [`docs`](docs) directory. Start with
[`docs/README.md`](docs/README.md) for an overview of the available guides,
including detailed explanations of the teleoperation and LeRobot stacks.

## SYSMO-32 Teleoperation Architecture

当前 SYSMO-32 实机遥操作链路是一个双臂统一控制架构。PICO4 Unity App 先通过 ZMQ 把左右手原始追踪数据送入 PICO4 detector；随后 `keypoint_transform.py` 将手部关键点和方向帧转换到内部右手系 VR frame，并分别发布右手、左手 transformed hand frame。

左右 `Sysmo32Operator` 继承 `XArmOperator` 的重定向逻辑，只替换 SYSMO-32 专用的 `H_R_V_SYSMO32` 坐标变换矩阵。Operator 在 reset 时记录机器人当前末端位姿和当前手部基准位姿；正常运行时计算手相对初始帧的运动，将其映射到 robot base frame，生成 `CartesianTarget(frame_id="base")`，并通过 ZMQ `endeff_coords` 发布给下游控制层。

SYSMO-32 真机控制层只启动一个 `Sysmo32RealControl`，不是左右臂各一个控制器。这个组件同时订阅左右 `CartesianTarget`、左右 reset、pause/resume、左右 transformed hand coords，并从 `/joint_states` 获取当前 12 关节反馈。每个控制周期中，它对左右最新笛卡尔目标分别做 IK，然后经过上层 arm smoother、命令限幅和 18 维命令组包。

当前默认 arm smoother 是 `jerk_limited_servo`，用于连续追踪流式 VR 目标；也保留 `min_snap` 和 `none` 模式。命令生成顺序为：

```text
CartesianTarget
  -> Sysmo32MujocoKinematics.solve_ik
  -> Sysmo32JerkLimitedServoSmoother 或 Sysmo32ArmTrajectorySmoother
  -> Sysmo32CommandLimiter
  -> Sysmo32CommandBuilder
  -> Sysmo32ArmCommand
```

真实 arm command 只有一个 ROS2 topic：

```text
/sysmo_left_arm_controller/commands
std_msgs/msg/Float64MultiArray
```

18 维 payload 格式固定为：

```text
data[0:6]    = left_arm_6
data[6:12]   = right_arm_6
data[12]     = speed_mode
data[13:17]  = reserved
data[17]     = neck_joint
```

不要新增 `/sysmo_right_arm_controller/commands`，也不要改变 `/sysmo_left_arm_controller/commands` 的消息类型。手部动作是独立的 `Sysmo32HandAction` 路径，不合并进 18 维 arm command。

`control_backend` 支持三种模式：

- `mujoco`: 只跑 MuJoCo mirror，主要用于 dry-run 和调试。
- `real`: 需要新鲜 `/joint_states`，通过 ROS2 给真机发布 18 维 arm command。
- `real_with_mujoco`: 真机命令路径与 `real` 相同，同时启动 MuJoCo mirror 观察同一条命令。

整体数据流：

```text
PICO4 Unity
  -> PICO4 detector
  -> keypoint_transform left/right
  -> Sysmo32Operator left/right
  -> CartesianTarget via ZMQ endeff_coords
  -> one bimanual Sysmo32RealControl
  -> /joint_states feedback
  -> IK + smoothing + safety limiting
  -> 18D Sysmo32ArmCommand
  -> /sysmo_left_arm_controller/commands
  -> optional MuJoCo mirror
```

## FA Teleoperation Adapter

FA 机器人通过 `robot_name=fa` 走独立配置、Operator、IK client、RealControl 和 16D upper-position command 适配层。FA real 模式不复用 SYSMO-32 的 18D arm command ABI。

```text
PICO4 Unity
  -> PICO4 detector
  -> keypoint_transform left/right
  -> FaOperator left/right
  -> CartesianTarget via ZMQ endeff_coords
  -> one bimanual FaRealControl
  -> /joint_states feedback
  -> FaArmIkClient / ik_7dof::IKSolver
  -> FA 7D seventh-order min-snap smoothing + limiter
  -> 16D Float64MultiArray
  -> /upper_position_controller/commands
  -> optional MuJoCo mirror
```

FA model root:

```text
/home/likunwei/dataCollection/beavr-bot/robots/fa_description
```

Current FA model defaults:

- model file: `robots/fa_description/urdf/fa_robot.urdf`
- SRDF file: `/home/likunwei/humanoid_ws/src/fa_moveit2_config/config/fa_robot.srdf`
- left/right arm joints: `*_shoulder_pitch`, `*_shoulder_roll`, `*_shoulder_yaw`, `*_elbow`, `*_wrist_yaw`, `*_wrist_pitch`, `*_wrist_roll`
- FA end-effector frame: `left_hand_base_link` / `right_hand_base_link`
- C++ IK source: `/home/likunwei/humanoid_ws/src/ik_7dof/include/ik_7dof/fa_ik_solver.hpp`
- Python IK module: `ik_7dof_pybind.FaIkSolver` from `/home/likunwei/humanoid_ws/src/ik_7dof`
- IK reference frame default: `pelvis`
- native command publisher: `FaNativeCommandPublisher`

FA 16D command mapping:

```text
data[0:7]    = left_arm_7
data[7:14]   = right_arm_7
data[14]     = neck_yaw_joint
data[15]     = neck_pitch_joint
```

Launch examples use the existing entrypoint and flags:

```bash
# build the C++ IK pybind module first
cd /home/likunwei/humanoid_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ik_7dof
source install/setup.bash

# dry-run / MuJoCo
cd /home/likunwei/dataCollection/beavr-bot
python -m beavr.teleop.main \
  --robot_name=fa \
  --laterality=bimanual \
  --teleop.flags.robot_interface=True \
  --teleop.flags.sim_env=True \
  --control_backend=mujoco

# real, using FA native /upper_position_controller/commands
python -m beavr.teleop.main \
  --robot_name=fa \
  --laterality=bimanual \
  --teleop.flags.robot_interface=True \
  --control_backend=real

# real + MuJoCo mirror
python -m beavr.teleop.main \
  --robot_name=fa \
  --laterality=bimanual \
  --teleop.flags.robot_interface=True \
  --teleop.flags.sim_env=True \
  --control_backend=real_with_mujoco
```

Robot-board deployment without a full `humanoid_ws`:

```bash
# On the development machine:
cd /home/likunwei/dataCollection/beavr-bot
scripts/package_ik7dof_runtime.sh /home/likunwei/humanoid_ws/install/ik_7dof /tmp/ik_7dof_runtime.tgz

# Copy /tmp/ik_7dof_runtime.tgz to the board, then on the board:
mkdir -p /opt/fa_runtime
tar -C /opt/fa_runtime -xzf /path/to/ik_7dof_runtime.tgz
export IK_7DOF_INSTALL_PREFIX=/opt/fa_runtime/ik_7dof
```

FA RViz command mirror:

```bash
cd /home/likunwei/dataCollection/beavr-bot
source /opt/ros/humble/setup.bash
colcon build \
  --base-paths robots/fa_description robots/fa_rviz_command_bridge \
  --packages-select fa_description fa_rviz_command_bridge
source install/setup.bash

# RViz follows the same 16D command stream as MuJoCo / real mode.
ros2 launch fa_rviz_command_bridge fa_command_rviz.launch.py

# If FA MuJoCo is already publishing /joint_states, avoid duplicate publishers:
ros2 launch fa_rviz_command_bridge fa_command_rviz.launch.py use_command_bridge:=false
```

If the runtime log repeatedly shows `PICO4: 暂未收到Unity原始数据`, no raw hand frames are reaching the BeaVR detector yet. In that state the operator will stay in reset waiting for a valid VR hand initialization frame, so neither MuJoCo nor RViz will move even though the FA model and IK are loaded.

For a headless smoke test:

```bash
ros2 launch fa_rviz_command_bridge fa_command_rviz.launch.py use_rviz:=false
ros2 topic pub --once /upper_position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.2, 0.0, -0.3, 0.0, 0.1, 0.0, 0.0, -0.2, 0.0, -0.3, 0.0, 0.1, 0.0, 0.0, 0.0]}"
ros2 topic echo --once /joint_states
```

Known fields that still need confirmation before hardware operation:

- physical TCP frame/site relative to the wrist/hand link
- whether `base` in BeaVR should map directly to FA `pelvis` or needs an external transform
- final home/ready joint poses, neck defaults, and production joint/velocity/acceleration/jerk limits

## Additional Features

### Apple Vision Pro Support
BeaVR only requires cartesian positions from VR headsets in the standard y-up right-hand coordinate frame used by VR systems.
Apple Vision Pro users can connect through the third-party Improbable AI's [Tracking Streamer App](https://github.com/Improbable-AI/VisionProTeleop) that provides independent hand pose tracking.
This app should seamlessly integrate with BeaVR as an alternative VR endpoint. Although, it has not been developed or tested for this purpose.

## Citation

If you use BeaVR in your research, please cite our work:

```bibtex
@misc{posadasnava2025beavr,
  title         = {BEAVR: Bimanual, multi-Embodiment, Accessible, Virtual Reality Teleoperation System for Robots},
  author        = {Alejandro Posadas-Nava and Alejandro Carrasco and Richard Linares},
  year          = {2025},
  eprint        = {2508.09606},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {Accepted for presentation at ICCR 2025, Kyoto},
  url           = {https://arxiv.org/abs/2508.09606}
}
```

## License

# MIT License

Copyright (c) 2025 MIT Arclab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.


## Acknowledgments

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/US_Air_Force_Logo_Solid_Colour.svg" alt="USAF Logo" width="60"/>
</p>

<p align="justify">
This work was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
</p>

<p align="center">
<sub><sup>© 2025 Massachusetts Institute of Technology</sup></sub>
</p>
