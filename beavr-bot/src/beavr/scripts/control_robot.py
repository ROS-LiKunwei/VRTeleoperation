# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python beavr/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python beavr/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python beavr/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python beavr/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python beavr/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import json
import logging
import multiprocessing  # Needed for process types (用于启动独立进程来处理遥操作等，避免阻塞主进程)
import os
import select
import sys
import termios
import time
import tty
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import numpy as np

# from safetensors.torch import load_file, save_file
from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from beavr.lerobot.common.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset_v21_to_v30
from beavr.lerobot.common.policies.factory import make_policy # 用于实例化AI策略模型
from beavr.lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from beavr.lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,     # 初始化键盘监听（用于控制录制流程）
    log_control_info,
    record_episode,             # 录制单个片段
    reset_environment,          # 重置环境（录制间隙用）
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,              # 录制前的预热阶段
)
from beavr.lerobot.common.robot_devices.robots.utils import (
    Robot,
    make_robot_from_config,     # 根据配置创建机器人对象
)
from beavr.lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from beavr.lerobot.common.utils.utils import init_logging, log_say
from beavr.lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################

# 用于存储遥操作相关的多进程列表，在运行时会被填充
_teleop_processes: list[multiprocessing.Process] | None = None  # Populated at runtime

_FA_REPLAY_PREP_TARGET_A = (
    0.0,
    1.40,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.40,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
_FA_REPLAY_PREP_TARGET_B = (
    -1.09,
    0.76,
    -0.57,
    -1.08,
    0.89,
    0.15,
    -0.35,
    -1.09,
    -0.76,
    0.57,
    -1.08,
    -0.89,
    0.15,
    0.35,
)
_FA_REPLAY_PREP_TARGET_C = (
    0.0,
    0.2,
    0.0,
    -0.0,
    0.0,
    0.0,
    0.0,
    -0.0,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
_FA_REPLAY_PREP_MOVE_DURATION_S = 4.0
_FA_REPLAY_TARGET_REACHED_TOLERANCE_RAD = 0.03
_FA_REPLAY_TARGET_REACHED_STABLE_SAMPLES = 3


class _FaDatasetReplayPublisher:
    """Publish FA dataset replay actions recorded as 7+7+2 joint/state vectors.

    FA 录制时的 action 不是通用 BeavrBot 的笛卡尔命令格式，而是：
    left_arm_7d_joint + right_arm_7d_joint + left/right gripper_state。
    因此回放时需要绕过 BeavrBot.send_action() 的笛卡尔拆包逻辑，直接发布
    FA 双臂 joint target 和 O6/手部 open/grasp 命令。
    """

    def __init__(self, fps: int | None):
        _extend_sys_path_from_ament_prefix()
        # 延迟导入 ROS2 依赖，避免普通录制/非 FA 回放在无 ROS 环境中导入失败。
        try:
            import rclpy
            from min_snap.msg import MinSnapTarget
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Int32
        except Exception as exc:
            raise RuntimeError(
                "FA dataset replay requires a sourced ROS2 environment with min_snap.msg.MinSnapTarget, "
                "sensor_msgs.msg.JointState, and std_msgs.msg.Int32 available. "
                "Run replay from a shell that has sourced both:\n"
                "  source /opt/ros/humble/setup.bash\n"
                "  source /home/likunwei/humanoid_ws/install/setup.bash\n"
                "Then start control_robot.py from the same shell."
            ) from exc

        from beavr.teleop.components.interface.robots.fa_command_builder import (
            FA_LEFT_ARM_JOINT_NAMES,
            FA_RIGHT_ARM_JOINT_NAMES,
        )
        from beavr.teleop.components.interface.robots.fa_real_control import FaRealControlConfig
        from beavr.teleop.configs.constants import robots

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._node = rclpy.create_node("fa_dataset_replay")
        self._robots = robots
        self._msg_type = Int32
        self._min_snap_msg_type = MinSnapTarget
        self._arm_joint_names = tuple(FA_LEFT_ARM_JOINT_NAMES + FA_RIGHT_ARM_JOINT_NAMES)
        self._latest_joint_state: dict[str, float] = {}
        config = FaRealControlConfig()
        topics = config.ros2
        # 双臂 joint-space 目标走 humanoid_ws/min_snap，和 FA real-control 内部发布路径保持一致。
        self._min_snap_pub = self._node.create_publisher(
            MinSnapTarget,
            topics.min_snap_target_topic,
            topics.min_snap_target_queue_size,
        )
        # 手部命令是独立 ROS Int32 topic，不属于 16D upper-body joint command。
        self._left_hand_pub = self._node.create_publisher(
            Int32, topics.left_hand_topic, topics.hand_command_queue_size
        )
        self._right_hand_pub = self._node.create_publisher(
            Int32, topics.right_hand_topic, topics.hand_command_queue_size
        )
        self._node.create_subscription(
            JointState,
            topics.joint_state_topic,
            self._on_joint_state,
            10,
        )
        self._open_ros_action = int(config.hand_open_ros_action)
        self._grasp_ros_action = int(config.hand_grasp_ros_action)
        # expected_duration_s 对齐 replay 帧率，使相邻帧目标按数据集采样节奏推进。
        self._expected_duration_s = 1.0 / fps if fps else config.min_snap_expected_duration_s
        self._max_velocity_rad_s = config.min_snap_max_velocity_rad_s
        self._max_acceleration_rad_s2 = config.min_snap_max_acceleration_rad_s2

    def publish(self, action) -> None:
        action_np = np.asarray(action, dtype=np.float32).flatten()
        if action_np.size != 16:
            raise ValueError(
                "FA replay action must contain 16 values (left7 + right7 + left/right hand), "
                f"got {action_np.size}"
            )

        # action layout: [left_arm_7, right_arm_7, left_hand_state, right_hand_state]
        self.publish_arm_target(action_np[:14])
        self._publish_hand(self._robots.LEFT, action_np[14])
        self._publish_hand(self._robots.RIGHT, action_np[15])
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def _publish_hand(self, hand_side: str, gripper_state: float) -> None:
        msg = self._msg_type()
        msg.data = self._gripper_state_to_ros_action(gripper_state)
        if hand_side == self._robots.LEFT:
            self._left_hand_pub.publish(msg)
        else:
            self._right_hand_pub.publish(msg)

    def _on_joint_state(self, msg) -> None:
        self._latest_joint_state = {
            name: float(position) for name, position in zip(msg.name, msg.position, strict=False)
        }

    def _gripper_state_to_ros_action(self, gripper_state: float) -> int:
        # 数据集里 0 表示打开，1 表示握紧；ROS 侧使用 FA 配置里的真实动作编号。
        return self._grasp_ros_action if float(gripper_state) >= 0.5 else self._open_ros_action

    def current_arm_position(self) -> np.ndarray | None:
        if not all(name in self._latest_joint_state for name in self._arm_joint_names):
            return None
        return np.asarray(
            [self._latest_joint_state[name] for name in self._arm_joint_names],
            dtype=np.float64,
        )

    def _target_max_error(self, target: np.ndarray) -> float | None:
        current = self.current_arm_position()
        if current is None:
            return None
        return float(np.max(np.abs(current - target)))

    def publish_arm_target(self, target, *, expected_duration_s: float | None = None) -> None:
        target_np = np.asarray(target, dtype=np.float64).flatten()
        if target_np.size != 14:
            raise ValueError(f"FA min-snap target must contain 14 arm joint values, got {target_np.size}")

        arm_msg = self._min_snap_msg_type()
        arm_msg.left_arm_target_rad = [float(value) for value in target_np[:7]]
        arm_msg.right_arm_target_rad = [float(value) for value in target_np[7:14]]
        arm_msg.expected_duration_s = float(
            self._expected_duration_s if expected_duration_s is None else expected_duration_s
        )
        arm_msg.max_velocity_rad_s = float(self._max_velocity_rad_s)
        arm_msg.max_acceleration_rad_s2 = float(self._max_acceleration_rad_s2)
        self._min_snap_pub.publish(arm_msg)
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def move_arm_target(
        self,
        target,
        *,
        duration_s: float,
        tolerance_rad: float = _FA_REPLAY_TARGET_REACHED_TOLERANCE_RAD,
        stable_samples: int = _FA_REPLAY_TARGET_REACHED_STABLE_SAMPLES,
        label: str = "target",
    ) -> np.ndarray:
        target_np = np.asarray(target, dtype=np.float64).flatten()
        precheck_error = None
        for _ in range(stable_samples):
            self._rclpy.spin_once(self._node, timeout_sec=0.02)
            precheck_error = self._target_max_error(target_np)
            if precheck_error is None or precheck_error > tolerance_rad:
                break
        else:
            logging.info(
                "FA replay arm target %s already reached; max_error=%.4frad <= %.4frad",
                label,
                float(precheck_error or 0.0),
                tolerance_rad,
            )
            return target_np

        logging.info("Publishing FA replay arm target %s via min_snap", label)
        self.publish_arm_target(target_np, expected_duration_s=duration_s)
        reached_samples = 0
        last_wait_log_s = 0.0
        while True:
            self._rclpy.spin_once(self._node, timeout_sec=0.02)
            current = self.current_arm_position()
            if current is None:
                now_s = time.time()
                if now_s - last_wait_log_s >= 1.0:
                    last_wait_log_s = now_s
                    logging.info("Waiting FA replay arm target %s: no complete /joint_states yet", label)
                continue
            max_error = float(np.max(np.abs(current - target_np)))
            if max_error <= tolerance_rad:
                reached_samples += 1
                if reached_samples >= stable_samples:
                    logging.info(
                        "FA replay arm target %s reached; max_error=%.4frad <= %.4frad",
                        label,
                        max_error,
                        tolerance_rad,
                    )
                    return target_np
            else:
                reached_samples = 0
                now_s = time.time()
                if now_s - last_wait_log_s >= 1.0:
                    last_wait_log_s = now_s
                    logging.info(
                        "Waiting FA replay arm target %s: max_error=%.4frad > %.4frad",
                        label,
                        max_error,
                        tolerance_rad,
                    )

    def close(self) -> None:
        self._node.destroy_node()


def _extend_sys_path_from_ament_prefix() -> None:
    """Restore ROS Python paths when users launch with PYTHONPATH=src.

    `source /opt/ros/.../setup.bash` 和 `source humanoid_ws/install/setup.bash`
    会设置 PYTHONPATH，但命令行前缀 `PYTHONPATH=src python ...` 会覆盖它。
    AMENT_PREFIX_PATH 通常还保留着，所以这里从每个 prefix 推导 Python 包目录。
    """
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        for relative in (
            Path("local") / "lib" / version_dir / "dist-packages",
            Path("lib") / version_dir / "dist-packages",
            Path("lib") / version_dir / "site-packages",
        ):
            candidate = Path(prefix) / relative
            if candidate.is_dir():
                candidate_text = str(candidate)
                if candidate_text not in sys.path:
                    sys.path.append(candidate_text)


def _is_fa_bimanual_replay_action(robot: Robot, action) -> bool:
    return getattr(robot, "robot_type", None) == "fa" and np.asarray(action).size == 16


def _load_local_v30_replay_actions(root: str | Path | None, episode: int):
    """Load actions from a local LeRobot v3.0 dataset without contacting Hugging Face.

    当前仓库的 LeRobotDataset loader 仍以 v2.1 jsonl 元数据为主。录制命令如果设置
    --control.dataset_format=v3.0，结束后会把本地数据转换成 parquet 元数据布局：
    meta/tasks.parquet, meta/episodes/...parquet, data/chunk-xxx/file-xxx.parquet。
    回放本地 v3.0 数据时直接读取这些 parquet 文件，避免误判本地缺文件后去 Hub 拉
    repo_id，例如 local/fa_test。
    """
    if root is None:
        return None
    dataset_root = Path(root)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return None

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    if str(info.get("codebase_version")) != "v3.0":
        return None

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Local LeRobot v3.0 replay requires pandas with a parquet engine "
            "(pyarrow or fastparquet) in the active Python environment."
        ) from exc

    # v3.0 episode 元数据按 chunk/file 分片，先合并索引表再定位目标 episode。
    episode_rows = []
    for episode_file in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        episode_rows.append(pd.read_parquet(episode_file))
    if not episode_rows:
        raise FileNotFoundError(f"No v3.0 episode metadata found under {dataset_root / 'meta' / 'episodes'}")

    episodes_df = pd.concat(episode_rows, ignore_index=True)
    matches = episodes_df[episodes_df["episode_index"] == int(episode)]
    if matches.empty:
        available = sorted(int(value) for value in episodes_df["episode_index"].tolist())
        raise ValueError(f"Episode {episode} not found in local v3.0 dataset. Available episodes: {available}")

    row = matches.iloc[0]
    data_path = info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )
    # data parquet 中保存真正逐帧数据；这里只取 action，回放不需要图片和 observation。
    data_df = pd.read_parquet(dataset_root / data_path)
    if "episode_index" in data_df.columns:
        data_df = data_df[data_df["episode_index"] == int(episode)]
    if "frame_index" in data_df.columns:
        data_df = data_df.sort_values("frame_index")
    if "action" not in data_df.columns:
        raise ValueError(f"Local v3.0 episode file has no action column: {dataset_root / data_path}")
    return list(data_df["action"]), int(info["fps"])


def _wait_for_replay_start_key() -> None:
    print("Dataset replay is ready. Press r/R to start motion.")
    if not sys.stdin.isatty():
        while True:
            value = input().strip()
            if value.lower() == "r":
                return
            print("Press r/R to start motion.")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char.lower() == "r":
                print("Replay start confirmed.")
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _run_fa_replay_arm_sequence(
    fa_replay_publisher: _FaDatasetReplayPublisher,
    targets: tuple[tuple[str, tuple[float, ...]], ...],
) -> np.ndarray | None:
    current_position = None
    for label, target in targets:
        current_position = fa_replay_publisher.move_arm_target(
            target,
            duration_s=_FA_REPLAY_PREP_MOVE_DURATION_S,
            label=label,
        )
    return current_position


class _RecordTerminalLogFilter(logging.Filter):
    """Keep record-mode terminal output focused on dataset collection prompts.
        终端日志过滤器。
        在“录制模式”下，底层可能会产生大量刷屏日志。这个过滤器的作用是屏蔽无用信息，
        只让核心的流程提示（如“开始预热”、“正在录制”、“重置环境”等）显示在终端上，保持界面清爽。
    """

    _PROMPT_MESSAGES = (
        "Warmup record",
        "Recording episode",
        "Reset the environment",
        "Re-record episode",
        "No frames recorded",
        "Stop recording",
        "Exiting",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True # 错误日志永远放行
        message = record.getMessage()
        # 只有 INFO 级别且以特定提示语开头的日志才会被输出
        return record.levelno == logging.INFO and message.startswith(self._PROMPT_MESSAGES)


def _configure_record_terminal_logging() -> None:
    """应用上面定义的日志过滤器。"""
    os.environ["BEAVR_RECORD_TERMINAL_QUIET"] = "1"
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_RecordTerminalLogFilter())


def _episode_buffer_size(dataset: LeRobotDataset) -> int:
    """获取当前录制片段(episode)缓冲区的大小，用于判断是否录到了有效数据。"""
    episode_buffer = getattr(dataset, "episode_buffer", None)
    if not episode_buffer:
        return 0
    return int(episode_buffer.get("size", 0))


def start_teleop_process(
    *,
    robot_name: str | None = None,
    laterality: str | None = None,
    operate: bool | None = None,
    wait_for_exit: bool = False,
):
    """
    启动遥操作（Teleoperation）的辅助进程栈（例如：VR检测器/摄像头/机器人接口）。[Launch the teleoperation helper stack (detector/cameras/robot interface)]
    因为视频流获取、VR手柄信号读取非常消耗资源，通常需要放在独立的进程中运行。
   
    Parameters
    ----------
    robot_name : str | None
        要加载的机器人组合的标识符(例如“leap,xarm7”或“leap”), 如果为*None*,
        我们将尝试从命令行界面(CLI)读取`--teleop_robot_name`(通过“beavr.configs.parser”)。
        如果该函数也不存在,回退到默认的“leap,xarm7”。

    operate : bool | None
        是否启动监听VR Detector并将命令转发给机器人的*Operator*进程。
        当*policy*通过`robot.send_action()`自行发布命令时，将此设置为``False``至关重要——
        同时运行两者会导致PUB/SUB冲突。
        当为*None*时,该值将自动派生:每当在命令行界面(CLI)上提供策略时（通过`--control.policy.`参数的存在来检测），该值将设置为``False``；否则，默认值为``True``。
    """

    global _teleop_processes
    if _teleop_processes is not None:
        logging.info("Teleoperation already running – skipping launch.")
        return

    # 延迟导入（Lazy import）：只有在需要用到遥操作时才导入这些库，加快脚本启动速度。
    from beavr.teleop.components import TeleOperator
    from beavr.teleop.main import MainConfig  # pylint: disable=import-error

    # ------------------------------------------------------------------
    # Resolve *robot_name*
    # ------------------------------------------------------------------
    if robot_name is None:
        # Try CLI override --teleop_robot_name=...
        robot_name = parser.parse_arg("teleop_robot_name")

    if robot_name is None:
        # Fallback for legacy flag --teleop.robot_name=...
        robot_name = parser.parse_arg("teleop.robot_name")

    if robot_name is None:
        # Ultimate fallback – use new multi-robot syntax
        robot_name = "leap,xarm7"

    if laterality is None:
        laterality = parser.parse_arg("teleop.laterality")

    if laterality is None:
        laterality = parser.parse_arg("teleop_laterality")

    if laterality is None:
        laterality = "right"

    # ------------------------------------------------------------------
    # Resolve *operate*
    # Operate 决定是否启动将命令转发给机器人的“Operator”进程。
    # 如果你在运行 AI 策略 (Policy)，AI 自己会发命令，此时要把 operate 设为 False，避免人类和 AI 抢夺控制权。
    # ------------------------------------------------------------------
    if operate is None:
        # CLI override (explicit)
        operate_cli = parser.parse_arg("teleop_operate")
        if operate_cli is not None:
            operate = operate_cli.lower() not in {"0", "false", "no", "off"}

    if operate is None:
        # 隐式启发式规则：如果命令行提供了 policy，就禁用操作员进程
        has_policy_path = parser.get_path_arg("control.policy") is not None
        has_policy_type = parser.get_type_arg("control.policy") is not None
        operate = not (has_policy_path or has_policy_type)

    logging.info(
        "Starting teleoperation helper with robot_name='%s', laterality='%s', operate=%s",
        robot_name,
        laterality,
        operate,
    )

    # 使用新的结构化配置为所选机器人组合构建配置
    # 使用robot_name创建MainConfig，并在teleop.flags中设置操作标志
    main_config = MainConfig(robot_name=robot_name, laterality=laterality)
    main_config.teleop.flags.operate = operate

    logging.info("Instantiating TeleOperator (Draccus version)…")
    # 构建并启动 TeleOperator 进程
    teleop = TeleOperator(main_config)
    _teleop_processes = teleop.get_processes()

    # Start all sub-processes.
    for p in _teleop_processes:
        p.start()

    # Give PUB sockets a moment to come up before the main thread proceeds.
    time.sleep(1)
    # 如果要求等待退出，则阻塞主线程直到子进程结束（通常在纯遥控模式下使用）
    if wait_for_exit:
        try:
            for p in _teleop_processes:
                p.join()
        except KeyboardInterrupt:
            logging.info("Teleoperation interrupted by user.")
        finally:
            stop_teleop_process()


def stop_teleop_process():
    """Terminate the TeleOperator's child processes if they are active."""
    global _teleop_processes
    if _teleop_processes:
        logging.info("Stopping teleoperation processes…")
        for p in _teleop_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        _teleop_processes = None


@safe_disconnect
def teleoperate():
    """Use the BeaVR teleoperation system by running teleop.py and waiting for it to complete."""
    start_teleop_process(wait_for_exit=True)


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    """
        数据录制的核心逻辑。
        用于通过遥操作(人控制)或策略模型(AI控制)来收集机器人的传感器数据和动作，并保存为数据集。
    """
    # When recording, the teleoperation system is assumed to be already running.
    # This function is now only responsible for the recording logic itself.
    try:
        if hasattr(robot, "manage_teleop_state"):
            robot.manage_teleop_state = cfg.manage_teleop_state
            logging.info("Recorder teleop-state management: %s", cfg.manage_teleop_state)

        # TODO: Add option to record logs
        # 处理数据集初始化：如果是断点续录(resume)，则加载已有数据集；否则新建。
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.repo_id,
                root=cfg.root,
            )
            if len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.num_image_writer_processes,
                    num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
        else:
            # Create empty dataset or load existing saved episodes
            sanity_check_dataset_name(cfg.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.repo_id,
                cfg.fps,
                root=cfg.root,
                robot=robot,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        if cfg.video:
            dataset.set_async_video_encoding(cfg.async_video_encoding)

        # 如果配置了策略（Policy），则实例化 AI 模型。如果是人类录制，这里为 None。
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        # 初始化键盘监听器，以便在终端里通过快捷键（如 ESC 退出，左右箭头重录）控制流程
        listener, events = init_keyboard_listener()

        # 录制前的预热：不保存数据，纯粹为了让硬件设备连接、画面同步，并允许操作员先把机器人移动到初始位置。
        # Execute a few seconds without recording to:
        # 1. teleoperate the robot to move it in starting position if no policy provided,
        # 2. give times to the robot devices to connect and start synchronizing,
        # 3. place the cameras windows on screen
        enable_teleoperation = policy is None
        log_say("Warmup record", cfg.play_sounds)
        warmup_record(
            robot,
            events,
            enable_teleoperation,
            cfg.warmup_time_s,
            cfg.display_data,
            cfg.fps,
        )

        recorded_episodes = 0
        # 主录制循环
        while True:
            if recorded_episodes >= cfg.num_episodes:
                break

            loop_start_t = time.perf_counter()
            logging.info(
                "[RecordTiming] episode_loop_start dataset_episode=%s recorded_episodes=%s pending_finalizations=%s",
                dataset.num_episodes,
                recorded_episodes,
                len(getattr(dataset, "_pending_episode_finalizations", [])),
            )
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            # 执行单个片段(episode)的录制
            record_start_t = time.perf_counter()
            record_episode(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_data=cfg.display_data,
                policy=policy,
                fps=cfg.fps,
                single_task=cfg.single_task,
            )
            logging.info(
                "[RecordTiming] record_episode_done dataset_episode=%s dt=%.3fs buffer_size=%s events=%s",
                dataset.num_episodes,
                time.perf_counter() - record_start_t,
                _episode_buffer_size(dataset),
                {k: events.get(k) for k in ["exit_early", "rerecord_episode", "stop_recording"]},
            )
            if events["stop_recording"]:
                if events["rerecord_episode"] or _episode_buffer_size(dataset) == 0:
                    dataset.clear_episode_buffer()
                    break
            # 键盘事件：用户按下要求重新录制当前片段
            if events["rerecord_episode"]:
                if not events["stop_recording"]:
                    log_say("Reset the environment", cfg.play_sounds)
                    reset_environment(robot, events, cfg.reset_time_s, cfg.fps) # 给时间重置物理环境
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer() # 清空刚刚录的废弃数据
                continue
            # 安全检查：如果没有记录到任何帧    
            if _episode_buffer_size(dataset) == 0:
                log_say(
                    "No frames recorded; episode not saved. Waiting for valid observation/action frames.",
                    cfg.play_sounds,
                )
                dataset.clear_episode_buffer()
                if events["stop_recording"]:
                    break
                if recorded_episodes < cfg.num_episodes:
                    log_say("Reset the environment", cfg.play_sounds)
                    reset_environment(robot, events, cfg.reset_time_s, cfg.fps)
                continue
            # 保存有效片段
            save_start_t = time.perf_counter()
            dataset.save_episode()
            logging.info(
                "[RecordTiming] save_episode_done saved_episode=%s dt=%.3fs pending_finalizations=%s",
                dataset.num_episodes - 1,
                time.perf_counter() - save_start_t,
                len(getattr(dataset, "_pending_episode_finalizations", [])),
            )
            recorded_episodes += 1

            if events["stop_recording"]:
                break

            # Execute a few seconds without recording to give time to manually reset the environment.
            # Save first so early-exited episodes are durable before reset or a later Ctrl+C.
            # 每个片段录制完后，留出时间让操作员把物体和机器人放回初始位置
            if recorded_episodes < cfg.num_episodes:
                log_say("Reset the environment", cfg.play_sounds)
                reset_start_t = time.perf_counter()
                reset_environment(robot, events, cfg.reset_time_s, cfg.fps)
                logging.info(
                    "[RecordTiming] reset_done dt=%.3fs events=%s total_loop_dt=%.3fs",
                    time.perf_counter() - reset_start_t,
                    {k: events.get(k) for k in ["exit_early", "rerecord_episode", "stop_recording"]},
                    time.perf_counter() - loop_start_t,
                )
        # 录制结束，清理键盘监听
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        stop_recording(robot, listener, cfg.display_data)
        dataset.wait_for_async_video_encoding(shutdown=True)
        # 处理数据集格式转换
        if cfg.dataset_format == "v3.0":
            convert_dataset_v21_to_v30(dataset.root)
        elif cfg.dataset_format != "v2.1":
            raise ValueError("control.dataset_format must be either 'v3.0' or 'v2.1'")
        # 选择性推送到 Hugging Face Hub（云端）
        if cfg.push_to_hub:
            dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

        log_say("Exiting", cfg.play_sounds)
        return dataset
    finally:
        # Cleanup is now handled by the main control_robot function
        pass


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    """
    回放模式。
    读取已录制的数据集中的某一个片段(episode)，并将其中的动作指令(action)直接发送给机器人，让机器人原样执行一遍。
    """
    # TODO: refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO: Add option to record logs

    # Prefer local v3.0 parquet replay when available; fall back to the regular
    # LeRobotDataset path for v2.1 datasets or Hub-backed datasets.
    local_v30_actions = _load_local_v30_replay_actions(cfg.root, cfg.episode)
    if local_v30_actions is None:
        dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
        actions = dataset.hf_dataset.select_columns("action")
        num_frames = dataset.num_frames
        replay_fps = cfg.fps or dataset.fps
    else:
        actions, dataset_fps = local_v30_actions
        num_frames = len(actions)
        replay_fps = cfg.fps or dataset_fps

    fa_replay_publisher = None
    is_fa_bimanual_replay = False
    if num_frames > 0:
        first_action_item = actions[0]
        first_action = first_action_item["action"] if isinstance(first_action_item, dict) else first_action_item
        if _is_fa_bimanual_replay_action(robot, first_action):
            is_fa_bimanual_replay = True
            # Initialize ROS publishers before opening cameras/robot adapters so a missing ROS environment
            # fails early and does not briefly connect hardware resources.
            fa_replay_publisher = _FaDatasetReplayPublisher(replay_fps)

    if not robot.is_connected:
        robot.connect()

    if is_fa_bimanual_replay:
        assert fa_replay_publisher is not None
        _wait_for_replay_start_key()
        logging.info("Moving FA replay arm targets A -> B before dataset replay.")
        _run_fa_replay_arm_sequence(
            fa_replay_publisher,
            (("A", _FA_REPLAY_PREP_TARGET_A), ("B", _FA_REPLAY_PREP_TARGET_B)),
        )

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    # 按帧率逐帧发送动作
    try:
        for idx in range(num_frames):
            start_episode_t = time.perf_counter()

            action_item = actions[idx]
            # v2.1/HF dataset returns {"action": ...}; local v3.0 helper returns the raw action vector.
            action = action_item["action"] if isinstance(action_item, dict) else action_item
            if _is_fa_bimanual_replay_action(robot, action):
                assert fa_replay_publisher is not None
                fa_replay_publisher.publish(action)
            else:
                robot.send_action(action) # 将当前帧的动作发送给机器人
            # 延时等待以匹配目标帧率(FPS)
            dt_s = time.perf_counter() - start_episode_t
            if replay_fps:
                busy_wait(1 / replay_fps - dt_s)

            dt_s = time.perf_counter() - start_episode_t
            log_control_info(robot, dt_s, fps=replay_fps)

        if is_fa_bimanual_replay:
            logging.info("Dataset replay finished. Moving FA replay arm targets B -> A -> C.")
            _run_fa_replay_arm_sequence(
                fa_replay_publisher,
                (
                    ("B", _FA_REPLAY_PREP_TARGET_B),
                    ("A", _FA_REPLAY_PREP_TARGET_A),
                    ("C", _FA_REPLAY_PREP_TARGET_C),
                ),
            )
            logging.info("FA replay sequence complete.")
    finally:
        if fa_replay_publisher is not None:
            fa_replay_publisher.close()


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    """
        脚本的主入口函数。
        负责初始化日志、解析配置、按需启动遥操作子进程、连接硬件，并路由到对应的模式（遥控、录制、回放）。
    """
    init_logging()
    if isinstance(cfg.control, RecordControlConfig):
        _configure_record_terminal_logging() # 如果是录制模式，开启日志过滤
    logging.info(pformat(asdict(cfg)))

    # Determine whether the teleoperation helper needs to run. Set
    # --teleop.start=false when an external teleop.py stack is already running.
    start_teleop = (
        isinstance(cfg.control, (TeleoperateControlConfig, RecordControlConfig)) and cfg.teleop.start
    )

    if start_teleop:
        start_teleop_process(
            robot_name=getattr(cfg.teleop, "robot_name", None),
            laterality=getattr(cfg.teleop, "laterality", None),
            operate=getattr(cfg.teleop, "operate", None),
        )
        logging.info("Waiting 5s for teleoperation system to initialize before robot connection...")
        # TODO: Improve this logic shouldn't need sleep
        time.sleep(5) # 强行等待5秒，确保遥操作系统硬件和通讯初始化完毕

    # 根据配置实例化机器人对象（控制硬件的接口） [Pass the controller to the robot adapter]
    robot = make_robot_from_config(cfg.robot)

    try:
        # 连接硬件
        if not robot.is_connected:
            robot.connect()
        # 根据配置文件指定的模式执行对应逻辑
        if isinstance(cfg.control, TeleoperateControlConfig):
            # The teleop process was already started, just wait for it to complete.
            global _teleop_processes
            if _teleop_processes:
                logging.info("Waiting for teleoperation process to complete...")
                for p in _teleop_processes:
                    p.join() # 纯遥控模式下，主进程在这里挂起，直到用户终止遥操作

        elif isinstance(cfg.control, RecordControlConfig):
            record(robot, cfg.control)
        elif isinstance(cfg.control, ReplayControlConfig):
            replay(robot, cfg.control)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 的中断信号
        logging.info("Main control loop interrupted by user.")
    finally:
        # This block ensures cleanup happens on normal exit or interrupt.
        logging.info("Starting cleanup...")
        if robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()
        # Always try to stop the teleop process on exit
        stop_teleop_process()
        logging.info("Cleanup complete. Exiting.")


if __name__ == "__main__":
    control_robot()
