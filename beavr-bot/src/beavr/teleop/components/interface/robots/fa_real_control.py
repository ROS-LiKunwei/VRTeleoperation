"""FA native upper-body real-control adapter."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.interface.robots.arm_command_publisher import MinSnapTargetPublisher
from beavr.teleop.components.interface.robots.fa_arm_ik_client import (
    FaArmIkConfig,
    FaArmIkClientBase,
    FaPybindIkClient,
)
from beavr.teleop.components.interface.robots.fa_command_builder import (
    FA_ARM_JOINT_COUNT,
    FA_NECK_JOINT_NAMES,
    FA_UPPER_COMMAND_LENGTH,
    FaCommandLimiter,
    FaJointStateCache,
    FaJointStateSnapshot,
    FaUpperPositionCommand,
    FaUpperPositionCommandBuilder,
    FaUpperPositionSafetyConfig,
)
from beavr.teleop.components.interface.robots.fa_mujoco_kinematics import (
    FaKinematicsConfig,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import ports, robots

logger = logging.getLogger(__name__)

FA_UPPER_COMMAND_TOPIC = "fa_upper_position_command"
FA_HAND_OPEN_COMMAND = 1
FA_HAND_GRASP_COMMAND = 2


def _default_beavr_bot_root() -> Path:
    return Path(os.environ.get("BEAVR_BOT_ROOT", Path(__file__).resolve().parents[6])).expanduser()


def _default_fa_urdf_path() -> str:
    env_value = os.environ.get("FA_URDF_PATH")
    if env_value:
        return str(Path(env_value).expanduser())
    return str(_default_beavr_bot_root() / "robots" / "fa_description" / "urdf" / "fa_robot.urdf")


def _default_fa_srdf_path() -> str:
    env_value = os.environ.get("FA_SRDF_PATH")
    if env_value:
        return str(Path(env_value).expanduser())
    for candidate in (
        Path.home() / "likunwei_ws" / "src" / "fa_moveit2_config" / "config" / "fa_robot.srdf",
        Path.home() / "humanoid_ws" / "src" / "fa_moveit2_config" / "config" / "fa_robot.srdf",
        Path("/home/likunwei/humanoid_ws/src/fa_moveit2_config/config/fa_robot.srdf"),
    ):
        if candidate.exists():
            return str(candidate)
    return ""


@dataclass
class FaRos2Topics:
    joint_state_topic: str = "/joint_states"
    upper_position_command_topic: str = "/upper_position_controller/commands"
    min_snap_target_topic: str = "/min_snap/target"
    left_hand_topic: str = "/left_topic_to_hand"
    right_hand_topic: str = "/right_topic_to_hand"
    upper_position_command_queue_size: int = 60
    hand_command_queue_size: int = 10
    min_snap_target_queue_size: int = 1
    joint_state_timeout_s: float = 1.0


@dataclass
class FaRealControlConfig:
    control_backend: str = "mujoco"
    ros2: FaRos2Topics = field(default_factory=FaRos2Topics)
    upper: FaUpperPositionSafetyConfig = field(default_factory=FaUpperPositionSafetyConfig)
    kinematics: FaKinematicsConfig = field(default_factory=FaKinematicsConfig)
    ik: FaArmIkConfig = field(
        default_factory=lambda: FaArmIkConfig(
            urdf_file=_default_fa_urdf_path(),
            srdf_file=_default_fa_srdf_path(),
        )
    )
    state_publish_fps: float = 30.0
    command_publish_hz: float = 1000.0
    safety_hold_arm_on_pause: bool = True
    pause_hold_heartbeat_hz: float = 20.0
    allow_mujoco_mirror_without_joint_state: bool = True
    max_ik_solution_jump_rad: float = 0.3
    ik_solution_jump_clip_rad: float = 0.3
    ik_max_position_error_m: float = 0.05
    ik_max_orientation_error_rad: float = 0.5
    min_snap_expected_duration_s: float = 0.016
    min_snap_max_velocity_rad_s: float = 1.5
    min_snap_max_acceleration_rad_s2: float = 15.0
    min_snap_target_publish_hz: float = 60.0
    min_snap_target_epsilon_rad: float = 0.002
    ik_cartesian_position_deadband_m: float = 0.012
    ik_cartesian_orientation_deadband_rad: float = 0.06
    ik_reachable_fallback_enabled: bool = True
    ik_reachable_fallback_iterations: int = 6
    ik_reachable_fallback_min_alpha: float = 0.05
    ik_reachable_fallback_orientation_alphas: tuple[float, ...] = (1.0, 0.75, 0.5)
    ik_return_recovery_enabled: bool = True
    ik_return_recovery_min_retreat_m: float = 0.015
    ik_return_recovery_max_position_error_m: float = 0.35
    ik_return_recovery_max_orientation_error_rad: float = 1.0
    ik_multi_seed_enabled: bool = True
    ik_escape_enabled: bool = True
    ik_escape_trigger_count: int = 3
    ik_escape_target_tolerance_rad: float = 0.04
    ik_escape_local_enabled: bool = True
    ik_escape_local_elbow_target_rad: float = -1.05
    ik_escape_local_max_joint_delta_rad: float = 0.25
    ik_singularity_output_filter_enabled: bool = True
    ik_singularity_filter_enter_problem_count: int = 2
    ik_singularity_filter_exit_problem_count: int = 0
    ik_singularity_filter_elbow_enter_rad: float = -1.20
    ik_singularity_filter_elbow_exit_rad: float = -1.05
    ik_singularity_filter_elbow_upper_enter_rad: float = 0.12
    ik_singularity_filter_elbow_upper_exit_rad: float = -0.05
    ik_singularity_output_filter_alpha: float = 0.85
    ik_singularity_output_filter_max_step_rad: float = 0.08
    ik_escape_left_arm_positions_rad: Optional[tuple[float, ...]] = None
    ik_escape_right_arm_positions_rad: Optional[tuple[float, ...]] = None
    initial_pose_enabled: bool = False
    initial_left_arm_positions_rad: Optional[tuple[float, ...]] = None
    initial_right_arm_positions_rad: Optional[tuple[float, ...]] = None
    initial_pose_duration_s: float = 4.0
    initial_pose_max_velocity_rad_s: float = 0.5
    initial_pose_max_acceleration_rad_s2: float = 2.0
    hand_open_ros_action: int = 21
    hand_grasp_ros_action: int = 20
    initial_hand_ros_action: Optional[int] = 21

    def __post_init__(self):
        self.max_ik_solution_jump_rad = max(0.0, float(self.max_ik_solution_jump_rad))
        self.ik_solution_jump_clip_rad = max(0.0, float(self.ik_solution_jump_clip_rad))
        self.ik_max_position_error_m = max(0.0, float(self.ik_max_position_error_m))
        self.ik_max_orientation_error_rad = max(0.0, float(self.ik_max_orientation_error_rad))
        self.min_snap_expected_duration_s = max(1e-4, float(self.min_snap_expected_duration_s))
        self.min_snap_max_velocity_rad_s = max(1e-6, float(self.min_snap_max_velocity_rad_s))
        self.min_snap_max_acceleration_rad_s2 = max(1e-6, float(self.min_snap_max_acceleration_rad_s2))
        self.min_snap_target_publish_hz = max(1.0, float(self.min_snap_target_publish_hz))
        self.min_snap_target_epsilon_rad = max(0.0, float(self.min_snap_target_epsilon_rad))
        self.ik_cartesian_position_deadband_m = max(0.0, float(self.ik_cartesian_position_deadband_m))
        self.ik_cartesian_orientation_deadband_rad = max(0.0, float(self.ik_cartesian_orientation_deadband_rad))
        self.ik_reachable_fallback_iterations = max(1, int(self.ik_reachable_fallback_iterations))
        self.ik_reachable_fallback_min_alpha = min(1.0, max(0.0, float(self.ik_reachable_fallback_min_alpha)))
        orientation_alphas = tuple(
            min(1.0, max(0.0, float(value)))
            for value in self.ik_reachable_fallback_orientation_alphas
        )
        if 1.0 not in orientation_alphas:
            orientation_alphas = (1.0, *orientation_alphas)
        self.ik_reachable_fallback_orientation_alphas = tuple(dict.fromkeys(orientation_alphas))
        self.ik_return_recovery_min_retreat_m = max(0.0, float(self.ik_return_recovery_min_retreat_m))
        self.ik_return_recovery_max_position_error_m = max(
            0.0, float(self.ik_return_recovery_max_position_error_m)
        )
        self.ik_return_recovery_max_orientation_error_rad = max(
            0.0, float(self.ik_return_recovery_max_orientation_error_rad)
        )
        self.ik_escape_trigger_count = max(1, int(self.ik_escape_trigger_count))
        self.ik_escape_target_tolerance_rad = max(1e-6, float(self.ik_escape_target_tolerance_rad))
        self.ik_escape_local_elbow_target_rad = float(self.ik_escape_local_elbow_target_rad)
        self.ik_escape_local_max_joint_delta_rad = max(0.0, float(self.ik_escape_local_max_joint_delta_rad))
        self.ik_singularity_filter_enter_problem_count = max(
            1, int(self.ik_singularity_filter_enter_problem_count)
        )
        self.ik_singularity_filter_exit_problem_count = max(
            0, int(self.ik_singularity_filter_exit_problem_count)
        )
        self.ik_singularity_filter_elbow_enter_rad = float(self.ik_singularity_filter_elbow_enter_rad)
        self.ik_singularity_filter_elbow_exit_rad = float(self.ik_singularity_filter_elbow_exit_rad)
        self.ik_singularity_filter_elbow_upper_enter_rad = float(
            self.ik_singularity_filter_elbow_upper_enter_rad
        )
        self.ik_singularity_filter_elbow_upper_exit_rad = float(
            self.ik_singularity_filter_elbow_upper_exit_rad
        )
        self.ik_singularity_output_filter_alpha = min(
            0.98, max(0.0, float(self.ik_singularity_output_filter_alpha))
        )
        self.ik_singularity_output_filter_max_step_rad = max(
            0.0, float(self.ik_singularity_output_filter_max_step_rad)
        )
        self.initial_pose_duration_s = max(1e-3, float(self.initial_pose_duration_s))
        self.initial_pose_max_velocity_rad_s = max(1e-6, float(self.initial_pose_max_velocity_rad_s))
        self.initial_pose_max_acceleration_rad_s2 = max(1e-6, float(self.initial_pose_max_acceleration_rad_s2))
        if self.initial_pose_enabled:
            self.initial_left_arm_positions_rad = tuple(
                float(v) for v in self._validate_initial_arm_pose(
                    self.initial_left_arm_positions_rad,
                    "initial_left_arm_positions_rad",
                )
            )
            self.initial_right_arm_positions_rad = tuple(
                float(v) for v in self._validate_initial_arm_pose(
                    self.initial_right_arm_positions_rad,
                    "initial_right_arm_positions_rad",
                )
            )
        self.ik_escape_left_arm_positions_rad = (
            None if self.ik_escape_left_arm_positions_rad is None
            else tuple(float(v) for v in self._validate_initial_arm_pose(
                self.ik_escape_left_arm_positions_rad,
                "ik_escape_left_arm_positions_rad",
            ))
        )
        self.ik_escape_right_arm_positions_rad = (
            None if self.ik_escape_right_arm_positions_rad is None
            else tuple(float(v) for v in self._validate_initial_arm_pose(
                self.ik_escape_right_arm_positions_rad,
                "ik_escape_right_arm_positions_rad",
            ))
        )

    @staticmethod
    def _validate_initial_arm_pose(values: Optional[Sequence[float]], label: str) -> np.ndarray:
        if values is None:
            raise ValueError(f"{label} must be set when initial_pose_enabled is true")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain {FA_ARM_JOINT_COUNT} finite values")
        return array


@dataclass
class _ArmIkUpdate:
    hand_side: str
    target_key: tuple
    active_goal: Optional[np.ndarray]
    dirty: bool
    last_safe: Optional[np.ndarray] = None
    last_ik_target: Optional[CartesianTarget] = None
    warn_key: Optional[str] = None
    warn_message: Optional[str] = None
    consume_target: bool = True
    counts_as_ik_problem: bool = False


class FaRos2Bridge:
    """ROS2 bridge for FA joint states and min-snap target publishing."""

    def __init__(
        self,
        topics: FaRos2Topics,
        require_ros: bool,
        node_name: str = "fa_real_control",
    ):
        self.topics = topics
        self.require_ros = require_ros
        self.available = False
        self.joint_cache = FaJointStateCache(topics.joint_state_timeout_s)
        self._rclpy = None
        self._node = None
        self._min_snap_target_publisher = None
        self._left_hand_pub = None
        self._right_hand_pub = None
        self._hand_msg_type = None
        if require_ros:
            self._init_ros2(node_name)

    def _init_ros2(self, node_name: str) -> None:
        try:
            import rclpy
            from min_snap.msg import MinSnapTarget
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Int32

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node(node_name)
            self._hand_msg_type = Int32
            self._min_snap_target_publisher = MinSnapTargetPublisher(
                ros_node=self._node,
                msg_type=MinSnapTarget,
                topic=self.topics.min_snap_target_topic,
                queue_size=self.topics.min_snap_target_queue_size,
            )
            self._left_hand_pub = self._node.create_publisher(
                Int32, self.topics.left_hand_topic, self.topics.hand_command_queue_size
            )
            self._right_hand_pub = self._node.create_publisher(
                Int32, self.topics.right_hand_topic, self.topics.hand_command_queue_size
            )
            self._node.create_subscription(JointState, self.topics.joint_state_topic, self._on_joint_state, 10)
            self.available = True
            logger.info(
                "FA ROS2 bridge connected: joint_state=%s min_snap_target=%s",
                self.topics.joint_state_topic,
                self.topics.min_snap_target_topic,
            )
        except Exception as exc:
            self.available = False
            message = f"FA ROS2 bridge unavailable: {exc}"
            logger.error(message)
            if self.require_ros:
                raise RuntimeError(message) from exc

    def _on_joint_state(self, msg) -> None:
        self.joint_cache.update_from_joint_state_msg(msg, now_s=time.time())

    def spin_once(self) -> None:
        if self.available and self._rclpy is not None and self._node is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def publish_hand_command(self, hand_side: str, action_id: int) -> bool:
        if not self.available or self._hand_msg_type is None:
            return False
        msg = self._hand_msg_type()
        msg.data = int(action_id)
        if hand_side == robots.LEFT:
            self._left_hand_pub.publish(msg)
        else:
            self._right_hand_pub.publish(msg)
        return True

    def publish_min_snap_target(
        self,
        command: FaUpperPositionCommand,
        expected_duration_s: float,
        max_velocity_rad_s: float,
        max_acceleration_rad_s2: float,
    ) -> bool:
        if not self.available or self._min_snap_target_publisher is None:
            return False
        return self._min_snap_target_publisher.publish(
            command.left_arm,
            command.right_arm,
            expected_duration_s,
            max_velocity_rad_s,
            max_acceleration_rad_s2,
        )

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


class FaRealControl(Component):
    """Bimanual FA real-control component using native 16D upper commands."""

    def __init__(
        self,
        host: str,
        control_backend: str,
        right_target_port: int,
        left_target_port: int,
        right_state_publish_port: int,
        left_state_publish_port: int,
        teleoperation_state_port: int,
        right_endeff_publish_port: Optional[int] = None,
        left_endeff_publish_port: Optional[int] = None,
        urdf_path: str = "",
        config: Optional[FaRealControlConfig] = None,
        ik_client: Optional[FaArmIkClientBase] = None,
        **_,
    ):
        self.notify_component_start("fa_real_control")
        self.host = host
        self.control_backend = control_backend
        urdf_path = urdf_path or _default_fa_urdf_path()
        self._validate_backend()
        self.config = config or FaRealControlConfig(control_backend=control_backend)
        self.config.control_backend = control_backend
        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._right_state_publish_port = right_state_publish_port
        self._left_state_publish_port = left_state_publish_port
        self._right_endeff_publish_port = right_endeff_publish_port or right_state_publish_port
        self._left_endeff_publish_port = left_endeff_publish_port or left_state_publish_port
        self._publisher_manager.register_topic(self.host, self._right_state_publish_port, "fa_right")
        self._publisher_manager.register_topic(self.host, self._left_state_publish_port, "fa_left")
        self._requires_resume_for_teleop = teleoperation_state_port is not None

        self._right_target_subscriber = ZMQSubscriber(
            host, right_target_port, "endeff_coords", message_type=CartesianTarget
        )
        self._left_target_subscriber = ZMQSubscriber(
            host, left_target_port, "endeff_coords", message_type=CartesianTarget
        )
        self._right_reset_subscriber = ZMQSubscriber(host, right_target_port, "reset", message_type=SessionCommand)
        self._left_reset_subscriber = ZMQSubscriber(host, left_target_port, "reset", message_type=SessionCommand)
        self._pause_subscriber = ZMQSubscriber(
            host, teleoperation_state_port, robots.PAUSE, message_type=SessionCommand
        )
        self._subscribers = [
            self._right_target_subscriber,
            self._left_target_subscriber,
            self._right_reset_subscriber,
            self._left_reset_subscriber,
            self._pause_subscriber,
        ]

        require_ros = True
        self._ros2 = FaRos2Bridge(
            self.config.ros2,
            require_ros=require_ros,
        )
        self._dry_joint_cache = FaJointStateCache(self.config.ros2.joint_state_timeout_s)
        self._dry_joint_cache.update(
            np.zeros(FA_ARM_JOINT_COUNT),
            np.zeros(FA_ARM_JOINT_COUNT),
            self.config.upper.neck_default_positions_rad,
            now_s=time.time(),
        )
        # FA runtime FK/IK is provided by the Pinocchio pybind solver. The
        # MuJoCo model collapses fixed hand links, so it is only used by the
        # command mirror instead of as a real-control FK fallback.
        self._kinematics = None
        self._ik_client = ik_client or self._make_default_ik_client()
        self._ik_clients: Dict[str, FaArmIkClientBase] = (
            {robots.LEFT: ik_client, robots.RIGHT: ik_client}
            if ik_client is not None
            else {robots.LEFT: self._make_default_ik_client(), robots.RIGHT: self._ik_client}
        )
        self._ik_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fa_arm_ik")
        self._pending_ik_futures: Dict[str, Future] = {}
        self._builder = FaUpperPositionCommandBuilder(self.config.upper)
        self._limiter = FaCommandLimiter(self.config.upper)

        self._teleop_active = not self._requires_resume_for_teleop
        self._needs_reset = True
        self._real_reset_ready = self.control_backend == "mujoco"
        self._latest_targets: Dict[str, Optional[CartesianTarget]] = {robots.LEFT: None, robots.RIGHT: None}
        self._latest_target_keys: Dict[str, Optional[tuple]] = {robots.LEFT: None, robots.RIGHT: None}
        self._active_arm_goals: Dict[str, Optional[np.ndarray]] = {robots.LEFT: None, robots.RIGHT: None}
        self._arm_goal_dirty: Dict[str, bool] = {robots.LEFT: False, robots.RIGHT: False}
        self._hand_init_ready: Dict[str, bool] = {robots.LEFT: False, robots.RIGHT: False}
        self._accept_cartesian_targets_after_s: Dict[str, float] = {robots.LEFT: 0.0, robots.RIGHT: 0.0}
        self._last_safe_arm_targets: Dict[str, Optional[np.ndarray]] = {robots.LEFT: None, robots.RIGHT: None}
        self._last_ik_cartesian_targets: Dict[str, Optional[CartesianTarget]] = {robots.LEFT: None, robots.RIGHT: None}
        self._ik_problem_counts: Dict[str, int] = {robots.LEFT: 0, robots.RIGHT: 0}
        self._arm_escape_active: Dict[str, bool] = {robots.LEFT: False, robots.RIGHT: False}
        self._ik_output_filter_active: Dict[str, bool] = {robots.LEFT: False, robots.RIGHT: False}
        self._ik_output_filter_goals: Dict[str, Optional[np.ndarray]] = {robots.LEFT: None, robots.RIGHT: None}
        self._pause_hold_command: Optional[FaUpperPositionCommand] = None
        self._resume_hold_until_target = False
        self._pending_resume_after_initial_pose = False
        self._last_pause_hold_publish_time = 0.0
        self._next_state_publish_time_s = 0.0
        self._last_published_upper_command: Optional[FaUpperPositionCommand] = None
        self._last_hand_commands: Dict[str, Optional[int]] = {robots.LEFT: None, robots.RIGHT: None}
        self._hand_gripper_states: Dict[str, int] = {robots.LEFT: 0, robots.RIGHT: 0}
        self._last_min_snap_target_command: Optional[FaUpperPositionCommand] = None
        self._last_min_snap_target_publish_time_s = 0.0
        self._startup_initial_pose_armed = self.config.initial_pose_enabled
        self._startup_initial_hand_armed = self.config.initial_hand_ros_action is not None
        self._initial_pose_started_at_s: Optional[float] = None
        self._initial_pose_done = not self._startup_initial_pose_armed
        self._teleop_ready_prompt_logged = not self._startup_initial_pose_armed
        self._last_initial_pose_publish_time_s = 0.0
        self._last_safety_log_time: Dict[str, float] = {}

    def _make_default_ik_client(self) -> FaArmIkClientBase:
        return FaPybindIkClient(self.config.ik)

    def _validate_backend(self) -> None:
        if self.control_backend not in ("real", "mujoco", "real_with_mujoco"):
            raise ValueError(f"control_backend must be one of: real, mujoco, real_with_mujoco; got {self.control_backend}")

    def stream(self):
        logger.info("Starting FA real-control backend=%s", self.control_backend)
        while True:
            start = time.time()
            self._ros2.spin_once()
            self._handle_session_command()
            self._publish_pause_hold_if_needed()
            self._publish_initial_pose_if_needed()
            if not self._initial_pose_ready():
                self._publish_lerobot_joint_states()
                command_period_s = 1.0 / max(1.0, float(self.config.command_publish_hz))
                time.sleep(max(0.0, command_period_s - (time.time() - start)))
                continue
            self._publish_initial_hand_pose_if_needed()
            self._handle_reset_requests()
            self._receive_cartesian_targets()
            self._publish_lerobot_joint_states()
            self._publish_upper_command_if_safe()
            command_period_s = 1.0 / max(1.0, float(self.config.command_publish_hz))
            time.sleep(max(0.0, command_period_s - (time.time() - start)))

    def _handle_session_command(self) -> None:
        msg = self._pause_subscriber.recv_keypoints()
        if msg is None:
            return
        if msg.command == robots.PAUSE:
            if getattr(self, "_pending_resume_after_initial_pose", False):
                self._cancel_pending_initial_pose_resume("pause command")
            if self._teleop_active:
                self._enter_pause("pause command")
        elif msg.command == robots.RESUME:
            if self._teleop_active or getattr(self, "_pending_resume_after_initial_pose", False):
                return
            resume_started_at_s = time.time()
            self._needs_reset = True
            self._real_reset_ready = self.control_backend == "mujoco"
            self._latest_targets = {robots.LEFT: None, robots.RIGHT: None}
            self._latest_target_keys = {robots.LEFT: None, robots.RIGHT: None}
            self._active_arm_goals = {robots.LEFT: None, robots.RIGHT: None}
            self._arm_goal_dirty = {robots.LEFT: False, robots.RIGHT: False}
            self._reset_ik_recovery_state()
            self._hand_init_ready = {robots.LEFT: False, robots.RIGHT: False}
            self._accept_cartesian_targets_after_s = {
                robots.LEFT: resume_started_at_s,
                robots.RIGHT: resume_started_at_s,
            }
            if not self._initial_pose_ready():
                self._pending_resume_after_initial_pose = True
                logger.info("FA resume requested; waiting initial pose before enabling teleop")
                return
            self._teleop_active = True
            snapshot = self._resume_hold_snapshot()
            if snapshot is not None:
                # Resume should seed the next IK solve from the actual arm state,
                # not from the last pre-pause IK branch.
                self._sync_ik_reference_from_snapshot(snapshot, reset_limiter=True)
                if self.config.safety_hold_arm_on_pause:
                    self._pause_hold_command = self._builder.build(snapshot.left_arm, snapshot.right_arm, snapshot.neck)
            self._resume_hold_until_target = self.config.safety_hold_arm_on_pause and self._pause_hold_command is not None

    def _cancel_pending_initial_pose_resume(self, reason: str) -> None:
        self._pending_resume_after_initial_pose = False
        self._resume_hold_until_target = False
        self._pause_hold_command = None
        self._teleop_active = False
        self._hand_init_ready = {robots.LEFT: False, robots.RIGHT: False}
        logger.info("FA pending teleop resume cancelled: %s", reason)

    def _initial_pose_ready(self, now_s: Optional[float] = None) -> bool:
        if getattr(self, "_initial_pose_done", True):
            return True
        if getattr(self, "_initial_pose_started_at_s", None) is None:
            return False
        now = time.time() if now_s is None else float(now_s)
        if now - self._initial_pose_started_at_s < self.config.initial_pose_duration_s:
            return False
        self._initial_pose_done = True
        if getattr(self, "_pending_resume_after_initial_pose", False):
            self._pending_resume_after_initial_pose = False
            self._teleop_active = True
            target_after_s = time.time()
            self._accept_cartesian_targets_after_s = {
                robots.LEFT: target_after_s,
                robots.RIGHT: target_after_s,
            }
            logger.info("FA initial pose ready; teleop resume enabled, waiting operator reset/Hand_Init_H")
        self._log_teleop_ready_once()
        return True

    def _log_teleop_ready_once(self) -> None:
        if getattr(self, "_teleop_ready_prompt_logged", False):
            return
        self._teleop_ready_prompt_logged = True
        logger.info("FA 已到达准备位置，可以开始遥操作。")

    def _publish_initial_hand_pose_if_needed(self) -> None:
        if not self._startup_initial_hand_armed:
            return
        if not self._initial_pose_done:
            return
        if not self._teleop_active:
            return
        action = self.config.initial_hand_ros_action
        if action is None or self.control_backend not in ("real", "real_with_mujoco"):
            self._startup_initial_hand_armed = False
            return
        left_published = self._ros2.publish_hand_command(robots.LEFT, int(action))
        right_published = self._ros2.publish_hand_command(robots.RIGHT, int(action))
        if not left_published or not right_published:
            self._warn_safety("initial_hand_unavailable", "FA initial hand publisher unavailable")
            return
        self._startup_initial_hand_armed = False
        logger.info("FA initial hand command published: left=%d right=%d", action, action)

    def _publish_initial_pose_if_needed(self) -> None:
        initial_pose_armed = getattr(self, "_startup_initial_pose_armed", self.config.initial_pose_enabled)
        if not initial_pose_armed or self._initial_pose_done:
            return
        if self._initial_pose_started_at_s is not None:
            self._republish_initial_pose_if_needed()
            return
        if not self.config.initial_pose_enabled:
            self._initial_pose_done = True
            self._startup_initial_pose_armed = False
            return
        if self._initial_pose_requires_fresh_joint_state() and not self._real_joint_state_fresh():
            self._warn_safety("initial_pose_joint_state_stale", "FA initial pose held until fresh /joint_states")
            return
        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            self._warn_safety("initial_pose_joint_state_missing", "FA initial pose held: missing joint state")
            return
        left = np.asarray(self.config.initial_left_arm_positions_rad, dtype=np.float64)
        right = np.asarray(self.config.initial_right_arm_positions_rad, dtype=np.float64)
        command = self._builder.build(left, right, snapshot.neck, timestamp_s=time.time())
        if not self._publish_min_snap_target_command(
            command,
            self.config.initial_pose_duration_s,
            self.config.initial_pose_max_velocity_rad_s,
            self.config.initial_pose_max_acceleration_rad_s2,
            warning_key="initial_pose_min_snap_unavailable",
            warning_message="FA initial pose min-snap target publisher unavailable",
            force=True,
        ):
            return
        self._initial_pose_started_at_s = time.time()
        self._startup_initial_pose_armed = False
        self._last_min_snap_target_command = command
        self._last_min_snap_target_publish_time_s = self._initial_pose_started_at_s
        self._last_initial_pose_publish_time_s = self._initial_pose_started_at_s
        self._last_published_upper_command = command
        self._active_arm_goals = {robots.LEFT: left.copy(), robots.RIGHT: right.copy()}
        self._arm_goal_dirty = {robots.LEFT: False, robots.RIGHT: False}
        self._last_safe_arm_targets = {robots.LEFT: left.copy(), robots.RIGHT: right.copy()}
        self._latest_targets = {robots.LEFT: None, robots.RIGHT: None}
        self._latest_target_keys = {robots.LEFT: None, robots.RIGHT: None}
        self._limiter.reset(command.values)
        logger.info(
            "FA initial teleop pose target published: duration=%.2fs max_vel=%.2f max_acc=%.2f",
            self.config.initial_pose_duration_s,
            self.config.initial_pose_max_velocity_rad_s,
            self.config.initial_pose_max_acceleration_rad_s2,
        )

    def _republish_initial_pose_if_needed(self) -> None:
        # Do not periodically resend the same initial pose target. The external
        # min-snap node treats every target as a new trajectory; repeated sends
        # reset the trajectory and show up as visible stutter.
        return

    def _initial_pose_requires_fresh_joint_state(self) -> bool:
        return self.control_backend in ("mujoco", "real", "real_with_mujoco")

    def _enter_pause(self, reason: str) -> None:
        self._teleop_active = False
        self._needs_reset = True
        self._real_reset_ready = self.control_backend == "mujoco"
        self._latest_targets = {robots.LEFT: None, robots.RIGHT: None}
        self._latest_target_keys = {robots.LEFT: None, robots.RIGHT: None}
        self._active_arm_goals = {robots.LEFT: None, robots.RIGHT: None}
        self._arm_goal_dirty = {robots.LEFT: False, robots.RIGHT: False}
        self._reset_ik_recovery_state()
        self._hand_init_ready = {robots.LEFT: False, robots.RIGHT: False}
        pause_started_at_s = time.time()
        self._accept_cartesian_targets_after_s = {
            robots.LEFT: pause_started_at_s,
            robots.RIGHT: pause_started_at_s,
        }
        snapshot = self._pause_hold_snapshot()
        if snapshot is not None:
            # The paused pose becomes the safe reference for the next resume.
            self._sync_ik_reference_from_snapshot(snapshot, reset_limiter=True)
            if self.config.safety_hold_arm_on_pause:
                self._pause_hold_command = self._builder.build(snapshot.left_arm, snapshot.right_arm, snapshot.neck)
                self._publish_pause_hold_if_needed(force=True, reason=reason)
        self._resume_hold_until_target = False

    def _pause_hold_snapshot(self) -> Optional[FaJointStateSnapshot]:
        if not self._real_joint_state_fresh():
            self._warn_safety("pause_joint_state_stale", "pause hold skipped: /joint_states is stale")
            return None
        return self._current_joint_snapshot()

    def _resume_hold_snapshot(self) -> Optional[FaJointStateSnapshot]:
        if not self._real_joint_state_fresh():
            self._warn_safety("resume_joint_state_stale", "resume hold uses previous pause target: /joint_states is stale")
            return None
        return self._current_joint_snapshot()

    def _sync_ik_reference_from_snapshot(self, snapshot: FaJointStateSnapshot, reset_limiter: bool = False) -> None:
        self._last_safe_arm_targets = {
            robots.LEFT: np.asarray(snapshot.left_arm, dtype=np.float64).copy(),
            robots.RIGHT: np.asarray(snapshot.right_arm, dtype=np.float64).copy(),
        }
        self._last_ik_cartesian_targets = {robots.LEFT: None, robots.RIGHT: None}
        self._reset_ik_recovery_state()
        if reset_limiter:
            self._limiter.reset(snapshot.upper_joints)

    def _reset_ik_recovery_state(self, hand_side: Optional[str] = None) -> None:
        if not hasattr(self, "_ik_problem_counts"):
            self._ik_problem_counts = {robots.LEFT: 0, robots.RIGHT: 0}
        if not hasattr(self, "_arm_escape_active"):
            self._arm_escape_active = {robots.LEFT: False, robots.RIGHT: False}
        if not hasattr(self, "_ik_output_filter_active"):
            self._ik_output_filter_active = {robots.LEFT: False, robots.RIGHT: False}
        if not hasattr(self, "_ik_output_filter_goals"):
            self._ik_output_filter_goals = {robots.LEFT: None, robots.RIGHT: None}
        sides = (robots.LEFT, robots.RIGHT) if hand_side is None else (hand_side,)
        for side in sides:
            self._ik_problem_counts[side] = 0
            self._arm_escape_active[side] = False
            self._ik_output_filter_active[side] = False
            self._ik_output_filter_goals[side] = None

    def _arm_escape_target(self, hand_side: str) -> Optional[np.ndarray]:
        if hand_side == robots.LEFT:
            configured = self.config.ik_escape_left_arm_positions_rad
            initial = self.config.initial_left_arm_positions_rad
        else:
            configured = self.config.ik_escape_right_arm_positions_rad
            initial = self.config.initial_right_arm_positions_rad
        values = configured if configured is not None else initial
        if values is None:
            return None
        target = np.asarray(values, dtype=np.float64)
        if target.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(target)):
            return None
        return target.copy()

    def _local_arm_escape_target(self, hand_side: str, current_arm: Sequence[float]) -> Optional[np.ndarray]:
        if not self.config.ik_escape_local_enabled:
            return self._arm_escape_target(hand_side)
        try:
            base = np.asarray(current_arm, dtype=np.float64).copy()
        except (TypeError, ValueError):
            return self._arm_escape_target(hand_side)
        if base.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(base)):
            return self._arm_escape_target(hand_side)

        target = base.copy()
        elbow_idx = 3
        desired_elbow = float(self.config.ik_escape_local_elbow_target_rad)
        max_delta = float(self.config.ik_escape_local_max_joint_delta_rad)
        if max_delta > 0.0:
            desired_elbow = float(np.clip(desired_elbow, base[elbow_idx] - max_delta, base[elbow_idx] + max_delta))
        target[elbow_idx] = desired_elbow
        return target

    def _record_arm_ik_outcome(self, update: _ArmIkUpdate) -> None:
        if not hasattr(self, "_ik_problem_counts"):
            self._ik_problem_counts = {robots.LEFT: 0, robots.RIGHT: 0}
        if not hasattr(self, "_arm_escape_active"):
            self._arm_escape_active = {robots.LEFT: False, robots.RIGHT: False}
        if update.counts_as_ik_problem:
            count = int(self._ik_problem_counts.get(update.hand_side, 0)) + 1
            self._ik_problem_counts[update.hand_side] = count
            if (
                self.config.ik_escape_enabled
                and count >= self.config.ik_escape_trigger_count
                and not self._arm_escape_active.get(update.hand_side, False)
                and self._arm_escape_target(update.hand_side) is not None
            ):
                self._arm_escape_active[update.hand_side] = True
                self._warn_safety(
                    f"{update.hand_side}_ik_escape_trigger",
                    (
                        f"{update.hand_side} IK escape mode triggered after "
                        f"{count} consecutive IK problem frames"
                    ),
                )
            return
        self._ik_problem_counts[update.hand_side] = 0
        self._arm_escape_active[update.hand_side] = False

    def _publish_pause_hold_if_needed(self, force: bool = False, reason: str = "pause hold") -> None:
        if not force:
            return
        if (
            (self._teleop_active and not getattr(self, "_resume_hold_until_target", False))
            or not self.config.safety_hold_arm_on_pause
            or self._pause_hold_command is None
        ):
            return
        now = time.time()
        command = FaUpperPositionCommand(now, self._pause_hold_command.values)
        if self._publish_upper_command_outputs(
            command,
            require_real_reset=False,
            allow_stale_real_hold=True,
            force_min_snap_publish=True,
        ):
            self._pause_hold_command = command
            self._last_pause_hold_publish_time = now

    def _handle_reset_requests(self) -> None:
        if not self._initial_pose_ready():
            self._warn_safety(
                "reset_initial_pose_pending",
                "reset held until FA startup initial pose is ready",
            )
            return
        if not self._teleop_active:
            self._warn_safety(
                "reset_teleop_inactive",
                "reset held until FA teleop resume is enabled after initial pose",
            )
            return
        for hand_side, subscriber, publish_port in (
            (robots.RIGHT, self._right_reset_subscriber, self._right_endeff_publish_port),
            (robots.LEFT, self._left_reset_subscriber, self._left_endeff_publish_port),
        ):
            if subscriber.recv_keypoints() is None:
                continue
            self._publish_current_endeff_homo(hand_side, publish_port)

    def _publish_current_endeff_homo(self, hand_side: str, publish_port: int) -> None:
        if not self._real_joint_state_fresh():
            self._warn_safety("reset_joint_state_stale", f"reset refused for {hand_side}: joint state stale")
            return
        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            self._warn_safety("reset_joint_state_missing", f"reset refused for {hand_side}: missing joint state")
            return
        arm_joints = snapshot.left_arm if hand_side == robots.LEFT else snapshot.right_arm
        homo = self._ik_client.compute_fk(hand_side, arm_joints)
        if homo is None and self._kinematics is not None and self._kinematics.available:
            homo = self._kinematics.fk(hand_side, snapshot.arm_joints)
        if homo is None:
            self._warn_safety("reset_fk_unavailable", f"reset refused for {hand_side}: FK failed")
            return
        self._publisher_manager.publish(
            self.host,
            publish_port,
            "endeff_homo",
            CartesianState(timestamp_s=time.time(), h_matrix=tuple(tuple(float(v) for v in row) for row in homo)),
        )
        self._needs_reset = False
        self._real_reset_ready = True
        self._hand_init_ready[hand_side] = True
        if not hasattr(self, "_accept_cartesian_targets_after_s"):
            self._accept_cartesian_targets_after_s = {robots.LEFT: 0.0, robots.RIGHT: 0.0}
        # After a pause/resume edge, stale Cartesian targets may still be the
        # latest ZMQ sample. Only consume targets produced after this fresh
        # Hand_Init_H response.
        self._accept_cartesian_targets_after_s[hand_side] = time.time()
        self._latest_targets[hand_side] = None
        self._latest_target_keys[hand_side] = None
        self._last_ik_cartesian_targets[hand_side] = None
        self._reset_ik_recovery_state(hand_side)
        self._sync_ik_reference_from_snapshot(snapshot, reset_limiter=True)
        self._active_arm_goals = {
            robots.LEFT: np.asarray(snapshot.left_arm, dtype=np.float64),
            robots.RIGHT: np.asarray(snapshot.right_arm, dtype=np.float64),
        }
        self._arm_goal_dirty = {robots.LEFT: False, robots.RIGHT: False}

    def _receive_cartesian_targets(self) -> None:
        for hand_side, subscriber in ((robots.LEFT, self._left_target_subscriber), (robots.RIGHT, self._right_target_subscriber)):
            msg = subscriber.recv_keypoints()
            if msg is None:
                continue
            if msg.hand_side != hand_side:
                self._warn_safety(f"{hand_side}_wrong_target_side", f"wrong target side: {msg.hand_side}")
                continue
            if not self._hand_init_ready.get(hand_side, False):
                self._warn_safety(
                    f"{hand_side}_hand_init_pending",
                    f"{hand_side} target held until Hand_Init_H reset completes",
                )
                continue
            min_target_timestamp_s = float(
                getattr(self, "_accept_cartesian_targets_after_s", {}).get(hand_side, 0.0)
            )
            target_timestamp_s = float(getattr(msg, "timestamp_s", 0.0) or 0.0)
            if min_target_timestamp_s > 0.0 and target_timestamp_s <= min_target_timestamp_s:
                self._warn_safety(
                    f"{hand_side}_stale_after_hand_init_target",
                    (
                        f"{hand_side} target held until a fresh post-Hand_Init_H frame arrives "
                        f"(target_ts={target_timestamp_s:.6f}, min_ts={min_target_timestamp_s:.6f})"
                    ),
                )
                continue
            # A fresh operator target means the post-resume rebaseline finished;
            # stop replaying the paused hold so this target owns the arm command.
            self._resume_hold_until_target = False
            self._latest_targets[hand_side] = msg
            self._publish_hand_command_on_edge(hand_side, msg.hand_command)

    def _publish_hand_command_on_edge(self, hand_side: str, hand_command) -> None:
        if hand_command is None:
            return
        try:
            command = int(hand_command)
        except (TypeError, ValueError):
            logger.warning("忽略无法解析的FA/O6手部命令: %s", hand_command)
            return
        if command == FA_HAND_GRASP_COMMAND:
            gripper_state = 1
            ros_action = self.config.hand_grasp_ros_action
        elif command == FA_HAND_OPEN_COMMAND:
            gripper_state = 0
            ros_action = self.config.hand_open_ros_action
        else:
            logger.warning("忽略未知FA/O6手部命令: %s", command)
            return
        if command == self._last_hand_commands[hand_side]:
            return
        if self.control_backend in ("real", "real_with_mujoco"):
            published = self._ros2.publish_hand_command(hand_side, ros_action)
            if not published and self.control_backend == "real":
                self._warn_safety("ros_hand_unavailable", f"FA/O6 hand publisher unavailable for {hand_side}")
        self._last_hand_commands[hand_side] = command
        self._hand_gripper_states[hand_side] = gripper_state
        logger.info(
            "FA/O6 hand command: %s command=%d state=%d ros_data=%d",
            hand_side,
            command,
            gripper_state,
            ros_action,
        )

    def _publish_lerobot_joint_states(self) -> None:
        if self.config.state_publish_fps <= 0.0:
            return
        now = time.time()
        if now < self._next_state_publish_time_s:
            return
        self._next_state_publish_time_s = now + 1.0 / float(self.config.state_publish_fps)
        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            return
        self._publish_lerobot_arm_state(robots.RIGHT, snapshot.right_arm, self._right_state_publish_port, now)
        self._publish_lerobot_arm_state(robots.LEFT, snapshot.left_arm, self._left_state_publish_port, now)

    def _publish_lerobot_arm_state(self, hand_side: str, joints: Sequence[float], port: int, now: float) -> None:
        hand_command = self._last_hand_commands.get(hand_side)
        if hand_command is None:
            hand_command = FA_HAND_OPEN_COMMAND
        hand_gripper_state = int(self._hand_gripper_states.get(hand_side, 0))
        state = {
            "joint_states": {"joint_position": [float(v) for v in joints], "timestamp": now},
            "joint_angles_rad": [float(v) for v in joints],
            "hand_command": int(hand_command),
            "hand_gripper_state": hand_gripper_state,
            "timestamp": now,
        }
        if self._last_published_upper_command is not None:
            command_joints = (
                self._last_published_upper_command.left_arm
                if hand_side == robots.LEFT
                else self._last_published_upper_command.right_arm
            )
            state["commanded_joint_states"] = {
                "joint_position": [float(v) for v in command_joints],
                "timestamp_s": self._last_published_upper_command.timestamp_s,
            }
        self._publisher_manager.publish(self.host, port, f"fa_{hand_side}", state)

    def _publish_upper_command_if_safe(self) -> None:
        if not self._teleop_active:
            return
        if not self._real_joint_state_fresh():
            self._warn_safety("joint_state_stale", "FA command rejected: /joint_states stale")
            return
        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            self._warn_safety("joint_state_missing", "FA command rejected: missing joint state")
            return
        self._update_active_arm_goals(snapshot)
        dirty_without_hand_init = [
            side for side, dirty in getattr(self, "_arm_goal_dirty", {}).items()
            if dirty and not self._hand_init_ready.get(side, False)
        ]
        if dirty_without_hand_init:
            self._warn_safety(
                "hand_init_pending",
                f"FA command held until Hand_Init_H reset completes for {dirty_without_hand_init}",
            )
            return
        if not any(getattr(self, "_arm_goal_dirty", {}).values()):
            return
        active_goals = getattr(self, "_active_arm_goals", {robots.LEFT: None, robots.RIGHT: None})
        if active_goals.get(robots.LEFT) is None and active_goals.get(robots.RIGHT) is None:
            return
        now = time.time()
        desired_left = (
            np.asarray(active_goals[robots.LEFT], dtype=np.float64)
            if active_goals.get(robots.LEFT) is not None
            else np.asarray(snapshot.left_arm, dtype=np.float64)
        )
        desired_right = (
            np.asarray(active_goals[robots.RIGHT], dtype=np.float64)
            if active_goals.get(robots.RIGHT) is not None
            else np.asarray(snapshot.right_arm, dtype=np.float64)
        )
        try:
            command = self._builder.build(desired_left, desired_right, snapshot.neck, timestamp_s=now)
            limited, reason = self._limiter.limit(command, now_s=command.timestamp_s)
        except Exception as exc:
            self._warn_safety("command_build_fail", f"FA command build failed: {exc}")
            return
        if limited is None:
            self._warn_safety("command_limited_reject", f"FA command rejected: {reason}")
            return
        published = self._publish_upper_command_outputs(limited, require_real_reset=True)
        if published and getattr(self, "_last_min_snap_target_command", None) is limited:
            self._update_arm_goal_dirty_after_publish(limited)

    def _update_arm_goal_dirty_after_publish(self, command: FaUpperPositionCommand) -> None:
        active_goals = getattr(self, "_active_arm_goals", {robots.LEFT: None, robots.RIGHT: None})
        if not hasattr(self, "_arm_goal_dirty"):
            self._arm_goal_dirty = {robots.LEFT: False, robots.RIGHT: False}
        tolerance = max(1e-6, float(self.config.min_snap_target_epsilon_rad))
        for hand_side, commanded in (
            (robots.LEFT, np.asarray(command.left_arm, dtype=np.float64)),
            (robots.RIGHT, np.asarray(command.right_arm, dtype=np.float64)),
        ):
            if not self._arm_goal_dirty.get(hand_side, False):
                continue
            goal = active_goals.get(hand_side)
            if goal is None:
                self._arm_goal_dirty[hand_side] = False
                continue
            remaining = float(np.max(np.abs(commanded - np.asarray(goal, dtype=np.float64))))
            if self._arm_escape_active.get(hand_side, False):
                escape_tolerance = max(tolerance, float(self.config.ik_escape_target_tolerance_rad))
                escaped = remaining <= escape_tolerance
                self._arm_goal_dirty[hand_side] = not escaped
                if escaped:
                    self._arm_escape_active[hand_side] = False
                    self._ik_problem_counts[hand_side] = 0
                    self._last_safe_arm_targets[hand_side] = np.asarray(goal, dtype=np.float64).copy()
                    self._last_ik_cartesian_targets[hand_side] = None
                    self._latest_target_keys[hand_side] = None
                    logger.info("%s IK escape complete; resuming Cartesian IK", hand_side)
                continue
            self._arm_goal_dirty[hand_side] = remaining > tolerance

    def _update_active_arm_goals(self, snapshot: FaJointStateSnapshot) -> None:
        self._ensure_arm_ik_workers()
        if not hasattr(self, "_latest_target_keys"):
            self._latest_target_keys = {robots.LEFT: None, robots.RIGHT: None}
        if not hasattr(self, "_active_arm_goals"):
            self._active_arm_goals = {robots.LEFT: None, robots.RIGHT: None}
        if not hasattr(self, "_arm_goal_dirty"):
            self._arm_goal_dirty = {robots.LEFT: False, robots.RIGHT: False}
        if not hasattr(self, "_last_ik_cartesian_targets"):
            self._last_ik_cartesian_targets = {robots.LEFT: None, robots.RIGHT: None}
        if not hasattr(self, "_pending_ik_futures"):
            self._pending_ik_futures = {}
        if not hasattr(self, "_arm_escape_active"):
            self._arm_escape_active = {robots.LEFT: False, robots.RIGHT: False}
        self._apply_completed_arm_ik_updates()
        for hand_side in (robots.LEFT, robots.RIGHT):
            target = self._latest_targets[hand_side]
            if target is None:
                continue
            target_key = self._cartesian_target_key(target)
            if self._arm_escape_active.get(hand_side, False):
                current_arm = snapshot.left_arm if hand_side == robots.LEFT else snapshot.right_arm
                escape_seed = self._last_safe_arm_targets.get(hand_side)
                escape_base = escape_seed if escape_seed is not None else current_arm
                escape_target = self._local_arm_escape_target(hand_side, escape_base)
                if escape_target is None:
                    self._arm_escape_active[hand_side] = False
                else:
                    pending = self._pending_ik_futures.pop(hand_side, None)
                    if pending is not None:
                        pending.cancel()
                    self._active_arm_goals[hand_side] = escape_target
                    self._arm_goal_dirty[hand_side] = True
                    self._latest_target_keys[hand_side] = ("escape", target_key)
                    continue
            if target_key == self._latest_target_keys.get(hand_side):
                continue
            pending = self._pending_ik_futures.get(hand_side)
            if pending is not None and not pending.done():
                continue
            last_safe = self._last_safe_arm_targets[hand_side]
            last_ik_target = self._last_ik_cartesian_targets[hand_side]
            if last_safe is not None and last_ik_target is not None and self._cartesian_target_within_ik_deadband(
                target, last_ik_target
            ):
                self._active_arm_goals[hand_side] = last_safe.copy()
                self._arm_goal_dirty[hand_side] = False
                self._latest_target_keys[hand_side] = target_key
                continue
            current_arm = snapshot.left_arm if hand_side == robots.LEFT else snapshot.right_arm
            current_arm_array = np.asarray(current_arm, dtype=np.float64)
            ik_seed = last_safe if last_safe is not None else current_arm_array
            self._pending_ik_futures[hand_side] = self._ik_executor.submit(
                self._solve_arm_ik_update,
                hand_side,
                target,
                target_key,
                last_ik_target,
                np.asarray(ik_seed, dtype=np.float64).copy(),
                current_arm_array.copy(),
                None if last_safe is None else np.asarray(last_safe, dtype=np.float64).copy(),
            )
        self._apply_completed_arm_ik_updates()

    def _ensure_arm_ik_workers(self) -> None:
        if not hasattr(self, "_ik_clients"):
            self._ik_clients = {robots.LEFT: self._ik_client, robots.RIGHT: self._ik_client}
        if not hasattr(self, "_ik_executor"):
            self._ik_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fa_arm_ik")
        if not hasattr(self, "_pending_ik_futures"):
            self._pending_ik_futures = {}

    def _apply_completed_arm_ik_updates(self) -> None:
        for hand_side, future in list(getattr(self, "_pending_ik_futures", {}).items()):
            if not future.done():
                continue
            self._pending_ik_futures.pop(hand_side, None)
            try:
                update = future.result()
            except Exception as exc:
                self._warn_safety(f"{hand_side}_ik_worker_fail", f"{hand_side} IK worker failed: {exc}")
                continue
            current_target = self._latest_targets.get(hand_side)
            if current_target is None:
                continue
            if update.warn_key is not None and update.warn_message is not None:
                self._warn_safety(update.warn_key, update.warn_message)
            self._record_arm_ik_outcome(update)
            if not update.consume_target:
                continue
            active_goal, last_safe = self._filter_arm_ik_output_if_needed(
                hand_side,
                update.active_goal,
                update.last_safe,
            )
            self._active_arm_goals[hand_side] = (
                None if active_goal is None else np.asarray(active_goal, dtype=np.float64)
            )
            self._arm_goal_dirty[hand_side] = bool(update.dirty)
            self._latest_target_keys[hand_side] = update.target_key
            if last_safe is not None:
                self._last_safe_arm_targets[hand_side] = np.asarray(last_safe, dtype=np.float64).copy()
            if update.last_ik_target is not None:
                self._last_ik_cartesian_targets[hand_side] = update.last_ik_target

    def _filter_arm_ik_output_if_needed(
        self,
        hand_side: str,
        active_goal: Optional[np.ndarray],
        last_safe: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.config.ik_singularity_output_filter_enabled or active_goal is None:
            return active_goal, last_safe
        if not hasattr(self, "_ik_output_filter_active"):
            self._ik_output_filter_active = {robots.LEFT: False, robots.RIGHT: False}
        if not hasattr(self, "_ik_output_filter_goals"):
            self._ik_output_filter_goals = {robots.LEFT: None, robots.RIGHT: None}

        goal = np.asarray(active_goal, dtype=np.float64)
        if goal.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(goal)):
            return active_goal, last_safe

        problem_count = int(getattr(self, "_ik_problem_counts", {}).get(hand_side, 0))
        elbow = float(goal[3])
        active = bool(self._ik_output_filter_active.get(hand_side, False))
        enter = (
            problem_count >= self.config.ik_singularity_filter_enter_problem_count
            or elbow <= self.config.ik_singularity_filter_elbow_enter_rad
            or elbow >= self.config.ik_singularity_filter_elbow_upper_enter_rad
        )
        exit_ready = (
            problem_count <= self.config.ik_singularity_filter_exit_problem_count
            and elbow >= self.config.ik_singularity_filter_elbow_exit_rad
            and elbow <= self.config.ik_singularity_filter_elbow_upper_exit_rad
        )

        if not active and enter:
            active = True
            self._ik_output_filter_active[hand_side] = True
            previous = self._last_published_arm_goal(hand_side)
            if previous is None:
                previous = getattr(self, "_active_arm_goals", {}).get(hand_side)
            self._ik_output_filter_goals[hand_side] = (
                None if previous is None else np.asarray(previous, dtype=np.float64).copy()
            )
            self._warn_safety(
                f"{hand_side}_ik_output_filter_enter",
                (
                    f"{hand_side} IK output filter enabled near singular/extended arm: "
                    f"elbow={elbow:.3f} problem_count={problem_count}"
                ),
            )
        elif active and exit_ready:
            active = False
            self._ik_output_filter_active[hand_side] = False
            self._ik_output_filter_goals[hand_side] = None
            self._warn_safety(
                f"{hand_side}_ik_output_filter_exit",
                (
                    f"{hand_side} IK output filter disabled after recovery: "
                    f"elbow={elbow:.3f} problem_count={problem_count}"
                ),
            )

        if not active:
            return active_goal, last_safe

        previous = self._ik_output_filter_goals.get(hand_side)
        if previous is None or previous.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(previous)):
            previous = goal
        alpha = float(self.config.ik_singularity_output_filter_alpha)
        filtered = alpha * previous + (1.0 - alpha) * goal
        max_step = float(self.config.ik_singularity_output_filter_max_step_rad)
        if max_step > 0.0:
            filtered = previous + np.clip(filtered - previous, -max_step, max_step)
        self._ik_output_filter_goals[hand_side] = filtered.copy()
        filtered_last_safe = filtered.copy() if last_safe is not None else last_safe
        return filtered, filtered_last_safe

    def _last_published_arm_goal(self, hand_side: str) -> Optional[np.ndarray]:
        command = getattr(self, "_last_published_upper_command", None)
        if command is None:
            return None
        arm = command.left_arm if hand_side == robots.LEFT else command.right_arm
        values = np.asarray(arm, dtype=np.float64)
        if values.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(values)):
            return None
        return values.copy()

    def _arm_ik_seed_candidates(
        self,
        hand_side: str,
        primary_seed: np.ndarray,
        current_arm: np.ndarray,
    ) -> list[np.ndarray]:
        seeds = [np.asarray(primary_seed, dtype=np.float64)]
        if self.config.ik_multi_seed_enabled:
            seeds.append(np.asarray(current_arm, dtype=np.float64))
            escape_target = self._local_arm_escape_target(hand_side, current_arm)
            if escape_target is not None:
                seeds.append(escape_target)
        unique: list[np.ndarray] = []
        for seed in seeds:
            if seed.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(seed)):
                continue
            if any(np.allclose(seed, previous, atol=1e-6, rtol=0.0) for previous in unique):
                continue
            unique.append(seed.copy())
        return unique or [np.asarray(primary_seed, dtype=np.float64).copy()]

    def _solution_jump_exceeds_limit(self, solved: np.ndarray, reference: np.ndarray) -> tuple[bool, float]:
        max_jump = float(np.max(np.abs(np.asarray(solved, dtype=np.float64) - np.asarray(reference, dtype=np.float64))))
        return self.config.max_ik_solution_jump_rad > 0.0 and max_jump > self.config.max_ik_solution_jump_rad, max_jump

    @staticmethod
    def _ik_result_score(ik) -> float:
        return float(ik.position_error) + 0.25 * float(ik.orientation_error)

    def _solve_best_arm_ik(
        self,
        hand_side: str,
        target: CartesianTarget,
        primary_seed: np.ndarray,
        current_arm: np.ndarray,
        ik_client: FaArmIkClientBase,
    ):
        candidates = []
        for seed in self._arm_ik_seed_candidates(hand_side, primary_seed, current_arm):
            ik = ik_client.solve(hand_side, target, seed)
            if not ik.success:
                continue
            solved = np.asarray(ik.q_target, dtype=np.float64)
            jump_exceeds, max_jump = self._solution_jump_exceeds_limit(solved, primary_seed)
            quality_exceeds = self._ik_error_exceeds_quality_limit(ik)
            candidates.append(
                {
                    "ik": ik,
                    "seed": seed,
                    "solved": solved,
                    "quality_exceeds": quality_exceeds,
                    "jump_exceeds": jump_exceeds,
                    "max_jump": max_jump,
                    "score": self._ik_result_score(ik),
                }
            )
        if not candidates:
            return None, None
        accepted = [
            item for item in candidates
            if not item["quality_exceeds"] and not item["jump_exceeds"]
        ]
        if accepted:
            return min(accepted, key=lambda item: (item["score"], item["max_jump"])), min(
                candidates, key=lambda item: item["score"]
            )
        return None, min(candidates, key=lambda item: item["score"])

    def _solve_arm_ik_update(
        self,
        hand_side: str,
        target: CartesianTarget,
        target_key: tuple,
        last_ik_target: Optional[CartesianTarget],
        ik_seed: np.ndarray,
        current_arm: np.ndarray,
        last_safe: Optional[np.ndarray],
    ) -> _ArmIkUpdate:
        ik_client = getattr(self, "_ik_clients", {}).get(hand_side, self._ik_client)
        accepted, best = self._solve_best_arm_ik(hand_side, target, ik_seed, current_arm, ik_client)
        if accepted is not None:
            ik = accepted["ik"]
            solved = np.asarray(accepted["solved"], dtype=np.float64)
            return _ArmIkUpdate(
                hand_side=hand_side,
                target_key=target_key,
                active_goal=solved,
                dirty=True,
                last_safe=solved.copy(),
                last_ik_target=target,
            )
        if best is None:
            if last_safe is None:
                return _ArmIkUpdate(
                    hand_side=hand_side,
                    target_key=target_key,
                    active_goal=None,
                    dirty=False,
                    consume_target=False,
                    warn_key=f"{hand_side}_ik_fail",
                    warn_message=f"{hand_side} IK failed for all seed candidates",
                    counts_as_ik_problem=True,
                )
            return _ArmIkUpdate(
                hand_side=hand_side,
                target_key=target_key,
                active_goal=last_safe.copy(),
                dirty=False,
                warn_key=f"{hand_side}_ik_fail",
                warn_message=f"{hand_side} IK failed for all seed candidates",
                counts_as_ik_problem=True,
            )

        ik = best["ik"]
        solved = np.asarray(best["solved"], dtype=np.float64)
        reference = ik_seed
        if self._ik_error_exceeds_quality_limit(ik):
            fallback_start = last_ik_target or self._cartesian_target_from_fk(
                hand_side, target, reference, ik_client
            )
            if self._cartesian_target_is_returning_from_reach(fallback_start, target) and (
                self._ik_within_return_recovery_limit(ik)
            ):
                max_jump = float(np.max(np.abs(solved - reference)))
                if self.config.max_ik_solution_jump_rad <= 0.0 or max_jump <= self.config.max_ik_solution_jump_rad:
                    return _ArmIkUpdate(
                        hand_side=hand_side,
                        target_key=target_key,
                        active_goal=solved,
                        dirty=True,
                        last_safe=solved.copy(),
                        last_ik_target=target,
                        warn_key=f"{hand_side}_ik_return_recovery",
                        warn_message=(
                            f"{hand_side} IK accepted returning approximate solution: "
                            f"position_error={ik.position_error:.3f}m "
                            f"orientation_error={ik.orientation_error:.3f}rad"
                        ),
                    )
            fallback = self._solve_reachable_ik_fallback(
                hand_side, fallback_start, target, reference, ik_client
            )
            if fallback is not None:
                fallback_ik, fallback_target, alpha = fallback
                solved = np.asarray(fallback_ik.q_target, dtype=np.float64)
                return _ArmIkUpdate(
                    hand_side=hand_side,
                    target_key=target_key,
                    active_goal=solved,
                    dirty=True,
                    last_safe=solved.copy(),
                    last_ik_target=fallback_target,
                    warn_key=f"{hand_side}_ik_reachable_fallback",
                    warn_message=(
                        f"{hand_side} IK target backed off: alpha={alpha:.3f} "
                        f"position_error={fallback_ik.position_error:.3f}m "
                        f"orientation_error={fallback_ik.orientation_error:.3f}rad"
                    ),
                    counts_as_ik_problem=True,
                )
            return _ArmIkUpdate(
                hand_side=hand_side,
                target_key=target_key,
                active_goal=np.asarray(reference, dtype=np.float64),
                dirty=False,
                warn_key=f"{hand_side}_ik_quality_hold",
                warn_message=(
                    f"{hand_side} IK solution held: position_error={ik.position_error:.3f}m "
                    f"orientation_error={ik.orientation_error:.3f}rad"
                ),
                counts_as_ik_problem=True,
            )

        jump_exceeds, max_jump = self._solution_jump_exceeds_limit(solved, reference)
        if jump_exceeds:
            return _ArmIkUpdate(
                hand_side=hand_side,
                target_key=target_key,
                active_goal=np.asarray(reference, dtype=np.float64),
                dirty=False,
                warn_key=f"{hand_side}_ik_solution_jump",
                warn_message=(
                    f"{hand_side} IK solution held: solution_jump={max_jump:.3f}rad "
                    f"limit={self.config.max_ik_solution_jump_rad:.3f}rad"
                ),
                counts_as_ik_problem=True,
            )
        return _ArmIkUpdate(
            hand_side=hand_side,
            target_key=target_key,
            active_goal=solved,
            dirty=True,
            last_safe=solved.copy(),
            last_ik_target=target,
        )

    def _cartesian_target_from_fk(
        self,
        hand_side: str,
        template: CartesianTarget,
        current_arm_q: Sequence[float],
        ik_client: FaArmIkClientBase,
    ) -> Optional[CartesianTarget]:
        homo = ik_client.compute_fk(hand_side, current_arm_q)
        if homo is None:
            return None
        matrix = np.asarray(homo, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return None
        return replace(
            template,
            position_m=tuple(float(v) for v in matrix[:3, 3]),
            orientation_xyzw=tuple(float(v) for v in self._rotation_to_quaternion_xyzw(matrix[:3, :3])),
        )

    def _solve_reachable_ik_fallback(
        self,
        hand_side: str,
        previous_target: Optional[CartesianTarget],
        current_target: CartesianTarget,
        reference: np.ndarray,
        ik_client: Optional[FaArmIkClientBase] = None,
    ) -> Optional[tuple[object, CartesianTarget, float]]:
        if not self.config.ik_reachable_fallback_enabled or previous_target is None:
            return None
        ik_client = ik_client or getattr(self, "_ik_clients", {}).get(hand_side, self._ik_client)
        best = None
        for orientation_alpha in self.config.ik_reachable_fallback_orientation_alphas:
            low = 0.0
            high = 1.0
            for _ in range(self.config.ik_reachable_fallback_iterations):
                alpha = 0.5 * (low + high)
                if alpha < self.config.ik_reachable_fallback_min_alpha:
                    break
                candidate = self._interpolate_cartesian_target(
                    previous_target,
                    current_target,
                    alpha,
                    orientation_alpha=orientation_alpha,
                )
                ik = ik_client.solve(hand_side, candidate, reference)
                if not ik.success or self._ik_error_exceeds_quality_limit(ik):
                    high = alpha
                    continue
                solved = np.asarray(ik.q_target, dtype=np.float64)
                max_jump = float(np.max(np.abs(solved - reference)))
                if self.config.max_ik_solution_jump_rad > 0.0 and max_jump > self.config.max_ik_solution_jump_rad:
                    high = alpha
                    continue
                best = (ik, candidate, alpha)
                low = alpha
            if best is not None:
                return best
        return best

    def _interpolate_cartesian_target(
        self,
        previous: CartesianTarget,
        current: CartesianTarget,
        alpha: float,
        orientation_alpha: Optional[float] = None,
    ) -> CartesianTarget:
        alpha = min(1.0, max(0.0, float(alpha)))
        orientation_alpha = alpha if orientation_alpha is None else min(1.0, max(0.0, float(orientation_alpha)))
        prev_pos = np.asarray(previous.position_m, dtype=np.float64)
        curr_pos = np.asarray(current.position_m, dtype=np.float64)
        position = tuple((prev_pos + (curr_pos - prev_pos) * alpha).tolist())
        orientation = tuple(
            self._slerp_quaternion_xyzw(previous.orientation_xyzw, current.orientation_xyzw, orientation_alpha).tolist()
        )
        return replace(current, position_m=position, orientation_xyzw=orientation)

    @staticmethod
    def _slerp_quaternion_xyzw(
        previous_xyzw: Sequence[float],
        current_xyzw: Sequence[float],
        alpha: float,
    ) -> np.ndarray:
        previous = np.asarray(previous_xyzw, dtype=np.float64)
        current = np.asarray(current_xyzw, dtype=np.float64)
        previous_norm = float(np.linalg.norm(previous))
        current_norm = float(np.linalg.norm(current))
        if previous_norm <= 0.0 or current_norm <= 0.0:
            return current if current_norm > 0.0 else np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        previous = previous / previous_norm
        current = current / current_norm
        dot = float(np.dot(previous, current))
        if dot < 0.0:
            current = -current
            dot = -dot
        dot = min(1.0, max(-1.0, dot))
        if dot > 0.9995:
            out = previous + alpha * (current - previous)
            return out / np.linalg.norm(out)
        theta_0 = float(np.arccos(dot))
        sin_theta_0 = float(np.sin(theta_0))
        theta = theta_0 * alpha
        scale_previous = float(np.sin(theta_0 - theta)) / sin_theta_0
        scale_current = float(np.sin(theta)) / sin_theta_0
        return scale_previous * previous + scale_current * current

    @staticmethod
    def _rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
        matrix = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = float(np.sqrt(trace + 1.0) * 2.0)
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        else:
            diag = np.diag(matrix)
            idx = int(np.argmax(diag))
            if idx == 0:
                scale = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0)
                qw = (matrix[2, 1] - matrix[1, 2]) / scale
                qx = 0.25 * scale
                qy = (matrix[0, 1] + matrix[1, 0]) / scale
                qz = (matrix[0, 2] + matrix[2, 0]) / scale
            elif idx == 1:
                scale = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0)
                qw = (matrix[0, 2] - matrix[2, 0]) / scale
                qx = (matrix[0, 1] + matrix[1, 0]) / scale
                qy = 0.25 * scale
                qz = (matrix[1, 2] + matrix[2, 1]) / scale
            else:
                scale = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0)
                qw = (matrix[1, 0] - matrix[0, 1]) / scale
                qx = (matrix[0, 2] + matrix[2, 0]) / scale
                qy = (matrix[1, 2] + matrix[2, 1]) / scale
                qz = 0.25 * scale
        quat = np.asarray((qx, qy, qz, qw), dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm <= 0.0 or not np.isfinite(norm):
            return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        return quat / norm

    def _ik_error_exceeds_quality_limit(self, ik: FaArmIkResult) -> bool:
        return (
            self.config.ik_max_position_error_m > 0.0
            and float(ik.position_error) > self.config.ik_max_position_error_m
        ) or (
            self.config.ik_max_orientation_error_rad > 0.0
            and float(ik.orientation_error) > self.config.ik_max_orientation_error_rad
        )

    def _ik_within_return_recovery_limit(self, ik: FaArmIkResult) -> bool:
        if not self.config.ik_return_recovery_enabled:
            return False
        return (
            self.config.ik_return_recovery_max_position_error_m <= 0.0
            or float(ik.position_error) <= self.config.ik_return_recovery_max_position_error_m
        ) and (
            self.config.ik_return_recovery_max_orientation_error_rad <= 0.0
            or float(ik.orientation_error) <= self.config.ik_return_recovery_max_orientation_error_rad
        )

    def _cartesian_target_is_returning_from_reach(
        self,
        previous: Optional[CartesianTarget],
        current: CartesianTarget,
    ) -> bool:
        if not self.config.ik_return_recovery_enabled or previous is None:
            return False
        retreat = float(self.config.ik_return_recovery_min_retreat_m)
        previous_pos = np.asarray(previous.position_m, dtype=np.float64)
        current_pos = np.asarray(current.position_m, dtype=np.float64)
        if previous_pos.shape != (3,) or current_pos.shape != (3,):
            return False
        forward_retreat = float(previous_pos[0] - current_pos[0])
        radial_retreat = float(np.linalg.norm(previous_pos) - np.linalg.norm(current_pos))
        return forward_retreat >= retreat or radial_retreat >= retreat

    def _cartesian_target_within_ik_deadband(self, current: CartesianTarget, previous: CartesianTarget) -> bool:
        position_delta = np.asarray(current.position_m, dtype=np.float64) - np.asarray(
            previous.position_m, dtype=np.float64
        )
        position_delta_m = float(np.linalg.norm(position_delta))
        orientation_delta_rad = self._quaternion_angle_delta_rad(
            current.orientation_xyzw, previous.orientation_xyzw
        )
        return (
            position_delta_m < self.config.ik_cartesian_position_deadband_m and
            orientation_delta_rad < self.config.ik_cartesian_orientation_deadband_rad
        )

    @staticmethod
    def _quaternion_angle_delta_rad(current_xyzw: Sequence[float], previous_xyzw: Sequence[float]) -> float:
        current = np.asarray(current_xyzw, dtype=np.float64)
        previous = np.asarray(previous_xyzw, dtype=np.float64)
        current_norm = float(np.linalg.norm(current))
        previous_norm = float(np.linalg.norm(previous))
        if current_norm <= 0.0 or previous_norm <= 0.0:
            return float("inf")
        current = current / current_norm
        previous = previous / previous_norm
        dot = abs(float(np.dot(current, previous)))
        dot = min(1.0, max(-1.0, dot))
        return 2.0 * float(np.arccos(dot))

    def _cartesian_target_key(self, target: CartesianTarget) -> tuple:
        timestamp_s = float(getattr(target, "timestamp_s", 0.0) or 0.0)
        if timestamp_s > 0.0:
            return ("timestamp", timestamp_s)
        return ("object", id(target))

    def _publish_upper_command_outputs(
        self,
        command: FaUpperPositionCommand,
        require_real_reset: bool,
        allow_stale_real_hold: bool = False,
        force_min_snap_publish: bool = False,
    ) -> bool:
        if len(command.values) != FA_UPPER_COMMAND_LENGTH:
            raise ValueError("Refusing to publish non-16D FA command")
        if self.control_backend in ("real", "real_with_mujoco"):
            state_ready = self._real_joint_state_fresh() or allow_stale_real_hold
            reset_ready = self._real_reset_ready or not require_real_reset
            if not state_ready or not reset_ready:
                self._warn_safety("real_reset_required", "FA real command held until fresh /joint_states and reset")
                return False
        if not force_min_snap_publish and not self._should_publish_min_snap_target(command):
            return True
        return self._publish_min_snap_target_command(
            command,
            self.config.min_snap_expected_duration_s,
            self.config.min_snap_max_velocity_rad_s,
            self.config.min_snap_max_acceleration_rad_s2,
            warning_key="min_snap_unavailable",
            warning_message="FA min-snap target publisher unavailable",
            force=True,
        )

    def _publish_min_snap_target_command(
        self,
        command: FaUpperPositionCommand,
        expected_duration_s: float,
        max_velocity_rad_s: float,
        max_acceleration_rad_s2: float,
        *,
        warning_key: str,
        warning_message: str,
        force: bool = False,
    ) -> bool:
        if not force and not self._should_publish_min_snap_target(command):
            return True
        if not self._ros2.publish_min_snap_target(
            command,
            expected_duration_s,
            max_velocity_rad_s,
            max_acceleration_rad_s2,
        ):
            self._warn_safety(warning_key, warning_message)
            return False
        self._last_min_snap_target_command = command
        self._last_min_snap_target_publish_time_s = time.time()
        self._last_published_upper_command = command
        return True

    def _should_publish_min_snap_target(self, command: FaUpperPositionCommand) -> bool:
        last = self._last_min_snap_target_command
        if last is None:
            return True
        current = np.asarray(command.values[:14], dtype=np.float64)
        previous = np.asarray(last.values[:14], dtype=np.float64)
        max_delta = float(np.max(np.abs(current - previous)))
        if max_delta < self.config.min_snap_target_epsilon_rad:
            return False
        min_period_s = 1.0 / self.config.min_snap_target_publish_hz
        if time.time() - self._last_min_snap_target_publish_time_s < min_period_s:
            return False
        return True

    def _current_joint_snapshot(self) -> Optional[FaJointStateSnapshot]:
        if self.control_backend == "mujoco":
            if self._real_joint_state_fresh():
                return self._ros2.joint_cache.snapshot
            return self._dry_joint_cache.snapshot
        if self._real_joint_state_fresh():
            return self._ros2.joint_cache.snapshot
        if self.control_backend == "real_with_mujoco" and self.config.allow_mujoco_mirror_without_joint_state:
            return self._dry_joint_cache.snapshot
        return self._ros2.joint_cache.snapshot

    def _real_joint_state_fresh(self) -> bool:
        return self._ros2.joint_cache.is_fresh()

    def _warn_safety(self, key: str, message: str) -> None:
        now = time.time()
        if now - self._last_safety_log_time.get(key, 0.0) < 1.0:
            return
        self._last_safety_log_time[key] = now
        logger.warning(message)

    def close(self):
        executor = getattr(self, "_ik_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self._ros2.close()
        for subscriber in getattr(self, "_subscribers", []):
            try:
                subscriber.close()
            except Exception:
                pass
        cleanup_zmq_resources()


__all__ = [
    "FA_UPPER_COMMAND_TOPIC",
    "FaRos2Topics",
    "FaRealControl",
    "FaRealControlConfig",
    "FaRos2Bridge",
]
