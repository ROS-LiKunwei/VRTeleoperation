"""FA native upper-body real-control adapter."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.interface.robots.arm_command_publisher import FaNativeCommandPublisher
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
from beavr.teleop.components.interface.robots.fa_trajectory import (
    FaArmTrajectoryConfig,
    FaArmTrajectorySmoother,
    FaJerkLimitedServoConfig,
    FaJerkLimitedServoSmoother,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import ports, robots

logger = logging.getLogger(__name__)

FA_UPPER_COMMAND_TOPIC = "fa_upper_position_command"


@dataclass
class FaRos2Topics:
    joint_state_topic: str = "/joint_states"
    upper_position_command_topic: str = "/upper_position_controller/commands"
    upper_position_command_queue_size: int = 60
    joint_state_timeout_s: float = 1.0


@dataclass
class FaRealControlConfig:
    control_backend: str = "mujoco"
    ros2: FaRos2Topics = field(default_factory=FaRos2Topics)
    upper: FaUpperPositionSafetyConfig = field(default_factory=FaUpperPositionSafetyConfig)
    kinematics: FaKinematicsConfig = field(default_factory=FaKinematicsConfig)
    ik: FaArmIkConfig = field(
        default_factory=lambda: FaArmIkConfig(
            urdf_file="/home/likunwei/dataCollection/beavr-bot/robots/fa_description/urdf/fa_robot.urdf",
            srdf_file="/home/likunwei/humanoid_ws/src/fa_moveit2_config/config/fa_robot.srdf",
        )
    )
    state_publish_fps: float = 30.0
    safety_hold_arm_on_pause: bool = True
    pause_hold_heartbeat_hz: float = 20.0
    allow_mujoco_mirror_without_joint_state: bool = True
    publish_upper_command_topic_in_mujoco: bool = False
    arm_trajectory_smoother: str = "min_snap"
    arm_trajectory_segment_time_s: float = 0.18
    arm_trajectory_min_duration_s: float = 0.06
    arm_trajectory_replan_threshold_rad: float = 0.0005
    arm_trajectory_max_acceleration_rad_s2: float = 12.0
    arm_servo_max_velocity_rad_s: float = 3.0
    arm_servo_max_acceleration_rad_s2: float = 10.0
    arm_servo_max_jerk_rad_s3: float = 120.0
    arm_servo_omega: float = 35.0
    arm_servo_damping_ratio: float = 1.0
    arm_servo_target_deadband_rad: float = 0.0005
    arm_servo_max_dt_s: float = 0.05
    max_ik_solution_jump_rad: float = 0.5

    def __post_init__(self):
        self.arm_trajectory_smoother = str(self.arm_trajectory_smoother or "none").strip().lower()
        if self.arm_trajectory_smoother in ("jerk", "servo", "online_servo"):
            self.arm_trajectory_smoother = "jerk_limited_servo"
        if self.arm_trajectory_smoother not in ("none", "min_snap", "jerk_limited_servo"):
            raise ValueError("arm_trajectory_smoother must be one of: none, min_snap, jerk_limited_servo")
        self.max_ik_solution_jump_rad = max(0.0, float(self.max_ik_solution_jump_rad))


class FaRos2Bridge:
    """ROS2 bridge for FA joint states and native 16D command publishing."""

    def __init__(self, topics: FaRos2Topics, require_ros: bool, node_name: str = "fa_real_control"):
        self.topics = topics
        self.require_ros = require_ros
        self.available = False
        self.joint_cache = FaJointStateCache(topics.joint_state_timeout_s)
        self._rclpy = None
        self._node = None
        self._upper_publisher = None
        if require_ros:
            self._init_ros2(node_name)

    def _init_ros2(self, node_name: str) -> None:
        try:
            import rclpy
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node(node_name)
            self._upper_publisher = FaNativeCommandPublisher(
                ros_node=self._node,
                msg_type=Float64MultiArray,
                topic=self.topics.upper_position_command_topic,
                queue_size=self.topics.upper_position_command_queue_size,
            )
            self._node.create_subscription(JointState, self.topics.joint_state_topic, self._on_joint_state, 10)
            self.available = True
            logger.info(
                "FA ROS2 bridge connected: joint_state=%s command=%s",
                self.topics.joint_state_topic,
                self.topics.upper_position_command_topic,
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

    def publish_upper_command(self, command: FaUpperPositionCommand) -> bool:
        if not self.available or self._upper_publisher is None:
            return False
        return self._upper_publisher.publish(command)

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
        upper_command_mirror_port: int = ports.SYSMO32_ARM_COMMAND_MIRROR_PORT,
        urdf_path: str = "/home/likunwei/dataCollection/beavr-bot/robots/fa_description/urdf/fa_robot.urdf",
        config: Optional[FaRealControlConfig] = None,
        ik_client: Optional[FaArmIkClientBase] = None,
        **_,
    ):
        self.notify_component_start("fa_real_control")
        self.host = host
        self.control_backend = control_backend
        self._validate_backend()
        self.config = config or FaRealControlConfig(control_backend=control_backend)
        self.config.control_backend = control_backend
        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._upper_command_mirror_port = upper_command_mirror_port
        self._right_state_publish_port = right_state_publish_port
        self._left_state_publish_port = left_state_publish_port
        self._right_endeff_publish_port = right_endeff_publish_port or right_state_publish_port
        self._left_endeff_publish_port = left_endeff_publish_port or left_state_publish_port
        self._publisher_manager.register_topic(self.host, self._upper_command_mirror_port, FA_UPPER_COMMAND_TOPIC)
        self._publisher_manager.register_topic(self.host, self._right_state_publish_port, "fa_right")
        self._publisher_manager.register_topic(self.host, self._left_state_publish_port, "fa_left")

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

        require_ros = self.control_backend in ("real", "real_with_mujoco") or (
            self.control_backend == "mujoco" and self.config.publish_upper_command_topic_in_mujoco
        )
        self._ros2 = FaRos2Bridge(self.config.ros2, require_ros=require_ros)
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
        self._builder = FaUpperPositionCommandBuilder(self.config.upper)
        self._limiter = FaCommandLimiter(self.config.upper)
        self._left_arm_smoother = self._make_arm_smoother(robots.LEFT)
        self._right_arm_smoother = self._make_arm_smoother(robots.RIGHT)

        self._teleop_active = True
        self._needs_reset = True
        self._real_reset_ready = self.control_backend == "mujoco"
        self._latest_targets: Dict[str, Optional[CartesianTarget]] = {robots.LEFT: None, robots.RIGHT: None}
        self._last_safe_arm_targets: Dict[str, Optional[np.ndarray]] = {robots.LEFT: None, robots.RIGHT: None}
        self._pause_hold_command: Optional[FaUpperPositionCommand] = None
        self._last_pause_hold_publish_time = 0.0
        self._next_state_publish_time_s = 0.0
        self._last_published_upper_command: Optional[FaUpperPositionCommand] = None
        self._last_safety_log_time: Dict[str, float] = {}

    def _make_default_ik_client(self) -> FaArmIkClientBase:
        return FaPybindIkClient(self.config.ik)

    def _validate_backend(self) -> None:
        if self.control_backend not in ("real", "mujoco", "real_with_mujoco"):
            raise ValueError(f"control_backend must be one of: real, mujoco, real_with_mujoco; got {self.control_backend}")

    def _make_arm_smoother(self, hand_side: str):
        if self.config.arm_trajectory_smoother == "min_snap":
            return FaArmTrajectorySmoother(
                self._make_arm_trajectory_config(hand_side, enabled=True),
                name=f"fa_{hand_side}",
            )
        enabled = self.config.arm_trajectory_smoother == "jerk_limited_servo"
        offset = 0 if hand_side == robots.LEFT else 7
        return FaJerkLimitedServoSmoother(
            FaJerkLimitedServoConfig(
                enabled=enabled,
                max_joint_velocity_rad_s=tuple([self.config.arm_servo_max_velocity_rad_s] * 7),
                max_joint_acceleration_rad_s2=tuple([self.config.arm_servo_max_acceleration_rad_s2] * 7),
                max_joint_jerk_rad_s3=tuple([self.config.arm_servo_max_jerk_rad_s3] * 7),
                omega=self.config.arm_servo_omega,
                damping_ratio=self.config.arm_servo_damping_ratio,
                target_deadband_rad=self.config.arm_servo_target_deadband_rad,
                max_dt_s=self.config.arm_servo_max_dt_s,
                joint_lower_limits_rad=self.config.upper.joint_lower_limits_rad[offset : offset + 7],
                joint_upper_limits_rad=self.config.upper.joint_upper_limits_rad[offset : offset + 7],
            ),
            name=f"fa_{hand_side}",
        )

    def _make_arm_trajectory_config(self, hand_side: str, enabled: bool) -> FaArmTrajectoryConfig:
        offset = 0 if hand_side == robots.LEFT else 7
        return FaArmTrajectoryConfig(
            joint_count=7,
            enabled=enabled,
            segment_time_s=self.config.arm_trajectory_segment_time_s,
            min_duration_s=self.config.arm_trajectory_min_duration_s,
            replan_threshold_rad=self.config.arm_trajectory_replan_threshold_rad,
            max_joint_velocity_rad_s=tuple([self.config.arm_servo_max_velocity_rad_s] * 7),
            max_joint_acceleration_rad_s2=tuple([self.config.arm_trajectory_max_acceleration_rad_s2] * 7),
            joint_lower_limits_rad=self.config.upper.joint_lower_limits_rad[offset : offset + 7],
            joint_upper_limits_rad=self.config.upper.joint_upper_limits_rad[offset : offset + 7],
        )

    def stream(self):
        logger.info("Starting FA real-control backend=%s", self.control_backend)
        while True:
            start = time.time()
            self._ros2.spin_once()
            self._handle_session_command()
            self._publish_pause_hold_if_needed()
            self._handle_reset_requests()
            self._receive_cartesian_targets()
            self._publish_lerobot_joint_states()
            self._publish_upper_command_if_safe()
            time.sleep(max(0.0, (1.0 / robots.VR_FREQ) - (time.time() - start)))

    def _handle_session_command(self) -> None:
        msg = self._pause_subscriber.recv_keypoints()
        if msg is None:
            return
        if msg.command == robots.PAUSE:
            if self._teleop_active:
                self._enter_pause("pause command")
        elif msg.command == robots.RESUME:
            if self._teleop_active:
                return
            self._teleop_active = True
            self._needs_reset = True
            self._real_reset_ready = self.control_backend == "mujoco"
            self._latest_targets = {robots.LEFT: None, robots.RIGHT: None}
            self._pause_hold_command = None

    def _enter_pause(self, reason: str) -> None:
        self._teleop_active = False
        self._needs_reset = True
        self._real_reset_ready = self.control_backend == "mujoco"
        self._latest_targets = {robots.LEFT: None, robots.RIGHT: None}
        snapshot = self._pause_hold_snapshot()
        if snapshot is not None:
            self._limiter.reset(snapshot.upper_joints)
            self._left_arm_smoother.reset(snapshot.left_arm)
            self._right_arm_smoother.reset(snapshot.right_arm)
            if self.config.safety_hold_arm_on_pause:
                self._pause_hold_command = self._builder.build(snapshot.left_arm, snapshot.right_arm, snapshot.neck)
                self._publish_pause_hold_if_needed(force=True, reason=reason)

    def _pause_hold_snapshot(self) -> Optional[FaJointStateSnapshot]:
        if self.control_backend in ("real", "real_with_mujoco") and not self._real_joint_state_fresh():
            self._warn_safety("pause_joint_state_stale", "pause hold skipped: /joint_states is stale")
            return None
        return self._current_joint_snapshot()

    def _publish_pause_hold_if_needed(self, force: bool = False, reason: str = "pause hold") -> None:
        if self._teleop_active or not self.config.safety_hold_arm_on_pause or self._pause_hold_command is None:
            return
        now = time.time()
        period = 1.0 / max(0.1, float(self.config.pause_hold_heartbeat_hz))
        if not force and now - self._last_pause_hold_publish_time < period:
            return
        command = FaUpperPositionCommand(now, self._pause_hold_command.values)
        if self._publish_upper_command_outputs(command, require_real_reset=False, allow_stale_real_hold=True):
            self._pause_hold_command = command
            self._last_pause_hold_publish_time = now

    def _handle_reset_requests(self) -> None:
        for hand_side, subscriber, publish_port in (
            (robots.RIGHT, self._right_reset_subscriber, self._right_endeff_publish_port),
            (robots.LEFT, self._left_reset_subscriber, self._left_endeff_publish_port),
        ):
            if subscriber.recv_keypoints() is None:
                continue
            self._publish_current_endeff_homo(hand_side, publish_port)

    def _publish_current_endeff_homo(self, hand_side: str, publish_port: int) -> None:
        snapshot = self._current_joint_snapshot()
        real_fresh = self._real_joint_state_fresh()
        if snapshot is None or (self.control_backend == "real" and not real_fresh):
            self._warn_safety("reset_joint_state_stale", f"reset refused for {hand_side}: joint state stale")
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
        self._real_reset_ready = real_fresh or self.control_backend == "mujoco"
        self._limiter.reset(snapshot.upper_joints)
        self._left_arm_smoother.reset(snapshot.left_arm)
        self._right_arm_smoother.reset(snapshot.right_arm)

    def _receive_cartesian_targets(self) -> None:
        for hand_side, subscriber in ((robots.LEFT, self._left_target_subscriber), (robots.RIGHT, self._right_target_subscriber)):
            msg = subscriber.recv_keypoints()
            if msg is None:
                continue
            if msg.hand_side != hand_side:
                self._warn_safety(f"{hand_side}_wrong_target_side", f"wrong target side: {msg.hand_side}")
                continue
            self._latest_targets[hand_side] = msg

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
        state = {
            "joint_states": {"joint_position": [float(v) for v in joints], "timestamp": now},
            "joint_angles_rad": [float(v) for v in joints],
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
        if self.control_backend == "real" and not self._real_joint_state_fresh():
            self._warn_safety("joint_state_stale", "FA command rejected: /joint_states stale")
            return
        if self.control_backend == "real_with_mujoco" and not self._real_joint_state_fresh():
            self._warn_safety("joint_state_stale", "FA command rejected: /joint_states stale")
            return
        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            self._warn_safety("joint_state_missing", "FA command rejected: missing joint state")
            return
        desired_left = np.asarray(snapshot.left_arm, dtype=np.float64)
        desired_right = np.asarray(snapshot.right_arm, dtype=np.float64)
        any_target = False
        for hand_side in (robots.LEFT, robots.RIGHT):
            target = self._latest_targets[hand_side]
            if target is None:
                continue
            current_arm = snapshot.left_arm if hand_side == robots.LEFT else snapshot.right_arm
            ik = self._ik_client.solve(hand_side, target, current_arm)
            if not ik.success:
                self._warn_safety(f"{hand_side}_ik_fail", f"{hand_side} IK failed: {ik.message}")
                last = self._last_safe_arm_targets[hand_side]
                if last is None:
                    continue
                solved = last
            else:
                solved = np.asarray(ik.q_target, dtype=np.float64)
                reference = self._last_safe_arm_targets[hand_side]
                if reference is None:
                    reference = np.asarray(current_arm, dtype=np.float64)
                max_jump = float(np.max(np.abs(solved - reference)))
                if self.config.max_ik_solution_jump_rad > 0.0 and max_jump > self.config.max_ik_solution_jump_rad:
                    self._warn_safety(
                        f"{hand_side}_ik_solution_jump",
                        f"{hand_side} IK solution jump held: {max_jump:.3f} rad",
                    )
                    last = self._last_safe_arm_targets[hand_side]
                    if last is None:
                        continue
                    solved = last
                else:
                    self._last_safe_arm_targets[hand_side] = solved.copy()
            if hand_side == robots.LEFT:
                desired_left = self._left_arm_smoother.sample(solved, snapshot.left_arm, now_s=time.time())
            else:
                desired_right = self._right_arm_smoother.sample(solved, snapshot.right_arm, now_s=time.time())
            any_target = True
        if not any_target:
            return
        try:
            command = self._builder.build(desired_left, desired_right, snapshot.neck, timestamp_s=time.time())
            limited, reason = self._limiter.limit(command, now_s=command.timestamp_s)
        except Exception as exc:
            self._warn_safety("command_build_fail", f"FA command build failed: {exc}")
            return
        if limited is None:
            self._warn_safety("command_limited_reject", f"FA command rejected: {reason}")
            return
        self._publish_upper_command_outputs(limited, require_real_reset=True)

    def _publish_upper_command_outputs(
        self,
        command: FaUpperPositionCommand,
        require_real_reset: bool,
        allow_stale_real_hold: bool = False,
    ) -> bool:
        if len(command.values) != FA_UPPER_COMMAND_LENGTH:
            raise ValueError("Refusing to publish non-16D FA command")
        if self.control_backend == "mujoco" and self.config.publish_upper_command_topic_in_mujoco:
            if not self._ros2.publish_upper_command(command):
                self._warn_safety("ros_upper_unavailable", "ROS2 FA upper publisher unavailable")
                return False
        if self.control_backend in ("real", "real_with_mujoco"):
            state_ready = self._real_joint_state_fresh() or allow_stale_real_hold
            reset_ready = self._real_reset_ready or not require_real_reset
            if not state_ready or not reset_ready:
                self._warn_safety("real_reset_required", "FA real command held until fresh /joint_states and reset")
                return False
            if not self._ros2.publish_upper_command(command):
                self._warn_safety("ros_upper_unavailable", "ROS2 FA upper publisher unavailable")
                return False
        if self.control_backend in ("mujoco", "real_with_mujoco"):
            self._publisher_manager.publish(self.host, self._upper_command_mirror_port, FA_UPPER_COMMAND_TOPIC, command)
            self._dry_joint_cache.update(command.left_arm, command.right_arm, command.neck, now_s=command.timestamp_s)
        self._last_published_upper_command = command
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
