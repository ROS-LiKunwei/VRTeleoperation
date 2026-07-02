""" sysmo32 专用 MuJoCo 镜像仿真层.

    与“MuJoCoSysmoSimulator”不同,此类不订阅“CartesianTarget”。
    它使用为真实手臂接口生成的精确18字段命令,并且仅应用左/右臂关节位置。
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.components import Component
from beavr.teleop.components.interface.robots.sysmo32_command import (
    SYSMO32_COMMAND_LENGTH,
    SYSMO32_HAND_ACTION_GRASP,
    SYSMO32_HAND_ACTION_RELEASE,
    SYSMO32_LEFT_JOINT_NAMES,
    SYSMO32_RIGHT_JOINT_NAMES,
    Sysmo32ArmCommand,
    Sysmo32HandAction,
)
from beavr.teleop.components.interface.robots.sysmo32_kinematics import Sysmo32MujocoKinematics
from beavr.teleop.components.interface.robots.sysmo32_real_control import (
    SYSMO32_ARM_COMMAND_TOPIC,
    SYSMO32_LEFT_HAND_ACTION_TOPIC,
    SYSMO32_RIGHT_HAND_ACTION_TOPIC,
)
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class Sysmo32MujocoCommandMirror(Component):
    """Apply real-format SYSMO-32 arm commands to MuJoCo."""

    def __init__(
        self,
        host: str,
        arm_command_port: int,
        hand_action_port: int,
        urdf_path: str,
        control_dt: float = 0.01,
        render: bool = True,
        load_model: bool = True,
        print_hand_action_only: bool = True,
        arm_command_source: str = "zmq",
        ros_arm_command_topic: str = "/sysmo_left_arm_controller/commands",
        publish_joint_states: bool = False,
        joint_state_topic: str = "/joint_states",
        joint_state_publish_hz: float = 50.0,
        subscribe_min_snap_target: bool = False,
        min_snap_target_topic: str = "/min_snap/target",
        arm_command_interpolation_steps: int = 5,
        interpolation_profile: str = "quintic",
        expected_command_length: int = SYSMO32_COMMAND_LENGTH,
        joint_state_joint_names: Sequence[str] | None = None,
        kinematics_type: str = "sysmo32",
    ):
        self.notify_component_start("sysmo32_mujoco_command_mirror")
        if arm_command_source not in ("none", "zmq", "ros2", "both"):
            raise ValueError("arm_command_source must be one of: none, zmq, ros2, both")
        if kinematics_type not in ("sysmo32", "fa"):
            raise ValueError("kinematics_type must be one of: sysmo32, fa")
        if subscribe_min_snap_target and kinematics_type != "fa":
            raise ValueError("subscribe_min_snap_target is currently supported only for FA mirror")
        interpolation_profile = str(interpolation_profile or "quintic").strip().lower()
        if interpolation_profile in ("septic", "seventh_order", "minimum_snap"):
            interpolation_profile = "min_snap"
        if interpolation_profile not in ("linear", "quintic", "min_snap"):
            raise ValueError("interpolation_profile must be one of: linear, quintic, min_snap")
        self.host = host
        self.control_dt = control_dt
        self.render = render
        self.print_hand_action_only = print_hand_action_only
        self.arm_command_source = arm_command_source
        self.ros_arm_command_topic = ros_arm_command_topic
        self.publish_joint_states = publish_joint_states
        self.joint_state_topic = joint_state_topic
        self.joint_state_publish_hz = max(0.1, float(joint_state_publish_hz))
        self.subscribe_min_snap_target = bool(subscribe_min_snap_target)
        self.min_snap_target_topic = min_snap_target_topic
        self.arm_command_interpolation_steps = max(1, int(arm_command_interpolation_steps))
        self.interpolation_profile = interpolation_profile
        self.expected_command_length = int(expected_command_length)
        self.joint_state_joint_names = tuple(joint_state_joint_names or (SYSMO32_LEFT_JOINT_NAMES + SYSMO32_RIGHT_JOINT_NAMES))
        self.kinematics_type = kinematics_type
        self.robot_label = "FA" if kinematics_type == "fa" else "SYSMO-32"
        self._arm_command_topic = SYSMO32_ARM_COMMAND_TOPIC
        self._arm_command_type = Sysmo32ArmCommand
        if self.kinematics_type == "fa":
            from beavr.teleop.components.interface.robots.fa_command_builder import FaUpperPositionCommand

            self._arm_command_topic = "fa_upper_position_command"
            self._arm_command_type = FaUpperPositionCommand
        self._last_no_hand_action_log_time = 0.0
        self._last_arm_pose_log_time = 0.0
        self._last_joint_state_publish_time = 0.0
        self._arm_joint_ids = []
        self._arm_qpos_addrs = []
        self._arm_dof_addrs = []
        self._hold_joint_positions: Optional[np.ndarray] = None
        self._trajectory_start_positions: Optional[np.ndarray] = None
        self._trajectory_target_positions: Optional[np.ndarray] = None
        self._trajectory_start_time_s: Optional[float] = None
        self._trajectory_duration_s: Optional[float] = None
        self._rclpy = None
        self._ros_node = None
        self._joint_state_msg_type = None
        self._joint_state_pub = None
        self._owns_rclpy_context = False
        # 订阅真实机械臂命令字段
        self._arm_command_subscriber = ZMQSubscriber(
            host,
            arm_command_port,
            self._arm_command_topic,
            message_type=self._arm_command_type,
        )
        # 订阅真实左手动作字段
        self._left_hand_action_subscriber = ZMQSubscriber(
            host,
            hand_action_port,
            SYSMO32_LEFT_HAND_ACTION_TOPIC,
            message_type=Sysmo32HandAction,
        )
        # 订阅真实右手动作字段
        self._right_hand_action_subscriber = ZMQSubscriber(
            host,
            hand_action_port,
            SYSMO32_RIGHT_HAND_ACTION_TOPIC,
            message_type=Sysmo32HandAction,
        )
        self._subscribers = [
            self._arm_command_subscriber,
            self._left_hand_action_subscriber,
            self._right_hand_action_subscriber,
        ]
        if self.arm_command_source in ("ros2", "both") or self.publish_joint_states or self.subscribe_min_snap_target:
            self._init_ros2_interfaces()

        self._kinematics: Optional[object] = None
        if load_model:
            self._kinematics = self._make_kinematics(urdf_path)
            if not self._kinematics.available:
                logger.warning(
                    "%s MuJoCo command mirror cannot load model; "
                    "falling back to command validation/logging only",
                    self.robot_label,
                )
            else:
                self._configure_arm_hold_state()

    def _make_kinematics(self, urdf_path: str):
        if self.kinematics_type == "fa":
            from beavr.teleop.components.interface.robots.fa_mujoco_kinematics import (
                FaKinematicsConfig,
                FaMujocoKinematics,
            )

            return FaMujocoKinematics(FaKinematicsConfig(model_path=urdf_path, require_endeff=False))
        return Sysmo32MujocoKinematics(urdf_path)

    def _init_ros2_interfaces(self) -> None:
        try:
            import rclpy
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy_context = True
            self._ros_node = rclpy.create_node("sysmo32_mujoco_command_mirror")
            if self.arm_command_source in ("ros2", "both"):
                self._ros_node.create_subscription(
                    Float64MultiArray,
                    self.ros_arm_command_topic,
                    self._on_ros_arm_command,
                    10,
                )
                logger.info(
                    "SYSMO-32 MuJoCo command mirror subscribed to ROS2 arm command topic %s",
                    self.ros_arm_command_topic,
                )
            if self.subscribe_min_snap_target:
                from min_snap.msg import MinSnapTarget

                self._ros_node.create_subscription(
                    MinSnapTarget,
                    self.min_snap_target_topic,
                    self._on_min_snap_target,
                    10,
                )
                logger.info(
                    "%s MuJoCo command mirror subscribed to ROS2 min-snap target topic %s",
                    self.robot_label,
                    self.min_snap_target_topic,
                )
            if self.publish_joint_states:
                self._joint_state_msg_type = JointState
                self._joint_state_pub = self._ros_node.create_publisher(JointState, self.joint_state_topic, 10)
                logger.info(
                    "SYSMO-32 MuJoCo command mirror publishing simulated joint feedback on %s",
                    self.joint_state_topic,
                )
        except Exception as exc:
            raise RuntimeError(
                f"SYSMO-32 MuJoCo command mirror cannot initialize ROS2 interfaces: {exc}"
            ) from exc

    def _spin_ros2_once(self) -> None:
        if self._rclpy is not None and self._ros_node is not None:
            self._rclpy.spin_once(self._ros_node, timeout_sec=0.0)

    def _on_ros_arm_command(self, msg) -> None:
        try:
            command = self._arm_command_type(timestamp_s=time.time(), values=tuple(float(v) for v in msg.data))
        except Exception as exc:
            logger.warning("[MuJoCo][ROS2 ArmCommand] invalid command: %s", exc)
            return
        self.apply_arm_command(command)

    def _on_min_snap_target(self, msg) -> None:
        if self.kinematics_type != "fa":
            return
        try:
            left = np.asarray(msg.left_arm_target_rad, dtype=np.float64)
            right = np.asarray(msg.right_arm_target_rad, dtype=np.float64)
            if left.shape != (7,) or right.shape != (7,):
                raise ValueError(f"expected 7+7 arm joints, got {left.shape} and {right.shape}")
            values = tuple(float(v) for v in np.concatenate([left, right, np.zeros(2, dtype=np.float64)]))
            command = self._arm_command_type(timestamp_s=time.time(), values=values)
            duration_s = float(getattr(msg, "expected_duration_s", 0.0) or 0.0)
        except Exception as exc:
            logger.warning("[MuJoCo][MinSnapTarget] invalid target: %s", exc)
            return
        self.apply_arm_command(command, duration_s=duration_s if duration_s > 0.0 else None)

    def stream(self):
        if self._kinematics is None or not self._kinematics.available:
            logger.info("%s MuJoCo command mirror running without model; logging only", self.robot_label)
            while True:
                self._receive_once()
                time.sleep(self.control_dt)

        import mujoco
        import mujoco.viewer

        logger.info("%s MuJoCo command mirror started", self.robot_label)
        if self.render:
            with mujoco.viewer.launch_passive(self._kinematics.model, self._kinematics.data) as viewer:
                while viewer.is_running():
                    # 收一次 arm command 和 hand action，并应用到mujoco
                    self._receive_once()
                    self._forward_kinematic_mirror(mujoco)
                    viewer.sync()
                    time.sleep(self.control_dt)
            return

        while True:
            self._receive_once()
            self._forward_kinematic_mirror(mujoco)
            time.sleep(self.control_dt)

    def _configure_arm_hold_state(self) -> None:
        """Cache arm joint addresses and hold the initial model pose until commands arrive."""

        if self._kinematics is None or not self._kinematics.available:
            return
        self._arm_joint_ids = self._kinematics.left_joint_ids + self._kinematics.right_joint_ids
        self._arm_qpos_addrs = [self._kinematics.model.jnt_qposadr[joint_id] for joint_id in self._arm_joint_ids]
        self._arm_dof_addrs = [self._kinematics.model.jnt_dofadr[joint_id] for joint_id in self._arm_joint_ids]
        self._hold_joint_positions = np.asarray(
            [self._kinematics.data.qpos[addr] for addr in self._arm_qpos_addrs],
            dtype=np.float64,
        )
        self._apply_arm_hold()

    def _forward_kinematic_mirror(self, mujoco_module) -> None:
        """Forward the model as a kinematic mirror instead of stepping free dynamics."""

        self._update_interpolated_hold(time.time())
        self._apply_arm_hold()
        mujoco_module.mj_forward(self._kinematics.model, self._kinematics.data)
        self._publish_joint_state_if_due()

    def _apply_arm_hold(self) -> None:
        if self._kinematics is None or self._hold_joint_positions is None:
            return
        for idx, qpos_addr in enumerate(self._arm_qpos_addrs):
            self._kinematics.data.qpos[qpos_addr] = self._hold_joint_positions[idx]
        for dof_addr in self._arm_dof_addrs:
            self._kinematics.data.qvel[dof_addr] = 0.0

    def _receive_once(self) -> None:
        if self.arm_command_source in ("ros2", "both") or self.subscribe_min_snap_target or self.publish_joint_states:
            self._spin_ros2_once()

        if self.arm_command_source in ("zmq", "both"):
            command = self._arm_command_subscriber.recv_keypoints()
            if command is not None:
                self.apply_arm_command(command)

        left_action = self._left_hand_action_subscriber.recv_keypoints()
        if left_action is not None:
            self.on_left_hand_action(left_action.action_id)

        right_action = self._right_hand_action_subscriber.recv_keypoints()
        if right_action is not None:
            self.on_right_hand_action(right_action.action_id)

    def apply_arm_command(self, command: Sysmo32ArmCommand, duration_s: Optional[float] = None) -> None:
        values = np.asarray(command.values, dtype=np.float64)
        expected_command_length = getattr(self, "expected_command_length", SYSMO32_COMMAND_LENGTH)
        if values.shape != (expected_command_length,) or not np.all(np.isfinite(values)):
            logger.warning("[MuJoCo][ArmCommand] invalid command shape/value: %s", values)
            return
        if self._kinematics is None or not self._kinematics.available:
            logger.debug("[MuJoCo][ArmCommand] received valid command without model: %s", values)
            return
        if values.size < len(self._arm_joint_ids):
            logger.warning(
                "[MuJoCo][ArmCommand] command length %d is shorter than configured arm joints %d",
                values.size,
                len(self._arm_joint_ids),
            )
            return

        now_s = time.time()
        self._update_interpolated_hold(now_s)
        target = []
        for idx, joint_id in enumerate(self._arm_joint_ids):
            low, high = self._kinematics.model.jnt_range[joint_id]
            target.append(float(np.clip(values[idx], low, high)))
        if self._hold_joint_positions is None:
            self._hold_joint_positions = np.asarray(
                [self._kinematics.data.qpos[addr] for addr in self._arm_qpos_addrs],
                dtype=np.float64,
            )
        self._trajectory_start_positions = self._hold_joint_positions.copy()
        self._trajectory_target_positions = np.asarray(target, dtype=np.float64)
        self._trajectory_start_time_s = now_s
        if duration_s is not None and np.isfinite(duration_s) and duration_s > 0.0:
            self._trajectory_duration_s = max(self.control_dt, float(duration_s))
        else:
            self._trajectory_duration_s = None
        self._update_interpolated_hold(now_s)
        self._log_applied_arm_command(values)

    def _update_interpolated_hold(self, now_s: float) -> None:
        if (
            self._trajectory_start_positions is None
            or self._trajectory_target_positions is None
            or self._trajectory_start_time_s is None
        ):
            return
        duration_s = self._trajectory_duration_s
        if duration_s is None:
            duration_s = max(self.control_dt, self.control_dt * self.arm_command_interpolation_steps)
        progress = (now_s - self._trajectory_start_time_s) / duration_s
        blend = _trajectory_blend(progress, getattr(self, "interpolation_profile", "quintic"))
        self._hold_joint_positions = self._trajectory_start_positions + (
            self._trajectory_target_positions - self._trajectory_start_positions
        ) * blend
        if progress >= 1.0:
            self._hold_joint_positions = self._trajectory_target_positions.copy()
            self._trajectory_start_positions = None
            self._trajectory_target_positions = None
            self._trajectory_start_time_s = None
            self._trajectory_duration_s = None

    def _publish_joint_state_if_due(self) -> None:
        if (
            not self.publish_joint_states
            or self._joint_state_pub is None
            or self._joint_state_msg_type is None
            or self._ros_node is None
            or self._kinematics is None
            or not self._kinematics.available
        ):
            return
        now_s = time.time()
        if now_s - self._last_joint_state_publish_time < 1.0 / self.joint_state_publish_hz:
            return
        self._last_joint_state_publish_time = now_s

        msg = self._joint_state_msg_type()
        msg.header.stamp = self._ros_node.get_clock().now().to_msg()
        msg.name = list(self.joint_state_joint_names)
        msg.position = [
            float(self._kinematics.data.qpos[qpos_addr]) for qpos_addr in self._arm_qpos_addrs
        ][: len(msg.name)]
        self._joint_state_pub.publish(msg)

    def _log_applied_arm_command(self, values: np.ndarray) -> None:
        now = time.time()
        if now - self._last_arm_pose_log_time >= 0.5:
            self._last_arm_pose_log_time = now
            left_site_id = getattr(self._kinematics, "left_site_id", -1)
            right_site_id = getattr(self._kinematics, "right_site_id", -1)
            if left_site_id >= 0 and right_site_id >= 0:
                left_site = self._kinematics.data.site_xpos[left_site_id].copy()
                right_site = self._kinematics.data.site_xpos[right_site_id].copy()
                logger.debug(
                    "[MuJoCo][ArmCommand] applied left=%s right=%s left_site=%s right_site=%s",
                    values[: len(self._kinematics.left_joint_ids)],
                    values[len(self._kinematics.left_joint_ids) : len(self._arm_joint_ids)],
                    left_site,
                    right_site,
                )
                return
            logger.debug(
                "[MuJoCo][ArmCommand] applied left=%s right=%s",
                values[: len(self._kinematics.left_joint_ids)],
                values[len(self._kinematics.left_joint_ids) : len(self._arm_joint_ids)],
            )
            return
        logger.debug(
            "[MuJoCo][ArmCommand] applied left=%s right=%s",
            values[: len(self._kinematics.left_joint_ids)],
            values[len(self._kinematics.left_joint_ids) : len(self._arm_joint_ids)],
        )

    def on_left_hand_action(self, action_id: int) -> None:
        self._log_hand_action(robots.LEFT, action_id)

    def on_right_hand_action(self, action_id: int) -> None:
        self._log_hand_action(robots.RIGHT, action_id)

    def _log_hand_action(self, hand_side: str, action_id: int) -> None:
        if action_id not in (SYSMO32_HAND_ACTION_RELEASE, SYSMO32_HAND_ACTION_GRASP):
            logger.warning("[MuJoCo][HandAction] %s invalid action=%s, print only, no execution", hand_side, action_id)
            return
        logger.info(
            "[MuJoCo][HandAction] %s action=%d, print only, no execution",
            hand_side,
            action_id,
        )

    def cleanup(self) -> None:
        for subscriber in getattr(self, "_subscribers", []):
            subscriber.stop()
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        if self._owns_rclpy_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        cleanup_zmq_resources()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = ["Sysmo32MujocoCommandMirror"]


def _quintic_blend(progress: float) -> float:
    tau = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def _min_snap_blend(progress: float) -> float:
    tau = float(np.clip(progress, 0.0, 1.0))
    return 35.0 * tau**4 - 84.0 * tau**5 + 70.0 * tau**6 - 20.0 * tau**7


def _trajectory_blend(progress: float, interpolation_profile: str = "quintic") -> float:
    if interpolation_profile == "linear":
        return float(np.clip(progress, 0.0, 1.0))
    if interpolation_profile == "min_snap":
        return _min_snap_blend(progress)
    return _quintic_blend(progress)
