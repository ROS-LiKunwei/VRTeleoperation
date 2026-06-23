"""SYSMO-32 real robot control adapter.

This component is deliberately bimanual and sysmo32-specific because the real
arm interface accepts one 18-field command containing both arms.

SYSMO-32 机器人的真实控制适配器，负责将笛卡尔空间的目标指令转换为关节角度命令，
并通过 ROS2 发布给真实机器人
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence

import numpy as np

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import InputFrame, SessionCommand
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.interface.robots.arm_command_publisher import (
    MinSnapTargetPublisher,
)
from beavr.teleop.components.interface.robots.sysmo32_command import (
    SYSMO32_HAND_ACTION_GRASP,
    SYSMO32_HAND_ACTION_RELEASE,
    Sysmo32ArmCommand,
    Sysmo32ArmSafetyConfig,
    Sysmo32CommandBuilder,
    Sysmo32CommandLimiter,
    Sysmo32HandAction,
    Sysmo32HandGestureMapper,
    Sysmo32JointStateCache,
    Sysmo32JointStateSnapshot,
    quaternion_angle_delta_rad,
)
from beavr.teleop.components.interface.robots.sysmo32_kinematics import Sysmo32MujocoKinematics
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import ports, robots

logger = logging.getLogger(__name__)

SYSMO32_ARM_COMMAND_TOPIC = "sysmo32_arm_command"
SYSMO32_LEFT_HAND_ACTION_TOPIC = "left_hand_action"
SYSMO32_RIGHT_HAND_ACTION_TOPIC = "right_hand_action"


@dataclass
class Sysmo32Ros2Topics:
    joint_state_topic: str = "/joint_states"
    arm_command_topic: str = "/sysmo_left_arm_controller/commands"
    min_snap_target_topic: str = "/min_snap/target"
    left_hand_topic: str = "/left_topic_to_hand"
    right_hand_topic: str = "/right_topic_to_hand"
    arm_command_queue_size: int = 60
    min_snap_target_queue_size: int = 10
    hand_command_queue_size: int = 10
    joint_state_timeout_s: float = 1.0


@dataclass
class Sysmo32HandConfig:
    default_action: int = SYSMO32_HAND_ACTION_RELEASE  # 默认动作：手释放
    grasp_action: int = SYSMO32_HAND_ACTION_GRASP  # 抓取动作：手抓取
    left_release_ros_action: Optional[int] = None
    left_grasp_ros_action: Optional[int] = None
    right_release_ros_action: Optional[int] = None
    right_grasp_ros_action: Optional[int] = None
    use_vr_hand_command_actions: bool = False
    vr_release_command: int = SYSMO32_HAND_ACTION_RELEASE
    vr_grasp_command: int = SYSMO32_HAND_ACTION_GRASP
    publish_on_change_only: bool = True  # 仅在动作变化时发布，避免重复发布
    heartbeat_hz: float = 3.0  # 心跳频率：3Hz
    grasp_enter_threshold_m: float = 0.035  # 抓取进入阈值：0.035m
    grasp_exit_threshold_m: float = 0.055  # 抓取退出阈值：0.055m
    confirm_frames: int = 3  # 确认帧数：3帧
    force_release_on_pause: bool = True  # 暂停时强制释放手
    force_release_on_timeout: bool = True  # 超时后强制释放手


@dataclass
class Sysmo32RealControlConfig:
    control_backend: str = "mujoco"  # 控制后端：real/mujoco/real_with_mujoco
    ros2: Sysmo32Ros2Topics = field(default_factory=Sysmo32Ros2Topics)  # ROS2 配置
    arm: Sysmo32ArmSafetyConfig = field(default_factory=Sysmo32ArmSafetyConfig)  # 手臂安全配置
    hand: Sysmo32HandConfig = field(default_factory=Sysmo32HandConfig)  # 手部配置
    state_publish_fps: float = 30.0  # LeRobot录制状态发布频率
    hand_frame_timeout_s: float = 1.0  # 手部帧超时时间：1.0s
    safety_hold_arm_on_pause: bool = True  # 暂停时保持手臂位置
    pause_hold_heartbeat_hz: float = 20.0  # 暂停保持命令心跳频率
    allow_placeholder_ik_for_mujoco: bool = False  # 允许降级 IK
    allow_mujoco_mirror_without_joint_state: bool = True
    mujoco_mirror_max_joint_velocity_rad_s: float = 3.0
    ik_nullspace_gain: float = 0.03
    ik_nullspace_step_limit_rad: float = 0.015
    ik_nullspace_reference_joints_rad: Optional[Sequence[float]] = None
    ik_orientation_weight: float = 0.2
    ik_max_joint_step_rad: float = 0.12
    ik_max_iter: int = 5
    ik_pos_tol_m: float = 1e-3
    ik_ori_tol_rad: float = 2e-2
    ik_profile_log_period_s: float = 1.0
    min_snap_expected_duration_s: float = 0.5
    min_snap_max_velocity_rad_s: float = 3.0
    min_snap_max_acceleration_rad_s2: float = 10.0

    def __post_init__(self):
        self.ik_nullspace_gain = max(0.0, float(self.ik_nullspace_gain))
        self.ik_nullspace_step_limit_rad = max(0.0, float(self.ik_nullspace_step_limit_rad))
        self.ik_orientation_weight = max(0.0, float(self.ik_orientation_weight))
        self.ik_max_joint_step_rad = max(0.0, float(self.ik_max_joint_step_rad))
        self.ik_max_iter = max(1, int(self.ik_max_iter))
        self.ik_pos_tol_m = max(0.0, float(self.ik_pos_tol_m))
        self.ik_ori_tol_rad = max(0.0, float(self.ik_ori_tol_rad))
        self.ik_profile_log_period_s = max(0.0, float(self.ik_profile_log_period_s))
        self.min_snap_expected_duration_s = max(1e-4, float(self.min_snap_expected_duration_s))
        self.min_snap_max_velocity_rad_s = max(1e-6, float(self.min_snap_max_velocity_rad_s))
        self.min_snap_max_acceleration_rad_s2 = max(1e-6, float(self.min_snap_max_acceleration_rad_s2))
        if self.ik_nullspace_reference_joints_rad is not None:
            reference = np.asarray(self.ik_nullspace_reference_joints_rad, dtype=np.float64)
            if reference.shape != (12,) or not np.all(np.isfinite(reference)):
                raise ValueError("ik_nullspace_reference_joints_rad must contain 12 finite values")
            self.ik_nullspace_reference_joints_rad = tuple(float(value) for value in reference)


# ROS2 桥接类,负责 ROS2 通信
class Sysmo32Ros2Bridge:
    """Optional ROS2 bridge.  It is inactive for pure MuJoCo dry-run."""

    def __init__(self, topics: Sysmo32Ros2Topics, require_ros: bool):
        self.topics = topics
        self.require_ros = require_ros
        self.available = False
        self.joint_cache = Sysmo32JointStateCache(topics.joint_state_timeout_s)  # 关节状态缓存
        self._rclpy = None
        self._node = None
        self._arm_pub = None
        self._min_snap_target_publisher = None
        self._left_hand_pub = None
        self._right_hand_pub = None
        if require_ros:
            self._init_ros2()  # 初始化发布器和订阅器

    def _init_ros2(self) -> None:
        try:
            import rclpy
            from min_snap.msg import MinSnapTarget
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Int32

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node("sysmo32_real_control")
            self._hand_msg_type = Int32
            self._min_snap_target_publisher = MinSnapTargetPublisher(
                ros_node=self._node,
                msg_type=MinSnapTarget,
                topic=self.topics.min_snap_target_topic,
                queue_size=self.topics.min_snap_target_queue_size,
            )
            self._left_hand_pub = self._node.create_publisher(
                Int32,
                self.topics.left_hand_topic,
                self.topics.hand_command_queue_size,
            )
            self._right_hand_pub = self._node.create_publisher(
                Int32,
                self.topics.right_hand_topic,
                self.topics.hand_command_queue_size,
            )
            self._node.create_subscription(
                JointState,
                self.topics.joint_state_topic,
                self._on_joint_state,
                10,
            )
            self.available = True
            logger.info(
                "SYSMO-32 ROS2 bridge connected: joint_state=%s min_snap_target=%s min_snap_output=%s",
                self.topics.joint_state_topic,
                self.topics.min_snap_target_topic,
                self.topics.arm_command_topic,
            )
        except Exception as exc:
            self.available = False
            message = f"SYSMO-32 ROS2 bridge unavailable: {exc}"
            logger.error(message)
            if self.require_ros:
                raise RuntimeError(message) from exc

    def _on_joint_state(self, msg) -> None:
        self.joint_cache.update_from_joint_state_msg(msg, now_s=time.time())

    def spin_once(self) -> None:
        if self.available and self._rclpy is not None and self._node is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def publish_min_snap_target(
        self,
        command: Sysmo32ArmCommand,
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

    def publish_hand_action(self, hand_side: str, action_id: int) -> bool:
        if not self.available:
            return False
        msg = self._hand_msg_type()
        msg.data = int(action_id)
        if hand_side == robots.LEFT:
            self._left_hand_pub.publish(msg)
        else:
            self._right_hand_pub.publish(msg)
        return True

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


class Sysmo32RealControl(Component):
    """Bimanual sysmo32 real-interface controller."""

    def __init__(
        self,
        host: str,
        control_backend: str,
        right_target_port: int,
        left_target_port: int,
        right_state_publish_port: int,
        left_state_publish_port: int,
        teleoperation_state_port: int,
        transformed_right_port: int,
        transformed_left_port: int,
        right_endeff_publish_port: Optional[int] = None,
        left_endeff_publish_port: Optional[int] = None,
        hand_action_mirror_port: int = ports.SYSMO32_HAND_ACTION_MIRROR_PORT,
        urdf_path: str = "robots/sysmo_description/urdf/sysmo32.urdf",
        config: Optional[Sysmo32RealControlConfig] = None,
    ):
        self.notify_component_start("sysmo32_real_control")
        self.host = host
        self.control_backend = control_backend
        self._validate_backend()
        self.config = config or Sysmo32RealControlConfig(control_backend=control_backend)
        self.config.control_backend = control_backend

        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._hand_action_mirror_port = hand_action_mirror_port
        self._right_state_publish_port = right_state_publish_port
        self._left_state_publish_port = left_state_publish_port
        self._right_endeff_publish_port = right_endeff_publish_port or right_state_publish_port
        self._left_endeff_publish_port = left_endeff_publish_port or left_state_publish_port
        self._publisher_manager.register_topic(
            self.host,
            self._hand_action_mirror_port,
            SYSMO32_LEFT_HAND_ACTION_TOPIC,
        )
        self._publisher_manager.register_topic(
            self.host,
            self._hand_action_mirror_port,
            SYSMO32_RIGHT_HAND_ACTION_TOPIC,
        )
        self._publisher_manager.register_topic(
            self.host,
            self._right_state_publish_port,
            "sysmo32_right",
        )
        self._publisher_manager.register_topic(
            self.host,
            self._left_state_publish_port,
            "sysmo32_left",
        )

        self._right_target_subscriber = ZMQSubscriber(
            host, right_target_port, "endeff_coords", message_type=CartesianTarget
        )
        self._left_target_subscriber = ZMQSubscriber(
            host, left_target_port, "endeff_coords", message_type=CartesianTarget
        )
        self._right_reset_subscriber = ZMQSubscriber(
            host, right_target_port, "reset", message_type=SessionCommand
        )
        self._left_reset_subscriber = ZMQSubscriber(
            host, left_target_port, "reset", message_type=SessionCommand
        )
        self._pause_subscriber = ZMQSubscriber(
            host, teleoperation_state_port, robots.PAUSE, message_type=SessionCommand
        )
        self._right_hand_subscriber = ZMQSubscriber(
            host,
            transformed_right_port,
            f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_COORDS}",
            message_type=InputFrame,
        )
        self._left_hand_subscriber = ZMQSubscriber(
            host,
            transformed_left_port,
            f"{robots.LEFT}_{robots.TRANSFORMED_HAND_COORDS}",
            message_type=InputFrame,
        )
        self._subscribers = [
            self._right_target_subscriber,
            self._left_target_subscriber,
            self._right_reset_subscriber,
            self._left_reset_subscriber,
            self._pause_subscriber,
            self._right_hand_subscriber,
            self._left_hand_subscriber,
        ]

        require_ros = True
        self._ros2 = Sysmo32Ros2Bridge(self.config.ros2, require_ros=require_ros)
        self._dry_joint_cache = Sysmo32JointStateCache(self.config.ros2.joint_state_timeout_s)
        self._dry_joint_cache.update(np.zeros(6), np.zeros(6), now_s=time.time())
        self._kinematics = Sysmo32MujocoKinematics(urdf_path)
        self._configure_ik_nullspace()

        self._builder = Sysmo32CommandBuilder(self.config.arm)
        self._limiter = Sysmo32CommandLimiter(self.config.arm)
        self._mujoco_limiter = Sysmo32CommandLimiter(self._make_mujoco_arm_safety_config())
        self._hand_mapper = Sysmo32HandGestureMapper(
            default_action=self.config.hand.default_action,
            grasp_action=self.config.hand.grasp_action,
            grasp_enter_threshold_m=self.config.hand.grasp_enter_threshold_m,
            grasp_exit_threshold_m=self.config.hand.grasp_exit_threshold_m,
            confirm_frames=self.config.hand.confirm_frames,
            hand_frame_timeout_s=self.config.hand_frame_timeout_s,
        )

        self._teleop_active = True
        self._needs_reset = True
        self._latest_targets: Dict[str, Optional[CartesianTarget]] = {robots.LEFT: None, robots.RIGHT: None}
        self._last_accepted_targets: Dict[str, Optional[CartesianTarget]] = {
            robots.LEFT: None,
            robots.RIGHT: None,
        }
        self._last_hand_actions = {
            robots.LEFT: self.config.hand.default_action,
            robots.RIGHT: self.config.hand.default_action,
        }
        self._last_vr_hand_commands: Dict[str, Optional[int]] = {robots.LEFT: None, robots.RIGHT: None}
        self._hand_frame_started = {robots.LEFT: False, robots.RIGHT: False}
        self._last_hand_publish_time = {robots.LEFT: 0.0, robots.RIGHT: 0.0}
        self._last_safety_log_time: Dict[str, float] = {}
        self._last_session_command: Optional[str] = None
        self._real_reset_ready = self.control_backend == "mujoco"
        self._next_state_publish_time_s = 0.0
        self._pause_hold_command: Optional[Sysmo32ArmCommand] = None
        self._last_pause_hold_publish_time = 0.0
        self._target_receive_count = {robots.LEFT: 0, robots.RIGHT: 0}
        self._last_target_rate_log_time = {robots.LEFT: 0.0, robots.RIGHT: 0.0}
        self._last_target_receive_time = {robots.LEFT: 0.0, robots.RIGHT: 0.0}
        self._arm_publish_count = 0
        self._last_arm_rate_log_time = 0.0
        self.enable_arm_command_value_debug = False
        self._last_arm_publish_time = 0.0
        self._last_real_timing_log_time = 0.0
        self._last_published_arm_command: Optional[Sysmo32ArmCommand] = None

    def _configure_ik_nullspace(self) -> None:
        if not hasattr(self._kinematics, "configure_nullspace"):
            return
        try:
            self._kinematics.configure_nullspace(
                reference_joints_rad=self.config.ik_nullspace_reference_joints_rad,
                gain=self.config.ik_nullspace_gain,
                step_limit_rad=self.config.ik_nullspace_step_limit_rad,
                orientation_weight=self.config.ik_orientation_weight,
                max_joint_step_rad=self.config.ik_max_joint_step_rad,
                max_iter=self.config.ik_max_iter,
                pos_tol_m=self.config.ik_pos_tol_m,
                ori_tol_rad=self.config.ik_ori_tol_rad,
                profile_log_period_s=self.config.ik_profile_log_period_s,
            )
        except TypeError:
            self._kinematics.configure_nullspace(
                reference_joints_rad=self.config.ik_nullspace_reference_joints_rad,
                gain=self.config.ik_nullspace_gain,
                step_limit_rad=self.config.ik_nullspace_step_limit_rad,
                orientation_weight=self.config.ik_orientation_weight,
                max_joint_step_rad=self.config.ik_max_joint_step_rad,
            )

    def _validate_backend(self) -> None:
        if self.control_backend not in ("real", "mujoco", "real_with_mujoco"):
            raise ValueError(
                f"control_backend must be one of: real, mujoco, real_with_mujoco; got {self.control_backend}"
            )

    def _make_mujoco_arm_safety_config(self) -> Sysmo32ArmSafetyConfig:
        """
        限幅使用真实机器人配置,但MuJoCo-only / mirror-only 模式下允许关节速度更大一点，
        否则仿真里动作太慢，看起来像没跟随
        """

        mirror_velocity = tuple(
            max(float(v), float(self.config.mujoco_mirror_max_joint_velocity_rad_s))
            for v in self.config.arm.max_joint_velocity_rad_s
        )
        return replace(self.config.arm, max_joint_velocity_rad_s=mirror_velocity)

    def stream(self):
        logger.info("Starting SYSMO-32 real-interface controller backend=%s", self.control_backend)
        if self.control_backend == "real_with_mujoco":
            logger.info(
                "SYSMO-32 real_with_mujoco requires fresh %s and a successful reset before "
                "publishing %s; MuJoCo mirrors only commands that pass the real publish gate",
                self.config.ros2.joint_state_topic,
                self.config.ros2.arm_command_topic,
            )
        while True:
            start = time.time()
            self._ros2.spin_once()
            self._handle_session_command()  # 处理暂停/恢复
            self._publish_pause_hold_if_needed()
            self._receive_hand_frames()  # 接收手部关键点
            self._publish_hand_actions_for_current_state()  # 发布手部动作
            self._handle_reset_requests()  # 处理重置请求
            self._receive_cartesian_targets()  # 接收笛卡尔目标
            self._publish_lerobot_joint_states()  # 按30Hz发布LeRobot录制状态
            self._publish_arm_command_if_safe()  # 发布安全的手臂命令
            elapsed = time.time() - start
            time.sleep(max(0.0, (1.0 / robots.VR_FREQ) - elapsed))

    def _handle_session_command(self) -> None:
        msg = self._pause_subscriber.recv_keypoints()
        if msg is None:
            return
        logger.info(
            "[Diag][REAL_SESSION_RX] command=%s teleop_active=%s needs_reset=%s",
            msg.command,
            self._teleop_active,
            self._needs_reset,
        )
        if msg.command == robots.PAUSE:
            self._last_session_command = robots.PAUSE
            self._enter_pause("pause command")
        elif msg.command == robots.RESUME:
            if self._teleop_active and self._last_session_command == robots.RESUME:
                return
            self._last_session_command = robots.RESUME
            self._teleop_active = True
            self._needs_reset = True
            self._real_reset_ready = (
                self.control_backend == "mujoco"
            )  # 没有真实 /joint_states 依赖，resume 后可以先认为 reset ready；其余情况不能直接 ready，必须等 fresh /joint_states + reset 成功后，才允许真实机械臂发命令
            self._last_accepted_targets = {robots.LEFT: None, robots.RIGHT: None}  # 清掉上一次接受过的目标
            self._pause_hold_command = None
            self._last_pause_hold_publish_time = 0.0
            logger.info("SYSMO-32 resume received: next targets require reset/rebaseline")

    def _enter_pause(self, reason: str) -> None:
        if not self._teleop_active:
            return
        self._teleop_active = False
        self._needs_reset = True
        self._real_reset_ready = self.control_backend == "mujoco"
        self._latest_targets = {robots.LEFT: None, robots.RIGHT: None}
        self._last_accepted_targets = {robots.LEFT: None, robots.RIGHT: None}
        self._hand_mapper.force_release()
        self._pause_hold_command = None
        self._last_pause_hold_publish_time = 0.0
        if self.config.hand.force_release_on_pause:
            for hand_side in (robots.LEFT, robots.RIGHT):
                if self._hand_frame_started[hand_side]:
                    self._publish_hand_action(hand_side, self.config.hand.default_action, reason, force=True)
        snapshot = self._pause_hold_snapshot()
        if snapshot is not None:
            self._limiter.reset(snapshot.all_joints)
            self._mujoco_limiter.reset(snapshot.all_joints)
            if self.config.safety_hold_arm_on_pause:
                self._pause_hold_command = self._builder.build(
                    snapshot.left_arm,
                    snapshot.right_arm,
                    timestamp_s=time.time(),
                )
                self._publish_pause_hold_if_needed(force=True, reason=reason)
        logger.info("SYSMO-32 paused immediately: %s", reason)

    def _pause_hold_snapshot(self) -> Optional[Sysmo32JointStateSnapshot]:
        if self.control_backend in ("real", "real_with_mujoco") and not self._real_joint_state_fresh():
            self._warn_safety(
                "pause_joint_state_stale",
                "pause hold skipped: /joint_states is stale, refusing to publish a stale hold target",
            )
            return None
        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            self._warn_safety("pause_joint_state_missing", "pause hold skipped: missing joint state")
        return snapshot

    def _publish_pause_hold_if_needed(self, force: bool = False, reason: str = "pause hold") -> None:
        if self._teleop_active or not self.config.safety_hold_arm_on_pause:
            return
        if self._pause_hold_command is None:
            return

        now = time.time()
        heartbeat_period = 1.0 / max(0.1, float(self.config.pause_hold_heartbeat_hz))
        if not force and now - self._last_pause_hold_publish_time < heartbeat_period:
            return

        command = Sysmo32ArmCommand(timestamp_s=now, values=self._pause_hold_command.values)
        published = self._publish_arm_command_outputs(
            command,
            real_joint_state_fresh=self._real_joint_state_fresh(),
            require_real_reset=False,
            allow_stale_real_hold=True,
        )
        if published:
            self._pause_hold_command = command
            self._last_pause_hold_publish_time = now
            logger.debug("SYSMO-32 pause hold command published reason=%s values=%s", reason, command.values)

    def _receive_hand_frames(self) -> None:
        for hand_side, subscriber in (
            (robots.LEFT, self._left_hand_subscriber),
            (robots.RIGHT, self._right_hand_subscriber),
        ):
            frame = subscriber.recv_keypoints()
            if frame is None:
                continue
            if getattr(frame, "hand_side", hand_side) != hand_side:
                self._warn_safety(
                    f"{hand_side}_hand_wrong_side",
                    f"{hand_side} hand frame ignored: frame side={getattr(frame, 'hand_side', None)}",
                )
                continue
            try:
                self._hand_mapper.update_from_keypoints(hand_side, frame.keypoints, now_s=time.time())
                self._hand_frame_started[hand_side] = True
                if self.config.hand.use_vr_hand_command_actions:
                    self._publish_vr_hand_command_on_edge(hand_side, frame.hand_command)
            except Exception as exc:
                self._warn_safety(f"{hand_side}_hand_invalid", f"{hand_side} hand invalid: {exc}")
                self._hand_mapper.force_release(hand_side)

    def _publish_hand_actions_for_current_state(self) -> None:
        now = time.time()
        for hand_side in (robots.LEFT, robots.RIGHT):
            if not self._hand_frame_started[hand_side]:
                continue
            if not self._teleop_active:
                action = self.config.hand.default_action
                reason = "pause"
            elif not self._hand_mapper.has_fresh_frame(hand_side, now_s=now):
                if not self.config.hand.force_release_on_timeout:
                    continue
                action = self._hand_mapper.force_release(hand_side)
                reason = "hand frame timeout"
            elif self.config.hand.use_vr_hand_command_actions:
                continue
            else:
                action = self._hand_mapper.action_for(hand_side, now_s=now)
                reason = "gesture"
            self._publish_hand_action(hand_side, action, reason)

    def _publish_vr_hand_command_on_edge(self, hand_side: str, hand_command) -> None:
        if hand_command is None:
            return
        try:
            command = int(hand_command)
        except (TypeError, ValueError):
            logger.warning("忽略无法解析的SYSMO-32/O6手部命令: %s", hand_command)
            return
        if command == self.config.hand.vr_grasp_command:
            action_id = self.config.hand.grasp_action
        elif command == self.config.hand.vr_release_command:
            action_id = self.config.hand.default_action
        else:
            logger.warning("忽略未知SYSMO-32/O6手部命令: %s", command)
            return
        if command == self._last_vr_hand_commands[hand_side]:
            return
        self._publish_hand_action(hand_side, action_id, "vr hand command", force=True)
        self._last_vr_hand_commands[hand_side] = command

    def _ros_hand_action_id(self, hand_side: str, action_id: int) -> int:
        if hand_side == robots.LEFT:
            if action_id == self.config.hand.grasp_action and self.config.hand.left_grasp_ros_action is not None:
                return int(self.config.hand.left_grasp_ros_action)
            if action_id == self.config.hand.default_action and self.config.hand.left_release_ros_action is not None:
                return int(self.config.hand.left_release_ros_action)
        else:
            if action_id == self.config.hand.grasp_action and self.config.hand.right_grasp_ros_action is not None:
                return int(self.config.hand.right_grasp_ros_action)
            if action_id == self.config.hand.default_action and self.config.hand.right_release_ros_action is not None:
                return int(self.config.hand.right_release_ros_action)
        return int(action_id)

    def _publish_hand_action(self, hand_side: str, action_id: int, reason: str, force: bool = False) -> None:
        now = time.time()
        ros_action_id = self._ros_hand_action_id(hand_side, int(action_id))
        heartbeat_period = 1.0 / max(0.1, self.config.hand.heartbeat_hz)
        changed = ros_action_id != self._last_hand_actions[hand_side]
        heartbeat_due = now - self._last_hand_publish_time[hand_side] >= heartbeat_period
        if not force and self.config.hand.publish_on_change_only and not changed and not heartbeat_due:
            return

        action = Sysmo32HandAction(now, hand_side, int(action_id), reason=reason)
        if self.control_backend in ("real", "real_with_mujoco"):
            published = self._ros2.publish_hand_action(hand_side, ros_action_id)
            if not published and self.control_backend == "real":
                self._warn_safety("ros_hand_unavailable", f"ROS2 hand publisher unavailable for {hand_side}")

        if self.control_backend in ("mujoco", "real_with_mujoco"):
            topic = (
                SYSMO32_LEFT_HAND_ACTION_TOPIC
                if hand_side == robots.LEFT
                else SYSMO32_RIGHT_HAND_ACTION_TOPIC
            )
            self._publisher_manager.publish(self.host, self._hand_action_mirror_port, topic, action)

        self._last_hand_actions[hand_side] = ros_action_id
        self._last_hand_publish_time[hand_side] = now
        logger.info(
            "SYSMO-32 hand action: %s action=%d ros_action=%d reason=%s",
            hand_side,
            action.action_id,
            ros_action_id,
            reason,
        )

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
        real_joint_state_fresh = self._real_joint_state_fresh()
        using_mujoco_fallback = (
            self.control_backend == "real_with_mujoco"
            and not real_joint_state_fresh
            and self._mujoco_joint_state_fallback_allowed()
        )
        if snapshot is None or (self.control_backend == "real" and not real_joint_state_fresh):
            self._warn_safety("reset_joint_state_stale", f"reset refused for {hand_side}: joint state stale")
            return
        if using_mujoco_fallback:
            self._warn_safety(
                "reset_mujoco_fallback",
                f"reset for {hand_side} uses MuJoCo fallback because /joint_states is stale; real arm remains held",
            )
        homo = self._fk(hand_side, snapshot)
        if homo is None:
            self._warn_safety("reset_fk_unavailable", f"reset refused for {hand_side}: FK unavailable")
            return
        self._publisher_manager.publish(
            host=self.host,
            port=publish_port,
            topic="endeff_homo",
            data=CartesianState(
                timestamp_s=time.time(),
                h_matrix=tuple(tuple(float(v) for v in row) for row in homo),
            ),
        )
        self._needs_reset = False
        self._real_reset_ready = real_joint_state_fresh
        self._limiter.reset(snapshot.all_joints)
        self._mujoco_limiter.reset(snapshot.all_joints)
        logger.info("SYSMO-32 reset pose published for %s on port %d", hand_side, publish_port)

    def _receive_cartesian_targets(self) -> None:
        for hand_side, subscriber in (
            (robots.LEFT, self._left_target_subscriber),
            (robots.RIGHT, self._right_target_subscriber),
        ):
            msg = subscriber.recv_keypoints()
            if msg is None:
                continue
            if msg.hand_side != hand_side:
                self._warn_safety(f"{hand_side}_wrong_target_side", f"wrong target side: {msg.hand_side}")
                continue
            self._latest_targets[hand_side] = msg
            self._log_target_receive_diag(hand_side, msg)

    def _log_target_receive_diag(self, hand_side: str, target: CartesianTarget) -> None:
        now = time.time()
        last_receive = self._last_target_receive_time[hand_side]
        if last_receive > 0.0:
            gap_s = now - last_receive
            if gap_s > 0.20:
                logger.warning(
                    "[Diag][REAL_TARGET_GAP] side=%s gap_ms=%.1f target_age_ms=%.1f port=%d",
                    hand_side,
                    gap_s * 1000.0,
                    (now - float(getattr(target, "timestamp_s", now) or now)) * 1000.0,
                    getattr(self._left_target_subscriber, "_port", -1)
                    if hand_side == robots.LEFT
                    else getattr(self._right_target_subscriber, "_port", -1),
                )
        self._last_target_receive_time[hand_side] = now
        if self._last_target_rate_log_time[hand_side] == 0.0:
            self._last_target_rate_log_time[hand_side] = now
        self._target_receive_count[hand_side] += 1
        window_s = now - self._last_target_rate_log_time[hand_side]
        if window_s >= 1.0:
            logger.info(
                "[Diag][REAL_TARGET_RATE] side=%s hz=%.1f count=%d last_target_age_ms=%.1f",
                hand_side,
                self._target_receive_count[hand_side] / window_s,
                self._target_receive_count[hand_side],
                (now - float(getattr(target, "timestamp_s", now) or now)) * 1000.0,
            )
            self._target_receive_count[hand_side] = 0
            self._last_target_rate_log_time[hand_side] = now

    def _publish_lerobot_joint_states(self) -> None:
        """Publish per-arm state dictionaries consumed by the LeRobot BeavrBot adapter."""

        state_publish_fps = float(self.config.state_publish_fps)
        if state_publish_fps <= 0.0:
            return

        now = time.time()
        if now < self._next_state_publish_time_s:
            return
        self._next_state_publish_time_s = now + (1.0 / state_publish_fps)

        if not self._joint_state_fresh():
            self._warn_safety("record_joint_state_stale", "LeRobot state publish skipped: stale joint state")
            return

        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            self._warn_safety(
                "record_joint_state_missing", "LeRobot state publish skipped: missing joint state"
            )
            return

        self._publish_lerobot_arm_state(robots.RIGHT, snapshot.right_arm, self._right_state_publish_port, now)
        self._publish_lerobot_arm_state(robots.LEFT, snapshot.left_arm, self._left_state_publish_port, now)

    def _publish_lerobot_arm_state(
        self,
        hand_side: str,
        joint_positions_rad: Sequence[float],
        publish_port: int,
        publish_time_s: float,
    ) -> None:
        topic = f"sysmo32_{hand_side}"
        state = {
            "joint_states": {
                "joint_position": [float(value) for value in joint_positions_rad],
                "timestamp": publish_time_s,
            },
            "joint_angles_rad": [float(value) for value in joint_positions_rad],
            "timestamp": publish_time_s,
        }

        target = self._latest_targets.get(hand_side)
        if target is not None:
            state["commanded_cartesian_state"] = {
                "commanded_cartesian_position": [
                    *[float(value) for value in target.position_m],
                    *[float(value) for value in target.orientation_xyzw],
                ],
                "timestamp_s": getattr(target, "timestamp_s", publish_time_s),
            }
        if self._last_published_arm_command is not None:
            command_joints = (
                self._last_published_arm_command.left_arm
                if hand_side == robots.LEFT
                else self._last_published_arm_command.right_arm
            )
            state["commanded_joint_states"] = {
                "joint_position": [float(value) for value in command_joints],
                "timestamp_s": self._last_published_arm_command.timestamp_s,
            }

        self._publisher_manager.publish(
            host=self.host,
            port=publish_port,
            topic=topic,
            data=state,
        )

    def _publish_arm_command_if_safe(self) -> None:
        loop_start_s = time.perf_counter()
        # 检查状态
        if not self._teleop_active:
            return
        real_joint_state_fresh = self._real_joint_state_fresh()
        if self.control_backend == "real" and not real_joint_state_fresh:
            self._warn_safety("joint_state_stale", "arm command rejected: joint state stale")
            return
        if (
            self.control_backend == "real_with_mujoco"
            and not real_joint_state_fresh
            and not self._mujoco_joint_state_fallback_allowed()
        ):
            self._warn_safety("joint_state_stale", "arm command rejected: joint state stale")
            return
        # 获取当前关节状态
        snapshot = self._current_joint_snapshot()
        if snapshot is None:
            self._warn_safety("joint_state_missing", "arm command rejected: missing joint state")
            return

        current = snapshot.all_joints
        desired_left = current[:6].copy()
        desired_right = current[6:].copy()
        any_target = False
        timing_entries = []
        # 对每个手臂求解 IK
        for hand_side in (robots.LEFT, robots.RIGHT):
            target = self._latest_targets[hand_side]
            if target is None:
                continue
            # 检查手部关键点是否weakly valid
            if not self._cartesian_target_fresh(target):
                self._warn_safety(
                    f"{hand_side}_target_stale",
                    f"{hand_side} arm held: CartesianTarget stale",
                )
                self._latest_targets[hand_side] = None
                continue
            target = self._sanitize_cartesian_target(hand_side, target)
            if target is None:
                continue
            ik_start_s = time.perf_counter()
            solved = self._solve_ik(hand_side, target, current)
            ik_ms = (time.perf_counter() - ik_start_s) * 1000.0
            if solved is None:
                self._warn_safety(f"{hand_side}_ik_fail", f"{hand_side} IK failed; arm held")
                continue
            if hand_side == robots.LEFT:
                desired_left = solved
            else:
                desired_right = solved
            self._last_accepted_targets[hand_side] = target
            any_target = True
            timing_entries.append(
                (
                    hand_side,
                    float(getattr(target, "timestamp_s", time.time()) or time.time()),
                    ik_ms,
                )
            )

        if not any_target:
            return
        # 构建命令并限幅
        mirror_only = self.control_backend == "mujoco"
        build_limit_start_s = time.perf_counter()
        try:
            command = self._builder.build(desired_left, desired_right, timestamp_s=time.time())
            limiter = self._mujoco_limiter if mirror_only else self._limiter
            limited, reason = limiter.limit(command, now_s=command.timestamp_s)
        except Exception as exc:
            self._warn_safety("command_build_fail", f"arm command build failed: {exc}")
            return
        if limited is None:
            self._warn_safety("command_limited_reject", f"arm command rejected: {reason}")
            return
        build_limit_ms = (time.perf_counter() - build_limit_start_s) * 1000.0
        publish_start_s = time.perf_counter()
        if not self._publish_arm_command_outputs(
            limited,
            real_joint_state_fresh=real_joint_state_fresh,
            require_real_reset=True,
        ):
            return
        publish_ms = (time.perf_counter() - publish_start_s) * 1000.0

        suffix = f" ({reason})" if reason else ""
        mode_suffix = " mirror_only" if mirror_only else ""
        self._log_arm_publish_diag(limited, current)
        self._log_real_timing_diag(
            timing_entries,
            build_limit_ms,
            publish_ms,
            (time.perf_counter() - loop_start_s) * 1000.0,
            reason,
        )
        if self.enable_arm_command_value_debug:
            logger.debug(
                "SYSMO-32 arm command published backend=%s%s%s values=%s",
                self.control_backend,
                mode_suffix,
                suffix,
                limited.values,
            )

    def _log_real_timing_diag(
        self,
        timing_entries: list[tuple[str, float, float]],
        build_limit_ms: float,
        publish_ms: float,
        loop_ms: float,
        limit_reason: str,
    ) -> None:
        now = time.time()
        if now - self._last_real_timing_log_time < 1.0:
            return
        self._last_real_timing_log_time = now
        if timing_entries:
            source_to_publish_ms = max((now - ts) * 1000.0 for _, ts, _ in timing_entries)
            ik_detail = ",".join(f"{side}:{ik_ms:.1f}" for side, _, ik_ms in timing_entries)
            sides = ",".join(side for side, _, _ in timing_entries)
        else:
            source_to_publish_ms = 0.0
            ik_detail = ""
            sides = ""
        logger.info(
            "[Diag][TIMING_REAL] backend=%s sides=%s source_to_publish_ms=%.1f ik_ms=%s "
            "build_limit_ms=%.1f publish_ms=%.1f loop_ms=%.1f limit=%s",
            self.control_backend,
            sides,
            source_to_publish_ms,
            ik_detail,
            build_limit_ms,
            publish_ms,
            loop_ms,
            limit_reason or "none",
        )

    def _log_arm_publish_diag(
        self,
        command: Optional[Sysmo32ArmCommand] = None,
        current_joints_rad: Optional[Sequence[float]] = None,
    ) -> None:
        now = time.time()
        if self._last_arm_publish_time > 0.0:
            gap_s = now - self._last_arm_publish_time
            if gap_s > 0.20:
                logger.warning(
                    "[Diag][REAL_ARM_COMMAND_GAP] backend=%s gap_ms=%.1f",
                    self.control_backend,
                    gap_s * 1000.0,
                )
        self._last_arm_publish_time = now
        if self._last_arm_rate_log_time == 0.0:
            self._last_arm_rate_log_time = now
        self._arm_publish_count += 1
        window_s = now - self._last_arm_rate_log_time
        if window_s >= 1.0:
            logger.info(
                "[Diag][REAL_ARM_COMMAND_RATE] backend=%s hz=%.1f count=%d",
                self.control_backend,
                self._arm_publish_count / window_s,
                self._arm_publish_count,
            )
            if command is not None and current_joints_rad is not None:
                current = np.asarray(current_joints_rad, dtype=np.float64)
                commanded = np.asarray(command.values[:12], dtype=np.float64)
                if current.shape == (12,) and commanded.shape == (12,):
                    error = commanded - current
                    logger.info(
                        "[Diag][REAL_ARM_COMMAND_DELTA] max_abs_error_rad=%.4f "
                        "command_range_rad=(%.4f,%.4f) current_range_rad=(%.4f,%.4f)",
                        float(np.max(np.abs(error))),
                        float(np.min(commanded)),
                        float(np.max(commanded)),
                        float(np.min(current)),
                        float(np.max(current)),
                    )
            self._arm_publish_count = 0
            self._last_arm_rate_log_time = now

    def _publish_arm_command_outputs(
        self,
        command: Sysmo32ArmCommand,
        real_joint_state_fresh: bool,
        require_real_reset: bool,
        allow_stale_real_hold: bool = False,
    ) -> bool:
        if self.control_backend in ("real", "real_with_mujoco"):
            real_state_gate_ready = real_joint_state_fresh or allow_stale_real_hold
            reset_gate_ready = self._real_reset_ready or not require_real_reset
            if not (real_state_gate_ready and reset_gate_ready) and self.control_backend == "real":
                self._warn_safety(
                    "real_reset_required", "real arm held until reset succeeds with fresh /joint_states"
                )
                return False
            if not (real_state_gate_ready and reset_gate_ready):
                self._warn_safety(
                    "real_with_mujoco_reset_required",
                    "/joint_states is stale or real reset is missing; holding both real command and MuJoCo mirror",
                )
                return False

        if not self._ros2.publish_min_snap_target(
            command,
            self.config.min_snap_expected_duration_s,
            self.config.min_snap_max_velocity_rad_s,
            self.config.min_snap_max_acceleration_rad_s2,
        ):
            self._warn_safety("min_snap_unavailable", "SYSMO-32 min_snap target publisher unavailable")
            return False

        self._last_published_arm_command = command
        return True

    def _current_joint_snapshot(self) -> Optional[Sysmo32JointStateSnapshot]:
        # 真实模式：通过 _ros2.joint_cache 获取关节状态（来自 ROS2）;
        # 模拟模式：使用 _dry_joint_cache 作为虚拟关节状态缓存
        if self.control_backend == "mujoco":
            if self._real_joint_state_fresh():
                return self._ros2.joint_cache.snapshot
            return self._dry_joint_cache.snapshot
        if self.control_backend == "real_with_mujoco" and self._real_joint_state_fresh():
            return self._ros2.joint_cache.snapshot
        if (
            self.control_backend == "real_with_mujoco"
            and not self._real_reset_ready
            and self._mujoco_joint_state_fallback_allowed()
        ):
            return self._dry_joint_cache.snapshot
        if self._real_joint_state_fresh():
            return self._ros2.joint_cache.snapshot
        if self.control_backend == "real_with_mujoco" and self._mujoco_joint_state_fallback_allowed():
            return self._dry_joint_cache.snapshot
        return self._ros2.joint_cache.snapshot

    def _joint_state_fresh(self) -> bool:
        if self.control_backend == "mujoco":
            return self._dry_joint_cache.snapshot is not None
        if self.control_backend == "real_with_mujoco" and self._mujoco_joint_state_fallback_allowed():
            return self._real_joint_state_fresh() or self._dry_joint_cache.snapshot is not None
        return self._ros2.joint_cache.is_fresh()

    def _real_joint_state_fresh(self) -> bool:
        return self._ros2.joint_cache.is_fresh()

    def _mujoco_joint_state_fallback_allowed(self) -> bool:
        return (
            self.control_backend in ("mujoco", "real_with_mujoco")
            and self.config.allow_mujoco_mirror_without_joint_state
            and self._dry_joint_cache.snapshot is not None
        )

    def _fk(self, hand_side: str, snapshot: Sysmo32JointStateSnapshot) -> Optional[np.ndarray]:
        # 如果 MuJoCo 运动学模型可用，使用真实 FK 计算
        if self._kinematics.available:
            return self._kinematics.fk(hand_side, snapshot.all_joints)
        # 降级方案：仅在纯模拟模式且允许占位符时使用
        if (
            self.control_backend in ("mujoco", "real_with_mujoco")
            and self.config.allow_placeholder_ik_for_mujoco
            and self._mujoco_joint_state_fallback_allowed()
        ):
            # Dry-run only fallback.  Real mode refuses reset without model FK.
            homo = np.eye(4)
            offset = np.asarray([0.28, 0.45 if hand_side == robots.LEFT else -0.45, 0.10], dtype=np.float64)
            homo[:3, 3] = offset
            return homo
        return None

    def _solve_ik(
        self, hand_side: str, target: CartesianTarget, current_joints: Sequence[float]
    ) -> Optional[np.ndarray]:
        if self._kinematics.available:
            return self._kinematics.solve_ik(hand_side, target, current_joints)
        if (
            self.control_backend in ("mujoco", "real_with_mujoco")
            and self.config.allow_placeholder_ik_for_mujoco
            and self._mujoco_joint_state_fallback_allowed()
        ):
            return self._kinematics.placeholder_ik(hand_side, target, current_joints)
        return None

    def _cartesian_target_fresh(self, target: CartesianTarget) -> bool:
        timestamp_s = getattr(target, "timestamp_s", 0.0) or 0.0
        return time.time() - timestamp_s <= self.config.hand_frame_timeout_s

    def _sanitize_cartesian_target(
        self, hand_side: str, target: CartesianTarget
    ) -> Optional[CartesianTarget]:
        pos = np.asarray(target.position_m, dtype=np.float64)
        quat = np.asarray(target.orientation_xyzw, dtype=np.float64)
        # 1. NaN/Inf 检查
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.all(np.isfinite(pos))
            or not np.all(np.isfinite(quat))
        ):
            self._warn_safety(f"{hand_side}_target_nan", f"{hand_side} target rejected: NaN/Inf")
            return None

        quat = self._normalize_quaternion_xyzw(quat)
        if quat is None:
            self._warn_safety(f"{hand_side}_target_quat", f"{hand_side} target rejected: invalid quaternion")
            return None

        # 2. 工作空间限幅
        clipped_pos = pos.copy()
        for axis_idx, axis in enumerate(("x", "y", "z")):
            low, high = self.config.arm.workspace_limits[axis]
            clipped_pos[axis_idx] = float(np.clip(clipped_pos[axis_idx], low, high))
        if not np.allclose(clipped_pos, pos, atol=1e-12):
            self._warn_safety(f"{hand_side}_workspace_clamped", f"{hand_side} target clamped to workspace")

        last = self._last_accepted_targets[hand_side]
        if last is not None:
            last_pos = np.asarray(last.position_m, dtype=np.float64)
            # 3. 位移步进限幅。不能直接拒绝，否则手腕移动稍快时会冻结跟随。
            delta = clipped_pos - last_pos
            translation_step = float(np.linalg.norm(delta))
            max_translation_step = float(self.config.arm.max_translation_step_m)
            if translation_step > max_translation_step > 0.0:
                clipped_pos = last_pos + delta * (max_translation_step / translation_step)
                self._warn_safety(
                    f"{hand_side}_translation_clamped",
                    f"{hand_side} target translation step {translation_step:.3f}m clamped to {max_translation_step:.3f}m",
                )

            # 4. 姿态步进限幅。使用 nlerp 逐步靠近目标四元数，保持链路继续运动。
            last_quat = self._normalize_quaternion_xyzw(np.asarray(last.orientation_xyzw, dtype=np.float64))
            if last_quat is not None:
                rotation_step = quaternion_angle_delta_rad(tuple(last_quat), tuple(quat))
                max_rotation_step = float(self.config.arm.max_rotation_step_rad)
                if rotation_step > max_rotation_step > 0.0:
                    quat = self._interpolate_quaternion_xyzw(
                        last_quat, quat, max_rotation_step / rotation_step
                    )
                    self._warn_safety(
                        f"{hand_side}_rotation_clamped",
                        f"{hand_side} target rotation step {rotation_step:.3f}rad clamped to {max_rotation_step:.3f}rad",
                    )

        return CartesianTarget(
            timestamp_s=target.timestamp_s,
            hand_side=target.hand_side,
            frame_id=target.frame_id,
            position_m=tuple(float(v) for v in clipped_pos),
            orientation_xyzw=tuple(float(v) for v in quat),
            linear_velocity_ff=target.linear_velocity_ff,
            angular_velocity_ff=target.angular_velocity_ff,
            stiffness=target.stiffness,
            damping=target.damping,
        )

    @staticmethod
    def _normalize_quaternion_xyzw(quat: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9 or not np.isfinite(norm):
            return None
        normalized = quat / norm
        if normalized[3] < 0.0:
            normalized = -normalized
        return normalized

    @classmethod
    def _interpolate_quaternion_xyzw(cls, start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if float(np.dot(start, end)) < 0.0:
            end = -end
        quat = (1.0 - fraction) * start + fraction * end
        normalized = cls._normalize_quaternion_xyzw(quat)
        if normalized is None:
            return start
        return normalized

    def _warn_safety(self, key: str, message: str) -> None:
        now = time.time()
        # 检查距离上次记录该警告是否超过1秒
        if now - self._last_safety_log_time.get(key, 0.0) < 1.0:
            return  # 未超过1秒，跳过本次记录
        # 更新最后记录时间
        self._last_safety_log_time[key] = now
        # 记录警告日志
        logger.warning("[SYSMO-32 Safety] %s", message)

    def cleanup(self) -> None:
        for subscriber in getattr(self, "_subscribers", []):
            subscriber.stop()
        self._ros2.close()
        cleanup_zmq_resources()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = [
    "SYSMO32_ARM_COMMAND_TOPIC",
    "SYSMO32_LEFT_HAND_ACTION_TOPIC",
    "SYSMO32_RIGHT_HAND_ACTION_TOPIC",
    "Sysmo32Ros2Topics",
    "Sysmo32HandConfig",
    "Sysmo32RealControlConfig",
    "Sysmo32RealControl",
    "Sysmo32Ros2Bridge",
]
