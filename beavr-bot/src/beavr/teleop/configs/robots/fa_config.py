"""FA robot teleoperation configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from beavr.teleop.common.configs.loader import Laterality, log_laterality_configuration
from beavr.teleop.components.interface.robots.fa_arm_ik_client import FaArmIkConfig
from beavr.teleop.components.interface.robots.fa_command_builder import (
    FA_NECK_JOINT_NAMES,
    FA_UPPER_COMMAND_LENGTH,
    FaUpperPositionSafetyConfig,
)
from beavr.teleop.components.interface.robots.fa_mujoco_kinematics import (
    FA_LEFT_ARM_JOINT_NAMES,
    FA_RIGHT_ARM_JOINT_NAMES,
    FaKinematicsConfig,
)
from beavr.teleop.components.interface.robots.fa_real_control import (
    FaRealControl,
    FaRealControlConfig,
    FaRos2Topics,
)
from beavr.teleop.components.operator.robots.fa_operator import H_R_V_FA, FaOperator
from beavr.teleop.configs.constants import network, ports, robots
from beavr.teleop.configs.robots import TeleopRobotConfig
from beavr.teleop.configs.robots.shared_components import SharedComponentRegistry
from beavr.teleop.configs.robots.sysmo32_config import (
    SYSMO32_LEFT_PORT_OFFSET,
    SYSMO32_RIGHT_PORT_OFFSET,
    Sysmo32RealCameraStreamerCfg,
)
from beavr.teleop.configs.robots.sysmo_mujoco_config import Sysmo32MujocoCommandMirrorCfg

logger = logging.getLogger(__name__)

ROBOT_NAME_FA = "fa"
FA_DESCRIPTION_PATH = "/home/likunwei/dataCollection/beavr-bot/robots/fa_description"
FA_URDF_PATH = f"{FA_DESCRIPTION_PATH}/urdf/fa_robot.urdf"
FA_SRDF_PATH = "/home/likunwei/humanoid_ws/src/fa_moveit2_config/config/fa_robot.srdf"
FA_UPPER_POSITION_COMMAND_TOPIC = "/upper_position_controller/commands"


@dataclass
class FaRealControlCfg:
    """FA bimanual control layer using the native 16D upper-position ABI."""

    host: str = network.HOST_ADDRESS
    control_backend: str = "mujoco"
    right_target_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_target_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET
    right_endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET
    right_state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    upper_command_mirror_port: int = ports.SYSMO32_ARM_COMMAND_MIRROR_PORT
    hand_action_mirror_port: int = ports.SYSMO32_HAND_ACTION_MIRROR_PORT
    urdf_path: str = FA_URDF_PATH
    config: FaRealControlConfig = field(
        default_factory=lambda: FaRealControlConfig(
            control_backend="mujoco",
            ros2=FaRos2Topics(
                joint_state_topic="/joint_states",
                upper_position_command_topic=FA_UPPER_POSITION_COMMAND_TOPIC,
                left_hand_topic="/left_topic_to_hand",
                right_hand_topic="/right_topic_to_hand",
                upper_position_command_queue_size=60,
                hand_command_queue_size=10,
                joint_state_timeout_s=1.0,
            ),
            upper=FaUpperPositionSafetyConfig(
                neck_default_positions_rad=(0.0, 0.0),
                joint_lower_limits_rad=(
                    -2.79,
                    -0.33,
                    -2.79,
                    -1.40,
                    -2.79,
                    -0.52,
                    -1.57,
                    -2.79,
                    -3.49,
                    -2.79,
                    -1.40,
                    -2.79,
                    -0.52,
                    -1.57,
                    -3.14,
                    -3.14,
                ),
                joint_upper_limits_rad=(
                    2.79,
                    3.49,
                    2.79,
                    0.26,
                    2.79,
                    0.52,
                    1.57,
                    2.79,
                    0.33,
                    2.79,
                    0.26,
                    2.79,
                    0.52,
                    1.57,
                    3.14,
                    3.14,
                ),
                max_joint_velocity_rad_s=tuple([3.0] * FA_UPPER_COMMAND_LENGTH),
                max_joint_jump_rad=0.5,
                max_translation_step_m=0.30,
                max_rotation_step_rad=0.5,
            ),
            state_publish_fps=30.0,
            safety_hold_arm_on_pause=True,
            pause_hold_heartbeat_hz=20.0,
            allow_mujoco_mirror_without_joint_state=True,
            publish_upper_command_topic_in_mujoco=False,
            arm_trajectory_smoother="min_snap",
            arm_servo_max_velocity_rad_s=3.0,
            arm_servo_max_acceleration_rad_s2=10.0,
            arm_servo_max_jerk_rad_s3=120.0,
            kinematics=FaKinematicsConfig(
                model_path=FA_URDF_PATH,
                left_joint_names=FA_LEFT_ARM_JOINT_NAMES,
                right_joint_names=FA_RIGHT_ARM_JOINT_NAMES,
                left_endeff_body="left_hand_base_link",
                right_endeff_body="right_hand_base_link",
                max_iter=8,
                dls_damping=0.1,
            ),
            ik=FaArmIkConfig(
                urdf_file=FA_URDF_PATH,
                srdf_file=FA_SRDF_PATH,
                module_name="ik_7dof_pybind",
                reference_frame="pelvis",
                max_iters=200,
                eps=1e-3,
            ),
        )
    )

    def _apply_backend_overrides(self):
        self.config.control_backend = self.control_backend
        if self.control_backend == "mujoco":
            self.config.publish_upper_command_topic_in_mujoco = False
        if self.control_backend == "real_with_mujoco":
            self.config.allow_mujoco_mirror_without_joint_state = False

    def __post_init__(self):
        self._apply_backend_overrides()

    def build(self):
        self._apply_backend_overrides()
        return FaRealControl(
            host=self.host,
            control_backend=self.control_backend,
            right_target_port=self.right_target_port,
            left_target_port=self.left_target_port,
            right_endeff_publish_port=self.right_endeff_publish_port,
            left_endeff_publish_port=self.left_endeff_publish_port,
            right_state_publish_port=self.right_state_publish_port,
            left_state_publish_port=self.left_state_publish_port,
            teleoperation_state_port=self.teleoperation_state_port,
            upper_command_mirror_port=self.upper_command_mirror_port,
            hand_action_mirror_port=self.hand_action_mirror_port,
            urdf_path=self.urdf_path,
            config=self.config,
        )


@dataclass
class FaOperatorCfg:
    host: str = network.HOST_ADDRESS
    transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT
    stream_configs: dict[str, Any] = field(
        default_factory=lambda: {"host": network.HOST_ADDRESS, "port": ports.CONTROL_STREAM_PORT}
    )
    stream_oculus: bool = True
    endeff_publish_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    endeff_subscribe_port: int = ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    moving_average_limit: int = 3
    arm_resolution_port: int = ports.KEYPOINT_STREAM_PORT
    use_filter: bool = False
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    hand_frame_timeout_s: float = 1.0
    rotation_delta_frame: str = "base"
    logging_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "log_dir": "logs",
            "log_poses": True,
            "log_prefix": "fa",
        }
    )
    hand_side: str = robots.RIGHT

    def build(self):
        return FaOperator(
            operator_name=f"fa_{self.hand_side}_operator",
            host=self.host,
            transformed_keypoints_port=self.transformed_keypoints_port,
            stream_configs=self.stream_configs,
            stream_oculus=self.stream_oculus,
            endeff_publish_port=self.endeff_publish_port,
            endeff_subscribe_port=self.endeff_subscribe_port,
            moving_average_limit=self.moving_average_limit,
            use_filter=self.use_filter,
            arm_resolution_port=self.arm_resolution_port,
            teleoperation_state_port=self.teleoperation_state_port,
            logging_config=self.logging_config,
            hand_side=self.hand_side,
            hand_frame_timeout_s=self.hand_frame_timeout_s,
            rotation_delta_frame=self.rotation_delta_frame,
            h_r_v=H_R_V_FA,
        )


@dataclass
@TeleopRobotConfig.register_subclass(ROBOT_NAME_FA)
class FaConfig:
    """FA bimanual robot config."""

    robot_name: str = ROBOT_NAME_FA
    laterality: Laterality = Laterality.BIMANUAL
    control_backend: str = "mujoco"
    model_root: str = FA_DESCRIPTION_PATH
    urdf_file: str = FA_URDF_PATH
    srdf_file: str = FA_SRDF_PATH
    upper_position_command_topic: str = FA_UPPER_POSITION_COMMAND_TOPIC
    upper_position_command_type: str = "std_msgs/msg/Float64MultiArray"
    upper_position_command_size: int = FA_UPPER_COMMAND_LENGTH
    left_arm_joint_names: tuple[str, ...] = FA_LEFT_ARM_JOINT_NAMES
    right_arm_joint_names: tuple[str, ...] = FA_RIGHT_ARM_JOINT_NAMES
    neck_joint_names: tuple[str, ...] = FA_NECK_JOINT_NAMES
    left_arm_joint_count: int = len(FA_LEFT_ARM_JOINT_NAMES)
    right_arm_joint_count: int = len(FA_RIGHT_ARM_JOINT_NAMES)
    home_joints_rad: tuple[float, ...] = field(default_factory=lambda: tuple([0.0] * 14))
    ready_joints_rad: tuple[float, ...] = field(default_factory=lambda: tuple([0.0] * 14))

    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    environment: list = field(default_factory=list)
    camera_streamers: list = field(default_factory=list)

    def __post_init__(self):
        if self.control_backend not in ("real", "mujoco", "real_with_mujoco"):
            raise ValueError("fa control_backend must be one of: real, mujoco, real_with_mujoco")
        log_laterality_configuration(self.laterality, ROBOT_NAME_FA)
        self._configure_for_laterality()

    def _configure_for_laterality(self):
        self.detector = []
        if self.laterality == Laterality.BIMANUAL:
            self.detector.append(
                SharedComponentRegistry.get_bimanual_detector_config(
                    host=network.HOST_ADDRESS,
                    detector_type="pico4",
                )
            )
        else:
            hand_side = robots.RIGHT if self.laterality == Laterality.RIGHT else robots.LEFT
            self.detector.append(
                SharedComponentRegistry.get_detector_config(
                    hand_side=hand_side,
                    host=network.HOST_ADDRESS,
                    detector_type="pico4",
                )
            )

        self.transforms = []
        if self.laterality in (Laterality.RIGHT, Laterality.BIMANUAL):
            self.transforms.append(
                SharedComponentRegistry.get_transform_config(
                    hand_side=robots.RIGHT,
                    host=network.HOST_ADDRESS,
                    keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                    moving_average_limit=3,
                )
            )
        if self.laterality in (Laterality.LEFT, Laterality.BIMANUAL):
            self.transforms.append(
                SharedComponentRegistry.get_transform_config(
                    hand_side=robots.LEFT,
                    host=network.HOST_ADDRESS,
                    keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                    moving_average_limit=3,
                )
            )

        self.camera_streamers = []
        if self.control_backend == "real":
            self.camera_streamers = [
                Sysmo32RealCameraStreamerCfg(
                    host=network.HOST_ADDRESS,
                    port=ports.SIM_IMAGE_PORT,
                    camera_name="front",
                    camera_type="opencv",
                    camera_index=0,
                    fps=30,
                    width=640,
                    height=480,
                    rotation=180,
                )
            ]

        self.robots = [FaRealControlCfg(host=network.HOST_ADDRESS, control_backend=self.control_backend)]

        self.operators = []
        if self.laterality in (Laterality.RIGHT, Laterality.BIMANUAL):
            self.operators.append(
                FaOperatorCfg(
                    transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                    endeff_publish_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    endeff_subscribe_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    hand_side=robots.RIGHT,
                    logging_config={
                        "enabled": False,
                        "log_dir": "logs",
                        "log_poses": True,
                        "log_prefix": "fa_right",
                    },
                )
            )
        if self.laterality in (Laterality.LEFT, Laterality.BIMANUAL):
            self.operators.append(
                FaOperatorCfg(
                    transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                    endeff_publish_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    endeff_subscribe_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    hand_side=robots.LEFT,
                    logging_config={
                        "enabled": False,
                        "log_dir": "logs",
                        "log_poses": True,
                        "log_prefix": "fa_left",
                    },
                )
            )

        self.environment = []
        if self.control_backend in ("mujoco", "real_with_mujoco"):
            self.environment = [
                Sysmo32MujocoCommandMirrorCfg(
                    host=network.HOST_ADDRESS,
                    arm_command_port=ports.SYSMO32_ARM_COMMAND_MIRROR_PORT,
                    hand_action_port=ports.SYSMO32_HAND_ACTION_MIRROR_PORT,
                    urdf_path=FA_URDF_PATH,
                    kinematics_type="fa",
                    control_dt=0.01,
                    render=True,
                    load_model=True,
                    print_hand_action_only=True,
                    arm_command_source="zmq" if self.control_backend == "mujoco" else "ros2",
                    ros_arm_command_topic=FA_UPPER_POSITION_COMMAND_TOPIC,
                    publish_joint_states=self.control_backend == "mujoco",
                    joint_state_topic="/joint_states",
                    joint_state_publish_hz=50.0,
                    arm_command_interpolation_steps=5,
                    interpolation_profile="min_snap",
                    expected_command_length=FA_UPPER_COMMAND_LENGTH,
                    joint_state_joint_names=FA_LEFT_ARM_JOINT_NAMES + FA_RIGHT_ARM_JOINT_NAMES,
                )
            ]

    def build(self):
        return {
            "robot_name": self.robot_name,
            "detector": [detector.build() for detector in self.detector],
            "transforms": [item.build() for item in self.transforms],
            "visualizers": [item.build() for item in self.visualizers],
            "robots": [item.build() for item in self.robots],
            "operators": [item.build() for item in self.operators],
            "environment": [item.build() for item in self.environment],
            "camera_streamers": [item.build() for item in self.camera_streamers],
        }


__all__ = ["FaConfig", "FaOperatorCfg", "FaRealControlCfg"]
