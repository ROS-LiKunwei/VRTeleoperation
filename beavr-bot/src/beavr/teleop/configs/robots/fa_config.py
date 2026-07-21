"""FA robot teleoperation configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
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
from beavr.teleop.components.operator.robots.fa_operator import FaOperator, H_R_V_FA
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


def _beavr_bot_root() -> Path:
    return Path(os.environ.get("BEAVR_BOT_ROOT", Path(__file__).resolve().parents[5])).expanduser()


def _first_existing_path(env_name: str, candidates: list[Path], default: str = "") -> str:
    env_value = os.environ.get(env_name)
    if env_value:
        return str(Path(env_value).expanduser())
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return default


ROBOT_NAME_FA = "fa"
BEAVR_BOT_ROOT = _beavr_bot_root()
FA_DESCRIPTION_PATH = str(BEAVR_BOT_ROOT / "robots" / "fa_description")
FA_URDF_PATH = _first_existing_path(
    "FA_URDF_PATH",
    [BEAVR_BOT_ROOT / "robots" / "fa_description" / "urdf" / "fa_robot.urdf"],
    str(BEAVR_BOT_ROOT / "robots" / "fa_description" / "urdf" / "fa_robot.urdf"),
)
FA_SRDF_PATH = _first_existing_path(
    "FA_SRDF_PATH",
    [
        Path.home() / "likunwei_ws" / "src" / "fa_moveit2_config" / "config" / "fa_robot.srdf",
        Path.home() / "humanoid_ws" / "src" / "fa_moveit2_config" / "config" / "fa_robot.srdf",
        Path("/home/likunwei/humanoid_ws/src/fa_moveit2_config/config/fa_robot.srdf"),
    ],
)
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
    urdf_path: str = FA_URDF_PATH
    config: FaRealControlConfig = field(
        default_factory=lambda: FaRealControlConfig(
            control_backend="mujoco",
            ros2=FaRos2Topics(
                joint_state_topic="/joint_states",
                upper_position_command_topic="",
                min_snap_target_topic="/min_snap/target",
                upper_position_command_queue_size=60,
                min_snap_target_queue_size=10,
                joint_state_timeout_s=1.0,
            ),
            upper=FaUpperPositionSafetyConfig(
                neck_default_positions_rad=(0.0, 0.0),
                joint_lower_limits_rad=(
                    -2.79, -0.33, -2.79, -1.40, -2.79, -0.52, -1.57,
                    -2.79, -3.49, -2.79, -1.40, -2.79, -0.52, -1.57,
                    -3.14, -3.14,
                ),
                joint_upper_limits_rad=(
                    2.79, 3.49, 2.79, 0.26, 2.79, 0.52, 1.57,
                    2.79, 0.33, 2.79, 0.26, 2.79, 0.52, 1.57,
                    3.14, 3.14,
                ),
                max_joint_velocity_rad_s=tuple([1.2] * FA_UPPER_COMMAND_LENGTH),
                max_joint_jump_rad=0.5,
                max_rate_limit_dt_s=0.02,
                max_translation_step_m=0.30,
                max_rotation_step_rad=0.5,
            ),
            state_publish_fps=30.0,
            command_publish_hz=100.0,
            safety_hold_arm_on_pause=True,
            pause_hold_heartbeat_hz=20.0,
            allow_mujoco_mirror_without_joint_state=True,
            max_ik_solution_jump_rad=0.3,
            ik_continuity_weight=0.05,
            ik_continuity_task_score_slack=0.003,
            ik_max_position_error_m=0.06,
            ik_max_orientation_error_rad=0.12,
            min_snap_expected_duration_s=0.04,
            min_snap_max_velocity_rad_s=1.0,
            min_snap_max_acceleration_rad_s2=6.0,
            min_snap_target_publish_hz=60.0,
            min_snap_target_epsilon_rad=0.002,
            ik_cartesian_position_deadband_m=0.01,
            ik_cartesian_orientation_deadband_rad=0.015,
            ik_reachable_fallback_enabled=True,
            ik_reachable_fallback_iterations=10,
            ik_reachable_fallback_min_alpha=0.001,
            ik_reachable_fallback_orientation_alphas=(1.0, 0.75, 0.5, 0.0),
            ik_return_recovery_enabled=True,
            ik_return_recovery_min_retreat_m=0.015,
            ik_return_recovery_max_position_error_m=0.35,
            ik_return_recovery_max_orientation_error_rad=1.0,
            ik_multi_seed_enabled=True,
            ik_escape_enabled=True,
            ik_escape_trigger_count=8,
            ik_escape_target_tolerance_rad=0.04,
            ik_escape_local_enabled=True,
            ik_escape_local_elbow_target_rad=-1.05,
            ik_escape_local_max_joint_delta_rad=0.25,
            ik_singularity_output_filter_enabled=True,
            ik_singularity_filter_enter_problem_count=2,
            ik_singularity_filter_exit_problem_count=0,
            ik_singularity_filter_elbow_enter_rad=-1.20,
            ik_singularity_filter_elbow_exit_rad=-1.05,
            ik_singularity_filter_elbow_upper_enter_rad=0.12,
            ik_singularity_filter_elbow_upper_exit_rad=-0.05,
            ik_singularity_output_filter_alpha=0.85,
            ik_singularity_output_filter_max_step_rad=0.08,
            initial_pose_enabled=True,
            initial_left_arm_positions_rad=(-1.05, 0.76, -0.62, -1.03, -0.68, 0.0, -0.35),
            initial_right_arm_positions_rad=(-1.05, -0.76, 0.62, -1.03, 0.68, 0.0, 0.35),
            initial_pose_duration_s=5.0,
            initial_pose_max_velocity_rad_s=0.8,
            initial_pose_max_acceleration_rad_s2=5.0,
            kinematics=FaKinematicsConfig(
                model_path=FA_URDF_PATH,
                left_joint_names=FA_LEFT_ARM_JOINT_NAMES,
                right_joint_names=FA_RIGHT_ARM_JOINT_NAMES,
                left_endeff_body="left_hand_base_link",
                right_endeff_body="right_hand_base_link",
                max_iter=100,
                dls_damping=0.1,
            ),
            ik=FaArmIkConfig(
                urdf_file=FA_URDF_PATH,
                srdf_file=FA_SRDF_PATH,
                module_name="ik_7dof_pybind",
                reference_frame="pelvis",
                max_iters=50,
                max_joint_step_rad=1.0,
                eps=1e-3,
                skip_svd_fallback=True,
                position_weight=1.0,
                orientation_weight=1.0,
                acceptable_position_error_m=0.02,
                acceptable_orientation_error_rad=0.05,
                continuity_nullspace_weight=0.01,
                comfort_nullspace_log_enabled=True,
                comfort_nullspace_weight=0.04,
                comfort_left_arm_positions_rad=(-1.0, 0.5, -0.9, -0.9, -0.5, 0.0, 0.0),
                comfort_right_arm_positions_rad=(-1.0, -0.5, 0.9, -0.9, 0.5, 0.0, 0.0),
            ),
        )
    )

    def _apply_backend_overrides(self):
        self.config.control_backend = self.control_backend
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
    moving_average_limit: int = 1
    arm_resolution_port: int = ports.KEYPOINT_STREAM_PORT
    use_filter: bool = False
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    hand_frame_timeout_s: float = 1.0
    rotation_delta_frame: str = "base"
    translation_scale: float = 1.0
    high_resolution_translation_scale: float = 1.0
    low_resolution_translation_scale: float = 1.0
    post_resume_stable_position_epsilon_m: float = 0.03
    post_resume_stable_orientation_epsilon_rad: float = 0.20
    post_resume_stable_dwell_s: float = 0.2
    right_transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT
    left_transformed_keypoints_port: int = ports.LEFT_KEYPOINT_TRANSFORM_PORT
    enable_vr_axis_calibration: bool = True
    require_calibration_each_enable: bool = True
    calibration_sample_duration_s: float = 0.4
    calibration_stable_dwell_s: float = 0.3
    calibration_stable_position_epsilon_m: float = 0.02
    calibration_ready_return_position_epsilon_m: float = 0.04
    calibration_ready_return_dwell_s: float = 0.20
    calibration_min_motion_distance_m: float = 0.06
    calibration_max_motion_distance_m: float = 0.60
    calibration_max_axis_abs_dot: float = 0.35
    calibration_max_left_right_direction_error_deg: float = 20.0
    calibration_max_left_right_distance_ratio_error: float = 0.35
    calibration_rotation_orthogonality_tolerance: float = 1e-4
    calibration_rotation_determinant_tolerance: float = 1e-4
    calibration_max_timestamp_skew_s: float = 0.15
    calibration_max_frame_age_s: float = 1.0
    tracking_origin_jump_detection_enabled: bool = True
    tracking_origin_jump_translation_m: float = 0.15
    tracking_origin_jump_rotation_deg: float = 15.0
    tracking_origin_jump_confirm_frames: int = 2
    tracking_origin_jump_interhand_change_m: float = 0.04
    calibration_audio_enabled: bool = True
    calibration_prompt_port: int = ports.FA_CALIBRATION_PROMPT_PORT
    calibration_prompt_topic: str = "fa_calibration_prompt"
    calibration_audio_cooldown_s: float = 2.5
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
            translation_scale=self.translation_scale,
            high_resolution_translation_scale=self.high_resolution_translation_scale,
            low_resolution_translation_scale=self.low_resolution_translation_scale,
            post_resume_stable_position_epsilon_m=self.post_resume_stable_position_epsilon_m,
            post_resume_stable_orientation_epsilon_rad=self.post_resume_stable_orientation_epsilon_rad,
            post_resume_stable_dwell_s=self.post_resume_stable_dwell_s,
            h_r_v=H_R_V_FA,
            right_transformed_keypoints_port=self.right_transformed_keypoints_port,
            left_transformed_keypoints_port=self.left_transformed_keypoints_port,
            enable_vr_axis_calibration=self.enable_vr_axis_calibration,
            require_calibration_each_enable=self.require_calibration_each_enable,
            calibration_sample_duration_s=self.calibration_sample_duration_s,
            calibration_stable_dwell_s=self.calibration_stable_dwell_s,
            calibration_stable_position_epsilon_m=self.calibration_stable_position_epsilon_m,
            calibration_ready_return_position_epsilon_m=self.calibration_ready_return_position_epsilon_m,
            calibration_ready_return_dwell_s=self.calibration_ready_return_dwell_s,
            calibration_min_motion_distance_m=self.calibration_min_motion_distance_m,
            calibration_max_motion_distance_m=self.calibration_max_motion_distance_m,
            calibration_max_axis_abs_dot=self.calibration_max_axis_abs_dot,
            calibration_max_left_right_direction_error_deg=self.calibration_max_left_right_direction_error_deg,
            calibration_max_left_right_distance_ratio_error=self.calibration_max_left_right_distance_ratio_error,
            calibration_rotation_orthogonality_tolerance=self.calibration_rotation_orthogonality_tolerance,
            calibration_rotation_determinant_tolerance=self.calibration_rotation_determinant_tolerance,
            calibration_max_timestamp_skew_s=self.calibration_max_timestamp_skew_s,
            calibration_max_frame_age_s=self.calibration_max_frame_age_s,
            tracking_origin_jump_detection_enabled=self.tracking_origin_jump_detection_enabled,
            tracking_origin_jump_translation_m=self.tracking_origin_jump_translation_m,
            tracking_origin_jump_rotation_deg=self.tracking_origin_jump_rotation_deg,
            tracking_origin_jump_confirm_frames=self.tracking_origin_jump_confirm_frames,
            tracking_origin_jump_interhand_change_m=self.tracking_origin_jump_interhand_change_m,
            calibration_audio_enabled=self.calibration_audio_enabled,
            calibration_prompt_port=self.calibration_prompt_port,
            calibration_prompt_topic=self.calibration_prompt_topic,
            calibration_audio_cooldown_s=self.calibration_audio_cooldown_s,
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
                    moving_average_limit=1,
                )
            )
        if self.laterality in (Laterality.LEFT, Laterality.BIMANUAL):
            self.transforms.append(
                SharedComponentRegistry.get_transform_config(
                    hand_side=robots.LEFT,
                    host=network.HOST_ADDRESS,
                    keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                    moving_average_limit=1,
                )
            )

        self.camera_streamers = []
        if self.control_backend == "real" and os.environ.get("FA_ENABLE_REAL_CAMERA", "0") == "1":
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
                    enable_vr_axis_calibration=self.laterality == Laterality.BIMANUAL,
                    calibration_audio_enabled=self.laterality == Laterality.BIMANUAL,
                    right_transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                    left_transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                    logging_config={"enabled": False, "log_dir": "logs", "log_poses": True, "log_prefix": "fa_right"},
                )
            )
        if self.laterality in (Laterality.LEFT, Laterality.BIMANUAL):
            self.operators.append(
                FaOperatorCfg(
                    transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                    endeff_publish_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    endeff_subscribe_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    hand_side=robots.LEFT,
                    enable_vr_axis_calibration=self.laterality == Laterality.BIMANUAL,
                    calibration_audio_enabled=False,
                    right_transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                    left_transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                    logging_config={"enabled": False, "log_dir": "logs", "log_poses": True, "log_prefix": "fa_left"},
                )
            )

        self.environment = []
        if self.control_backend in ("mujoco", "real_with_mujoco"):
            self.environment.append(
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
                    arm_command_source="none",
                    ros_arm_command_topic="",
                    subscribe_min_snap_target=True,
                    min_snap_target_topic="/min_snap/target",
                    publish_joint_states=self.control_backend == "mujoco",
                    joint_state_topic="/joint_states",
                    joint_state_publish_hz=50.0,
                    arm_command_interpolation_steps=5,
                    interpolation_profile="min_snap",
                    expected_command_length=FA_UPPER_COMMAND_LENGTH,
                    joint_state_joint_names=FA_LEFT_ARM_JOINT_NAMES + FA_RIGHT_ARM_JOINT_NAMES,
                )
            )

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
