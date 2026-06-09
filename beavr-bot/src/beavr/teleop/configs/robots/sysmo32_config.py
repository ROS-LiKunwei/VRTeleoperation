"""
SYSMO-32双臂机器人配置模块

支持右手、左手或双手(bimanual)模式的SYSMO-32遥操作配置。
SYSMO-32是6自由度双臂机器人,每臂6个旋转关节,共12个关节。

与XArm7配置的区别:
    - 使用Sysmo32Robot替代XArm7Robot
    - 使用MuJoCo仿真替代XArm SDK
    - 每臂6个关节(XArm7是7个)
    - 双臂共享同一个base_link
    - 端口偏移量不同,避免与XArm7冲突

端口映射(SYSMO-32专用,基于XARM端口偏移):
    右臂：
        - endeff_publish: 10012 (XARM_ENDEFF_PUBLISH_PORT + 2)
        - endeff_subscribe: 10011 (XARM_ENDEFF_SUBSCRIBE_PORT + 2)
        - state_publish: 10018 (XARM_STATE_PUBLISH_PORT + 2)
    左臂：
        - endeff_publish: 10014 (XARM_ENDEFF_PUBLISH_PORT + 4)
        - endeff_subscribe: 10013 (XARM_ENDEFF_SUBSCRIBE_PORT + 4)
        - state_publish: 10020 (XARM_STATE_PUBLISH_PORT + 4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from beavr.lerobot.common.robot_devices.cameras.configs import (
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from beavr.teleop.common.configs.loader import Laterality, log_laterality_configuration
from beavr.teleop.components.interface.robots.sysmo32_command import Sysmo32ArmSafetyConfig
from beavr.teleop.components.interface.robots.sysmo32_real_control import (
    Sysmo32HandConfig,
    Sysmo32RealControl,
    Sysmo32RealControlConfig,
    Sysmo32Ros2Topics,
)
from beavr.teleop.components.interface.robots.sysmo32_robot import Sysmo32Robot
from beavr.teleop.components.operator.robots.sysmo32_operator import Sysmo32Operator
from beavr.teleop.configs.constants import network, ports, robots
from beavr.teleop.configs.robots import TeleopRobotConfig
from beavr.teleop.configs.robots.shared_components import SharedComponentRegistry
from beavr.teleop.configs.robots.sysmo_mujoco_config import Sysmo32MujocoCommandMirrorCfg

logger = logging.getLogger(__name__)

# SYSMO-32端口偏移（避免与XArm7端口冲突）
SYSMO32_RIGHT_PORT_OFFSET = 2
SYSMO32_LEFT_PORT_OFFSET = 4


@dataclass
class Sysmo32RobotCfg:
    """SYSMO-32单臂机器人接口配置"""

    host: str = network.HOST_ADDRESS
    robot_ip: str = "127.0.0.1"
    is_right_arm: bool = True
    endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    joint_subscribe_port: int = ports.XARM_JOINT_SUBSCRIBE_PORT
    reset_subscribe_port: int = ports.XARM_RESET_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    home_subscribe_port: int = ports.XARM_HOME_SUBSCRIBE_PORT
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    hand_side: str = robots.RIGHT
    simulation_mode: bool = True
    enable_ros2_bridge: bool = False
    left_hand_topic: str = "/left_topic_to_hand"
    right_hand_topic: str = "/right_topic_to_hand"
    arm_command_topic: str = "/sysmo_left_arm_controller/commands"
    ik_urdf_path: str = "robots/sysmo_description/urdf/sysmo32.urdf"
    recorder_config: dict[str, Any] = field(
        default_factory=lambda: {
            "robot_identifier": "right_sysmo32",
            "recorded_data": [
                robots.RECORDED_DATA_JOINT_STATES,
                robots.RECORDED_DATA_XARM_CARTESIAN_STATES,
                robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                robots.RECORDED_DATA_JOINT_ANGLES_RAD,
            ],
        }
    )

    def build(self):
        return Sysmo32Robot(
            host=self.host,
            robot_ip=self.robot_ip,
            is_right_arm=self.is_right_arm,
            simulation_mode=self.simulation_mode,
            enable_ros2_bridge=self.enable_ros2_bridge,
            left_hand_topic=self.left_hand_topic,
            right_hand_topic=self.right_hand_topic,
            arm_command_topic=self.arm_command_topic,
            ik_urdf_path=self.ik_urdf_path,
            endeff_publish_port=self.endeff_publish_port,
            endeff_subscribe_port=self.endeff_subscribe_port,
            joint_subscribe_port=self.joint_subscribe_port,
            reset_subscribe_port=self.reset_subscribe_port,
            state_publish_port=self.state_publish_port,
            home_subscribe_port=self.home_subscribe_port,
            teleoperation_state_port=self.teleoperation_state_port,
        )


@dataclass
class Sysmo32RealControlCfg:
    """SYSMO-32真实接口控制层配置。

    单个组件同时处理左右臂，因为真实机械臂接口是一条18维双臂命令。
    """

    host: str = network.HOST_ADDRESS  # 网络通信配置
    control_backend: str = "mujoco"  # 控制后端：real/mujoco/real_with_mujoco
    # ZMQ 端口配置（用于接收目标、发布状态等）
    right_target_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_target_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET
    right_endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET
    right_state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    transformed_right_port: int = ports.KEYPOINT_TRANSFORM_PORT
    transformed_left_port: int = ports.LEFT_KEYPOINT_TRANSFORM_PORT
    arm_command_mirror_port: int = ports.SYSMO32_ARM_COMMAND_MIRROR_PORT
    hand_action_mirror_port: int = ports.SYSMO32_HAND_ACTION_MIRROR_PORT
    # URDF 模型路径
    urdf_path: str = "robots/sysmo_description/urdf/sysmo32.urdf"
    # 嵌套配置对象
    config: Sysmo32RealControlConfig = field(
        default_factory=lambda: Sysmo32RealControlConfig(
            control_backend="mujoco",
            ros2=Sysmo32Ros2Topics(
                joint_state_topic="/joint_states",
                arm_command_topic="/sysmo_left_arm_controller/commands",
                left_hand_topic="/left_topic_to_hand",
                right_hand_topic="/right_topic_to_hand",
                arm_command_queue_size=60,
                hand_command_queue_size=10,
                joint_state_timeout_s=0.5,
            ),
            arm=Sysmo32ArmSafetyConfig(
                speed_mode=0.0,
                reserved=(0.0, 0.0, 0.0, 0.0),
                neck_joint=0.0,
                max_joint_velocity_rad_s=tuple([0.8] * 12),
                max_translation_step_m=0.02,
                max_rotation_step_rad=0.08,
            ),
            hand=Sysmo32HandConfig(
                default_action=1,
                grasp_action=2,
                publish_on_change_only=True,
                heartbeat_hz=3.0,
                grasp_enter_threshold_m=0.035,
                grasp_exit_threshold_m=0.055,
                confirm_frames=3,
                force_release_on_pause=True,
                force_release_on_timeout=True,
            ),
            state_publish_fps=30.0,
            hand_frame_timeout_s=0.3,
            safety_hold_arm_on_pause=True,
            pause_hold_heartbeat_hz=20.0,
            allow_placeholder_ik_for_mujoco=True,
            allow_mujoco_mirror_without_joint_state=True,
            mujoco_mirror_max_joint_velocity_rad_s=3.0,
            publish_arm_command_topic_in_mujoco=False,
        )
    )

    def __post_init__(self):
        self.config.control_backend = self.control_backend
        if self.control_backend == "mujoco":
            self.config.publish_arm_command_topic_in_mujoco = True
        if self.control_backend == "real_with_mujoco":
            self.config.allow_mujoco_mirror_without_joint_state = False

    def build(self):
        self.config.control_backend = self.control_backend
        if self.control_backend == "mujoco":
            self.config.publish_arm_command_topic_in_mujoco = True
        if self.control_backend == "real_with_mujoco":
            self.config.allow_mujoco_mirror_without_joint_state = False
        return Sysmo32RealControl(
            host=self.host,
            control_backend=self.control_backend,
            right_target_port=self.right_target_port,
            left_target_port=self.left_target_port,
            right_endeff_publish_port=self.right_endeff_publish_port,
            left_endeff_publish_port=self.left_endeff_publish_port,
            right_state_publish_port=self.right_state_publish_port,
            left_state_publish_port=self.left_state_publish_port,
            teleoperation_state_port=self.teleoperation_state_port,
            transformed_right_port=self.transformed_right_port,
            transformed_left_port=self.transformed_left_port,
            arm_command_mirror_port=self.arm_command_mirror_port,
            hand_action_mirror_port=self.hand_action_mirror_port,
            urdf_path=self.urdf_path,
            config=self.config,
        )


@dataclass
class Sysmo32OperatorCfg:
    """SYSMO-32单臂Operator配置"""

    host: str = network.HOST_ADDRESS
    transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT
    stream_configs: dict[str, Any] = field(
        default_factory=lambda: {
            "host": network.HOST_ADDRESS,
            "port": ports.CONTROL_STREAM_PORT,
        }
    )
    stream_oculus: bool = True
    endeff_publish_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    endeff_subscribe_port: int = ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET
    moving_average_limit: int = 3
    arm_resolution_port: int = ports.KEYPOINT_STREAM_PORT
    use_filter: bool = False
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    hand_frame_timeout_s: float = 0.3
    rotation_delta_frame: str = "base"
    logging_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "log_dir": "logs",
            "log_poses": True,
            "log_prefix": "sysmo32",
        }
    )
    hand_side: str = robots.RIGHT

    def build(self):
        return Sysmo32Operator(
            operator_name=f"sysmo32_{self.hand_side}_operator",
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
        )


@dataclass
class Sysmo32RealCameraStreamerCfg:
    """Publish a real robot camera feed to the VR app image stream."""

    host: str = network.HOST_ADDRESS
    port: int = ports.SIM_IMAGE_PORT
    camera_name: str = "front"
    camera_type: str = "opencv"
    camera_index: int | str = 0
    serial_number: int | None = None
    realsense_name: str | None = None
    fps: int = 30
    width: int = 640
    height: int = 480
    rotation: int | None = None
    color_mode: str = "rgb"

    def build(self):
        from beavr.teleop.components.camera.real_camera_streamer import RealCameraStreamer

        if self.camera_type == "opencv":
            camera_config = OpenCVCameraConfig(
                camera_index=self.camera_index,
                fps=self.fps,
                width=self.width,
                height=self.height,
                rotation=self.rotation,
                color_mode=self.color_mode,
            )
        elif self.camera_type == "intelrealsense":
            camera_config = IntelRealSenseCameraConfig(
                name=self.realsense_name,
                serial_number=self.serial_number,
                fps=self.fps,
                width=self.width,
                height=self.height,
                rotation=self.rotation,
                color_mode=self.color_mode,
            )
        else:
            raise ValueError(f"Unsupported camera_type: {self.camera_type}")

        return RealCameraStreamer(
            camera_config=camera_config,
            host=self.host,
            port=self.port,
            camera_name=self.camera_name,
        )


ROBOT_NAME_SYSMO32 = "sysmo32"


@dataclass
@TeleopRobotConfig.register_subclass(ROBOT_NAME_SYSMO32)
class Sysmo32Config:
    """
    SYSMO-32双臂机器人完整配置。

    支持右手、左手或双手模式。
    使用@TeleopRobotConfig.register_subclass装饰器注册，
    可通过 --robot_name=sysmo32 参数启动。
    """

    robot_name: str = ROBOT_NAME_SYSMO32
    laterality: Laterality = Laterality.BIMANUAL
    control_backend: str = "mujoco"

    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    environment: list = field(default_factory=list)
    camera_streamers: list = field(default_factory=list)

    def __post_init__(self):
        if self.control_backend not in ("real", "mujoco", "real_with_mujoco"):
            raise ValueError("sysmo32 control_backend must be one of: real, mujoco, real_with_mujoco")
        log_laterality_configuration(self.laterality, ROBOT_NAME_SYSMO32)
        self._configure_for_laterality()

    def _configure_for_laterality(self):
        """根据laterality配置所有组件"""

        # Detector配置 - 使用PICO4检测器
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

        # Transform配置
        self.transforms = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.transforms.append(
                SharedComponentRegistry.get_transform_config(
                    hand_side=robots.RIGHT,
                    host=network.HOST_ADDRESS,
                    keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                    moving_average_limit=3,
                )
            )

        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.transforms.append(
                SharedComponentRegistry.get_transform_config(
                    hand_side=robots.LEFT,
                    host=network.HOST_ADDRESS,
                    keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                    moving_average_limit=3,
                )
            )

        # MuJoCo dry-run/mirror should not require a physical camera.
        self.camera_streamers = []
        if self.control_backend == "real":
            self.camera_streamers = [
                Sysmo32RealCameraStreamerCfg(
                    host=network.HOST_ADDRESS,
                    port=ports.SIM_IMAGE_PORT,
                    camera_name="front",
                    camera_type="opencv",
                    camera_index=6,
                    fps=30,
                    width=640,
                    height=480,
                    rotation=180,
                )
            ]

        # Robot配置：sysmo32真实接口是一条双臂18维命令，因此只启动一个双臂控制层。
        self.robots = [
            Sysmo32RealControlCfg(
                host=network.HOST_ADDRESS,
                control_backend=self.control_backend,
                right_target_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                left_target_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET,
                right_endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                left_endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET,
                right_state_publish_port=ports.XARM_STATE_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                left_state_publish_port=ports.XARM_STATE_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET,
                teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
                transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                arm_command_mirror_port=ports.SYSMO32_ARM_COMMAND_MIRROR_PORT,
                hand_action_mirror_port=ports.SYSMO32_HAND_ACTION_MIRROR_PORT,
            )
        ]

        # Operator配置
        self.operators = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.operators.append(
                Sysmo32OperatorCfg(
                    host=network.HOST_ADDRESS,
                    transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                    stream_configs={
                        "host": network.HOST_ADDRESS,
                        "port": ports.CONTROL_STREAM_PORT,
                    },
                    stream_oculus=True,
                    endeff_publish_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    endeff_subscribe_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    moving_average_limit=3,
                    arm_resolution_port=ports.KEYPOINT_STREAM_PORT,
                    use_filter=False,
                    teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                    hand_side=robots.RIGHT,
                    logging_config={
                        "enabled": False,
                        "log_dir": "logs",
                        "log_poses": True,
                        "log_prefix": "sysmo32_right",
                    },
                )
            )

        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.operators.append(
                Sysmo32OperatorCfg(
                    host=network.HOST_ADDRESS,
                    transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                    stream_configs={
                        "host": network.HOST_ADDRESS,
                        "port": ports.CONTROL_STREAM_PORT,
                    },
                    stream_oculus=True,
                    endeff_publish_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    endeff_subscribe_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    moving_average_limit=3,
                    arm_resolution_port=ports.KEYPOINT_STREAM_PORT,
                    use_filter=False,
                    teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                    hand_side=robots.LEFT,
                    logging_config={
                        "enabled": False,
                        "log_dir": "logs",
                        "log_poses": True,
                        "log_prefix": "sysmo32_left",
                    },
                )
            )

        # Environment配置：独立MuJoCo层接收真实接口格式的18维命令。
        self.environment = []
        if self.control_backend in ("mujoco", "real_with_mujoco"):
            self.environment = [
                Sysmo32MujocoCommandMirrorCfg(
                    host=network.HOST_ADDRESS,
                    arm_command_port=ports.SYSMO32_ARM_COMMAND_MIRROR_PORT,
                    hand_action_port=ports.SYSMO32_HAND_ACTION_MIRROR_PORT,
                    urdf_path="robots/sysmo_description/urdf/sysmo32.urdf",
                    control_dt=0.01,
                    render=True,
                    load_model=True,
                    print_hand_action_only=True,
                    arm_command_source="ros2",
                    ros_arm_command_topic="/sysmo_left_arm_controller/commands",
                    publish_joint_states=self.control_backend == "mujoco",
                    joint_state_topic="/joint_states",
                    joint_state_publish_hz=50.0,
                    arm_command_interpolation_steps=5,
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
