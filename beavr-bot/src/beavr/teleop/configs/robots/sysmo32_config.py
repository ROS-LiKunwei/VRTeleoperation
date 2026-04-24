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
        - state_publish: 10019 (XARM_STATE_PUBLISH_PORT + 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from beavr.teleop.common.configs.loader import Laterality, log_laterality_configuration
from beavr.teleop.components.interface.robots.sysmo32_robot import Sysmo32Robot
from beavr.teleop.components.operator.robots.sysmo32_operator import Sysmo32Operator
from beavr.teleop.configs.constants import network, ports, robots
from beavr.teleop.configs.robots import TeleopRobotConfig
from beavr.teleop.configs.robots.shared_components import SharedComponentRegistry
from beavr.teleop.configs.robots.sysmo_mujoco_config import MuJoCoSimConfig

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
            endeff_publish_port=self.endeff_publish_port,
            endeff_subscribe_port=self.endeff_subscribe_port,
            joint_subscribe_port=self.joint_subscribe_port,
            reset_subscribe_port=self.reset_subscribe_port,
            state_publish_port=self.state_publish_port,
            home_subscribe_port=self.home_subscribe_port,
            teleoperation_state_port=self.teleoperation_state_port,
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

    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    environment: list = field(default_factory=list)

    def __post_init__(self):
        log_laterality_configuration(self.laterality, ROBOT_NAME_SYSMO32)
        self._configure_for_laterality()

    def _configure_for_laterality(self):
        """根据laterality配置所有组件"""

        # Detector配置
        self.detector = []
        if self.laterality == Laterality.BIMANUAL:
            self.detector.append(
                SharedComponentRegistry.get_bimanual_detector_config(
                    host=network.HOST_ADDRESS,
                )
            )
        else:
            hand_side = robots.RIGHT if self.laterality == Laterality.RIGHT else robots.LEFT
            self.detector.append(
                SharedComponentRegistry.get_detector_config(
                    hand_side=hand_side,
                    host=network.HOST_ADDRESS,
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

        # Robot配置
        self.robots = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.robots.append(
                Sysmo32RobotCfg(
                    host=network.HOST_ADDRESS,
                    is_right_arm=True,
                    endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    reset_subscribe_port=ports.XARM_RESET_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    state_publish_port=ports.XARM_STATE_PUBLISH_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                    home_subscribe_port=ports.XARM_HOME_SUBSCRIBE_PORT,
                    teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                    hand_side=robots.RIGHT,
                    simulation_mode=True,
                    recorder_config={
                        "robot_identifier": "right_sysmo32",
                        "recorded_data": [
                            robots.RECORDED_DATA_JOINT_STATES,
                            robots.RECORDED_DATA_XARM_CARTESIAN_STATES,
                            robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                            robots.RECORDED_DATA_JOINT_ANGLES_RAD,
                        ],
                    },
                )
            )

        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.robots.append(
                Sysmo32RobotCfg(
                    host=network.HOST_ADDRESS,
                    is_right_arm=False,
                    endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    reset_subscribe_port=ports.XARM_RESET_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    state_publish_port=ports.XARM_STATE_PUBLISH_PORT + SYSMO32_LEFT_PORT_OFFSET,
                    home_subscribe_port=ports.XARM_HOME_SUBSCRIBE_PORT,
                    teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                    hand_side=robots.LEFT,
                    simulation_mode=True,
                    recorder_config={
                        "robot_identifier": "left_sysmo32",
                        "recorded_data": [
                            robots.RECORDED_DATA_JOINT_STATES,
                            robots.RECORDED_DATA_XARM_CARTESIAN_STATES,
                            robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                            robots.RECORDED_DATA_JOINT_ANGLES_RAD,
                        ],
                    },
                )
            )

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

        # Environment配置（MuJoCo仿真）
        self.environment = [
            MuJoCoSimConfig(
                host=network.HOST_ADDRESS,
                urdf_path="configs/robots/sysmo32.urdf",
                right_endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET,
                left_endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET,
                render=True,
                simulation_mode=True,
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
        }
