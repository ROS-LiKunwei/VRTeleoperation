"""Simulation-oriented config for XArm7 robot (right/left/bimanual)."""

from __future__ import annotations

from dataclasses import dataclass, field

from beavr.teleop.common.configs.loader import Laterality
from beavr.teleop.components.interface.robots.xarm7_robot import XArm7Robot
from beavr.teleop.configs.constants import network, ports
from beavr.teleop.configs.robots import TeleopRobotConfig
from beavr.teleop.configs.robots.xarm7_config import XArm7Config


@dataclass
class XArm7SimRobotCfg:
    host: str = network.HOST_ADDRESS
    robot_ip: str = network.RIGHT_XARM_IP
    is_right_arm: bool = True
    simulation_mode: bool = True
    endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT
    endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT
    joint_subscribe_port: int = ports.XARM_JOINT_SUBSCRIBE_PORT
    reset_subscribe_port: int = ports.XARM_RESET_SUBSCRIBE_PORT
    state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT
    home_subscribe_port: int = ports.XARM_HOME_SUBSCRIBE_PORT
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT

    def build(self):
        return XArm7Robot(
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
@TeleopRobotConfig.register_subclass("xarm7_sim")
class XArm7SimConfig:
    robot_name: str = "xarm7_sim"
    laterality: Laterality = Laterality.BIMANUAL

    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    environment: list = field(default_factory=list)

    def __post_init__(self):
        # Reuse official xarm7 component graph and only replace robot interface with sim-enabled variant.
        base_cfg = XArm7Config(laterality=self.laterality)
        self.detector = base_cfg.detector
        self.transforms = base_cfg.transforms
        self.visualizers = base_cfg.visualizers
        self.operators = base_cfg.operators

        sim_robots = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            sim_robots.append(
                XArm7SimRobotCfg(
                    host=network.HOST_ADDRESS,
                    robot_ip=network.RIGHT_XARM_IP,
                    is_right_arm=True,
                    endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT,
                    endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT,
                    joint_subscribe_port=ports.XARM_JOINT_SUBSCRIBE_PORT,
                    reset_subscribe_port=ports.XARM_RESET_SUBSCRIBE_PORT,
                    state_publish_port=ports.XARM_STATE_PUBLISH_PORT,
                    home_subscribe_port=ports.XARM_HOME_SUBSCRIBE_PORT,
                    teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                )
            )

        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            sim_robots.append(
                XArm7SimRobotCfg(
                    host=network.HOST_ADDRESS,
                    robot_ip=network.LEFT_XARM_IP,
                    is_right_arm=False,
                    endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT + 2,
                    endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + 2,
                    joint_subscribe_port=ports.XARM_JOINT_SUBSCRIBE_PORT + 1,
                    reset_subscribe_port=ports.XARM_RESET_SUBSCRIBE_PORT + 2,
                    state_publish_port=ports.XARM_STATE_PUBLISH_PORT + 1,
                    home_subscribe_port=ports.XARM_HOME_SUBSCRIBE_PORT,
                    teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                )
            )

        self.robots = sim_robots

    def build(self):
        return {
            "robot_name": self.robot_name,
            "detector": [detector.build() for detector in self.detector],
            "transforms": [item.build() for item in self.transforms],
            "visualizers": [item.build() for item in self.visualizers],
            "operators": [item.build() for item in self.operators],
            "robots": [item.build() for item in self.robots],
            "environment": [item.build() if hasattr(item, "build") else item for item in self.environment],
        }
