"""SYSMO-32 MuJoCo simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from beavr.teleop.configs.constants import network, ports

# Keep these offsets aligned with sysmo32_config.py.
SYSMO32_RIGHT_PORT_OFFSET = 2
SYSMO32_LEFT_PORT_OFFSET = 4


@dataclass
class MuJoCoSimConfig:
    """Existing high-level CartesianTarget MuJoCo simulator config."""

    host: str = network.HOST_ADDRESS
    urdf_path: str = "robots/sysmo_description/urdf/sysmo32.urdf"
    right_endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET
    simulation_mode: bool = True
    render: bool = True

    def build(self):
        from beavr.teleop.components.simulation.mujoco_sim import MuJoCoSysmoSimulator

        return MuJoCoSysmoSimulator(
            host=self.host,
            right_endeff_subscribe_port=self.right_endeff_subscribe_port,
            left_endeff_subscribe_port=self.left_endeff_subscribe_port,
            urdf_path=self.urdf_path,
            simulation_mode=self.simulation_mode,
            render=self.render,
        )


@dataclass
class Sysmo32MujocoCommandMirrorCfg:
    """MuJoCo mirror for the real 18-field SYSMO-32 arm command."""

    host: str = network.HOST_ADDRESS
    arm_command_port: int = ports.SYSMO32_ARM_COMMAND_MIRROR_PORT
    hand_action_port: int = ports.SYSMO32_HAND_ACTION_MIRROR_PORT
    urdf_path: str = "robots/sysmo_description/urdf/sysmo32.urdf"
    control_dt: float = 0.01
    render: bool = True
    load_model: bool = True
    print_hand_action_only: bool = True
    arm_command_source: str = "zmq"
    ros_arm_command_topic: str = "/sysmo_left_arm_controller/commands"
    publish_joint_states: bool = False
    joint_state_topic: str = "/joint_states"
    joint_state_publish_hz: float = 50.0
    subscribe_min_snap_target: bool = False
    min_snap_target_topic: str = "/min_snap/target"
    arm_command_interpolation_steps: int = 5
    interpolation_profile: str = "quintic"
    expected_command_length: int = 18
    joint_state_joint_names: Optional[tuple[str, ...]] = None
    kinematics_type: str = "sysmo32"

    def build(self):
        from beavr.teleop.components.simulation.sysmo32_mujoco_command_sim import (
            Sysmo32MujocoCommandMirror,
        )

        return Sysmo32MujocoCommandMirror(
            host=self.host,
            arm_command_port=self.arm_command_port,
            hand_action_port=self.hand_action_port,
            urdf_path=self.urdf_path,
            control_dt=self.control_dt,
            render=self.render,
            load_model=self.load_model,
            print_hand_action_only=self.print_hand_action_only,
            arm_command_source=self.arm_command_source,
            ros_arm_command_topic=self.ros_arm_command_topic,
            publish_joint_states=self.publish_joint_states,
            joint_state_topic=self.joint_state_topic,
            joint_state_publish_hz=self.joint_state_publish_hz,
            subscribe_min_snap_target=self.subscribe_min_snap_target,
            min_snap_target_topic=self.min_snap_target_topic,
            arm_command_interpolation_steps=self.arm_command_interpolation_steps,
            interpolation_profile=self.interpolation_profile,
            expected_command_length=self.expected_command_length,
            joint_state_joint_names=self.joint_state_joint_names,
            kinematics_type=self.kinematics_type,
        )


__all__ = ["MuJoCoSimConfig", "Sysmo32MujocoCommandMirrorCfg"]
