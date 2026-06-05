"""SYSMO-32 MuJoCo simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass

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
        )


__all__ = ["MuJoCoSimConfig", "Sysmo32MujocoCommandMirrorCfg"]
