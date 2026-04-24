"""
SYSMO-32 MuJoCo仿真配置模块

定义MuJoCo仿真环境的配置参数,包括URDF路径、端口配置等。

端口映射说明：
    右手Sysmo32Operator发布CartesianTarget到XARM_ENDEFF_SUBSCRIBE_PORT+2(10011)
    左手Sysmo32Operator发布CartesianTarget到XARM_ENDEFF_SUBSCRIBE_PORT+4(10013)
    MuJoCo仿真器分别订阅这两个端口,接收左右手的末端执行器目标

与sysmo32_config.py中的端口偏移保持一致:
    SYSMO32_RIGHT_PORT_OFFSET = 2
    SYSMO32_LEFT_PORT_OFFSET = 4
"""

from dataclasses import dataclass, field
from typing import Optional

from beavr.teleop.configs.constants import network, ports

# 与sysmo32_config.py中的端口偏移保持一致
SYSMO32_RIGHT_PORT_OFFSET = 2
SYSMO32_LEFT_PORT_OFFSET = 4


@dataclass
class MuJoCoSimConfig:
    """MuJoCo仿真环境配置"""
    host: str = network.HOST_ADDRESS
    urdf_path: str = "configs/robots/sysmo32.urdf"
    right_endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_RIGHT_PORT_OFFSET
    left_endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT + SYSMO32_LEFT_PORT_OFFSET
    render: bool = True
    simulation_mode: bool = True

    def build(self):
        """构建MuJoCo仿真器实例"""
        from beavr.teleop.components.simulation.mujoco_sim import MuJoCoSysmoSimulator

        return MuJoCoSysmoSimulator(
            host=self.host,
            right_endeff_subscribe_port=self.right_endeff_subscribe_port,
            left_endeff_subscribe_port=self.left_endeff_subscribe_port,
            urdf_path=self.urdf_path,
            simulation_mode=self.simulation_mode,
            render=self.render,
        )
