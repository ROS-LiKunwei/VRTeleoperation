""" sysmo32 专用 MuJoCo 镜像仿真层.

    与“MuJoCoSysmoSimulator”不同,此类不订阅“CartesianTarget”。
    它使用为真实手臂接口生成的精确18字段命令,并且仅应用左/右臂关节位置。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.components import Component
from beavr.teleop.components.interface.robots.sysmo32_command import (
    SYSMO32_COMMAND_LENGTH,
    SYSMO32_HAND_ACTION_GRASP,
    SYSMO32_HAND_ACTION_RELEASE,
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
    ):
        self.notify_component_start("sysmo32_mujoco_command_mirror")
        self.host = host
        self.control_dt = control_dt
        self.render = render
        self.print_hand_action_only = print_hand_action_only
        self._last_no_hand_action_log_time = 0.0
        self._last_arm_pose_log_time = 0.0
        self._arm_joint_ids = []
        self._arm_qpos_addrs = []
        self._arm_dof_addrs = []
        self._hold_joint_positions: Optional[np.ndarray] = None
        # 订阅真实机械臂命令字段
        self._arm_command_subscriber = ZMQSubscriber(
            host,
            arm_command_port,
            SYSMO32_ARM_COMMAND_TOPIC,
            message_type=Sysmo32ArmCommand,
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

        self._kinematics: Optional[Sysmo32MujocoKinematics] = None
        if load_model:
            # 加载真实机械臂模型
            self._kinematics = Sysmo32MujocoKinematics(urdf_path)
            if not self._kinematics.available:
                logger.warning(
                    "SYSMO-32 MuJoCo command mirror cannot load model; "
                    "falling back to command validation/logging only"
                )
            else:
                self._configure_arm_hold_state()

    def stream(self):
        if self._kinematics is None or not self._kinematics.available:
            logger.info("SYSMO-32 MuJoCo command mirror running without model; logging only")
            while True:
                self._receive_once()
                time.sleep(self.control_dt)

        import mujoco
        import mujoco.viewer

        logger.info("SYSMO-32 MuJoCo command mirror started")
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

        self._apply_arm_hold()
        mujoco_module.mj_forward(self._kinematics.model, self._kinematics.data)

    def _apply_arm_hold(self) -> None:
        if self._kinematics is None or self._hold_joint_positions is None:
            return
        for idx, qpos_addr in enumerate(self._arm_qpos_addrs):
            self._kinematics.data.qpos[qpos_addr] = self._hold_joint_positions[idx]
        for dof_addr in self._arm_dof_addrs:
            self._kinematics.data.qvel[dof_addr] = 0.0

    def _receive_once(self) -> None:
        command = self._arm_command_subscriber.recv_keypoints()
        if command is not None:
            self.apply_arm_command(command)

        left_action = self._left_hand_action_subscriber.recv_keypoints()
        if left_action is not None:
            self.on_left_hand_action(left_action.action_id)

        right_action = self._right_hand_action_subscriber.recv_keypoints()
        if right_action is not None:
            self.on_right_hand_action(right_action.action_id)

    def apply_arm_command(self, command: Sysmo32ArmCommand) -> None:
        values = np.asarray(command.values, dtype=np.float64)
        if values.shape != (SYSMO32_COMMAND_LENGTH,) or not np.all(np.isfinite(values)):
            logger.warning("[MuJoCo][ArmCommand] invalid command shape/value: %s", values)
            return
        if self._kinematics is None or not self._kinematics.available:
            logger.debug("[MuJoCo][ArmCommand] received valid command without model: %s", values)
            return

        hold = []
        for idx, joint_id in enumerate(self._arm_joint_ids):
            qpos_addr = self._kinematics.model.jnt_qposadr[joint_id]
            low, high = self._kinematics.model.jnt_range[joint_id]
            hold.append(float(np.clip(values[idx], low, high)))
        self._hold_joint_positions = np.asarray(hold, dtype=np.float64)
        self._apply_arm_hold()
        for joint_id in self._arm_joint_ids:
            dof_addr = self._kinematics.model.jnt_dofadr[joint_id]
            self._kinematics.data.qvel[dof_addr] = 0.0
        self._kinematics._mujoco.mj_forward(self._kinematics.model, self._kinematics.data)
        self._log_applied_arm_command(values)

    def _log_applied_arm_command(self, values: np.ndarray) -> None:
        now = time.time()
        if now - self._last_arm_pose_log_time >= 0.5:
            self._last_arm_pose_log_time = now
            left_site = self._kinematics.data.site_xpos[self._kinematics.left_site_id].copy()
            right_site = self._kinematics.data.site_xpos[self._kinematics.right_site_id].copy()
            logger.debug(
                "[MuJoCo][ArmCommand] applied left=%s right=%s left_site=%s right_site=%s",
                values[:6],
                values[6:12],
                left_site,
                right_site,
            )
            return
        logger.debug("[MuJoCo][ArmCommand] applied left=%s right=%s", values[:6], values[6:12])

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
        cleanup_zmq_resources()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = ["Sysmo32MujocoCommandMirror"]
