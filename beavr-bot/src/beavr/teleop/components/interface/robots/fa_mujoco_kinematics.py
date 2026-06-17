"""FA FK/IK helper backed by the local FA MuJoCo/URDF model."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from beavr.teleop.components.interface.robots.sysmo32_kinematics import _damped_least_squares_delta
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


FA_LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
FA_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


@dataclass(frozen=True)
class FaKinematicsConfig:
    model_path: str = "robots/fa_description/urdf/fa_robot.urdf"
    left_joint_names: tuple[str, ...] = FA_LEFT_ARM_JOINT_NAMES
    right_joint_names: tuple[str, ...] = FA_RIGHT_ARM_JOINT_NAMES
    left_endeff_site: str = "left_fa_endeff"
    right_endeff_site: str = "right_fa_endeff"
    left_endeff_body: str = "left_hand_base_link"
    right_endeff_body: str = "right_hand_base_link"
    require_endeff: bool = True
    dls_damping: float = 0.1
    max_iter: int = 8
    orientation_weight: float = 0.2
    max_joint_step_rad: float = 0.08
    pos_tol_m: float = 1e-3
    ori_tol_rad: float = 2e-2


class FaMujocoKinematics:
    """Small FA FK/IK wrapper.

    The class is intentionally configuration-driven because the FA native ROS2
    control ABI and final end-effector frames still need hardware confirmation.
    """

    def __init__(self, config: FaKinematicsConfig | str | None = None, **kwargs):
        if isinstance(config, str):
            config = FaKinematicsConfig(model_path=config)
        elif config is None:
            config = FaKinematicsConfig(**kwargs)
        self.config = config
        self.available = False
        self.load_error: str | None = None
        self._mujoco = None
        self.model = None
        self.data = None
        self.left_joint_ids: list[int] = []
        self.right_joint_ids: list[int] = []
        self.left_site_id = -1
        self.right_site_id = -1
        self.left_body_id = -1
        self.right_body_id = -1
        self._joint_qpos_addrs: dict[str, list[int]] = {robots.LEFT: [], robots.RIGHT: []}
        self._joint_dof_addrs: dict[str, list[int]] = {robots.LEFT: [], robots.RIGHT: []}
        self._jacp = None
        self._jacr = None
        self._current_quat = np.zeros(4, dtype=np.float64)
        self._inv_current_quat = np.zeros(4, dtype=np.float64)
        self._quat_error = np.zeros(4, dtype=np.float64)
        self._axis_angle = np.zeros(3, dtype=np.float64)
        self._last_safe_target: dict[str, Optional[np.ndarray]] = {robots.LEFT: None, robots.RIGHT: None}
        self._load_model()

    @property
    def arm_joint_count(self) -> int:
        return len(self.config.left_joint_names)

    def configure_nullspace(self, **kwargs) -> None:
        if "max_iter" in kwargs and kwargs["max_iter"] is not None:
            self.config = dataclass_replace(self.config, max_iter=max(1, int(kwargs["max_iter"])))
        if "orientation_weight" in kwargs and kwargs["orientation_weight"] is not None:
            self.config = dataclass_replace(
                self.config, orientation_weight=max(0.0, float(kwargs["orientation_weight"]))
            )
        if "max_joint_step_rad" in kwargs and kwargs["max_joint_step_rad"] is not None:
            self.config = dataclass_replace(
                self.config, max_joint_step_rad=max(0.0, float(kwargs["max_joint_step_rad"]))
            )

    def _load_model(self) -> None:
        try:
            import mujoco

            self._mujoco = mujoco
            path = self._resolve_model_path(self.config.model_path)
            if not path.exists():
                raise FileNotFoundError(f"FA model file not found: {path}")
            xml_string = path.read_text()
            xml_string = self._resolve_mesh_paths(xml_string, path)
            xml_string = self._insert_endeff_sites(xml_string)
            self.model = mujoco.MjModel.from_xml_string(xml_string)
            self.data = mujoco.MjData(self.model)
            self.left_joint_ids = self._joint_ids(self.config.left_joint_names)
            self.right_joint_ids = self._joint_ids(self.config.right_joint_names)
            self.left_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.config.left_endeff_site
            )
            self.right_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.config.right_endeff_site
            )
            self.left_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.left_endeff_body
            )
            self.right_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.right_endeff_body
            )
            missing = []
            if len(self.left_joint_ids) != len(self.config.left_joint_names):
                missing.append("left arm joints")
            if len(self.right_joint_ids) != len(self.config.right_joint_names):
                missing.append("right arm joints")
            if self.config.require_endeff and self.left_site_id < 0 and self.left_body_id < 0:
                missing.append(
                    f"left end-effector site/body {self.config.left_endeff_site}/{self.config.left_endeff_body}"
                )
            if self.config.require_endeff and self.right_site_id < 0 and self.right_body_id < 0:
                missing.append(
                    f"right end-effector site/body {self.config.right_endeff_site}/{self.config.right_endeff_body}"
                )
            if missing:
                raise ValueError(f"FA model loaded but missing: {', '.join(missing)}")
            self._joint_qpos_addrs = {
                robots.LEFT: [self.model.jnt_qposadr[jid] for jid in self.left_joint_ids],
                robots.RIGHT: [self.model.jnt_qposadr[jid] for jid in self.right_joint_ids],
            }
            self._joint_dof_addrs = {
                robots.LEFT: [self.model.jnt_dofadr[jid] for jid in self.left_joint_ids],
                robots.RIGHT: [self.model.jnt_dofadr[jid] for jid in self.right_joint_ids],
            }
            self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            self.available = True
            logger.info("FA MuJoCo FK/IK model loaded from %s", path)
        except Exception as exc:
            self.available = False
            self.load_error = str(exc)
            logger.warning("FA MuJoCo FK/IK unavailable: %s", exc)

    def _resolve_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[6]
        return repo_root / path

    def _resolve_mesh_paths(self, xml_string: str, model_path: Path) -> str:
        fa_root = model_path.parents[1] if model_path.parent.name in ("urdf", "mjcf") else model_path.parent
        mesh_dir = fa_root / "meshes"

        def replace_package(match):
            return f'filename="{mesh_dir / match.group(1)}"'

        xml_string = re.sub(
            r'filename="package://(?:sysmo_description|fa_description)/meshes/([^"]+)"',
            replace_package,
            xml_string,
        )

        def replace_mjcf_file(match):
            mesh_file = Path(match.group(1))
            if not mesh_file.is_absolute():
                mesh_file = mesh_dir / mesh_file.name
            return f'file="{mesh_file}"'

        return re.sub(r'file="([^"]+\.(?:STL|stl))"', replace_mjcf_file, xml_string)

    def _insert_endeff_sites(self, xml_string: str) -> str:
        if "<mujoco" in xml_string:
            xml_string = self._insert_site_into_mjcf_body(
                xml_string,
                self.config.left_endeff_body,
                f'<site name="{self.config.left_endeff_site}" pos="0 0 0" size="0.015"/>',
            )
            return self._insert_site_into_mjcf_body(
                xml_string,
                self.config.right_endeff_body,
                f'<site name="{self.config.right_endeff_site}" pos="0 0 0" size="0.015"/>',
            )
        return xml_string

    def _insert_site_into_mjcf_body(self, xml_string: str, body_name: str, site_xml: str) -> str:
        pattern = rf'(<body\s+name\s*=\s*["\']{re.escape(body_name)}["\'][^>]*>)'
        updated, count = re.subn(pattern, rf"\1\n        {site_xml}", xml_string, count=1)
        if count != 1:
            raise ValueError(f"FA MuJoCo body not found for site insertion: {body_name}")
        return updated

    def _joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        ids = []
        for name in joint_names:
            joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                logger.warning("FA joint not found in MuJoCo model: %s", name)
            else:
                ids.append(joint_id)
        return ids

    def fk(self, hand_side: str, all_joints_rad: Sequence[float]) -> Optional[np.ndarray]:
        if not self.available:
            return None
        joints = np.asarray(all_joints_rad, dtype=np.float64)
        expected = len(self.config.left_joint_names) + len(self.config.right_joint_names)
        if joints.shape != (expected,) or not np.all(np.isfinite(joints)):
            return None
        self._set_all_joints(joints)
        self._mujoco.mj_forward(self.model, self.data)
        site_id, body_id = self._endeff_ids(hand_side)
        homo = np.eye(4)
        if site_id >= 0:
            homo[:3, 3] = self.data.site_xpos[site_id].copy()
            homo[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3).copy()
        else:
            homo[:3, 3] = self.data.xpos[body_id].copy()
            homo[:3, :3] = self.data.xmat[body_id].reshape(3, 3).copy()
        return homo

    def solve_ik(
        self,
        hand_side: str,
        target: CartesianTarget,
        all_joints_rad: Sequence[float],
        max_iter: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if not self.available:
            return self._last_safe_target[hand_side]
        joints = np.asarray(all_joints_rad, dtype=np.float64)
        expected = len(self.config.left_joint_names) + len(self.config.right_joint_names)
        if joints.shape != (expected,) or not np.all(np.isfinite(joints)):
            return self._last_safe_target[hand_side]
        target_pos, target_quat = self._target_to_mujoco(target)
        joint_ids = self.left_joint_ids if hand_side == robots.LEFT else self.right_joint_ids
        site_id, body_id = self._endeff_ids(hand_side)
        arm_slice = slice(0, self.arm_joint_count) if hand_side == robots.LEFT else slice(
            self.arm_joint_count, self.arm_joint_count * 2
        )
        qpos_addrs = self._joint_qpos_addrs[hand_side]
        dof_addrs = self._joint_dof_addrs[hand_side]
        self._set_all_joints(joints)
        best_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs])
        best_error = np.inf
        iterations = self.config.max_iter if max_iter is None else max(1, int(max_iter))
        for _ in range(iterations):
            self._mujoco.mj_forward(self.model, self.data)
            self._mujoco.mju_mat2Quat(
                self._current_quat,
                self._endeff_mat(site_id, body_id).flatten(),
            )
            pos_error = target_pos - self._endeff_pos(site_id, body_id)
            self._mujoco.mju_negQuat(self._inv_current_quat, self._current_quat)
            self._mujoco.mju_mulQuat(self._quat_error, target_quat, self._inv_current_quat)
            self._mujoco.mju_quat2Vel(self._axis_angle, self._quat_error, 1.0)
            task_error = float(
                np.linalg.norm(pos_error) + self.config.orientation_weight * np.linalg.norm(self._axis_angle)
            )
            current_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs])
            if task_error < best_error:
                best_error = task_error
                best_qpos = current_qpos.copy()
            if (
                np.linalg.norm(pos_error) < self.config.pos_tol_m
                and np.linalg.norm(self._axis_angle) < self.config.ori_tol_rad
            ):
                break
            self._jacp.fill(0.0)
            self._jacr.fill(0.0)
            if site_id >= 0:
                self._mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site_id)
            else:
                self._mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, body_id)
            jac = np.vstack([self._jacp, self.config.orientation_weight * self._jacr])[:, dof_addrs]
            error = np.concatenate([pos_error, self.config.orientation_weight * self._axis_angle])
            try:
                delta = _damped_least_squares_delta(
                    jac,
                    error,
                    current_qpos,
                    reference_qpos=None,
                    damping=self.config.dls_damping,
                    nullspace_gain=0.0,
                    nullspace_step_limit_rad=0.0,
                )
            except np.linalg.LinAlgError:
                return self._last_safe_target[hand_side]
            norm = float(np.linalg.norm(delta))
            if norm > self.config.max_joint_step_rad > 0.0:
                delta *= self.config.max_joint_step_rad / norm
            for idx, addr in enumerate(qpos_addrs):
                self.data.qpos[addr] += delta[idx]
            self._clip_joint_qpos(joint_ids, qpos_addrs)
        result = np.asarray(joints[arm_slice], dtype=np.float64).copy()
        result[:] = best_qpos
        if not np.all(np.isfinite(result)):
            return self._last_safe_target[hand_side]
        self._last_safe_target[hand_side] = result.copy()
        return result

    def _set_all_joints(self, joints: np.ndarray) -> None:
        for idx, joint_id in enumerate(self.left_joint_ids + self.right_joint_ids):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = joints[idx]
        self._clip_joint_qpos(
            self.left_joint_ids + self.right_joint_ids,
            [self.model.jnt_qposadr[jid] for jid in self.left_joint_ids + self.right_joint_ids],
        )

    def _clip_joint_qpos(self, joint_ids: Sequence[int], qpos_addrs: Sequence[int]) -> None:
        for joint_id, qpos_addr in zip(joint_ids, qpos_addrs, strict=False):
            low, high = self.model.jnt_range[joint_id]
            self.data.qpos[qpos_addr] = np.clip(self.data.qpos[qpos_addr], low, high)

    def _target_to_mujoco(self, target: CartesianTarget) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(target.position_m, dtype=np.float64)
        quat_xyzw = np.asarray(target.orientation_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat_xyzw)
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0]) if norm < 1e-9 else quat_xyzw / norm
        return pos, np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    def _endeff_ids(self, hand_side: str) -> tuple[int, int]:
        if hand_side == robots.LEFT:
            return self.left_site_id, self.left_body_id
        return self.right_site_id, self.right_body_id

    def _endeff_pos(self, site_id: int, body_id: int) -> np.ndarray:
        if site_id >= 0:
            return self.data.site_xpos[site_id]
        return self.data.xpos[body_id]

    def _endeff_mat(self, site_id: int, body_id: int) -> np.ndarray:
        if site_id >= 0:
            return self.data.site_xmat[site_id].reshape(3, 3)
        return self.data.xmat[body_id].reshape(3, 3)


def dataclass_replace(config: FaKinematicsConfig, **kwargs) -> FaKinematicsConfig:
    data = config.__dict__.copy()
    data.update(kwargs)
    return FaKinematicsConfig(**data)


__all__ = [
    "FA_LEFT_ARM_JOINT_NAMES",
    "FA_RIGHT_ARM_JOINT_NAMES",
    "FaKinematicsConfig",
    "FaMujocoKinematics",
]
