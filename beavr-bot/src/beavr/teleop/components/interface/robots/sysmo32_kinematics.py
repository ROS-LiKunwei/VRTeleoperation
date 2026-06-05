"""SYSMO-32 FK/IK helper based on the local MuJoCo model."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from beavr.teleop.components.interface.robots.sysmo32_command import (
    SYSMO32_LEFT_JOINT_NAMES,
    SYSMO32_RIGHT_JOINT_NAMES,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class Sysmo32MujocoKinematics:
    """Small FK/IK wrapper for SYSMO-32.
        真实机器人重置路径需要从“/joint_states”获取正向运动学(FK)数据。
        同时,还需要使用反向运动学(IK)将“Sysmo32Operator”发出的笛卡尔目标转换为每条手臂的六个指令关节。
    """

    LEFT_ENDEFF_SITE = "left_endeff"
    RIGHT_ENDEFF_SITE = "right_endeff"

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.available = False
        self._mujoco = None
        self.model = None
        self.data = None
        self.left_joint_ids = []
        self.right_joint_ids = []
        self.left_site_id = -1
        self.right_site_id = -1
        self._load_model()

    def _load_model(self) -> None:
        try:
            import mujoco

            self._mujoco = mujoco
            path = Path(self.urdf_path)
            if not path.is_absolute():
                repo_root = Path(__file__).resolve().parents[6] # 向上找6层到仓库根目录
                path = repo_root / self.urdf_path
            if not path.exists():
                raise FileNotFoundError(f"SYSMO-32 URDF not found: {path}")
            
            # 将 URDF 中相对路径的 STL 网格文件转换为绝对路径。
            source_model = mujoco.MjModel.from_xml_path(str(path))
            xml_path = "/tmp/sysmo32_kinematics.xml"
            mujoco.mj_saveLastXML(xml_path, source_model)
            xml_string = Path(xml_path).read_text()
            xml_string = self._resolve_exported_mesh_paths(xml_string, path.parent)
            # 在左右臂的末端连杆上各插入一个虚拟位点（site），用于计算末端位姿。
            xml_string = self._insert_site_into_body(
                xml_string,
                "left_arm_J6_Link",
                '<site name="left_endeff" pos="0 0.05 0" rgba="0 1 0 1" size="0.02"/>',
            )
            xml_string = self._insert_site_into_body(
                xml_string,
                "right_arm_J6_Link",
                '<site name="right_endeff" pos="0 -0.05 0" rgba="1 0 0 1" size="0.02"/>',
            )

            self.model = mujoco.MjModel.from_xml_string(xml_string)
            self.data = mujoco.MjData(self.model)
            self.left_joint_ids = self._joint_ids(SYSMO32_LEFT_JOINT_NAMES)
            self.right_joint_ids = self._joint_ids(SYSMO32_RIGHT_JOINT_NAMES)
            self.left_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.LEFT_ENDEFF_SITE)
            self.right_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.RIGHT_ENDEFF_SITE)
            self.available = (
                len(self.left_joint_ids) == 6
                and len(self.right_joint_ids) == 6
                and self.left_site_id >= 0
                and self.right_site_id >= 0
            )
            if self.available:
                logger.info("SYSMO-32 MuJoCo FK/IK model loaded from %s", path)
            else:
                logger.warning("SYSMO-32 MuJoCo FK/IK model loaded incompletely; FK/IK disabled")
        except Exception as exc:
            self.available = False
            logger.warning("SYSMO-32 MuJoCo FK/IK unavailable: %s", exc)

    def _resolve_exported_mesh_paths(self, xml_string: str, urdf_dir: Path) -> str:
        def resolve_mesh_file(match):
            mesh_file = match.group(1)
            # 提取文件名，如 "meshes/left_arm.stl"
            mesh_path = Path(mesh_file)
            if not mesh_path.is_absolute():
                # 如果是相对路径，转换为绝对路径
                mesh_path = (urdf_dir / mesh_path).resolve()
            return f'file="{mesh_path}"'
        # 使用正则表达式查找并替换所有 STL 文件引用
        return re.sub(r'file="([^"]+\.(?:STL|stl))"', resolve_mesh_file, xml_string)

    # 向 MuJoCo XML 模型中的指定连杆（body）插入一个位点（site）元素
    def _insert_site_into_body(self, xml_string: str, body_name: str, site_xml: str) -> str:
        # 构建正则表达式，匹配目标 body 标签
        pattern = rf'(<body name="{re.escape(body_name)}"[^>]*>)'
        # 定义替换模板：保留原 body 标签，后面追加 site 元素
        replacement = rf"\1\n        {site_xml}"
        # 执行替换（只替换第一个匹配项）
        updated_xml, count = re.subn(pattern, replacement, xml_string, count=1)
        # 验证是否成功找到并替换
        if count != 1:
            raise ValueError(f"MuJoCo body not found for site insertion: {body_name}")
        return updated_xml

    def _joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        ids = []
        for name in joint_names:
            # 查询关节名称对应的 ID
            joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            # 检查是否找到（ID >= 0 表示找到，< 0 表示未找到）
            if joint_id < 0:
                logger.warning("SYSMO-32 joint not found in MuJoCo model: %s", name)
            else:
                ids.append(joint_id)
        return ids

    def fk(self, hand_side: str, all_joints_rad: Sequence[float]) -> Optional[np.ndarray]:
        # 检查模型是否可用
        if not self.available:
            return None
        joints = np.asarray(all_joints_rad, dtype=np.float64)
        if joints.shape != (12,) or not np.all(np.isfinite(joints)):
            return None
        # 设置 MuJoCo 模型中的关节角度
        self._set_all_joints(joints)
        self._mujoco.mj_forward(self.model, self.data)
        site_id = self.left_site_id if hand_side == robots.LEFT else self.right_site_id
        # 获取末端位点的位置和姿态
        pos = self.data.site_xpos[site_id].copy()
        mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        homo = np.eye(4)
        homo[:3, :3] = mat
        homo[:3, 3] = pos
        return homo

    def solve_ik(
        self,
        hand_side: str,
        target: CartesianTarget,
        all_joints_rad: Sequence[float],
        max_iter: int = 80,
    ) -> Optional[np.ndarray]:
        if not self.available:
            return None
        joints = np.asarray(all_joints_rad, dtype=np.float64)
        if joints.shape != (12,) or not np.all(np.isfinite(joints)):
            return None
        # 把目标位姿转换成 MuJoCo 格式 [pos + w x y z]
        target_pos, target_quat = self._target_to_mujoco(target)
        joint_ids = self.left_joint_ids if hand_side == robots.LEFT else self.right_joint_ids
        site_id = self.left_site_id if hand_side == robots.LEFT else self.right_site_id
        arm_slice = slice(0, 6) if hand_side == robots.LEFT else slice(6, 12)
        # 把当前关节角写入 MuJoCo,IK从当前关节状态开始迭代，而不是从零位开始
        self._set_all_joints(joints)

        qpos_addrs = [self.model.jnt_qposadr[jid] for jid in joint_ids] # 表示某个 joint 在 data.qpos 里的地址
        dof_addrs = [self.model.jnt_dofadr[jid] for jid in joint_ids]   # 表示某个 joint 在速度空间 data.qvel / Jacobian 列里的地址
        
        best_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs]) # 保存当前这条手臂的 6 个关节角作为初始最优解
        best_err = np.inf

        for _ in range(max_iter):
            self._mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[site_id].copy() # site_xpos[site_id]是末端在世界坐标系下的位置
            current_mat = self.data.site_xmat[site_id].reshape(3, 3).copy() # site_xmat[site_id]是末端在世界坐标系下的旋转矩阵
            current_quat = np.zeros(4)
            self._mujoco.mju_mat2Quat(current_quat, current_mat.flatten())

            pos_error = target_pos - current_pos
            inv_current = np.zeros(4)
            quat_error = np.zeros(4)
            self._mujoco.mju_negQuat(inv_current, current_quat)
            self._mujoco.mju_mulQuat(quat_error, target_quat, inv_current)# 四元数误差
            axis_angle = np.zeros(3)
            self._mujoco.mju_quat2Vel(axis_angle, quat_error, 1.0) # 四元数误差转轴角速度形式

            err_norm = float(np.linalg.norm(pos_error) + 0.25 * np.linalg.norm(axis_angle))
            # 保存当前最优解
            if err_norm < best_err:
                best_err = err_norm
                best_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs])
            # 判断是否满足停止条件
            if np.linalg.norm(pos_error) < 1e-3 and np.linalg.norm(axis_angle) < 0.05:
                break
            
            # 计算末端位姿的Jacobian矩阵
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id) # jacp为位置 Jacobian，jacr为旋转矩阵 Jacobian
            jac = np.vstack([jacp, 0.25 * jacr])[:, dof_addrs] # 合并位置和旋转 Jacobian
            error = np.concatenate([pos_error, 0.25 * axis_angle])
            try:
                delta = np.linalg.solve(jac.T @ jac + 0.01 * np.eye(6), jac.T @ error) # DLS
            except np.linalg.LinAlgError:
                return None
            delta_norm = np.linalg.norm(delta)
            # 限制单步关节增量
            if delta_norm > 0.08:
                delta *= 0.08 / delta_norm
            # 更新 MuJoCo qpos
            for idx, addr in enumerate(qpos_addrs):
                self.data.qpos[addr] += delta[idx]
            # 关节限位
            self._clip_joint_qpos(joint_ids, qpos_addrs)

        result = np.asarray(joints[arm_slice], dtype=np.float64).copy()
        result[:] = best_qpos
        if not np.all(np.isfinite(result)):
            return None
        return result
    # 保守的反向运动学降级方案 ，仅作为 MuJoCo IK 不可用时的备用策略，不用于真实机器人控制
    def placeholder_ik(self, hand_side: str, target: CartesianTarget, all_joints_rad: Sequence[float]) -> np.ndarray:
        """Conservative dry-run fallback, never used for real robot safety."""
        # 获取当前关节角度
        joints = np.asarray(all_joints_rad, dtype=np.float64)
        arm = joints[:6].copy() if hand_side == robots.LEFT else joints[6:12].copy()
        # 提取目标位置
        pos = np.asarray(target.position_m, dtype=np.float64)
        # 计算微小的关节增量（仅用 XYZ 位置，乘以 0.05 的缩放因子）
        delta = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0], dtype=np.float64) * 0.05
        # 添加增量并限制最大变化量（-0.02 ~ 0.02 rad）
        return arm + np.clip(delta, -0.02, 0.02)

    def _set_all_joints(self, joints: np.ndarray) -> None:
        # 遍历所有关节（左臂 + 右臂，共12个）
        for idx, joint_id in enumerate(self.left_joint_ids + self.right_joint_ids):
            # 获取该关节在 qpos 数组中的内存地址
            qpos_addr = self.model.jnt_qposadr[joint_id]
            # 将关节角度写入 data.qpos 数组中
            self.data.qpos[qpos_addr] = joints[idx]
        # 对所有关节角度进行限位裁剪    
        self._clip_joint_qpos(self.left_joint_ids + self.right_joint_ids, [
            self.model.jnt_qposadr[jid] for jid in self.left_joint_ids + self.right_joint_ids
        ])

    def _clip_joint_qpos(self, joint_ids: Sequence[int], qpos_addrs: Sequence[int]) -> None:
        for joint_id, qpos_addr in zip(joint_ids, qpos_addrs, strict=False):
            low, high = self.model.jnt_range[joint_id]
            self.data.qpos[qpos_addr] = np.clip(self.data.qpos[qpos_addr], low, high)

    def _target_to_mujoco(self, target: CartesianTarget) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(target.position_m, dtype=np.float64)
        quat_xyzw = np.asarray(target.orientation_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat_xyzw)
        if norm < 1e-9:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            quat_xyzw = quat_xyzw / norm
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        return pos, quat_wxyz


__all__ = ["Sysmo32MujocoKinematics"]
