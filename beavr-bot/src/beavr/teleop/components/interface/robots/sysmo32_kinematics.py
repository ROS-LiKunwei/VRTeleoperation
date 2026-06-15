"""SYSMO-32 FK/IK helper based on the local MuJoCo model."""

from __future__ import annotations

import logging
import math
import re
import time
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

SYSMO32_TOTAL_ARM_JOINTS = len(SYSMO32_LEFT_JOINT_NAMES) + len(SYSMO32_RIGHT_JOINT_NAMES)
SYSMO32_LEFT_ELBOW_INDEX = SYSMO32_LEFT_JOINT_NAMES.index("left_elbow_joint")
SYSMO32_RIGHT_ELBOW_INDEX = len(SYSMO32_LEFT_JOINT_NAMES) + SYSMO32_RIGHT_JOINT_NAMES.index(
    "right_elbow_joint"
)
SYSMO32_LEFT_WRIST_YAW_INDEX = SYSMO32_LEFT_JOINT_NAMES.index("left_wrist_yaw_joint")
SYSMO32_RIGHT_WRIST_YAW_INDEX = len(SYSMO32_LEFT_JOINT_NAMES) + SYSMO32_RIGHT_JOINT_NAMES.index(
    "right_wrist_yaw_joint"
)
SYSMO32_HUMAN_ELBOW_REFERENCE_RAD = 0.1
SYSMO32_WRIST_YAW_REFERENCE_RAD = 0.0
SYSMO32_ELBOW_SIGN_EPS_RAD = 1e-4
SYSMO32_ELBOW_TASK_ERROR_SLACK = 0.02
SYSMO32_WRIST_YAW_TASK_ERROR_SLACK = 0.03
SYSMO32_WRIST_YAW_SCORE_WEIGHT = 0.05
SYSMO32_WRIST_YAW_NULLSPACE_WEIGHT = 4.0
SYSMO32_IK_ORIENTATION_WEIGHT = 0.2
SYSMO32_IK_MAX_JOINT_STEP_RAD = 0.08


def _default_nullspace_reference_joints() -> np.ndarray:
    reference = np.zeros(SYSMO32_TOTAL_ARM_JOINTS, dtype=np.float64)
    reference[SYSMO32_LEFT_ELBOW_INDEX] = -SYSMO32_HUMAN_ELBOW_REFERENCE_RAD
    reference[SYSMO32_RIGHT_ELBOW_INDEX] = SYSMO32_HUMAN_ELBOW_REFERENCE_RAD
    reference[SYSMO32_LEFT_WRIST_YAW_INDEX] = SYSMO32_WRIST_YAW_REFERENCE_RAD
    reference[SYSMO32_RIGHT_WRIST_YAW_INDEX] = SYSMO32_WRIST_YAW_REFERENCE_RAD
    return reference


def _default_nullspace_reference_weights() -> np.ndarray:
    weights = np.ones(SYSMO32_TOTAL_ARM_JOINTS, dtype=np.float64)
    weights[SYSMO32_LEFT_WRIST_YAW_INDEX] = SYSMO32_WRIST_YAW_NULLSPACE_WEIGHT
    weights[SYSMO32_RIGHT_WRIST_YAW_INDEX] = SYSMO32_WRIST_YAW_NULLSPACE_WEIGHT
    return weights


def _with_human_elbow_reference(reference_joints_rad: Sequence[float]) -> np.ndarray:
    reference = np.asarray(reference_joints_rad, dtype=np.float64).copy()
    if reference.shape != (SYSMO32_TOTAL_ARM_JOINTS,) or not np.all(np.isfinite(reference)):
        raise ValueError(
            f"SYSMO-32 IK nullspace reference must contain {SYSMO32_TOTAL_ARM_JOINTS} finite joints"
        )
    if reference[SYSMO32_LEFT_ELBOW_INDEX] >= -SYSMO32_ELBOW_SIGN_EPS_RAD:
        reference[SYSMO32_LEFT_ELBOW_INDEX] = -SYSMO32_HUMAN_ELBOW_REFERENCE_RAD
    if reference[SYSMO32_RIGHT_ELBOW_INDEX] <= SYSMO32_ELBOW_SIGN_EPS_RAD:
        reference[SYSMO32_RIGHT_ELBOW_INDEX] = SYSMO32_HUMAN_ELBOW_REFERENCE_RAD
    return reference


def _elbow_local_index(hand_side: str) -> int:
    return SYSMO32_LEFT_ELBOW_INDEX if hand_side == robots.LEFT else SYSMO32_RIGHT_ELBOW_INDEX - len(
        SYSMO32_LEFT_JOINT_NAMES
    )


def _wrist_yaw_local_index(hand_side: str) -> int:
    return (
        SYSMO32_LEFT_WRIST_YAW_INDEX
        if hand_side == robots.LEFT
        else SYSMO32_RIGHT_WRIST_YAW_INDEX - len(SYSMO32_LEFT_JOINT_NAMES)
    )


def _angle_distance_rad(value: float, reference: float) -> float:
    delta = float(value) - float(reference)
    return abs(math.atan2(math.sin(delta), math.cos(delta)))


def _wrist_yaw_reference_error(hand_side: str, qpos: Sequence[float], reference_qpos: Sequence[float]) -> float:
    idx = _wrist_yaw_local_index(hand_side)
    q = np.asarray(qpos, dtype=np.float64)
    ref = np.asarray(reference_qpos, dtype=np.float64)
    return _angle_distance_rad(float(q[idx]), float(ref[idx]))


def _elbow_sign_satisfied(hand_side: str, qpos: Sequence[float]) -> bool:
    elbow = float(np.asarray(qpos, dtype=np.float64)[_elbow_local_index(hand_side)])
    if hand_side == robots.LEFT:
        return elbow < -SYSMO32_ELBOW_SIGN_EPS_RAD
    return elbow > SYSMO32_ELBOW_SIGN_EPS_RAD


def _elbow_sign_violation(hand_side: str, qpos: Sequence[float]) -> float:
    elbow = float(np.asarray(qpos, dtype=np.float64)[_elbow_local_index(hand_side)])
    if hand_side == robots.LEFT:
        return max(0.0, elbow + SYSMO32_ELBOW_SIGN_EPS_RAD)
    return max(0.0, SYSMO32_ELBOW_SIGN_EPS_RAD - elbow)


def _select_elbow_preferred_result(
    best_qpos: np.ndarray,
    best_task_error: float,
    best_valid_qpos: Optional[np.ndarray],
    best_valid_task_error: float,
) -> np.ndarray:
    if (
        best_valid_qpos is not None
        and best_valid_task_error <= best_task_error + SYSMO32_ELBOW_TASK_ERROR_SLACK
    ):
        return best_valid_qpos
    return best_qpos


def _select_posture_preferred_result(
    hand_side: str,
    best_qpos: np.ndarray,
    best_task_error: float,
    best_valid_qpos: Optional[np.ndarray],
    best_valid_task_error: float,
    reference_qpos: np.ndarray,
) -> np.ndarray:
    selected = _select_elbow_preferred_result(
        best_qpos,
        best_task_error,
        best_valid_qpos,
        best_valid_task_error,
    )
    if (
        best_valid_qpos is not None
        and best_valid_task_error <= best_task_error + SYSMO32_WRIST_YAW_TASK_ERROR_SLACK
        and _wrist_yaw_reference_error(hand_side, best_valid_qpos, reference_qpos)
        < _wrist_yaw_reference_error(hand_side, selected, reference_qpos)
    ):
        return best_valid_qpos
    return selected


def _limit_delta_norm(delta: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if max_norm > 0.0 and norm > max_norm:
        return delta * (max_norm / norm)
    return delta


def _damped_least_squares_delta(
    jacobian: np.ndarray,
    error: np.ndarray,
    current_qpos: np.ndarray,
    reference_qpos: Optional[np.ndarray],
    damping: float,
    nullspace_gain: float,
    nullspace_step_limit_rad: float,
    nullspace_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    使用阻尼最小二乘法（DLS）计算关节的位移量（Delta q），并支持零空间姿态控制。
    该方法常用于解决逆运动学（IK）问题，特别是在奇异点附近时，阻尼项能防止关节速度/位移过大。

    参数:
        jacobian: 雅可比矩阵 (J)，形状为 (任务空间维度, 关节数量)
        error: 任务空间误差 (末端执行器目标位姿 - 当前位姿)，形状为 (任务空间维度,)
        current_qpos: 当前关节位置 (q)，形状为 (关节数量,)
        reference_qpos: 期望的参考关节位置 (用于零空间控制，保持特定姿态)，可选
        damping: 阻尼系数 (lambda)，用于正则化，值越大越稳定但追踪误差可能会增加
        nullspace_gain: 零空间控制增益，决定了向参考姿态靠拢的强度
        nullspace_step_limit_rad: 零空间关节位移的幅度限制（弧度），防止姿态调整过猛

    返回:
        np.ndarray: 计算得出的关节位置变化量 (delta q)
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    current_qpos = np.asarray(current_qpos, dtype=np.float64)
    # 雅可比矩阵必须是2维的，且误差向量的维度需与雅可比矩阵的行数(任务维度)一致
    if jacobian.ndim != 2 or error.shape != (jacobian.shape[0],):
        raise ValueError("jacobian/error shape mismatch")
    # 当前关节向量的维度需与雅可比矩阵的列数(关节自由度)一致
    if current_qpos.shape != (jacobian.shape[1],):
        raise ValueError("current_qpos shape mismatch")

    task_count, joint_count = jacobian.shape
    regularized_task_metric = jacobian @ jacobian.T
    regularized_task_metric.flat[:: task_count + 1] += float(damping) ** 2
    # 6x6 DLS: dq = J.T @ solve(J @ J.T + lambda^2 I, err)
    task_delta = jacobian.T @ np.linalg.solve(regularized_task_metric, error)

    # 4. 零空间（Nullspace）姿态控制判定
    # 如果没有提供参考姿态，或者零空间增益小于等于0，则只返回主任务计算结果
    if reference_qpos is None or nullspace_gain <= 0.0:
        return task_delta

    reference_qpos = np.asarray(reference_qpos, dtype=np.float64)
    # 如果参考姿态形状不匹配，或者包含 NaN/Inf 等无效值，则放弃零空间控制
    if reference_qpos.shape != current_qpos.shape or not np.all(np.isfinite(reference_qpos)):
        return task_delta

    # 5. 计算零空间投影与姿态调整
    # 零空间投影矩阵：N = I - J_pinv * J
    # 作用是过滤掉会影响末端主任务（位姿）的关节运动，只保留不改变末端位姿的冗余运动
    jacobian_pinv_times_jacobian = jacobian.T @ np.linalg.solve(regularized_task_metric, jacobian)
    projection = np.eye(joint_count) - jacobian_pinv_times_jacobian
    posture_error = reference_qpos - current_qpos
    if nullspace_weights is not None:
        weights = np.asarray(nullspace_weights, dtype=np.float64)
        if weights.shape == posture_error.shape and np.all(np.isfinite(weights)):
            posture_error = posture_error * weights
    # 在零空间中朝着参考姿态移动：delta_q_posture = N * (gain * (q_ref - q_current))
    posture_delta = projection @ (float(nullspace_gain) * posture_error)
    # 对零空间引起的关节位移幅度进行截断限制，防止单步调整幅度过大造成震荡
    posture_delta = _limit_delta_norm(posture_delta, float(nullspace_step_limit_rad))
    # 最终的关节位移量 = 主任务位移量 + 零空间姿态调整位移量
    return task_delta + posture_delta


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
        self._nullspace_reference_joints = _default_nullspace_reference_joints()
        self._nullspace_reference_weights = _default_nullspace_reference_weights()
        self._nullspace_gain = 0.03
        self._nullspace_step_limit_rad = 0.015
        self._elbow_branch_penalty_weight = 0.02
        self._orientation_weight = SYSMO32_IK_ORIENTATION_WEIGHT
        self._max_joint_step_rad = SYSMO32_IK_MAX_JOINT_STEP_RAD
        self._max_iter = 5
        self._pos_tol_m = 1e-3
        self._ori_tol_rad = 2e-2
        self._profile_log_period_s = 1.0
        self._last_profile_log_time_s = 0.0
        self._joint_qpos_addrs: dict[str, list[int]] = {robots.LEFT: [], robots.RIGHT: []}
        self._joint_dof_addrs: dict[str, list[int]] = {robots.LEFT: [], robots.RIGHT: []}
        self._jacp = None
        self._jacr = None
        self._current_quat = np.zeros(4, dtype=np.float64)
        self._inv_current_quat = np.zeros(4, dtype=np.float64)
        self._quat_error = np.zeros(4, dtype=np.float64)
        self._axis_angle = np.zeros(3, dtype=np.float64)
        self._load_model()

    def configure_nullspace(
        self,
        reference_joints_rad: Optional[Sequence[float]] = None,
        gain: Optional[float] = None,
        step_limit_rad: Optional[float] = None,
        orientation_weight: Optional[float] = None,
        max_joint_step_rad: Optional[float] = None,
        max_iter: Optional[int] = None,
        pos_tol_m: Optional[float] = None,
        ori_tol_rad: Optional[float] = None,
        profile_log_period_s: Optional[float] = None,
    ) -> None:
        if reference_joints_rad is not None:
            self._nullspace_reference_joints = _with_human_elbow_reference(reference_joints_rad)
        if gain is not None:
            self._nullspace_gain = max(0.0, float(gain))
        if step_limit_rad is not None:
            self._nullspace_step_limit_rad = max(0.0, float(step_limit_rad))
        if orientation_weight is not None:
            self._orientation_weight = max(0.0, float(orientation_weight))
        if max_joint_step_rad is not None:
            self._max_joint_step_rad = max(0.0, float(max_joint_step_rad))
        if max_iter is not None:
            self._max_iter = max(1, int(max_iter))
        if pos_tol_m is not None:
            self._pos_tol_m = max(0.0, float(pos_tol_m))
        if ori_tol_rad is not None:
            self._ori_tol_rad = max(0.0, float(ori_tol_rad))
        if profile_log_period_s is not None:
            self._profile_log_period_s = max(0.0, float(profile_log_period_s))

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
        max_iter: Optional[int] = None,
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
        total_start = time.perf_counter()
        # 把当前 /joint_states 关节角写入 MuJoCo，IK 从真实当前关节状态开始迭代。
        set_start = time.perf_counter()
        self._set_all_joints(joints)
        set_q_ms = (time.perf_counter() - set_start) * 1000.0

        qpos_addrs = self._joint_qpos_addrs[hand_side] # 表示某个 joint 在 data.qpos 里的地址
        dof_addrs = self._joint_dof_addrs[hand_side]   # 表示某个 joint 在速度空间 data.qvel / Jacobian 列里的地址
        
        best_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs]) # 保存当前这条手臂的 6 个关节角作为初始最优解
        reference_qpos = self._nullspace_reference_joints[arm_slice].copy()
        reference_weights = self._nullspace_reference_weights[arm_slice].copy()
        best_task_error = np.inf
        best_score = np.inf
        best_valid_qpos = None
        best_valid_task_error = np.inf
        best_valid_score = np.inf
        nullspace_settle_iter = 0
        iter_count = 0
        max_iter = self._max_iter if max_iter is None else max(1, int(max_iter))
        fk_ms = 0.0
        err_ms = 0.0
        jac_ms = 0.0
        solve_ms = 0.0
        update_ms = 0.0

        for _ in range(max_iter):
            iter_count += 1
            fk_start = time.perf_counter()
            self._mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[site_id] # site_xpos[site_id]是末端在世界坐标系下的位置
            current_mat = self.data.site_xmat[site_id].reshape(3, 3) # site_xmat[site_id]是末端在世界坐标系下的旋转矩阵
            fk_ms += (time.perf_counter() - fk_start) * 1000.0

            err_start = time.perf_counter()
            self._mujoco.mju_mat2Quat(self._current_quat, current_mat.flatten())

            pos_error = target_pos - current_pos
            self._mujoco.mju_negQuat(self._inv_current_quat, self._current_quat)
            self._mujoco.mju_mulQuat(self._quat_error, target_quat, self._inv_current_quat)# 四元数误差
            self._mujoco.mju_quat2Vel(self._axis_angle, self._quat_error, 1.0) # 四元数误差转轴角速度形式

            current_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs])
            task_error = float(
                np.linalg.norm(pos_error) + self._orientation_weight * np.linalg.norm(self._axis_angle)
            )
            score = task_error
            if self._nullspace_gain > 0.0:
                posture_error = (current_qpos - reference_qpos) * reference_weights
                score += 1e-4 * float(np.linalg.norm(posture_error))
                score += SYSMO32_WRIST_YAW_SCORE_WEIGHT * _wrist_yaw_reference_error(
                    hand_side, current_qpos, reference_qpos
                )
            elbow_violation = _elbow_sign_violation(hand_side, current_qpos)
            if elbow_violation > 0.0:
                score += self._elbow_branch_penalty_weight * elbow_violation
            # 保存当前最优解
            if task_error < best_task_error or (
                np.isclose(task_error, best_task_error) and score < best_score
            ):
                best_task_error = task_error
                best_score = score
                best_qpos = current_qpos.copy()
            if _elbow_sign_satisfied(hand_side, current_qpos) and (
                task_error < best_valid_task_error
                or (np.isclose(task_error, best_valid_task_error) and score < best_valid_score)
            ):
                best_valid_task_error = task_error
                best_valid_score = score
                best_valid_qpos = current_qpos.copy()
            err_ms += (time.perf_counter() - err_start) * 1000.0
            
            # 计算末端位姿的Jacobian矩阵
            jac_start = time.perf_counter()
            self._jacp.fill(0.0)
            self._jacr.fill(0.0)
            self._mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site_id) # jacp为位置 Jacobian，jacr为旋转矩阵 Jacobian
            jac = np.vstack([self._jacp, self._orientation_weight * self._jacr])[:, dof_addrs] # 合并位置和旋转 Jacobian
            error = np.concatenate([pos_error, self._orientation_weight * self._axis_angle])
            jac_ms += (time.perf_counter() - jac_start) * 1000.0
            try:
                solve_start = time.perf_counter()
                delta = _damped_least_squares_delta(
                    jac,
                    error,
                    current_qpos,
                    reference_qpos,
                    damping=0.1,
                    nullspace_gain=self._nullspace_gain,
                    nullspace_step_limit_rad=self._nullspace_step_limit_rad,
                    nullspace_weights=reference_weights,
                ) # DLS + 零空间初始姿态次任务
                solve_ms += (time.perf_counter() - solve_start) * 1000.0
            except np.linalg.LinAlgError:
                return None
            delta_norm = np.linalg.norm(delta)
            task_satisfied = (
                np.linalg.norm(pos_error) < self._pos_tol_m
                and np.linalg.norm(self._axis_angle) < self._ori_tol_rad
            )
            # 判断是否满足停止条件；若零空间项仍在生效，最多额外整理几步，避免实时循环跑满迭代。
            if task_satisfied:
                nullspace_settle_iter += 1
                if delta_norm < 1e-5 or nullspace_settle_iter >= 2:
                    break
            else:
                nullspace_settle_iter = 0
            # 限制单步关节增量
            if self._max_joint_step_rad > 0.0 and delta_norm > self._max_joint_step_rad:
                delta *= self._max_joint_step_rad / delta_norm
            # 更新 MuJoCo qpos
            update_start = time.perf_counter()
            for idx, addr in enumerate(qpos_addrs):
                self.data.qpos[addr] += delta[idx]
            # 关节限位
            self._clip_joint_qpos(joint_ids, qpos_addrs)
            update_ms += (time.perf_counter() - update_start) * 1000.0

        result = np.asarray(joints[arm_slice], dtype=np.float64).copy()
        result[:] = _select_posture_preferred_result(
            hand_side,
            best_qpos,
            best_task_error,
            best_valid_qpos,
            best_valid_task_error,
            reference_qpos,
        )
        if not np.all(np.isfinite(result)):
            return None
        self._log_ik_profile(
            hand_side=hand_side,
            iterations=iter_count,
            total_ms=(time.perf_counter() - total_start) * 1000.0,
            set_q_ms=set_q_ms,
            fk_ms=fk_ms,
            err_ms=err_ms,
            jac_ms=jac_ms,
            solve_ms=solve_ms,
            update_ms=update_ms,
            task_error=best_task_error,
        )
        return result

    def _log_ik_profile(
        self,
        hand_side: str,
        iterations: int,
        total_ms: float,
        set_q_ms: float,
        fk_ms: float,
        err_ms: float,
        jac_ms: float,
        solve_ms: float,
        update_ms: float,
        task_error: float,
    ) -> None:
        if self._profile_log_period_s <= 0.0:
            return
        now = time.time()
        if now - self._last_profile_log_time_s < self._profile_log_period_s:
            return
        self._last_profile_log_time_s = now
        logger.info(
            "[Diag][IK_PROFILE] side=%s iter=%d total_ms=%.2f set_q_ms=%.2f "
            "fk_ms=%.2f err_ms=%.2f jac_ms=%.2f solve_ms=%.2f update_ms=%.2f task_error=%.4f",
            hand_side,
            iterations,
            total_ms,
            set_q_ms,
            fk_ms,
            err_ms,
            jac_ms,
            solve_ms,
            update_ms,
            task_error,
        )
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
