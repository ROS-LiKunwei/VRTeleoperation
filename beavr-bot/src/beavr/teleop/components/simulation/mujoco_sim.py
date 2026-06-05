"""
MuJoCo仿真环境模块 - SYSMO-32双臂机器人

本模块是BeaVR-bot遥操作系统的仿真终端，负责在MuJoCo物理引擎中
渲染SYSMO-32双臂机器人，并根据PICO4手势的相对位姿控制双臂末端移动。

数据流位置：
    pico4.py → keypoint_transform.py → xarm7_operator.py → [本模块: mujoco_sim.py]

功能：
    1. 加载SYSMO-32 URDF模型到MuJoCo仿真环境
    2. 订阅XArmOperator发布的CartesianTarget命令
    3. 使用MuJoCo逆运动学(IK)求解关节角度
    4. 驱动仿真中的机器人双臂运动
    5. 提供可视化渲染窗口

通信协议：
    - 接收：ZMQ SUB套接字，从xarm7_operator.py订阅CartesianTarget对象
    - 无发送（仿真终端节点）

端口配置：
    - 右手末端命令订阅端口：10010 (XARM_ENDEFF_PUBLISH_PORT)
    - 左手末端命令订阅端口：10010 (XARM_ENDEFF_PUBLISH_PORT)
"""

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.interface.interface_types import Sysmo32JointCommand
from beavr.teleop.components.interface.robots.sysmo32_robot import SYSMO32_JOINT_COMMAND_TOPIC
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="microseconds")


class MuJoCoSysmoSimulator(Component):
    """
    MuJoCo仿真环境 - SYSMO-32双臂机器人。

    数据流角色：
        本类是遥操作系统数据流的仿真终端，
        负责在MuJoCo物理引擎中渲染SYSMO-32双臂机器人，
        并根据PICO4手势的相对位姿控制双臂末端执行器移动。

    工作流程：
        1. 加载SYSMO-32 URDF模型到MuJoCo
        2. 订阅XArmOperator发布的CartesianTarget命令
        3. 将笛卡尔空间目标(位置+四元数)转换为关节角度(IK)
        4. 设置MuJoCo仿真中的关节角度
        5. 步进仿真并渲染

    IK求解方法：
        使用MuJoCo内置的ik求解器（jacobi_pinv方法），
        或使用解析IK方法（如果可用）。
        当前实现使用MuJoCo的inverse kinematics API。
    """

    # SYSMO-32关节名称映射
    LEFT_JOINT_NAMES = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
    ]

    RIGHT_JOINT_NAMES = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
    ]

    LEFT_ENDEFF_SITE = "left_endeff"
    RIGHT_ENDEFF_SITE = "right_endeff"

    POSITION_SERVO_KP = 150.0
    JOINT_DAMPING = 2.5
    SIMULATION_TIMESTEP = 0.002
    ENABLE_GRAVITY_COMPENSATION = True
    HOME_JOINT_POSITIONS = {
        "left_shoulder_roll_joint": 0.5,
        "right_shoulder_roll_joint": -0.5,
        "left_elbow_joint": -1.2,
        "right_elbow_joint": 1.2,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
    }

    def __init__(
        self,
        host: str,
        right_endeff_subscribe_port: int,
        left_endeff_subscribe_port: int,
        urdf_path: str,
        simulation_mode: bool = True,
        render: bool = True,
    ):
        """
        初始化MuJoCo仿真环境。

        Args:
            host: 网络主机地址（ZMQ通信地址）。
            right_endeff_subscribe_port: 右手末端命令订阅端口。
            left_endeff_subscribe_port: 左手末端命令订阅端口。
            urdf_path: SYSMO-32 URDF文件路径。
            simulation_mode: 是否为仿真模式（始终为True）。
            render: 是否启用可视化渲染窗口。
        """
        self.notify_component_start("mujoco_sysmo_simulator")
        self.host = host
        self.simulation_mode = simulation_mode
        self.render = render
        path = Path(urdf_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[5] / urdf_path
        self.urdf_path = str(path)

        self._ik_tolerance = 0.01 # 位置误差收敛阈值
        self._ik_orientation_tolerance = 0.15 # 姿态误差阈值
        # 完整位姿 IK 不稳定时，进入 fallback，降低姿态误差权重，优先保证位置跟踪。
        self._ik_fallback_orientation_weight = 0.05 # 进入姿态优化近似阈值
        self._ik_fallback_position_tolerance = 0.02 # 进入位置优先近似阈值
        self._ik_reject_position_tolerance = 0.08 
        self._ik_singularity_condition_threshold = 250.0 # 雅可比矩阵条件数阈值。条件数过大说明接近奇异位形，IK 解可能不稳定
        self._ik_joint_step_limit = 0.08 # 每次 IK 迭代允许的最大关节增量，单位 rad，防止单步求解跳太大
        self._ik_full_max_iter = 80
        self._ik_approx_max_iter = 80
        self._last_ik_diag_log_time = {}
        self._ik_reject_active = {robots.LEFT: False, robots.RIGHT: False} # 记录左右臂当前是否处于 IK 拒绝状态
        self._joint_actuator_ids = {} # 缓存 MuJoCo actuator id，后面根据 joint name 找对应 actuator 控制关节
        self._joint_hold_qpos = None # 保存当前要保持的关节位置。没有新目标或 pause 时，可以让关节保持上一安全位置
        self._initial_endeff_poses = {} # 缓存左右臂初始末端位姿，用于 IK 诊断和相对目标分析。
        self._ik_logger = self._setup_ik_logger()

        # 初始化MuJoCo模型
        self.model = None
        self.data = None
        self._load_model()
        self._left_joint_ids = []
        self._right_joint_ids = []
        self._left_endeff_site_id = None
        self._right_endeff_site_id = None
        self._cache_joint_ids()
        self._cache_actuator_ids()
        self._configure_position_servos()
        self._cache_initial_endeff_poses()

        # 初始化ZMQ订阅者: 接收robot层IK后的关节命令
        self._right_endeff_subscriber = ZMQSubscriber(
            host=host,
            port=right_endeff_subscribe_port,
            topic=SYSMO32_JOINT_COMMAND_TOPIC,
            message_type=Sysmo32JointCommand,
        )

        self._left_endeff_subscriber = ZMQSubscriber(
            host=host,
            port=left_endeff_subscribe_port,
            topic=SYSMO32_JOINT_COMMAND_TOPIC,
            message_type=Sysmo32JointCommand,
        )

        # 计时器
        self.timer = FrequencyTimer(robots.VR_FREQ)  # 30Hz
        self._ik_logger.info(
            "IK参数: tolerance=%s, orientation_tolerance=%s, "
            "fallback_orientation_weight=%s, reject_position_tolerance=%s, "
            "singularity_condition_threshold=%s",
            self._ik_tolerance,
            self._ik_orientation_tolerance,
            self._ik_fallback_orientation_weight,
            self._ik_reject_position_tolerance,
            self._ik_singularity_condition_threshold,
        )

        logger.info(f"MuJoCo SYSMO-32仿真器初始化完成, URDF: {self.urdf_path}")

    def _setup_ik_logger(self):
        repo_root = Path(__file__).resolve().parents[5]
        ik_log_dir = repo_root / "Log" / "IK"
        ik_log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")[:-3]
        log_file = ik_log_dir / f"ik_run_{timestamp}_pid{os.getpid()}.log"

        ik_logger = logging.getLogger(f"{__name__}.ik.{id(self)}")
        ik_logger.setLevel(logging.DEBUG)
        ik_logger.propagate = False
        for handler in list(ik_logger.handlers):
            ik_logger.removeHandler(handler)
            handler.close()

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            MicrosecondFormatter(
                "[%(levelname)s] %(asctime)s %(processName)s %(name)s: %(message)s",
                datefmt="%H:%M:%S.%f",
            )
        )
        ik_logger.addHandler(file_handler)
        ik_logger.info(f"IK log file created: {log_file}")
        return ik_logger

    def _load_model(self):
        """
        加载SYSMO-32 URDF模型到MuJoCo，并添加末端执行器site。

        步骤：
        1. 加载URDF文件
        2. 导出为MuJoCo XML格式（使用mj_saveLastXML）
        3. 在XML中添加末端执行器site（left_endeff, right_endeff）
        4. 重新加载修改后的XML
        """
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF文件不存在: {self.urdf_path}")

        try:
            temp_model = mujoco.MjModel.from_xml_path(self.urdf_path)

            # 导出为MuJoCo XML
            xml_path = "/tmp/sysmo32_mujoco.xml"
            mujoco.mj_saveLastXML(xml_path, temp_model)
            with open(xml_path, "r") as f:
                xml_string = f.read()

            xml_string = self._resolve_exported_mesh_paths(xml_string)
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

            actuator_xml = self._build_position_actuator_xml(temp_model)
            xml_string = xml_string.replace("</mujoco>", f"{actuator_xml}\n</mujoco>")

            self.model = mujoco.MjModel.from_xml_string(xml_string)
            self.data = mujoco.MjData(self.model)
            self.model.opt.timestep = self.SIMULATION_TIMESTEP
            mujoco.mj_forward(self.model, self.data)
            logger.info(
                f"MuJoCo模型加载成功: {self.model.nq} 个自由度, "
                f"{self.model.njnt} 个关节, {self.model.nsite} 个site"
            )
        except Exception as e:
            logger.error(f"MuJoCo模型加载失败: {e}")
            raise

    def _resolve_exported_mesh_paths(self, xml_string):
        urdf_dir = Path(self.urdf_path).resolve().parent

        def resolve_mesh_file(match):
            mesh_file = match.group(1)
            mesh_path = Path(mesh_file)
            if not mesh_path.is_absolute():
                mesh_path = (urdf_dir / mesh_path).resolve()
            return f'file="{mesh_path}"'

        return re.sub(r'file="([^"]+\.(?:STL|stl))"', resolve_mesh_file, xml_string)

    def _insert_site_into_body(self, xml_string, body_name, site_xml):
        pattern = rf'(<body name="{re.escape(body_name)}"[^>]*>)'
        replacement = rf"\1\n        {site_xml}"
        updated_xml, count = re.subn(pattern, replacement, xml_string, count=1)
        if count != 1:
            raise ValueError(f"未找到MuJoCo body，无法添加site: {body_name}")
        return updated_xml

    def _build_position_actuator_xml(self, source_model):
        actuator_lines = ["  <actuator>"]
        for joint_name in self.LEFT_JOINT_NAMES + self.RIGHT_JOINT_NAMES:
            actuator_name = self._actuator_name_for_joint(joint_name)
            joint_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            low, high = source_model.jnt_range[joint_id]
            actuator_lines.append(
                (
                    f'    <position name="{actuator_name}" joint="{joint_name}" '
                    f'kp="{self.POSITION_SERVO_KP}" ctrllimited="true" '
                    f'ctrlrange="{low} {high}"/>'
                )
            )
        actuator_lines.append("  </actuator>")
        return "\n".join(actuator_lines)

    def _actuator_name_for_joint(self, joint_name):
        if joint_name.endswith("_joint"):
            return joint_name[: -len("_joint")]
        return joint_name

    def _cache_joint_ids(self):
        """
        缓存关节和site的MuJoCo内部ID，避免每帧查找。

        MuJoCo使用整数ID来引用关节和site，
        缓存这些ID可以提高运行时性能。
        """
        if self.model is None:
            return

        # 缓存左手关节ID
        for name in self.LEFT_JOINT_NAMES:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    self._left_joint_ids.append(joint_id)
                else:
                    logger.warning(f"左手关节未找到: {name}")
            except Exception as e:
                logger.warning(f"查找左手关节ID失败: {name}, 错误: {e}")

        # 缓存右手关节ID
        for name in self.RIGHT_JOINT_NAMES:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    self._right_joint_ids.append(joint_id)
                else:
                    logger.warning(f"右手关节未找到: {name}")
            except Exception as e:
                logger.warning(f"查找右手关节ID失败: {name}, 错误: {e}")

        # 缓存末端执行器site ID
        try:
            self._left_endeff_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.LEFT_ENDEFF_SITE
            )
            if self._left_endeff_site_id < 0:
                logger.warning(f"左手末端site未找到: {self.LEFT_ENDEFF_SITE}，将使用最后一个左手link")
        except Exception:
            self._left_endeff_site_id = None

        try:
            self._right_endeff_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.RIGHT_ENDEFF_SITE
            )
            if self._right_endeff_site_id < 0:
                logger.warning(f"右手末端site未找到: {self.RIGHT_ENDEFF_SITE}，将使用最后一个右手link")
        except Exception:
            self._right_endeff_site_id = None

        logger.info(f"关节ID缓存完成: 左手{len(self._left_joint_ids)}个, 右手{len(self._right_joint_ids)}个")

    def _cache_actuator_ids(self):
        self._joint_actuator_ids = {}
        if self.model is None:
            return

        for joint_name in self.LEFT_JOINT_NAMES + self.RIGHT_JOINT_NAMES:
            actuator_name = self._actuator_name_for_joint(joint_name)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0:
                self._joint_actuator_ids[joint_name] = actuator_id
            else:
                logger.warning(f"MuJoCo位置执行器未找到: {actuator_name}")

        logger.info(f"位置执行器缓存完成: {len(self._joint_actuator_ids)}个")

    def _configure_position_servos(self):
        if self.model is None or self.data is None:
            return

        self.model.opt.timestep = self.SIMULATION_TIMESTEP
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        self._apply_home_joint_positions()

        for actuator_id in self._joint_actuator_ids.values():
            self.model.actuator_gainprm[actuator_id, 0] = self.POSITION_SERVO_KP
            self.model.actuator_biasprm[actuator_id, 1] = -self.POSITION_SERVO_KP

        for joint_id in self._left_joint_ids + self._right_joint_ids:
            dof_addr = self.model.jnt_dofadr[joint_id]
            self.model.dof_damping[dof_addr] = self.JOINT_DAMPING

            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_id = self._joint_actuator_ids.get(joint_name)
            if actuator_id is not None:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.ctrl[actuator_id] = self.data.qpos[qpos_addr]

        self._joint_hold_qpos = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        logger.info(
            "MuJoCo位置伺服参数已配置: "
            f"kp={self.POSITION_SERVO_KP}, joint_damping={self.JOINT_DAMPING}, "
            f"timestep={self.SIMULATION_TIMESTEP}, "
            f"gravity_compensation={self.ENABLE_GRAVITY_COMPENSATION}, "
            f"home_joints={self.HOME_JOINT_POSITIONS}"
        )

    def _apply_home_joint_positions(self):
        for joint_name, home_qpos in self.HOME_JOINT_POSITIONS.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                logger.warning(f"Home姿态关节未找到: {joint_name}")
                continue

            qpos_addr = self.model.jnt_qposadr[joint_id]
            low, high = self.model.jnt_range[joint_id]
            self.data.qpos[qpos_addr] = np.clip(home_qpos, low, high)

    def _apply_arm_gravity_compensation(self):
        if not self.ENABLE_GRAVITY_COMPENSATION:
            return

        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        for joint_id in self._left_joint_ids + self._right_joint_ids:
            dof_addr = self.model.jnt_dofadr[joint_id]
            self.data.qfrc_applied[dof_addr] = self.data.qfrc_bias[dof_addr]

    def _apply_joint_position_holds(self):
        if self._joint_hold_qpos is None:
            return

        for joint_id in self._left_joint_ids + self._right_joint_ids:
            qpos_addr = self.model.jnt_qposadr[joint_id]
            dof_addr = self.model.jnt_dofadr[joint_id]
            hold_qpos = self._joint_hold_qpos[qpos_addr]
            self.data.qpos[qpos_addr] = hold_qpos
            self.data.qvel[dof_addr] = 0.0

            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_id = self._joint_actuator_ids.get(joint_name)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = hold_qpos

    def _cache_initial_endeff_poses(self):
        self._initial_endeff_poses = {}
        for side, joint_ids, site_id in (
            (robots.LEFT, self._left_joint_ids, self._left_endeff_site_id),
            (robots.RIGHT, self._right_joint_ids, self._right_endeff_site_id),
        ):
            if not joint_ids:
                continue
            pos, quat = self._get_endeff_pose(joint_ids, site_id)
            self._initial_endeff_poses[side] = (pos, quat)
            self._ik_logger.info(
                (f"MuJoCo初始末端位姿[{side}]: pos={self._fmt_array(pos)}, quat_wxyz={self._fmt_array(quat)}")
            )

    def _get_endeff_pose(self, joint_ids, endeff_site_id):
        mujoco.mj_forward(self.model, self.data)
        if endeff_site_id is not None and endeff_site_id >= 0:
            pos = self.data.site_xpos[endeff_site_id].copy()
            mat = self.data.site_xmat[endeff_site_id].reshape(3, 3).copy()
        else:
            last_joint_body_id = self.model.jnt_bodyid[joint_ids[-1]]
            pos = self.data.xpos[last_joint_body_id].copy()
            mat = self.data.xmat[last_joint_body_id].reshape(3, 3).copy()

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return pos, quat

    def _quat_relative(self, reference_quat, target_quat):
        inv_reference = np.zeros(4)
        relative = np.zeros(4)
        mujoco.mju_negQuat(inv_reference, reference_quat)
        mujoco.mju_mulQuat(relative, target_quat, inv_reference)
        return relative

    def _cartesian_to_mujoco_pos(self, position_m, orientation_xyzw):
        """
        将CartesianTarget的位姿转换为MuJoCo格式。

        MuJoCo使用4x4齐次变换矩阵表示位姿，
        本方法将位置(米)和四元数(xyzw)转换为MuJoCo的site pos和quat格式。

        注意坐标系差异：
        - PICO/Unity使用左手坐标系
        - MuJoCo使用右手坐标系
        可能需要进行坐标轴变换。

        Args:
            position_m: 目标位置 (x, y, z) 米
            orientation_xyzw: 目标姿态四元数 (x, y, z, w)

        Returns:
            tuple: (pos, quat)
                - pos: numpy数组 [x, y, z]
                - quat: numpy数组 [w, x, y, z] (MuJoCo四元数格式)
        """
        pos = np.array(position_m, dtype=np.float64)

        # MuJoCo四元数格式为 [w, x, y, z]，与scipy的 [x, y, z, w] 不同
        quat_xyzw = np.array(orientation_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat_xyzw)
        if norm > 1e-6:
            quat_xyzw = quat_xyzw / norm
        else:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])

        # 转换为MuJoCo格式 [w, x, y, z]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat_wxyz

    def _clamp_joint_qpos(self, joint_ids, qpos_addrs):
        hit_limit = False
        for jid, addr in zip(joint_ids, qpos_addrs, strict=False):
            low = self.model.jnt_range[jid, 0]
            high = self.model.jnt_range[jid, 1]
            before = self.data.qpos[addr]
            self.data.qpos[addr] = np.clip(before, low, high)
            hit_limit = hit_limit or not np.isclose(before, self.data.qpos[addr], atol=1e-9)
        return hit_limit

    def _log_ik_diagnostics(self, side, mode, diagnostics, level=logging.INFO):
        current_time = time.time()
        key = f"{side}:{mode}"
        if current_time - self._last_ik_diag_log_time.get(key, 0.0) < 1.0:
            return
        self._last_ik_diag_log_time[key] = current_time
        self._ik_logger.log(
            level,
            (
                f"IK诊断[{side}][{mode}]\n"
                f"  status: converged={diagnostics['converged']}, "
                f"iter={diagnostics['iterations']}, "
                f"pos_err={diagnostics['pos_error_norm']:.4f}m, "
                f"ori_err={diagnostics['ori_error_norm']:.4f}rad, "
                f"cond={diagnostics['condition_number']:.1f}, "
                f"singular={diagnostics['near_singular']}, "
                f"hit_limit={diagnostics['hit_limit']}\n"
                f"  initial: pos={self._fmt_array(diagnostics['initial_pos'])}, "
                f"quat_wxyz={self._fmt_array(diagnostics['initial_quat'])}\n"
                f"  target_relative_to_initial: "
                f"pos={self._fmt_array(diagnostics['target_relative_pos'])}, "
                f"quat_wxyz={self._fmt_array(diagnostics['target_relative_quat'])}\n"
                f"  target: pos={self._fmt_array(diagnostics['target_pos'])}, "
                f"quat_wxyz={self._fmt_array(diagnostics['target_quat'])}\n"
                f"  current_start: pos={self._fmt_array(diagnostics['current_start_pos'])}, "
                f"quat_wxyz={self._fmt_array(diagnostics['current_start_quat'])}\n"
                f"  current_end: pos={self._fmt_array(diagnostics['current_end_pos'])}, "
                f"quat_wxyz={self._fmt_array(diagnostics['current_end_quat'])}"
            ),
        )

    def _fmt_array(self, values):
        return np.array2string(
            np.asarray(values, dtype=np.float64),
            precision=4,
            suppress_small=False,
            separator=",",
        )

    def _warn_ik_fallback(self, side):
        current_time = time.time()
        key = f"{side}:fallback_reason"
        if current_time - self._last_ik_diag_log_time.get(key, 0.0) < 1.0:
            return
        self._last_ik_diag_log_time[key] = current_time
        self._ik_logger.warning(f"IK[{side}]: 姿态目标可能不可达或接近奇异，使用位置优先近似解。")

    def _build_pose_diagnostics(self, side, target_pos, target_quat):
        initial_pos, initial_quat = self._initial_endeff_poses.get(
            side,
            (np.full(3, np.nan), np.full(4, np.nan)),
        )
        return {
            "initial_pos": initial_pos.copy(),
            "initial_quat": initial_quat.copy(),
            "target_relative_pos": target_pos - initial_pos,
            "target_relative_quat": self._quat_relative(initial_quat, target_quat)
            if np.all(np.isfinite(initial_quat))
            else np.full(4, np.nan),
        }

    def _solve_ik_pass(
        self,
        joint_ids,
        target_pos,
        target_quat,
        endeff_site_id,
        orientation_weight,
        max_iter,
        side,
    ):
        """Run one IK pass. orientation_weight=0 gives a position-only approximation."""
        qpos_addrs = [self.model.jnt_qposadr[jid] for jid in joint_ids]
        joint_dof_addrs = [self.model.jnt_dofadr[jid] for jid in joint_ids]
        saved_qpos = self.data.qpos.copy()
        best_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs])
        best_score = np.inf
        hit_limit = False
        near_singular = False
        condition_number = 0.0
        max_condition_number = 0.0
        pos_error_norm = np.inf
        ori_error_norm = np.inf
        converged = False
        iterations = 0
        current_start_pos = None
        current_start_quat = None
        current_end_pos = None
        current_end_quat = None

        for iteration in range(max_iter):
            iterations = iteration + 1
            mujoco.mj_forward(self.model, self.data)

            if endeff_site_id is not None and endeff_site_id >= 0:
                current_pos = self.data.site_xpos[endeff_site_id].copy()
                current_mat = self.data.site_xmat[endeff_site_id].reshape(3, 3).copy()
            else:
                last_joint_body_id = self.model.jnt_bodyid[joint_ids[-1]]
                current_pos = self.data.xpos[last_joint_body_id].copy()
                current_mat = self.data.xmat[last_joint_body_id].reshape(3, 3).copy()

            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten())
            if current_start_pos is None:
                current_start_pos = current_pos.copy()
                current_start_quat = current_quat.copy()
            current_end_pos = current_pos.copy()
            current_end_quat = current_quat.copy()

            inv_current_quat = np.zeros(4)
            quat_error = np.zeros(4)
            mujoco.mju_negQuat(inv_current_quat, current_quat)
            mujoco.mju_mulQuat(quat_error, target_quat, inv_current_quat)
            axis_angle = np.zeros(3)
            mujoco.mju_quat2Vel(axis_angle, quat_error, 1.0)
            ori_error_norm = np.linalg.norm(axis_angle)

            score = pos_error_norm + orientation_weight * ori_error_norm
            if score < best_score:
                best_score = score
                best_qpos = np.array([self.data.qpos[addr] for addr in qpos_addrs])

            orientation_satisfied = (
                orientation_weight <= 0.0 or ori_error_norm < self._ik_orientation_tolerance
            )
            if pos_error_norm < self._ik_tolerance and orientation_satisfied:
                converged = True
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            if endeff_site_id is not None and endeff_site_id >= 0:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, endeff_site_id)
            else:
                last_joint_body_id = self.model.jnt_bodyid[joint_ids[-1]]
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, last_joint_body_id)

            if orientation_weight > 0.0:
                jacobian = np.vstack([jacp, orientation_weight * jacr])
                error = np.concatenate([pos_error, orientation_weight * axis_angle])
            else:
                jacobian = jacp
                error = pos_error

            jacobian_sub = jacobian[:, joint_dof_addrs]
            try:
                condition_number = float(np.linalg.cond(jacobian_sub))
            except np.linalg.LinAlgError:
                condition_number = np.inf
            if (
                not np.isfinite(condition_number)
                or condition_number > self._ik_singularity_condition_threshold
            ):
                near_singular = True
            if not np.isfinite(condition_number):
                max_condition_number = np.inf
            elif np.isfinite(max_condition_number):
                max_condition_number = max(max_condition_number, condition_number)

            damping = 0.1
            jt_j = jacobian_sub.T @ jacobian_sub + damping**2 * np.eye(len(joint_ids))
            delta_q = np.linalg.solve(jt_j, jacobian_sub.T @ error)
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > self._ik_joint_step_limit:
                delta_q *= self._ik_joint_step_limit / delta_norm

            for i, addr in enumerate(qpos_addrs):
                self.data.qpos[addr] += delta_q[i]
            hit_limit = self._clamp_joint_qpos(joint_ids, qpos_addrs) or hit_limit

        result = best_qpos.copy()
        self.data.qpos[:] = saved_qpos
        if current_start_pos is None:
            mujoco.mj_forward(self.model, self.data)
            if endeff_site_id is not None and endeff_site_id >= 0:
                current_start_pos = self.data.site_xpos[endeff_site_id].copy()
                current_mat = self.data.site_xmat[endeff_site_id].reshape(3, 3).copy()
            else:
                last_joint_body_id = self.model.jnt_bodyid[joint_ids[-1]]
                current_start_pos = self.data.xpos[last_joint_body_id].copy()
                current_mat = self.data.xmat[last_joint_body_id].reshape(3, 3).copy()
            current_start_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_start_quat, current_mat.flatten())
            current_end_pos = current_start_pos.copy()
            current_end_quat = current_start_quat.copy()

        diagnostics = {
            "converged": converged,
            "iterations": iterations,
            "pos_error_norm": float(pos_error_norm),
            "ori_error_norm": float(ori_error_norm),
            "condition_number": float(max_condition_number),
            "near_singular": near_singular,
            "hit_limit": hit_limit,
            "target_pos": target_pos.copy(),
            "target_quat": target_quat.copy(),
            "current_start_pos": current_start_pos.copy(),
            "current_start_quat": current_start_quat.copy(),
            "current_end_pos": current_end_pos.copy(),
            "current_end_quat": current_end_quat.copy(),
        }
        diagnostics.update(self._build_pose_diagnostics(side, target_pos, target_quat))
        return result, diagnostics

    def _solve_ik(self, joint_ids, target_pos, target_quat, endeff_site_id=None, side="unknown"):
        """
        使用MuJoCo内置IK求解器计算关节角度。

        IK求解流程：
        1. 获取当前末端执行器位姿
        2. 计算位姿误差（位置误差+姿态误差）
        3. 计算雅可比矩阵
        4. 使用伪逆方法计算关节角度增量
        5. 更新关节角度
        6. 重复直到收敛或达到最大迭代次数

        Args:
            joint_ids: 关节ID列表
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标姿态 [w, x, y, z] (MuJoCo格式)
            endeff_site_id: 末端执行器site ID（可选）

        Returns:
            numpy数组: 求解得到的关节角度，长度与joint_ids相同
        """
        if not joint_ids or self.model is None:
            return None

        full_result, full_diag = self._solve_ik_pass(
            joint_ids,
            target_pos,
            target_quat,
            endeff_site_id,
            orientation_weight=1.0,
            max_iter=self._ik_full_max_iter,
            side=side,
        )
        full_unstable = (
            full_diag["near_singular"]
            or full_diag["hit_limit"]
            or (
                full_diag["pos_error_norm"] < self._ik_fallback_position_tolerance
                and full_diag["ori_error_norm"] > self._ik_orientation_tolerance
            )
        )

        if not full_unstable and full_diag["converged"]:
            self._log_ik_diagnostics(side, "full", full_diag, level=logging.DEBUG)
            self._ik_reject_active[side] = False
            return full_result

        self._log_ik_diagnostics(side, "full_unstable", full_diag, level=logging.WARNING)
        approx_result, approx_diag = self._solve_ik_pass(
            joint_ids,
            target_pos,
            target_quat,
            endeff_site_id,
            orientation_weight=self._ik_fallback_orientation_weight,
            max_iter=self._ik_approx_max_iter,
            side=side,
        )
        self._log_ik_diagnostics(side, "approx_position_priority", approx_diag, level=logging.WARNING)
        self._warn_ik_fallback(side)

        if (
            not approx_diag["converged"]
            and approx_diag["pos_error_norm"] > self._ik_reject_position_tolerance
        ):
            if not self._ik_reject_active.get(side, False):
                self._ik_logger.error(
                    (
                        f"IK[{side}]: 近似解仍不可达，拒绝应用目标并保持上一帧。\n"
                        f"  status: pos_err={approx_diag['pos_error_norm']:.4f}m, "
                        f"reject_threshold={self._ik_reject_position_tolerance:.4f}m, "
                        f"hit_limit={approx_diag['hit_limit']}, "
                        f"singular={approx_diag['near_singular']}\n"
                        f"  target: pos={self._fmt_array(approx_diag['target_pos'])}\n"
                        f"  current_start: pos={self._fmt_array(approx_diag['current_start_pos'])}\n"
                        f"  current_end: pos={self._fmt_array(approx_diag['current_end_pos'])}"
                    )
                )
            self._ik_reject_active[side] = True
            return None

        self._ik_reject_active[side] = False
        return approx_result

    def _apply_endeff_target(self, target: CartesianTarget, joint_ids, endeff_site_id):
        """
        将笛卡尔空间目标应用到MuJoCo仿真。

        流程：
        1. 将CartesianTarget转换为MuJoCo位姿格式
        2. 使用IK求解关节角度
        3. 设置仿真中的关节角度

        Args:
            target: CartesianTarget对象（位置+四元数）
            joint_ids: 对应手臂的关节ID列表
            endeff_site_id: 末端执行器site ID
        """
        if target is None or not joint_ids:
            return

        try:
            if not self._is_valid_target(target):
                logger.warning(
                    f"跳过 {target.hand_side} 手无效目标: pos/orientation 包含 NaN/Inf 或四元数退化"
                )
                return

            target_pos, target_quat = self._cartesian_to_mujoco_pos(
                target.position_m, target.orientation_xyzw
            )

            joint_angles = self._solve_ik(
                joint_ids,
                target_pos,
                target_quat,
                endeff_site_id,
                side=target.hand_side,
            )

            if joint_angles is not None:
                for i, jid in enumerate(joint_ids):
                    qpos_addr = self.model.jnt_qposadr[jid]
                    # 限制关节角度在URDF定义的范围内
                    low = self.model.jnt_range[jid, 0]
                    high = self.model.jnt_range[jid, 1]
                    clamped = np.clip(joint_angles[i], low, high)
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    actuator_id = self._joint_actuator_ids.get(joint_name)
                    if actuator_id is not None:
                        self.data.ctrl[actuator_id] = clamped
                    if self._joint_hold_qpos is not None:
                        self._joint_hold_qpos[qpos_addr] = clamped
                    self.data.qpos[qpos_addr] = clamped
                    self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0

        except Exception as e:
            logger.error(f"应用末端目标失败: {e}")

    def _is_valid_target(self, target: CartesianTarget) -> bool:
        pos = np.asarray(target.position_m, dtype=np.float64)
        quat = np.asarray(target.orientation_xyzw, dtype=np.float64)
        return (
            pos.shape == (3,)
            and quat.shape == (4,)
            and np.all(np.isfinite(pos))
            and np.all(np.isfinite(quat))
            and np.linalg.norm(quat) > 1e-6
        )

    def _is_valid_joint_command(self, command: Sysmo32JointCommand, expected_side: str) -> bool:
        if command is None or command.hand_side != expected_side:
            return False
        joints = np.asarray(command.arm_joint_positions_rad, dtype=np.float64)
        return joints.shape == (len(self.RIGHT_JOINT_NAMES),) and np.all(np.isfinite(joints))

    def _apply_joint_command(self, command: Sysmo32JointCommand, joint_ids):
        if command is None or not joint_ids:
            return

        joint_angles = np.asarray(command.arm_joint_positions_rad, dtype=np.float64)
        for i, jid in enumerate(joint_ids):
            qpos_addr = self.model.jnt_qposadr[jid]
            low = self.model.jnt_range[jid, 0]
            high = self.model.jnt_range[jid, 1]
            clamped = np.clip(joint_angles[i], low, high)
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            actuator_id = self._joint_actuator_ids.get(joint_name)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = clamped
            if self._joint_hold_qpos is not None:
                self._joint_hold_qpos[qpos_addr] = clamped
            self.data.qpos[qpos_addr] = clamped
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0

    def _receive_targets(self):
        """
        从ZMQ订阅者接收robot层IK后的关节目标命令。

        分别接收右手和左手的Sysmo32JointCommand命令，
        并缓存到_left_target和_right_target。
        """
        # 接收右手目标
        right_msg = self._right_endeff_subscriber.recv_keypoints()
        if right_msg is not None:
            if not self._is_valid_joint_command(right_msg, robots.RIGHT):
                logger.warning("MuJoCo: 跳过右手无效或错侧关节目标")
                right_msg = None
        if right_msg is not None:
            self._right_target = right_msg
            logger.debug(
                f"右手IK关节目标: {np.array2string(np.asarray(right_msg.arm_joint_positions_rad), precision=4)}"
            )
        elif self._right_target is None:
            self._log_no_joint_command(robots.RIGHT, self._right_endeff_subscriber._port)

        # 接收左手目标
        left_msg = self._left_endeff_subscriber.recv_keypoints()
        if left_msg is not None:
            if not self._is_valid_joint_command(left_msg, robots.LEFT):
                logger.warning("MuJoCo: 跳过左手无效或错侧关节目标")
                left_msg = None
        if left_msg is not None:
            self._left_target = left_msg
            logger.debug(
                f"左手IK关节目标: {np.array2string(np.asarray(left_msg.arm_joint_positions_rad), precision=4)}"
            )
        elif self._left_target is None:
            self._log_no_joint_command(robots.LEFT, self._left_endeff_subscriber._port)

    def _log_no_joint_command(self, hand_side, port):
        current_time = time.time()
        if current_time - self._last_no_joint_command_log_time.get(hand_side, 0.0) < 1.0:
            return
        self._last_no_joint_command_log_time[hand_side] = current_time
        logger.info(
            f"MuJoCo: 暂未收到{hand_side}手robot层IK关节命令，等待 "
            f"topic={SYSMO32_JOINT_COMMAND_TOPIC}, port={port}"
        )

    def stream(self):
        """
        主仿真循环。

        主循环流程：
        1. 以VR_FREQ(30Hz)频率运行
        2. 接收CartesianTarget命令
        3. 对每只手臂：
           a. 将笛卡尔目标转换为MuJoCo位姿格式
           b. 使用IK求解关节角度
           c. 设置仿真中的关节角度
        4. 步进MuJoCo仿真
        5. 渲染可视化窗口（如果启用）
        """
        logger.info("MuJoCo SYSMO-32仿真器启动")

        if self.render:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running():
                    self.timer.start_loop()
                    # 接收目标命令
                    self._receive_targets()
                    # 应用右手目标
                    self._apply_endeff_target(
                        self._right_target,
                        self._right_joint_ids,
                    )
                    # 应用左手目标
                    self._apply_endeff_target(
                        self._left_target,
                        self._left_joint_ids,
                    )
                    # 把关节保持在上一帧/当前目标位置，避免 MuJoCo 物理仿真把手臂自然带偏。 这里会设置：qpos、qvel=0、actuator ctrl
                    self._apply_joint_position_holds()
                    # 给左右臂关节加重力补偿，让手臂不因为重力下垂
                    self._apply_arm_gravity_compensation()
                    # 步进仿真
                    mujoco.mj_step(self.model, self.data)
                    # 因为 mj_step() 后物理状态可能有微小漂移，所以再把关节拉回目标保持位置
                    self._apply_joint_position_holds()
                    # 根据当前 qpos/qvel/ctrl 重新计算 MuJoCo 派生状态，比如 link pose、site pose、传感器等。
                    mujoco.mj_forward(self.model, self.data)
                    # 更新渲染
                    viewer.sync()
                    self.timer.end_loop()
            return

        while True:
            self.timer.start_loop()
            self._receive_targets()
            self._apply_endeff_target(
                self._right_target,
                self._right_joint_ids,
                self._right_endeff_site_id,
            )
            self._apply_endeff_target(
                self._left_target,
                self._left_joint_ids,
                self._left_endeff_site_id,
            )
            self._apply_joint_position_holds()
            self._apply_arm_gravity_compensation()
            mujoco.mj_step(self.model, self.data)
            self._apply_joint_position_holds()
            mujoco.mj_forward(self.model, self.data)
            self.timer.end_loop()

    def cleanup(self):
        """清理资源。"""
        try:
            if hasattr(self, "_right_endeff_subscriber"):
                self._right_endeff_subscriber.stop()
            if hasattr(self, "_left_endeff_subscriber"):
                self._left_endeff_subscriber.stop()
            cleanup_zmq_resources()
            logger.info("MuJoCo仿真器资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")

    def __del__(self):
        self.cleanup()
