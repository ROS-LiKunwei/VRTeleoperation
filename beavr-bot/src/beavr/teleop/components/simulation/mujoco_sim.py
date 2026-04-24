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
import time
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


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
        self.render = render
        self.urdf_path = urdf_path

        # 初始化MuJoCo模型
        self.model = None
        self.data = None
        self._load_model()

        # 初始化ZMQ订阅者
        self._right_endeff_subscriber = ZMQSubscriber(
            host=host,
            port=right_endeff_subscribe_port,
            topic="endeff_coords",
            message_type=CartesianTarget,
        )

        self._left_endeff_subscriber = ZMQSubscriber(
            host=host,
            port=left_endeff_subscribe_port,
            topic="endeff_coords",
            message_type=CartesianTarget,
        )

        # 计时器
        self.timer = FrequencyTimer(robots.VR_FREQ)  # 30Hz

        # 关节索引缓存
        self._left_joint_ids = []
        self._right_joint_ids = []
        self._left_endeff_site_id = None
        self._right_endeff_site_id = None
        self._cache_joint_ids()

        # 目标位姿缓存
        self._left_target: Optional[CartesianTarget] = None
        self._right_target: Optional[CartesianTarget] = None

        # IK参数
        self._ik_max_iter = 100
        self._ik_tolerance = 1e-4

        logger.info(f"MuJoCo SYSMO-32仿真器初始化完成, URDF: {urdf_path}")

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

            # 在left_arm_J6_Link body中添加site
            xml_string = xml_string.replace(
                '<body name="left_arm_J6_Link"',
                '<body name="left_arm_J6_Link">\n'
                '        <site name="left_endeff" pos="0 0.07 0" rgba="0 1 0 1" size="0.02"/>',
            )

            # 在right_arm_J6_Link body中添加site
            xml_string = xml_string.replace(
                '<body name="right_arm_J6_Link"',
                '<body name="right_arm_J6_Link">\n'
                '        <site name="right_endeff" pos="0 -0.07 0" rgba="1 0 0 1" size="0.02"/>',
            )

            self.model = mujoco.MjModel.from_xml_string(xml_string)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_step(self.model, self.data)
            logger.info(
                f"MuJoCo模型加载成功: {self.model.nq} 个自由度, "
                f"{self.model.njnt} 个关节, {self.model.nsite} 个site"
            )
        except Exception as e:
            logger.error(f"MuJoCo模型加载失败: {e}")
            raise

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

        logger.info(
            f"关节ID缓存完成: 左手{len(self._left_joint_ids)}个, "
            f"右手{len(self._right_joint_ids)}个"
        )

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

    def _solve_ik(self, joint_ids, target_pos, target_quat, endeff_site_id=None):
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

        # 获取关节的qpos地址
        qpos_addrs = []
        for jid in joint_ids:
            qpos_addr = self.model.jnt_qposadr[jid]
            qpos_addrs.append(qpos_addr)

        # 保存当前关节角度（用于恢复）
        saved_qpos = self.data.qpos.copy()

        # IK迭代
        for iteration in range(self._ik_max_iter):
            # 前向运动学
            mujoco.mj_forward(self.model, self.data)

            # 获取当前末端位姿
            if endeff_site_id is not None and endeff_site_id >= 0:
                current_pos = self.data.site_xpos[endeff_site_id].copy()
                current_mat = self.data.site_xmat[endeff_site_id].reshape(3, 3).copy()
            else:
                # 使用最后一个关节的body位姿
                last_joint_body_id = self.model.jnt_bodyid[joint_ids[-1]]
                current_pos = self.data.xpos[last_joint_body_id].copy()
                current_mat = self.data.xmat[last_joint_body_id].reshape(3, 3).copy()

            # 计算位置误差
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # 计算姿态误差
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten())
            quat_error = np.zeros(4)
            mujoco.mju_negQuat(quat_error, target_quat)
            mujoco.mju_mulQuat(quat_error, quat_error, current_quat)
            # 将四元数误差转换为轴角
            axis_angle = np.zeros(3)
            mujoco.mju_quat2Vel(axis_angle, quat_error, 1.0)
            ori_error_norm = np.linalg.norm(axis_angle)

            # 检查收敛
            if pos_error_norm < self._ik_tolerance and ori_error_norm < 0.01:
                break

            # 组合误差向量
            error = np.concatenate([pos_error, axis_angle])

            # 计算雅可比矩阵
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))

            if endeff_site_id is not None and endeff_site_id >= 0:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, endeff_site_id)
            else:
                last_joint_body_id = self.model.jnt_bodyid[joint_ids[-1]]
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, last_joint_body_id)

            jacobian = np.vstack([jacp, jacr])

            # 提取相关关节的雅可比列
            joint_dof_addrs = []
            for jid in joint_ids:
                dof_addr = self.model.jnt_dofadr[jid]
                joint_dof_addrs.append(dof_addr)

            jacobian_sub = jacobian[:, joint_dof_addrs]

            # 伪逆方法求解关节角度增量
            damping = 0.1
            jt_j = jacobian_sub.T @ jacobian_sub + damping**2 * np.eye(len(joint_ids))
            delta_q = np.linalg.solve(jt_j, jacobian_sub.T @ error)

            # 更新关节角度
            step_size = 0.5
            for i, addr in enumerate(qpos_addrs):
                self.data.qpos[addr] += step_size * delta_q[i]

        # 提取求解结果
        result = np.array([self.data.qpos[addr] for addr in qpos_addrs])

        # 恢复原始关节角度（IK求解不直接修改仿真状态）
        self.data.qpos[:] = saved_qpos

        return result

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
            target_pos, target_quat = self._cartesian_to_mujoco_pos(
                target.position_m, target.orientation_xyzw
            )

            joint_angles = self._solve_ik(joint_ids, target_pos, target_quat, endeff_site_id)

            if joint_angles is not None:
                for i, jid in enumerate(joint_ids):
                    qpos_addr = self.model.jnt_qposadr[jid]
                    # 限制关节角度在URDF定义的范围内
                    low = self.model.jnt_range[jid, 0]
                    high = self.model.jnt_range[jid, 1]
                    clamped = np.clip(joint_angles[i], low, high)
                    self.data.qpos[qpos_addr] = clamped

        except Exception as e:
            logger.error(f"应用末端目标失败: {e}")

    def _receive_targets(self):
        """
        从ZMQ订阅者接收笛卡尔空间目标命令。

        分别接收右手和左手的CartesianTarget命令，
        并缓存到_left_target和_right_target。
        """
        # 接收右手目标
        right_msg = self._right_endeff_subscriber.recv_keypoints()
        if right_msg is not None:
            self._right_target = right_msg
            logger.debug(
                f"右手目标: pos=({right_msg.position_m[0]:.3f}, "
                f"{right_msg.position_m[1]:.3f}, {right_msg.position_m[2]:.3f})"
            )

        # 接收左手目标
        left_msg = self._left_endeff_subscriber.recv_keypoints()
        if left_msg is not None:
            self._left_target = left_msg
            logger.debug(
                f"左手目标: pos=({left_msg.position_m[0]:.3f}, "
                f"{left_msg.position_m[1]:.3f}, {left_msg.position_m[2]:.3f})"
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
                        self._right_endeff_site_id,
                    )

                    # 应用左手目标
                    self._apply_endeff_target(
                        self._left_target,
                        self._left_joint_ids,
                        self._left_endeff_site_id,
                    )

                    # 步进仿真
                    mujoco.mj_step(self.model, self.data)

                    # 更新渲染
                    viewer.sync()

                    self.timer.end_loop()
        else:
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

                mujoco.mj_step(self.model, self.data)

                self.timer.end_loop()

    def cleanup(self):
        """清理资源。"""
        try:
            if hasattr(self, '_right_endeff_subscriber'):
                self._right_endeff_subscriber.stop()
            if hasattr(self, '_left_endeff_subscriber'):
                self._left_endeff_subscriber.stop()
            cleanup_zmq_resources()
            logger.info("MuJoCo仿真器资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")

    def __del__(self):
        self.cleanup()
