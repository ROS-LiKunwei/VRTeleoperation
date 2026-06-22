"""
SYSMO-32双臂机器人接口模块

本模块实现了SYSMO-32双臂机器人的BeaVR-bot接口，支持仿真模式和实机模式。
SYSMO-32是6自由度双臂机器人，每臂6个旋转关节，共12个关节。

与XArm7的区别：
    - SYSMO-32每臂6个关节（XArm7是7个）
    - SYSMO-32双臂固定在同一个base_link上（XArm7是单臂独立）
    - SYSMO-32的关节限位与XArm7不同
    - SYSMO-32的初始位姿与XArm7不同

关节顺序（与URDF一致）：
    左臂：left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
          left_elbow, left_wrist_yaw, left_wrist_pitch
    右臂：right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
          right_elbow, right_wrist_yaw, right_wrist_pitch

数据流位置：
    xarm7_operator.py → [本模块: sysmo32_robot.py] → 物理机器人/MuJoCo仿真
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from beavr.teleop.common.math.orientation import quat_to_axis_angle
from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.ops import Ops
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.interface.interface_types import (
    CartesianState,
    CommandedCartesianState,
    Sysmo32JointCommand,
)
from beavr.teleop.components.interface.robots.arm_command_publisher import MinSnapTargetPublisher
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)

# SYSMO-32常量定义
SYSMO32_NUM_JOINTS_PER_ARM = 6
SYSMO32_TOTAL_JOINTS = 12
SYSMO32_HOME_JS = np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32)

# SYSMO-32双臂初始位姿（笛卡尔空间，毫米+轴角）
SYSMO32_BIMANUAL_LEFT_HOME = [278.7504, 445.4513, 99.2101, 0.2427, 0.7903, -1.3631]
SYSMO32_BIMANUAL_RIGHT_HOME = [278.7504, -445.4513, 99.2101, -0.2427, 0.7903, 1.3631]

# SYSMO-32缩放因子（毫米→米）
SYSMO32_SCALE_FACTOR = 1000
SYSMO32_HAND_OPEN_COMMAND = 1
SYSMO32_HAND_GRASP_COMMAND = 2
SYSMO32_ARM_COMMAND_TOPIC = "/sysmo_left_arm_controller/commands"
SYSMO32_JOINT_COMMAND_TOPIC = "sysmo32_joint_command"
SYSMO32_LEFT_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
]
SYSMO32_RIGHT_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
]
SYSMO32_LEFT_ENDEFF_SITE = "left_endeff"
SYSMO32_RIGHT_ENDEFF_SITE = "right_endeff"


class Sysmo32RosBridge:
    """Optional ROS2 publishers for the SYSMO-32 real robot interface."""

    _shared_left_arm_joints = np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float64)
    _shared_right_arm_joints = np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float64)

    def __init__(
        self,
        node_name: str,
        is_right_arm: bool,
        left_hand_topic: str = "/left_topic_to_hand",
        right_hand_topic: str = "/right_topic_to_hand",
        arm_command_topic: str = SYSMO32_ARM_COMMAND_TOPIC,
        min_snap_target_topic: str = "/min_snap/target",
    ):
        self._enabled = False
        self._last_hand_command: Optional[int] = None
        self._hand_topic = right_hand_topic if is_right_arm else left_hand_topic
        self._arm_command_topic = arm_command_topic
        self._min_snap_target_topic = min_snap_target_topic
        self._is_right_arm = is_right_arm

        try:
            import rclpy
            from min_snap.msg import MinSnapTarget
            from std_msgs.msg import Int32
        except ImportError as exc:
            logger.warning(f"ROS2 bridge disabled for {node_name}: {exc}")
            self._rclpy = None
            self._node = None
            self._hand_msg_type = None
            self._min_snap_msg_type = None
            return

        self._rclpy = rclpy
        self._hand_msg_type = Int32
        self._min_snap_msg_type = MinSnapTarget
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node(node_name)
        self._hand_publisher = self._node.create_publisher(Int32, self._hand_topic, 10)
        self._min_snap_target_publisher = MinSnapTargetPublisher(
            self._node,
            MinSnapTarget,
            self._min_snap_target_topic,
            10,
        )
        self._enabled = True
        logger.info(
            f"ROS2 bridge enabled for {node_name}, hand topic={self._hand_topic}, "
            f"min_snap target={self._min_snap_target_topic}, min_snap output={self._arm_command_topic}"
        )

    @property
    def enabled(self):
        return self._enabled

    def publish_hand_command(self, command: Optional[int], force: bool = False):
        if not self._enabled or command is None:
            return

        command = int(command)
        if command not in (SYSMO32_HAND_OPEN_COMMAND, SYSMO32_HAND_GRASP_COMMAND):
            logger.warning(f"忽略未知灵巧手命令: {command}")
            return

        if not force and command == self._last_hand_command:
            return

        msg = self._hand_msg_type()
        msg.data = command
        logger.debug(
            f"[真实机器人发送前] ROS2 hand msg topic={self._hand_topic}, type=std_msgs/Int32, data={msg.data}"
        )
        self._hand_publisher.publish(msg)
        self._last_hand_command = command
        logger.info(f"发布灵巧手命令到 {self._hand_topic}: {command}")

    def publish_arm_command(self, arm_joint_positions_rad):
        if not self._enabled or arm_joint_positions_rad is None:
            return

        joints = np.asarray(arm_joint_positions_rad, dtype=np.float64).reshape(-1)
        if joints.size != SYSMO32_NUM_JOINTS_PER_ARM or not np.all(np.isfinite(joints)):
            logger.warning(f"忽略无效SYSMO-32关节命令: {joints}")
            return

        if self._is_right_arm:
            Sysmo32RosBridge._shared_right_arm_joints = joints.copy()
        else:
            Sysmo32RosBridge._shared_left_arm_joints = joints.copy()

        logger.debug(
            "[真实机器人发送前] ROS2 min_snap target topic=%s left=%s right=%s",
            self._min_snap_target_topic,
            Sysmo32RosBridge._shared_left_arm_joints,
            Sysmo32RosBridge._shared_right_arm_joints,
        )
        self._min_snap_target_publisher.publish(
            Sysmo32RosBridge._shared_left_arm_joints,
            Sysmo32RosBridge._shared_right_arm_joints,
            expected_duration_s=0.5,
            max_velocity_rad_s=3.0,
            max_acceleration_rad_s2=10.0,
        )

    def close(self):
        if not self._enabled or self._node is None:
            return
        self._node.destroy_node()
        self._enabled = False


def build_sysmo32_full_command(left_arm_joints, right_arm_joints, speed_mode=0.0):
    left = np.asarray(left_arm_joints, dtype=np.float64).reshape(SYSMO32_NUM_JOINTS_PER_ARM)
    right = np.asarray(right_arm_joints, dtype=np.float64).reshape(SYSMO32_NUM_JOINTS_PER_ARM)
    return [
        *[float(value) for value in left],
        *[float(value) for value in right],
        float(speed_mode),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


class Sysmo32MujocoIKSolver:
    """MuJoCo Jacobian IK used by Sysmo32Robot before real/sim command output."""

    def __init__(self, urdf_path: str, is_right_arm: bool):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("SYSMO-32 IK requires the mujoco Python package") from exc

        self.mujoco = mujoco
        self.is_right_arm = is_right_arm
        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.model = None
        self.data = None
        self.joint_names = SYSMO32_RIGHT_JOINT_NAMES if is_right_arm else SYSMO32_LEFT_JOINT_NAMES
        self.endeff_site = SYSMO32_RIGHT_ENDEFF_SITE if is_right_arm else SYSMO32_LEFT_ENDEFF_SITE
        self.endeff_body = "right_arm_J6_Link" if is_right_arm else "left_arm_J6_Link"
        self.joint_ids = []
        self.qpos_addrs = []
        self.dof_addrs = []
        self.endeff_site_id = None
        self.max_iter = 80
        self.tolerance = 1e-3
        self.orientation_tolerance = 0.05
        self.orientation_weight = 0.25
        self.joint_step_limit = 0.08
        self._load_model()

    def _resolve_urdf_path(self, urdf_path):
        candidate = Path(urdf_path)
        if candidate.exists():
            return str(candidate)

        repo_root = Path(__file__).resolve().parents[6]
        for base in (repo_root, repo_root / "beavr-bot"):
            resolved = base / urdf_path
            if resolved.exists():
                return str(resolved)

        raise FileNotFoundError(f"SYSMO-32 IK URDF not found: {urdf_path}")

    def _load_model(self):
        mujoco = self.mujoco
        temp_model = mujoco.MjModel.from_xml_path(self.urdf_path)
        xml_path = f"/tmp/sysmo32_ik_{os.getpid()}_{'right' if self.is_right_arm else 'left'}.xml"
        mujoco.mj_saveLastXML(xml_path, temp_model)
        with open(xml_path, "r") as file:
            xml_string = file.read()

        xml_string = self._resolve_exported_mesh_paths(xml_string)
        site_pos = "0 -0.05 0" if self.is_right_arm else "0 0.05 0"
        xml_string = self._insert_site_into_body(
            xml_string,
            self.endeff_body,
            f'<site name="{self.endeff_site}" pos="{site_pos}" size="0.02"/>',
        )

        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self._cache_ids()
        mujoco.mj_forward(self.model, self.data)
        logger.info(
            f"SYSMO-32 IK初始化完成 side={'right' if self.is_right_arm else 'left'}, urdf={self.urdf_path}"
        )

    def _resolve_exported_mesh_paths(self, xml_string):
        urdf_dir = Path(self.urdf_path).resolve().parent

        def resolve_mesh_file(match):
            mesh_path = Path(match.group(1))
            if not mesh_path.is_absolute():
                mesh_path = (urdf_dir / mesh_path).resolve()
            return f'file="{mesh_path}"'

        return re.sub(r'file="([^"]+\.(?:STL|stl))"', resolve_mesh_file, xml_string)

    def _insert_site_into_body(self, xml_string, body_name, site_xml):
        pattern = rf'(<body name="{re.escape(body_name)}"[^>]*>)'
        updated_xml, count = re.subn(pattern, rf"\1\n        {site_xml}", xml_string, count=1)
        if count != 1:
            raise ValueError(f"未找到MuJoCo body，无法添加site: {body_name}")
        return updated_xml

    def _cache_ids(self):
        mujoco = self.mujoco
        self.joint_ids = []
        self.qpos_addrs = []
        self.dof_addrs = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"SYSMO-32 IK joint not found: {joint_name}")
            self.joint_ids.append(joint_id)
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])
            self.dof_addrs.append(self.model.jnt_dofadr[joint_id])

        self.endeff_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.endeff_site)
        if self.endeff_site_id < 0:
            raise ValueError(f"SYSMO-32 IK site not found: {self.endeff_site}")

    def solve(self, position_m, orientation_xyzw):
        if self.model is None or self.data is None:
            return None

        target_pos = np.asarray(position_m, dtype=np.float64)
        target_quat = self._xyzw_to_wxyz(orientation_xyzw)
        if target_pos.shape != (3,) or target_quat.shape != (4,):
            return None
        if not np.all(np.isfinite(target_pos)) or not np.all(np.isfinite(target_quat)):
            return None

        mujoco = self.mujoco
        best_qpos = np.array([self.data.qpos[addr] for addr in self.qpos_addrs], dtype=np.float64)
        best_score = np.inf
        converged = False
        pos_error_norm = np.inf
        ori_error_norm = np.inf

        for _ in range(self.max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.endeff_site_id].copy()
            current_mat = self.data.site_xmat[self.endeff_site_id].reshape(3, 3).copy()
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten())

            pos_error = target_pos - current_pos
            pos_error_norm = float(np.linalg.norm(pos_error))

            inv_current_quat = np.zeros(4)
            quat_error = np.zeros(4)
            mujoco.mju_negQuat(inv_current_quat, current_quat)
            mujoco.mju_mulQuat(quat_error, target_quat, inv_current_quat)
            axis_angle = np.zeros(3)
            mujoco.mju_quat2Vel(axis_angle, quat_error, 1.0)
            ori_error_norm = float(np.linalg.norm(axis_angle))

            score = pos_error_norm + self.orientation_weight * ori_error_norm
            if score < best_score:
                best_score = score
                best_qpos = np.array([self.data.qpos[addr] for addr in self.qpos_addrs])

            if pos_error_norm < self.tolerance and ori_error_norm < self.orientation_tolerance:
                converged = True
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.endeff_site_id)
            jacobian = np.vstack([jacp, self.orientation_weight * jacr])
            error = np.concatenate([pos_error, self.orientation_weight * axis_angle])
            jacobian_sub = jacobian[:, self.dof_addrs]

            damping = 0.1
            jt_j = jacobian_sub.T @ jacobian_sub + damping**2 * np.eye(len(self.joint_ids))
            delta_q = np.linalg.solve(jt_j, jacobian_sub.T @ error)
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > self.joint_step_limit:
                delta_q *= self.joint_step_limit / delta_norm

            for i, addr in enumerate(self.qpos_addrs):
                self.data.qpos[addr] += delta_q[i]
                low, high = self.model.jnt_range[self.joint_ids[i]]
                self.data.qpos[addr] = np.clip(self.data.qpos[addr], low, high)

        for value, addr in zip(best_qpos, self.qpos_addrs, strict=False):
            self.data.qpos[addr] = value
        mujoco.mj_forward(self.model, self.data)

        logger.debug(
            f"SYSMO-32 IK result side={'right' if self.is_right_arm else 'left'} "
            f"converged={converged}, pos_err={pos_error_norm:.5f}, ori_err={ori_error_norm:.5f}, "
            f"joints={np.array2string(best_qpos, precision=5, separator=', ')}"
        )
        return best_qpos.astype(np.float64)

    def _xyzw_to_wxyz(self, orientation_xyzw):
        quat_xyzw = np.asarray(orientation_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat_xyzw)
        if norm > 1e-6:
            quat_xyzw = quat_xyzw / norm
        else:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _axis_angle_to_matrix(axis_angle):
    rotvec = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-9:
        return np.eye(3)

    x, y, z = rotvec / angle
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


class MockSysmo32Control:
    """
    SYSMO-32仿真控制器（Mock模式）。

    在没有物理机器人时使用，模拟SYSMO-32双臂的运动响应。
    每臂6个关节，接收笛卡尔空间命令后更新模拟状态。
    """

    def __init__(self, ip="127.0.0.1", simulation_mode=True, is_right_arm=True):
        self.simulation_mode = simulation_mode
        self._is_right_arm = is_right_arm

        # 模拟关节位置（6个关节）
        home = SYSMO32_BIMANUAL_RIGHT_HOME if is_right_arm else SYSMO32_BIMANUAL_LEFT_HOME
        self._joint_positions = np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32)
        self._cartesian_position = np.array(home, dtype=np.float32)

    def _init_control(self):
        return 0

    def get_arm_states(self):
        return {
            "joint_position": self._joint_positions,
            "joint_velocity": np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32),
            "joint_torque": np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32),
            "timestamp": time.time(),
        }

    def get_arm_position(self):
        return self._joint_positions

    def get_arm_velocity(self):
        return np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32)

    def get_arm_torque(self):
        return np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32)

    def get_arm_cartesian_coords(self):
        return self._cartesian_position

    def get_cartesian_state(self):
        return {
            "cartesian_position": self._cartesian_position,
            "timestamp": time.time(),
        }

    def get_arm_pose(self):
        rotation = _axis_angle_to_matrix(self._cartesian_position[3:6])
        translation = np.array(self._cartesian_position[:3]) / SYSMO32_SCALE_FACTOR
        return np.block([[rotation, translation[:, np.newaxis]], [0, 0, 0, 1]])

    def move_arm_joint(self, joint_angles):
        self._joint_positions = np.array(joint_angles, dtype=np.float32)
        return 0

    def move_arm_cartesian(self, cartesian_pos, duration=3):
        if len(cartesian_pos) == 7:
            pos_m = np.asarray(cartesian_pos[0:3], dtype=np.float32)
            quat_xyzw = np.asarray(cartesian_pos[3:7], dtype=np.float32)
            self._cartesian_position[:3] = pos_m * SYSMO32_SCALE_FACTOR
            self._cartesian_position[3:6] = quat_to_axis_angle(quat_xyzw)
        elif len(cartesian_pos) == 6:
            self._cartesian_position = np.asarray(cartesian_pos, dtype=np.float32)
        return 0

    def arm_control(self, cartesian_pos):
        return self.move_arm_cartesian(cartesian_pos)

    def home_arm(self):
        home = SYSMO32_BIMANUAL_RIGHT_HOME if self._is_right_arm else SYSMO32_BIMANUAL_LEFT_HOME
        self._joint_positions = np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32)
        self._cartesian_position = np.array(home, dtype=np.float32)
        return 0

    @property
    def robot(self):
        class MockRobot:
            def set_mode_and_state(self, mode, state):
                return True

        return MockRobot()


class Sysmo32Robot(RobotWrapper):
    """
    SYSMO-32双臂机器人遥操作接口和状态发布器。

    数据流角色：
        本类是遥操作系统数据流的机器人接口层，
        负责接收xarm7_operator发布的CartesianTarget命令，
        驱动SYSMO-32双臂机器人运动（实机或仿真）。

    与XArm7Robot的区别：
        - 每臂6个关节（XArm7是7个）
        - 双臂共享同一个base_link
        - 使用MockSysmo32Control替代DexArmControl
        - 关节状态为6维向量

    订阅Topic：
        - 'endeff_coords': 笛卡尔空间目标命令（来自xarm7_operator）
        - 'reset': 重置命令
        - 'home': 归零命令
        - 'pause': 暂停/恢复命令

    发布Topic：
        - 'endeff_homo': 末端执行器齐次变换矩阵（用于Operator重置）
        - '{robot_name}': 机器人状态字典（用于数据记录）
    """

    def __init__(
        self,
        host,
        endeff_subscribe_port,
        joint_subscribe_port,
        home_subscribe_port,
        reset_subscribe_port,
        teleoperation_state_port,
        robot_ip="127.0.0.1",
        is_right_arm=True,
        simulation_mode: bool = True,
        enable_ros2_bridge: bool = False,
        left_hand_topic: str = "/left_topic_to_hand",
        right_hand_topic: str = "/right_topic_to_hand",
        arm_command_topic: str = SYSMO32_ARM_COMMAND_TOPIC,
        ik_urdf_path: str = "robots/sysmo_description/urdf/sysmo32.urdf",
        endeff_publish_port: int = 10009,
        state_publish_port: int = 10010,
        **kwargs,
    ):
        """
        初始化SYSMO-32机器人接口。

        Args:
            host: 网络主机地址（ZMQ通信地址）。
            endeff_subscribe_port: 末端执行器命令订阅端口。
            joint_subscribe_port: 关节命令订阅端口。
            home_subscribe_port: 归零命令订阅端口。
            reset_subscribe_port: 重置命令订阅端口。
            teleoperation_state_port: 遥操作状态端口。
            robot_ip: 机器人IP地址（实机模式使用）。
            is_right_arm: 是否为右臂（True）或左臂（False）。
            simulation_mode: 是否为仿真模式（默认True）。
            enable_ros2_bridge: 是否启用ROS2实机桥接。
            left_hand_topic: 左灵巧手ROS2命令话题。
            right_hand_topic: 右灵巧手ROS2命令话题。
            arm_command_topic: SYSMO-32 ROS2 Float64MultiArray机械臂命令话题。
            ik_urdf_path: 用于MuJoCo Jacobian IK的SYSMO-32 URDF路径。
            endeff_publish_port: 末端执行器数据发布端口。
            state_publish_port: 机器人状态发布端口。
        """
        if not endeff_publish_port:
            raise ValueError("Sysmo32Robot requires an 'endeff_publish_port'")
        if not state_publish_port:
            raise ValueError("Sysmo32Robot requires a 'state_publish_port'")

        # 使用Mock控制器（SYSMO-32暂无实机SDK）
        self._controller = MockSysmo32Control(
            ip=robot_ip,
            simulation_mode=simulation_mode,
            is_right_arm=is_right_arm,
        )

        self._is_right_arm = is_right_arm
        self._data_frequency = robots.VR_FREQ
        self._ros2_bridge = None
        if enable_ros2_bridge or not simulation_mode:
            self._ros2_bridge = Sysmo32RosBridge(
                node_name=f"{self.name}_ros_bridge",
                is_right_arm=is_right_arm,
                left_hand_topic=left_hand_topic,
                right_hand_topic=right_hand_topic,
                arm_command_topic=arm_command_topic,
            )
        self._ik_solver = Sysmo32MujocoIKSolver(ik_urdf_path, is_right_arm=is_right_arm)

        # ZMQ订阅者
        self._cartesian_coords_subscriber = ZMQSubscriber(
            host=host,
            port=endeff_subscribe_port,
            topic="endeff_coords",
            message_type=CartesianTarget,
        )

        self._reset_subscriber = ZMQSubscriber(
            host=host,
            port=reset_subscribe_port,
            topic="reset",
            message_type=SessionCommand,
        )

        self._home_subscriber = ZMQSubscriber(
            host=host,
            port=home_subscribe_port,
            topic="home",
        )

        self._arm_teleop_state_subscriber = Ops(
            arm_teleop_state_subscriber=ZMQSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic="pause",
                message_type=SessionCommand,
            )
        )

        self._subscribers = {
            "cartesian_coords": self._cartesian_coords_subscriber,
            "reset": self._reset_subscriber,
            "home": self._home_subscriber,
            "teleop_state": self._arm_teleop_state_subscriber.get_arm_teleop_state,
        }

        # ZMQ发布者
        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._publisher_host = host
        self._endeff_publish_port = endeff_publish_port
        self._state_publish_port = state_publish_port

        # 状态缓存
        self._latest_cartesian_coords = None
        self._latest_joint_state = None
        self._latest_cartesian_state_timestamp = 0
        self._latest_joint_state_timestamp = 0
        self._latest_commanded_cartesian_position = None
        self._latest_commanded_cartesian_timestamp = 0.0
        self._latest_commanded_joint_position = None
        self._latest_hand_command = None
        self._last_real_robot_command_log_time = 0.0
        self._real_robot_command_log_interval = 1.0
        self._last_no_cartesian_cmd_log_time = 0.0
        self._last_ik_failure_log_time = 0.0

        # 录制控制
        self._is_recording_enabled = False

        # 握手协调
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"

        self._handshake_coordinator.start_server(
            subscriber_id=self._handshake_server_id,
            bind_host="*",
            port=robots.TELEOP_HANDSHAKE_PORT + (103 if self._is_right_arm else 104),
        )
        logger.info(f"Handshake server started for {self.name}")

        self._is_homed = False

    @property
    def name(self):
        return f"sysmo32_{'right' if self._is_right_arm else 'left'}"

    @property
    def recorder_functions(self):
        return {
            "joint_states": self.get_joint_state,
            "operator_cartesian_states": self.get_cartesian_state_from_operator,
            "sysmo32_cartesian_states": self.get_robot_actual_cartesian_position,
            "commanded_cartesian_state": self.get_cartesian_commanded_position,
            "joint_angles_rad": self.get_joint_position,
        }

    @property
    def data_frequency(self):
        return self._data_frequency

    def get_joint_state(self):
        arm_states = self._controller.get_arm_states()
        if arm_states is None or arm_states.get("joint_position") is None:
            return None
        return {
            "joint_position": list(np.array(arm_states["joint_position"], dtype=np.float32)),
            "timestamp": arm_states.get("timestamp", time.time()),
        }

    def get_joint_velocity(self):
        return self._controller.get_arm_velocity()

    def get_joint_torque(self):
        return self._controller.get_arm_torque()

    def get_cartesian_state(self):
        return self._controller.get_cartesian_state()

    def get_joint_position(self):
        arm_position = self._controller.get_arm_position()
        if arm_position is None:
            return None
        return list(np.array(arm_position, dtype=np.float32))

    def get_cartesian_position(self):
        return self._controller.get_arm_cartesian_coords()

    def reset(self):
        return self._controller._init_control()

    def get_teleop_state(self):
        return self._arm_teleop_state_subscriber.get_arm_teleop_state()

    def get_pose(self):
        return self._controller.get_arm_pose()

    def home(self):
        return self._controller.home_arm()

    def move(self, input_angles):
        self._controller.move_arm_joint(input_angles)

    def move_coords(self, cartesian_coords, duration=3):
        self._controller.move_arm_cartesian(cartesian_coords, duration=duration)

    def arm_control(self, cartesian_coords):
        self._controller.arm_control(cartesian_coords)

    def move_velocity(self, input_velocity_values, duration):
        pass

    def get_cartesian_state_from_operator(self):
        if self._latest_cartesian_coords is None:
            return None
        position = tuple(np.asarray(self._latest_cartesian_coords, dtype=np.float32).tolist())
        return CartesianState(position_m=position, timestamp_s=self._latest_cartesian_state_timestamp)

    def get_joint_state_from_operator(self):
        if self._latest_joint_state is None:
            return None
        return {
            "joint_position": list(np.array(self._latest_joint_state, dtype=np.float32)),
            "timestamp": self._latest_joint_state_timestamp,
        }

    def get_cartesian_commanded_position(self):
        if self._latest_commanded_cartesian_position is None:
            return None
        return CommandedCartesianState(
            commanded_cartesian_position=self._latest_commanded_cartesian_position.tolist()
            if isinstance(self._latest_commanded_cartesian_position, np.ndarray)
            else list(self._latest_commanded_cartesian_position),
            timestamp_s=self._latest_commanded_cartesian_timestamp,
        )

    def get_robot_actual_cartesian_position(self):
        cartesian_state = self.get_cartesian_position()
        position = tuple(np.asarray(cartesian_state, dtype=np.float32).tolist())
        return CartesianState(position_m=position, timestamp_s=time.time())

    def get_robot_actual_joint_position(self):
        return self.get_joint_state()

    def send_robot_pose(self):
        pose_homo = self._controller.get_arm_pose()
        try:
            h_matrix = tuple(tuple(float(x) for x in row) for row in pose_homo)
            self._publisher_manager.publish(
                host=self._publisher_host,
                port=self._endeff_publish_port,
                topic="endeff_homo",
                data=CartesianState(
                    timestamp_s=time.time(),
                    h_matrix=h_matrix,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to publish robot pose for {self.name}: {e}")

    def check_reset(self):
        reset_bool = self._reset_subscriber.recv_keypoints()
        return reset_bool is not None

    def check_home(self):
        home_bool = self._home_subscriber.recv_keypoints()
        if home_bool == robots.ARM_TELEOP_STOP:
            return True
        elif home_bool == robots.ARM_TELEOP_CONT:
            return False
        return False

    def stream(self):
        """
        主流循环：接收笛卡尔命令并驱动机器人运动。

        流程：
        1. 归零机器人
        2. 进入SERVO-READY模式
        3. 循环：
           a. 检查归零/重置命令
           b. 检查遥操作状态（暂停/恢复）
           c. 接收CartesianTarget命令
           d. 驱动机器人运动
           e. 发布当前状态
        """
        self.home()

        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()

        while True:
            current_time = time.time()

            if current_time >= next_frame_time:
                next_frame_time = current_time + target_interval

                if self.check_home() and not self._is_homed:
                    self.home()
                    self._is_homed = True
                    self.send_robot_pose()
                elif not self.check_home() and self._is_homed:
                    self._is_homed = False

                if self.check_reset():
                    self.send_robot_pose()

                if self.get_teleop_state() == robots.ARM_TELEOP_STOP:
                    continue

                msg = self._cartesian_coords_subscriber.recv_keypoints()
                cmd = msg
                if cmd is not None:
                    logger.debug(
                        f"{self.name}: 收到operator 7维位姿命令 "
                        f"pos={cmd.position_m}, quat={cmd.orientation_xyzw}, hand_command={cmd.hand_command}"
                    )
                    self._latest_hand_command = cmd.hand_command
                    self.publish_hand_command(cmd.hand_command)
                    self._latest_commanded_cartesian_position = np.concatenate(
                        [
                            np.asarray(cmd.position_m, dtype=np.float32),
                            np.asarray(cmd.orientation_xyzw, dtype=np.float32),
                        ]
                    )
                    self._latest_commanded_cartesian_timestamp = cmd.timestamp_s
                    joint_angles = self.solve_ik(cmd)
                    if joint_angles is not None:
                        self._latest_commanded_joint_position = joint_angles
                        self.publish_joint_command(joint_angles, cmd)
                        self.publish_arm_command(joint_angles)
                else:
                    self._log_no_cartesian_command()

                if self._latest_commanded_joint_position is not None:
                    self.move(self._latest_commanded_joint_position)

                self.publish_current_state()

                sleep_time = max(0, next_frame_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def publish_current_state(self):
        """
        收集并发布机器人当前状态。

        发布的状态字典包含：
        - joint_states: 关节位置和速度
        - operator_cartesian_states: Operator计算的笛卡尔目标
        - sysmo32_cartesian_states: 机器人实际笛卡尔位置
        - commanded_cartesian_state: 命令的笛卡尔位姿
        - joint_angles_rad: 关节角度（弧度）
        """
        publish_time = time.time()

        joint_states = self.get_joint_state()
        operator_cart = self.get_cartesian_state_from_operator()
        robot_cart = self.get_robot_actual_cartesian_position()
        commanded_cart = self.get_cartesian_commanded_position()
        joint_angles_rad = self.get_joint_position()

        current_state_dict = {}
        if joint_states is not None:
            current_state_dict["joint_states"] = joint_states
        if operator_cart is not None:
            current_state_dict["operator_cartesian_states"] = operator_cart.to_dict()
        if robot_cart is not None:
            current_state_dict["sysmo32_cartesian_states"] = robot_cart.to_dict()
        if commanded_cart is not None:
            current_state_dict["commanded_cartesian_state"] = commanded_cart.to_dict()
        if self._latest_commanded_joint_position is not None:
            current_state_dict["commanded_joint_angles_rad"] = [
                float(value) for value in self._latest_commanded_joint_position
            ]
        if joint_angles_rad is not None:
            current_state_dict["joint_angles_rad"] = joint_angles_rad
        if self._latest_hand_command is not None:
            current_state_dict["hand_command"] = int(self._latest_hand_command)

        current_state_dict["timestamp"] = publish_time

        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._state_publish_port,
            topic=self.name,
            data=current_state_dict,
        )

    def publish_hand_command(self, hand_command):
        if self._ros2_bridge is None:
            return
        self._ros2_bridge.publish_hand_command(hand_command)

    def publish_arm_command(self, joint_angles):
        if self._ros2_bridge is None:
            return
        self._ros2_bridge.publish_arm_command(joint_angles)

    def solve_ik(self, cmd: CartesianTarget):
        joint_angles = self._ik_solver.solve(cmd.position_m, cmd.orientation_xyzw)
        if joint_angles is None:
            self._log_ik_failure("IK逆解返回None", cmd)
            return None
        if joint_angles.shape != (SYSMO32_NUM_JOINTS_PER_ARM,) or not np.all(np.isfinite(joint_angles)):
            self._log_ik_failure(f"IK逆解结果无效: {joint_angles}", cmd)
            return None
        return joint_angles

    def publish_joint_command(self, joint_angles, cmd: CartesianTarget):
        left = joint_angles if not self._is_right_arm else np.zeros(SYSMO32_NUM_JOINTS_PER_ARM)
        right = joint_angles if self._is_right_arm else np.zeros(SYSMO32_NUM_JOINTS_PER_ARM)
        full_command = build_sysmo32_full_command(left, right)
        command = Sysmo32JointCommand(
            timestamp_s=time.time(),
            hand_side=cmd.hand_side,
            arm_joint_positions_rad=tuple(float(value) for value in joint_angles),
            full_command=tuple(full_command),
            hand_command=cmd.hand_command,
        )
        self._log_real_robot_command(joint_angles, command)
        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._endeff_publish_port,
            topic=SYSMO32_JOINT_COMMAND_TOPIC,
            data=command,
        )
        logger.debug(
            f"{self.name}: 已发布IK关节命令 topic={SYSMO32_JOINT_COMMAND_TOPIC}, "
            f"port={self._endeff_publish_port}"
        )

    def _log_no_cartesian_command(self):
        current_time = time.time()
        if current_time - self._last_no_cartesian_cmd_log_time < 1.0:
            return
        self._last_no_cartesian_cmd_log_time = current_time
        logger.debug(
            f"{self.name}: 暂未收到operator 7维位姿命令，等待 topic=endeff_coords, "
            f"port={self._cartesian_coords_subscriber._port}"
        )

    def _log_ik_failure(self, reason, cmd: CartesianTarget):
        current_time = time.time()
        if current_time - self._last_ik_failure_log_time < 1.0:
            return
        self._last_ik_failure_log_time = current_time
        logger.warning(
            f"{self.name}: {reason}，跳过本帧。输入7维 pos={cmd.position_m}, quat={cmd.orientation_xyzw}"
        )

    def _log_real_robot_command(self, joint_angles, command: Sysmo32JointCommand):
        current_time = time.time()
        if current_time - self._last_real_robot_command_log_time < self._real_robot_command_log_interval:
            return

        self._last_real_robot_command_log_time = current_time
        joints = np.array2string(np.asarray(joint_angles, dtype=np.float32), precision=5, separator=", ")
        logger.debug(
            f"[真实机器人发送前] {self.name} IK joint command "
            f"arm_joint_positions_rad={joints}, full_command={list(command.full_command)}, "
            f"hand_command={command.hand_command}, timestamp_s={command.timestamp_s:.6f}"
        )

    def __del__(self):
        if hasattr(self, "_ros2_bridge") and self._ros2_bridge is not None:
            self._ros2_bridge.close()
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            self._handshake_coordinator.stop_server(self._handshake_server_id)
        cleanup_zmq_resources()
