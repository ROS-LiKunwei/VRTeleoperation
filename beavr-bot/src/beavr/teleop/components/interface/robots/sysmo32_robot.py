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
import time

import numpy as np

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
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)

# SYSMO-32常量定义
SYSMO32_NUM_JOINTS_PER_ARM = 6
SYSMO32_TOTAL_JOINTS = 12
SYSMO32_HOME_JS = np.zeros(SYSMO32_NUM_JOINTS_PER_ARM, dtype=np.float32)

# SYSMO-32双臂初始位姿（笛卡尔空间，毫米+轴角）
SYSMO32_BIMANUAL_LEFT_HOME = [206, 186, 475, 3.142, 0, 0]
SYSMO32_BIMANUAL_RIGHT_HOME = [206, -186, 475, 3.142, 0, 0]

# SYSMO-32缩放因子（毫米→米）
SYSMO32_SCALE_FACTOR = 1000


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
        rotation = np.eye(3)
        translation = np.array(self._cartesian_position[:3]) / SYSMO32_SCALE_FACTOR
        return np.block([[rotation, translation[:, np.newaxis]], [0, 0, 0, 1]])

    def move_arm_joint(self, joint_angles):
        self._joint_positions = np.array(joint_angles, dtype=np.float32)
        return 0

    def move_arm_cartesian(self, cartesian_pos, duration=3):
        if len(cartesian_pos) == 7:
            pos_m = np.asarray(cartesian_pos[0:3], dtype=np.float32)
            self._cartesian_position[:3] = pos_m * SYSMO32_SCALE_FACTOR
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
            message_type=SessionCommand,
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

        # 录制控制
        self._is_recording_enabled = False

        # 握手协调
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"

        self._handshake_coordinator.start_server(
            subscriber_id=self._handshake_server_id,
            bind_host="*",
            port=robots.TELEOP_HANDSHAKE_PORT + (3 if self._is_right_arm else 4),
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
                    self._latest_commanded_cartesian_position = np.concatenate(
                        [
                            np.asarray(cmd.position_m, dtype=np.float32),
                            np.asarray(cmd.orientation_xyzw, dtype=np.float32),
                        ]
                    )
                    self._latest_commanded_cartesian_timestamp = cmd.timestamp_s

                if self._latest_commanded_cartesian_position is not None:
                    self.move_coords(self._latest_commanded_cartesian_position)

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
        if joint_angles_rad is not None:
            current_state_dict["joint_angles_rad"] = joint_angles_rad

        current_state_dict["timestamp"] = publish_time

        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._state_publish_port,
            topic=self.name,
            data=current_state_dict,
        )

    def __del__(self):
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            self._handshake_coordinator.stop_server(self._handshake_server_id)
        cleanup_zmq_resources()
