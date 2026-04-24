import logging
import time
from copy import deepcopy as copy
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from beavr.teleop.common.logging.logger import PoseLogger
from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import (
    SerializationError,
    cleanup_zmq_resources,
    get_global_context,
)
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components.detector.detector_types import (
    ButtonEvent,
    InputFrame,
    SessionCommand,
)
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.operator import CartesianTarget
from beavr.teleop.components.operator.operator_base import Operator
from beavr.teleop.components.operator.solvers.filters import CompStateFilter
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class XArmOperator(Operator):
    """
    XArm 机器人遥操作基类。
    负责处理 VR 手部追踪数据的接收、坐标系转换、滤波平滑以及发布机器人目标位姿。
    特定的手臂配置（例如，左/右）应继承此类，并提供适当的变换矩阵。
    """

    def __init__(
        self,
        operator_name: str,
        host: str,
        transformed_keypoints_port: int,
        stream_configs: Dict[str, Any],
        stream_oculus: bool,
        endeff_publish_port: int,
        endeff_subscribe_port: int,
        moving_average_limit: int,
        h_r_v: np.ndarray,  # Transformation matrix Robot base to VR base
        h_t_v: np.ndarray,  # Transformation matrix Hand Tracking base to VR base
        use_filter: bool = True,
        arm_resolution_port: Optional[int] = None,
        teleoperation_state_port: Optional[int] = None,
        logging_config: Optional[Dict[str, Any]] = None,
        hand_side: str = robots.RIGHT,
    ):
        """
        Initializes the XArmOperator.

        Args:
            operator_name: Name for this operator instance (e.g., 'xarm7_right_operator').
            host: Network host address for ZMQ communication.
            transformed_keypoints_port: 用于接收变换后的手部关键点的端口.
            stream_configs: 数据流配置信息.
            stream_oculus: 指示是否使用 Oculus 推流的标志位.
            endeff_publish_port: 用于发布末端执行器（机器人手部）指令的端口.
            endeff_subscribe_port: 用于订阅末端执行器当前状态的端口.
            moving_average_limit: 滑动平均滤波器的样本数量（目前暂未使用）.
            h_r_v: 机器人基座坐标系到 VR 坐标系的 4x4 齐次变换矩阵.
            h_t_v: 手部追踪坐标系到 VR 坐标系的 4x4 齐次变换矩阵.
            use_filter: 是否启用互补状态滤波器.
            arm_resolution_port: （可选）用于接收机械臂精度控制消息的端口.
            teleoperation_state_port: （可选）用于接收遥操作重置或暂停消息的端口.
            logging_config: （可选）用于姿态日志记录的配置字典.
            hand_side: 手侧（'left' 左手或 'right' 右手），用于确定关键点订阅的正确话题.
        """
        # 初始化基础属性
        self.operator_name = operator_name
        self.hand_side = hand_side
        self.notify_component_start(self.operator_name)
        self._host, self._port = host, transformed_keypoints_port

        # Transformation matrices specific to the arm setup
        self.h_r_v = h_r_v  # 机器人基座到VR坐标系的变换矩阵
        self.h_t_v = h_t_v  # 手部追踪到VR坐标系的变换矩阵

        # 初始化ZMQ上下文和订阅者
        self._context = get_global_context()

        # Determine the correct topic based on hand side
        if hand_side == robots.RIGHT:
            frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"
        else:  # LEFT
            frame_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_FRAME}"

        # 接收从 keypoint_transform.py 处理后发送过来的手部姿态数据,InputFrame 对象包含了手部在空间中的位置和方向向量
        self._arm_transformed_keypoint_subscriber = ZMQSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic=frame_topic,
            context=self._context,
            message_type=InputFrame,
        )

        # Optional subscribers
        self._arm_resolution_subscriber = None
        # TODO: Remove the literal in the topic arg use a constant.
        if arm_resolution_port:
            # 用于切换机械臂的控制精度
            self._arm_resolution_subscriber = ZMQSubscriber(
                host=host,
                port=arm_resolution_port,
                topic="button",
                context=self._context,
                message_type=ButtonEvent,
            )

        # TODO: Remove the literal in the topic arg use a constant.
        # 监听遥操作session的状态控制命令,包含"pause", "resume", "reset", "home"
        self._arm_teleop_state_subscriber = None
        if teleoperation_state_port:
            self._arm_teleop_state_subscriber = ZMQSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic="pause",
                context=self._context,
                message_type=SessionCommand,
            )

        # 接收机器人末端执行器当前的真实位姿，CartesianState 对象包含手臂末端的homogeneous matrix
        # 主要用于 Reset（重置） 阶段。系统需要知道机器人的当前位置，才能将其设为遥操作的起始“零点”，实现人手与机械臂的位姿对齐
        self.endeff_homo_subscriber = ZMQSubscriber(
            host=host,
            port=endeff_subscribe_port,
            topic="endeff_homo",
            context=self._context,
            message_type=CartesianState,
        )

        # 订阅者字典，便于在程序结束或对象销毁时，通过循环自动关闭所有网络连接，释放系统资源
        self._subscribers = {
            "endeff_homo": self.endeff_homo_subscriber,
            "teleop_state": self._arm_teleop_state_subscriber,
            "resolution_scale": self._arm_resolution_subscriber,
        }

        # Using the centralized publisher manager
        self._publisher_manager = ZMQPublisherManager.get_instance(self._context)
        self._publisher_host = host
        self._publisher_port = endeff_publish_port

        self._stream_oculus = stream_oculus
        self.stream_configs = stream_configs

        # State initialization
        self.arm_teleop_state = robots.ARM_TELEOP_CONT  # 遥操作状态：默认持续控制
        self.resolution_scale = 1.0  # 分辨率缩放因子：默认1.0
        self.is_first_frame = True  # 是否为第一帧：默认为True，需要重置
        self._timer = FrequencyTimer(robots.VR_FREQ)  # 频率定时器：控制运行频率
        self._robot = None  # 占位符：潜在的机器人接口对象
        self.real = False  # 占位符：指示仿真vs真实机器人

        # Transformation matrices state
        self.robot_init_h: Optional[np.ndarray] = None  # 机器人初始齐次矩阵
        self.robot_moving_h: Optional[np.ndarray] = None  # 机器人当前移动齐次矩阵
        self.hand_init_h: Optional[np.ndarray] = None  # 手部初始齐次矩阵
        self.hand_moving_h: Optional[np.ndarray] = None  # 手部当前移动齐次矩阵
        self.hand_init_t: Optional[np.ndarray] = None  # 手部初始平移向量
        self.last_valid_hand_frame: Optional[np.ndarray] = None  # 缓存最后接收到的手部帧

        # Filter setup
        self.use_filter = use_filter  # 是否使用滤波器
        self.comp_filter: Optional[CompStateFilter] = None  # 互补滤波器实例

        # 滑动窗口设置 (Currently unused in _apply_retargeted_angles)
        self.moving_average_queue = []  # 移动平均队列
        self.moving_average_limit = moving_average_limit  # 移动平均限制
        self.hand_frames = []  # 手部帧队列（可能与moving_average_queue重复）

        # 位置和方向的分离滑动窗口限制 (Currently unused)
        self.orientation_average_limit = min(10, moving_average_limit * 2)  # 方向移动平均限制
        self.orientation_queue = []  # 方向队列

        # 跟踪之前的朝向以进行稳定性检测 (Currently unused)
        self.prev_orientation: Optional[np.ndarray] = None  # 前一个方向
        self.last_sent_orientation: Optional[np.ndarray] = None  # 最后发送的方向
        self.ori_update_counter: int = 0  # 方向更新计数器

        # Initialize pose logger based on config
        self.logging_config = logging_config or {"enabled": False}  # 日志配置
        self.logging_enabled = self.logging_config.get("enabled", False)  # 是否启用日志
        self.pose_logger: Optional[PoseLogger] = None  # 姿态日志记录器

        if self.logging_enabled:
            log_filename = self.logging_config.get("filename", f"{self.operator_name}_poses.csv")
            logger.info(
                f"Initializing pose logger for {self.operator_name} with config: {self.logging_config}"
            )
            self.pose_logger = PoseLogger(filename=log_filename)  # Pass filename if specified
        else:
            self.pose_logger = None

        # Initialize handshake coordination for this operator
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{operator_name}_handshake"

        # 为该操作员启动具有唯一端口的握手服务器，使用操作符名称哈希来避免端口冲突
        # 目的：生成唯一的服务器标识，区分左右臂的握手过程
        operator_port_offset = hash(operator_name) % 100
        # 动态端口计算，在运行双臂时避免端口冲突
        handshake_port = robots.TELEOP_HANDSHAKE_PORT + operator_port_offset

        try:
            # 正式在计算出的端口上启动服务：当下游的机器人控制器（sysmo32_robot.py）启动时，
            # 它会先通过这个端口与 Operator “握手”。只有握手成功，确认双方协议和端口匹配，正式的遥操作数据流才会开始传输。
            self._handshake_coordinator.start_server(
                subscriber_id=self._handshake_server_id,
                bind_host="*",
                port=handshake_port,
            )
            logger.info(f"Handshake server started for {operator_name} on port {handshake_port}")
        except Exception as e:
            logger.warning(f"Failed to start handshake server for {operator_name}: {e}")

    @property
    def timer(self) -> FrequencyTimer:
        """Returns the frequency timer instance."""
        return self._timer

    @property
    def robot(self) -> Any:
        """Returns the robot interface object (placeholder)."""
        return self._robot

    @property
    def transformed_arm_keypoint_subscriber(self) -> ZMQSubscriber:
        """Returns the subscriber for transformed hand keypoints."""
        return self._arm_transformed_keypoint_subscriber

    @property
    def transformed_hand_keypoint_subscriber(self) -> None:
        """Required property from the Operator abstract class, returning None."""
        return None

    def return_real(self) -> bool:
        """Returns whether the operator is controlling a real robot (placeholder)."""
        return self.real

    # ------------------------------
    # Frame / Matrix utilities
    # ------------------------------
    def _get_hand_frame(self) -> Optional[np.ndarray]:
        """
        从 ZMQ 订阅者获取最新的手部帧数据。
        如果当前没有立即可用的新数据，则使用缓存的上一次数值。

        返回:
            一个 4x3 的 numpy 数组，代表手部坐标帧 ([t; R_列1; R_列2; R_列3])
            如果没有可用的有效帧，则返回 None。
        """
        # 尝试以非阻塞方式获取新数据
        data = self._arm_transformed_keypoint_subscriber.recv_keypoints()

        if data is not None:
            # 处理新数据 - 预期接收到包含 frame_vectors (帧向量) 的 InputFrame 对象
            try:
                if data.frame_vectors is not None:
                    # frame_vectors 应该是一个包含 4 个元组的序列 (origin + 3 basis vectors)
                    # Convert from Tuple[Tuple[float, float, float], ...] to numpy array (4, 3)
                    frame_data = np.array(data.frame_vectors, dtype=np.float64).reshape(4, 3)
                    self.last_valid_hand_frame = frame_data  # 缓存这个新的有效帧
                    return frame_data

            except Exception as e:
                logger.error(f"Error processing InputFrame data: {e}")
                # 如果处理失败，则进入后续逻辑尝试返回缓存帧

        # 如果没有新数据或处理失败，检查是否存在已缓存的帧，如果有则返回它
        if self.last_valid_hand_frame is not None:
            logger.info(f"No new data, returning cached frame: {self.last_valid_hand_frame}")
            return self.last_valid_hand_frame

        # 如果既没有新数据也没有缓存帧，返回 None
        return None

    def _turn_frame_to_homo_mat(self, frame: np.ndarray) -> np.ndarray:
        """
            将手部追踪获得的几何帧4x3数据 (origin + 3 basis vectors) 转换为机器人学中标准的齐次变换矩阵
        Args:
            frame: A 4x3 numpy array ([t; R_col1; R_col2; R_col3]).

        Returns:
            A 4x4 homogeneous transformation matrix.
        """
        if frame is None or frame.shape != (4, 3):
            raise ValueError("Input frame must be a 4x3 numpy array.")
        t = frame[0]  # 提取平移向量（手腕位置）
        r_cols = frame[1:]  # 提取旋转矩阵列（3x3）

        homo_mat = np.eye(4)  # 初始化4x4单位矩阵
        # The frame stores columns of R, so transpose r_cols to get R
        homo_mat[:3, :3] = r_cols.T  # 旋转部分：转置得到旋转矩阵
        homo_mat[:3, 3] = t  # 平移部分：设置平移向量
        # homo_mat[3, 3] = 1 # Already set by np.eye(4)

        return homo_mat

    def _homo2cart(self, homo_mat: np.ndarray) -> np.ndarray:
        """
        Converts a 4x4 homogeneous matrix to a 7D Cartesian pose vector.

        Args:
            homo_mat: A 4x4 homogeneous transformation matrix.

        Returns:
            A 7D numpy array [x, y, z, qx, qy, qz, qw].
        """
        t = homo_mat[:3, 3]  # 提取平移部分
        # Ensure the rotation matrix is valid before converting to quaternion
        r_mat = self.project_to_rotation_matrix(homo_mat[:3, :3])  # 确保旋转矩阵有效
        r_quat = Rotation.from_matrix(r_mat).as_quat()  # [qx, qy, qz, qw]  # 转换为四元数

        cart = np.concatenate([t, r_quat], axis=0)  # 拼接位置和四元数
        return cart

    def cart2homo(self, cart: np.ndarray) -> np.ndarray:
        """
        Converts a 7D Cartesian pose vector back to a 4x4 homogeneous matrix.

        Args:
            cart: A 7D numpy array [x, y, z, qx, qy, qz, qw].

        Returns:
            A 4x4 homogeneous transformation matrix.
        """
        if cart is None or cart.shape != (7,):
            raise ValueError("Input cart must be a 7D numpy array.")
        homo = np.eye(4)  # 初始化4x4单位矩阵
        t = cart[:3]  # 提取位置部分
        # Normalize quaternion before converting to matrix
        quat = cart[3:]  # 提取四元数部分
        norm = np.linalg.norm(quat)  # 计算四元数范数
        if norm > 1e-6:  # Avoid division by zero
            quat /= norm  # 归一化四元数
        else:
            # Handle zero quaternion case (e.g., default to identity rotation)
            quat = np.array([0.0, 0.0, 0.0, 1.0])  # 默认单位四元数

        r_mat = Rotation.from_quat(quat).as_matrix()  # 四元数转旋转矩阵
        homo[:3, 3] = t  # 设置平移部分
        homo[:3, :3] = r_mat  # 设置旋转部分
        return homo

    def project_to_rotation_matrix(self, r_mat: np.ndarray) -> np.ndarray:
        """
        使用奇异值分解(SVD)将一个近似旋转的3x3矩阵调整为一个有效的SO(3)旋转矩阵,确保行列式为+1(去除反射) 
        作用：矩阵纠偏,保证了无论计算如何波动，发给机器人的始终是一个物理上可实现的、平滑的旋转状态。
            在遥操作过程中，旋转矩阵经过多次乘法运算、缩放或网络传输，会因为浮点数精度误差逐渐失去其正交特性（不再是标准旋转矩阵）。
        如果直接将带误差的矩阵发给机器人，可能会导致：
            1) 机器人姿态抖动。
            2) 逆运动学(IK)解算失败，因为非法旋转矩阵无法计算出关节角。
            3) 模型拉伸或扭曲。
        Args:
            r_mat: A 3x3 numpy array, potentially close to a rotation matrix.

        Returns:
            A valid 3x3 rotation matrix.
        """
        try:
            u, _, vt = np.linalg.svd(r_mat)  # 执行SVD分解,将矩阵分解为两个正交阵 U 和 V^T
            r_fixed = u @ vt  # 重建旋转矩阵,通过将中间的奇异值矩阵 \Sigma 强制设为单位矩阵 I，重新构建出一个完美的正交矩阵。

            # 正交矩阵的行列式可能是 +1（旋转）或 -1（镜像/反射），确保行列式为 +1（旋转）
            if np.linalg.det(r_fixed) < 0:
                vt[-1, :] *= -1  # 翻转Vt的最后一行的符号，将其强制修正为行列式为 +1 的纯旋转矩阵
                # 注意：在固定行列式时，通常更倾向于调整Vt而非U
                r_fixed = u @ vt  # 重新计算R
            return r_fixed
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning identity matrix.")
            return np.eye(3)  #  fallback

    def _get_resolution_scale_mode(self) -> float:
        # TODO: We may not need this anymore I am not too sure what the use case is.
        # Instead we can default or make this configurable but do we really need it
        # during real time operation?

        """Gets the resolution scale mode from the subscriber."""
        if not self._arm_resolution_subscriber:
            return 1.0  # default if subscriber not configured

        # Use NOBLOCK to avoid waiting if no message is present
        data = self._arm_resolution_subscriber.recv_keypoints()
        if data is None:
            # Keep the current resolution scale if no new message
            return self.resolution_scale
        try:
            # Expect ButtonEvent
            scale_mode = data.value

            # Update internal resolution scale based on mode
            if scale_mode == robots.ARM_HIGH_RESOLUTION:
                self.resolution_scale = 1.0
            elif scale_mode == robots.ARM_LOW_RESOLUTION:
                self.resolution_scale = 0.6
            return self.resolution_scale  # Return the updated scale
        except Exception as e:
            logger.error(f"Error processing resolution scale data: {e}")
            return self.resolution_scale  # Return current scale on error

    def _get_arm_teleop_state(self) -> int:
        """Gets the arm teleoperation state (STOP/CONT) from the subscriber."""
        if not self._arm_teleop_state_subscriber:
            # Default to CONT if no subscriber, assuming continuous operation unless stopped externally
            return robots.ARM_TELEOP_CONT

        # Use NOBLOCK to avoid waiting
        data = self._arm_teleop_state_subscriber.recv_keypoints()
        if data is None:
            return self.arm_teleop_state  # Return current state if no new message
        try:
            # Expect SessionCommand
            if data.command == robots.PAUSE:
                return robots.ARM_TELEOP_STOP
            elif data.command == robots.RESUME:
                return robots.ARM_TELEOP_CONT
            else:
                return self.arm_teleop_state

        except Exception:
            return self.arm_teleop_state  # Return current state on error

    # ------------------------------
    # Teleop reset logic
    # ------------------------------
    def _reset_teleop(self) -> Optional[np.ndarray]:
        """
        通过捕捉当前机器人和手的姿态来重置远程操作baseline.
        发送重置信号并等待机器人的当前位姿,获取后更新robot_moving_h、robot_init_h、hand_init_h.

        Returns:
            The initial moving hand frame (4x3) captured after reset, or None on failure.
        """

        logger.info(f"****** {self.operator_name}: RESETTING TELEOP ******")
        # 1. 向下游（机器人或仿真器）发送重置请求，要求获取机器人当前的末端位姿
        # TODO: Remove the literal in the topic arg use a constant.
        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._publisher_port,
            topic="reset",
            data=SessionCommand(timestamp_s=time.time(), command="reset"),
        )
        robot_frame_homo = self.endeff_homo_subscriber.recv_keypoints()

        # Keep trying until we get a response
        while robot_frame_homo is None:
            self._publisher_manager.publish(
                host=self._publisher_host,
                port=self._publisher_port,
                topic="reset",
                data=SessionCommand(timestamp_s=time.time(), command="reset"),
            )
            robot_frame_homo = self.endeff_homo_subscriber.recv_keypoints()
            time.sleep(0.01)

        try:
            h = np.array(robot_frame_homo.h_matrix, dtype=np.float64).reshape(4, 4)  # 转换为4x4矩阵
            self.robot_init_h = h  # 保存机器人初始姿态
            # Validate if it's close to a homogeneous matrix
            if not np.allclose(self.robot_init_h[3, :], [0, 0, 0, 1]):
                logger.warning(
                    f"Warning ({self.operator_name}): Received robot frame is not a valid homogeneous matrix. Resetting bottom row."
                )
                self.robot_init_h[3, :] = [0, 0, 0, 1]  # 重置底部行
            # Ensure rotation part is valid SO(3)
            self.robot_init_h[:3, :3] = self.project_to_rotation_matrix(self.robot_init_h[:3, :3])  # 确保旋转矩阵有效

        except Exception:
            # logger.error(f"ERROR ({self.operator_name}): Failed to process received robot frame: {e}")
            # 如果解析失败，标记为第一帧状态（以便下次循环重试）并退出
            self.is_first_frame = True  # Stay in reset state
            return None

        # 将当前计算出的目标位姿初始化为初始位姿
        self.robot_moving_h = copy(self.robot_init_h)  # 初始化机器人移动姿态为初始姿态
        logger.info(f"{self.operator_name} Robot init H:\n{self.robot_init_h}")
        
        # 4. 获取手部的初始位姿
        first_hand_frame = None
        while first_hand_frame is None:
            # 阻塞等待直到获取到有效的 VR 手部追踪数据
            first_hand_frame = self._get_hand_frame()  # 等待有效手部帧
            time.sleep(0.01)

        try:
            self.hand_init_h = self._turn_frame_to_homo_mat(first_hand_frame)  # 转换为齐次矩阵
            self.hand_init_t = copy(self.hand_init_h[:3, 3])  # Store initial hand translation
            logger.info(f"{self.operator_name} Hand init H:\n{self.hand_init_h}")
        except ValueError as e:
            logger.error(f"ERROR ({self.operator_name}): Failed to convert initial hand frame to matrix: {e}")
            self.is_first_frame = True  # Stay in reset state
            return None
        
        # 5. 完成重置
        self.is_first_frame = False  # Reset successful
        self.comp_filter = None  # Reset filter, will be initialized on first _apply call
        logger.info(f"{self.operator_name}: TELEOP RESET COMPLETE")
        logger.info(f"[{self.operator_name}] hand_init_h\n{self.hand_init_h}")
        return first_hand_frame  # Return the frame used for initialization

    # ------------------------------
    # Main teleop: transforms
    # ------------------------------
    def _fix_quaternion_flips(self, quats: np.ndarray) -> np.ndarray:
        """
        通过防止相对于序列中第一个四元数的“跨半球翻转”，确保四元数表示的连续性。
        简单来说，就是确保相邻两帧四元数的数值表达在同一个“半球”内，避免符号突变

        Args:
            quats: A numpy array of quaternions (Nx4).

        Returns:
            A numpy array of quaternions (Nx4) with flips corrected.
        """
        # 如果数组为空或只有一个元素，无需进行连续性检查
        if quats is None or len(quats) <= 1:
            return quats
        
        # 初始化修正后的列表，将第一个四元数作为初始参考基准
        fixed = [quats[0]]  
        for q in quats[1:]:
            # 1. 计算当前四元数 q 与上一个已修正四元数 (fixed[-1]) 的点积 (Dot Product)
            # 点积的大小反映了两个四元数在四维球体空间中的“距离”或夹角
            dot = np.sum(fixed[-1] * q)  # 计算点积
            # 2. 判断是否发生了半球翻转
            # 如果点积小于 0，意味着两个四元数之间的夹角大于 90 度（在四元数空间中）
            # 这通常表示四元数从 q 跳变到了其等效的 -q 表示形式
            if dot < 0:
                fixed.append(-q)  # 翻转四元数
            else:
                fixed.append(q)  # 保持不变
        return np.array(fixed)

    def _apply_retargeted_angles(self):
        """
        整个遥操作系统的核心执行引擎。
        它在每一个循环周期内，完成从"获取手部原始位姿" -> "应用重定向" -> "计算并发布机器人目标指令" 的全过程
        逻辑包括：处理状态变化（重置/暂停）、应用坐标变换、滤波平滑以及发布指令
        """

        # 1. 检查系统状态变化 (Pause/Resume, Resolution)
        new_arm_teleop_state = self._get_arm_teleop_state()  # 从订阅者获取当前遥操作开关状态
        self.resolution_scale = self._get_resolution_scale_mode()  # Update resolution scale

        # 确定是否需要触发重置：
        # - 第一次运行 (is_first_frame)
        # - 或者状态从 STOP 切换回 CONT（恢复运行）时
        needs_reset = self.is_first_frame or (
            self.arm_teleop_state == robots.ARM_TELEOP_STOP and new_arm_teleop_state == robots.ARM_TELEOP_CONT
        )  # 检查是否需要重置

        # 在检查完状态跳变后，更新当前状态变量
        self.arm_teleop_state = new_arm_teleop_state  # 更新遥操作状态

        # 决定当前周期是否需要对外发布机器人指令（只有在 CONT 状态下才发布）
        publish_commands = self.arm_teleop_state == robots.ARM_TELEOP_CONT  # 是否发布命令

        # 2. 处理重置逻辑（同步人手与机器人的起始位置）
        if needs_reset:
            moving_hand_frame = self._reset_teleop()  # 调用重置方法，返回当前手部基准帧
            if moving_hand_frame is None:
                logger.error(f"ERROR ({self.operator_name}): Reset failed, cannot proceed.")
                return  # Exit if reset failed
            # Reset is done, is_first_frame is now False
        else:
            # 3. 正常运行模式：获取当前帧的手部追踪数据
            moving_hand_frame = self._get_hand_frame()  # 获取当前手部帧

        # 如果无法获取有效的手部帧（可能丢包或追踪丢失），跳过本轮循环
        if moving_hand_frame is None:
            logger.warning(f"Warning ({self.operator_name}): No valid hand frame received, skipping cycle.")
            return

        # 安全检查：确保重置阶段生成的初始位姿已存在
        if self.robot_init_h is None or self.hand_init_h is None:
            logger.error(
                f"ERROR ({self.operator_name}): Initial robot or hand poses not set. Triggering reset."
            )
            self.is_first_frame = True  # Force reset on next cycle
            return

        # 4. Convert current hand frame(4x3) to Homogeneous Matrix
        try:
            self.hand_moving_h = self._turn_frame_to_homo_mat(moving_hand_frame)  # 转换为齐次矩阵
        except ValueError as e:
            logger.error(f"Error ({self.operator_name}): Could not convert moving hand frame: {e}")
            return  # Skip cycle if conversion fails

        # 5. Calculate Relative Transformation
        # 手部当前相对于初始点的变化 = (初始手部矩阵的逆) × 当前手部矩阵
        # Use solve for potentially better numerical stability than inv
        try:
            h_hi_hh_inv = np.linalg.inv(self.hand_init_h)  # 初始手部姿态的逆矩阵
            h_ht_hi = h_hi_hh_inv @ self.hand_moving_h  # 手部相对于起始姿态的相对运动
            # Alternative using solve: H_HT_HI = np.linalg.solve(self.hand_init_H, self.hand_moving_H)
        except np.linalg.LinAlgError:
            logger.error(f"Error ({self.operator_name}): Could not invert initial hand matrix. Resetting.")
            self.is_first_frame = True
            return

        # 6. 坐标系空间重定向 Apply Coordinate Transformations (using provided H_R_V and H_T_V)
        
        # 将手部坐标系(T)下的运动量，映射到机器人基座坐标系(R)下
        try:
            h_r_v_inv = np.linalg.inv(self.h_r_v)  # 机器人基座到VR坐标系的逆矩阵
            h_t_v_inv = np.linalg.inv(self.h_t_v)  # 手部到VR坐标系的逆矩阵

            # 计算手部相对旋转在机器人基座坐标系下的等效变换 (基底变换/相似变换: P^-1 * R * P)
            # 旋转部分变换：结合 H_R_V 矩阵，将手部的相对转动量映射到机器人空间
            h_ht_hi_r = h_r_v_inv[:3, :3] @ h_ht_hi[:3, :3] @ self.h_r_v[:3, :3]  # handRelRotInVRWorld_2_robotBase=VRBase_2_robotBase * VRHand_2_VRHandInit * robotBase_2_VRWorld
            # 平移部分变换：通过左乘一个 h_t_v_inv（手部坐标系的逆映射），其实是在做一个轴向置换（Axis Swapping），目的是直接把手部在空间里的位移方向，强行“掰”成机器人基座能够理解的方向
            h_ht_hi_t = h_t_v_inv[:3, :3] @ h_ht_hi[:3, 3] * self.resolution_scale  # handRelMov_2_VRHandTrack = R{VRBase_2_VRHandTrack} * T{Hand_Relative_Motion}

        except np.linalg.LinAlgError:
            logger.error(f"Error ({self.operator_name}): Could not invert H_R_V or H_T_V matrix.")
            # Handle error appropriately, maybe reset or use identity
            return

        # 修正计算出的旋转矩阵，确保其正交合法（防止由于连续乘法导致的数值漂移）
        h_ht_hi_r = self.project_to_rotation_matrix(h_ht_hi_r)  # 确保旋转矩阵有效

        # 在机器人基座坐标系中进行相对仿射变换
        relative_affine_in_robot_frame = np.eye(4)  # 初始化4x4单位矩阵
        relative_affine_in_robot_frame[:3, :3] = h_ht_hi_r  # 设置旋转部分
        relative_affine_in_robot_frame[:3, 3] = h_ht_hi_t  # 设置平移部分

        # 7. 计算机器人最终目标位姿
        # H_RT_RH = H_RI_RH * relative_affine_in_robot_frame
        h_rt_rh = self.robot_init_h @ relative_affine_in_robot_frame  # 计算目标机器人位姿

        # 再次确保最终输出的矩阵旋转部分是合法的
        h_rt_rh[:3, :3] = self.project_to_rotation_matrix(h_rt_rh[:3, :3])  # 确保旋转矩阵有效
        self.robot_moving_h = copy(h_rt_rh)  # 缓存当前计算结果

        # 8. 格式转换：将 4x4 齐次矩阵转换为 [x, y, z, qx, qy, qz, qw] 7维笛卡尔坐标格式
        cart_target_raw = self._homo2cart(self.robot_moving_h) 

        # 9. 应用滤波器（平滑抖动）
        if self.use_filter:
            # 如果是重置后的第一帧，初始化互补滤波器
            if self.comp_filter is None:
                # 使用第一帧中的*原始*目标姿态作为初始滤波器状态
                self.comp_filter = CompStateFilter(
                    init_state=cart_target_raw,
                    pos_ratio=0.7,  # 位置平滑系数
                    ori_ratio=0.85, # 姿态平滑系数
                    adaptive=True,
                )
                cart_target_filtered = cart_target_raw  # 第一帧直接使用原始值
            else:
                # 正常滤波：融合历史信息与当前输入，减少手部震颤
                cart_target_filtered = self.comp_filter(cart_target_raw)
        else:
            cart_target_filtered = cart_target_raw  # No filtering

        # 10. 准备过滤后的姿态数据以供发布（四元数方向正半球）
        position = cart_target_filtered[0:3]  # 提取位置部分
        orientation_quat = cart_target_filtered[3:7].copy()  # 提取四元数部分

        # 归一化四元数并强制约束到“正半球”(w >= 0)，防止数学上的翻转导致机器人动作跳变
        norm = np.linalg.norm(orientation_quat)  # 计算四元数范数
        if norm < 1e-6:
            orientation_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 默认单位四元数
        else:
            orientation_quat = orientation_quat / norm  # 归一化四元数
            if orientation_quat[3] < 0:  # 如果 w 分量为负，则整体取反
                orientation_quat = -orientation_quat  # 确保w分量为正

        # 11. 构建统一的 Contract 对象进行发布
        cartesian_cmd = CartesianTarget(
            timestamp_s=time.time(),
            hand_side=self.hand_side,
            frame_id="base",  # 坐标参考系为机器人基座
            position_m=(float(position[0]), float(position[1]), float(position[2])),  # 目标位置
            orientation_xyzw=(
                float(orientation_quat[0]),
                float(orientation_quat[1]),
                float(orientation_quat[2]),
                float(orientation_quat[3]),
            ),  # 目标姿态四元数
        )

        # 仅在系统处于运行模式CONT mode时，将指令通过 ZMQ 发布给下游
        if publish_commands:
            try:
                # TODO: Remove the literal in the topic arg use a constant.
                self._publisher_manager.publish(
                    host=self._publisher_host,
                    port=self._publisher_port,
                    topic="endeff_coords",  # 发布话题
                    data=cartesian_cmd,  # 发布数据
                )
                # logger.info(f"Published end-effector command: {command_data}")
            except (ConnectionError, SerializationError) as e:
                logger.error(f"Failed to publish end-effector command: {e}")
            except Exception as e:
                logger.error(f"Unexpected error publishing command: {e}")

        # 12. 记录日志（可选）：保存初始和实时的矩阵数据，用于后续离线分析
        if self.logging_enabled and self.pose_logger:
            try:
                # 在记录之前，确保所有矩阵都有效
                if (
                    self.hand_init_h is not None
                    and self.robot_init_h is not None
                    and self.hand_moving_h is not None
                    and self.robot_moving_h is not None
                ):
                    self.pose_logger.log_frame(
                        self.hand_init_h,
                        self.robot_init_h,
                        self.hand_moving_h,
                        self.robot_moving_h,  # 记录滤波前的原始目标值
                    )
            except Exception as e:
                logger.error(f"Error logging frame ({self.operator_name}): {e}")

    def moving_average(self, action: np.ndarray, queue: list, limit: int) -> np.ndarray:
        """
        对输入的动作（例如目标位姿向量）应用简单的滑动平均滤波
        注意：目前该方法在主循环 `_apply_retargeted_angles` 中并未被使用

        Args:
            action: 当前时刻最新的数据点（例如 7D 的笛卡尔位姿向量 [x,y,z,qx,qy,qz,qw]）。
            queue: 充当滑动窗口队列的 Python 列表（保存历史数据）。
            limit: 队列的最大长度（即滑动窗口的大小）。

        Returns:
            The averaged action.
        """
        # 1. 入队：将最新的一帧数据放到队列的末尾
        queue.append(action) 
        
        # 2. 维护窗口大小：如果当前保存的历史数据量超过了设定的上限 (limit)
        if len(queue) > limit:
            queue.pop(0)  
            
        # 3. 安全检查：在计算均值前确保队列不为空（极少数异常情况下触发）
        if not queue:
            return action  # 如果为空，直接返回当前数据（不做处理）
        
        # 4. 计算均值：沿着 axis=0（列方向）对队列中的所有历史向量求算术平均值
        return np.mean(queue, axis=0)  

    def run(self):
        # TODO: Call this method stream to align with rest of the codebase
        """The main execution loop for the operator."""
        try:
            while True:
                with self.timer:  # Ensures loop runs at desired frequency (e.g., VR_FREQ)
                    self._apply_retargeted_angles()  # 执行主要遥操作逻辑
        except KeyboardInterrupt:
            # 捕获用户在终端按下 Ctrl+C 的中断信号
            logger.info(f"{self.operator_name} received KeyboardInterrupt. Cleaning up...")
        finally:
            self.cleanup()

    def __del__(self):
        """Destructor ensures cleanup is called."""
        # Safely clean up subscribers if they were initialized
        # 1. 安全地关闭所有可选的 ZMQ 订阅者 (Subscribers)
        if hasattr(self, "_subscribers") and self._subscribers:
            for subscriber in self._subscribers.values():
                if subscriber:  # Check if subscriber is not None
                    try:
                        subscriber.stop()  # 停止订阅者
                    except Exception as e:
                        logger.warning(
                            f"Error stopping subscriber in {getattr(self, 'operator_name', 'unknown')}: {e}"
                        )
        # 2. 停止握手服务器 (Handshake Server)
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            try:
                self._handshake_coordinator.stop_server(self._handshake_server_id)  # 停止握手服务器
            except Exception as e:
                logger.warning(f"Error stopping handshake server: {e}")

        # 3. 全局 ZMQ 资源清理
        # ZMQ (ZeroMQ) 如果不手动清理其上下文 (Context) 和套接字 (Sockets)，
        # 很容易导致内存泄漏或端口被死锁占用（报 "Address already in use" 错误）。
        cleanup_zmq_resources()  