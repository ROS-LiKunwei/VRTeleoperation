"""
手部关键点坐标变换模块

本模块是BeaVR-bot遥操作系统的第二环,负责对手部关键点进行坐标变换和平滑处理，
将原始的VR手部坐标转换为适合机器人控制的标准化坐标。

数据流位置：
    pico4.py → [本模块: keypoint_transform.py] → xarm7_operator.py → xarm7_robot.py

主要功能：
    1. 手腕坐标系平移：以手腕为原点，将所有关键点平移到以手腕为中心的坐标系
    2. 旋转矩阵变换：将手部坐标旋转到标准朝向（目前使用单位矩阵，保持原始坐标不变）
    3. 手部方向坐标系计算：基于食指/中指/小指关节计算手部朝向的3D坐标系
    4. 滑动平均平滑：对坐标和方向帧进行时域平滑，减少抖动
    5. 正交化处理：确保方向帧向量保持正交性

通信协议：
    - 接收：ZMQ SUB套接字，从pico4.py订阅InputFrame对象
    - 发送：ZMQ PUB套接字，发布变换后的InputFrame对象给下游组件

端口配置：
    - 订阅端口：8088 (KEYPOINT_STREAM_PORT)
    - 右手发布端口：8092 (KEYPOINT_TRANSFORM_PORT)
    - 左手发布端口：8093 (LEFT_KEYPOINT_TRANSFORM_PORT)

发布Topic：
    - 右手坐标：right_transformed_hand_coords
    - 右手帧：right_transformed_hand_frame
    - 左手坐标：left_transformed_hand_coords
    - 左手帧：left_transformed_hand_frame

26关节索引定义（OCULUS_JOINTS）：
    0: Wrist（手腕）
    1: Palm（手掌）
    拇指(2-5): Metacarpal, Proximal, Distal, Tip
    食指(6-10): Metacarpal, Proximal, Intermediate, Distal, Tip
    中指(11-15): Metacarpal, Proximal, Intermediate, Distal, Tip
    无名指(16-20): Metacarpal, Proximal, Intermediate, Distal, Tip
    小指(21-25): Metacarpal, Proximal, Intermediate, Distal, Tip
"""

import logging
import time
from copy import deepcopy as copy
from enum import IntEnum

import numpy as np

from beavr.teleop.common.math.vectorops import moving_average
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import InputFrame, SessionCommand
from beavr.teleop.components.detector.vr.log_keypoints import KeypointLogger
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)

UNITY_LEFT_TO_INTERNAL_RIGHT = np.array([1.0, 1.0, -1.0], dtype=np.float64)


def _safe_normalize(vector, min_norm=1e-6):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < min_norm:
        return None
    return vector / norm


class HandMode(IntEnum):
    """手部数据模式枚举"""

    ABSOLUTE = 1  # 绝对模式：关键点为世界坐标系下的绝对位置
    RELATIVE = 0  # 相对模式：关键点为相对于手腕的位置


class TransformHandPositionCoords(Component):
    """
    手部关键点坐标变换组件。

    数据流角色：
        本类是遥操作系统数据流的第二环，
        负责将pico4.py接收到的原始手部关键点进行坐标变换和平滑处理,
        输出标准化的手部坐标和方向帧给下游的xarm7_operator.py。

    变换流程：
        1. 接收InputFrame对象(包含26个关节的原始坐标)
        2. 平移：以手腕为原点，将所有关键点平移到手腕中心坐标系
        3. 旋转：应用旋转矩阵将手部坐标旋转到标准朝向
           (目前使用单位矩阵,保持原始坐标不变,TODO: 后续替换为实际旋转矩阵）
        4. 计算手部方向帧：基于食指/中指/小指关节计算手部朝向的3D坐标系
        5. 滑动平均平滑：对坐标和方向帧进行时域平滑
        6. 正交化处理：确保方向帧向量保持正交性
        7. 发布变换后的InputFrame对象

    输出数据：
        - transformed_keypoints:变换后的关键点坐标（以手腕为原点）
        - coordinate_frame:手部方向帧 [origin, x_vec, y_vec, z_vec]
          - origin:手腕在世界坐标系下的位置
          - x_vec, y_vec, z_vec:手部朝向的三个正交基向量
    """

    def __init__(
        self,
        host: str,
        keypoint_sub_port: int,
        keypoint_transform_pub_port: int,
        hand_side: str = robots.RIGHT,
        moving_average_limit: int = 5,
        enable_logging: bool = False,
        log_dir: str = "data/keypoint_logs",
        auto_save_interval: int = 100,
    ):
        """
        初始化手部关键点坐标变换组件。

        Args:
            host: 网络主机地址(ZMQ通信地址)。
            keypoint_sub_port: 订阅关键点数据的端口号(从pico4.py接收)。
            keypoint_transform_pub_port: 发布变换后关键点的端口号(发送给xarm7_operator.py)。
            hand_side: 手侧，'right'（右手）或 'left'（左手）。
            moving_average_limit: 滑动平均窗口大小,默认5帧。
            enable_logging: 是否启用关键点日志记录,默认False。
            log_dir: 日志保存目录，默认"data/keypoint_logs"。
            auto_save_interval: 自动保存间隔(帧数),默认100。
        """
        if hand_side not in [robots.LEFT, robots.RIGHT]:
            raise ValueError(f"hand_side must be {robots.LEFT} or {robots.RIGHT}")

        self.hand_side = hand_side

        """
        组件生命周期管理与系统状态通知
        Component 的父类:
            系统监控 (Monitoring): 整个系统的控制中心。可以通过接收这个通知，知道“哦，右手坐标变换模块已经成功启动并准备就绪了”。
            日志追踪 (Logging & Debugging): 在排查 bug 时非常有用。如果系统没收到手部数据，开发者可以去查日志，看看对应的组件有没有输出类似 “right_hand_keypoint_transform started” 的信息。如果没有，说明系统在初始化阶段就挂了；如果有，说明是后续的网络通信或计算出了问题。
            解耦设计： 这种设计让各个模块高度解耦。当前模块不需要关心谁在监听它的状态,它只需要负责大喊一声“我上线了”,剩下的由框架的基类(Component)来处理（比如写入日志文件、或者通过 ZMQ 发送一个心跳包）。
        """
        component_name = f"{hand_side}_hand_keypoint_transform"
        self.notify_component_start(component_name)

        self.host = host
        self.keypoint_sub_port = keypoint_sub_port
        self.keypoint_transform_pub_port = keypoint_transform_pub_port

        # 初始化订阅者：从pico4.py订阅原始关键点数据
        if hand_side == robots.RIGHT:
            self.keypoint_subscriber = ZMQSubscriber(self.host, self.keypoint_sub_port, robots.RIGHT)
        else:
            self.keypoint_subscriber = ZMQSubscriber(self.host, self.keypoint_sub_port, robots.LEFT)
        self.pause_subscriber = ZMQSubscriber(
            self.host,
            self.keypoint_sub_port,
            robots.PAUSE,
            message_type=SessionCommand,
        )

        self.publisher_manager = ZMQPublisherManager.get_instance()

        # 定义发布Topic名称
        if hand_side == robots.RIGHT:
            self.coords_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_COORDS}"
            self.frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"
            self.absolute_mode = robots.ABSOLUTE
            self.relative_mode = robots.RELATIVE
        else:
            self.coords_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_COORDS}"
            self.frame_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_FRAME}"
            self.absolute_mode = robots.ABSOLUTE
            self.relative_mode = robots.RELATIVE
        self.publisher_manager.register_topic(self.host, self.keypoint_transform_pub_port, self.coords_topic)
        self.publisher_manager.register_topic(self.host, self.keypoint_transform_pub_port, self.frame_topic)

        self.timer = FrequencyTimer(robots.VR_FREQ)  # 30Hz频率

        # 定义用于稳定帧计算的关键关节索引
        self.wrist_idx = 0  # 手腕（第0个关节）
        self.index_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][0]  # 食指掌指关节（第7个关节）
        self.middle_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][1]  # 中指掌指关节（第12个关节）
        self.pinky_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][-1]  # 小指掌指关节（第22个关节）

        # 滑动平均队列
        self.moving_average_limit = moving_average_limit
        self.coord_moving_average_queue, self.frame_moving_average_queue = [], []
        self.averaged_keypoints = None
        self.averaged_coordinate_frame = None
        self.arm_teleop_state = robots.ARM_TELEOP_CONT
        self._resume_settle_frames_remaining = 0

        # 初始化关键点日志记录器
        self.keypoint_logger = None
        if enable_logging:
            self.keypoint_logger = KeypointLogger(
                hand_side=hand_side,
                log_dir=log_dir,
                auto_save_interval=auto_save_interval,
                moving_average_limit=moving_average_limit,
            )
        self._last_invalid_frame_log_time = 0.0
        self._last_no_raw_input_log_time = 0.0
        self._last_receive_raw_input_log_time = 0.0
        self._last_publish_frame_log_time = 0.0
        self._last_timing_log_time = 0.0

    def _clear_smoothing_state(self):
        self.coord_moving_average_queue.clear()
        self.frame_moving_average_queue.clear()
        self.averaged_keypoints = None
        self.averaged_coordinate_frame = None

    def _get_arm_teleop_state(self):
        data = self.pause_subscriber.recv_keypoints()
        if data is None:
            return self.arm_teleop_state

        try:
            if data.command == robots.PAUSE:
                new_state = robots.ARM_TELEOP_STOP
            elif data.command == robots.RESUME:
                new_state = robots.ARM_TELEOP_CONT
            else:
                return self.arm_teleop_state
        except Exception:
            return self.arm_teleop_state

        previous_state = self.arm_teleop_state
        if new_state != previous_state:
            self._clear_smoothing_state()
            self.keypoint_subscriber.recv_keypoints()
            if new_state == robots.ARM_TELEOP_STOP:
                self._resume_settle_frames_remaining = 0
                logger.info(f"{self.hand_side}_hand_keypoint_transform: pause received, stop publishing hand frames")
            else:
                self._resume_settle_frames_remaining = max(1, int(self.moving_average_limit))
                logger.info(
                    f"{self.hand_side}_hand_keypoint_transform: resume received, "
                    f"rebuilding baseline input over {self._resume_settle_frames_remaining} frames"
                )

        self.arm_teleop_state = new_state
        return self.arm_teleop_state

    def _warn_invalid_frame(self, reason):
        current_time = time.time()
        if current_time - self._last_invalid_frame_log_time >= 1.0:
            self._last_invalid_frame_log_time = current_time
            logger.warning(f"{self.hand_side}_hand_keypoint_transform: 跳过无效手部帧: {reason}")

    def _log_no_raw_input(self):
        current_time = time.time()
        if current_time - self._last_no_raw_input_log_time >= 1.0:
            self._last_no_raw_input_log_time = current_time
            logger.debug(
                f"{self.hand_side}_hand_keypoint_transform: 暂未收到PICO/Unity原始手部数据, "
                f"订阅 topic={self.hand_side} port={self.keypoint_sub_port}"
            )

    def _log_receive_raw_input(self, input_frame, reshaped_keypoints):
        current_time = time.time()
        if current_time - self._last_receive_raw_input_log_time >= 1.0:
            self._last_receive_raw_input_log_time = current_time
            logger.debug(
                f"{self.hand_side}_hand_keypoint_transform: 已收到PICO/Unity原始手部数据 "
                f"topic={self.hand_side} port={self.keypoint_sub_port} "
                f"frame_hand={input_frame.hand_side} wrist={reshaped_keypoints[0].tolist()} "
                f"is_relative={input_frame.is_relative} hand_command={input_frame.hand_command}"
            )

    def _log_publish_frame(self, coordinate_frame, hand_command):
        current_time = time.time()
        if current_time - self._last_publish_frame_log_time >= 1.0:
            self._last_publish_frame_log_time = current_time
            origin = np.asarray(coordinate_frame[0], dtype=np.float64).tolist()
            logger.debug(
                f"{self.hand_side}_hand_keypoint_transform: 已发布变换后手部方向帧 "
                f"topic={self.frame_topic} port={self.keypoint_transform_pub_port} "
                f"origin={origin} hand_command={hand_command}"
            )

    def _validate_hand_coords(self, hand_coords):
        if hand_coords is None or hand_coords.shape != (robots.OCULUS_NUM_KEYPOINTS, 3):
            return False, "关键点形状错误"
        if not np.all(np.isfinite(hand_coords)):
            return False, "关键点包含 NaN/Inf"

        wrist = hand_coords[self.wrist_idx]
        distances = np.linalg.norm(hand_coords - wrist, axis=1)
        if float(np.max(distances)) < 1e-4:
            return False, "所有关键点几乎重合"

        return True, ""

    def _unity_left_handed_to_internal_right_handed(self, hand_coords):
        """
        PICO Unity Integration SDK reports positions in Unity's left-handed
        world frame: +X right, +Y up, +Z forward. The teleop stack uses an
        internal right-handed VR frame: +X right, +Y up, +Z backward.
        """
        return hand_coords * UNITY_LEFT_TO_INTERNAL_RIGHT

    def _get_hand_coords(self):
        """
        从ZMQ订阅者获取手部坐标数据。
        从pico4.py发布的InputFrame对象中提取关键点数据,并将其reshape为(26, 3)的形状。

        Returns:
            tuple: (data_type, coordinates, hand_command, source_timestamp_s)
                - data_type: HandMode.RELATIVE 或 HandMode.ABSOLUTE
                - coordinates: numpy数组,形状为(26, 3),26个关节的xyz坐标
                - hand_command: 灵巧手命令，1=松开，2=抓取；无数据时为None。
                - 无数据时返回 (None, None, None, None)
        """
        # 1. 接收数据：从 ZMQ 网络队列中非阻塞地拉取最新的一帧手部数据
        input_frame = self.keypoint_subscriber.recv_keypoints()
        if input_frame is None:
            self._log_no_raw_input()
            return None, None, None, None

        # 2. 转换为 Numpy 数组：将接收到的通常是 Python List 格式的数据转化为高效的 numpy 数组
        keypoints = np.asanyarray(input_frame.keypoints)

        logger.debug(
            f"_get_hand_coords: Received keypoints for {input_frame.hand_side} hand. "
            f"Length: {len(input_frame.keypoints)}, First 3 values: {input_frame.keypoints[:3] if len(input_frame.keypoints) > 0 else 'empty'}"
        )

        # 3. 完整性校验
        expected_count = robots.OCULUS_NUM_KEYPOINTS * 3  # 26 * 3 = 78
        actual_count = keypoints.size

        if actual_count != expected_count:
            logger.warning(
                f"_get_hand_coords: keypoint count mismatch. "
                f"Expected {expected_count} elements ({robots.OCULUS_NUM_KEYPOINTS} points × 3 coords), "
                f"got {actual_count}. Skipping this frame."
            )
            return None, None, None, None

        # 4. 判断坐标系模式：根据传过来的标志位，判断这组坐标是绝对世界坐标 (ABSOLUTE) 还是相对坐标 (RELATIVE)
        data_type = self.relative_mode if input_frame.is_relative else self.absolute_mode

        # 5. 重塑为 (26, 3) 形状：将 78 个元素的数组转换为 (26, 3) 的二维矩阵，每一行代表一个关节点的 [x, y, z]，方便后续做矩阵乘法和向量计算
        reshaped_keypoints = keypoints.reshape(robots.OCULUS_NUM_KEYPOINTS, 3)
        self._log_receive_raw_input(input_frame, reshaped_keypoints)
        logger.debug(
            f"_get_hand_coords: Reshaped keypoints for {input_frame.hand_side} hand. "
            f"Shape: {reshaped_keypoints.shape}, Wrist position: {reshaped_keypoints[0]}"
        )

        return data_type, reshaped_keypoints, input_frame.hand_command, input_frame.timestamp_s

    def _orthogonalize_frame(self, x_vec, y_vec, z_vec):
        """
        使用Gram-Schmidt正交化过程确保三个向量形成正交坐标系。

        正交化步骤：
        1. 归一化x向量
        2. 将y向量投影到x向量的正交补空间，然后归一化
        3. 通过x和y的叉积计算z向量，确保正交性

        Args:
            x_vec: x方向向量
            y_vec: y方向向量
            z_vec: z方向向量

        Returns:
            tuple: (x_vec, y_vec, z_vec) 正交化后的三个单位向量
        """
        x_vec = _safe_normalize(x_vec)
        if x_vec is None:
            return None

        y_vec = y_vec - np.dot(y_vec, x_vec) * x_vec
        y_vec = _safe_normalize(y_vec)
        if y_vec is None:
            return None

        z_vec = np.cross(x_vec, y_vec)
        z_vec = _safe_normalize(z_vec)
        if z_vec is None:
            return None

        return x_vec, y_vec, z_vec

    def _get_stable_coord_frame(self, hand_coords):
        """
        使用多个手部关键点计算更稳定的坐标系。

        基于手腕、食指掌指关节、中指掌指关节和小指掌指关节，
        计算手掌的法向量、朝向和侧向，形成稳定的3D坐标系。

        该坐标系用于将手部关键点旋转到标准朝向（目前未使用，
        因为旋转矩阵已替换为单位矩阵）。

        Args:
            hand_coords: 手部坐标数组，形状为(26, 3)，以手腕为原点

        Returns:
            list: [x_vec, y_vec, z_vec] 三个正交基向量
        """
        wrist = hand_coords[self.wrist_idx]
        v1 = hand_coords[self.index_knuckle_idx] - wrist  # 手腕→食指掌指
        v2 = hand_coords[self.pinky_knuckle_idx] - wrist  # 手腕→小指掌指
        v3 = hand_coords[self.middle_knuckle_idx] - wrist  # 手腕→中指掌指

        palm_normal = _safe_normalize(np.cross(v1, v3))  # 手掌法向量（Z方向）
        palm_direction = _safe_normalize((v1 + v2 + v3) / 3)  # 手掌朝向（Y方向）
        if palm_normal is None or palm_direction is None:
            return None
        cross_product = _safe_normalize(np.cross(palm_direction, palm_normal))  # 侧向（X方向）
        if cross_product is None:
            return None

        frame = self._orthogonalize_frame(cross_product, palm_direction, palm_normal)
        if frame is None:
            return None
        x_vec, y_vec, z_vec = frame

        return [x_vec, y_vec, z_vec]

    def _get_stable_hand_dir_frame(self, hand_coords):
        """
        使用多个手部关键点计算手部方向帧。

        方向帧包含4个向量：[手腕位置, x_vec, y_vec, z_vec]
        这个方向帧会被下游的xarm7_operator.py用来：
        1. 构建手部的4x4齐次变换矩阵
        2. 计算手部相对于初始帧的运动
        3. 将手部运动映射到机器人末端执行器

        Args:
            hand_coords: 手部坐标数组，形状为(26, 3)，原始世界坐标

        Returns:
            list: [wrist, x_vec, y_vec, z_vec]
                - wrist: 手腕在世界坐标系下的位置
                - x_vec, y_vec, z_vec: 手部朝向的三个正交基向量
        """
        wrist = hand_coords[self.wrist_idx]
        v1 = hand_coords[self.index_knuckle_idx] - wrist
        v2 = hand_coords[self.pinky_knuckle_idx] - wrist
        v3 = hand_coords[self.middle_knuckle_idx] - wrist

        palm_normal_raw = np.cross(v1, v3)
        # PICO reports left-hand gestures with mirrored handedness relative to
        # the right hand. Flip the palm normal so downstream operators always
        # receive the same right-handed frame convention for both hands.
        if self.hand_side == robots.LEFT:
            palm_normal_raw = -palm_normal_raw

        palm_normal = _safe_normalize(palm_normal_raw)
        palm_direction = _safe_normalize((v1 + v2 + v3) / 3)

        if palm_normal is None or palm_direction is None:
            return None
        cross_product = _safe_normalize(np.cross(palm_direction, palm_normal))
        if cross_product is None:
            return None

        frame = self._orthogonalize_frame(cross_product, palm_normal, palm_direction)
        if frame is None:
            return None
        x_vec, y_vec, z_vec = frame

        return [wrist, x_vec, y_vec, z_vec]

    def transform_keypoints(self, hand_coords):
        """
        对手部关键点进行坐标变换。

        变换步骤：
        1. 平移：以手腕为原点，将所有关键点平移到手腕中心坐标系
           translated_coords = hand_coords - hand_coords[0]
        2. 旋转：应用旋转矩阵将手部坐标旋转到标准朝向
           目前使用单位矩阵(np.eye(3))，保持原始坐标不变
           TODO: 后续替换为实际的旋转矩阵
        3. 计算手部方向帧：基于原始坐标计算手部朝向的3D坐标系

        Args:
            hand_coords: 手部坐标数组，形状为(26, 3)，原始世界坐标

        Returns:
            tuple: (transformed_keypoints, coordinate_frame)
                - transformed_keypoints: 变换后的关键点坐标，形状为(26, 3)
                - coordinate_frame: 手部方向帧 [wrist, x_vec, y_vec, z_vec]
        """
        is_valid, invalid_reason = self._validate_hand_coords(hand_coords)
        if not is_valid:
            self._warn_invalid_frame(invalid_reason)
            return None, None
        hand_coords = self._unity_left_handed_to_internal_right_handed(hand_coords)

        # 步骤1：平移 - 以手腕为原点
        translated_coords = copy(hand_coords) - hand_coords[0]

        # 步骤2：旋转 - 目前使用单位矩阵，保持原始坐标不变
        # TODO: 先使用单位矩阵作为旋转矩阵，保持原始坐标不变
        # # Use the new, more stable coordinate frame method
        # original_coord_frame = self._get_stable_coord_frame(translated_coords)
        # # Finding the rotation matrix and rotating the coordinates
        # rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T
        rotation_matrix = np.eye(3)

        transformed_keypoints = (rotation_matrix @ translated_coords.T).T

        # 步骤3：计算手部方向帧（使用原始坐标，不是平移后的坐标）
        coordinate_frame = self._get_stable_hand_dir_frame(hand_coords)
        if coordinate_frame is None:
            self._warn_invalid_frame("无法从手掌关键点建立稳定坐标系")
            return None, None

        return transformed_keypoints, coordinate_frame

    def _log_frame(self, keypoints, coordinate_frame):
        """记录帧数据到日志文件（如果启用）。"""
        if self.keypoint_logger is not None:
            self.keypoint_logger.log_frame(keypoints, coordinate_frame)

    def stream(self):
        """
        主流循环：处理手部关键点坐标变换。

        主循环流程：
        1. 以VR_FREQ(30Hz)频率运行
        2. 从pico4.py订阅原始关键点数据
        3. 执行坐标变换（平移+旋转）
        4. 对变换后的坐标和方向帧进行滑动平均平滑
        5. 对方向帧进行正交化处理
        6. 封装为InputFrame对象
        7. 发布变换后的数据给下游的xarm7_operator.py
        """
        while True:
            self.timer.start_loop()
            if self._get_arm_teleop_state() == robots.ARM_TELEOP_STOP:
                self._clear_smoothing_state()
                self.keypoint_subscriber.recv_keypoints()
                self.timer.end_loop()
                continue

            transform_start_s = time.perf_counter()

            # 1.从pico4.py订阅原始关键点数据
            data_type, hand_coords, hand_command, source_timestamp_s = self._get_hand_coords()

            if hand_coords is None or data_type is None:
                self.timer.end_loop()
                continue

            # 2. 执行坐标变换
            (
                transformed_keypoints,  # transformed_keypoints 是相对于手腕的
                coordinate_frame,  # [wrist, x_vec, y_vec, z_vec],手腕在真实 VR 空间中的绝对坐标 + 手在世界坐标系下的绝对朝向
            ) = self.transform_keypoints(hand_coords)
            if transformed_keypoints is None or coordinate_frame is None:
                self.timer.end_loop()
                continue

            # 3. 对变换后的坐标进行滑动平均平滑
            self.averaged_keypoints = moving_average(
                transformed_keypoints,
                self.coord_moving_average_queue,
                self.moving_average_limit,
            )

            # 4. 对方向帧向量进行滑动平均平滑
            self.averaged_coordinate_frame = moving_average(
                coordinate_frame,
                self.frame_moving_average_queue,
                self.moving_average_limit,
            )

            # 5. 确保方向帧向量保持正交性
            origin = self.averaged_coordinate_frame[0]
            x_vec = _safe_normalize(self.averaged_coordinate_frame[1])
            y_vec = _safe_normalize(self.averaged_coordinate_frame[2])
            z_vec = _safe_normalize(self.averaged_coordinate_frame[3])
            if x_vec is None or y_vec is None or z_vec is None:
                self._warn_invalid_frame("平滑后的方向向量退化")
                self.timer.end_loop()
                continue

            # 重新正交化
            orthogonal_frame = self._orthogonalize_frame(x_vec, y_vec, z_vec)
            if orthogonal_frame is None:
                self._warn_invalid_frame("正交化方向帧失败")
                self.timer.end_loop()
                continue
            x_vec, y_vec, z_vec = orthogonal_frame

            # 重构正交帧
            self.averaged_coordinate_frame = [origin, x_vec, y_vec, z_vec]

            if self._resume_settle_frames_remaining > 0:
                self._resume_settle_frames_remaining -= 1
                self.timer.end_loop()
                continue

            # 6. 封装为InputFrame对象
            data = InputFrame(
                timestamp_s=source_timestamp_s or time.time(),
                hand_side=self.hand_side,
                keypoints=self.averaged_keypoints,
                is_relative=data_type == self.relative_mode,
                frame_vectors=self.averaged_coordinate_frame,
                hand_command=hand_command,
            )

            # 7. 发布变换后的坐标给下游组件
            self.publisher_manager.publish(
                host=self.host,
                port=self.keypoint_transform_pub_port,
                topic=self.coords_topic,
                data=data,
            )
            if self.keypoint_logger is not None:
                self._log_frame(self.averaged_keypoints, self.averaged_coordinate_frame)
            # 8.发布方向帧给下游组件（xarm7_operator.py使用frame_topic）
            self.publisher_manager.publish(
                host=self.host,
                port=self.keypoint_transform_pub_port,
                topic=self.frame_topic,
                data=data,
            )
            self._log_timing_diag(data.timestamp_s, transform_start_s)
            self._log_publish_frame(self.averaged_coordinate_frame, hand_command)

            self.timer.end_loop()

    def _log_timing_diag(self, source_timestamp_s: float, transform_start_s: float) -> None:
        now = time.time()
        if now - self._last_timing_log_time < 1.0:
            return
        self._last_timing_log_time = now
        logger.info(
            "[Diag][TIMING_TRANSFORM] side=%s source_to_pub_ms=%.1f transform_loop_ms=%.1f",
            self.hand_side,
            (now - float(source_timestamp_s or now)) * 1000.0,
            (time.perf_counter() - transform_start_s) * 1000.0,
        )

    def cleanup(self):
        """清理资源并保存日志数据。"""
        if self.keypoint_logger is not None:
            logger.info("Cleanup called. Saving final logged data...")
            self.keypoint_logger.save_data()

        self.keypoint_subscriber.stop()
        self.pause_subscriber.stop()
        cleanup_zmq_resources()

    def __del__(self):
        self.cleanup()
