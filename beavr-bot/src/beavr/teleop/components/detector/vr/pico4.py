"""
PICO4 VR手部追踪探测器模块

本模块是BeaVR-bot遥操作系统的第一环，负责从PICO4 VR头显接收手部关键点数据，
解析后通过ZMQ发布给下游的坐标变换组件（keypoint_transform.py）。

数据流位置：
    PICO4 VR头显 (Unity端) → [本模块: pico4.py] → keypoint_transform.py → xarm7_operator.py → xarm7_robot.py

通信协议：
    - 接收：ZMQ PULL套接字，从PICO4 Unity应用接收原始手部数据
    - 发送：ZMQ PUB套接字，发布解析后的InputFrame对象给下游组件

数据格式：
    - 接收格式："<type_marker>:x,y,z|x,y,z|..." （字符串，冒号前为类型标记，冒号后为26个关节坐标）
    - 发送格式：InputFrame对象（包含时间戳、手侧、关键点序列、相对/绝对模式、方向帧向量）

端口配置：
    - 右手数据接收端口：8087 (RIGHT_HAND_PICO4_PORT)
    - 左手数据接收端口：8110 (LEFT_HAND_PICO4_PORT)
    - 关键点发布端口：8088 (KEYPOINT_STREAM_PORT)
    - 按钮事件端口：8095 (RESOLUTION_BUTTON_PORT)
    - 暂停/恢复端口：8100 (TELEOP_RESET_PORT)
"""

import logging
import time
from typing import Optional, Union

import zmq

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.utils import create_pull_socket
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import (
    ButtonEvent,
    InputFrame,
    SessionCommand,
)
from beavr.teleop.configs.constants import network, robots

logger = logging.getLogger(__name__)


class PICO4VRHandDetector(Component):
    """
    PICO4 VR手部追踪探测器，支持左手、右手或双手追踪。

    该类基于提供的手部配置动态配置自身，无需单独的单手和双手探测器类。

    数据流角色：
        本类是整个遥操作系统数据流的起点（BeaVR-bot端），
        负责从PICO4 VR头显的Unity应用接收手部关键点数据，
        解析为结构化的InputFrame对象后发布给下游组件。

    工作流程：
        1. 初始化ZMQ PULL套接字，连接到PICO4 Unity应用的PUSH端口
        2. 在主循环中，以VR_FREQ(30Hz)频率接收原始手部数据
        3. 解析原始数据格式："<type_marker>:x,y,z|x,y,z|..."
           - type_marker: "relative"（相对模式）或 "absolute"（绝对模式）
           - 坐标部分：26个关节的xyz坐标，用"|"分隔
        4. 将解析后的数据封装为InputFrame对象
        5. 通过ZMQ PUB套接字发布InputFrame对象给下游的keypoint_transform.py

    手势控制：
        - 左手食指捏合：启动相对数据模式（StreamRelativeData）
        - 左手中指捏合：启动绝对数据模式（StreamAbsoluteData）
        - 左手无名指捏合：停止遥操作
        - 高/低分辨率按钮：控制运动缩放比例
    """

    def __init__(
        self,
        host: str,
        pico4_pub_port: int,
        button_port: int,
        teleop_reset_port: int,
        hand_config: Union[str, str] = robots.RIGHT,
        right_hand_port: Optional[int] = None,
        left_hand_port: Optional[int] = None,
    ):
        """
        初始化PICO4 VR手部追踪探测器组件。

        Args:
            host: PICO4 VR头显的主机地址（ZMQ通信地址）。
            pico4_pub_port: 发布关键点数据的端口号（发布给keypoint_transform.py）。
            button_port: 按钮事件的端口号（接收分辨率切换命令）。
            teleop_reset_port: 遥控重置命令的端口号（接收暂停/恢复命令）。
            hand_config: 配置模式 - 'left'（左手）、'right'（右手）或 'bimanual'（双手）。
            right_hand_port: 右手数据端口（右手/双手模式需要），默认8087。
            left_hand_port: 左手数据端口（左手/双手模式需要），默认8110。
        """
        self.notify_component_start(robots.VR_DETECTOR)

        self.host = host
        self.pico4_pub_port = pico4_pub_port
        self.button_port = button_port
        self.teleop_reset_port = teleop_reset_port
        self.hand_config = hand_config

        # 根据配置验证并设置手部端口
        self._configure_hand_ports(right_hand_port, left_hand_port)

        # 初始化发布器和计时器
        self.publisher_manager = ZMQPublisherManager.get_instance()
        self.timer = FrequencyTimer(robots.VR_FREQ)  # 30Hz频率
        self.last_received = {}
        self.sockets = None  # 延迟初始化套接字

        # 接收频率统计
        self._receive_counts = {}
        self._last_receive_freq_log_time = {}
        self._receive_frequencies = {}
        self._last_receive_time = {}
        self._last_wrist_log_time = 0.0
        self._last_full_joint_log_time = 0.0
        self._freq_calc_interval = 1.0  # 1秒计算一次频率
        self._frame_index = 0  # 帧索引，用于匹配三个环节的数据

    def _configure_hand_ports(self, right_hand_port: Optional[int], left_hand_port: Optional[int]):
        """
        根据手部配置配置手部端口。

        根据hand_config参数（right/left/bimanual），设置需要接收数据的手部端口。
        默认端口：右手8087，左手8110。
        """
        self.hand_ports = {}

        if self.hand_config in [robots.RIGHT, robots.BIMANUAL]:
            if right_hand_port is None:
                right_hand_port = network.RIGHT_HAND_PICO4_PORT  # 8087
            self.hand_ports[robots.RIGHT] = right_hand_port

        if self.hand_config in [robots.LEFT, robots.BIMANUAL]:
            if left_hand_port is None:
                left_hand_port = network.LEFT_HAND_PICO4_PORT  # 8110
            self.hand_ports[robots.LEFT] = left_hand_port

    def _initialize_sockets(self):
        """
        根据手部配置初始化ZMQ sockets。

        创建以下套接字：
        - 手部关键点套接字（PULL）：接收PICO4 Unity应用发送的手部数据
          - socket_key: "RightHand" 或 "LeftHand"
        - 按钮套接字（PULL）：接收分辨率切换命令
        - 暂停套接字（PULL）：接收暂停/恢复命令
        """
        self.sockets = {}

        # 创建手部特定的关键点套接字
        for hand_side, port in self.hand_ports.items():
            # 使用与PICO发送匹配的套接字名称
            socket_key = "RightHand" if hand_side == "right" else "LeftHand"
            self.sockets[socket_key] = create_pull_socket(self.host, port)

        # 按钮和暂停的共享套接字（只需要一个实例）
        self.sockets[robots.BUTTON] = create_pull_socket(self.host, self.button_port)
        self.sockets[robots.PAUSE] = create_pull_socket(self.host, self.teleop_reset_port)

    def _process_keypoints(self, data):
        """
        将原始关键点数据处理为坐标值列表。

        解析PICO4 Unity应用发送的原始数据格式：
        "<type_marker>:x1,y1,z1|x2,y2,z2|...|x26,y26,z26"

        解析步骤：
        1. 解码字节流为字符串
        2. 按冒号分割，获取类型标记和坐标部分
        3. 按竖线分割坐标部分，获取每个关节的xyz坐标
        4. 按逗号分割每个坐标，转换为float值
        5. 验证坐标数量是否为26*3=78

        Args:
            data: 原始字节数据，格式为 b"<type_marker>:x,y,z|x,y,z|..."

        Returns:
            list: 扁平化的坐标值列表，长度应为78（26个关节×3个坐标）。
                  解析失败时返回空列表。
        """
        try:
            data_str = data.decode().strip()
            values = []

            # 解析坐标（格式：<hand>:x,y,z|x,y,z|x,y,z）
            if ':' not in data_str:
                logger.warning(f"_process_keypoints: 数据格式错误，缺少冒号分隔符: {data_str}")
                return []

            parts = data_str.split(":")
            if len(parts) < 2:
                logger.warning(f"_process_keypoints: 数据格式错误，分割后部分不足: {data_str}")
                return []

            coords_part = parts[1].strip()
            if not coords_part:
                logger.warning(f"_process_keypoints: 坐标部分为空: {data_str}")
                return []

            coords = coords_part.split("|")
            if not coords:
                logger.warning(f"_process_keypoints: 坐标列表为空: {data_str}")
                return []

            for coord in coords:
                coord = coord.strip()
                if not coord:
                    continue
                coord_values = coord.split(",")[:3]
                if len(coord_values) != 3:
                    logger.warning(f"_process_keypoints: 坐标格式错误: {coord}")
                    continue
                try:
                    values.extend(float(val) for val in coord_values)
                except ValueError as e:
                    logger.warning(f"_process_keypoints: 坐标值转换错误: {e}, 坐标: {coord}")
                    continue

            expected_count = 26 * 3  # 26个关节，每个3个坐标
            actual_count = len(values)
            if actual_count != expected_count:
                logger.warning(
                    f"_process_keypoints: 坐标数量不匹配. "
                    f"期望 {expected_count} 个值, 得到 {actual_count}."
                )

            return values
        except Exception as e:
            logger.error(f"_process_keypoints: 处理数据时出错: {e}")
            return []

    def _receive_data(self, socket_name):
        """
        从指定ZMQ套接字接收数据（非阻塞模式）。

        Args:
            socket_name: 套接字名称（如"RightHand"、"LeftHand"、robots.BUTTON、robots.PAUSE）

        Returns:
            bytes: 接收到的原始字节数据，无数据时返回None。
        """
        if self.sockets is None:
            return None
        try:
            data = self.sockets[socket_name].recv(zmq.NOBLOCK)
            self.last_received[socket_name] = int(time.time())
            return data
        except zmq.Again:
            return None

    def _parse_wrist_data(self, data):
        """
        解析手腕部数据，用于调试日志打印。

        从原始数据中提取手腕（第0个关节）和手掌（第1个关节）的坐标。

        Args:
            data: 原始字节数据

        Returns:
            str: 格式化的手腕部数据字符串，如 "手腕=x,y,z 手掌=x,y,z"
        """
        try:
            data_str = data.decode().strip()
            parts = data_str.split(":")
            if len(parts) < 2:
                return "格式错误"

            coords = parts[1].split("|")
            if len(coords) < 2:
                return "数据不足"

            wrist = coords[0].strip()
            palm = coords[1].strip()
            return f"手腕={wrist} 手掌={palm}"
        except Exception as e:
            return f"解析错误: {e}"

    def _parse_full_joint_data(self, data):
        """
        解析26个关节数据，用于调试日志打印。

        从原始数据中提取所有26个关节的坐标，并添加索引标注。

        Args:
            data: 原始字节数据

        Returns:
            str: 格式化的26关节数据字符串，如 "0:x,y,z 1:x,y,z ... 25:x,y,z"
        """
        try:
            data_str = data.decode().strip()
            parts = data_str.split(":")
            if len(parts) < 2:
                return "格式错误"

            coords = parts[1].split("|")
            result = []

            for i in range(min(26, len(coords))):
                joint = coords[i].strip()
                result.append(f"{i}:{joint}")

            return " ".join(result)
        except Exception as e:
            return f"解析错误: {e}"

    def stream(self):
        """
        统一VR手部检测的主流循环。

        主循环流程：
        1. 以VR_FREQ(30Hz)频率运行
        2. 对每只配置的手：
           a. 从ZMQ PULL套接字接收原始手部数据
           b. 解析原始数据为坐标值列表
           c. 判断数据模式（相对/绝对）
           d. 封装为InputFrame对象
           e. 通过ZMQ PUB套接字发布给下游组件（keypoint_transform.py）
           f. 定期打印调试日志（频率、手腕数据、26关节数据）
        3. 处理按钮事件和暂停/恢复命令
        """
        logger.info(f"Starting PICO4 VR hand detection with configuration: {self.hand_config}")

        # 延迟初始化sockets
        self._initialize_sockets()
        self.last_received = dict.fromkeys(self.sockets, 0)

        while True:
            self.timer.start_loop()

            # 处理所有配置的手部的关键点数据
            for hand_side in self.hand_ports:
                # 使用与PICO发送匹配的套接字名称
                socket_key = "RightHand" if hand_side == "right" else "LeftHand"
                keypoint_data = self._receive_data(socket_key)

                if keypoint_data is not None:
                    # 处理并发布此手的关键点
                    keypoints = self._process_keypoints(keypoint_data)
                    is_relative = not keypoint_data.decode().strip().startswith(robots.ABSOLUTE)
                    # TODO: 我们真的只需要发布一次！
                    # 我们可以将所有信息存储在单个模式表中
                    # 发布InputFrame对象给下游的keypoint_transform.py
                    # InputFrame包含：时间戳、手侧、关键点序列、相对/绝对模式、方向帧向量
                    self.publisher_manager.publish(
                        host=self.host,
                        port=self.pico4_pub_port,
                        topic=hand_side,
                        data=InputFrame(
                            timestamp_s=time.time(),
                            hand_side=hand_side,
                            keypoints=keypoints,
                            is_relative=is_relative,
                            frame_vectors=None,
                        ),
                    )

                    # 接收频率统计
                    if socket_key not in self._receive_counts:
                        self._receive_counts[socket_key] = 0
                        self._last_receive_freq_log_time[socket_key] = time.time()
                        self._receive_frequencies[socket_key] = 0.0
                        self._last_receive_time[socket_key] = time.time()
                        self._last_wrist_log_time = 0.0

                    self._receive_counts[socket_key] += 1
                    current_time = time.time()
                    if current_time - self._last_receive_freq_log_time[socket_key] >= self._freq_calc_interval:
                        self._receive_frequencies[socket_key] = self._receive_counts[socket_key] / (current_time - self._last_receive_freq_log_time[socket_key])
                        self._receive_counts[socket_key] = 0
                        self._last_receive_freq_log_time[socket_key] = current_time
                        logger.info(f"[Bot接收] {socket_key} 接收频率: {self._receive_frequencies[socket_key]:.1f} Hz")

                    # 定期打印手腕部数据（每3秒）
                    if current_time - getattr(self, '_last_wrist_log_time', 0) >= 3.0:
                        self._last_wrist_log_time = current_time
                        wrist_data = self._parse_wrist_data(keypoint_data)
                        logger.info(f"[Bot接收] index={self._frame_index} {socket_key} 手腕数据: {wrist_data}")
                        self._frame_index += 1

                    # 定期打印26个坐标系数据（每5秒）
                    if current_time - getattr(self, '_last_full_joint_log_time', 0) >= 5.0:
                        self._last_full_joint_log_time = current_time
                        full_joint_data = self._parse_full_joint_data(keypoint_data)
                        logger.info(f"[Bot接收] index={self._frame_index} {socket_key} 26关节数据: {full_joint_data}")
                        self._frame_index += 1

                    # 定期打印接收到的位姿信息
                    if current_time - self._last_receive_time.get(socket_key, 0) >= 3.0:
                        self._last_receive_time[socket_key] = current_time
                        pose_sample = keypoints[:9] if len(keypoints) >= 9 else keypoints
                        logger.info(f"[Bot接收] {socket_key} 位姿样本: {pose_sample}")

                    logger.debug(
                        f"PICO4: 发布 {hand_side} 手数据到端口 {self.pico4_pub_port}, "
                        f"关键点数量: {len(keypoints)}, 相对模式: {is_relative}"
                    )

            # 处理并发布按钮状态（在手部之间共享）
            if button_data := self._receive_data(robots.BUTTON):
                hand_side = (
                    robots.RIGHT if robots.RIGHT in self.hand_ports else list(self.hand_ports.keys())[0]
                )

                self.publisher_manager.publish(
                    host=self.host,
                    port=self.pico4_pub_port,
                    topic=robots.BUTTON,
                    data=ButtonEvent(
                        timestamp_s=time.time(),
                        hand_side=hand_side,
                        name=robots.BUTTON,
                        value=robots.ARM_LOW_RESOLUTION
                        if button_data == b"Low"
                        else robots.ARM_HIGH_RESOLUTION,
                    ),
                )

            # 处理并发布暂停状态（在手部之间共享）
            if pause_data := self._receive_data(robots.PAUSE):
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.pico4_pub_port,
                    topic=robots.PAUSE,
                    data=SessionCommand(
                        timestamp_s=time.time(),
                        command="resume" if pause_data == b"Low" else "pause",
                    ),
                )

            self.timer.end_loop()

        # TODO: 我们需要比这更好的清理方法
        # 退出时清理sockets
        for name, socket in self.sockets.items():
            socket.close()
            logger.info(f"Closed {name} socket")
        logger.info("Stopped PICO4 VR hand detection process.")
