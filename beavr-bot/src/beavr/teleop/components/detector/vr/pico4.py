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
    PICO4 VR手部追踪探测器,支持左手、右手或双手追踪。

    该类基于提供的手部配置动态配置自身，无需单独的单手和双手探测器类。
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
            host: PICO4 VR头显的主机地址。
            pico4_pub_port: 发布关键点数据的端口号。
            button_port: 按钮事件的端口号。
            teleop_reset_port: 遥控重置命令的端口号。
            hand_config: 配置模式 - 'left'、'right' 或 'bimanual'
            right_hand_port: 右手数据端口（右手/双手模式需要）
            left_hand_port: 左手数据端口（左手/双手模式需要）
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
        self.timer = FrequencyTimer(robots.VR_FREQ)
        self.last_received = {}
        self.sockets = None  # 延迟初始化套接字

    def _configure_hand_ports(self, right_hand_port: Optional[int], left_hand_port: Optional[int]):
        """根据手部配置配置手部端口。"""
        self.hand_ports = {}

        if self.hand_config in [robots.RIGHT, robots.BIMANUAL]:
            if right_hand_port is None:
                right_hand_port = network.RIGHT_HAND_PICO4_PORT
            self.hand_ports[robots.RIGHT] = right_hand_port

        if self.hand_config in [robots.LEFT, robots.BIMANUAL]:
            if left_hand_port is None:
                left_hand_port = network.LEFT_HAND_PICO4_PORT
            self.hand_ports[robots.LEFT] = left_hand_port

    def _initialize_sockets(self):
        """根据手部配置初始化套接字。"""
        self.sockets = {}

        # 创建手部特定的关键点套接字
        for hand_side, port in self.hand_ports.items():
            socket_key = f"{robots.KEYPOINTS}_{hand_side}"
            self.sockets[socket_key] = create_pull_socket(self.host, port)

        # 按钮和暂停的共享套接字（只需要一个实例）
        self.sockets[robots.BUTTON] = create_pull_socket(self.host, self.button_port)
        self.sockets[robots.PAUSE] = create_pull_socket(self.host, self.teleop_reset_port)

    def _process_keypoints(self, data):
        """将原始关键点数据处理为坐标值列表。"""
        data_str = data.decode().strip()
        values = []

        # 解析坐标（格式：<hand>:x,y,z|x,y,z|x,y,z）
        coords = data_str.split(":")[1].strip().split("|")
        for coord in coords:
            values.extend(float(val) for val in coord.split(",")[:3])

        return values

    def _receive_data(self, socket_name):
        """从套接字接收数据。"""
        if self.sockets is None:
            return None
        try:
            data = self.sockets[socket_name].recv(zmq.NOBLOCK)
            self.last_received[socket_name] = int(time.time())
            return data
        except zmq.Again:
            return None

    def stream(self):
        """统一VR手部检测的主流循环。"""
        logger.info(f"Starting PICO4 VR hand detection with configuration: {self.hand_config}")
        
        # 延迟初始化套接字
        self._initialize_sockets()
        self.last_received = dict.fromkeys(self.sockets, 0)

        while True:
            self.timer.start_loop()

            # 处理所有配置的手部的关键点数据
            for hand_side in self.hand_ports:
                socket_key = f"{robots.KEYPOINTS}_{hand_side}"
                keypoint_data = self._receive_data(socket_key)

                if keypoint_data is not None:
                    # 处理并发布此手的关键点
                    keypoints = self._process_keypoints(keypoint_data)
                    is_relative = not keypoint_data.decode().strip().startswith(robots.ABSOLUTE)

                    # TODO: 我们真的只需要发布一次！
                    # 我们可以将所有信息存储在单个模式表中

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

            # 处理并发布按钮状态（在手部之间共享）
            if button_data := self._receive_data(robots.BUTTON):
                # 对于按钮事件，使用第一个配置的手部侧作为源
                # 或在双手设置中默认使用'right'
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
        # 退出时清理套接字
        for name, socket in self.sockets.items():
            socket.close()
            logger.info(f"Closed {name} socket")
        logger.info("Stopped PICO4 VR hand detection process.")
