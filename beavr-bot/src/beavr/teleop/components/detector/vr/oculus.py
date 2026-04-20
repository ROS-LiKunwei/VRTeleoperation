import logging
import time
from typing import Optional, Union

import zmq # 核心：BeaVR 使用 ZMQ 协议在各个独立进程之间极速传输数据

# 引入 BeaVR 框架内部的基础组件和数据类型定义
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


class OculusVRHandDetector(Component):
    """ 
        统一的 Oculus VR 手部检测器，能够处理左手、右手或双手的检测。
        这个类会根据所提供的手动配置来动态地进行自我配置，消除了对单独的单手和双手检测器类别的需求。
    """

    def __init__(
        self,
        host: str,
        oculus_pub_port: int,
        button_port: int,
        teleop_reset_port: int,
        hand_config: Union[str, str] = robots.RIGHT,
        right_hand_port: Optional[int] = None,
        left_hand_port: Optional[int] = None,
    ):
        """
        初始化统一的OculusVRHandDetector组件。
        【PICO适配注意】:PICO 的检测器初始化也需要这些参数，确保能对接后端的端口。
        Args:
            host: Oculus VR头显的主机地址。
            oculus_pub_port: 发布关键点数据的端口号。
            button_port: 按钮事件的端口号。
            teleop_reset_port: 遥控重置命令的端口号。
            hand_config: 配置模式 - 'left', 'right', or 'bimanual'
            right_hand_port: 右手数据端口号（仅用于右手或双手配置）。
            left_hand_port: 左手数据端口号（仅用于左手或双手配置）。
        """
        # 通知系统该组件已启动
        self.notify_component_start(robots.VR_DETECTOR)

        self.host = host
        self.oculus_pub_port = oculus_pub_port
        self.button_port = button_port
        self.teleop_reset_port = teleop_reset_port
        self.hand_config = hand_config

        # 1. 根据传入的配置（左手/右手/双手），确定需要监听哪些端口
        self._configure_hand_ports(right_hand_port, left_hand_port)

        # 2. 建立与这些端口的 ZMQ 连接
        self._initialize_sockets()

        # 3. 初始化发布者（用于将处理好的数据发给下游的机器人控制器）和定时器
        self.publisher_manager = ZMQPublisherManager.get_instance()
        self.timer = FrequencyTimer(robots.VR_FREQ) # 确保以固定频率运行
        self.last_received = dict.fromkeys(self.sockets, 0)

    def _configure_hand_ports(self, right_hand_port: Optional[int], left_hand_port: Optional[int]):
        """ 根据配置分配左右手的独立通讯端口 """
        self.hand_ports = {}
        # 如果配置了右手或双手，分配右手端口
        if self.hand_config in [robots.RIGHT, robots.BIMANUAL]:
            if right_hand_port is None:
                right_hand_port = network.RIGHT_HAND_PORT
            self.hand_ports[robots.RIGHT] = right_hand_port
        # 如果配置了左手或双手，分配左手端口
        if self.hand_config in [robots.LEFT, robots.BIMANUAL]:
            if left_hand_port is None:
                left_hand_port = network.LEFT_HAND_PORT
            self.hand_ports[robots.LEFT] = left_hand_port

    def _initialize_sockets(self):
        """ 初始化所有的 ZMQ Pull Socket(作为接收端等待数据) """
        self.sockets = {}

        # 为每个手创建一个 ZMQ Pull Socket，用于接收骨骼关键点数据
        for hand_side, port in self.hand_ports.items():
            socket_key = f"{robots.KEYPOINTS}_{hand_side}"
            self.sockets[socket_key] = create_pull_socket(self.host, port)

        # 创建一个 ZMQ Pull Socket，用于接收按钮事件和暂停事件 (only one instance needed)
        self.sockets[robots.BUTTON] = create_pull_socket(self.host, self.button_port)
        self.sockets[robots.PAUSE] = create_pull_socket(self.host, self.teleop_reset_port)

    def _process_keypoints(self, data):
        """
        处理原始的骨骼关键点数据，将其转换为坐标列表。
        【PICO适配最关键的一步】:
        原本 Oculus 的 Unity 客户端发送的数据是一个拼起来的字符串，长这样：
        "right:0.1,0.2,0.3|0.4,0.5,0.6|..."
        所以这里用了 .decode().split() 来暴力切分字符串。
        但在你的环境里,TeleVuer 很可能会发送结构化的 JSON 数据！你需要在这里修改解析逻辑。
        """
        data_str = data.decode().strip()
        values = []

        # 按冒号分割，取后面的坐标部分，再按竖线分割每个关节(format: <hand>:x,y,z|x,y,z|x,y,z)
        coords = data_str.split(":")[1].strip().split("|")
        for coord in coords:
            values.extend(float(val) for val in coord.split(",")[:3])

        return values

    def _receive_data(self, socket_name):
        """ 从指定的 socket 非阻塞地接收数据 """
        try:
            data = self.sockets[socket_name].recv(zmq.NOBLOCK)
            self.last_received[socket_name] = time.time()
            return data
        except zmq.Again:
            return None # 如果没收到数据就不管，直接返回 None

    def stream(self):
        """ 核心流式循环：持续监听、处理并转发数据 """
        logger.info(f"Starting VR hand detection with configuration: {self.hand_config}")

        while True:
            self.timer.start_loop() # 维持固定的循环频率

            # 1. 轮询每一只手的关键点数据
            for hand_side in self.hand_ports:
                socket_key = f"{robots.KEYPOINTS}_{hand_side}"
                keypoint_data = self._receive_data(socket_key)

                if keypoint_data is not None:
                    # 处理并发布此手的关键点信息
                    keypoints = self._process_keypoints(keypoint_data)
                    # 判断数据是绝对坐标还是相对坐标 (取决于字符串开头)
                    is_relative = not keypoint_data.decode().strip().startswith(robots.ABSOLUTE)

                    # TODO: 我们真的只需要发布一次！
                    # 我们可以将所有信息存储在一个单个模式表中
                    
                    # 将解析后的数据打包成标准 InputFrame，通过发布者(Publisher)发给机器人的运动学解算模块
                    self.publisher_manager.publish(
                        host=self.host,
                        port=self.oculus_pub_port,
                        topic=hand_side,
                        data=InputFrame(
                            timestamp_s=time.time(),
                            hand_side=hand_side,
                            keypoints=keypoints,
                            is_relative=is_relative,
                            frame_vectors=None,
                        ),
                    )

            # 2. 处理手柄按钮事件（用于切换机械臂分辨率/精度等）(双手共享)
            if button_data := self._receive_data(robots.BUTTON):
                # 对于按钮事件，在双手设置中，使用第一个配置的手侧作为源侧，或默认使用“右手”
                hand_side = (
                    robots.RIGHT if robots.RIGHT in self.hand_ports else list(self.hand_ports.keys())[0]
                )
                
                # 将字符串 "Low" 转换为对应的枚举状态并发布
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.oculus_pub_port,
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

            # 3. 处理系统的暂停/恢复指令(双手共享)
            if pause_data := self._receive_data(robots.PAUSE):
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.oculus_pub_port,
                    topic=robots.PAUSE,
                    data=SessionCommand(
                        timestamp_s=time.time(),
                        command="resume" if pause_data == b"Low" else "pause",
                    ),
                )

            self.timer.end_loop() # 结束本次循环控制频率

        # TODO: 我们需要比这更好的清理机制。退出时清理sokets(通信通道/数据接收端点）
        # 清理机制（目前是死循环，这里理论上走不到，除非外部强杀）
        for name, socket in self.sockets.items():
            socket.close()
            logger.info(f"Closed {name} socket")
        logger.info("Stopped VR hand detection process.")
