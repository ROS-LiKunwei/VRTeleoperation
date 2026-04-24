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


class OculusVRHandDetector(Component):
    """
    统一的 Oculus VR 手部检测器，能够处理左手、右手或双手的检测。
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
        self.notify_component_start(robots.VR_DETECTOR)

        self.host = host
        self.oculus_pub_port = oculus_pub_port
        self.button_port = button_port
        self.teleop_reset_port = teleop_reset_port
        self.hand_config = hand_config

        self._configure_hand_ports(right_hand_port, left_hand_port)
        self._initialize_sockets()

        self.publisher_manager = ZMQPublisherManager.get_instance()
        self.timer = FrequencyTimer(robots.VR_FREQ)
        self.last_received = dict.fromkeys(self.sockets, 0)

    def _configure_hand_ports(self, right_hand_port: Optional[int], left_hand_port: Optional[int]):
        self.hand_ports = {}
        if self.hand_config in [robots.RIGHT, robots.BIMANUAL]:
            if right_hand_port is None:
                right_hand_port = network.RIGHT_HAND_PORT
            self.hand_ports[robots.RIGHT] = right_hand_port
        if self.hand_config in [robots.LEFT, robots.BIMANUAL]:
            if left_hand_port is None:
                left_hand_port = network.LEFT_HAND_PORT
            self.hand_ports[robots.LEFT] = left_hand_port

    def _initialize_sockets(self):
        self.sockets = {}
        for hand_side, port in self.hand_ports.items():
            socket_key = f"{robots.KEYPOINTS}_{hand_side}"
            self.sockets[socket_key] = create_pull_socket(self.host, port)
        self.sockets[robots.BUTTON] = create_pull_socket(self.host, self.button_port)
        self.sockets[robots.PAUSE] = create_pull_socket(self.host, self.teleop_reset_port)

    def _process_keypoints(self, data):
        """
        处理原始的骨骼关键点数据，将其转换为坐标列表。
        """
        data_str = data.decode().strip()
        values = []

        parts = data_str.split(":")
        if len(parts) >= 2:
            coords = parts[1].strip().split("|")
        else:
            return []

        for coord in coords:
            try:
                values.extend(float(val) for val in coord.split(",")[:3])
            except ValueError:
                pass

        return values

    def _receive_data(self, socket_name):
        try:
            data = self.sockets[socket_name].recv(zmq.NOBLOCK)
            self.last_received[socket_name] = time.time()
            return data
        except zmq.Again:
            return None

    def stream(self):
        logger.info(f"Starting VR hand detection with configuration: {self.hand_config}")

        while True:
            self.timer.start_loop()

            for hand_side in self.hand_ports:
                socket_key = f"{robots.KEYPOINTS}_{hand_side}"
                keypoint_data = self._receive_data(socket_key)

                if keypoint_data is not None:
                    keypoints = self._process_keypoints(keypoint_data)
                    is_relative = not keypoint_data.decode().strip().startswith(robots.ABSOLUTE)

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

                    logger.info(f"[Bot接收] {hand_side} 关键点数据长度: {len(keypoints)}")

            if button_data := self._receive_data(robots.BUTTON):
                hand_side = (
                    robots.RIGHT if robots.RIGHT in self.hand_ports else list(self.hand_ports.keys())[0]
                )
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

            self.timer.end_loop()

        for name, socket in self.sockets.items():
            socket.close()
            logger.info(f"Closed {name} socket")
        logger.info("Stopped VR hand detection process.")
