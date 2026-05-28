import logging
import time

from beavr.lerobot.common.robot_devices.cameras.configs import (
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from beavr.lerobot.common.robot_devices.cameras.intelrealsense import (
    IntelRealSenseCamera,
)
from beavr.lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from beavr.teleop.common.network.publisher import ZMQCompressedImageTransmitter
from beavr.teleop.components.component import Component

logger = logging.getLogger(__name__)


class RealCameraStreamer(Component):
    """Read frames from a real camera and publish them to the VR image stream."""

    def __init__(
        self,
        camera_config: OpenCVCameraConfig | IntelRealSenseCameraConfig,
        host: str,
        port: int,
        camera_name: str = "front",
    ):
        self.notify_component_start(f"real_camera_streamer_{camera_name}")
        self.camera_config = camera_config
        self.host = host
        self.port = port
        self.camera_name = camera_name
        self.camera = self._make_camera(camera_config)
        self.publisher = ZMQCompressedImageTransmitter(host=host, port=port)

    @staticmethod
    def _make_camera(camera_config: OpenCVCameraConfig | IntelRealSenseCameraConfig):
        if isinstance(camera_config, OpenCVCameraConfig):
            return OpenCVCamera(camera_config)
        if isinstance(camera_config, IntelRealSenseCameraConfig):
            return IntelRealSenseCamera(camera_config)
        raise ValueError(f"Unsupported camera config type: {type(camera_config)}")

    def stream(self):
        logger.info(
            "Starting real camera stream '%s' on tcp://*:%s",
            self.camera_name,
            self.port,
        )
        self.camera.connect()

        if self.camera.fps is None:
            self.camera.read()
        else:
            self.camera.async_read()

        target_dt = 1.0 / self.camera.fps if self.camera.fps else 0.0
        try:
            while True:
                start = time.perf_counter()
                frame = self.camera.read() if self.camera.fps is None else self.camera.async_read()

                if isinstance(frame, tuple):
                    frame = frame[0]

                self.publisher.send_image(frame)

                if target_dt > 0:
                    elapsed = time.perf_counter() - start
                    time.sleep(max(0.0, target_dt - elapsed))
        finally:
            try:
                self.camera.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting real camera '%s': %s", self.camera_name, exc)
            self.publisher.stop()
