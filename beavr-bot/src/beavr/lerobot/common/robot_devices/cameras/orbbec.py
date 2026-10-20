import logging
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .configs import OrbbecCameraConfig
from .recording_fault import CameraFault

logger = logging.getLogger(__name__)

# Keep one SDK context alive for every Pipeline. This is important when two
# Orbbec devices are opened by the same recording process.
_context: Any | None = None
_context_lock = threading.Lock()


def _sdk_context():
    global _context
    with _context_lock:
        if _context is None:
            from pyorbbecsdk import Context

            _context = Context()
        return _context


class OrbbecCamera:
    """Synchronized RGB + millimetre uint16 depth capture via pyorbbecsdk."""

    def __init__(self, config: OrbbecCameraConfig):
        self.width, self.height, self.fps = config.width, config.height, config.fps
        self.logs = {}
        self.config = config
        self.pipeline = None
        self.sdk_config = None
        self._align_filter = None
        self.effective_controls = None
        self.stream_profiles = {}
        self._connected = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._latest_color: NDArray[Any] | None = None
        self._latest_depth: NDArray[Any] | None = None
        self._capture_error = None
        self._latest_timestamp = 0.0
        self._sequence = 0
        self._last_read_sequence = 0

    def __repr__(self) -> str:
        return f"OrbbecCamera({self.config.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def use_depth(self) -> bool:
        return self.config.use_depth

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        devices = _sdk_context().query_devices()
        result = []
        for index in range(devices.get_count()):
            info = devices[index].get_device_info()
            result.append({"name": info.get_name(), "serial_number": info.get_serial_number()})
        return result

    def _find_device(self):
        devices = _sdk_context().query_devices()
        available = []
        for index in range(devices.get_count()):
            device = devices[index]
            serial = device.get_device_info().get_serial_number()
            available.append(serial)
            if serial == self.config.serial_number:
                return device
        raise RuntimeError(
            f"Orbbec serial {self.config.serial_number!r} was not found; connected: {available}"
        )

    def _select_default_profile(self, sensor_type):
        """Use the SDK's known-compatible default stream profile.

        Selecting color and depth profiles independently by requested output
        size can produce a pair which starts successfully but never yields a
        complete frameset on Gemini 305. Output resizing is handled after the
        synchronized frameset has been received.
        """
        profiles = self.pipeline.get_stream_profile_list(sensor_type)
        profile = profiles.get_default_video_stream_profile()
        logger.info(
            "%s: SDK stream %sx%s@%s, output %sx%s",
            self,
            profile.get_width(),
            profile.get_height(),
            profile.get_fps(),
            self.width,
            self.height,
        )
        self.stream_profiles[str(sensor_type)] = {
            "width": profile.get_width(), "height": profile.get_height(),
            "fps": profile.get_fps(), "format": str(profile.get_format()),
        }
        expected = self.config.expected_stream_profiles.get(str(sensor_type))
        if expected is not None and expected != self.stream_profiles[str(sensor_type)]:
            raise ValueError(f"{self} SDK profile changed from recorded {expected} to {self.stream_profiles[str(sensor_type)]}")
        if profile.get_fps() != self.fps:
            logger.warning("%s native %s is %s FPS; requested sampling is %s Hz and may repeat frames",
                           self, sensor_type, profile.get_fps(), self.fps)
        return profile

    def connect(self, warmup: bool = True) -> None:
        if self._connected:
            raise RuntimeError(f"{self} is already connected")
        from pyorbbecsdk import Config, OBSensorType, Pipeline

        device = self._find_device()
        self.pipeline = Pipeline(device)
        self.sdk_config = Config()
        color_profile = self._select_default_profile(OBSensorType.COLOR_SENSOR)
        self.sdk_config.enable_stream(color_profile)
        if self.use_depth:
            depth_profile = self._select_default_profile(OBSensorType.DEPTH_SENSOR)
            self.sdk_config.enable_stream(depth_profile)
        if self.use_depth and self.config.align_depth_to_color:
            from pyorbbecsdk import AlignFilter, OBStreamType
            self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.pipeline.start(self.sdk_config)
        self._connected = True
        try:
            if self.config.manual_mode:
                import pyorbbecsdk as sdk
                from .manual_controls import apply_orbbec_manual_controls
                self.effective_controls = apply_orbbec_manual_controls(
                    device, self.config.manual_controls, sdk
                )
                logger.info("%s fixed camera controls: %s", self, self.effective_controls)
        except Exception:
            self.disconnect()
            raise
        self._capture_error = None
        self._latest_color = self._latest_depth = None
        self._latest_timestamp = 0.0
        self._sequence = self._last_read_sequence = 0
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"orbbec-{self.config.serial_number}",
            daemon=True,
        )
        self._thread.start()
        try:
            if warmup:
                timeout = max(
                    self.config.frame_timeout_ms,
                    int(self.config.warmup_s * 1000) + 1000,
                )
                self.async_read(timeout_ms=timeout)
        except Exception:
            # A Pipeline owns native SDK threads. Always release them when
            # warm-up fails, otherwise interpreter shutdown can abort.
            self.disconnect()
            raise
        logger.info("%s connected using the Orbbec SDK", self)

    @staticmethod
    def _color_to_rgb(frame) -> NDArray[Any]:
        import cv2
        from pyorbbecsdk import OBFormat

        width, height, fmt = frame.get_width(), frame.get_height(), frame.get_format()
        data = np.asanyarray(frame.get_data())
        if fmt == OBFormat.RGB:
            return data.reshape(height, width, 3).copy()
        if fmt == OBFormat.BGR:
            return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        if fmt == OBFormat.MJPG:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("failed to decode Orbbec MJPG color frame")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if fmt == OBFormat.YUYV:
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUY2)
        if fmt == OBFormat.UYVY:
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_UYVY)
        raise RuntimeError(f"unsupported Orbbec color format: {fmt}")

    def _decode_frames(self, frames):
        import cv2

        if self._align_filter is not None:
            aligned = self._align_filter.process(frames)
            if aligned is None:
                return None
            frames = aligned.as_frame_set()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if self.use_depth else None
        if color_frame is None or (self.use_depth and depth_frame is None):
            return None
        color = self._color_to_rgb(color_frame)
        if self.config.color_mode == "bgr":
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        depth = None
        if depth_frame is not None:
            if self.config.align_depth_to_color and (depth_frame.get_height(), depth_frame.get_width()) != color.shape[:2]:
                raise RuntimeError("D2C alignment did not produce RGB-sized depth")
            units = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()
            )
            depth_mm = np.rint(units.astype(np.float32) * depth_frame.get_depth_scale())
            depth_mm[(depth_mm < 0) | (depth_mm > 65535)] = 0
            depth = depth_mm.astype(np.uint16)
            min_mm = round(self.config.min_depth_m * 1000)
            max_mm = round(self.config.max_depth_m * 1000)
            invalid = (depth > max_mm) | ((depth < min_mm) & (depth != 0))
            depth[invalid] = 0

            if self.config.crop_x_min is not None and color.shape[:2] != depth.shape:
                raise RuntimeError(
                    f"{self} color/depth sizes differ before aligned crop: "
                    f"{color.shape[:2]} vs {depth.shape}"
                )

            if self.config.crop_x_min is not None:
                native_h, native_w = depth.shape
                x0, x1 = self.config.crop_x_min, self.config.crop_x_max
                if x1 > native_w:
                    raise ValueError(
                        f"{self} horizontal crop [{x0}, {x1}) exceeds native width {native_w}"
                    )
                crop_w = x1 - x0
                if self.config.crop_y_min is not None:
                    y0, y1 = self.config.crop_y_min, self.config.crop_y_max
                    if y1 > native_h:
                        raise ValueError(
                            f"{self} vertical crop [{y0}, {y1}) exceeds native height {native_h}"
                        )
                else:
                    crop_h = round(crop_w * self.height / self.width)
                    if crop_h > native_h:
                        raise ValueError(
                            f"{self} aspect-preserving crop height {crop_h} exceeds native height {native_h}"
                        )
                    y0 = (native_h - crop_h) // 2
                    y1 = y0 + crop_h
                color = color[y0:y1, x0:x1]
                depth = depth[y0:y1, x0:x1]

            if color.shape[:2] != (self.height, self.width):
                color = cv2.resize(
                    color, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
            if depth.shape != (self.height, self.width):
                depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.uint16, copy=False)
            if self.config.map_depth_to_full_uint16:
                valid = depth > 0
                mapped = np.zeros(depth.shape, dtype=np.uint16)
                # Preserve zero as the invalid-depth sentinel. Orbbec metric
                # depth is integer millimetres, so the transform can be
                # inverted and rounded back to the original millimetre value.
                lo_mm = max(1, min_mm)
                span_mm = max_mm - lo_mm
                if span_mm <= 0:
                    raise ValueError("configured depth interval is too small to map")
                mapped[valid] = np.rint(
                    (depth[valid].astype(np.float32) - lo_mm)
                    / span_mm
                    * 65534.0
                    + 1.0
                ).astype(np.uint16)
                depth = mapped
        if color.shape[:2] != (self.height, self.width):
            color = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return color, depth

    def _capture_loop(self) -> None:
        try:
            ready_at = time.monotonic() + self.config.warmup_s
            while not self._stop_event.is_set():
                frames = self.pipeline.wait_for_frames(self.config.frame_timeout_ms)
                if not frames or time.monotonic() < ready_at:
                    continue
                pair = self._decode_frames(frames)
                if pair is None:
                    continue
                with self._condition:
                    self._latest_color, self._latest_depth = pair
                    self._latest_timestamp = time.monotonic()
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as exc:
            if not self._stop_event.is_set():
                with self._condition:
                    self._capture_error = repr(exc)
                logger.exception("%s capture thread stopped", self)
        finally:
            with self._condition:
                self._condition.notify_all()

    def read(self) -> NDArray[Any]:
        return self.async_read(self.config.frame_timeout_ms)

    def async_read(self, timeout_ms: float = 2000) -> NDArray[Any]:
        deadline = time.monotonic() + timeout_ms / 1000
        with self._condition:
            while self._latest_color is None or self._sequence == self._last_read_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for a synchronized frame from {self}")
                self._condition.wait(remaining)
            if self._latest_color is None:
                raise RuntimeError(f"{self} has not captured a color frame")
            self._last_read_sequence = self._sequence
            return self._latest_color.copy()

    def read_latest_rgbd_with_metadata(
        self,
        max_age_ms: int = 2000,
        *,
        newer_than_sequence: int | None = None,
        timeout_ms: float | None = None,
    ):
        """Return the latest RGB-D pair with a local capture time and sequence.

        The monotonic timestamp is recorded when the SDK frameset has been
        decoded by the capture thread.  It deliberately uses the host clock so
        observations from multiple cameras can be compared with locally
        received ROS joint-state samples without assuming synchronized device
        clocks.
        """
        deadline = time.monotonic() + (
            self.config.frame_timeout_ms if timeout_ms is None else timeout_ms
        ) / 1000.0
        with self._condition:
            while True:
                if self._capture_error is not None:
                    raise CameraFault(self, "capture_thread_failed", self._capture_error)
                complete = self._latest_color is not None and (
                    not self.use_depth or self._latest_depth is not None
                )
                sequence_is_new = (
                    newer_than_sequence is None or self._sequence > newer_than_sequence
                )
                if complete and sequence_is_new:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = (
                        "No complete RGB-D frame"
                        if not complete
                        else f"sequence {self._sequence} is not newer than {newer_than_sequence}"
                    )
                    raise CameraFault(self, "missing_rgbd", detail)
                self._condition.wait(remaining)

            age_ms = (time.monotonic() - self._latest_timestamp) * 1000
            # Older deployed camera configs predate ``max_frame_age_ms``.
            # The caller-provided limit remains authoritative in that case.
            configured_max_age_ms = getattr(
                self.config, "max_frame_age_ms", max_age_ms
            )
            if age_ms > min(max_age_ms, configured_max_age_ms):
                raise CameraFault(self, "stale_rgbd", f"Latest RGB-D frame is {age_ms:.1f} ms old")
            color = self._latest_color.copy()
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            metadata = {
                "sequence": self._sequence,
                "captured_at_monotonic_s": self._latest_timestamp,
                "age_ms": max(0.0, age_ms),
            }
            return color, depth, metadata

    def read_latest_rgbd(self, max_age_ms: int = 2000):
        color, depth, _metadata = self.read_latest_rgbd_with_metadata(max_age_ms)
        return color, depth

    def read_latest(self, max_age_ms: int = 2000) -> NDArray[Any]:
        return self.read_latest_rgbd(max_age_ms)[0]

    def read_depth(self) -> NDArray[Any]:
        return self.read_latest_depth(self.config.frame_timeout_ms)

    def read_latest_depth(self, max_age_ms: int = 2000) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"Depth is disabled for {self}")
        depth = self.read_latest_rgbd(max_age_ms)[1]
        if depth is None:
            raise RuntimeError(f"{self} has not captured a depth frame")
        return depth

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._stop_event.set()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                logger.exception("Failed to stop %s cleanly", self)
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"{self} capture thread did not stop; refusing overlapping reconnect")
        self._thread = None
        self.pipeline = None
        self.sdk_config = None
        self._align_filter = None
        self.effective_controls = None
        self.stream_profiles = {}
        self._connected = False
        logger.info("%s disconnected", self)
