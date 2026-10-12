# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass, field

import draccus


@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(0, 30, 640, 480)
    OpenCVCameraConfig(0, 60, 640, 480)
    OpenCVCameraConfig(0, 90, 640, 480)
    OpenCVCameraConfig(0, 30, 1280, 720)
    ```
    """

    camera_index: int | str = 0
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 60, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 90, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 30, 1280, 720)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, rotation=90)
    ```
    """

    name: str | None = None
    serial_number: int | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    # Fixed photometric controls; exported SDK values can be reused at inference.
    manual_mode: bool = True
    manual_controls: dict[str, float] = field(default_factory=dict)
    force_hardware_reset: bool = True
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        # bool is stronger than is None, since it works with empty strings
        if bool(self.name) and bool(self.serial_number):
            raise ValueError(
                f"One of them must be set: name or serial_number, but {self.name=} and {self.serial_number=} provided."
            )

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("zmq")
@dataclass
class ZMQCameraConfig(CameraConfig):
    """Configuration for a camera that streams data over ZMQ."""

    host: str = "localhost"
    port: int = 5556
    topic_type: str = "RGB"  # "RGB" or "Depth"
    rotation: int | None = None


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Orbbec RGB-D camera configuration using the official Orbbec SDK.

    ``width``/``height`` describe the arrays returned to LeRobot. The backend
    uses the device default stream profiles, resizing RGB normally and depth with
    nearest-neighbour interpolation.
    """

    serial_number: str
    fps: int = 30
    width: int = 640
    height: int = 480
    channels: int = 3
    color_mode: str = "rgb"
    align_depth_to_color: bool = False
    expected_stream_profiles: dict[str, dict] = field(default_factory=dict)
    use_depth: bool = True
    # Fixed photometric controls; exported SDK values can be reused at inference.
    manual_mode: bool = True
    manual_controls: dict[str, float] = field(default_factory=dict)
    warmup_s: float = 2.0
    frame_timeout_ms: int = 2000
    max_frame_age_ms: int = 300
    reconnect_attempts: int = 3
    min_depth_m: float = 0.0
    max_depth_m: float = 65.535
    # Reversibly map the configured millimetre interval to 1..65535 for
    # full-range uint16 PNG storage. Zero remains reserved for invalid depth.
    map_depth_to_full_uint16: bool = False
    # Optional horizontal crop in native-frame pixel coordinates. The vertical
    # crop is centered automatically to match the configured output aspect ratio.
    crop_x_min: int | None = None
    crop_x_max: int | None = None
    crop_y_min: int | None = None
    crop_y_max: int | None = None

    def __post_init__(self) -> None:
        if self.color_mode not in ("rgb", "bgr"):
            raise ValueError("color_mode must be rgb or bgr")
        if not self.serial_number:
            raise ValueError("serial_number must not be empty")
        if self.fps is None or self.width is None or self.height is None:
            raise ValueError("fps, width and height must all be specified")
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("fps, width and height must be positive")
        if self.warmup_s < 0 or self.frame_timeout_ms <= 0:
            raise ValueError("warmup_s must be non-negative and frame_timeout_ms must be positive")
        if self.max_frame_age_ms <= 0 or not 0 <= self.reconnect_attempts <= 3:
            raise ValueError("max_frame_age_ms must be positive; reconnect_attempts must be 0..3")
        if not 0 <= self.min_depth_m < self.max_depth_m <= 65.535:
            raise ValueError("depth range must satisfy 0 <= min_depth_m < max_depth_m <= 65.535")
        if (self.crop_x_min is None) != (self.crop_x_max is None):
            raise ValueError("crop_x_min and crop_x_max must be specified together")
        if self.crop_x_min is not None and not (0 <= self.crop_x_min < self.crop_x_max):
            raise ValueError("horizontal crop must satisfy 0 <= crop_x_min < crop_x_max")
        if (self.crop_y_min is None) != (self.crop_y_max is None):
            raise ValueError("crop_y_min and crop_y_max must be specified together")
        if self.crop_y_min is not None and not (0 <= self.crop_y_min < self.crop_y_max):
            raise ValueError("vertical crop must satisfy 0 <= crop_y_min < crop_y_max")
        if self.crop_y_min is not None and self.crop_x_min is None:
            raise ValueError("an explicit vertical crop requires a horizontal crop")
        if self.crop_y_min is not None:
            crop_w = self.crop_x_max - self.crop_x_min
            crop_h = self.crop_y_max - self.crop_y_min
            if crop_w * self.height != crop_h * self.width:
                raise ValueError("crop and output aspect ratios must be identical")
