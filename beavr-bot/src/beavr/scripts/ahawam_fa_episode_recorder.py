"""Crash-safe recorder for one FA AHA-WAM real-robot episode.

This module is deliberately independent of ROS and camera drivers.  The FA
client supplies immutable RGB-D/state snapshots while this recorder writes
three H.264 RGB streams, three lossless Blosc-Zstd depth streams, and JSONL metadata
on a bounded background queue.

FA's historical LeRobot contract is important here: a training action is the
*next measured state*, not the command sent to ``/min_snap/target``.  Raw frame
rows therefore store ``state`` and ``sent_command`` as separate fields.  An
exporter must construct ``action[t] = state[t + 1]``; the last frame has no
training transition.  A sent command is stored only after the caller reports
that its ROS publish returned successfully.

Episodes remain below ``.in_progress/<uuid>`` until every JSONL/video artifact
has been closed and validated.  Final publication is one atomic directory
rename to ``episodes/<uuid>`` with a ``COMMITTED`` marker already inside it.
Any validation, queue, disk, or encoder failure is fail-closed and leaves the
staging directory available for diagnosis.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


SCHEMA_VERSION = 1
CAMERA_KEYS = ("head", "wrist_CV2756100033", "wrist_CV2856D0001C")
CAMERA_SHAPES = {
    "head": (720, 1280, 3),
    "wrist_CV2756100033": (530, 848, 3),
    "wrist_CV2856D0001C": (530, 848, 3),
}
DEPTH_SHAPES = {key: shape[:2] for key, shape in CAMERA_SHAPES.items()}
LEFT_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
RIGHT_JOINT_NAMES = tuple(name.replace("left_", "right_") for name in LEFT_JOINT_NAMES)
STATE_NAMES = (
    LEFT_JOINT_NAMES
    + RIGHT_JOINT_NAMES
    + (
        "left_gripper_latch",
        "right_gripper_latch",
    )
)
STATE_DIM = 16
VALID_COMMAND_SOURCES = frozenset({"model", "human", "hold"})
VALID_OUTCOMES = frozenset({"success", "failure", "aborted", "error", "unlabeled"})
WORKER_STOP_TIMEOUT_SEC = 30.0
_STOP = object()


class RecorderError(RuntimeError):
    """Base class for FA episode recorder errors."""


class RecorderBusyError(RecorderError):
    """Another process owns the dataset writer lock."""


class RecorderValidationError(RecorderError, ValueError):
    """Input cannot satisfy the on-disk FA recording contract."""


class RecorderFailedError(RecorderError):
    """The fail-closed recorder can no longer accept data."""


@dataclass(frozen=True)
class SentCommandSnapshot:
    """A 16D command whose ROS publish returned successfully.

    ``receipt`` describes what the caller considers successful delivery.  ROS
    publishers normally cannot prove hardware execution, so callers should use
    explicit wording such as ``{"transport": "ros_publish_returned"}`` rather
    than claiming actuator acknowledgement.
    """

    sent_command: tuple[float, ...]
    source: str
    sequence: int
    sent_at_monotonic_s: float
    sent_at_wall_s: float
    chunk: Optional[int]
    step: Optional[int]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class _FramePacket:
    frame_index: int
    sequence: int
    captured_at: float
    images: dict[str, np.ndarray]
    depths: dict[str, np.ndarray]
    state: tuple[float, ...]
    command: Optional[SentCommandSnapshot]
    camera_metadata: dict[str, Any]
    state_metadata: Optional[dict[str, Any]]
    sync_metadata: Optional[dict[str, Any]]
    intervention: Optional[dict[str, Any]]


@dataclass(frozen=True)
class _EventPacket:
    event_index: int
    event_type: str
    recorded_at: float
    payload: dict[str, Any]


class _PyAVH264Encoder:
    """Streaming H.264 encoder; PyAV is imported only on construction."""

    def __init__(
        self,
        *,
        path: Path,
        fps: int,
        width: int,
        height: int,
        crf: int,
        gop: int,
    ) -> None:
        try:
            import av  # type: ignore
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RecorderError("PyAV is required for H.264 episode recording") from exc
        self._av = av
        self._container = av.open(str(path), mode="w", format="mp4")
        try:
            self._stream = self._container.add_stream(
                "libx264",
                rate=fps,
                options={
                    "crf": str(crf),
                    "g": str(gop),
                    "preset": "ultrafast",
                    "tune": "zerolatency",
                    "bf": "0",
                },
            )
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = "yuv420p"
            self._stream.codec_context.gop_size = gop
            # Three auto-threaded x264 encoders starved the Orbbec capture
            # threads on the deployed ARM host.  One worker thread per stream
            # keeps total CPU concurrency bounded; the recorder queue absorbs
            # short encoding jitter without dropping observations.
            self._stream.codec_context.thread_count = 1
        except Exception:
            self._container.close()
            raise
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RecorderError("cannot write to a closed video encoder")
        video_frame = self._av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._closed = True


def _default_encoder_factory(**kwargs: Any) -> Any:
    if _jetson_h264_available():
        return _GStreamerJetsonH264Encoder(**kwargs)
    return _PyAVH264Encoder(**kwargs)


_JETSON_H264_AVAILABLE: Optional[bool] = None


def _jetson_h264_available() -> bool:
    global _JETSON_H264_AVAILABLE
    if _JETSON_H264_AVAILABLE is not None:
        return _JETSON_H264_AVAILABLE
    executable = shutil.which("gst-inspect-1.0")
    if executable is None:
        _JETSON_H264_AVAILABLE = False
        return False
    try:
        result = subprocess.run(
            [executable, "nvv4l2h264enc"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
            check=False,
        )
        _JETSON_H264_AVAILABLE = result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        _JETSON_H264_AVAILABLE = False
    return _JETSON_H264_AVAILABLE


class _GStreamerJetsonH264Encoder:
    """Jetson NVENC H.264 writer fed with RGB24 frames over stdin."""

    def __init__(
        self,
        *,
        path: Path,
        fps: int,
        width: int,
        height: int,
        crf: int,
        gop: int,
    ) -> None:
        del crf  # NVENC uses a high fixed bitrate instead of software CRF.
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.expected_shape = (self.height, self.width, 3)
        self._closed = False
        gst = shutil.which("gst-launch-1.0")
        if gst is None:
            raise RecorderError("gst-launch-1.0 disappeared after encoder probe")
        command = [
            gst,
            "-q",
            "fdsrc",
            "fd=0",
            "!",
            "rawvideoparse",
            "format=rgb",
            f"width={self.width}",
            f"height={self.height}",
            f"framerate={int(fps)}/1",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=RGBA",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            "maxperf-enable=1",
            "bitrate=8000000",
            f"iframeinterval={int(gop)}",
            "insert-sps-pps=1",
            "!",
            "h264parse",
            "!",
            "qtmux",
            "!",
            "filesink",
            f"location={self.path}",
        ]
        nice = shutil.which("nice")
        if nice is not None:
            command = [nice, "-n", "5", *command]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RecorderError(f"could not start Jetson H.264 encoder: {exc}") from exc

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RecorderError("cannot write to a closed Jetson encoder")
        if frame.dtype != np.uint8 or frame.shape != self.expected_shape:
            raise RecorderValidationError(
                f"Jetson RGB frame expected {self.expected_shape}/uint8, "
                f"got {frame.shape}/{frame.dtype}"
            )
        if self._process.stdin is None:
            raise RecorderError("Jetson encoder stdin is unavailable")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RecorderError(
                f"Jetson H.264 encoder stopped with code {self._process.poll()}"
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            return_code = self._process.wait(timeout=30.0)
        except subprocess.TimeoutExpired as exc:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3.0)
            raise RecorderError("Jetson H.264 encoder did not stop") from exc
        if return_code != 0:
            raise RecorderError(
                f"Jetson H.264 encoder exited with code {return_code}"
            )


_DEPTH_STREAM_MAGIC = b"AHA-DEPTH-BLOSC-V1\n"


class _BloscZstdDepthEncoder:
    """Framed, lossless uint16-mm depth with fast Blosc-Zstd compression."""

    def __init__(self, *, path: Path, fps: int, width: int, height: int) -> None:
        try:
            import blosc  # type: ignore
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RecorderError("python-blosc is required for depth recording") from exc
        # The recorder already has one worker per camera.  Prevent Blosc from
        # creating another large thread pool that could starve camera capture.
        blosc.set_nthreads(1)
        self._blosc = blosc
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.expected_shape = (self.height, self.width)
        self._file = self.path.open("xb")
        header = json.dumps(
            {
                "dtype": "<u2",
                "shape": [self.height, self.width],
                "fps": int(fps),
                "compressor": "blosc-zstd",
                "clevel": 1,
                "shuffle": "bitshuffle",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._file.write(_DEPTH_STREAM_MAGIC)
        self._file.write(struct.pack("<I", len(header)))
        self._file.write(header)
        self._closed = False

    def write(self, depth: np.ndarray) -> None:
        if self._closed:
            raise RecorderError("cannot write to a closed depth encoder")
        if depth.dtype != np.uint16 or depth.shape != self.expected_shape:
            raise RecorderValidationError(
                f"depth frame expected {self.expected_shape}/uint16, "
                f"got {depth.shape}/{depth.dtype}"
            )
        payload = self._blosc.compress(
            np.ascontiguousarray(depth),
            typesize=2,
            clevel=1,
            cname="zstd",
            shuffle=self._blosc.BITSHUFFLE,
        )
        self._file.write(struct.pack("<I", len(payload)))
        self._file.write(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._closed = True


def _default_depth_encoder_factory(**kwargs: Any) -> _BloscZstdDepthEncoder:
    return _BloscZstdDepthEncoder(**kwargs)


def _lower_encoder_thread_priority() -> None:
    """Keep camera capture/observation threads ahead of background encoders."""

    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 5)
    except (AttributeError, OSError):
        # Non-Linux test hosts may not expose per-thread setpriority.
        pass


class _MultiCameraSegmentWriter:
    """Write synchronized RGB H.264 and lossless depth Blosc-Zstd segments."""

    def __init__(
        self,
        staging_path: Path,
        *,
        fps: int,
        segment_frames: int,
        encoder_factory: Callable[..., Any],
        depth_encoder_factory: Callable[..., Any],
    ) -> None:
        self.staging_path = staging_path
        self.fps = fps
        self.segment_frames = segment_frames
        self.encoder_factory = encoder_factory
        self.depth_encoder_factory = depth_encoder_factory
        self._encode_executor = ThreadPoolExecutor(
            max_workers=len(CAMERA_KEYS) * 2,
            thread_name_prefix="fa-rgbd-encode",
            initializer=_lower_encoder_thread_priority,
        )
        self.camera_dirs: dict[str, Path] = {}
        self.depth_dirs: dict[str, Path] = {}
        videos_dir = staging_path / "videos"
        videos_dir.mkdir(exist_ok=False)
        for key in CAMERA_KEYS:
            path = videos_dir / key
            path.mkdir(exist_ok=False)
            self.camera_dirs[key] = path
            _fsync_directory(path)
            depth_path = videos_dir / f"{key}_depth"
            depth_path.mkdir(exist_ok=False)
            self.depth_dirs[key] = depth_path
            _fsync_directory(depth_path)
        _fsync_directory(videos_dir)
        _fsync_directory(staging_path)
        self._encoders: dict[str, Any] = {}
        self._depth_encoders: dict[str, Any] = {}
        self._partials: dict[str, Path] = {}
        self._depth_partials: dict[str, Path] = {}
        self._segment_index = -1
        self._segment_frame_count = 0
        self._segment_start_frame = 0
        self.segments: list[dict[str, Any]] = []

    def _open_segment(self, frame_index: int) -> None:
        self._segment_index += 1
        self._segment_frame_count = 0
        self._segment_start_frame = frame_index
        stem = f"segment-{self._segment_index:06d}"
        try:
            for key in CAMERA_KEYS:
                shape = CAMERA_SHAPES[key]
                partial = self.camera_dirs[key] / f"{stem}.mp4.partial"
                encoder = self.encoder_factory(
                    path=partial,
                    fps=self.fps,
                    width=shape[1],
                    height=shape[0],
                    crf=30,
                    gop=self.fps,
                )
                if not callable(getattr(encoder, "write", None)) or not callable(
                    getattr(encoder, "close", None)
                ):
                    raise RecorderValidationError(
                        f"encoder_factory returned invalid encoder for {key}"
                    )
                self._partials[key] = partial
                self._encoders[key] = encoder
                depth_shape = DEPTH_SHAPES[key]
                depth_partial = self.depth_dirs[key] / f"{stem}.depth.bl2.partial"
                depth_encoder = self.depth_encoder_factory(
                    path=depth_partial,
                    fps=self.fps,
                    width=depth_shape[1],
                    height=depth_shape[0],
                )
                if not callable(getattr(depth_encoder, "write", None)) or not callable(
                    getattr(depth_encoder, "close", None)
                ):
                    raise RecorderValidationError(
                        f"depth_encoder_factory returned invalid encoder for {key}"
                    )
                self._depth_partials[key] = depth_partial
                self._depth_encoders[key] = depth_encoder
        except Exception:
            self.abort()
            raise

    def write(
        self, packet: _FramePacket
    ) -> tuple[int, int, dict[str, str], dict[str, str]]:
        if not self._encoders:
            self._open_segment(packet.frame_index)
        elif self._segment_frame_count >= self.segment_frames:
            self._finalize_segment()
            self._open_segment(packet.frame_index)
        futures = []
        for key in CAMERA_KEYS:
            futures.append(
                self._encode_executor.submit(
                    self._encoders[key].write, packet.images[key]
                )
            )
            futures.append(
                self._encode_executor.submit(
                    self._depth_encoders[key].write, packet.depths[key]
                )
            )
        for future in futures:
            future.result()
        segment_frame = self._segment_frame_count
        self._segment_frame_count += 1
        rgb_filename = f"segment-{self._segment_index:06d}.mp4"
        depth_filename = f"segment-{self._segment_index:06d}.depth.bl2"
        paths = {key: f"videos/{key}/{rgb_filename}" for key in CAMERA_KEYS}
        depth_paths = {
            key: f"videos/{key}_depth/{depth_filename}" for key in CAMERA_KEYS
        }
        return self._segment_index, segment_frame, paths, depth_paths

    def _finalize_segment(self) -> None:
        if not self._encoders:
            return
        encoders, partials = self._encoders, self._partials
        depth_encoders, depth_partials = self._depth_encoders, self._depth_partials
        self._encoders, self._partials = {}, {}
        self._depth_encoders, self._depth_partials = {}, {}
        try:
            for key in CAMERA_KEYS:
                encoders[key].close()
                depth_encoders[key].close()
            final_paths: dict[str, str] = {}
            final_depth_paths: dict[str, str] = {}
            for key in CAMERA_KEYS:
                partial = partials[key]
                if not partial.is_file() or partial.stat().st_size <= 0:
                    raise RecorderError(f"video encoder produced no data: {partial}")
                _fsync_file(partial)
                final = Path(str(partial).removesuffix(".partial"))
                os.replace(partial, final)
                _fsync_directory(self.camera_dirs[key])
                final_paths[key] = str(final.relative_to(self.staging_path))
                depth_partial = depth_partials[key]
                if not depth_partial.is_file() or depth_partial.stat().st_size <= 0:
                    raise RecorderError(
                        f"depth encoder produced no data: {depth_partial}"
                    )
                _fsync_file(depth_partial)
                depth_final = Path(str(depth_partial).removesuffix(".partial"))
                os.replace(depth_partial, depth_final)
                _fsync_directory(self.depth_dirs[key])
                final_depth_paths[key] = str(
                    depth_final.relative_to(self.staging_path)
                )
            self.segments.append(
                {
                    "segment_index": self._segment_index,
                    "start_frame": self._segment_start_frame,
                    "frame_count": self._segment_frame_count,
                    "videos": final_paths,
                    "depth_videos": final_depth_paths,
                }
            )
        except Exception:
            for encoder in (*encoders.values(), *depth_encoders.values()):
                try:
                    encoder.close()
                except Exception:
                    pass
            raise

    def close(self) -> list[dict[str, Any]]:
        try:
            self._finalize_segment()
            return copy.deepcopy(self.segments)
        finally:
            self._encode_executor.shutdown(wait=True, cancel_futures=True)

    def abort(self) -> None:
        for encoder in (*self._encoders.values(), *self._depth_encoders.values()):
            try:
                encoder.close()
            except Exception:
                pass
        self._encoders = {}
        self._depth_encoders = {}
        self._partials = {}
        self._depth_partials = {}
        self._encode_executor.shutdown(wait=True, cancel_futures=True)


class FaEpisodeRecorder:
    """Transactionally record one FA AHA-WAM episode.

    ``encoder_factory`` writes RGB and receives ``path``, ``fps``, ``width``,
    ``height``, ``crf`` and ``gop``. ``depth_encoder_factory`` writes HxW
    uint16-mm depth and receives ``path``, ``fps``, ``width`` and ``height``.
    Injection keeps tests independent from PyAV and real hardware.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        fps: int = 30,
        segment_sec: float = 10.0,
        queue_size: int = 128,
        min_free_gb: float = 5.0,
        config: Optional[Mapping[str, Any]] = None,
        encoder_factory: Optional[Callable[..., Any]] = None,
        depth_encoder_factory: Optional[Callable[..., Any]] = None,
        validate_timing: bool = True,
        max_relative_fps_error: float = 0.25,
        max_frame_gap_factor: float = 3.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.fps = _validate_positive_int(fps, "fps")
        self.segment_sec = _validate_positive_finite(segment_sec, "segment_sec")
        self.segment_frames = max(1, round(self.fps * self.segment_sec))
        self.queue_size = _validate_positive_int(queue_size, "queue_size")
        self.min_free_gb = _validate_nonnegative_finite(min_free_gb, "min_free_gb")
        self.min_free_bytes = int(self.min_free_gb * 1024**3)
        if not isinstance(validate_timing, bool):
            raise RecorderValidationError("validate_timing must be bool")
        self.validate_timing = validate_timing
        self.max_relative_fps_error = _validate_nonnegative_finite(
            max_relative_fps_error, "max_relative_fps_error"
        )
        self.max_frame_gap_factor = _validate_positive_finite(
            max_frame_gap_factor, "max_frame_gap_factor"
        )
        self.config = _json_compatible({} if config is None else config, "config")
        self._uses_default_encoder = encoder_factory is None
        self._uses_default_depth_encoder = depth_encoder_factory is None
        self.rgb_encoder_backend = (
            "gstreamer_nvv4l2h264enc"
            if encoder_factory is None and _jetson_h264_available()
            else "pyav_libx264"
            if encoder_factory is None
            else "injected"
        )
        self.encoder_factory = encoder_factory or _default_encoder_factory
        self.depth_encoder_factory = (
            depth_encoder_factory or _default_depth_encoder_factory
        )

        self._lock_fd: Optional[int] = None
        self._lifecycle_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._failure: Optional[BaseException] = None
        self._failed_event = threading.Event()
        self._ready = threading.Event()
        self._accepting = False
        self._state = "initializing"
        self._submitted_frames = 0
        self._submitted_events = 0
        self._written_frames = 0
        self._written_events = 0
        self._last_sequence: Optional[int] = None
        self._last_captured_at: Optional[float] = None
        self._saw_task_command = False
        self._complete_task_command_count = 0
        self._partial_task_command_count = 0
        self._task_command_sources: set[str] = set()
        self._task_output_may_have_changed = False
        self._worker_summary: Optional[dict[str, Any]] = None

        self.root.mkdir(parents=True, exist_ok=True)
        self._acquire_writer_lock()
        try:
            _ensure_free_space(self.root, self.min_free_bytes)
            (self.root / ".in_progress").mkdir(exist_ok=True)
            (self.root / "episodes").mkdir(exist_ok=True)
            self.episode_uuid = str(uuid.uuid4())
            self.staging_path = self.root / ".in_progress" / self.episode_uuid
            self.final_path = self.root / "episodes" / self.episode_uuid
            self.staging_path.mkdir(exist_ok=False)
            created_at = _utc_now()
            self._initial_metadata = {
                "schema_version": SCHEMA_VERSION,
                "recorder": "AHA-WAM-FA-real-episode",
                "episode_uuid": self.episode_uuid,
                "status": "recording",
                "created_at_utc": created_at,
                "fps": self.fps,
                "segment_sec": self.segment_sec,
                "segment_frames": self.segment_frames,
                "camera_keys": list(CAMERA_KEYS),
                "camera_shapes": {key: list(CAMERA_SHAPES[key]) for key in CAMERA_KEYS},
                "camera_dtype": "uint8",
                "depth_camera_keys": list(CAMERA_KEYS),
                "depth_shapes": {key: list(DEPTH_SHAPES[key]) for key in CAMERA_KEYS},
                "depth_dtype": "uint16",
                "depth_unit": "millimetres",
                "depth_valid_range_mm": [0, 1000],
                "state_shape": [STATE_DIM],
                "state_dtype": "float32",
                "state_names": list(STATE_NAMES),
                "state_semantics": {
                    "indices_0_13": "measured /joint_states arm positions in radians",
                    "indices_14_15": "left/right binary gripper command latch",
                },
                "sent_command_shape": [STATE_DIM],
                "sent_command_dtype": "float32",
                "sent_command_sources": sorted(VALID_COMMAND_SOURCES),
                "sent_command_nullable_before_first_send": True,
                "training_export_contract": {
                    "observation_state": "frames[t].state",
                    "action": "frames[t+1].state",
                    "action_shape": [STATE_DIM],
                    "last_frame_has_training_transition": False,
                    "sent_command_is_not_training_action": True,
                },
                "camera_sampling_contract": {
                    "latest_frame_reuse_allowed": True,
                    "fresh_sequence_per_recorder_frame_required": False,
                    "per_camera_sequence_and_timestamp_retained": True,
                },
                "motion_evidence_contract": {
                    "motion_executed_true_requires_hardware_execution_ack": True,
                    "current_client_has_hardware_execution_ack": False,
                    "motion_executed_false": (
                        "motion disabled or no model/human task output was published"
                    ),
                    "motion_executed_null": (
                        "task output may have been published, but physical execution "
                        "is not verified"
                    ),
                    "command_publication_is_not_execution_confirmation": True,
                },
                "video_codec": {
                    "codec": "h264",
                    "backend": self.rgb_encoder_backend,
                    "pixel_format": "yuv420p",
                    "quality": (
                        {"bitrate": 8000000}
                        if self.rgb_encoder_backend == "gstreamer_nvv4l2h264enc"
                        else {"crf": 30}
                    ),
                    "gop": self.fps,
                    "preset": (
                        "maxperf"
                        if self.rgb_encoder_backend == "gstreamer_nvv4l2h264enc"
                        else "ultrafast"
                    ),
                    "tune": (
                        None
                        if self.rgb_encoder_backend == "gstreamer_nvv4l2h264enc"
                        else "zerolatency"
                    ),
                    "encoder_threads_per_stream": 1,
                },
                "depth_video_codec": {
                    "container": "aha_framed_depth_v1",
                    "codec": "blosc-zstd",
                    "pixel_format": "uint16le",
                    "encoding": "uint16_mm_blosc_zstd_v1",
                    "compression_level": 1,
                    "shuffle": "bitshuffle",
                    "encoder_threads_per_stream": 1,
                },
                "video_validation": (
                    "full_rgb_and_exact_depth_decode"
                    if self._uses_default_encoder
                    and self._uses_default_depth_encoder
                    else "injected_encoder_contract"
                ),
                "motion_enabled": self.config.get("enable_motion"),
                "config": self.config,
                "config_sha256": hashlib.sha256(
                    _canonical_json(self.config).encode("utf-8")
                ).hexdigest(),
            }
            _atomic_write_json(
                self.staging_path / "metadata.json", self._initial_metadata
            )
            self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.queue_size)
            self._worker = threading.Thread(
                target=self._writer_main,
                name=f"fa-episode-writer-{self.episode_uuid[:8]}",
                daemon=True,
            )
            self._worker.start()
            self._ready.wait()
            self.raise_if_failed()
            self._accepting = True
            self._state = "recording"
        except Exception:
            worker = getattr(self, "_worker", None)
            if worker is not None and worker.is_alive():
                self._failed_event.set()
                worker.join(timeout=WORKER_STOP_TIMEOUT_SEC)
            if worker is None or not worker.is_alive():
                self._release_writer_lock()
            raise

    @property
    def frame_count(self) -> int:
        with self._lifecycle_lock:
            return self._submitted_frames

    def submit_observation(
        self,
        *,
        images: Mapping[str, np.ndarray],
        depths: Optional[Mapping[str, np.ndarray]] = None,
        state: np.ndarray,
        camera_metadata: Mapping[str, Any],
        state_metadata: Optional[Mapping[str, Any]],
        sync_metadata: Optional[Mapping[str, Any]],
        sequence: int,
        captured_at: float,
        sent_command: Optional[Any],
        intervention_snapshot: Optional[Any] = None,
    ) -> int:
        """Explicit client-facing wrapper around :meth:`submit_frame`."""

        return self.submit_frame(
            {
                "images": images,
                "depths": depths,
                "state": state,
                "camera_metadata": camera_metadata,
                "state_metadata": state_metadata,
                "sync_metadata": sync_metadata,
                "sequence": sequence,
                "captured_at": captured_at,
            },
            sent_command,
            intervention_snapshot,
        )

    def submit_frame(
        self,
        snapshot: Any,
        command_snapshot: Optional[Any],
        intervention_snapshot: Optional[Any] = None,
    ) -> int:
        """Queue one RGB-D/state observation and the latest successful command.

        Camera sequences may repeat because FA wrist cameras can run at 10 Hz
        while the recorder runs at 30 Hz.  Only the recorder's own ``sequence``
        and ``captured_at`` must increase strictly.
        """

        try:
            raw_images = _snapshot_field(snapshot, "images")
            images = _validate_images(raw_images)
            raw_depths = _snapshot_field(snapshot, "depths")
            depths = _validate_depths(raw_depths)
            state = _validate_vector(_snapshot_field(snapshot, "state"), "state")
            sequence = _validate_nonnegative_int(
                _snapshot_field(snapshot, "sequence"), "sequence"
            )
            captured_at = _validate_timestamp(
                _snapshot_field(snapshot, "captured_at"), "captured_at"
            )
            command = _normalize_command_snapshot(command_snapshot)
            if command is not None and command.sent_at_monotonic_s > captured_at:
                raise RecorderValidationError(
                    "sent command is newer than the observation: "
                    f"command={command.sent_at_monotonic_s}, observation={captured_at}"
                )
            camera_metadata = _validate_camera_metadata(
                _snapshot_field(snapshot, "camera_metadata")
            )
            raw_state_metadata = _optional_snapshot_field(snapshot, "state_metadata")
            state_metadata = (
                None
                if raw_state_metadata is None
                else _structured_json(raw_state_metadata, "state_metadata")
            )
            raw_sync_metadata = _optional_snapshot_field(snapshot, "sync_metadata")
            sync_metadata = (
                None
                if raw_sync_metadata is None
                else _structured_json(raw_sync_metadata, "sync_metadata")
            )
            intervention = (
                None
                if intervention_snapshot is None
                else _structured_json(intervention_snapshot, "intervention_snapshot")
            )
        except Exception as exc:
            self._set_failure(exc)
            raise

        with self._lifecycle_lock:
            self._require_accepting()
            if self._last_sequence is not None and sequence <= self._last_sequence:
                self._fail_and_raise(
                    RecorderValidationError(
                        "sequence must increase strictly: "
                        f"previous={self._last_sequence}, current={sequence}"
                    )
                )
            if (
                self._last_captured_at is not None
                and captured_at <= self._last_captured_at
            ):
                self._fail_and_raise(
                    RecorderValidationError(
                        "captured_at must increase strictly: "
                        f"previous={self._last_captured_at}, current={captured_at}"
                    )
                )
            frame_index = self._submitted_frames
            packet = _FramePacket(
                frame_index=frame_index,
                sequence=sequence,
                captured_at=captured_at,
                images={key: images[key].copy() for key in CAMERA_KEYS},
                depths={key: depths[key].copy() for key in CAMERA_KEYS},
                state=tuple(float(value) for value in state),
                command=command,
                camera_metadata=camera_metadata,
                state_metadata=state_metadata,
                sync_metadata=sync_metadata,
                intervention=intervention,
            )
            self._put_nowait(packet)
            self._submitted_frames += 1
            if command is not None and command.source in {"model", "human"}:
                self._saw_task_command = True
                self._task_command_sources.add(command.source)
                self._task_output_may_have_changed = True
            self._last_sequence = sequence
            self._last_captured_at = captured_at
            return frame_index

    def record_event(self, event_type: str, **payload: Any) -> int:
        """Queue one diagnostic/intervention event in recorder order."""

        try:
            clean_type = _validate_nonempty_string(event_type, "event_type")
            clean_payload = _structured_json(payload, "event payload")
        except Exception as exc:
            self._set_failure(exc)
            raise
        with self._lifecycle_lock:
            self._require_accepting()
            event_index = self._submitted_events
            self._put_nowait(
                _EventPacket(
                    event_index=event_index,
                    event_type=clean_type,
                    recorded_at=time.time(),
                    payload=clean_payload,
                )
            )
            self._submitted_events += 1
            return event_index

    def note_successful_command(self, source: str) -> None:
        """Synchronously retain evidence of accepted task motion."""

        clean_source = _validate_command_source(source)
        if clean_source in {"model", "human"}:
            with self._lifecycle_lock:
                self._saw_task_command = True
                self._complete_task_command_count += 1
                self._task_command_sources.add(clean_source)
                # Every complete FA command includes an arm-topic publish.
                # ROS publish return is evidence of command publication only,
                # not acknowledgement that the physical trajectory executed.
                self._task_output_may_have_changed = True

    def note_partial_command(self, receipt: Mapping[str, Any]) -> None:
        """Retain conservative motion evidence from a partial ROS write."""

        clean_receipt = _structured_json(receipt, "partial command receipt")
        source = _validate_command_source(clean_receipt.get("source"))
        for name in (
            "arm_published",
            "left_hand_published",
            "right_hand_published",
            "left_hand_changed",
            "right_hand_changed",
            "complete",
        ):
            if not isinstance(clean_receipt.get(name), bool):
                raise RecorderValidationError(
                    f"partial command receipt.{name} must be boolean"
                )
        if clean_receipt["complete"]:
            raise RecorderValidationError(
                "note_partial_command requires receipt.complete=false"
            )
        if source in {"model", "human"}:
            with self._lifecycle_lock:
                self._partial_task_command_count += 1
                self._task_command_sources.add(source)
                self._task_output_may_have_changed = bool(
                    self._task_output_may_have_changed
                    or clean_receipt["arm_published"]
                    or (
                        clean_receipt["left_hand_published"]
                        and clean_receipt["left_hand_changed"]
                    )
                    or (
                        clean_receipt["right_hand_published"]
                        and clean_receipt["right_hand_changed"]
                    )
                )

    def raise_if_failed(self) -> None:
        with self._error_lock:
            failure = self._failure
        if failure is not None:
            raise RecorderFailedError(
                "FA episode recorder failed; staging retained at "
                f"{getattr(self, 'staging_path', '<not-created>')}: {failure}"
            ) from failure

    def finalize(
        self,
        outcome: str,
        *,
        reason: Optional[str] = None,
        failure_reason: Optional[str] = None,
        intervention_summary: Optional[Any] = None,
    ) -> Path:
        """Validate and atomically publish one explicitly labelled episode."""

        clean_outcome, clean_reason, clean_failure_reason = _validate_outcome(
            outcome, reason=reason, failure_reason=failure_reason
        )
        interventions = _structured_json(
            {} if intervention_summary is None else intervention_summary,
            "intervention_summary",
        )
        with self._lifecycle_lock:
            self._require_accepting()
            if self._submitted_frames <= 0:
                raise RecorderValidationError(
                    "cannot finalize an episode with no frames"
                )
            self._accepting = False
            self._state = "finalizing"
        try:
            self._stop_worker_normally()
            self.raise_if_failed()
            _ensure_free_space(self.root, self.min_free_bytes)
            assert self._worker_summary is not None
            manifest = self._validate_staging(self._worker_summary)
            ended_at = _utc_now()
            motion_executed = self._motion_executed()
            command_publication = self._command_publication_evidence()
            outcome_document = {
                "schema_version": SCHEMA_VERSION,
                "episode_uuid": self.episode_uuid,
                "outcome": clean_outcome,
                "success": (
                    True
                    if clean_outcome == "success"
                    else False if clean_outcome == "failure" else None
                ),
                "outcome_reason": clean_reason,
                "failure_reason": clean_failure_reason,
                "intervention_summary": interventions,
                "frame_count": manifest["frame_count"],
                "transition_count": manifest["transition_count"],
                "event_count": manifest["event_count"],
                "motion_executed": motion_executed,
                "command_publication": command_publication,
                "started_at_utc": self._initial_metadata["created_at_utc"],
                "ended_at_utc": ended_at,
            }
            metadata = dict(self._initial_metadata)
            metadata.update(
                status="complete",
                ended_at_utc=ended_at,
                frame_count=manifest["frame_count"],
                transition_count=manifest["transition_count"],
                event_count=manifest["event_count"],
                motion_executed=motion_executed,
                command_publication=command_publication,
                capture_timing=manifest["capture_timing"],
            )
            _atomic_write_json(self.staging_path / "manifest.json", manifest)
            _atomic_write_json(self.staging_path / "outcome.json", outcome_document)
            _atomic_write_json(self.staging_path / "metadata.json", metadata)
            _fsync_tree_files(self.staging_path)
            _fsync_directory(self.staging_path)

            index_row = {
                "schema_version": SCHEMA_VERSION,
                "episode_uuid": self.episode_uuid,
                "path": f"episodes/{self.episode_uuid}",
                "outcome": clean_outcome,
                "failure_reason": clean_failure_reason,
                "frame_count": manifest["frame_count"],
                "transition_count": manifest["transition_count"],
                "motion_executed": motion_executed,
                "command_publication": command_publication,
                "started_at_utc": self._initial_metadata["created_at_utc"],
                "ended_at_utc": ended_at,
            }
            index_path = self.root / "episodes.jsonl"
            old_index = _read_index_bytes(index_path)
            prepared_index = _prepare_index_update(index_path, old_index, index_row)
            committed_document = {
                "schema_version": SCHEMA_VERSION,
                "episode_uuid": self.episode_uuid,
                "committed_at_utc": _utc_now(),
            }
            _atomic_write_json(self.staging_path / "COMMITTED", committed_document)
            _fsync_directory(self.staging_path)
            moved = False
            try:
                os.replace(self.staging_path, self.final_path)
                moved = True
                _fsync_directory(self.root / "episodes")
                os.replace(prepared_index, index_path)
                _fsync_directory(self.root)
            except Exception:
                if prepared_index.exists():
                    prepared_index.unlink()
                if (
                    moved
                    and self.final_path.exists()
                    and not self.staging_path.exists()
                ):
                    os.replace(self.final_path, self.staging_path)
                committed = self.staging_path / "COMMITTED"
                if committed.exists():
                    committed.unlink()
                _restore_index(index_path, old_index)
                raise
            self._state = "finalized"
            self._release_writer_lock()
            return self.final_path
        except Exception as exc:
            self._set_failure(exc)
            worker_alive = (
                getattr(self, "_worker", None) is not None and self._worker.is_alive()
            )
            if not worker_alive:
                self._mark_staging_failed(
                    exc,
                    _effective_error_summary(interventions, exc),
                    outcome_reason=f"{type(exc).__name__}: {exc}",
                )
                self._release_writer_lock()
            self._state = "failed"
            if isinstance(exc, RecorderError):
                raise
            raise RecorderFailedError(
                f"finalize failed; staging retained at {self.staging_path}: {exc}"
            ) from exc

    def abort(
        self,
        reason: str,
        *,
        outcome: str = "aborted",
        intervention_summary: Optional[Any] = None,
    ) -> Path:
        """Stop without publishing and retain the incomplete staging directory."""

        if outcome not in {"aborted", "error"}:
            raise RecorderValidationError("abort outcome must be 'aborted' or 'error'")
        clean_reason = _validate_nonempty_string(reason, "reason")
        interventions = _structured_json(
            {} if intervention_summary is None else intervention_summary,
            "intervention_summary",
        )
        with self._lifecycle_lock:
            if self._state == "finalized":
                return self.final_path
            if self._state in {"aborted", "closed", "failed"}:
                return self.staging_path
            self._accepting = False
            self._state = "aborting"
        self._failed_event.set()
        self._worker.join(timeout=WORKER_STOP_TIMEOUT_SEC)
        if self._worker.is_alive():
            failure = RecorderFailedError(
                "writer did not stop; dataset lock remains held"
            )
            self._set_failure(failure)
            self._state = "failed"
            raise failure
        with self._error_lock:
            existing_failure = self._failure
        effective_outcome = "error" if existing_failure is not None else outcome
        effective_failure = existing_failure or RecorderError(clean_reason)
        effective_reason = (
            f"{type(existing_failure).__name__}: {existing_failure}"
            if existing_failure is not None
            else clean_reason
        )
        if effective_outcome == "error":
            interventions = _effective_error_summary(
                interventions,
                effective_failure,
                requested_reason=clean_reason,
            )
        self._mark_staging_failed(
            effective_failure,
            interventions,
            outcome=effective_outcome,
            outcome_reason=effective_reason,
        )
        self._state = "failed" if existing_failure is not None else "aborted"
        self._release_writer_lock()
        return self.staging_path

    def close(self) -> Path:
        if self._state == "finalized":
            return self.final_path
        return self.abort("recorder closed without finalize")

    def __enter__(self) -> "FaEpisodeRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._state != "finalized":
            reason = (
                "context exited without finalize"
                if exc is None
                else f"context exception: {exc!r}"
            )
            self.abort(reason, outcome="error" if exc is not None else "aborted")

    def _writer_main(self) -> None:
        frames_file = events_file = None
        videos: Optional[_MultiCameraSegmentWriter] = None
        frame_count = event_count = 0
        normal_stop = False
        try:
            frames_file = (self.staging_path / "frames.jsonl").open(
                "x", encoding="utf-8", buffering=1
            )
            events_file = (self.staging_path / "events.jsonl").open(
                "x", encoding="utf-8", buffering=1
            )
            videos = _MultiCameraSegmentWriter(
                self.staging_path,
                fps=self.fps,
                segment_frames=self.segment_frames,
                encoder_factory=self.encoder_factory,
                depth_encoder_factory=self.depth_encoder_factory,
            )
            self._ready.set()
            while True:
                if self._failed_event.is_set():
                    break
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _STOP:
                    normal_stop = True
                    break
                if isinstance(item, _FramePacket):
                    if frame_count % self.fps == 0:
                        _ensure_free_space(self.root, self.min_free_bytes)
                    (
                        segment_index,
                        segment_frame,
                        paths,
                        depth_paths,
                    ) = videos.write(item)
                    command = item.command
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "frame_index": item.frame_index,
                        "sequence": item.sequence,
                        "captured_at": item.captured_at,
                        "state": list(item.state),
                        "sent_command": (
                            None if command is None else list(command.sent_command)
                        ),
                        "sent_command_source": (
                            "none" if command is None else command.source
                        ),
                        "sent_command_sequence": (
                            None if command is None else command.sequence
                        ),
                        "sent_command_chunk": (
                            None if command is None else command.chunk
                        ),
                        "sent_command_step": None if command is None else command.step,
                        "sent_command_at_monotonic_s": (
                            None if command is None else command.sent_at_monotonic_s
                        ),
                        "sent_command_at_wall_s": (
                            None if command is None else command.sent_at_wall_s
                        ),
                        "sent_command_receipt": (
                            None if command is None else command.receipt
                        ),
                        "segment_index": segment_index,
                        "segment_frame_index": segment_frame,
                        "videos": paths,
                        "depth_videos": depth_paths,
                        "depth_sha256": {
                            key: hashlib.sha256(
                                np.ascontiguousarray(item.depths[key]).tobytes()
                            ).hexdigest()
                            for key in CAMERA_KEYS
                        },
                        "camera_metadata": item.camera_metadata,
                        "state_metadata": item.state_metadata,
                        "sync_metadata": item.sync_metadata,
                        "intervention": item.intervention,
                    }
                    _write_json_line(frames_file, row)
                    frame_count += 1
                    self._written_frames = frame_count
                    if frame_count % self.fps == 0:
                        frames_file.flush()
                        os.fsync(frames_file.fileno())
                elif isinstance(item, _EventPacket):
                    _write_json_line(
                        events_file,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "event_index": item.event_index,
                            "type": item.event_type,
                            "recorded_at": item.recorded_at,
                            "payload": item.payload,
                        },
                    )
                    event_count += 1
                    self._written_events = event_count
                else:  # pragma: no cover - private queue
                    raise RecorderError(f"unknown writer item: {type(item)!r}")
            if normal_stop and not self._failed_event.is_set():
                segments = videos.close()
                for handle in (frames_file, events_file):
                    handle.flush()
                    os.fsync(handle.fileno())
                self._worker_summary = {
                    "frame_count": frame_count,
                    "event_count": event_count,
                    "segments": segments,
                }
            else:
                videos.abort()
        except Exception as exc:
            if videos is not None:
                videos.abort()
            self._set_failure(exc)
        finally:
            self._ready.set()
            for handle in (frames_file, events_file):
                if handle is not None:
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except Exception:
                        pass
                    try:
                        handle.close()
                    except Exception:
                        pass

    def _validate_staging(self, summary: dict[str, Any]) -> dict[str, Any]:
        if summary["frame_count"] != self._submitted_frames:
            raise RecorderError(
                f"frame count mismatch: submitted={self._submitted_frames}, "
                f"written={summary['frame_count']}"
            )
        if summary["event_count"] != self._submitted_events:
            raise RecorderError(
                f"event count mismatch: submitted={self._submitted_events}, "
                f"written={summary['event_count']}"
            )
        segments = summary["segments"]
        if (
            not segments
            or sum(row["frame_count"] for row in segments) != self._submitted_frames
        ):
            raise RecorderError("video segment manifest does not cover all frames")
        if list(self.staging_path.rglob("*.partial")):
            raise RecorderError("unfinished video segment remains in staging")
        frame_rows = _read_jsonl(self.staging_path / "frames.jsonl")
        event_rows = _read_jsonl(self.staging_path / "events.jsonl")
        if (
            len(frame_rows) != self._submitted_frames
            or len(event_rows) != self._submitted_events
        ):
            raise RecorderError("JSONL row count differs from submitted count")
        for expected, row in enumerate(frame_rows):
            if row.get("frame_index") != expected:
                raise RecorderError(f"non-contiguous frame_index at row {expected}")
            if "action" in row:
                raise RecorderError(
                    "raw frame row must not contain a misleading action field"
                )
            if len(row.get("state", [])) != STATE_DIM:
                raise RecorderError(f"invalid state dimension at row {expected}")
            depth_hashes = row.get("depth_sha256")
            if not isinstance(depth_hashes, dict) or set(depth_hashes) != set(
                CAMERA_KEYS
            ):
                raise RecorderError(f"invalid depth hashes at row {expected}")
            sent = row.get("sent_command")
            if sent is not None and len(sent) != STATE_DIM:
                raise RecorderError(f"invalid sent_command dimension at row {expected}")
        timing = _capture_timing_stats(
            [row["captured_at"] for row in frame_rows],
            fps=self.fps,
            max_relative_error=self.max_relative_fps_error,
            max_gap_factor=self.max_frame_gap_factor,
        )
        if self.validate_timing and timing["valid"] is False:
            raise RecorderValidationError(
                "capture timing violates contract: " + "; ".join(timing["violations"])
            )
        for segment in segments:
            if not isinstance(segment.get("depth_videos"), dict) or set(
                segment["depth_videos"]
            ) != set(CAMERA_KEYS):
                raise RecorderError("depth video segment must cover all FA cameras")
            for key in CAMERA_KEYS:
                path = self.staging_path / segment["videos"][key]
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RecorderError(f"missing or empty video: {path}")
                if self._uses_default_encoder:
                    _validate_h264_video(
                        path,
                        expected_frames=segment["frame_count"],
                        expected_fps=self.fps,
                        expected_shape=CAMERA_SHAPES[key],
                    )
                depth_path = self.staging_path / segment["depth_videos"][key]
                if not depth_path.is_file() or depth_path.stat().st_size <= 0:
                    raise RecorderError(f"missing or empty depth video: {depth_path}")
                if self._uses_default_depth_encoder:
                    start = int(segment["start_frame"])
                    stop = start + int(segment["frame_count"])
                    _validate_blosc_depth_stream(
                        depth_path,
                        expected_hashes=[
                            row["depth_sha256"][key] for row in frame_rows[start:stop]
                        ],
                        expected_fps=self.fps,
                        expected_shape=DEPTH_SHAPES[key],
                    )
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_uuid": self.episode_uuid,
            "frame_count": self._submitted_frames,
            "transition_count": max(0, self._submitted_frames - 1),
            "event_count": self._submitted_events,
            "segments": segments,
            "capture_timing": timing,
            "camera_keys": list(CAMERA_KEYS),
            "depth_camera_keys": list(CAMERA_KEYS),
            "training_export_contract": {
                "observation": "frames[t]",
                "action": "frames[t+1].state",
                "sent_command_is_not_training_action": True,
            },
        }

    def _require_accepting(self) -> None:
        self.raise_if_failed()
        if not self._accepting or self._state != "recording":
            raise RecorderError(f"recorder is not accepting data: state={self._state}")

    def _motion_executed(self) -> Optional[bool]:
        enabled = self.config.get("enable_motion")
        if enabled is False:
            return False
        with self._lifecycle_lock:
            task_output_may_have_changed = self._task_output_may_have_changed
        if not task_output_may_have_changed:
            return False if enabled is True else None
        # The current FA interface observes joint state but receives no
        # command/trajectory execution acknowledgement.  Do not turn a ROS
        # publish return into a claim that physical motion executed.
        return None

    def _command_publication_evidence(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            return {
                "complete_task_command_observed": self._saw_task_command,
                "successful_publish_notification_count": (
                    self._complete_task_command_count
                ),
                "partial_task_command_count": self._partial_task_command_count,
                "task_command_sources": sorted(self._task_command_sources),
                "task_output_may_have_changed": self._task_output_may_have_changed,
                "physical_execution_verified": False,
            }

    def _put_nowait(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            failure = RecorderFailedError(
                f"writer queue full (capacity={self.queue_size}); refusing to drop data"
            )
            self._set_failure(failure)
            raise failure from exc

    def _fail_and_raise(self, failure: BaseException) -> None:
        self._set_failure(failure)
        raise failure

    def _set_failure(self, failure: BaseException) -> None:
        with self._error_lock:
            if self._failure is None:
                self._failure = failure
        self._failed_event.set()

    def _stop_worker_normally(self) -> None:
        deadline = time.monotonic() + WORKER_STOP_TIMEOUT_SEC
        stop_enqueued = False
        while self._worker.is_alive() and not stop_enqueued:
            with self._error_lock:
                if self._failure is not None:
                    break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._queue.put(_STOP, timeout=min(0.1, remaining))
                stop_enqueued = True
            except queue.Full:
                continue
        self._worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._worker.is_alive():
            failure = RecorderFailedError(
                "writer did not stop; dataset lock remains held"
            )
            self._set_failure(failure)
            raise failure

    def _mark_staging_failed(
        self,
        failure: BaseException,
        interventions: dict[str, Any],
        *,
        outcome: str = "error",
        outcome_reason: Optional[str] = None,
    ) -> None:
        if not getattr(self, "staging_path", None) or not self.staging_path.exists():
            return
        now = _utc_now()
        motion_executed = self._motion_executed()
        command_publication = self._command_publication_evidence()
        metadata = dict(self._initial_metadata)
        metadata.update(
            status="failed" if outcome == "error" else "aborted",
            failure=repr(failure),
            ended_at_utc=now,
            frame_count=self._written_frames,
            submitted_frame_count=self._submitted_frames,
            event_count=self._written_events,
            submitted_event_count=self._submitted_events,
            motion_executed=motion_executed,
            command_publication=command_publication,
        )
        outcome_document = {
            "schema_version": SCHEMA_VERSION,
            "episode_uuid": self.episode_uuid,
            "outcome": outcome,
            "success": None,
            "outcome_reason": (
                str(failure) if outcome_reason is None else outcome_reason
            ),
            "failure_reason": None,
            "intervention_summary": interventions,
            "frame_count": self._written_frames,
            "submitted_frame_count": self._submitted_frames,
            "event_count": self._written_events,
            "submitted_event_count": self._submitted_events,
            "motion_executed": motion_executed,
            "command_publication": command_publication,
            "started_at_utc": self._initial_metadata["created_at_utc"],
            "ended_at_utc": now,
            "incomplete": True,
        }
        for path, document in (
            (self.staging_path / "metadata.json", metadata),
            (self.staging_path / "outcome.json", outcome_document),
        ):
            try:
                _atomic_write_json(path, document)
            except Exception:
                pass

    def _acquire_writer_lock(self) -> None:
        lock_path = self.root / ".fa_episode_writer.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RecorderBusyError(
                f"dataset already has an active writer: {self.root}"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} acquired_at={_utc_now()}\n".encode())
        os.fsync(fd)
        self._lock_fd = fd

    def _release_writer_lock(self) -> None:
        fd, self._lock_fd = self._lock_fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _snapshot_field(snapshot: Any, name: str) -> Any:
    value = _optional_snapshot_field(snapshot, name)
    if value is None:
        raise RecorderValidationError(f"snapshot.{name} is required")
    return value


def _optional_snapshot_field(snapshot: Any, name: str) -> Any:
    if isinstance(snapshot, Mapping):
        return snapshot.get(name)
    return getattr(snapshot, name, None)


def _validate_images(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise RecorderValidationError("images must be a mapping")
    if set(value) != set(CAMERA_KEYS):
        raise RecorderValidationError(
            f"images keys must be exactly {CAMERA_KEYS}, got {tuple(value)}"
        )
    result: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        image = value[key]
        if not isinstance(image, np.ndarray):
            raise RecorderValidationError(f"images[{key!r}] must be numpy.ndarray")
        if image.dtype != np.uint8 or image.shape != CAMERA_SHAPES[key]:
            raise RecorderValidationError(
                f"images[{key!r}] expected {CAMERA_SHAPES[key]}/uint8, "
                f"got {image.shape}/{image.dtype}"
            )
        result[key] = image
    return result


def _validate_depths(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise RecorderValidationError("depths must be a mapping")
    if set(value) != set(CAMERA_KEYS):
        raise RecorderValidationError(
            f"depths keys must be exactly {CAMERA_KEYS}, got {tuple(value)}"
        )
    result: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        depth = value[key]
        if not isinstance(depth, np.ndarray):
            raise RecorderValidationError(f"depths[{key!r}] must be numpy.ndarray")
        if depth.dtype != np.uint16 or depth.shape != DEPTH_SHAPES[key]:
            raise RecorderValidationError(
                f"depths[{key!r}] expected {DEPTH_SHAPES[key]}/uint16, "
                f"got {depth.shape}/{depth.dtype}"
            )
        if int(depth.max(initial=0)) > 1000:
            raise RecorderValidationError(
                f"depths[{key!r}] exceeds 1000 mm configured training range"
            )
        result[key] = depth
    return result


def _validate_camera_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(CAMERA_KEYS):
        raise RecorderValidationError(
            f"camera_metadata keys must be exactly {CAMERA_KEYS}"
        )
    result: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        entry = value[key]
        if not isinstance(entry, Mapping):
            raise RecorderValidationError(f"camera_metadata[{key!r}] must be a mapping")
        clean = _structured_json(entry, f"camera_metadata.{key}")
        _validate_nonnegative_int(
            clean.get("sequence"), f"camera_metadata.{key}.sequence"
        )
        _validate_timestamp(
            clean.get("captured_at_monotonic_s"),
            f"camera_metadata.{key}.captured_at_monotonic_s",
        )
        result[key] = clean
    return result


def _copy_command_snapshot(value: SentCommandSnapshot) -> SentCommandSnapshot:
    return SentCommandSnapshot(
        sent_command=tuple(value.sent_command),
        source=value.source,
        sequence=value.sequence,
        sent_at_monotonic_s=value.sent_at_monotonic_s,
        sent_at_wall_s=value.sent_at_wall_s,
        chunk=value.chunk,
        step=value.step,
        receipt=copy.deepcopy(value.receipt),
    )


def _normalize_command_snapshot(value: Any) -> Optional[SentCommandSnapshot]:
    """Accept the recorder register or the FA client's ``SentCommand.to_dict``."""

    if value is None:
        return None
    if isinstance(value, SentCommandSnapshot):
        snapshot = _copy_command_snapshot(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        if not isinstance(value, Mapping):
            raise RecorderValidationError(
                "sent_command must be None, SentCommandSnapshot, or a mapping"
            )
        raw_command = value.get("command", value.get("sent_command"))
        try:
            command_array = np.asarray(raw_command, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RecorderValidationError(
                "sent_command.command is not a numeric vector"
            ) from exc
        command = _validate_vector(command_array, "sent_command.command")
        source = _validate_command_source(value.get("source"))
        sequence = _validate_nonnegative_int(
            value.get("sequence"), "sent_command.sequence"
        )
        sent_at_monotonic_s = _validate_timestamp(
            value.get("sent_at_monotonic_s"), "sent_command.sent_at_monotonic_s"
        )
        sent_at_wall_s = _validate_timestamp(
            value.get("sent_at_wall_s"), "sent_command.sent_at_wall_s"
        )
        receipt = _structured_json(value.get("receipt"), "sent_command.receipt")
        snapshot = SentCommandSnapshot(
            sent_command=tuple(float(item) for item in command),
            source=source,
            sequence=sequence,
            sent_at_monotonic_s=sent_at_monotonic_s,
            sent_at_wall_s=sent_at_wall_s,
            chunk=_validate_optional_nonnegative_int(
                value.get("chunk"), "sent_command.chunk"
            ),
            step=_validate_optional_nonnegative_int(
                value.get("step"), "sent_command.step"
            ),
            receipt=receipt,
        )
    if snapshot.receipt.get("complete") is not True:
        raise RecorderValidationError("sent_command.receipt.complete must be true")
    receipt_source = snapshot.receipt.get("source")
    if receipt_source is not None and receipt_source != snapshot.source:
        raise RecorderValidationError("sent_command source differs from receipt source")
    receipt_time = snapshot.receipt.get("sent_at_monotonic_s")
    if receipt_time is not None and not math.isclose(
        _validate_timestamp(receipt_time, "sent_command.receipt.sent_at_monotonic_s"),
        snapshot.sent_at_monotonic_s,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RecorderValidationError("sent_command time differs from receipt time")
    publish_flags = (
        "arm_published",
        "left_hand_published",
        "right_hand_published",
    )
    if any(flag in snapshot.receipt for flag in publish_flags) and not all(
        snapshot.receipt.get(flag) is True for flag in publish_flags
    ):
        raise RecorderValidationError("sent_command receipt is a partial FA publish")
    return _copy_command_snapshot(snapshot)


def _validate_vector(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RecorderValidationError(f"{name} must be numpy.ndarray")
    if value.dtype != np.float32 or value.shape != (STATE_DIM,):
        raise RecorderValidationError(
            f"{name} expected ({STATE_DIM},)/float32, got {value.shape}/{value.dtype}"
        )
    if not np.isfinite(value).all():
        raise RecorderValidationError(f"{name} contains NaN or infinity")
    return value


def _validate_command_source(value: Any) -> str:
    source = _validate_nonempty_string(value, "command source")
    if source not in VALID_COMMAND_SOURCES:
        raise RecorderValidationError(
            f"command source must be one of {sorted(VALID_COMMAND_SOURCES)}"
        )
    return source


def _validate_outcome(
    outcome: Any,
    *,
    reason: Optional[Any],
    failure_reason: Optional[Any],
) -> tuple[str, Optional[str], Optional[str]]:
    clean_outcome = _validate_nonempty_string(outcome, "outcome")
    if clean_outcome not in VALID_OUTCOMES:
        raise RecorderValidationError(
            f"outcome must be one of {sorted(VALID_OUTCOMES)}"
        )
    clean_reason = (
        None if reason is None else _validate_nonempty_string(reason, "reason")
    )
    clean_failure = (
        None
        if failure_reason is None
        else _validate_nonempty_string(failure_reason, "failure_reason")
    )
    if clean_outcome == "success":
        if clean_reason is not None or clean_failure is not None:
            raise RecorderValidationError("success accepts no reason or failure_reason")
    elif clean_outcome == "failure":
        if clean_reason is not None or clean_failure is None:
            raise RecorderValidationError("failure requires only failure_reason")
    else:
        if clean_reason is None or clean_failure is not None:
            raise RecorderValidationError(
                f"{clean_outcome} requires only a non-empty reason"
            )
    return clean_outcome, clean_reason, clean_failure


def _validate_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecorderValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecorderValidationError(f"{name} must be a positive integer")
    return value


def _validate_nonnegative_int(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise RecorderValidationError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_optional_nonnegative_int(value: Any, name: str) -> Optional[int]:
    return None if value is None else _validate_nonnegative_int(value, name)


def _validate_nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecorderValidationError(f"{name} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RecorderValidationError(f"{name} must be finite and non-negative")
    return result


def _validate_positive_finite(value: Any, name: str) -> float:
    result = _validate_nonnegative_finite(value, name)
    if result <= 0:
        raise RecorderValidationError(f"{name} must be positive")
    return result


def _validate_timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RecorderValidationError(f"{name} must be a finite non-negative timestamp")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RecorderValidationError(f"{name} must be a finite non-negative timestamp")
    return result


def _json_compatible(value: Any, name: str) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_compatible(to_dict(), name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecorderValidationError(f"{name} contains NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _json_compatible(value.item(), name)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RecorderValidationError(f"{name} contains NaN or infinity")
        return _json_compatible(value.tolist(), name)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecorderValidationError(f"{name} mapping keys must be strings")
            result[key] = _json_compatible(item, f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item, f"{name}[]") for item in value]
    raise RecorderValidationError(f"{name} has unsupported type {type(value).__name__}")


def _structured_json(value: Any, name: str) -> dict[str, Any]:
    result = _json_compatible(value, name)
    if not isinstance(result, dict):
        raise RecorderValidationError(f"{name} must be a mapping or expose to_dict()")
    return result


def _effective_error_summary(
    intervention_summary: Mapping[str, Any],
    error: BaseException,
    *,
    requested_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Retain the reported label while making recorder failure authoritative."""

    result = copy.deepcopy(dict(intervention_summary))
    error_text = f"{type(error).__name__}: {error}"
    recorder_errors = result.get("recorder_errors", [])
    if not isinstance(recorder_errors, list):
        recorder_errors = [recorder_errors]
    if error_text not in recorder_errors:
        recorder_errors.append(error_text)
    result.update(
        {
            "effective_outcome": "error",
            "effective_reason": error_text,
            "recorder_errors": recorder_errors,
        }
    )
    if requested_reason is not None:
        result["requested_abort_reason"] = requested_reason
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_json_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(_canonical_json(value) + "\n")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode()
    _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecorderError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RecorderError(
                    f"JSONL row at {path}:{line_number} is not an object"
                )
            rows.append(value)
    return rows


def _read_index_bytes(path: Path) -> Optional[bytes]:
    if not path.exists():
        return None
    payload = path.read_bytes()
    for line_number, line in enumerate(payload.decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecorderError(f"invalid episode index at line {line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("episode_uuid"), str):
            raise RecorderError(f"invalid episode index row at line {line_number}")
    return payload


def _prepare_index_update(
    path: Path, old_payload: Optional[bytes], row: Mapping[str, Any]
) -> Path:
    old = b"" if old_payload is None else old_payload
    if old and not old.endswith(b"\n"):
        old += b"\n"
    existing = {
        json.loads(line)["episode_uuid"]
        for line in old.decode().splitlines()
        if line.strip()
    }
    if row["episode_uuid"] in existing:
        raise RecorderError(f"duplicate episode UUID: {row['episode_uuid']}")
    temporary = path.parent / f".{path.name}.{row['episode_uuid']}.prepared"
    with temporary.open("xb") as handle:
        handle.write(old + (_canonical_json(row) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _restore_index(path: Path, payload: Optional[bytes]) -> None:
    if payload is None:
        if path.exists():
            path.unlink()
            _fsync_directory(path.parent)
        return
    _atomic_write_bytes(path, payload)


def _capture_timing_stats(
    timestamps: list[Any],
    *,
    fps: int,
    max_relative_error: float,
    max_gap_factor: float,
) -> dict[str, Any]:
    expected_interval = 1.0 / fps
    if len(timestamps) < 2:
        return {
            "valid": None,
            "assessed": False,
            "target_fps": fps,
            "frame_count": len(timestamps),
            "violations": [],
        }
    numeric = [float(value) for value in timestamps]
    intervals = [later - earlier for earlier, later in zip(numeric, numeric[1:])]
    if not all(math.isfinite(value) and value > 0 for value in intervals):
        raise RecorderError("captured_at is not finite and strictly increasing")
    duration = numeric[-1] - numeric[0]
    effective_fps = (len(numeric) - 1) / duration
    relative_error = abs(effective_fps / fps - 1.0)
    max_gap = max(intervals)
    max_allowed_gap = max_gap_factor * expected_interval
    violations = []
    if relative_error > max_relative_error:
        violations.append(
            f"effective_fps={effective_fps:.6f} differs from target={fps} by "
            f"{relative_error:.1%}"
        )
    if max_gap > max_allowed_gap:
        violations.append(f"max_gap={max_gap:.6f}s exceeds {max_allowed_gap:.6f}s")
    return {
        "valid": not violations,
        "assessed": True,
        "target_fps": fps,
        "frame_count": len(numeric),
        "duration_sec": duration,
        "effective_fps": effective_fps,
        "relative_fps_error": relative_error,
        "max_gap_sec": max_gap,
        "max_allowed_gap_sec": max_allowed_gap,
        "violations": violations,
    }


def _validate_h264_video(
    path: Path,
    *,
    expected_frames: int,
    expected_fps: int,
    expected_shape: tuple[int, int, int],
) -> None:
    try:
        import av  # type: ignore
    except ImportError as exc:  # pragma: no cover - default encoder already imports it
        raise RecorderError("PyAV disappeared before validation") from exc
    container = av.open(str(path), mode="r")
    try:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise RecorderError(f"expected one video stream in {path}")
        stream = streams[0]
        if str(stream.codec_context.name).lower() != "h264":
            raise RecorderError(f"expected H.264 in {path}")
        if (stream.width, stream.height) != (expected_shape[1], expected_shape[0]):
            raise RecorderError(f"video dimensions mismatch in {path}")
        if stream.average_rate is None or abs(
            float(stream.average_rate) - expected_fps
        ) > max(0.01, expected_fps * 0.01):
            raise RecorderError(f"video fps mismatch in {path}")
        decoded = 0
        for frame in container.decode(stream):
            decoded += 1
            if (frame.width, frame.height) != (expected_shape[1], expected_shape[0]):
                raise RecorderError(f"decoded frame dimensions mismatch in {path}")
        if decoded != expected_frames:
            raise RecorderError(
                f"decoded frame count mismatch in {path}: {decoded} != {expected_frames}"
            )
    finally:
        container.close()


def _read_exact(stream: Any, count: int, path: Path) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise RecorderError(f"truncated depth stream: {path}")
    return value


def _validate_blosc_depth_stream(
    path: Path,
    *,
    expected_hashes: list[str],
    expected_fps: int,
    expected_shape: tuple[int, int],
) -> None:
    try:
        import blosc  # type: ignore
    except ImportError as exc:  # pragma: no cover - default encoder already imports it
        raise RecorderError("python-blosc disappeared before depth validation") from exc
    with path.open("rb") as stream:
        if _read_exact(stream, len(_DEPTH_STREAM_MAGIC), path) != _DEPTH_STREAM_MAGIC:
            raise RecorderError(f"invalid depth stream magic in {path}")
        header_length = struct.unpack("<I", _read_exact(stream, 4, path))[0]
        if header_length <= 0 or header_length > 4096:
            raise RecorderError(f"invalid depth stream header length in {path}")
        try:
            header = json.loads(_read_exact(stream, header_length, path))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecorderError(f"invalid depth stream header in {path}") from exc
        expected_header = {
            "dtype": "<u2",
            "shape": list(expected_shape),
            "fps": expected_fps,
            "compressor": "blosc-zstd",
            "clevel": 1,
            "shuffle": "bitshuffle",
        }
        if header != expected_header:
            raise RecorderError(f"depth stream header mismatch in {path}")
        frame_bytes = int(np.prod(expected_shape)) * np.dtype("<u2").itemsize
        for index, expected_hash in enumerate(expected_hashes):
            payload_length = struct.unpack("<I", _read_exact(stream, 4, path))[0]
            if payload_length <= 0 or payload_length > frame_bytes + 65536:
                raise RecorderError(f"invalid compressed depth frame in {path}")
            try:
                raw = blosc.decompress(_read_exact(stream, payload_length, path))
            except Exception as exc:
                raise RecorderError(
                    f"could not decompress depth frame {index} in {path}"
                ) from exc
            if len(raw) != frame_bytes:
                raise RecorderError(f"depth frame size mismatch in {path}")
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected_hash:
                raise RecorderError(
                    f"lossless depth verification failed in {path} at frame {index}"
                )
        if stream.read(1):
            raise RecorderError(f"excess depth frame data in {path}")


def _ensure_free_space(path: Path, min_free_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    if free < min_free_bytes:
        raise RecorderError(
            f"insufficient free disk space at {path}: free={free}, required={min_free_bytes}"
        )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            _fsync_file(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "CAMERA_KEYS",
    "CAMERA_SHAPES",
    "DEPTH_SHAPES",
    "FaEpisodeRecorder",
    "RecorderBusyError",
    "RecorderError",
    "RecorderFailedError",
    "RecorderValidationError",
    "STATE_DIM",
    "STATE_NAMES",
    "SentCommandSnapshot",
    "VALID_COMMAND_SOURCES",
    "VALID_OUTCOMES",
]
