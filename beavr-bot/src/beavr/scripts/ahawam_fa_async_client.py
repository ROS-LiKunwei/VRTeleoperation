#!/usr/bin/env python3
"""FA V6 inference, exclusive intervention gateway, and episode recorder.

Run this from the beavr-bot environment on the robot brain. It reuses the
recording camera_settings.json, including RGB-D profiles and fixed manual
camera controls. Motion is disabled by default.  Concurrent FA VR is safe only
when it is launched with ``--ahawam_intervention_mode=true`` so it publishes
isolated human candidates instead of writing the hardware topics directly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import json
import math
import pickle
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from beavr.scripts.ahawam_intervention import (
    ControlMode,
    EpisodeOutcome,
    OperatorIntervention,
    TerminalKeyListener,
)

CAMERA_KEYS = ("head", "wrist_CV2756100033", "wrist_CV2856D0001C")
LEFT_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
)
RIGHT_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_JOINTS)
FA_GRIPPER_CLOSE_THRESHOLD = 0.8
_NDARRAY_MARKER = "__ahawam_ndarray_v1__"


class TakeoverRequested(RuntimeError):
    """The model pipeline has stopped and a measured hold is required."""


class EpisodeTerminated(RuntimeError):
    def __init__(self, outcome: EpisodeOutcome, reason: str):
        super().__init__(reason)
        self.outcome = outcome
        self.reason = reason


class ObservationStreamError(RuntimeError):
    """Camera/state/recording faults for which human motion is unsafe."""


class ModelFailure(RuntimeError):
    """Inference/response fault that may safely hand off after a measured hold."""


@dataclass(frozen=True)
class FaCommandReceipt:
    source: str
    sent_at_monotonic_s: float
    arm_published: bool
    left_hand_published: bool
    right_hand_published: bool
    left_hand_changed: bool
    right_hand_changed: bool

    @property
    def complete(self) -> bool:
        return (
            self.arm_published
            and self.left_hand_published
            and self.right_hand_published
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "sent_at_monotonic_s": self.sent_at_monotonic_s,
            "arm_published": self.arm_published,
            "left_hand_published": self.left_hand_published,
            "right_hand_published": self.right_hand_published,
            "left_hand_changed": self.left_hand_changed,
            "right_hand_changed": self.right_hand_changed,
            "complete": self.complete,
        }


class PartialCommandPublishError(RuntimeError):
    """At least one FA command topic was written before publication failed."""

    def __init__(self, message: str, receipt: FaCommandReceipt):
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True)
class SentCommand:
    sequence: int
    sent_at_monotonic_s: float
    sent_at_wall_s: float
    source: str
    command: np.ndarray
    receipt: FaCommandReceipt

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "sent_at_monotonic_s": self.sent_at_monotonic_s,
            "sent_at_wall_s": self.sent_at_wall_s,
            "source": self.source,
            "command": self.command.astype(float).tolist(),
            "receipt": self.receipt.to_dict(),
        }


class SentCommandHistory:
    """Causally bind an observation only to commands sent before its time."""

    def __init__(self, maxlen: int = 4096) -> None:
        self._lock = threading.RLock()
        self._commands: deque[SentCommand] = deque(maxlen=maxlen)
        self._sequence = 0

    def mark_sent(
        self,
        command: np.ndarray,
        source: str,
        receipt: FaCommandReceipt,
    ) -> SentCommand:
        vector = np.asarray(command, dtype=np.float32)
        if vector.shape != (16,) or not np.isfinite(vector).all():
            raise ValueError("sent FA command must be a finite 16D vector")
        if not receipt.complete:
            raise ValueError("cannot mark an incomplete FA command as sent")
        with self._lock:
            self._sequence += 1
            item = SentCommand(
                sequence=self._sequence,
                sent_at_monotonic_s=receipt.sent_at_monotonic_s,
                sent_at_wall_s=time.time(),
                source=str(source),
                command=vector.copy(),
                receipt=receipt,
            )
            self._commands.append(item)
            return item

    def snapshot_at(self, observation_monotonic_s: float) -> Optional[SentCommand]:
        timestamp = float(observation_monotonic_s)
        with self._lock:
            for item in reversed(self._commands):
                if item.sent_at_monotonic_s <= timestamp:
                    return SentCommand(
                        sequence=item.sequence,
                        sent_at_monotonic_s=item.sent_at_monotonic_s,
                        sent_at_wall_s=item.sent_at_wall_s,
                        source=item.source,
                        command=item.command.copy(),
                        receipt=item.receipt,
                    )
        return None

    def latest(self) -> Optional[SentCommand]:
        with self._lock:
            if not self._commands:
                return None
            return self.snapshot_at(float("inf"))


@dataclass(frozen=True)
class FaObservation:
    sequence: int
    captured_at_monotonic_s: float
    captured_at_wall_s: float
    images: dict[str, np.ndarray]
    state: np.ndarray
    camera_metadata: dict[str, dict[str, object]]
    state_metadata: dict[str, object]
    sync_metadata: dict[str, object]
    command: Optional[SentCommand]


def binarize_gripper(value: float) -> float:
    return float(value >= FA_GRIPPER_CLOSE_THRESHOLD)


def _encode_for_wire(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        if array.dtype.hasobject:
            raise TypeError("Object-dtype NumPy arrays are not supported")
        return {
            _NDARRAY_MARKER: True,
            "dtype": array.dtype.str,
            "shape": tuple(int(dim) for dim in array.shape),
            "data": array.tobytes(order="C"),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _encode_for_wire(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_encode_for_wire(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_encode_for_wire(value) for value in obj)
    return obj


def _decode_from_wire(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_NDARRAY_MARKER) is True:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(int(dim) for dim in obj["shape"])
        data = obj["data"]
        if dtype.hasobject or any(dim < 0 for dim in shape) or not isinstance(data, bytes):
            raise ValueError("Malformed AHA-WAM ndarray payload")
        if len(data) != math.prod(shape) * dtype.itemsize:
            raise ValueError("AHA-WAM ndarray payload size mismatch")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if isinstance(obj, dict):
        return {key: _decode_from_wire(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode_from_wire(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_decode_from_wire(value) for value in obj)
    return obj


def recv_message(conn: socket.socket) -> tuple[Any, int]:
    def recv_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            packet = conn.recv(size - len(data))
            if not packet:
                raise ConnectionError("remote closed connection")
            data.extend(packet)
        return bytes(data)

    (size,) = struct.unpack("!I", recv_exact(4))
    payload = recv_exact(size)
    return _decode_from_wire(pickle.loads(payload)), size


def send_message(conn: socket.socket, message: Any) -> int:
    payload = pickle.dumps(_encode_for_wire(message), protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!I", len(payload)) + payload)
    return len(payload)


class PolicyConnection:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        connect_timeout: Optional[float] = None,
    ):
        self.sock = socket.create_connection(
            (host, port),
            timeout=timeout if connect_timeout is None else connect_timeout,
        )
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._request_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False

    def request(self, message):
        with self._request_lock:
            if self._closed:
                raise ConnectionError("policy connection is closed")
            send_message(self.sock, message)
            reply, _ = recv_message(self.sock)
            if not isinstance(reply, dict) or not reply.get("ok", False):
                raise RuntimeError(f"Policy request failed: {reply}")
            return reply

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass


class FaRos:
    def __init__(self, args):
        try:
            import rclpy
            from min_snap.msg import MinSnapTarget
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Int32
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FA ROS2 messages are unavailable. Before activating the beavr "
                "environment, source /opt/ros/humble/setup.bash and "
                "/home/likunwei/humanoid_ws/install/setup.bash"
            ) from exc

        if not rclpy.ok():
            rclpy.init(args=None)
        self.rclpy = rclpy
        self.node = rclpy.create_node("ahawam_fa_async_client")
        self._state_condition = threading.Condition()
        self._spin_stop = threading.Event()
        self._spin_error = None
        self.latest = None
        self.latest_at = 0.0
        self.latest_source_at = 0.0
        self.source_clock_offset_s = None
        self.latest_sequence = 0
        self._joint_history = deque(maxlen=4096)
        self.last_joint_message_at = 0.0
        self.last_missing_joints = ()
        self.left_gripper = float(args.initial_left_gripper_state)
        self.right_gripper = float(args.initial_right_gripper_state)
        self._human_condition = threading.Condition()
        self._human_arm: Optional[np.ndarray] = None
        self._human_left_gripper = self.left_gripper
        self._human_right_gripper = self.right_gripper
        self._human_sequence = 0
        self._human_arm_sequence = 0
        self._human_received_at = 0.0
        self._human_error: Optional[BaseException] = None
        # The ROS queue retains only the newest pending message. A dedicated
        # spin thread consumes feedback continuously, while a bounded local
        # history supports timestamp matching to camera observations.
        self.node.create_subscription(JointState, args.joint_state_topic, self._state, 1)
        self.arm_pub = self.node.create_publisher(MinSnapTarget, args.arm_command_topic, 1)
        self.left_hand_pub = self.node.create_publisher(Int32, args.left_hand_topic, 10)
        self.right_hand_pub = self.node.create_publisher(Int32, args.right_hand_topic, 10)
        self.MinSnapTarget, self.Int32 = MinSnapTarget, Int32
        if getattr(args, "enable_operator_intervention", False):
            human_arm_topic = str(args.human_arm_input_topic)
            human_left_topic = str(args.human_left_hand_input_topic)
            human_right_topic = str(args.human_right_hand_input_topic)
            real_topics = {
                str(args.arm_command_topic),
                str(args.left_hand_topic),
                str(args.right_hand_topic),
            }
            candidate_topics = {
                human_arm_topic,
                human_left_topic,
                human_right_topic,
            }
            if len(candidate_topics) != 3 or real_topics & candidate_topics:
                raise ValueError(
                    "FA human candidate topics must be distinct from each other "
                    "and from the real robot command topics"
                )
            self.node.create_subscription(
                MinSnapTarget, human_arm_topic, self._human_arm_callback, 1
            )
            self.node.create_subscription(
                Int32,
                human_left_topic,
                lambda msg: self._human_hand_callback("left", msg),
                10,
            )
            self.node.create_subscription(
                Int32,
                human_right_topic,
                lambda msg: self._human_hand_callback("right", msg),
                10,
            )
        self._spin_thread = threading.Thread(
            target=self._spin_feedback,
            name="ahawam-fa-joint-feedback",
            daemon=True,
        )
        self._spin_thread.start()

    def _human_arm_callback(self, msg) -> None:
        try:
            left = np.asarray(msg.left_arm_target_rad, dtype=np.float32)
            right = np.asarray(msg.right_arm_target_rad, dtype=np.float32)
            if (
                left.shape != (7,)
                or right.shape != (7,)
                or not np.isfinite(left).all()
                or not np.isfinite(right).all()
            ):
                raise ValueError(
                    "human MinSnapTarget must contain finite left/right 7D arms"
                )
            arm = np.concatenate([left, right]).astype(np.float32, copy=False)
            with self._human_condition:
                self._human_arm = arm
                self._human_sequence += 1
                self._human_arm_sequence = self._human_sequence
                self._human_received_at = time.monotonic()
                self._human_condition.notify_all()
        except BaseException as exc:
            with self._human_condition:
                self._human_error = exc
                self._human_condition.notify_all()

    def _human_hand_callback(self, side: str, msg) -> None:
        try:
            action_id = int(msg.data)
            if action_id not in (20, 21):
                raise ValueError(
                    f"human {side} hand action must be 20(close) or 21(open)"
                )
            value = 1.0 if action_id == 20 else 0.0
            with self._human_condition:
                if side == "left":
                    self._human_left_gripper = value
                else:
                    self._human_right_gripper = value
                self._human_sequence += 1
                self._human_received_at = time.monotonic()
                self._human_condition.notify_all()
        except BaseException as exc:
            with self._human_condition:
                self._human_error = exc
                self._human_condition.notify_all()

    @property
    def human_command_sequence(self) -> int:
        with self._human_condition:
            return self._human_sequence

    def begin_human_handoff(self) -> int:
        """Discard pre-handoff hand state and seed the current real latches."""

        with self._human_condition:
            # Candidate hand messages received during AUTO must not overwrite
            # the gripper state most recently sent by the model/hold gateway.
            # Any callback after this boundary acquires the same lock, advances
            # the event sequence, and is therefore an authorized human edge.
            self._human_left_gripper = float(self.left_gripper)
            self._human_right_gripper = float(self.right_gripper)
            return self._human_sequence

    def wait_human_action(
        self,
        *,
        newer_than_sequence: int,
        timeout_s: float,
        max_age_s: float,
        fallback_arm: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Return only a fresh VR proposal produced after the handoff edge."""

        fallback = np.asarray(fallback_arm, dtype=np.float32)
        if fallback.shape != (14,) or not np.isfinite(fallback).all():
            raise ValueError("human fallback arm must be a finite 14D vector")
        deadline = time.monotonic() + float(timeout_s)
        with self._human_condition:
            while True:
                if self._human_error is not None:
                    raise RuntimeError("invalid FA human command proposal") from self._human_error
                now = time.monotonic()
                age_s = now - self._human_received_at
                if self._human_sequence > int(newer_than_sequence) and age_s <= float(
                    max_age_s
                ):
                    # A hand edge can arrive while the VR arm target is
                    # stationary and therefore not republished.  Never reuse a
                    # pre-handoff arm target for that edge: hold the last
                    # authorized arm until a newer arm proposal is observed.
                    arm = (
                        self._human_arm.copy()
                        if self._human_arm is not None
                        and self._human_arm_sequence
                        > int(newer_than_sequence)
                        else fallback.copy()
                    )
                    action = np.concatenate(
                        [
                            arm,
                            [self._human_left_gripper, self._human_right_gripper],
                        ]
                    ).astype(np.float32)
                    return action, {
                        "sequence": self._human_sequence,
                        "arm_sequence": self._human_arm_sequence,
                        "received_at_monotonic_s": self._human_received_at,
                        "age_ms": max(0.0, age_s * 1000.0),
                    }
                remaining = deadline - now
                if remaining <= 0.0:
                    raise TimeoutError("no fresh post-handoff FA human command")
                self._human_condition.wait(timeout=min(0.02, remaining))

    def validate_command_topology(self, args, timeout_s: float) -> None:
        """Fail before motion when another node can bypass this arbiter."""

        if not args.enable_motion:
            return
        deadline = time.monotonic() + float(timeout_s)
        real_topics = (
            args.arm_command_topic,
            args.left_hand_topic,
            args.right_hand_topic,
        )
        human_topics = (
            args.human_arm_input_topic,
            args.human_left_hand_input_topic,
            args.human_right_hand_input_topic,
        )
        while True:
            real_counts = {
                topic: int(self.node.count_publishers(topic))
                for topic in real_topics
            }
            human_counts = {
                topic: int(self.node.count_publishers(topic))
                for topic in human_topics
            }
            real_is_exclusive = all(count == 1 for count in real_counts.values())
            human_ready = (
                not args.enable_operator_intervention
                or all(count == 1 for count in human_counts.values())
            )
            if real_is_exclusive and human_ready:
                return
            if time.monotonic() >= deadline:
                details = {
                    "real_publishers_including_this_client": real_counts,
                    "human_candidate_publishers": human_counts,
                }
                if not real_is_exclusive:
                    raise RuntimeError(
                        "FA command ownership is not exclusive; stop direct "
                        "teleop publishers and use "
                        "--ahawam_intervention_mode=true. "
                        f"topology={details}"
                    )
                raise RuntimeError(
                    "FA human candidate publisher is missing; start VR teleop "
                    "with --ahawam_intervention_mode=true before the AHA client. "
                    f"topology={details}"
                )
            time.sleep(0.05)

    def assert_exclusive_real_publishers(self, args) -> None:
        """Continuously enforce that this gateway is the sole real writer."""

        if not getattr(args, "enable_motion", False):
            return
        counts = {
            topic: int(self.node.count_publishers(topic))
            for topic in (
                args.arm_command_topic,
                args.left_hand_topic,
                args.right_hand_topic,
            )
        }
        if any(count != 1 for count in counts.values()):
            raise RuntimeError(
                "FA command ownership changed after startup; refusing to "
                "publish while a direct writer can bypass the AHA-WAM gateway. "
                f"publishers_including_this_client={counts}"
            )

    def assert_runtime_command_topology(self, args) -> None:
        """Monitor real ownership and the single configured human source."""

        self.assert_exclusive_real_publishers(args)
        if not (
            getattr(args, "enable_motion", False)
            and getattr(args, "enable_operator_intervention", False)
        ):
            return
        counts = {
            topic: int(self.node.count_publishers(topic))
            for topic in (
                args.human_arm_input_topic,
                args.human_left_hand_input_topic,
                args.human_right_hand_input_topic,
            )
        }
        if any(count != 1 for count in counts.values()):
            raise RuntimeError(
                "FA human candidate ownership changed after startup; expected "
                "exactly one PICO/VR candidate publisher per topic. "
                f"publishers={counts}"
            )

    def _spin_feedback(self):
        try:
            while not self._spin_stop.is_set():
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
        except Exception as exc:
            with self._state_condition:
                self._spin_error = exc
                self._state_condition.notify_all()

    def _state(self, msg):
        values = dict(zip(msg.name, msg.position))
        received_at = time.monotonic()
        received_wall_at = time.time()
        source_at = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        missing_joints = tuple(
            name for name in LEFT_JOINTS + RIGHT_JOINTS if name not in values
        )
        with self._state_condition:
            self.last_joint_message_at = received_at
            self.last_missing_joints = missing_joints
            if not missing_joints:
                observed_clock_offset_s = received_wall_at - source_at
                if self.source_clock_offset_s is None:
                    self.source_clock_offset_s = observed_clock_offset_s
                else:
                    # The minimum observed receive-minus-source offset removes
                    # non-negative transport delay while retaining the fixed
                    # clock offset between the ROS publisher and this client.
                    self.source_clock_offset_s = min(
                        self.source_clock_offset_s, observed_clock_offset_s
                    )
                self.latest = np.asarray(
                    [values[name] for name in LEFT_JOINTS + RIGHT_JOINTS], dtype=np.float32
                )
                self.latest_at = received_at
                self.latest_source_at = source_at
                self.latest_sequence += 1
                corrected_source_delay_s = max(
                    0.0, observed_clock_offset_s - self.source_clock_offset_s
                )
                self._joint_history.append(
                    {
                        "sequence": self.latest_sequence,
                        "received_at_monotonic_s": received_at,
                        "sample_at_monotonic_s": received_at - corrected_source_delay_s,
                        "source_stamp_s": source_at,
                        "arm": self.latest.copy(),
                        "left_gripper": self.left_gripper,
                        "right_gripper": self.right_gripper,
                    }
                )
            self._state_condition.notify_all()

    def state_snapshot(
        self,
        timeout_s: float,
        max_age_s: float,
        newer_than_sequence: int | None = None,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        deadline = time.monotonic() + timeout_s
        with self._state_condition:
            while True:
                now_monotonic = time.monotonic()
                now_wall = time.time()
                receive_age_s = now_monotonic - self.latest_at
                raw_source_age_s = (
                    now_wall - self.latest_source_at
                    if self.latest_source_at > 0.0 else float("inf")
                )
                source_age_s = (
                    raw_source_age_s - self.source_clock_offset_s
                    if self.source_clock_offset_s is not None else float("inf")
                )
                sequence_is_new = (
                    newer_than_sequence is None
                    or self.latest_sequence > newer_than_sequence
                )
                source_is_fresh = -0.005 <= source_age_s <= max_age_s
                if self.latest is not None and sequence_is_new and source_is_fresh:
                    state = np.concatenate(
                        [self.latest.copy(), [self.left_gripper, self.right_gripper]]
                    ).astype(np.float32)
                    return state, {
                        "sequence": self.latest_sequence,
                        "source_stamp_s": self.latest_source_at,
                        "source_age_ms": max(0.0, source_age_s * 1000.0),
                        "raw_source_age_ms": raw_source_age_s * 1000.0,
                        "source_clock_offset_ms": self.source_clock_offset_s * 1000.0,
                        "receive_age_ms": max(0.0, receive_age_s * 1000.0),
                    }
                if self._spin_error is not None:
                    raise RuntimeError("Joint-state feedback thread failed") from self._spin_error
                remaining = deadline - now_monotonic
                if remaining <= 0:
                    break
                self._state_condition.wait(timeout=min(0.05, remaining))

            if self.latest is None:
                detail = (
                    f"last message missing {self.last_missing_joints}"
                    if self.last_joint_message_at else "no /joint_states message received"
                )
            else:
                detail = (
                    f"last complete sample source_age={source_age_s * 1000:.1f} ms "
                    f"raw_source_age={raw_source_age_s * 1000:.1f} ms "
                    f"source_clock_offset={self.source_clock_offset_s * 1000:.1f} ms "
                    f"receive_age={receive_age_s * 1000:.1f} ms "
                    f"sequence={self.latest_sequence} newer_than={newer_than_sequence}"
                )
            raise TimeoutError(f"No fresh complete 14-joint /joint_states sample: {detail}")

    def state_at_observation(
        self,
        observation_monotonic_s: float,
        timeout_s: float,
        max_age_s: float,
        max_skew_s: float,
        newer_than_sequence: int | None = None,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Return the joint sample closest to a host-monotonic image time."""
        deadline = time.monotonic() + timeout_s
        with self._state_condition:
            while True:
                candidates = [
                    sample
                    for sample in self._joint_history
                    if newer_than_sequence is None
                    or int(sample["sequence"]) > newer_than_sequence
                ]
                nearest = (
                    min(
                        candidates,
                        key=lambda sample: abs(
                            float(sample["sample_at_monotonic_s"])
                            - observation_monotonic_s
                        ),
                    )
                    if candidates
                    else None
                )
                now_monotonic = time.monotonic()
                if nearest is not None:
                    sample_at = float(nearest["sample_at_monotonic_s"])
                    signed_skew_s = sample_at - observation_monotonic_s
                    sample_age_s = now_monotonic - sample_at
                    if abs(signed_skew_s) <= max_skew_s and sample_age_s <= max_age_s:
                        state = np.concatenate(
                            [
                                nearest["arm"],
                                [nearest["left_gripper"], nearest["right_gripper"]],
                            ]
                        ).astype(np.float32)
                        now_wall = time.time()
                        raw_source_age_s = now_wall - float(nearest["source_stamp_s"])
                        source_age_s = (
                            raw_source_age_s - self.source_clock_offset_s
                            if self.source_clock_offset_s is not None
                            else float("inf")
                        )
                        return state, {
                            "sequence": int(nearest["sequence"]),
                            "source_stamp_s": float(nearest["source_stamp_s"]),
                            "source_age_ms": max(0.0, source_age_s * 1000.0),
                            "receive_age_ms": max(
                                0.0,
                                (
                                    now_monotonic
                                    - float(nearest["received_at_monotonic_s"])
                                )
                                * 1000.0,
                            ),
                            "sample_at_monotonic_s": sample_at,
                            "sample_age_ms": max(0.0, sample_age_s * 1000.0),
                            "joint_observation_skew_ms": signed_skew_s * 1000.0,
                        }

                    # A future feedback sample can only improve the match when
                    # every available sample still precedes the image time.
                    if sample_at >= observation_monotonic_s:
                        break
                if self._spin_error is not None:
                    raise RuntimeError("Joint-state feedback thread failed") from self._spin_error
                remaining = deadline - now_monotonic
                if remaining <= 0:
                    break
                self._state_condition.wait(timeout=min(0.01, remaining))

            if nearest is None:
                detail = "no eligible complete joint sample in history"
            else:
                sample_at = float(nearest["sample_at_monotonic_s"])
                detail = (
                    f"nearest sequence={nearest['sequence']} "
                    f"skew={(sample_at - observation_monotonic_s) * 1000.0:.1f} ms "
                    f"age={(time.monotonic() - sample_at) * 1000.0:.1f} ms"
                )
            raise TimeoutError(f"No joint state matched the camera observation: {detail}")

    def state(self, timeout_s: float, max_age_s: float) -> np.ndarray:
        state, _metadata = self.state_snapshot(timeout_s, max_age_s)
        return state

    def publish(
        self,
        action: np.ndarray,
        args,
        *,
        source: str = "model",
    ) -> FaCommandReceipt:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("FA publish action must be a finite 16D vector")
        self.assert_exclusive_real_publishers(args)
        msg = self.MinSnapTarget()
        msg.left_arm_target_rad = action[:7].astype(float).tolist()
        msg.right_arm_target_rad = action[7:14].astype(float).tolist()
        msg.expected_duration_s = 1.0 / args.hz
        msg.max_velocity_rad_s = args.max_velocity_rad_s
        msg.max_acceleration_rad_s2 = args.max_acceleration_rad_s2
        arm_published = False
        left_complete = False
        right_complete = False
        left_changed = False
        right_changed = False
        try:
            self.arm_pub.publish(msg)
            arm_published = True
            for side, value, old, pub in (
                ("left", action[14], self.left_gripper, self.left_hand_pub),
                ("right", action[15], self.right_gripper, self.right_hand_pub),
            ):
                new = binarize_gripper(value)
                changed = new != old
                if changed:
                    hand = self.Int32()
                    hand.data = 20 if new else 21
                    pub.publish(hand)
                if side == "left":
                    self.left_gripper = new
                    left_complete = True
                    left_changed = changed
                else:
                    self.right_gripper = new
                    right_complete = True
                    right_changed = changed
        except BaseException as exc:
            receipt = FaCommandReceipt(
                source=str(source),
                sent_at_monotonic_s=time.monotonic(),
                arm_published=arm_published,
                left_hand_published=left_complete,
                right_hand_published=right_complete,
                left_hand_changed=left_changed,
                right_hand_changed=right_changed,
            )
            if arm_published or left_complete or right_complete:
                raise PartialCommandPublishError(
                    "FA command publication failed after a partial ROS write",
                    receipt,
                ) from exc
            raise
        return FaCommandReceipt(
            source=str(source),
            sent_at_monotonic_s=time.monotonic(),
            arm_published=True,
            left_hand_published=True,
            right_hand_published=True,
            left_hand_changed=left_changed,
            right_hand_changed=right_changed,
        )

    def close(self):
        self._spin_stop.set()
        with self._state_condition:
            self._state_condition.notify_all()
        with self._human_condition:
            self._human_condition.notify_all()
        self._spin_thread.join(timeout=1.0)
        if self._spin_thread.is_alive():
            print("WARNING: joint-state feedback thread did not stop", flush=True)
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def open_cameras(
    settings_path: Path,
    attempts: int = 5,
    retry_delay_s: float = 3.0,
):
    """Open the inference camera set with bounded, fully-cleaned retries."""
    from beavr.lerobot.common.robot_devices.cameras.manual_controls import load_camera_settings
    from beavr.lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs

    last_error = None
    for attempt in range(1, attempts + 1):
        cameras = {}
        try:
            configs = load_camera_settings(settings_path)
            if set(configs) != set(CAMERA_KEYS):
                raise ValueError(
                    f"Camera config mismatch: expected {CAMERA_KEYS}, got {tuple(configs)}"
                )
            cameras = make_cameras_from_configs(configs)
            for key in CAMERA_KEYS:
                cameras[key].connect(warmup=True)
            if attempt > 1:
                print(f"Opened all FA cameras on attempt {attempt}/{attempts}")
            return cameras
        except Exception as exc:
            last_error = exc
            print(
                f"Camera startup attempt {attempt}/{attempts} failed: {exc}",
                flush=True,
            )
            # connect() cleans the camera that failed. Also release cameras
            # which connected earlier in this attempt before rebuilding all
            # Pipeline objects, so retries never overlap native SDK threads.
            for camera in cameras.values():
                try:
                    camera.disconnect()
                except Exception as cleanup_exc:
                    print(f"Camera cleanup failed: {cleanup_exc}", flush=True)
            cameras.clear()
            gc.collect()
            if attempt < attempts:
                time.sleep(retry_delay_s)
    raise RuntimeError(
        f"Could not open all FA cameras after {attempts} attempts: {last_error}"
    )


def read_images(cameras, max_age_ms: int, newer_than_sequences=None):
    result = {}
    metadata = {}
    newer_than_sequences = newer_than_sequences or {}
    for key in CAMERA_KEYS:
        rgb, _depth, frame_metadata = cameras[key].read_latest_rgbd_with_metadata(
            max_age_ms,
            newer_than_sequence=newer_than_sequences.get(key),
        )
        expected = (720, 1280, 3) if key == "head" else (530, 848, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError(f"{key} produced {rgb.shape}/{rgb.dtype}; expected {expected}/uint8")
        result[key] = np.ascontiguousarray(rgb)
        metadata[key] = frame_metadata
    return result, metadata


def camera_observation_time(camera_metadata, max_camera_skew_ms: float):
    captured_at = {
        key: float(camera_metadata[key]["captured_at_monotonic_s"])
        for key in CAMERA_KEYS
    }
    observation_time = float(np.median(list(captured_at.values())))
    camera_skew_ms = (max(captured_at.values()) - min(captured_at.values())) * 1000.0
    if camera_skew_ms > max_camera_skew_ms:
        raise RuntimeError(
            f"Camera frame skew is {camera_skew_ms:.1f} ms; "
            f"limit is {max_camera_skew_ms:.1f} ms"
        )
    return observation_time, camera_skew_ms


def image_state_sync_metadata(
    camera_metadata,
    state_metadata,
    observation_time: float,
    max_image_state_skew_ms: float,
):
    joint_time = float(state_metadata["sample_at_monotonic_s"])
    signed_skews_ms = {
        key: (joint_time - float(camera_metadata[key]["captured_at_monotonic_s"]))
        * 1000.0
        for key in CAMERA_KEYS
    }
    max_abs_skew_ms = max(abs(value) for value in signed_skews_ms.values())
    if max_abs_skew_ms > max_image_state_skew_ms:
        raise RuntimeError(
            f"Camera/joint skew is {max_abs_skew_ms:.1f} ms; "
            f"limit is {max_image_state_skew_ms:.1f} ms"
        )
    return {
        "observation_time_monotonic_s": observation_time,
        "joint_sample_time_monotonic_s": joint_time,
        "joint_observation_skew_ms": (joint_time - observation_time) * 1000.0,
        "max_abs_camera_joint_skew_ms": max_abs_skew_ms,
        "camera_joint_skew_ms": signed_skews_ms,
    }


class FaObservationStream:
    """Single producer shared by policy inference and continuous recording."""

    def __init__(
        self,
        cameras,
        ros: FaRos,
        args,
        command_history: SentCommandHistory,
        on_observation: Optional[Callable[[FaObservation], None]] = None,
    ) -> None:
        self.cameras = cameras
        self.ros = ros
        self.args = args
        self.command_history = command_history
        self.on_observation = on_observation
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[FaObservation] = None
        self._error: Optional[BaseException] = None
        self._sequence = 0

    @property
    def captured_frames(self) -> int:
        with self._condition:
            return self._sequence

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("FA observation stream was already started")
        self._thread = threading.Thread(
            target=self._run,
            name="ahawam-fa-observation-stream",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        last_camera_sequences = {key: 0 for key in CAMERA_KEYS}
        last_joint_sequence = 0
        period_s = 1.0 / float(self.args.record_fps)
        try:
            while not self._stop_event.is_set():
                started = time.monotonic()
                self.ros.assert_runtime_command_topology(self.args)
                images, camera_metadata = read_images(
                    self.cameras,
                    self.args.max_camera_age_ms,
                    newer_than_sequences=last_camera_sequences,
                )
                observation_time, camera_skew_ms = camera_observation_time(
                    camera_metadata,
                    self.args.max_camera_skew_ms,
                )
                state, state_metadata = self.ros.state_at_observation(
                    observation_time,
                    self.args.state_timeout_s,
                    self.args.max_state_age_ms / 1000.0,
                    self.args.max_image_state_skew_ms / 1000.0,
                    newer_than_sequence=last_joint_sequence,
                )
                sync_metadata = image_state_sync_metadata(
                    camera_metadata,
                    state_metadata,
                    observation_time,
                    self.args.max_image_state_skew_ms,
                )
                sync_metadata["camera_skew_ms"] = camera_skew_ms
                last_joint_sequence = int(state_metadata["sequence"])
                last_camera_sequences = {
                    key: int(camera_metadata[key]["sequence"])
                    for key in CAMERA_KEYS
                }
                frozen_images: dict[str, np.ndarray] = {}
                for key, value in images.items():
                    image = np.ascontiguousarray(value)
                    image.setflags(write=False)
                    frozen_images[key] = image
                frozen_state = np.ascontiguousarray(state, dtype=np.float32)
                frozen_state.setflags(write=False)
                self._sequence += 1
                observation = FaObservation(
                    sequence=self._sequence,
                    captured_at_monotonic_s=observation_time,
                    captured_at_wall_s=time.time(),
                    images=frozen_images,
                    state=frozen_state,
                    camera_metadata=camera_metadata,
                    state_metadata=state_metadata,
                    sync_metadata=sync_metadata,
                    command=self.command_history.snapshot_at(observation_time),
                )
                if self.on_observation is not None:
                    self.on_observation(observation)
                with self._condition:
                    self._latest = observation
                    self._condition.notify_all()
                remaining = period_s - (time.monotonic() - started)
                if remaining > 0.0:
                    self._stop_event.wait(remaining)
        except BaseException as exc:
            if not self._stop_event.is_set():
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

    def wait_next(
        self, newer_than_sequence: int, timeout_s: float
    ) -> FaObservation:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                if self._error is not None:
                    raise ObservationStreamError(
                        "FA observation stream failed"
                    ) from self._error
                if (
                    self._latest is not None
                    and self._latest.sequence > int(newer_than_sequence)
                ):
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise ObservationStreamError(
                        "timed out waiting for the next FA observation"
                    )
                self._condition.wait(timeout=min(0.05, remaining))

    def raise_if_failed(self) -> None:
        with self._condition:
            if self._error is not None:
                raise ObservationStreamError(
                    "FA observation stream failed"
                ) from self._error

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.args.state_timeout_s + 1.0))
            if self._thread.is_alive():
                raise RuntimeError("FA observation stream did not stop")


def compose_training_mosaic(images):
    """Run the exact FA training transform before network transport."""
    import torch
    import torchvision.transforms.functional as TF

    def center_crop_resize(frames, height, width):
        from math import gcd
        source_h, source_w = frames.shape[-2:]
        divisor = gcd(height, width)
        ratio_h, ratio_w = height // divisor, width // divisor
        units = min(source_h // ratio_h, source_w // ratio_w)
        crop_h, crop_w = ratio_h * units, ratio_w * units
        top, left = (source_h - crop_h) // 2, (source_w - crop_w) // 2
        return TF.resize(
            frames[..., top:top + crop_h, left:left + crop_w],
            [height, width], interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

    views = []
    for key in CAMERA_KEYS:
        tensor = (
            torch.from_numpy(np.ascontiguousarray(images[key]))
            .permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        views.append(center_crop_resize(tensor, 256, 320))
    head = center_crop_resize(views[0], 256, 320)
    left = center_crop_resize(views[1], 128, 160)
    right = center_crop_resize(views[2], 128, 160)
    mosaic = torch.cat((head, torch.cat((left, right), dim=-1)), dim=-2)[0]
    return mosaic.permute(1, 2, 0).contiguous().numpy()


def validate_action(chunk) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 16 or not np.isfinite(chunk).all():
        raise ValueError(f"Invalid FA action chunk: {chunk.shape}")
    return chunk


def format_gripper_output_log(chunk: np.ndarray, execute: int, chunk_index: int) -> str:
    executed = chunk[:execute]
    left_raw = executed[:, 14].astype(float).tolist()
    right_raw = executed[:, 15].astype(float).tolist()
    return "chunk_gripper_output=" + json.dumps(
        {
            "chunk": chunk_index,
            "executed_steps": execute,
            "close_threshold": FA_GRIPPER_CLOSE_THRESHOLD,
            "action_14_left_raw": left_raw,
            "action_15_right_raw": right_raw,
            "action_14_left_binary": [binarize_gripper(value) for value in left_raw],
            "action_15_right_binary": [binarize_gripper(value) for value in right_raw],
        },
        separators=(",", ":"),
    )


def limit_fa_target(
    target: np.ndarray,
    measured: np.ndarray,
    previous_command: Optional[np.ndarray],
    max_joint_delta_rad: float,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    measured = np.asarray(measured, dtype=np.float32)
    if target.shape != (16,) or measured.shape != (16,):
        raise ValueError("FA target and measured state must both be 16D")
    if not np.isfinite(target).all() or not np.isfinite(measured).all():
        raise ValueError("FA target/state contains NaN or Inf")
    reference = (
        measured
        if previous_command is None
        else np.asarray(previous_command, dtype=np.float32)
    )
    if reference.shape != (16,) or not np.isfinite(reference).all():
        raise ValueError("previous FA command must be a finite 16D vector")
    result = target.copy()
    delta = np.clip(
        target[:14] - reference[:14],
        -float(max_joint_delta_rad),
        float(max_joint_delta_rad),
    )
    result[:14] = reference[:14] + delta
    result[14:] = [binarize_gripper(value) for value in result[14:]]
    return result


def recovery_takeover_reason(reply: dict[str, object]) -> Optional[str]:
    """Validate and consume the shared AHA recovery-controller contract."""

    recovery = reply.get("recovery_controller")
    if recovery is None:
        return None
    if not isinstance(recovery, dict):
        raise ValueError("recovery_controller must be a mapping")
    intervention_required = recovery.get("operator_intervention_required", False)
    if not isinstance(intervention_required, bool):
        raise ValueError("operator_intervention_required must be boolean")
    if not intervention_required:
        return None
    mode = str(recovery.get("mode", "diagnostic")).lower()
    if mode not in {"active", "diagnostic"}:
        raise ValueError(f"invalid recovery mode: {mode!r}")
    if mode != "active":
        return None
    return str(recovery.get("reason") or "policy_requested_operator_intervention")


def poll_operator_controls(
    listener: TerminalKeyListener,
    operator: OperatorIntervention,
    *,
    chunk: Optional[int],
    step: Optional[int],
) -> None:
    for key in listener.poll():
        normalized = key.lower()
        if normalized == "i":
            operator.request_takeover(
                "operator_pressed_i",
                automatic=False,
                chunk=chunk,
                step=step,
            )
            raise TakeoverRequested("operator_pressed_i")
        outcome = {
            "s": EpisodeOutcome.SUCCESS,
            "f": EpisodeOutcome.FAILURE,
            "q": EpisodeOutcome.ABORTED,
            "\x1b": EpisodeOutcome.ABORTED,
        }.get(normalized)
        if outcome is not None:
            reason = {
                EpisodeOutcome.SUCCESS: "operator_labeled_success",
                EpisodeOutcome.FAILURE: "operator_labeled_failure",
                EpisodeOutcome.ABORTED: "operator_aborted",
            }[outcome]
            operator.finish(outcome, reason, "operator")
            raise EpisodeTerminated(outcome, reason)


def send_authorized_command(
    *,
    ros: FaRos,
    args,
    operator: OperatorIntervention,
    history: SentCommandHistory,
    command: np.ndarray,
    source: str,
) -> Optional[SentCommand]:
    """Serialize ownership, ROS publication, and successful-send registry."""

    with operator.command_authority(source):
        if not args.enable_motion:
            return None
        receipt = ros.publish(command, args, source=source)
        return history.mark_sent(command, source, receipt)


def safe_measured_hold(
    *,
    ros: FaRos,
    args,
    operator: OperatorIntervention,
    history: SentCommandHistory,
) -> np.ndarray:
    """Publish one fresh measured arm hold while preserving gripper latches."""

    measured, _metadata = ros.state_snapshot(
        args.state_timeout_s,
        args.max_state_age_ms / 1000.0,
    )
    hold = np.asarray(measured, dtype=np.float32).copy()
    send_authorized_command(
        ros=ros,
        args=args,
        operator=operator,
        history=history,
        command=hold,
        source="hold",
    )
    return hold


def wait_until_deadline_with_controls(
    deadline: float,
    *,
    listener: TerminalKeyListener,
    operator: OperatorIntervention,
    observation_stream: FaObservationStream,
    chunk: Optional[int],
    step: Optional[int],
    recorder=None,
) -> None:
    while True:
        poll_operator_controls(
            listener, operator, chunk=chunk, step=step
        )
        observation_stream.raise_if_failed()
        if recorder is not None:
            recorder.raise_if_failed()
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(0.02, remaining))


def _policy_exchange(
    policy: PolicyConnection,
    observation: FaObservation,
    args,
    chunk_index: int,
) -> dict[str, object]:
    images = {
        key: np.asarray(value).copy()
        for key, value in observation.images.items()
    }
    wire_images = (
        images
        if args.image_transport == "raw_cameras"
        else {"front": compose_training_mosaic(images)}
    )
    policy.request(
        {
            "type": "image",
            "images": wire_images,
            "state": np.asarray(observation.state).copy(),
            "timestamp": observation.captured_at_wall_s,
            "frame_sequence": chunk_index,
            "observation_id": observation.sequence,
            "expect_reply": True,
        }
    )
    return policy.request(
        {
            "type": "action_request",
            "after_frame_sequence": chunk_index - 1,
            "require_paired_observation": True,
        }
    )


def _print_observation_feedback(
    observation: FaObservation,
    previous_target: Optional[np.ndarray],
    chunk_index: int,
) -> None:
    state = np.asarray(observation.state)
    target_error = (
        None if previous_target is None else state[:14] - previous_target[:14]
    )
    camera_metadata = observation.camera_metadata
    state_metadata = observation.state_metadata
    sync_metadata = observation.sync_metadata
    print(
        "chunk_feedback="
        + json.dumps(
            {
                "chunk": chunk_index,
                "observation_sequence": observation.sequence,
                "joint_state_sequence": int(state_metadata["sequence"]),
                "source_stamp_s": state_metadata["source_stamp_s"],
                "source_age_ms": round(
                    float(state_metadata["source_age_ms"]), 3
                ),
                "receive_age_ms": round(
                    float(state_metadata["receive_age_ms"]), 3
                ),
                "joint_sample_age_ms": round(
                    float(state_metadata["sample_age_ms"]), 3
                ),
                "joint_observation_skew_ms": round(
                    float(sync_metadata["joint_observation_skew_ms"]), 3
                ),
                "max_abs_camera_joint_skew_ms": round(
                    float(sync_metadata["max_abs_camera_joint_skew_ms"]), 3
                ),
                "camera_skew_ms": round(
                    float(sync_metadata["camera_skew_ms"]), 3
                ),
                "camera_frames": {
                    key: {
                        "sequence": int(camera_metadata[key]["sequence"]),
                        "age_ms": round(
                            float(camera_metadata[key]["age_ms"]), 3
                        ),
                        "joint_skew_ms": round(
                            float(sync_metadata["camera_joint_skew_ms"][key]),
                            3,
                        ),
                    }
                    for key in CAMERA_KEYS
                },
                "actual_arm_rad": state[:14].astype(float).tolist(),
                "previous_target_arm_rad": (
                    None
                    if previous_target is None
                    else previous_target[:14].astype(float).tolist()
                ),
                "actual_minus_previous_target_rad": (
                    None
                    if target_error is None
                    else target_error.astype(float).tolist()
                ),
                "max_abs_target_error_rad": (
                    None
                    if target_error is None
                    else float(np.abs(target_error).max(initial=0.0))
                ),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def run_human_control(
    *,
    ros: FaRos,
    args,
    operator: OperatorIntervention,
    listener: TerminalKeyListener,
    observation_stream: FaObservationStream,
    history: SentCommandHistory,
    initial_hold: np.ndarray,
    recorder=None,
) -> np.ndarray:
    """Forward only post-handoff VR proposals through the sole real gateway."""

    proposal_sequence = ros.begin_human_handoff()
    operator.enter_human()
    print(
        "[INTERVENTION] HUMAN active; move PICO/VR to send a fresh candidate. "
        "Press s=success, f=failure, q/Esc=abort.",
        flush=True,
    )
    previous = np.asarray(initial_hold, dtype=np.float32).copy()
    started = time.monotonic()
    step = 0
    while True:
        poll_operator_controls(
            listener, operator, chunk=None, step=step
        )
        observation_stream.raise_if_failed()
        if recorder is not None:
            recorder.raise_if_failed()
        if (
            args.human_control_timeout_s > 0.0
            and time.monotonic() - started >= args.human_control_timeout_s
        ):
            operator.finish(
                EpisodeOutcome.UNLABELED,
                "human_control_timeout",
                "runtime",
            )
            raise EpisodeTerminated(
                EpisodeOutcome.UNLABELED, "human_control_timeout"
            )
        try:
            proposal, metadata = ros.wait_human_action(
                newer_than_sequence=proposal_sequence,
                timeout_s=0.02,
                max_age_s=args.max_human_command_age_ms / 1000.0,
                fallback_arm=previous[:14],
            )
        except TimeoutError:
            continue
        proposal_sequence = int(metadata["sequence"])
        measured = ros.state(
            args.state_timeout_s, args.max_state_age_ms / 1000.0
        )
        limited = limit_fa_target(
            proposal,
            measured,
            previous,
            args.max_joint_delta_rad,
        )
        sent = send_authorized_command(
            ros=ros,
            args=args,
            operator=operator,
            history=history,
            command=limited,
            source="human",
        )
        if recorder is not None and sent is not None:
            recorder.note_successful_command("human")
        previous = limited
        step += 1
        print(
            f"[HUMAN] step={step} proposal_sequence={proposal_sequence} "
            f"age_ms={float(metadata['age_ms']):.1f} motion={args.enable_motion}",
            flush=True,
        )


def run(args):
    """Run exactly one FA rollout and, when configured, one recorded episode."""

    if args.enable_motion and args.confirm_motion != "RUN":
        raise SystemExit(
            "Motion requires both --enable-motion and --confirm-motion RUN"
        )
    if args.require_outcome_label and not args.enable_operator_intervention:
        raise SystemExit(
            "--require-outcome-label requires --enable-operator-intervention"
        )

    cameras = {}
    ros: Optional[FaRos] = None
    policy: Optional[PolicyConnection] = None
    recorder = None
    observation_stream: Optional[FaObservationStream] = None
    executor: Optional[ThreadPoolExecutor] = None
    operator = OperatorIntervention(enabled=args.enable_operator_intervention)
    listener = TerminalKeyListener(enabled=args.enable_operator_intervention)
    listener_entered = False
    history = SentCommandHistory()
    episode_reason = "startup_incomplete"
    pending_error: Optional[BaseException] = None
    cleanup_errors: list[BaseException] = []
    recorded_partial_failures: set[int] = set()

    def close_policy() -> None:
        nonlocal policy
        current, policy = policy, None
        if current is not None:
            current.close()

    def record_event(event_type: str, **payload: object) -> None:
        if recorder is not None:
            recorder.record_event(event_type, **payload)

    def record_partial_failure(
        error: BaseException, **payload: object
    ) -> None:
        """Record one partial multi-topic write without masking its root error."""

        if not isinstance(error, PartialCommandPublishError):
            return
        identity = id(error)
        if identity in recorded_partial_failures:
            return
        recorded_partial_failures.add(identity)
        if recorder is None:
            return
        try:
            receipt = error.receipt.to_dict()
            recorder.note_partial_command(receipt)
            recorder.record_event(
                "partial_command_publish",
                receipt=receipt,
                fatal=True,
                **payload,
            )
        except BaseException as recorder_exc:
            cleanup_errors.append(recorder_exc)

    def establish_hold(reason: str) -> np.ndarray:
        if ros is None:
            raise RuntimeError("cannot establish FA hold before ROS startup")
        hold = safe_measured_hold(
            ros=ros,
            args=args,
            operator=operator,
            history=history,
        )
        record_event(
            "safe_hold",
            reason=reason,
            state=hold.astype(float).tolist(),
            motion_enabled=args.enable_motion,
        )
        return hold

    def enter_human_after_hold(reason: str) -> None:
        nonlocal episode_reason
        episode_reason = reason
        close_policy()
        hold = establish_hold(reason)
        record_event(
            "takeover",
            reason=reason,
            operator=operator.snapshot().to_dict(),
        )
        assert ros is not None and observation_stream is not None
        try:
            run_human_control(
                ros=ros,
                args=args,
                operator=operator,
                listener=listener,
                observation_stream=observation_stream,
                history=history,
                initial_hold=hold,
                recorder=recorder,
            )
        except EpisodeTerminated as terminal:
            episode_reason = terminal.reason
            establish_hold(f"episode_{terminal.outcome.value}")
            print(
                f"[OUTCOME] {terminal.outcome.value} reason={terminal.reason}",
                flush=True,
            )

    try:
        listener.__enter__()
        listener_entered = True
        if args.enable_operator_intervention and not listener.available:
            raise RuntimeError(
                "--enable-operator-intervention requires an interactive TTY; "
                "refusing to start without i/s/f/q controls"
            )

        # No command or camera is opened before the operator input contract is
        # known to be usable.  Motion remains double-gated by the flags above.
        ros = FaRos(args)
        ros.validate_command_topology(args, args.command_topology_timeout_s)
        cameras = open_cameras(
            args.camera_settings,
            attempts=args.camera_open_retries,
            retry_delay_s=args.camera_retry_delay_s,
        )

        if args.record_dir is not None:
            from beavr.scripts.ahawam_fa_episode_recorder import FaEpisodeRecorder

            recorder = FaEpisodeRecorder(
                args.record_dir,
                fps=args.record_fps,
                segment_sec=args.record_segment_sec,
                queue_size=args.record_queue_size,
                min_free_gb=args.record_min_free_gb,
                config={
                    "platform": "FA",
                    "prompt": args.prompt,
                    "enable_motion": args.enable_motion,
                    "arm_command_topic": args.arm_command_topic,
                    "left_hand_topic": args.left_hand_topic,
                    "right_hand_topic": args.right_hand_topic,
                    "human_arm_input_topic": args.human_arm_input_topic,
                    "human_left_hand_input_topic": args.human_left_hand_input_topic,
                    "human_right_hand_input_topic": args.human_right_hand_input_topic,
                    "camera_settings": str(args.camera_settings),
                    "record_fps": args.record_fps,
                    "state_units": "radians_plus_binary_gripper_latches",
                },
            )

        def submit_observation(observation: FaObservation) -> None:
            if recorder is None:
                return
            recorder.submit_observation(
                images=observation.images,
                state=np.asarray(observation.state),
                camera_metadata=observation.camera_metadata,
                state_metadata=observation.state_metadata,
                sync_metadata=observation.sync_metadata,
                sequence=observation.sequence,
                captured_at=observation.captured_at_monotonic_s,
                sent_command=(
                    None
                    if observation.command is None
                    else observation.command.to_dict()
                ),
                intervention_snapshot=operator.snapshot().to_dict(),
            )

        observation_stream = FaObservationStream(
            cameras,
            ros,
            args,
            history,
            on_observation=submit_observation,
        )
        observation_stream.start()
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ahawam-fa-policy"
        )

        # Start recording before connecting to the brain.  A connection/startup
        # failure is a model-side failure: after a measured hold the operator
        # can complete the same continuously recorded episode through VR.
        try:
            policy = PolicyConnection(
                args.host,
                args.port,
                args.socket_timeout_s,
                connect_timeout=args.connect_timeout_s,
            )
        except Exception as exc:
            raise ModelFailure(f"policy connection failed: {exc}") from exc

        assert policy is not None
        reset_future = executor.submit(
            policy.request, {"type": "reset", "instruction": args.prompt}
        )
        while not reset_future.done():
            wait_until_deadline_with_controls(
                time.monotonic() + 0.02,
                listener=listener,
                operator=operator,
                observation_stream=observation_stream,
                chunk=None,
                step=None,
                recorder=recorder,
            )
        try:
            print(reset_future.result())
        except Exception as exc:
            raise ModelFailure(f"policy reset failed: {exc}") from exc

        last_command: Optional[np.ndarray] = None
        last_observation_sequence = 0
        for chunk_index in range(args.max_chunks):
            observation = observation_stream.wait_next(
                last_observation_sequence, args.observation_timeout_s
            )
            last_observation_sequence = observation.sequence
            _print_observation_feedback(
                observation, last_command, chunk_index
            )
            t0 = time.perf_counter()
            assert policy is not None
            future: Future[dict[str, object]] = executor.submit(
                _policy_exchange,
                policy,
                observation,
                args,
                chunk_index,
            )
            while not future.done():
                wait_until_deadline_with_controls(
                    time.monotonic() + 0.02,
                    listener=listener,
                    operator=operator,
                    observation_stream=observation_stream,
                    chunk=chunk_index,
                    step=None,
                    recorder=recorder,
                )
            try:
                reply = future.result()
                takeover_reason = recovery_takeover_reason(reply)
                if takeover_reason is not None:
                    operator.request_takeover(
                        takeover_reason,
                        automatic=True,
                        chunk=chunk_index,
                        step=None,
                    )
                    raise TakeoverRequested(takeover_reason)
                chunk = validate_action(reply.get("action_chunk"))
                first_delta = np.abs(
                    chunk[0, :14] - np.asarray(observation.state)[:14]
                )
                if (
                    first_delta.max(initial=0.0)
                    > args.max_first_target_delta_rad
                ):
                    raise ValueError(
                        "first target jump too large: "
                        f"{first_delta.max():.4f} rad"
                    )
            except (TakeoverRequested, EpisodeTerminated):
                raise
            except Exception as exc:
                raise ModelFailure(
                    f"policy action failed at chunk {chunk_index}: {exc}"
                ) from exc

            execute = min(args.execute_actions, len(chunk))
            print(
                f"chunk={chunk_index} model_ms={reply.get('model_latency_ms')} "
                f"rtt_ms={(time.perf_counter()-t0)*1000:.1f} "
                f"execute={execute} motion={args.enable_motion}",
                flush=True,
            )
            print(
                format_gripper_output_log(chunk, execute, chunk_index),
                flush=True,
            )
            for step, target in enumerate(chunk[:execute]):
                poll_operator_controls(
                    listener,
                    operator,
                    chunk=chunk_index,
                    step=step,
                )
                observation_stream.raise_if_failed()
                if recorder is not None:
                    recorder.raise_if_failed()
                measured = ros.state(
                    args.state_timeout_s, args.max_state_age_ms / 1000.0
                )
                limited = limit_fa_target(
                    target,
                    measured,
                    last_command,
                    args.max_joint_delta_rad,
                )
                try:
                    sent = send_authorized_command(
                        ros=ros,
                        args=args,
                        operator=operator,
                        history=history,
                        command=limited,
                        source="model",
                    )
                    if recorder is not None and sent is not None:
                        recorder.note_successful_command("model")
                except PartialCommandPublishError as exc:
                    record_partial_failure(
                        exc,
                        chunk=chunk_index,
                        step=step,
                    )
                    raise
                last_command = limited
                wait_until_deadline_with_controls(
                    time.monotonic() + 1.0 / args.hz,
                    listener=listener,
                    operator=operator,
                    observation_stream=observation_stream,
                    chunk=chunk_index,
                    step=step,
                    recorder=recorder,
                )

        episode_reason = "max_chunks_completed"
        establish_hold(episode_reason)
        if args.require_outcome_label:
            print("[LABEL] press s=success, f=failure, q/Esc=abort", flush=True)
            label_started = time.monotonic()
            while True:
                poll_operator_controls(
                    listener, operator, chunk=None, step=None
                )
                observation_stream.raise_if_failed()
                if recorder is not None:
                    recorder.raise_if_failed()
                if (
                    args.outcome_label_timeout_s > 0.0
                    and time.monotonic() - label_started
                    >= args.outcome_label_timeout_s
                ):
                    operator.finish(
                        EpisodeOutcome.UNLABELED,
                        "outcome_label_timeout",
                        "runtime",
                    )
                    episode_reason = "outcome_label_timeout"
                    break
                time.sleep(0.02)
        else:
            operator.finish(
                EpisodeOutcome.UNLABELED, episode_reason, "runtime"
            )
    except TakeoverRequested as exc:
        if operator.mode is not ControlMode.SAFE_HOLD:
            operator.request_takeover(
                str(exc), automatic=False, chunk=None, step=None
            )
        try:
            enter_human_after_hold(str(exc))
        except BaseException as handoff_exc:
            record_partial_failure(handoff_exc, phase="human_handoff")
            pending_error = handoff_exc
            episode_reason = f"{type(handoff_exc).__name__}: {handoff_exc}"
            if operator.mode is not ControlMode.FINALIZED:
                operator.finish(EpisodeOutcome.ERROR, episode_reason, "runtime")
    except ModelFailure as exc:
        if not args.enable_operator_intervention:
            pending_error = exc
            episode_reason = str(exc)
        else:
            operator.request_takeover(
                str(exc), automatic=True, chunk=None, step=None
            )
            try:
                enter_human_after_hold(str(exc))
            except BaseException as handoff_exc:
                record_partial_failure(handoff_exc, phase="human_handoff")
                pending_error = handoff_exc
                episode_reason = f"{type(handoff_exc).__name__}: {handoff_exc}"
                if operator.mode is not ControlMode.FINALIZED:
                    operator.finish(
                        EpisodeOutcome.ERROR, episode_reason, "runtime"
                    )
    except EpisodeTerminated as terminal:
        episode_reason = terminal.reason
        close_policy()
        if ros is not None:
            try:
                establish_hold(f"episode_{terminal.outcome.value}")
            except BaseException as hold_exc:
                record_partial_failure(hold_exc, phase="terminal_hold")
                # The operator's label remains available as reported history,
                # but a failed terminal hold makes the committed episode an
                # effective ERROR even if the final shutdown retry succeeds.
                pending_error = hold_exc
                episode_reason = f"{type(hold_exc).__name__}: {hold_exc}"
        print(
            f"[OUTCOME] {terminal.outcome.value} reason={terminal.reason}",
            flush=True,
        )
    except KeyboardInterrupt:
        episode_reason = "keyboard_interrupt"
        close_policy()
        if operator.mode in {ControlMode.AUTO, ControlMode.HUMAN}:
            operator.request_takeover(
                episode_reason, automatic=False, chunk=None, step=None
            )
        if ros is not None:
            try:
                establish_hold(episode_reason)
            except Exception as hold_exc:
                record_partial_failure(hold_exc, phase="keyboard_interrupt_hold")
                cleanup_errors.append(hold_exc)
        if operator.mode is not ControlMode.FINALIZED:
            operator.finish(
                EpisodeOutcome.ABORTED, episode_reason, "operator"
            )
    except BaseException as exc:
        pending_error = exc
        episode_reason = f"{type(exc).__name__}: {exc}"
        close_policy()
        record_partial_failure(exc, phase="runtime")
        if operator.mode in {ControlMode.AUTO, ControlMode.HUMAN}:
            operator.request_takeover(
                episode_reason, automatic=True, chunk=None, step=None
            )
        if ros is not None:
            try:
                establish_hold("runtime_error")
            except Exception as hold_exc:
                record_partial_failure(hold_exc, phase="runtime_error_hold")
                cleanup_errors.append(hold_exc)
        if operator.mode is not ControlMode.FINALIZED:
            operator.finish(EpisodeOutcome.ERROR, episode_reason, "runtime")
    finally:
        close_policy()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if ros is not None and args.enable_motion:
            try:
                if operator.mode is ControlMode.HUMAN:
                    operator.request_takeover(
                        "client_shutdown",
                        automatic=True,
                        chunk=None,
                        step=None,
                    )
                # Always retry one final fresh measured hold.  A previous hold
                # may have been followed by a human/model command, and a
                # partial ROS publish is deliberately not present in the
                # successful-command history.
                establish_hold("client_shutdown")
            except Exception as exc:
                record_partial_failure(exc, phase="client_shutdown_hold")
                cleanup_errors.append(exc)
        if observation_stream is not None:
            try:
                observation_stream.stop()
                # Re-check after join so a camera/state/topology failure that
                # raced with an operator terminal key cannot be committed as
                # success/failure.
                observation_stream.raise_if_failed()
            except Exception as exc:
                cleanup_errors.append(exc)
        for key, camera in cameras.items():
            try:
                camera.disconnect()
            except Exception as exc:
                cleanup_errors.append(exc)
                print(f"Camera cleanup failed for {key}: {exc}", flush=True)
        if ros is not None:
            try:
                ros.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if listener_entered:
            try:
                listener.__exit__(None, None, None)
            except Exception as exc:
                cleanup_errors.append(exc)

        if operator.mode is not ControlMode.FINALIZED:
            effective = (
                EpisodeOutcome.ERROR
                if pending_error is not None or cleanup_errors
                else EpisodeOutcome.UNLABELED
            )
            operator.finish(effective, episode_reason, "runtime")

        if recorder is not None:
            effective_outcome = (
                EpisodeOutcome.ERROR
                if pending_error is not None or cleanup_errors
                else operator.outcome
            )
            effective_reason = episode_reason
            if pending_error is not None:
                effective_reason = (
                    f"{type(pending_error).__name__}: {pending_error}"
                )
            elif cleanup_errors:
                first_cleanup_error = cleanup_errors[0]
                effective_reason = (
                    f"{type(first_cleanup_error).__name__}: "
                    f"{first_cleanup_error}"
                )
            events = [event.to_dict() for event in operator.events()]
            intervention_summary = operator.snapshot().to_dict()
            reported_reason = (
                intervention_summary.get("final_reason") or episode_reason
            )
            intervention_summary.update(
                {
                    "assisted": any(
                        event["event"] == "human_entered" for event in events
                    ),
                    "events": events,
                    "reported_outcome": operator.outcome.value,
                    "reported_reason": reported_reason,
                    "effective_outcome": effective_outcome.value,
                    "effective_reason": effective_reason,
                    "cleanup_errors": [
                        f"{type(item).__name__}: {item}"
                        for item in cleanup_errors
                    ],
                }
            )
            try:
                recorder.raise_if_failed()
                if recorder.frame_count <= 0:
                    recorder.abort(
                        "episode ended before first complete observation",
                        outcome=EpisodeOutcome.ERROR.value,
                        intervention_summary=intervention_summary,
                    )
                else:
                    final_path = recorder.finalize(
                        effective_outcome.value,
                        reason=(
                            effective_reason
                            if effective_outcome
                            not in {EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE}
                            else None
                        ),
                        failure_reason=(
                            args.failure_reason or episode_reason
                            if effective_outcome is EpisodeOutcome.FAILURE
                            else None
                        ),
                        intervention_summary=intervention_summary,
                    )
                    print(f"[RECORD] finalized episode: {final_path}", flush=True)
            except Exception as exc:
                cleanup_errors.append(exc)
                try:
                    staging = recorder.abort(
                        f"recorder finalization failed: {type(exc).__name__}: {exc}",
                        outcome=EpisodeOutcome.ERROR.value,
                        intervention_summary=intervention_summary,
                    )
                    print(f"[RECORD] staging retained: {staging}", flush=True)
                except Exception as abort_exc:
                    cleanup_errors.append(abort_exc)
            finally:
                try:
                    recorder.close()
                except Exception as exc:
                    cleanup_errors.append(exc)

    if pending_error is not None:
        raise pending_error
    if cleanup_errors:
        raise cleanup_errors[0]


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=10000)
    p.add_argument("--camera-settings", type=Path, required=True)
    p.add_argument("--camera-open-retries", type=int, default=5)
    p.add_argument("--camera-retry-delay-s", type=float, default=3.0)
    p.add_argument("--prompt", default="Open the drawer, put in the building blocks, and close the drawer")
    p.add_argument("--connect-timeout-s", type=float, default=5.0)
    p.add_argument("--socket-timeout-s", type=float, default=1000.0)
    p.add_argument("--state-timeout-s", type=float, default=5.0)
    p.add_argument("--observation-timeout-s", type=float, default=10.0)
    p.add_argument(
        "--command-topology-timeout-s",
        type=float,
        default=2.0,
        help="ROS discovery wait before enforcing exclusive command ownership",
    )
    p.add_argument("--max-state-age-ms", type=int, default=200)
    p.add_argument("--max-camera-age-ms", type=int, default=300)
    p.add_argument(
        "--max-camera-skew-ms",
        type=float,
        default=150.0,
        help="Maximum capture-time spread across the three camera frames",
    )
    p.add_argument(
        "--max-image-state-skew-ms",
        type=float,
        default=150.0,
        help="Maximum time difference between any camera frame and matched joint state",
    )
    p.add_argument(
        "--image-transport", choices=("float_mosaic", "raw_cameras"),
        default="float_mosaic",
        help="float_mosaic is exact training preprocessing with lower bandwidth",
    )
    p.add_argument("--max-chunks", type=int, default=100000)
    p.add_argument("--execute-actions", type=int, default=16)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--max-joint-delta-rad", type=float, default=0.03)
    p.add_argument("--max-first-target-delta-rad", type=float, default=0.30)
    p.add_argument("--max-velocity-rad-s", type=float, default=0.3)
    p.add_argument("--max-acceleration-rad-s2", type=float, default=2.0)
    p.add_argument("--initial-left-gripper-state", type=int, choices=(0, 1), default=0)
    p.add_argument("--initial-right-gripper-state", type=int, choices=(0, 1), default=0)
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--arm-command-topic", default="/min_snap/target")
    p.add_argument("--left-hand-topic", default="/left_topic_to_hand")
    p.add_argument("--right-hand-topic", default="/right_topic_to_hand")
    p.add_argument(
        "--enable-operator-intervention",
        action="store_true",
        help=(
            "enable i=takeover, s=success, f=failure, q/Esc=abort; FA VR "
            "must run in --ahawam_intervention_mode=true candidate mode"
        ),
    )
    p.add_argument(
        "--human-arm-input-topic",
        default="/ahawam/fa/human/min_snap_target",
    )
    p.add_argument(
        "--human-left-hand-input-topic",
        default="/ahawam/fa/human/left_hand",
    )
    p.add_argument(
        "--human-right-hand-input-topic",
        default="/ahawam/fa/human/right_hand",
    )
    p.add_argument("--max-human-command-age-ms", type=float, default=200.0)
    p.add_argument(
        "--human-control-timeout-s",
        type=float,
        default=0.0,
        help="maximum takeover duration; 0 waits indefinitely",
    )
    p.add_argument(
        "--record-dir",
        type=Path,
        help="append one crash-safe FA episode under this rollout root",
    )
    p.add_argument(
        "--record-fps",
        type=int,
        default=10,
        help="continuous raw capture rate; FA wrist cameras are natively 10 Hz",
    )
    p.add_argument("--record-segment-sec", type=float, default=10.0)
    p.add_argument(
        "--record-queue-size",
        type=int,
        default=32,
        help="bounded raw-frame queue (~176 MiB at the default FA geometry)",
    )
    p.add_argument("--record-min-free-gb", type=float, default=5.0)
    p.add_argument(
        "--require-outcome-label",
        action="store_true",
        help="hold after autonomous execution until s/f/q labels the episode",
    )
    p.add_argument(
        "--outcome-label-timeout-s",
        type=float,
        default=0.0,
        help="maximum label wait; 0 waits indefinitely",
    )
    p.add_argument(
        "--failure-reason",
        help="optional task-failure description stored when f is pressed",
    )
    p.add_argument("--enable-motion", action="store_true")
    p.add_argument("--confirm-motion", default="")
    args = p.parse_args(argv)
    if args.camera_open_retries <= 0:
        p.error("--camera-open-retries must be positive")
    if args.camera_retry_delay_s < 0:
        p.error("--camera-retry-delay-s must be non-negative")
    if args.max_camera_skew_ms <= 0 or args.max_image_state_skew_ms <= 0:
        p.error("camera and image-state skew limits must be positive")
    positive = (
        args.socket_timeout_s,
        args.connect_timeout_s,
        args.state_timeout_s,
        args.observation_timeout_s,
        args.command_topology_timeout_s,
        args.max_state_age_ms,
        args.max_camera_age_ms,
        args.max_chunks,
        args.execute_actions,
        args.hz,
        args.max_joint_delta_rad,
        args.max_first_target_delta_rad,
        args.max_velocity_rad_s,
        args.max_acceleration_rad_s2,
        args.max_human_command_age_ms,
        args.record_fps,
        args.record_segment_sec,
        args.record_queue_size,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        p.error("timeouts, rates, limits, counts, and ages must be positive")
    nonnegative = (
        args.record_min_free_gb,
        args.human_control_timeout_s,
        args.outcome_label_timeout_s,
    )
    if any(not np.isfinite(value) or value < 0 for value in nonnegative):
        p.error("record free space and optional timeouts must be non-negative")
    if args.require_outcome_label and not args.enable_operator_intervention:
        p.error(
            "--require-outcome-label requires --enable-operator-intervention"
        )
    real_topics = {
        args.arm_command_topic,
        args.left_hand_topic,
        args.right_hand_topic,
    }
    human_topics = {
        args.human_arm_input_topic,
        args.human_left_hand_input_topic,
        args.human_right_hand_input_topic,
    }
    if len(human_topics) != 3 or real_topics & human_topics:
        p.error(
            "human candidate topics must be distinct and must not equal real "
            "robot command topics"
        )
    args.camera_settings = args.camera_settings.expanduser().resolve()
    if not args.camera_settings.is_file():
        p.error(f"camera settings not found: {args.camera_settings}")
    if args.record_dir is not None:
        args.record_dir = args.record_dir.expanduser().resolve()
    if args.failure_reason is not None:
        args.failure_reason = args.failure_reason.strip()
        if not args.failure_reason:
            p.error("--failure-reason must be non-empty")
    return args


if __name__ == "__main__":
    run(parse_args())
