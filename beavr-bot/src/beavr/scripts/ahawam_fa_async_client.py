#!/usr/bin/env python3
"""FA V6 three-camera inference client; motion is disabled by default.

Run this from the beavr-bot environment on the robot brain. It reuses the
recording camera_settings.json, including RGB-D profiles and fixed manual
camera controls. Do not run FA teleop concurrently when --enable-motion is
used because both would publish robot commands.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_KEYS = ("head", "wrist_CV2756100033", "wrist_CV2856D0001C")
LEFT_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
)
RIGHT_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_JOINTS)
FA_GRIPPER_CLOSE_THRESHOLD = 0.8
_NDARRAY_MARKER = "__ahawam_ndarray_v1__"


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
    def __init__(self, host: str, port: int, timeout: float):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def request(self, message):
        send_message(self.sock, message)
        reply, _ = recv_message(self.sock)
        if not isinstance(reply, dict) or not reply.get("ok", False):
            raise RuntimeError(f"Policy request failed: {reply}")
        return reply

    def close(self):
        try:
            self.sock.close()
        except Exception:
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
                "/home/wangzebei/likunwei_ws/install/setup.bash"
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
        self.last_joint_message_at = 0.0
        self.last_missing_joints = ()
        self.left_gripper = float(args.initial_left_gripper_state)
        self.right_gripper = float(args.initial_right_gripper_state)
        # Only retain the newest feedback sample. A dedicated spin thread keeps
        # consuming it while policy inference blocks the main thread.
        self.node.create_subscription(JointState, args.joint_state_topic, self._state, 1)
        self.arm_pub = self.node.create_publisher(MinSnapTarget, args.arm_command_topic, 1)
        self.left_hand_pub = self.node.create_publisher(Int32, args.left_hand_topic, 10)
        self.right_hand_pub = self.node.create_publisher(Int32, args.right_hand_topic, 10)
        self.MinSnapTarget, self.Int32 = MinSnapTarget, Int32
        self._spin_thread = threading.Thread(
            target=self._spin_feedback,
            name="ahawam-fa-joint-feedback",
            daemon=True,
        )
        self._spin_thread.start()

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

    def state(self, timeout_s: float, max_age_s: float) -> np.ndarray:
        state, _metadata = self.state_snapshot(timeout_s, max_age_s)
        return state

    def publish(self, action: np.ndarray, args):
        msg = self.MinSnapTarget()
        msg.left_arm_target_rad = action[:7].astype(float).tolist()
        msg.right_arm_target_rad = action[7:14].astype(float).tolist()
        msg.expected_duration_s = 1.0 / args.hz
        msg.max_velocity_rad_s = args.max_velocity_rad_s
        msg.max_acceleration_rad_s2 = args.max_acceleration_rad_s2
        self.arm_pub.publish(msg)
        for value, old, pub in (
            (action[14], self.left_gripper, self.left_hand_pub),
            (action[15], self.right_gripper, self.right_hand_pub),
        ):
            new = binarize_gripper(value)
            if new != old:
                hand = self.Int32(); hand.data = 20 if new else 21; pub.publish(hand)
            if pub is self.left_hand_pub:
                self.left_gripper = new
            else:
                self.right_gripper = new

    def close(self):
        self._spin_stop.set()
        with self._state_condition:
            self._state_condition.notify_all()
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


def read_images(cameras, max_age_ms: int):
    result = {}
    for key in CAMERA_KEYS:
        rgb, _depth = cameras[key].read_latest_rgbd(max_age_ms)
        expected = (720, 1280, 3) if key == "head" else (530, 848, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError(f"{key} produced {rgb.shape}/{rgb.dtype}; expected {expected}/uint8")
        result[key] = np.ascontiguousarray(rgb)
    return result


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


def run(args):
    if args.enable_motion and args.confirm_motion != "RUN":
        raise SystemExit("Motion requires both --enable-motion and --confirm-motion RUN")
    cameras = {}
    ros = None
    policy = None
    try:
        # Validate ROS message availability and server reachability before
        # opening native camera pipelines. Every later failure is covered by
        # the common cleanup path below.
        ros = FaRos(args)
        policy = PolicyConnection(args.host, args.port, args.socket_timeout_s)
        cameras = open_cameras(
            args.camera_settings,
            attempts=args.camera_open_retries,
            retry_delay_s=args.camera_retry_delay_s,
        )
        print(policy.request({"type": "reset", "instruction": args.prompt}))
        last_command = None
        last_chunk_state_sequence = 0
        for chunk_index in range(args.max_chunks):
            state, state_metadata = ros.state_snapshot(
                args.state_timeout_s,
                args.max_state_age_ms / 1000.0,
                newer_than_sequence=last_chunk_state_sequence,
            )
            last_chunk_state_sequence = int(state_metadata["sequence"])
            previous_target = None if last_command is None else last_command[:14]
            target_error = None if previous_target is None else state[:14] - previous_target
            print(
                "chunk_feedback=" + json.dumps(
                    {
                        "chunk": chunk_index,
                        "joint_state_sequence": last_chunk_state_sequence,
                        "source_stamp_s": state_metadata["source_stamp_s"],
                        "source_age_ms": round(float(state_metadata["source_age_ms"]), 3),
                        "receive_age_ms": round(float(state_metadata["receive_age_ms"]), 3),
                        "actual_arm_rad": state[:14].astype(float).tolist(),
                        "previous_target_arm_rad": (
                            None
                            if previous_target is None
                            else previous_target.astype(float).tolist()
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
            images = read_images(cameras, args.max_camera_age_ms)
            wire_images = (
                images if args.image_transport == "raw_cameras"
                else {"front": compose_training_mosaic(images)}
            )
            t0 = time.perf_counter()
            policy.request({
                "type": "image", "images": wire_images, "state": state,
                "timestamp": time.time(), "frame_sequence": chunk_index,
                "observation_id": chunk_index, "expect_reply": True,
            })
            # Bind ActionDiT to the state carried by this exact image message.
            # ``chunk_index - 1`` requires the server to select this frame or a
            # newer one and prevents accidental reuse of the previous frame.
            reply = policy.request({
                "type": "action_request",
                "after_frame_sequence": chunk_index - 1,
                "require_paired_observation": True,
            })
            chunk = validate_action(reply.get("action_chunk"))
            first_delta = np.abs(chunk[0, :14] - state[:14])
            if first_delta.max(initial=0.0) > args.max_first_target_delta_rad:
                raise RuntimeError(f"First target jump too large: {first_delta.max():.4f} rad")
            execute = min(args.execute_actions, len(chunk))
            print(
                f"chunk={chunk_index} model_ms={reply.get('model_latency_ms')} "
                f"rtt_ms={(time.perf_counter()-t0)*1000:.1f} execute={execute} "
                f"motion={args.enable_motion}"
            )
            print(format_gripper_output_log(chunk, execute, chunk_index), flush=True)
            for target in chunk[:execute]:
                measured = ros.state(args.state_timeout_s, args.max_state_age_ms / 1000.0)
                reference = measured if last_command is None else last_command
                limited = target.copy()
                limited[:14] = reference[:14] + np.clip(
                    target[:14] - reference[:14], -args.max_joint_delta_rad, args.max_joint_delta_rad
                )
                limited[14:] = np.asarray(
                    [binarize_gripper(value) for value in limited[14:]],
                    dtype=np.float32,
                )
                if args.enable_motion:
                    ros.publish(limited, args)
                last_command = limited
                time.sleep(1.0 / args.hz)
    finally:
        if policy is not None:
            policy.close()
        if ros is not None:
            ros.close()
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception as exc:
                print(f"Camera cleanup failed: {exc}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True); p.add_argument("--port", type=int, default=10000)
    p.add_argument("--camera-settings", type=Path, required=True)
    p.add_argument("--camera-open-retries", type=int, default=5)
    p.add_argument("--camera-retry-delay-s", type=float, default=3.0)
    p.add_argument("--prompt", default="Open the drawer, put in the building blocks, and close the drawer")
    p.add_argument("--socket-timeout-s", type=float, default=1000.0)
    p.add_argument("--state-timeout-s", type=float, default=5.0)
    p.add_argument("--max-state-age-ms", type=int, default=200)
    p.add_argument("--max-camera-age-ms", type=int, default=300)
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
    p.add_argument("--enable-motion", action="store_true")
    p.add_argument("--confirm-motion", default="")
    args = p.parse_args()
    if args.camera_open_retries <= 0:
        p.error("--camera-open-retries must be positive")
    if args.camera_retry_delay_s < 0:
        p.error("--camera-retry-delay-s must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
