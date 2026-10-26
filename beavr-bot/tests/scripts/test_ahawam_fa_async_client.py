import json
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from beavr.scripts.ahawam_fa_async_client import (
    CAMERA_KEYS,
    FaObservation,
    FaRos,
    LEFT_JOINTS,
    RIGHT_JOINTS,
    binarize_gripper,
    camera_observation_time,
    format_gripper_output_log,
    image_state_sync_metadata,
    _policy_exchange,
    validate_synchronous_reply,
)


class _FakePublisher:
    def publish(self, _message):
        pass


class _FakeNode:
    def __init__(self):
        self.subscription_depth = None
        self.subscription_callback = None
        self.destroyed = False

    def create_subscription(self, _message_type, _topic, callback, depth):
        self.subscription_depth = depth
        self.subscription_callback = callback
        return object()

    def create_publisher(self, _message_type, _topic, _depth):
        return _FakePublisher()

    def destroy_node(self):
        self.destroyed = True


def _install_fake_ros_modules(monkeypatch):
    node = _FakeNode()
    state = {"ok": True, "spin_calls": 0}
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: state["ok"]
    rclpy.init = lambda args=None: state.__setitem__("ok", True)
    rclpy.create_node = lambda _name: node

    def spin_once(_node, timeout_sec):
        state["spin_calls"] += 1
        time.sleep(timeout_sec)

    rclpy.spin_once = spin_once
    rclpy.shutdown = lambda: state.__setitem__("ok", False)

    class MinSnapTarget:
        pass

    class JointState:
        pass

    class Int32:
        pass

    modules = {
        "rclpy": rclpy,
        "min_snap": types.ModuleType("min_snap"),
        "min_snap.msg": types.ModuleType("min_snap.msg"),
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg"),
        "std_msgs": types.ModuleType("std_msgs"),
        "std_msgs.msg": types.ModuleType("std_msgs.msg"),
    }
    modules["min_snap.msg"].MinSnapTarget = MinSnapTarget
    modules["sensor_msgs.msg"].JointState = JointState
    modules["std_msgs.msg"].Int32 = Int32
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return node, state


def _args():
    return SimpleNamespace(
        joint_state_topic="/joint_states",
        arm_command_topic="/min_snap/target",
        left_hand_topic="/left_topic_to_hand",
        right_hand_topic="/right_topic_to_hand",
        initial_left_gripper_state=0,
        initial_right_gripper_state=0,
    )


def _joint_state(stamp_s, position_offset=0.0):
    seconds = int(stamp_s)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=seconds,
                nanosec=int((stamp_s - seconds) * 1e9),
            )
        ),
        name=list(LEFT_JOINTS + RIGHT_JOINTS),
        position=list(np.linspace(-0.7, 0.7, 14) + position_offset),
    )


def test_gripper_close_threshold_is_inclusive_at_point_eight():
    assert binarize_gripper(0.7999) == 0.0
    assert binarize_gripper(0.8) == 1.0
    assert binarize_gripper(1.0) == 1.0


def test_gripper_output_log_records_raw_and_binary_executed_actions():
    chunk = np.zeros((3, 16), dtype=np.float32)
    chunk[:, 14] = [0.79, 0.8, 0.9]
    chunk[:, 15] = [0.85, 0.75, 0.8]

    line = format_gripper_output_log(chunk, execute=2, chunk_index=7)
    prefix, payload = line.split("=", 1)
    record = json.loads(payload)

    assert prefix == "chunk_gripper_output"
    assert record["chunk"] == 7
    assert record["executed_steps"] == 2
    assert record["close_threshold"] == 0.8
    assert record["action_14_left_raw"] == pytest.approx([0.79, 0.8])
    assert record["action_15_right_raw"] == pytest.approx([0.85, 0.75])
    assert record["action_14_left_binary"] == [0.0, 1.0]
    assert record["action_15_right_binary"] == [1.0, 0.0]


def _server_stage():
    return {
        "stage_id": 1,
        "stage": "open_drawer",
        "instruction": "Pull the drawer open while holding the handle.",
        "controller_enabled": False,
    }


def test_policy_exchange_is_serial_paired_and_uses_server_stage():
    state = np.arange(16, dtype=np.float32)
    observation = FaObservation(
        sequence=9,
        captured_at_monotonic_s=1.0,
        captured_at_wall_s=2.0,
        images={key: np.zeros((2, 3, 3), dtype=np.uint8) for key in CAMERA_KEYS},
        state=state,
        camera_metadata={},
        state_metadata={},
        sync_metadata={},
        command=None,
    )

    class Policy:
        def __init__(self):
            self.messages = []

        def request(self, message):
            self.messages.append(message)
            if message["type"] == "image":
                return {"ok": True, "type": "image_ack"}
            return {
                "ok": True,
                "type": "action_chunk",
                "action_chunk": np.zeros((16, 16), dtype=np.float32),
                "stage_language": _server_stage(),
            }

    policy = Policy()
    reply = _policy_exchange(
        policy,
        observation,
        SimpleNamespace(image_transport="raw_cameras"),
        chunk_index=3,
    )

    assert [message["type"] for message in policy.messages] == [
        "image",
        "action_request",
    ]
    assert policy.messages[0]["observation_id"] == 9
    assert policy.messages[1]["after_frame_sequence"] == 2
    assert policy.messages[1]["require_paired_observation"] is True
    np.testing.assert_array_equal(policy.messages[1]["state"], state)
    assert reply["stage_language"]["stage"] == "open_drawer"


def test_server_stage_is_required_without_local_fallback():
    with pytest.raises(RuntimeError, match="missing stage_language"):
        validate_synchronous_reply(
            {"type": "action_chunk"},
            frame_sequence=0,
            observation_id=1,
            state=np.zeros(16, dtype=np.float32),
        )


def test_fa_ros_continuously_spins_depth_one_and_checks_source_stamp(monkeypatch):
    node, ros_state = _install_fake_ros_modules(monkeypatch)
    ros = FaRos(_args())
    try:
        assert node.subscription_depth == 1
        time.sleep(0.06)
        assert ros_state["spin_calls"] >= 1

        # The publisher clock may be ahead of the client clock. Calibrate that
        # fixed offset without treating a genuinely old sample as fresh.
        node.subscription_callback(_joint_state(time.time() + 0.09))
        observed, metadata = ros.state_snapshot(
            timeout_s=0.1, max_age_s=0.2, newer_than_sequence=0
        )
        assert observed.shape == (16,)
        assert metadata["sequence"] == 1
        assert metadata["source_age_ms"] < 200
        assert metadata["raw_source_age_ms"] < -50
        assert metadata["source_clock_offset_ms"] < -50

        # A callback executed now must not make an old sampled state fresh.
        node.subscription_callback(_joint_state(time.time() - 1.0))
        with pytest.raises(TimeoutError, match="source_age="):
            ros.state_snapshot(
                timeout_s=0.01, max_age_s=0.2, newer_than_sequence=1
            )
    finally:
        ros.close()

    assert node.destroyed
    assert not ros._spin_thread.is_alive()


def test_fa_ros_matches_historical_joint_state_to_image_time(monkeypatch):
    node, _ros_state = _install_fake_ros_modules(monkeypatch)
    ros = FaRos(_args())
    try:
        node.subscription_callback(_joint_state(time.time(), position_offset=0.0))
        first_sample_time = ros._joint_history[-1]["sample_at_monotonic_s"]
        time.sleep(0.01)
        node.subscription_callback(_joint_state(time.time(), position_offset=1.0))

        observed, metadata = ros.state_at_observation(
            first_sample_time,
            timeout_s=0.1,
            max_age_s=0.2,
            max_skew_s=0.005,
            newer_than_sequence=0,
        )
        assert observed[:14] == pytest.approx(np.linspace(-0.7, 0.7, 14))
        assert metadata["sequence"] == 1
        assert abs(metadata["joint_observation_skew_ms"]) < 1.0
    finally:
        ros.close()


def test_fa_ros_rejects_stale_historical_match_without_waiting(monkeypatch):
    node, _ros_state = _install_fake_ros_modules(monkeypatch)
    ros = FaRos(_args())
    try:
        node.subscription_callback(_joint_state(time.time()))
        observation_time = ros._joint_history[-1]["sample_at_monotonic_s"]
        time.sleep(0.03)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="No joint state matched"):
            ros.state_at_observation(
                observation_time,
                timeout_s=0.5,
                max_age_s=0.01,
                max_skew_s=0.15,
                newer_than_sequence=0,
            )
        assert time.monotonic() - started < 0.1
    finally:
        ros.close()


def test_camera_and_joint_observation_skew_limits():
    camera_metadata = {
        key: {
            "sequence": index + 1,
            "captured_at_monotonic_s": 10.0 + index * 0.04,
            "age_ms": 20.0,
        }
        for index, key in enumerate(CAMERA_KEYS)
    }
    observation_time, camera_skew_ms = camera_observation_time(
        camera_metadata,
        max_camera_skew_ms=90.0,
    )
    assert observation_time == pytest.approx(10.04)
    assert camera_skew_ms == pytest.approx(80.0)

    state_metadata = {"sample_at_monotonic_s": 10.045}
    sync = image_state_sync_metadata(
        camera_metadata,
        state_metadata,
        observation_time,
        max_image_state_skew_ms=50.0,
    )
    assert sync["joint_observation_skew_ms"] == pytest.approx(5.0)
    assert sync["max_abs_camera_joint_skew_ms"] == pytest.approx(45.0)

    with pytest.raises(RuntimeError, match="Camera frame skew"):
        camera_observation_time(camera_metadata, max_camera_skew_ms=50.0)
    with pytest.raises(RuntimeError, match="Camera/joint skew"):
        image_state_sync_metadata(
            camera_metadata,
            state_metadata,
            observation_time,
            max_image_state_skew_ms=40.0,
        )


def test_camera_observation_rejects_aligned_but_stale_set():
    camera_metadata = {
        key: {
            "sequence": index + 1,
            "captured_at_monotonic_s": 10.0 + index * 0.01,
            "age_ms": 0.0,
        }
        for index, key in enumerate(CAMERA_KEYS)
    }

    with pytest.raises(RuntimeError, match="Camera set is stale"):
        camera_observation_time(
            camera_metadata,
            max_camera_skew_ms=150.0,
            max_camera_age_ms=300.0,
            now_monotonic_s=10.5,
        )
