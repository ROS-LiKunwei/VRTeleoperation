import sys
from types import ModuleType

import numpy as np

from beavr.scripts.control_robot import (
    _FA_REPLAY_APPROACH_SEQUENCE,
    _FA_REPLAY_P2_TO_P3_MOVE_DURATION_S,
    _FA_REPLAY_P3_TO_P4_MOVE_DURATION_S,
    _FA_REPLAY_INITIAL_RIGHT_OPEN_ACTION,
    _FA_REPLAY_RIGHT_OPEN_ACTION,
    _FA_REPLAY_RETURN_SEQUENCE,
    _FA_REPLAY_WAYPOINT_1,
    _FA_REPLAY_WAYPOINT_2,
    _FA_REPLAY_WAYPOINT_3,
    _FA_REPLAY_WAYPOINT_4,
    _FaDatasetReplayPublisher,
    _run_fa_replay_arm_sequence,
)


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _FakeNode:
    def __init__(self):
        self.publishers = []
        self.destroyed = False

    def create_publisher(self, _msg_type, _topic, _queue_size):
        publisher = _FakePublisher()
        self.publishers.append(publisher)
        return publisher

    def create_subscription(self, _msg_type, _topic, _callback, _queue_size):
        return object()

    def destroy_node(self):
        self.destroyed = True


class _FakeRclpy:
    def __init__(self):
        self.node = _FakeNode()
        self.spin_calls = 0

    def ok(self):
        return False

    def init(self, args=None):
        self.init_args = args

    def create_node(self, _name):
        return self.node

    def spin_once(self, _node, timeout_sec=0.0):
        self.spin_calls += 1


class _FakeMinSnapTarget:
    pass


class _FakeInt32:
    pass


class _FakeJointState:
    pass


def test_fa_replay_waypoint_order_and_values():
    expected_1 = (0.0, 0.2, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected_2 = (0.0, 1.4, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -1.4, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected_3 = (0.0, 1.4, 0.0, -1.4, 0.0, 0.0, 0.0, -0.0, -1.4, 0.0, -1.4, 0.0, 0.0, 0.0)
    expected_4 = (
        -0.868398, 0.992635, -0.978102, -1.870111, -0.383045, 0.250036, -0.009134,
        -0.969093, -1.124586, 0.831005, -1.463140, 0.269165, -0.044380, 0.134951,
    )
    assert _FA_REPLAY_WAYPOINT_1 == expected_1
    assert _FA_REPLAY_WAYPOINT_2 == expected_2
    assert _FA_REPLAY_WAYPOINT_3 == expected_3
    assert _FA_REPLAY_WAYPOINT_4 == expected_4
    assert _FA_REPLAY_APPROACH_SEQUENCE == (
        ("P1", expected_1), ("P2", expected_2), ("P3", expected_3), ("P4", expected_4)
    )
    assert _FA_REPLAY_RETURN_SEQUENCE == (
        ("P4", expected_4), ("P3", expected_3), ("P2", expected_2), ("P1", expected_1)
    )
    assert _FA_REPLAY_P2_TO_P3_MOVE_DURATION_S == 10.5
    assert _FA_REPLAY_P3_TO_P4_MOVE_DURATION_S == 7.5


def test_fa_replay_uses_transition_durations_for_approach_to_p4():
    class FakeReplayPublisher:
        def __init__(self):
            self.calls = []

        def move_arm_target(self, target, *, duration_s, label):
            self.calls.append((label, duration_s))
            return np.asarray(target)

    publisher = FakeReplayPublisher()
    _run_fa_replay_arm_sequence(
        publisher,
        _FA_REPLAY_APPROACH_SEQUENCE,
        duration_overrides_s={
            "P3": _FA_REPLAY_P2_TO_P3_MOVE_DURATION_S,
            "P4": _FA_REPLAY_P3_TO_P4_MOVE_DURATION_S,
        },
    )
    assert publisher.calls == [("P1", 4.0), ("P2", 4.0), ("P3", 10.5), ("P4", 7.5)]


def test_fa_dataset_replay_publisher_splits_arm_and_hand_actions(monkeypatch):
    fake_rclpy = _FakeRclpy()
    min_snap = ModuleType("min_snap")
    min_snap_msg = ModuleType("min_snap.msg")
    min_snap_msg.MinSnapTarget = _FakeMinSnapTarget
    min_snap.msg = min_snap_msg
    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    std_msgs_msg.Int32 = _FakeInt32
    std_msgs.msg = std_msgs_msg
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = _FakeJointState
    sensor_msgs.msg = sensor_msgs_msg
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "min_snap", min_snap)
    monkeypatch.setitem(sys.modules, "min_snap.msg", min_snap_msg)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)

    publisher = _FaDatasetReplayPublisher(fps=20)
    action = np.arange(16, dtype=np.float32)
    action[14] = 0.0
    action[15] = 1.0

    publisher.publish(action)

    arm_pub, left_hand_pub, right_hand_pub = fake_rclpy.node.publishers
    arm_msg = arm_pub.messages[0]
    assert arm_msg.left_arm_target_rad == [float(value) for value in range(7)]
    assert arm_msg.right_arm_target_rad == [float(value) for value in range(7, 14)]
    assert arm_msg.expected_duration_s == 0.05
    assert left_hand_pub.messages[0].data == 21
    assert right_hand_pub.messages[0].data == 20
    # publish_arm_target() services ROS once after the arm command, then
    # publish() services it again after both hand commands.
    assert fake_rclpy.spin_calls == 2

    publisher.close()
    assert fake_rclpy.node.destroyed


def test_fa_dataset_replay_uses_distinct_right_hand_initial_and_track_open_actions(monkeypatch):
    fake_rclpy = _FakeRclpy()
    min_snap = ModuleType("min_snap")
    min_snap_msg = ModuleType("min_snap.msg")
    min_snap_msg.MinSnapTarget = _FakeMinSnapTarget
    min_snap.msg = min_snap_msg
    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    std_msgs_msg.Int32 = _FakeInt32
    std_msgs.msg = std_msgs_msg
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = _FakeJointState
    sensor_msgs.msg = sensor_msgs_msg
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "min_snap", min_snap)
    monkeypatch.setitem(sys.modules, "min_snap.msg", min_snap_msg)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)

    publisher = _FaDatasetReplayPublisher(fps=30)
    publisher.initialize_right_hand()

    action = np.zeros(16, dtype=np.float32)
    publisher.publish(action)

    _arm_pub, left_hand_pub, right_hand_pub = fake_rclpy.node.publishers
    assert _FA_REPLAY_INITIAL_RIGHT_OPEN_ACTION == 21
    assert _FA_REPLAY_RIGHT_OPEN_ACTION == 22
    assert [message.data for message in right_hand_pub.messages] == [21, 22]
    assert [message.data for message in left_hand_pub.messages] == [21]
