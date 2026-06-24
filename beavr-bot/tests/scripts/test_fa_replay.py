import sys
from types import ModuleType

import numpy as np

from beavr.scripts.control_robot import _FaDatasetReplayPublisher


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
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "min_snap", min_snap)
    monkeypatch.setitem(sys.modules, "min_snap.msg", min_snap_msg)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)

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
    assert fake_rclpy.spin_calls == 1

    publisher.close()
    assert fake_rclpy.node.destroyed
