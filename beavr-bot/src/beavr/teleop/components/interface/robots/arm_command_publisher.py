"""ROS2 publisher helpers for robot command targets."""

from __future__ import annotations

from typing import Sequence


class MinSnapTargetPublisher:
    """Publish raw joint goals to the external humanoid_ws min_snap node."""

    def __init__(self, ros_node: object, msg_type: type, topic: str, queue_size: int):
        self.topic = topic
        self._msg_type = msg_type
        self._publisher = ros_node.create_publisher(msg_type, topic, queue_size)

    def publish(
        self,
        left_arm_target_rad: Sequence[float],
        right_arm_target_rad: Sequence[float],
        expected_duration_s: float,
        max_velocity_rad_s: float,
        max_acceleration_rad_s2: float,
    ) -> bool:
        msg = self._msg_type()
        msg.left_arm_target_rad = [float(value) for value in left_arm_target_rad]
        msg.right_arm_target_rad = [float(value) for value in right_arm_target_rad]
        msg.expected_duration_s = float(expected_duration_s)
        msg.max_velocity_rad_s = float(max_velocity_rad_s)
        msg.max_acceleration_rad_s2 = float(max_acceleration_rad_s2)
        self._publisher.publish(msg)
        return True
