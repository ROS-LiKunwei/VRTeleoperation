"""Convert SYSMO-32 real arm commands into interpolated JointState messages."""

from __future__ import annotations

import math
from typing import Iterable

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
)
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
ARM_JOINT_COUNT = len(ARM_JOINT_NAMES)
SYSMO32_COMMAND_LENGTH = 18


class Sysmo32CommandToJointState(Node):
    """Subscribe to the real 18D command topic and drive a RViz robot model."""

    def __init__(self) -> None:
        super().__init__("sysmo32_command_to_joint_state")

        self.command_topic = str(
            self.declare_parameter("command_topic", "/sysmo_left_arm_controller/commands").value
        )
        self.joint_state_topic = str(
            self.declare_parameter("joint_state_topic", "/joint_states").value
        )
        self.interpolation_steps = max(1, int(self.declare_parameter("interpolation_steps", 5).value))
        self.publish_rate_hz = max(1.0, float(self.declare_parameter("publish_rate_hz", 100.0).value))
        self.idle_publish_rate_hz = max(
            0.0, float(self.declare_parameter("idle_publish_rate_hz", 2.0).value)
        )
        self.expected_command_length = int(
            self.declare_parameter("expected_command_length", SYSMO32_COMMAND_LENGTH).value
        )

        self._current_positions = [0.0] * ARM_JOINT_COUNT
        self._start_positions = [0.0] * ARM_JOINT_COUNT
        self._target_positions: list[float] | None = None
        self._step_index = 0
        self._last_idle_publish_time = None

        self._publisher = self.create_publisher(JointState, self.joint_state_topic, 10)
        self._subscription = self.create_subscription(
            Float64MultiArray,
            self.command_topic,
            self._on_command,
            10,
        )
        self._timer = self.create_timer(1.0 / self.publish_rate_hz, self._on_timer)

        self.get_logger().info(
            f"Listening to {self.command_topic}, publishing interpolated JointState to "
            f"{self.joint_state_topic} (quintic steps={self.interpolation_steps}, "
            f"rate={self.publish_rate_hz:.1f} Hz)"
        )

    def _on_command(self, msg: Float64MultiArray) -> None:
        values = list(msg.data)
        if self.expected_command_length > 0 and len(values) != self.expected_command_length:
            self.get_logger().warning(
                f"Ignoring command with length {len(values)}; expected {self.expected_command_length}"
            )
            return
        if len(values) < ARM_JOINT_COUNT:
            self.get_logger().warning(
                f"Ignoring command with length {len(values)}; need at least {ARM_JOINT_COUNT} arm joints"
            )
            return

        target = [float(v) for v in values[:ARM_JOINT_COUNT]]
        if not _all_finite(target):
            self.get_logger().warning("Ignoring command containing NaN or Inf")
            return

        self._start_positions = list(self._current_positions)
        self._target_positions = target
        self._step_index = 0

    def _on_timer(self) -> None:
        if self._target_positions is None:
            self._publish_idle_if_due()
            return

        self._step_index += 1
        progress = min(1.0, self._step_index / float(self.interpolation_steps))
        alpha = _quintic_blend(progress)
        self._current_positions = [
            start + (target - start) * alpha
            for start, target in zip(self._start_positions, self._target_positions)
        ]
        self._publish_joint_state()

        if self._step_index >= self.interpolation_steps:
            self._target_positions = None
            self._step_index = 0

    def _publish_idle_if_due(self) -> None:
        if self.idle_publish_rate_hz <= 0.0:
            return
        now = self.get_clock().now()
        if self._last_idle_publish_time is not None:
            elapsed_s = (now - self._last_idle_publish_time).nanoseconds / 1e9
            if elapsed_s < 1.0 / self.idle_publish_rate_hz:
                return
        self._publish_joint_state()
        self._last_idle_publish_time = now

    def _publish_joint_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = list(self._current_positions)
        self._publisher.publish(msg)


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _quintic_blend(progress: float) -> float:
    tau = min(1.0, max(0.0, float(progress)))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Sysmo32CommandToJointState()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
