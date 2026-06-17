"""Arm command publisher abstraction for real robot backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from beavr.teleop.components.interface.robots.sysmo32_command import (
    SYSMO32_COMMAND_LENGTH,
    Sysmo32ArmCommand,
)
from beavr.teleop.components.interface.robots.fa_command_builder import (
    FA_UPPER_COMMAND_LENGTH,
    FaUpperPositionCommand,
)


class ArmCommandPublisherBase(abc.ABC):
    """Backend-independent arm command publisher."""

    @abc.abstractmethod
    def publish(self, command: Sysmo32ArmCommand) -> bool:
        """Publish one arm command."""


@dataclass
class Sysmo32CompatibleCommandPublisher(ArmCommandPublisherBase):
    """Publish the fixed 18D SYSMO-32-compatible Float64MultiArray payload."""

    ros_node: object
    msg_type: type
    topic: str
    queue_size: int

    def __post_init__(self):
        self._publisher = self.ros_node.create_publisher(
            self.msg_type,
            self.topic,
            self.queue_size,
        )

    def publish(self, command: Sysmo32ArmCommand) -> bool:
        if len(command.values) != SYSMO32_COMMAND_LENGTH:
            raise ValueError("Refusing to publish non-18D SYSMO-32-compatible command")
        msg = self.msg_type()
        msg.data = command.to_list()
        self._publisher.publish(msg)
        return True


class FaNativeCommandPublisher(ArmCommandPublisherBase):
    """Publish FA native 16D upper position commands."""

    def __init__(self, ros_node: object, msg_type: type, topic: str, queue_size: int):
        self.topic = topic
        self._msg_type = msg_type
        self._publisher = ros_node.create_publisher(
            msg_type,
            topic,
            queue_size,
        )

    def publish(self, command: FaUpperPositionCommand) -> bool:
        if len(command.values) != FA_UPPER_COMMAND_LENGTH:
            raise ValueError("Refusing to publish non-16D FA upper position command")
        msg = self._msg_type()
        msg.data = command.to_list()
        self._publisher.publish(msg)
        return True
