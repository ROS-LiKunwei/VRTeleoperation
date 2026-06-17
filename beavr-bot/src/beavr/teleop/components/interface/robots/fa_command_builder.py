"""FA native upper-body command helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FA_ARM_JOINT_COUNT = 7
FA_NECK_JOINT_COUNT = 2
FA_UPPER_COMMAND_LENGTH = 16
FA_LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
FA_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
FA_NECK_JOINT_NAMES = ("neck_yaw_joint", "neck_pitch_joint")
FA_UPPER_JOINT_NAMES = FA_LEFT_ARM_JOINT_NAMES + FA_RIGHT_ARM_JOINT_NAMES + FA_NECK_JOINT_NAMES


@dataclass(frozen=True)
class FaUpperPositionCommand:
    """Exact 16-field command for `/upper_position_controller/commands`."""

    timestamp_s: float
    values: Tuple[float, ...]

    def __post_init__(self):
        if len(self.values) != FA_UPPER_COMMAND_LENGTH:
            raise ValueError(f"FA upper position command length must be 16, got {len(self.values)}")
        if not np.all(np.isfinite(np.asarray(self.values, dtype=np.float64))):
            raise ValueError("FA upper position command contains NaN/Inf")

    @property
    def left_arm(self) -> Tuple[float, ...]:
        return self.values[0:7]

    @property
    def right_arm(self) -> Tuple[float, ...]:
        return self.values[7:14]

    @property
    def neck(self) -> Tuple[float, float]:
        return self.values[14:16]

    def to_list(self) -> list[float]:
        return [float(v) for v in self.values]


@dataclass
class FaUpperPositionSafetyConfig:
    """Safety limits for FA 7+7+2 upper position commands."""

    neck_default_positions_rad: Tuple[float, float] = (0.0, 0.0)
    joint_lower_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([-3.14] * 16))
    joint_upper_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([3.14] * 16))
    max_joint_velocity_rad_s: Tuple[float, ...] = field(default_factory=lambda: tuple([3.0] * 16))
    max_joint_jump_rad: float = 0.5
    max_translation_step_m: float = 0.30
    max_rotation_step_rad: float = 0.5
    workspace_limits: dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.5, 1.5)}
    )

    def __post_init__(self):
        if len(self.neck_default_positions_rad) != FA_NECK_JOINT_COUNT:
            raise ValueError("neck_default_positions_rad must contain 2 values")
        for name, values in (
            ("joint_lower_limits_rad", self.joint_lower_limits_rad),
            ("joint_upper_limits_rad", self.joint_upper_limits_rad),
            ("max_joint_velocity_rad_s", self.max_joint_velocity_rad_s),
        ):
            if len(values) != FA_UPPER_COMMAND_LENGTH:
                raise ValueError(f"{name} must contain 16 values")


class FaUpperPositionCommandBuilder:
    """Build the exact 16D FA upper position command payload."""

    def __init__(self, safety_config: Optional[FaUpperPositionSafetyConfig] = None):
        self.config = safety_config or FaUpperPositionSafetyConfig()

    def build(
        self,
        left_arm_joints_rad: Sequence[float],
        right_arm_joints_rad: Sequence[float],
        neck_joints_rad: Optional[Sequence[float]] = None,
        timestamp_s: Optional[float] = None,
    ) -> FaUpperPositionCommand:
        left = _as_joint_vector(left_arm_joints_rad, FA_ARM_JOINT_COUNT, "left_arm")
        right = _as_joint_vector(right_arm_joints_rad, FA_ARM_JOINT_COUNT, "right_arm")
        neck_source = self.config.neck_default_positions_rad if neck_joints_rad is None else neck_joints_rad
        neck = _as_joint_vector(neck_source, FA_NECK_JOINT_COUNT, "neck")
        values = tuple(float(v) for v in np.concatenate([left, right, neck]))
        return FaUpperPositionCommand(timestamp_s=timestamp_s or time.time(), values=values)


class FaCommandLimiter:
    """Validate, clip and rate-limit FA upper position commands."""

    def __init__(self, safety_config: Optional[FaUpperPositionSafetyConfig] = None):
        self.config = safety_config or FaUpperPositionSafetyConfig()
        self._last_command: Optional[FaUpperPositionCommand] = None
        self._last_publish_time_s: Optional[float] = None

    @property
    def last_command(self) -> Optional[FaUpperPositionCommand]:
        return self._last_command

    def reset(self, current_upper_joints_rad: Optional[Sequence[float]] = None) -> None:
        self._last_command = None
        self._last_publish_time_s = None
        if current_upper_joints_rad is not None:
            joints = _as_joint_vector(current_upper_joints_rad, FA_UPPER_COMMAND_LENGTH, "current_upper_joints_rad")
            self._last_command = FaUpperPositionCommand(time.time(), tuple(float(v) for v in joints))
            self._last_publish_time_s = self._last_command.timestamp_s

    def limit(
        self, command: FaUpperPositionCommand, now_s: Optional[float] = None
    ) -> tuple[Optional[FaUpperPositionCommand], str]:
        now = time.time() if now_s is None else float(now_s)
        raw = np.asarray(command.values, dtype=np.float64)
        if raw.shape != (FA_UPPER_COMMAND_LENGTH,) or not np.all(np.isfinite(raw)):
            return None, "command contains NaN/Inf or wrong length"

        lower = np.asarray(self.config.joint_lower_limits_rad, dtype=np.float64)
        upper = np.asarray(self.config.joint_upper_limits_rad, dtype=np.float64)
        clipped = np.clip(raw, lower, upper)
        reason_parts = []
        if not np.allclose(clipped, raw, atol=1e-12):
            reason_parts.append("joint position clipped")

        if self._last_command is not None:
            last = np.asarray(self._last_command.values, dtype=np.float64)
            jump = np.abs(clipped - last)
            max_joint_jump = float(self.config.max_joint_jump_rad)
            if max_joint_jump > 0.0 and np.max(jump) > max_joint_jump:
                clipped = last + np.clip(clipped - last, -max_joint_jump, max_joint_jump)
                reason_parts.append(f"joint jump limited: {float(np.max(jump)):.3f} rad")
            dt = max(1e-3, now - (self._last_publish_time_s or now))
            max_delta = np.asarray(self.config.max_joint_velocity_rad_s, dtype=np.float64) * dt
            limited = last + np.clip(clipped - last, -max_delta, max_delta)
            if not np.allclose(limited, clipped, atol=1e-12):
                reason_parts.append("joint velocity limited")
            clipped = limited
            logger.debug(
                "FA command delta max: left=%.4f right=%.4f neck=%.4f",
                float(np.max(np.abs(clipped[:7] - last[:7]))),
                float(np.max(np.abs(clipped[7:14] - last[7:14]))),
                float(np.max(np.abs(clipped[14:16] - last[14:16]))),
            )

        limited_command = FaUpperPositionCommand(now, tuple(float(v) for v in clipped))
        self._last_command = limited_command
        self._last_publish_time_s = now
        return limited_command, ", ".join(reason_parts)


@dataclass
class FaJointStateSnapshot:
    timestamp_s: float
    left_arm: Tuple[float, ...]
    right_arm: Tuple[float, ...]
    neck: Tuple[float, float]

    @property
    def upper_joints(self) -> np.ndarray:
        return np.asarray(self.left_arm + self.right_arm + self.neck, dtype=np.float64)

    @property
    def arm_joints(self) -> np.ndarray:
        return np.asarray(self.left_arm + self.right_arm, dtype=np.float64)


class FaJointStateCache:
    """Parse FA 7+7+2 upper joint positions from sensor_msgs/JointState by name."""

    def __init__(self, joint_state_timeout_s: float = 0.8):
        self.joint_state_timeout_s = float(joint_state_timeout_s)
        self._snapshot: Optional[FaJointStateSnapshot] = None
        self._last_missing_joint: Optional[str] = None

    @property
    def snapshot(self) -> Optional[FaJointStateSnapshot]:
        return self._snapshot

    @property
    def last_missing_joint(self) -> Optional[str]:
        return self._last_missing_joint

    def update_from_joint_state_msg(
        self, msg, now_s: Optional[float] = None
    ) -> Optional[FaJointStateSnapshot]:
        name_to_position = dict(zip(list(msg.name), list(msg.position), strict=False))
        try:
            left = tuple(float(name_to_position[name]) for name in FA_LEFT_ARM_JOINT_NAMES)
            right = tuple(float(name_to_position[name]) for name in FA_RIGHT_ARM_JOINT_NAMES)
            neck = tuple(float(name_to_position[name]) for name in FA_NECK_JOINT_NAMES)
        except KeyError as exc:
            self._last_missing_joint = str(exc)
            logger.warning("JointState missing FA upper joint: %s", exc)
            return self._snapshot
        self._last_missing_joint = None
        return self.update(left, right, neck, now_s=now_s)

    def update(
        self,
        left_arm: Sequence[float],
        right_arm: Sequence[float],
        neck: Sequence[float],
        now_s: Optional[float] = None,
    ) -> FaJointStateSnapshot:
        left = tuple(float(v) for v in _as_joint_vector(left_arm, FA_ARM_JOINT_COUNT, "left_arm"))
        right = tuple(float(v) for v in _as_joint_vector(right_arm, FA_ARM_JOINT_COUNT, "right_arm"))
        neck_values = tuple(float(v) for v in _as_joint_vector(neck, FA_NECK_JOINT_COUNT, "neck"))
        self._snapshot = FaJointStateSnapshot(time.time() if now_s is None else float(now_s), left, right, neck_values)
        return self._snapshot

    def is_fresh(self, now_s: Optional[float] = None) -> bool:
        if self._snapshot is None:
            return False
        now = time.time() if now_s is None else float(now_s)
        return now - self._snapshot.timestamp_s <= self.joint_state_timeout_s


def _as_joint_vector(values: Sequence[float], expected_count: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain {expected_count} finite joint values")
    return array


__all__ = [
    "FA_ARM_JOINT_COUNT",
    "FA_NECK_JOINT_COUNT",
    "FA_UPPER_COMMAND_LENGTH",
    "FA_LEFT_ARM_JOINT_NAMES",
    "FA_RIGHT_ARM_JOINT_NAMES",
    "FA_NECK_JOINT_NAMES",
    "FA_UPPER_JOINT_NAMES",
    "FaUpperPositionCommand",
    "FaUpperPositionSafetyConfig",
    "FaUpperPositionCommandBuilder",
    "FaCommandLimiter",
    "FaJointStateCache",
    "FaJointStateSnapshot",
]
