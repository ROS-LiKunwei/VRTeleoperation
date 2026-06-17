"""FA joint smoothing helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

_MIN_SNAP_MAX_S_DOT = 2.1875
_MIN_SNAP_MAX_ABS_S_DDOT = 7.513188404399293


@dataclass(frozen=True)
class FaArmTrajectoryConfig:
    joint_count: int = 7
    enabled: bool = True
    segment_time_s: float = 0.18
    min_duration_s: float = 0.06
    replan_threshold_rad: float = 0.0005
    max_joint_velocity_rad_s: Tuple[float, ...] = field(default_factory=lambda: tuple([3.0] * 7))
    max_joint_acceleration_rad_s2: Tuple[float, ...] = field(default_factory=lambda: tuple([12.0] * 7))
    joint_lower_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([-3.14] * 7))
    joint_upper_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([3.14] * 7))

    def __post_init__(self):
        for name, values in (
            ("max_joint_velocity_rad_s", self.max_joint_velocity_rad_s),
            ("max_joint_acceleration_rad_s2", self.max_joint_acceleration_rad_s2),
            ("joint_lower_limits_rad", self.joint_lower_limits_rad),
            ("joint_upper_limits_rad", self.joint_upper_limits_rad),
        ):
            if len(values) != self.joint_count:
                raise ValueError(f"{name} must contain {self.joint_count} values")
        object.__setattr__(self, "segment_time_s", max(0.0, float(self.segment_time_s)))
        object.__setattr__(self, "min_duration_s", max(1e-4, float(self.min_duration_s)))
        object.__setattr__(self, "replan_threshold_rad", max(0.0, float(self.replan_threshold_rad)))


@dataclass(frozen=True)
class FaJerkLimitedServoConfig:
    joint_count: int = 7
    enabled: bool = True
    max_joint_velocity_rad_s: Tuple[float, ...] = field(default_factory=lambda: tuple([3.0] * 7))
    max_joint_acceleration_rad_s2: Tuple[float, ...] = field(default_factory=lambda: tuple([10.0] * 7))
    max_joint_jerk_rad_s3: Tuple[float, ...] = field(default_factory=lambda: tuple([120.0] * 7))
    omega: float = 35.0
    damping_ratio: float = 1.0
    target_deadband_rad: float = 0.0005
    max_dt_s: float = 0.05
    joint_lower_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([-3.14] * 7))
    joint_upper_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([3.14] * 7))

    def __post_init__(self):
        for name, values in (
            ("max_joint_velocity_rad_s", self.max_joint_velocity_rad_s),
            ("max_joint_acceleration_rad_s2", self.max_joint_acceleration_rad_s2),
            ("max_joint_jerk_rad_s3", self.max_joint_jerk_rad_s3),
            ("joint_lower_limits_rad", self.joint_lower_limits_rad),
            ("joint_upper_limits_rad", self.joint_upper_limits_rad),
        ):
            if len(values) != self.joint_count:
                raise ValueError(f"{name} must contain {self.joint_count} values")


class FaJerkLimitedServoSmoother:
    """Dimension-configurable online joint servo for FA arms."""

    def __init__(self, config: Optional[FaJerkLimitedServoConfig] = None, name: str = "fa_arm"):
        self.config = config or FaJerkLimitedServoConfig()
        self.name = name
        self._position: Optional[np.ndarray] = None
        self._velocity = np.zeros(self.config.joint_count, dtype=np.float64)
        self._acceleration = np.zeros(self.config.joint_count, dtype=np.float64)
        self._last_time_s: Optional[float] = None

    def reset(self, current_joints_rad: Optional[Sequence[float]] = None) -> None:
        joints = None if current_joints_rad is None else self._as_joint_vector(current_joints_rad, "reset")
        self._position = None if joints is None else self._clamp_to_joint_limits(joints)
        self._velocity = np.zeros(self.config.joint_count, dtype=np.float64)
        self._acceleration = np.zeros(self.config.joint_count, dtype=np.float64)
        self._last_time_s = None

    def sample(
        self,
        target_joints_rad: Sequence[float],
        current_joints_rad: Optional[Sequence[float]] = None,
        now_s: Optional[float] = None,
    ) -> np.ndarray:
        now = time.time() if now_s is None else float(now_s)
        target = self._clamp_to_joint_limits(self._as_joint_vector(target_joints_rad, "target"))
        if not self.config.enabled:
            self.reset(target)
            return target
        current = (
            self._clamp_to_joint_limits(self._as_joint_vector(current_joints_rad, "current"))
            if current_joints_rad is not None
            else None
        )
        if self._position is None:
            self._position = current.copy() if current is not None else target.copy()
            self._last_time_s = now
            return self._position.copy()
        if self._last_time_s is None:
            self._last_time_s = now
            return self._position.copy()

        dt = max(1e-4, min(float(self.config.max_dt_s), now - self._last_time_s))
        self._last_time_s = now
        error = target - self._position
        max_vel = np.maximum(np.asarray(self.config.max_joint_velocity_rad_s, dtype=np.float64), 1e-6)
        max_acc = np.maximum(np.asarray(self.config.max_joint_acceleration_rad_s2, dtype=np.float64), 1e-6)
        max_jerk = np.maximum(np.asarray(self.config.max_joint_jerk_rad_s3, dtype=np.float64), 1e-6)
        desired_acc = (
            float(self.config.omega) ** 2 * error
            - 2.0 * float(self.config.damping_ratio) * float(self.config.omega) * self._velocity
        )
        desired_acc = np.clip(desired_acc, -max_acc, max_acc)
        acc_delta = np.clip(desired_acc - self._acceleration, -max_jerk * dt, max_jerk * dt)
        new_acc = np.clip(self._acceleration + acc_delta, -max_acc, max_acc)
        new_vel = np.clip(self._velocity + new_acc * dt, -max_vel, max_vel)
        new_pos = self._position + new_vel * dt
        overshoot = (target - self._position) * (target - new_pos) < 0.0
        if np.any(overshoot):
            new_pos[overshoot] = target[overshoot]
            new_vel[overshoot] = 0.0
            new_acc[overshoot] = 0.0
        close = (np.abs(target - new_pos) <= float(self.config.target_deadband_rad)) & (
            np.abs(new_vel) <= max_vel * 0.02
        )
        if np.any(close):
            new_pos[close] = target[close]
            new_vel[close] = 0.0
            new_acc[close] = 0.0
        self._position = self._clamp_to_joint_limits(new_pos)
        self._velocity = new_vel
        self._acceleration = new_acc
        return self._position.copy()

    def _clamp_to_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.config.joint_lower_limits_rad, dtype=np.float64)
        upper = np.asarray(self.config.joint_upper_limits_rad, dtype=np.float64)
        return np.clip(joints, lower, upper)

    def _as_joint_vector(self, values: Sequence[float], label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.config.joint_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain {self.config.joint_count} finite joint values")
        return array


class FaArmTrajectorySmoother:
    """Per-arm seventh-order minimum-snap trajectory replanner for FA."""

    def __init__(self, config: Optional[FaArmTrajectoryConfig] = None, name: str = "fa_arm"):
        self.config = config or FaArmTrajectoryConfig()
        self.name = name
        self._start: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None
        self._start_time_s = 0.0
        self._duration_s = self.config.segment_time_s
        self._last_sample: Optional[np.ndarray] = None

    def reset(self, current_joints_rad: Optional[Sequence[float]] = None) -> None:
        joints = None if current_joints_rad is None else self._as_joint_vector(current_joints_rad, "reset")
        self._start = None if joints is None else self._clamp_to_joint_limits(joints)
        self._goal = None if joints is None else self._clamp_to_joint_limits(joints)
        self._last_sample = None if joints is None else self._clamp_to_joint_limits(joints)
        self._start_time_s = time.time()
        self._duration_s = self.config.segment_time_s

    def sample(
        self,
        target_joints_rad: Sequence[float],
        current_joints_rad: Optional[Sequence[float]] = None,
        now_s: Optional[float] = None,
    ) -> np.ndarray:
        now = time.time() if now_s is None else float(now_s)
        target = self._clamp_to_joint_limits(self._as_joint_vector(target_joints_rad, "target"))
        if not self.config.enabled:
            self.reset(target)
            return target

        current = (
            self._clamp_to_joint_limits(self._as_joint_vector(current_joints_rad, "current"))
            if current_joints_rad is not None
            else None
        )
        if self._goal is None:
            start = current if current is not None else target
            self._start_min_snap_trajectory(start, target, now)
        elif np.max(np.abs(target - self._goal)) >= self.config.replan_threshold_rad:
            if self._trajectory_active(now):
                self._update_active_goal(target, now)
            else:
                self._start_min_snap_trajectory(self._sample_active(now), target, now)

        sample = self._sample_active(now)
        self._last_sample = sample.copy()
        return sample

    def _start_min_snap_trajectory(self, start: np.ndarray, goal: np.ndarray, now_s: float) -> None:
        self._start = self._clamp_to_joint_limits(start)
        self._goal = self._clamp_to_joint_limits(goal)
        self._start_time_s = now_s
        self._duration_s = self._compute_constrained_duration(self._goal - self._start)

    def _update_active_goal(self, goal: np.ndarray, now_s: float) -> None:
        active_sample = self._sample_active(now_s)
        goal = self._clamp_to_joint_limits(goal)
        elapsed_s = max(0.0, now_s - self._start_time_s)
        remaining_duration_s = self._compute_constrained_duration(goal - active_sample)
        duration_s = max(self._duration_s, elapsed_s + remaining_duration_s)
        tau = np.clip(elapsed_s / duration_s, 0.0, 1.0) if duration_s > 0.0 else 1.0
        blend = min_snap_position_blend(tau)
        if blend >= 1.0 - 1e-9:
            start = active_sample
        else:
            start = (active_sample - goal * blend) / (1.0 - blend)

        self._start = self._clamp_to_joint_limits(start)
        self._goal = goal
        self._duration_s = duration_s

    def _compute_constrained_duration(self, delta: np.ndarray) -> float:
        duration = max(self.config.segment_time_s, self.config.min_duration_s)
        abs_delta = np.abs(delta)
        max_vel = np.maximum(np.asarray(self.config.max_joint_velocity_rad_s, dtype=np.float64), 1e-6)
        max_acc = np.maximum(np.asarray(self.config.max_joint_acceleration_rad_s2, dtype=np.float64), 1e-6)
        duration = max(duration, float(np.max(abs_delta * _MIN_SNAP_MAX_S_DOT / max_vel)))
        duration = max(duration, float(np.max(np.sqrt(abs_delta * _MIN_SNAP_MAX_ABS_S_DDOT / max_acc))))
        return max(duration, self.config.min_duration_s)

    def _sample_active(self, now_s: float) -> np.ndarray:
        if self._start is None or self._goal is None:
            if self._last_sample is None:
                return np.zeros(self.config.joint_count, dtype=np.float64)
            return self._last_sample.copy()
        if self._duration_s <= 0.0:
            return self._goal.copy()
        tau = np.clip((now_s - self._start_time_s) / self._duration_s, 0.0, 1.0)
        blend = min_snap_position_blend(tau)
        return self._clamp_to_joint_limits(self._start + (self._goal - self._start) * blend)

    def _trajectory_active(self, now_s: float) -> bool:
        return self._start is not None and self._goal is not None and now_s < self._start_time_s + self._duration_s

    def _clamp_to_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.config.joint_lower_limits_rad, dtype=np.float64)
        upper = np.asarray(self.config.joint_upper_limits_rad, dtype=np.float64)
        return np.clip(joints, lower, upper)

    def _as_joint_vector(self, values: Sequence[float], label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.config.joint_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain {self.config.joint_count} finite joint values")
        return array


def min_snap_position_blend(progress: float) -> float:
    tau = min(1.0, max(0.0, float(progress)))
    return float(35.0 * tau**4 - 84.0 * tau**5 + 70.0 * tau**6 - 20.0 * tau**7)


__all__ = [
    "FaArmTrajectoryConfig",
    "FaArmTrajectorySmoother",
    "FaJerkLimitedServoConfig",
    "FaJerkLimitedServoSmoother",
    "min_snap_position_blend",
]
