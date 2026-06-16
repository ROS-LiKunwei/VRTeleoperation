"""Joint trajectory helpers for SYSMO-32 arms."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MIN_SNAP_MAX_S_DOT = 2.1875
_MIN_SNAP_MAX_ABS_S_DDOT = 7.513188404399293


@dataclass(frozen=True)
class Sysmo32ArmTrajectoryConfig:
    enabled: bool = True
    segment_time_s: float = 0.18
    min_duration_s: float = 0.06
    replan_threshold_rad: float = 0.0005
    max_joint_velocity_rad_s: Tuple[float, ...] = field(default_factory=lambda: tuple([3.0] * 6))
    max_joint_acceleration_rad_s2: Tuple[float, ...] = field(default_factory=lambda: tuple([12.0] * 6))
    joint_lower_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([-3.14] * 6))
    joint_upper_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([3.14] * 6))

    def __post_init__(self):
        if len(self.max_joint_velocity_rad_s) != 6:
            raise ValueError("max_joint_velocity_rad_s must contain 6 values")
        if len(self.max_joint_acceleration_rad_s2) != 6:
            raise ValueError("max_joint_acceleration_rad_s2 must contain 6 values")
        if len(self.joint_lower_limits_rad) != 6 or len(self.joint_upper_limits_rad) != 6:
            raise ValueError("joint limits must contain 6 values")
        object.__setattr__(self, "segment_time_s", max(0.0, float(self.segment_time_s)))
        object.__setattr__(self, "min_duration_s", max(1e-4, float(self.min_duration_s)))
        object.__setattr__(self, "replan_threshold_rad", max(0.0, float(self.replan_threshold_rad)))


@dataclass(frozen=True)
class Sysmo32JerkLimitedServoConfig:
    enabled: bool = True
    max_joint_velocity_rad_s: Tuple[float, ...] = field(default_factory=lambda: tuple([3.0] * 6))
    max_joint_acceleration_rad_s2: Tuple[float, ...] = field(default_factory=lambda: tuple([10.0] * 6))
    max_joint_jerk_rad_s3: Tuple[float, ...] = field(default_factory=lambda: tuple([120.0] * 6))
    omega: float = 35.0
    damping_ratio: float = 1.0
    target_deadband_rad: float = 0.0005
    max_dt_s: float = 0.05
    joint_lower_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([-3.14] * 6))
    joint_upper_limits_rad: Tuple[float, ...] = field(default_factory=lambda: tuple([3.14] * 6))

    def __post_init__(self):
        for name, values in (
            ("max_joint_velocity_rad_s", self.max_joint_velocity_rad_s),
            ("max_joint_acceleration_rad_s2", self.max_joint_acceleration_rad_s2),
            ("max_joint_jerk_rad_s3", self.max_joint_jerk_rad_s3),
        ):
            if len(values) != 6:
                raise ValueError(f"{name} must contain 6 values")
        if len(self.joint_lower_limits_rad) != 6 or len(self.joint_upper_limits_rad) != 6:
            raise ValueError("joint limits must contain 6 values")
        object.__setattr__(self, "omega", max(0.0, float(self.omega)))
        object.__setattr__(self, "damping_ratio", max(0.0, float(self.damping_ratio)))
        object.__setattr__(self, "target_deadband_rad", max(0.0, float(self.target_deadband_rad)))
        object.__setattr__(self, "max_dt_s", max(1e-4, float(self.max_dt_s)))


class Sysmo32JerkLimitedServoSmoother:
    """Online joint servo with velocity, acceleration and jerk limits.

    This smoother is intended for VR teleoperation where the target changes
    every frame. It tracks the latest target directly instead of creating a
    point-to-point trajectory for every small target update.
    """

    def __init__(
        self,
        config: Optional[Sysmo32JerkLimitedServoConfig] = None,
        name: str = "arm",
    ) -> None:
        self.config = config or Sysmo32JerkLimitedServoConfig()
        self.name = name
        self._position: Optional[np.ndarray] = None
        self._velocity = np.zeros(6, dtype=np.float64)
        self._acceleration = np.zeros(6, dtype=np.float64)
        self._last_time_s: Optional[float] = None

    def reset(self, current_joints_rad: Optional[Sequence[float]] = None) -> None:
        joints = None if current_joints_rad is None else self._as_joint_vector(current_joints_rad, "reset")
        self._position = None if joints is None else self._clamp_to_joint_limits(joints)
        self._velocity = np.zeros(6, dtype=np.float64)
        self._acceleration = np.zeros(6, dtype=np.float64)
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

        dt = max(1e-4, min(self.config.max_dt_s, now - self._last_time_s))
        self._last_time_s = now

        error = target - self._position
        max_vel = np.maximum(np.asarray(self.config.max_joint_velocity_rad_s, dtype=np.float64), 1e-6)
        max_acc = np.maximum(np.asarray(self.config.max_joint_acceleration_rad_s2, dtype=np.float64), 1e-6)
        max_jerk = np.maximum(np.asarray(self.config.max_joint_jerk_rad_s3, dtype=np.float64), 1e-6)

        desired_acceleration = (
            self.config.omega**2 * error
            - 2.0 * self.config.damping_ratio * self.config.omega * self._velocity
        )
        desired_acceleration = np.clip(desired_acceleration, -max_acc, max_acc)
        acceleration_delta = np.clip(
            desired_acceleration - self._acceleration,
            -max_jerk * dt,
            max_jerk * dt,
        )
        new_acceleration = np.clip(self._acceleration + acceleration_delta, -max_acc, max_acc)
        new_velocity = np.clip(self._velocity + new_acceleration * dt, -max_vel, max_vel)
        new_position = self._position + new_velocity * dt

        overshoot = (target - self._position) * (target - new_position) < 0.0
        if np.any(overshoot):
            new_position[overshoot] = target[overshoot]
            new_velocity[overshoot] = 0.0
            new_acceleration[overshoot] = 0.0

        close = (np.abs(target - new_position) <= self.config.target_deadband_rad) & (
            np.abs(new_velocity) <= max_vel * 0.02
        )
        if np.any(close):
            new_position[close] = target[close]
            new_velocity[close] = 0.0
            new_acceleration[close] = 0.0

        self._position = self._clamp_to_joint_limits(new_position)
        self._velocity = new_velocity
        self._acceleration = new_acceleration
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self._acceleration.copy()

    def _clamp_to_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.config.joint_lower_limits_rad, dtype=np.float64)
        upper = np.asarray(self.config.joint_upper_limits_rad, dtype=np.float64)
        return np.clip(joints, lower, upper)

    @staticmethod
    def _as_joint_vector(values: Sequence[float], label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain 6 finite joint values")
        return array


class Sysmo32ArmTrajectorySmoother:
    """Per-arm seventh-order minimum-snap trajectory replanner.

    New IK targets replan a trajectory from the current actual command sample
    to the limited target. The trajectory duration is stretched so the blend
    respects configured joint velocity and acceleration limits.
    """

    def __init__(
        self,
        config: Optional[Sysmo32ArmTrajectoryConfig] = None,
        name: str = "arm",
    ) -> None:
        self.config = config or Sysmo32ArmTrajectoryConfig()
        self.name = name
        self._start: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None
        self._start_time_s = 0.0
        self._duration_s = self.config.segment_time_s
        self._last_sample: Optional[np.ndarray] = None

    def reset(self, current_joints_rad: Optional[Sequence[float]] = None) -> None:
        joints = None if current_joints_rad is None else self._as_joint_vector(current_joints_rad, "reset")
        self._start = None if joints is None else joints.copy()
        self._goal = None if joints is None else joints.copy()
        self._last_sample = None if joints is None else joints.copy()
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
                start = self._sample_active(now)
                self._start_min_snap_trajectory(start, target, now)

        sample = self._sample_active(now)
        self._last_sample = sample.copy()
        return sample

    def _start_min_snap_trajectory(self, start: np.ndarray, goal: np.ndarray, now_s: float) -> None:
        self._start = self._clamp_to_joint_limits(start)
        self._goal = self._clamp_to_joint_limits(goal)
        self._start_time_s = now_s
        self._duration_s = self._compute_constrained_duration(self._goal - self._start)
        logger.debug(
            "SYSMO-32 %s min-snap trajectory duration=%.3fs delta_max=%.4frad",
            self.name,
            self._duration_s,
            float(np.max(np.abs(self._goal - self._start))),
        )

    def _update_active_goal(self, goal: np.ndarray, now_s: float) -> None:
        active_sample = self._sample_active(now_s)
        goal = self._clamp_to_joint_limits(goal)
        elapsed_s = max(0.0, now_s - self._start_time_s)
        remaining_duration_s = self._compute_constrained_duration(goal - active_sample)
        duration_s = max(self._duration_s, elapsed_s + remaining_duration_s)
        tau = np.clip(elapsed_s / duration_s, 0.0, 1.0) if duration_s > 0.0 else 1.0
        blend = self._min_snap_position_blend(tau)
        if blend >= 1.0 - 1e-9:
            start = active_sample
        else:
            start = (active_sample - goal * blend) / (1.0 - blend)

        self._start = start
        self._goal = goal
        self._duration_s = duration_s
        logger.debug(
            "SYSMO-32 %s min-snap trajectory goal updated duration=%.3fs remaining=%.3fs delta_max=%.4frad",
            self.name,
            self._duration_s,
            max(0.0, self._duration_s - elapsed_s),
            float(np.max(np.abs(goal - active_sample))),
        )

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
                return np.zeros(6, dtype=np.float64)
            return self._last_sample.copy()
        if self._duration_s <= 0.0:
            return self._goal.copy()
        tau = np.clip((now_s - self._start_time_s) / self._duration_s, 0.0, 1.0)
        blend = self._min_snap_position_blend(tau)
        return self._clamp_to_joint_limits(self._start + (self._goal - self._start) * blend)

    def _trajectory_active(self, now_s: float) -> bool:
        return self._start is not None and self._goal is not None and now_s < self._start_time_s + self._duration_s

    @staticmethod
    def _min_snap_position_blend(tau: float) -> float:
        return float(35.0 * tau**4 - 84.0 * tau**5 + 70.0 * tau**6 - 20.0 * tau**7)

    def _clamp_to_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.config.joint_lower_limits_rad, dtype=np.float64)
        upper = np.asarray(self.config.joint_upper_limits_rad, dtype=np.float64)
        return np.clip(joints, lower, upper)

    @staticmethod
    def _as_joint_vector(values: Sequence[float], label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain 6 finite joint values")
        return array


__all__ = [
    "Sysmo32ArmTrajectoryConfig",
    "Sysmo32ArmTrajectorySmoother",
    "Sysmo32JerkLimitedServoConfig",
    "Sysmo32JerkLimitedServoSmoother",
]
