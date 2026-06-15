"""Seventh-order minimum-snap joint trajectory helpers for SYSMO-32 arms."""

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


__all__ = ["Sysmo32ArmTrajectoryConfig", "Sysmo32ArmTrajectorySmoother"]
