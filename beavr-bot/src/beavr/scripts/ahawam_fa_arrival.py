"""Feedback arrival barriers for FA chunk execution and subsequent camera input."""

from collections import deque
import math
import time

import numpy as np


def wait_for_robot_arrival(
    ros,
    target,
    *,
    tolerance_rad: float = 0.02,
    stability_rad: float = 0.005,
    stable_seconds: float = 0.2,
    timeout_s: float = 15.0,
    state_timeout_s: float = 5.0,
    max_state_age_s: float = 0.2,
    newer_than_sequence=None,
):
    """Wait for fresh, stable 14-joint feedback without publishing commands."""

    limits = (
        tolerance_rad,
        stability_rad,
        stable_seconds,
        timeout_s,
        state_timeout_s,
        max_state_age_s,
    )
    if any(not math.isfinite(value) or value <= 0 for value in limits):
        raise ValueError("arrival limits must be finite and positive")
    if stable_seconds >= timeout_s:
        raise ValueError("arrival timeout must exceed the stability window")
    goal = (
        None
        if target is None
        else np.asarray(target, dtype=np.float64).reshape(-1)[:14]
    )
    if goal is not None and (
        goal.shape != (14,) or not np.isfinite(goal).all()
    ):
        raise ValueError("arrival target must contain 14 finite joint positions")

    started = time.monotonic()
    deadline = started + timeout_s
    history = deque()
    sequence = newer_than_sequence
    error = None
    spread = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"robot did not settle within {timeout_s}s: "
                f"max_error_rad={error}, spread_rad={spread}; "
                "no next observation/inference"
            )
        try:
            state, metadata = ros.state_snapshot(
                min(state_timeout_s, remaining),
                max_state_age_s,
                newer_than_sequence=sequence,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "arrival wait lost fresh joint feedback; "
                "no next observation/inference"
            ) from exc
        now = time.monotonic()
        if now > deadline:
            raise TimeoutError(
                "robot arrival deadline exceeded; no next observation/inference"
            )
        new_sequence = int(metadata["sequence"])
        if sequence is not None and new_sequence <= sequence:
            raise RuntimeError(
                "arrival wait requires strictly newer joint feedback"
            )
        sequence = new_sequence
        age_s = float(metadata["source_age_ms"]) / 1000.0
        if not math.isfinite(age_s) or not 0 <= age_s <= max_state_age_s:
            raise RuntimeError(
                "arrival wait received stale or invalid joint feedback"
            )
        sample_time = now - age_s
        if sample_time < started:
            continue
        joints = np.asarray(state, dtype=np.float64).reshape(-1)[:14]
        if joints.shape != (14,) or not np.isfinite(joints).all():
            raise RuntimeError(
                "arrival wait requires 14 finite measured joints"
            )
        error = None if goal is None else float(np.abs(joints - goal).max())
        if error is not None and error > tolerance_rad:
            history.clear()
            continue
        history.append((sample_time, joints.copy()))
        cutoff = sample_time - stable_seconds
        while len(history) > 1 and history[1][0] <= cutoff:
            history.popleft()
        spread = float(
            np.ptp(np.stack([row[1] for row in history]), axis=0).max()
        )
        if (
            len(history) >= 3
            and sample_time - history[0][0] >= stable_seconds
            and spread <= stability_rad
        ):
            return {
                "joint_state_sequence": sequence,
                "max_error_rad": error,
                "spread_rad": spread,
                "wait_s": now - started,
                "ready_at_monotonic_s": time.monotonic(),
                "mode": (
                    "target_reached"
                    if goal is not None
                    else "initial_stationary"
                ),
            }
