#!/usr/bin/env python3
"""Replay recorded FA Cartesian targets through legacy and continuous IK policies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from beavr.teleop.components.interface.robots.fa_arm_ik_client import FaPybindIkClient
from beavr.teleop.components.interface.robots.fa_real_control import FaRealControl
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots
from beavr.teleop.configs.robots.fa_config import FaRealControlCfg


def load_targets(path: Path) -> tuple[str, np.ndarray, list[CartesianTarget]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"IK CSV contains no rows: {path}")
    side = rows[0]["hand_side"]
    initial_q = np.asarray([float(rows[0][f"current_q{i}"]) for i in range(7)], dtype=np.float64)
    targets = []
    seen = set()
    for row in rows:
        timestamp = float(row["target_timestamp_s"])
        if timestamp in seen:
            continue
        seen.add(timestamp)
        targets.append(
            CartesianTarget(
                timestamp_s=timestamp,
                hand_side=side,
                frame_id=row["frame_id"],
                position_m=tuple(float(row[f"target_{axis}"]) for axis in "xyz"),
                orientation_xyzw=tuple(float(row[f"target_q{axis}"]) for axis in "xyzw"),
                hand_command=int(row["hand_command"] or 0),
            )
        )
    return side, initial_q, targets


def make_controller(
    continuity_weight: float,
    continuity_nullspace_weight: float,
) -> tuple[FaRealControl, FaPybindIkClient]:
    config = FaRealControlCfg().config
    config.ik_continuity_weight = float(continuity_weight)
    config.ik = replace(
        config.ik,
        continuity_nullspace_weight=float(continuity_nullspace_weight),
        log_enabled=False,
        comfort_nullspace_log_enabled=False,
    )
    client = FaPybindIkClient(config.ik)
    controller = FaRealControl.__new__(FaRealControl)
    controller.config = config
    controller._ik_client = client
    controller._ik_clients = {robots.LEFT: client, robots.RIGHT: client}
    controller._approx_ik_target_cache = {robots.LEFT: None, robots.RIGHT: None}
    return controller, client


def rotation_error_rad(actual: np.ndarray, target_xyzw: tuple[float, ...]) -> float:
    x, y, z, w = target_xyzw
    target = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    cosine = float(np.clip((np.trace(actual.T @ target) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def record_step(metrics: dict, client: FaPybindIkClient, side: str, q: np.ndarray, target: CartesianTarget) -> None:
    previous = metrics["last_q"]
    if previous is not None:
        delta = np.abs(q - previous)
        metrics["max_steps"].append(float(np.max(delta)))
        metrics["path_length"] += float(np.linalg.norm(q - previous))
        metrics["joint_path_lengths"] += delta
    metrics["last_q"] = q.copy()
    pose = client.compute_fk(side, q)
    if pose is None:
        return
    metrics["position_errors"].append(float(np.linalg.norm(pose[:3, 3] - np.asarray(target.position_m))))
    metrics["orientation_errors"].append(rotation_error_rad(pose[:3, :3], target.orientation_xyzw))


def new_metrics(initial_q: np.ndarray) -> dict:
    return {
        "last_q": initial_q.copy(),
        "max_steps": [],
        "path_length": 0.0,
        "joint_path_lengths": np.zeros(7, dtype=np.float64),
        "position_errors": [],
        "orientation_errors": [],
        "holds": 0,
        "joint_clips": 0,
        "cartesian_fallbacks": 0,
        "outcomes": Counter(),
    }


def replay_legacy(path: Path) -> dict:
    side, q, targets = load_targets(path)
    controller, client = make_controller(0.0, 0.0)
    controller.config.max_ik_solution_jump_rad = 0.3
    metrics = new_metrics(q)
    previous_target = None
    for target in targets:
        accepted, best = controller._solve_best_arm_ik(side, target, q, q, client)
        if accepted is not None:
            q = np.asarray(accepted["solved"], dtype=np.float64)
            previous_target = target
        elif best is None:
            metrics["holds"] += 1
        elif best["quality_exceeds"]:
            fallback = controller._solve_reachable_ik_fallback(side, previous_target, target, q, client)
            if fallback is None:
                metrics["holds"] += 1
            else:
                ik, previous_target, _ = fallback
                q = np.asarray(ik.q_target, dtype=np.float64)
                metrics["cartesian_fallbacks"] += 1
        elif best["jump_exceeds"]:
            q = q + np.clip(np.asarray(best["solved"]) - q, -0.1, 0.1)
            previous_target = target
            metrics["joint_clips"] += 1
        else:
            q = np.asarray(best["solved"], dtype=np.float64)
            previous_target = target
        record_step(metrics, client, side, q, target)
    return summarize(path, side, "legacy", targets, metrics, controller)


def replay_continuous(path: Path, max_retries: int, continuity_nullspace_weight: float) -> dict:
    side, q, targets = load_targets(path)
    controller, client = make_controller(0.05, continuity_nullspace_weight)
    metrics = new_metrics(q)
    previous_target = None
    for target in targets:
        for _ in range(max_retries + 1):
            update = controller._solve_arm_ik_update(
                side,
                target,
                ("timestamp", target.timestamp_s),
                previous_target,
                q.copy(),
                q.copy(),
                q.copy(),
            )
            if update.warn_key and "cartesian_fallback" in update.warn_key:
                metrics["cartesian_fallbacks"] += 1
            metrics["outcomes"][update.warn_key or "accepted"] += 1
            if not update.dirty:
                metrics["holds"] += 1
            if update.active_goal is not None:
                q = np.asarray(update.active_goal, dtype=np.float64)
            if update.last_ik_target is not None:
                previous_target = update.last_ik_target
            if not update.retry_target:
                break
        record_step(metrics, client, side, q, target)
    return summarize(path, side, "continuous", targets, metrics, controller)


def summarize(path: Path, side: str, policy: str, targets: list, metrics: dict, controller) -> dict:
    steps = np.asarray(metrics["max_steps"], dtype=np.float64)
    pos = np.asarray(metrics["position_errors"], dtype=np.float64)
    ori = np.asarray(metrics["orientation_errors"], dtype=np.float64)
    reachable = (pos <= controller.config.ik_max_position_error_m) & (
        ori <= controller.config.ik_max_orientation_error_rad
    )
    return {
        "file": str(path),
        "side": side,
        "policy": policy,
        "targets": len(targets),
        "eps": controller.config.ik.eps,
        "continuity_nullspace_weight": controller.config.ik.continuity_nullspace_weight,
        "joint_step_p95_rad": float(np.quantile(steps, 0.95)) if steps.size else 0.0,
        "joint_step_max_rad": float(np.max(steps)) if steps.size else 0.0,
        "joint_steps_over_0_3": int(np.sum(steps > 0.3)),
        "joint_path_length_rad": metrics["path_length"],
        "joint_path_length_by_index_rad": metrics["joint_path_lengths"].tolist(),
        "fk_position_p95_m": float(np.quantile(pos, 0.95)) if pos.size else 0.0,
        "fk_position_max_m": float(np.max(pos)) if pos.size else 0.0,
        "fk_orientation_p95_rad": float(np.quantile(ori, 0.95)) if ori.size else 0.0,
        "fk_orientation_max_rad": float(np.max(ori)) if ori.size else 0.0,
        "reachable_rate": float(np.mean(reachable)) if reachable.size else 0.0,
        "holds": metrics["holds"],
        "joint_clips": metrics["joint_clips"],
        "cartesian_fallbacks": metrics["cartesian_fallbacks"],
        "outcomes": dict(metrics["outcomes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--continuity-nullspace-weight", type=float, default=0.01)
    args = parser.parse_args()
    results = []
    for path in args.csv:
        results.append(replay_legacy(path))
        results.append(
            replay_continuous(
                path,
                max(0, args.max_retries),
                max(0.0, args.continuity_nullspace_weight),
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
