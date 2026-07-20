#!/usr/bin/python3
"""Replay a local LeRobot v3.0 FA episode into MoveIt demo RViz controllers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path("/home/likunwei/dataCollection/beavr-bot/datasets/data_show_final6")
DEFAULT_EPISODE = 2
DEFAULT_START_DELAY_S = 1.0

LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a local LeRobot v3.0 FA dataset episode and send its two arm joint "
            "trajectory to the ros2_control JointTrajectoryController instances created "
            "by `ros2 launch fa_moveit2_config demo.launch.py`."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE)
    parser.add_argument("--left-action", default="/leftArm_controller/follow_joint_trajectory")
    parser.add_argument("--right-action", default="/rightArm_controller/follow_joint_trajectory")
    parser.add_argument("--speed-scale", type=float, default=1.0, help=">1 is faster, <1 is slower.")
    parser.add_argument("--start-delay-s", type=float, default=DEFAULT_START_DELAY_S)
    parser.add_argument("--goal-time-tolerance-s", type=float, default=2.0)
    parser.add_argument(
        "--source",
        choices=("action", "observation.state"),
        default="action",
        help="Dataset vector column to replay. `action` is the recorded next joint target.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only load and summarize the episode.")
    return parser.parse_args()


def _load_v30_episode(root: Path, episode: int, source: str) -> tuple[np.ndarray, np.ndarray, int, Path]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot v3.0 info file: {info_path}")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected codebase_version v3.0 in {info_path}, got {info.get('codebase_version')!r}")

    episodes_parts = [
        pd.read_parquet(path)
        for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ]
    if not episodes_parts:
        raise FileNotFoundError(f"No episode metadata parquet files under {root / 'meta' / 'episodes'}")
    episodes_df = pd.concat(episodes_parts, ignore_index=True)

    matches = episodes_df[episodes_df["episode_index"] == int(episode)]
    if matches.empty:
        available = sorted(int(v) for v in episodes_df["episode_index"].tolist())
        raise ValueError(f"Episode {episode} not found. Available episodes: {available}")

    episode_row = matches.iloc[0]
    data_path = root / info["data_path"].format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing episode data parquet: {data_path}")

    data_df = pd.read_parquet(data_path)
    if "episode_index" in data_df.columns:
        data_df = data_df[data_df["episode_index"] == int(episode)]
    if data_df.empty:
        raise ValueError(f"No rows for episode {episode} in {data_path}")
    if "frame_index" in data_df.columns:
        data_df = data_df.sort_values("frame_index")
    if source not in data_df.columns:
        raise ValueError(f"Column {source!r} not found in {data_path}; columns={list(data_df.columns)}")

    vectors = np.stack([np.asarray(value, dtype=np.float64).reshape(-1) for value in data_df[source]])
    if vectors.ndim != 2 or vectors.shape[1] < 14:
        raise ValueError(f"Expected {source!r} vectors with at least 14 values, got shape {vectors.shape}")
    if not np.isfinite(vectors[:, :14]).all():
        raise ValueError(f"Episode {episode} contains NaN or Inf in the first 14 arm joints")

    if "timestamp" in data_df.columns:
        timestamps = data_df["timestamp"].to_numpy(dtype=np.float64)
    else:
        fps = int(info.get("fps") or 30)
        timestamps = np.arange(len(vectors), dtype=np.float64) / float(fps)

    timestamps = _normalize_timestamps(timestamps, int(info.get("fps") or 30))
    return timestamps, vectors[:, :14], int(info.get("fps") or 30), data_path


def _normalize_timestamps(timestamps: np.ndarray, fps: int) -> np.ndarray:
    if timestamps.size == 0:
        raise ValueError("Episode contains no frames")

    normalized = timestamps.astype(np.float64, copy=True)
    normalized -= normalized[0]
    fallback_step = 1.0 / float(fps or 30)
    for idx in range(1, len(normalized)):
        if not math.isfinite(float(normalized[idx])) or normalized[idx] <= normalized[idx - 1]:
            normalized[idx] = normalized[idx - 1] + fallback_step
    normalized[0] = max(0.0, normalized[0])
    return normalized


def _duration_to_msg(seconds: float):
    from builtin_interfaces.msg import Duration

    if seconds < 0.0:
        raise ValueError(f"Duration must be non-negative, got {seconds}")
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def _build_goal(
    *,
    joint_names: Sequence[str],
    positions: np.ndarray,
    timestamps: np.ndarray,
    speed_scale: float,
    goal_time_tolerance_s: float,
):
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    if speed_scale <= 0.0:
        raise ValueError(f"--speed-scale must be > 0, got {speed_scale}")

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    goal.goal_time_tolerance = _duration_to_msg(goal_time_tolerance_s)

    for row, timestamp_s in zip(positions, timestamps, strict=True):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in row]
        point.time_from_start = _duration_to_msg(float(timestamp_s) / speed_scale)
        goal.trajectory.points.append(point)
    return goal


def _stamp_goal_header(node, goal, start_delay_s: float) -> None:
    from rclpy.duration import Duration

    stamp = node.get_clock().now() + Duration(seconds=float(start_delay_s))
    goal.trajectory.header.stamp = stamp.to_msg()


def _send_goal(node, action_client, goal, label: str, timeout_s: float):
    import rclpy

    if not action_client.wait_for_server(timeout_sec=timeout_s):
        raise RuntimeError(f"Timed out waiting for {label} action server: {action_client._action_name}")

    future = action_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    goal_handle = future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError(f"{label} trajectory goal was rejected")
    return goal_handle


def _wait_result(node, goal_handle, label: str) -> None:
    import rclpy

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result()
    if result is None:
        raise RuntimeError(f"{label} trajectory finished without a result")
    if result.result.error_code != 0:
        raise RuntimeError(
            f"{label} trajectory failed with error_code={result.result.error_code}: "
            f"{result.result.error_string}"
        )


def _replay_to_rviz(
    timestamps: np.ndarray,
    arm_positions: np.ndarray,
    *,
    left_action: str,
    right_action: str,
    speed_scale: float,
    start_delay_s: float,
    goal_time_tolerance_s: float,
) -> None:
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient

    rclpy.init(args=None)
    node = rclpy.create_node("fa_v30_episode_rviz_replay")
    try:
        left_goal = _build_goal(
            joint_names=LEFT_ARM_JOINT_NAMES,
            positions=arm_positions[:, :7],
            timestamps=timestamps,
            speed_scale=speed_scale,
            goal_time_tolerance_s=goal_time_tolerance_s,
        )
        right_goal = _build_goal(
            joint_names=RIGHT_ARM_JOINT_NAMES,
            positions=arm_positions[:, 7:14],
            timestamps=timestamps,
            speed_scale=speed_scale,
            goal_time_tolerance_s=goal_time_tolerance_s,
        )
        _stamp_goal_header(node, left_goal, start_delay_s)
        _stamp_goal_header(node, right_goal, start_delay_s)

        left_client = ActionClient(node, FollowJointTrajectory, left_action)
        right_client = ActionClient(node, FollowJointTrajectory, right_action)

        print(f"Waiting for action servers: {left_action}, {right_action}")
        left_handle = _send_goal(node, left_client, left_goal, "left arm", 10.0)
        right_handle = _send_goal(node, right_client, right_goal, "right arm", 10.0)
        duration_s = float(timestamps[-1] - timestamps[0]) / speed_scale
        print(f"Replay started: {len(timestamps)} frames, duration={duration_s:.3f}s")

        _wait_result(node, left_handle, "left arm")
        _wait_result(node, right_handle, "right arm")
        print("Replay finished successfully.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    args = _parse_args()
    timestamps, arm_positions, fps, data_path = _load_v30_episode(
        args.dataset_root, args.episode, args.source
    )

    print(
        f"Loaded episode {args.episode} from {data_path}: "
        f"frames={len(timestamps)}, dataset_fps={fps}, "
        f"duration={timestamps[-1] / args.speed_scale:.3f}s, source={args.source}"
    )
    print(f"First arm target: {np.array2string(arm_positions[0], precision=4)}")
    print(f"Last arm target:  {np.array2string(arm_positions[-1], precision=4)}")

    if args.dry_run:
        return

    _replay_to_rviz(
        timestamps,
        arm_positions,
        left_action=args.left_action,
        right_action=args.right_action,
        speed_scale=args.speed_scale,
        start_delay_s=args.start_delay_s,
        goal_time_tolerance_s=args.goal_time_tolerance_s,
    )


if __name__ == "__main__":
    main()
