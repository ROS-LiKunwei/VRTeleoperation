"""Export one committed FA AHA rollout to the cache-only LeRobot v3 layout.

The visual streams remain in the source episode (or are uploaded beside this
export).  This exporter mirrors the state/action parquet contract used by the
FA RGB-only cache datasets: observation is measured state[t], and action is
measured state[t+1].  The default exports the autonomous prefix.  Passing
``--include-human`` exports one continuous assisted rollout made from the model
and human-commanded frames while removing startup and SAFE_HOLD repeats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd


STATE_NAMES = [
    *(f"fa_left_joint_{index}" for index in range(7)),
    *(f"fa_right_joint_{index}" for index in range(7)),
    "left_hand_gripper_state",
    "right_hand_gripper_state",
]
ACTION_NAMES = [
    *(f"fa_left_next_joint_{index}_rad" for index in range(7)),
    *(f"fa_right_next_joint_{index}_rad" for index in range(7)),
    "left_hand_next_gripper_state",
    "right_hand_next_gripper_state",
]


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, list]:
    array = np.asarray(values)
    axis = 0
    return {
        "min": np.min(array, axis=axis).reshape(-1).tolist(),
        "max": np.max(array, axis=axis).reshape(-1).tolist(),
        "mean": np.mean(array, axis=axis, dtype=np.float64).reshape(-1).tolist(),
        "std": np.std(array, axis=axis, dtype=np.float64).reshape(-1).tolist(),
        "count": [int(array.shape[0])],
    }


def _feature_stats(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, dict[str, list]]:
    length = len(states)
    frame_index = np.arange(length, dtype=np.int64)
    zero = np.zeros(length, dtype=np.int64)
    return {
        "observation.state": _stats(states),
        "action": _stats(actions),
        "timestamp": _stats(timestamps[:, None]),
        "frame_index": _stats(frame_index[:, None]),
        "episode_index_stats": _stats(zero[:, None]),
        "index": _stats(frame_index[:, None]),
        "task_index": _stats(zero[:, None]),
    }


def _video_feature(shape: list[int]) -> dict:
    height, width, channels = shape
    return {
        "dtype": "video",
        "shape": [height, width, channels],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 5,
            "video.channels": channels,
            "has_audio": False,
            "source_encoding": "segmented_h264_yuv420p",
        },
    }


def _contiguous_ranges(indices: list[int]) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append(
                {
                    "raw_start_frame_inclusive": start,
                    "raw_stop_frame_exclusive": previous + 1,
                }
            )
            start = index
        previous = index
    ranges.append(
        {
            "raw_start_frame_inclusive": start,
            "raw_stop_frame_exclusive": previous + 1,
        }
    )
    return ranges


def export_episode(
    source: Path,
    output: Path,
    task: str,
    *,
    include_human: bool = False,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not (source / "COMMITTED").is_file():
        raise ValueError("source episode is not COMMITTED")
    metadata = _load_json(source / "metadata.json")
    outcome = _load_json(source / "outcome.json")
    frames = _load_jsonl(source / "frames.jsonl")
    if len(frames) != int(metadata["frame_count"]):
        raise ValueError("frames.jsonl count differs from metadata")
    states_all = np.asarray([row["state"] for row in frames], dtype=np.float32)
    if states_all.shape != (len(frames), 16) or not np.isfinite(states_all).all():
        raise ValueError("source states must be finite [N,16] float32")

    task_frames = [
        index
        for index, row in enumerate(frames)
        if row.get("sent_command_source") in {"model", "human"}
    ]
    if not task_frames:
        raise ValueError("episode has no model/human task command")
    if include_human:
        if outcome.get("outcome") != "success":
            raise ValueError("assisted export requires a successful episode")
        selected = task_frames
    else:
        start = task_frames[0]
        stop = next(
            (
                index
                for index in range(start + 1, len(frames))
                if frames[index].get("intervention", {}).get("mode") != "AUTO"
            ),
            len(frames),
        )
        selected = list(range(start, stop))
    if len(selected) < 2:
        raise ValueError("trimmed episode has no state transition")

    # Each row is observation at one retained frame and measured action at the
    # next retained frame.  With --include-human this joins the autonomous and
    # human runs across the removed stationary SAFE_HOLD interval.
    observations = states_all[selected[:-1]]
    actions = states_all[selected[1:]]
    length = len(observations)
    fps = int(metadata["fps"])
    timestamps = np.arange(length, dtype=np.float32) / float(fps)
    indices = np.arange(length, dtype=np.int64)
    zeros = np.zeros(length, dtype=np.int64)

    features = datasets.Features(
        {
            "observation.state": datasets.Sequence(
                datasets.Value("float32"), length=16
            ),
            "action": datasets.Sequence(datasets.Value("float32"), length=16),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
        }
    )
    table = datasets.Dataset.from_dict(
        {
            "observation.state": observations,
            "action": actions,
            "timestamp": timestamps,
            "frame_index": indices,
            "episode_index": zeros,
            "index": indices,
            "task_index": zeros,
        },
        features=features,
    )

    data_dir = output / "data" / "chunk-000"
    episodes_dir = output / "meta" / "episodes" / "chunk-000"
    episode_stats_dir = output / "meta" / "episodes_stats" / "chunk-000"
    data_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)
    episode_stats_dir.mkdir(parents=True)
    table.to_parquet(data_dir / "file-000.parquet")

    camera_shapes = metadata["camera_shapes"]
    camera_keys = list(metadata["camera_keys"])
    info_features = {
        "observation.state": {
            "shape": [16],
            "dtype": "float32",
            "names": STATE_NAMES,
        },
        "action": {"shape": [16], "dtype": "float32", "names": ACTION_NAMES},
    }
    for key in camera_keys:
        feature = _video_feature(camera_shapes[key])
        feature["info"]["video.fps"] = fps
        info_features[f"observation.images.{key}"] = feature
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        info_features[key] = {"dtype": dtype, "shape": [1], "names": None}

    info = {
        "codebase_version": "v3.0",
        "robot_type": "fa",
        "total_episodes": 1,
        "total_frames": length,
        "total_tasks": 1,
        "total_videos": len(camera_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": info_features,
        "episodes_path": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "episodes_stats_path": "meta/episodes_stats/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    meta_dir = output / "meta"
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    pd.DataFrame([{"task_index": 0, "task": task}]).to_parquet(
        meta_dir / "tasks.parquet", index=False
    )

    episode_row = {
        "episode_index": 0,
        "tasks": [task],
        "length": length,
        "dataset_from_index": 0,
        "dataset_to_index": length,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "data/from_index": 0,
        "data/to_index": length,
    }
    for key in camera_keys:
        prefix = f"videos/observation.images.{key}"
        episode_row.update(
            {
                f"{prefix}/chunk_index": 0,
                f"{prefix}/file_index": 0,
                f"{prefix}/from_timestamp": 0.0,
                f"{prefix}/to_timestamp": float(length / fps),
            }
        )
    pd.DataFrame([episode_row]).to_parquet(
        episodes_dir / "file-000.parquet", index=False
    )
    stats = _feature_stats(observations, actions, timestamps)
    pd.DataFrame([{"episode_index": 0, **stats}]).to_parquet(
        episode_stats_dir / "file-000.parquet", index=False
    )
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    camera_settings = source.parents[4] / "configs/cameras/sysmo32_rgbd_aligned_native.json"
    if camera_settings.is_file():
        shutil.copy2(camera_settings, meta_dir / "camera_settings.json")

    ranges = _contiguous_ranges(selected)
    skipped_bridges = []
    for left, right in zip(selected[:-1], selected[1:]):
        if right == left + 1:
            continue
        delta = np.abs(states_all[right] - states_all[left])
        skipped_bridges.append(
            {
                "raw_left_frame": left,
                "raw_right_frame": right,
                "skipped_frame_count": right - left - 1,
                "elapsed_s": float(
                    frames[right]["captured_at"] - frames[left]["captured_at"]
                ),
                "max_abs_state_delta": float(delta.max(initial=0.0)),
            }
        )
    command_source_counts = {
        source_name: sum(
            frames[index].get("sent_command_source") == source_name
            for index in selected
        )
        for source_name in ("model", "human")
    }
    report = {
        "schema": "fa_rgb_only_cache_dataset_v1",
        "source_episode_uuid": metadata["episode_uuid"],
        "source_episode_path_after_upload": "raw_episode",
        "source_outcome": outcome["outcome"],
        "source_outcome_reason": outcome["outcome_reason"],
        "source_frame_count": len(frames),
        "source_fps": fps,
        "trim": {
            "selection": (
                "model_and_human_commanded_frames"
                if include_human
                else "autonomous_contiguous_prefix"
            ),
            "retained_raw_frame_ranges": ranges,
            "dropped_frame_count": len(frames) - len(selected),
            "exported_transition_count": length,
            "interior_frames_removed": sum(
                item["skipped_frame_count"] for item in skipped_bridges
            ),
            "skipped_interval_bridges": skipped_bridges,
            "retained_command_source_frame_counts": command_source_counts,
        },
        "training_contract": {
            "observation.state": "source frames[t].state",
            "action": "source frames[t+1].state",
            "sent_command_used_as_action": False,
        },
        "rgb_camera_keys": [f"observation.images.{key}" for key in camera_keys],
        "depth_preserved_in_raw_episode": True,
        "videos_omitted_from_cache_dataset": True,
        "must_not_merge_into_success_only_dataset": outcome["outcome"] != "success",
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "source_frames_sha256": _sha256(source / "frames.jsonl"),
    }
    (meta_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="teleoperate FA")
    parser.add_argument(
        "--include-human",
        action="store_true",
        help=(
            "export model and human-commanded frames as one assisted rollout, "
            "dropping startup and SAFE_HOLD repeats"
        ),
    )
    args = parser.parse_args()
    report = export_episode(
        args.source,
        args.output,
        args.task,
        include_human=args.include_human,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
