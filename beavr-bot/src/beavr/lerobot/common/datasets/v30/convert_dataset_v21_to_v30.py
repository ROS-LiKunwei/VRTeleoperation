#!/usr/bin/env python

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from beavr.lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    INFO_PATH,
    load_json,
    load_jsonlines,
    write_json,
)

CODEBASE_VERSION_V30 = "v3.0"
V30_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V30_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
V30_TASKS_PATH = "meta/tasks.parquet"
V30_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V30_EPISODES_STATS_PATH = "meta/episodes_stats/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

logger = logging.getLogger(__name__)


def convert_dataset_v21_to_v30(
    dataset_root: str | Path,
    *,
    keep_v21_backup: bool = False,
) -> Path:
    """Convert a locally recorded LeRobot v2.1 dataset directory to v3.0 layout.

    The current recorder writes reliable v2.1 episodes incrementally.  This helper
    performs the v3.0 finalization step after recording by reorganizing the files
    into file-based parquet/video paths and parquet metadata tables.
    """

    root = Path(dataset_root)
    info = load_json(root / INFO_PATH)
    version = str(info.get("codebase_version", ""))
    if version == CODEBASE_VERSION_V30:
        logger.info("Dataset at %s is already LeRobot %s", root, CODEBASE_VERSION_V30)
        return root
    if version != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 dataset at {root}, got codebase_version={version!r}")

    episodes = _load_jsonlines_if_exists(root / "meta/episodes.jsonl")
    tasks = _load_jsonlines_if_exists(root / "meta/tasks.jsonl")
    episodes_stats = _load_jsonlines_if_exists(root / "meta/episodes_stats.jsonl")

    tmp_root = root.parent / f".{root.name}.v30_tmp"
    backup_root = root.parent / f".{root.name}.v21_backup"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)
    tmp_root.mkdir(parents=True)

    try:
        _write_v30_data_files(root, tmp_root, info, episodes)
        _write_v30_video_files(root, tmp_root, info, episodes)
        _write_v30_metadata(tmp_root, info, tasks, episodes, episodes_stats)

        root.rename(backup_root)
        tmp_root.rename(root)
        if keep_v21_backup:
            logger.info("LeRobot v2.1 backup kept at %s", backup_root)
        else:
            shutil.rmtree(backup_root)
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        if backup_root.exists() and not root.exists():
            backup_root.rename(root)
        raise

    logger.info("Converted LeRobot dataset at %s from v2.1 to v3.0", root)
    return root


def _load_jsonlines_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonlines(path)


def _episode_chunk_file_indices(episode_index: int, chunks_size: int) -> tuple[int, int]:
    chunk_index = episode_index // chunks_size
    file_index = episode_index % chunks_size
    return chunk_index, file_index


def _format_v21_data_path(info: dict[str, Any], episode_index: int) -> Path:
    episode_chunk = episode_index // int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    return Path(info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index))


def _format_v21_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    episode_chunk = episode_index // int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    return Path(
        info["video_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def _write_v30_data_files(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk_index, file_index = _episode_chunk_file_indices(episode_index, chunks_size)
        src = src_root / _format_v21_data_path(info, episode_index)
        dst = dst_root / V30_DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_v30_video_files(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    if not info.get("video_path"):
        return

    video_keys = [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]
    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk_index, file_index = _episode_chunk_file_indices(episode_index, chunks_size)
        for video_key in video_keys:
            src = src_root / _format_v21_video_path(info, episode_index, video_key)
            if not src.exists():
                continue
            dst = dst_root / V30_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _write_v30_metadata(
    dst_root: Path,
    info: dict[str, Any],
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    episodes_stats: list[dict[str, Any]],
) -> None:
    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    v30_info = dict(info)
    v30_info["codebase_version"] = CODEBASE_VERSION_V30
    v30_info["data_path"] = V30_DATA_PATH
    v30_info["video_path"] = V30_VIDEO_PATH if info.get("video_path") else None
    v30_info["episodes_path"] = V30_EPISODES_PATH
    v30_info["episodes_stats_path"] = V30_EPISODES_STATS_PATH
    write_json(v30_info, dst_root / INFO_PATH)

    pd.DataFrame(tasks, columns=["task_index", "task"]).to_parquet(dst_root / V30_TASKS_PATH)

    episode_rows = []
    dataset_from_index = 0
    for episode in episodes:
        row = _v30_episode_row(episode, info, dataset_from_index)
        episode_rows.append(row)
        dataset_from_index = int(row["dataset_to_index"])
    _write_chunked_parquet_rows(episode_rows, dst_root, V30_EPISODES_PATH, chunks_size)

    stats_rows = []
    for item in episodes_stats:
        row = {"episode_index": int(item["episode_index"])}
        stats = dict(item.get("stats", {}))
        if "episode_index" in stats:
            stats["episode_index_stats"] = stats.pop("episode_index")
        row.update(stats)
        stats_rows.append(row)
    if stats_rows:
        _write_chunked_parquet_rows(stats_rows, dst_root, V30_EPISODES_STATS_PATH, chunks_size)


def _v30_episode_row(
    episode: dict[str, Any],
    info: dict[str, Any],
    dataset_from_index: int,
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    chunk_index, file_index = _episode_chunk_file_indices(episode_index, chunks_size)
    length = int(episode["length"])
    fps = float(info["fps"])
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "tasks": episode.get("tasks", []),
        "length": length,
        "dataset_from_index": dataset_from_index,
        "dataset_to_index": dataset_from_index + length,
        "data/chunk_index": chunk_index,
        "data/file_index": file_index,
        "data/from_index": 0,
        "data/to_index": length,
    }

    if info.get("video_path"):
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") != "video":
                continue
            prefix = f"videos/{key}"
            row[f"{prefix}/chunk_index"] = chunk_index
            row[f"{prefix}/file_index"] = file_index
            row[f"{prefix}/from_timestamp"] = 0.0
            row[f"{prefix}/to_timestamp"] = max(0.0, (length - 1) / fps)

    return row


def _write_chunked_parquet_rows(
    rows: list[dict[str, Any]],
    root: Path,
    path_template: str,
    chunks_size: int,
) -> None:
    if not rows:
        return
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        chunk_index = int(row["episode_index"]) // chunks_size
        by_chunk.setdefault(chunk_index, []).append(row)

    for chunk_index, chunk_rows in by_chunk.items():
        path = root / path_template.format(chunk_index=chunk_index, file_index=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(chunk_rows).to_parquet(path)
