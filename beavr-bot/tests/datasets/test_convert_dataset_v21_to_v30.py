from pathlib import Path

import pandas as pd

from beavr.lerobot.common.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_dataset_v21_to_v30,
)


def test_convert_dataset_v21_to_v30_rewrites_layout(tmp_path: Path):
    root = tmp_path / "whx" / "sysmo32_dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)

    pd.DataFrame(
        {
            "observation.state": [[0.0, 1.0], [2.0, 3.0]],
            "action": [[0.1], [0.2]],
            "timestamp": [0.0, 1.0 / 30.0],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "index": [0, 1],
            "task_index": [0, 0],
        }
    ).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    pd.DataFrame(
        {
            "observation.state": [[4.0, 5.0]],
            "action": [[0.3]],
            "timestamp": [0.0],
            "frame_index": [0],
            "episode_index": [1],
            "index": [2],
            "task_index": [0],
        }
    ).to_parquet(root / "data/chunk-000/episode_000001.parquet")

    (root / "meta/info.json").write_text(
        """
{
  "codebase_version": "v2.1",
  "robot_type": "sysmo32",
  "total_episodes": 2,
  "total_frames": 3,
  "total_tasks": 1,
  "total_videos": 0,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 30,
  "splits": {"train": "0:2"},
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": null,
  "features": {
    "observation.state": {"dtype": "float32", "shape": [2], "names": ["j0", "j1"]},
    "action": {"dtype": "float32", "shape": [1], "names": ["a0"]},
    "timestamp": {"dtype": "float32", "shape": [1], "names": null},
    "frame_index": {"dtype": "int64", "shape": [1], "names": null},
    "episode_index": {"dtype": "int64", "shape": [1], "names": null},
    "index": {"dtype": "int64", "shape": [1], "names": null},
    "task_index": {"dtype": "int64", "shape": [1], "names": null}
  }
}
""".strip()
    )
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "task"}\n')
    (root / "meta/episodes.jsonl").write_text(
        '{"episode_index": 0, "tasks": ["task"], "length": 2}\n'
        '{"episode_index": 1, "tasks": ["task"], "length": 1}\n'
    )
    (root / "meta/episodes_stats.jsonl").write_text(
        '{"episode_index": 0, "stats": {"episode_index": {"min": [0]}, "observation.state": {"mean": [1.0, 2.0]}}}\n'
        '{"episode_index": 1, "stats": {"episode_index": {"min": [1]}, "observation.state": {"mean": [4.0, 5.0]}}}\n'
    )

    convert_dataset_v21_to_v30(root)

    assert (root / "data/chunk-000/file-000.parquet").exists()
    assert (root / "data/chunk-000/file-001.parquet").exists()
    assert not (root / "data/chunk-000/episode_000000.parquet").exists()

    info = pd.read_json(root / "meta/info.json", typ="series")
    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    assert tasks.to_dict("records") == [{"task_index": 0, "task": "task"}]

    episodes = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    assert episodes["dataset_from_index"].tolist() == [0, 2]
    assert episodes["dataset_to_index"].tolist() == [2, 3]
    assert episodes["data/file_index"].tolist() == [0, 1]

    episodes_stats = pd.read_parquet(root / "meta/episodes_stats/chunk-000/file-000.parquet")
    assert episodes_stats["episode_index"].tolist() == [0, 1]
    assert "episode_index_stats" in episodes_stats.columns
