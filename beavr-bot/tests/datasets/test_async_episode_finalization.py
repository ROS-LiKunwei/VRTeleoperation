import json
import threading
from pathlib import Path

import numpy as np

from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def test_async_episode_finalization_defers_video_and_stats(tmp_path, monkeypatch):
    allow_encode = threading.Event()

    def fake_encode_video_frames(_img_dir, video_path, _fps, overwrite=False):
        allow_encode.wait(timeout=2.0)
        Path(video_path).write_bytes(b"fake-mp4")

    def fake_get_video_info(_video_path):
        return {
            "video.fps": 30,
            "video.codec": "fake",
            "video.pix_fmt": "yuv420p",
            "video.height": 4,
            "video.width": 4,
            "video.channels": 3,
        }

    monkeypatch.setattr(
        "beavr.lerobot.common.datasets.lerobot_dataset.encode_video_frames",
        fake_encode_video_frames,
    )
    monkeypatch.setattr(
        "beavr.lerobot.common.datasets.lerobot_dataset.get_video_info",
        fake_get_video_info,
    )

    dataset = LeRobotDataset.create(
        repo_id="test/async_episode",
        fps=30,
        root=tmp_path / "async_episode",
        robot_type="test_robot",
        features={
            "observation.images.front": {
                "dtype": "video",
                "shape": (3, 4, 4),
                "names": ["channels", "height", "width"],
            },
            "action": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["action"],
            },
        },
        use_videos=True,
    )
    dataset.set_async_video_encoding(True)

    frame = {
        "observation.images.front": np.zeros((3, 4, 4), dtype=np.uint8),
        "action": np.array([0.0], dtype=np.float32),
        "task": "test",
    }
    dataset.add_frame(frame)
    dataset.save_episode()

    assert list(dataset.root.rglob("*.parquet"))
    assert not list(dataset.root.rglob("*.mp4"))
    assert not (dataset.root / "meta/episodes_stats.jsonl").exists()

    allow_encode.set()
    dataset.wait_for_async_video_encoding(shutdown=True)

    assert list(dataset.root.rglob("*.mp4"))
    assert (dataset.root / "meta/episodes_stats.jsonl").exists()
    with open(dataset.root / "meta/info.json") as f:
        info = json.load(f)
    assert info["features"]["observation.images.front"]["info"]["video.fps"] == 30
