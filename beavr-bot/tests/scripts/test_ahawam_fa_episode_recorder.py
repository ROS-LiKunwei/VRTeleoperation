import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import beavr.scripts.ahawam_fa_episode_recorder as recorder_module
from beavr.scripts.ahawam_fa_isolated_recorder import IsolatedFaEpisodeRecorder
from beavr.scripts.ahawam_fa_episode_recorder import (
    CAMERA_KEYS,
    CAMERA_SHAPES,
    DEPTH_SHAPES,
    FaEpisodeRecorder,
    RecorderBusyError,
    RecorderFailedError,
    RecorderValidationError,
)


class _FakeEncoder:
    def __init__(self, *, path, fps, width, height, crf, gop):
        self.path = Path(path)
        self.expected_shape = (height, width, 3)
        self.frames = 0
        self.closed = False
        assert fps > 0
        assert crf == 30
        assert gop == fps

    def write(self, frame):
        assert not self.closed
        assert frame.shape == self.expected_shape
        assert frame.dtype == np.uint8
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.path.write_bytes(f"fake-h264-frames={self.frames}".encode())
        self.closed = True


class _FakeDepthEncoder:
    def __init__(self, *, path, fps, width, height):
        self.path = Path(path)
        self.expected_shape = (height, width)
        self.frames = 0
        self.closed = False
        assert fps > 0

    def write(self, frame):
        assert not self.closed
        assert frame.shape == self.expected_shape
        assert frame.dtype == np.uint16
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.path.write_bytes(f"fake-depth-frames={self.frames}".encode())
        self.closed = True


def _images(seed=0):
    return {
        key: np.full(shape, seed + index, dtype=np.uint8)
        for index, (key, shape) in enumerate(CAMERA_SHAPES.items())
    }


def _depths(seed=0):
    return {
        key: np.full(shape, min(1000, 100 + seed + index), dtype=np.uint16)
        for index, (key, shape) in enumerate(DEPTH_SHAPES.items())
    }


def _camera_metadata(sequence, captured_at):
    return {
        key: {
            "sequence": sequence + index,
            "captured_at_monotonic_s": captured_at + index * 0.001,
            "age_ms": 1.5,
        }
        for index, key in enumerate(CAMERA_KEYS)
    }


def _sent_command(*, source="model", sequence=1, sent_at=9.9):
    return {
        "sequence": sequence,
        "sent_at_monotonic_s": sent_at,
        "sent_at_wall_s": 1_800_000_000.0 + sent_at,
        "source": source,
        "command": np.linspace(-0.5, 0.5, 16, dtype=np.float32).tolist(),
        "receipt": {
            "source": source,
            "sent_at_monotonic_s": sent_at,
            "arm_published": True,
            "left_hand_published": True,
            "right_hand_published": True,
            "left_hand_changed": False,
            "right_hand_changed": False,
            "complete": True,
        },
    }


def _submit(
    recorder,
    *,
    frame_sequence,
    captured_at,
    camera_sequence,
    sent_command=None,
    seed=0,
):
    return recorder.submit_observation(
        images=_images(seed),
        depths=_depths(seed),
        state=np.linspace(-1.0, 1.0, 16, dtype=np.float32) + seed,
        camera_metadata=_camera_metadata(camera_sequence, captured_at),
        state_metadata={
            "sequence": frame_sequence,
            "sample_at_monotonic_s": captured_at,
        },
        sync_metadata={"max_abs_camera_joint_skew_ms": 2.0},
        sequence=frame_sequence,
        captured_at=captured_at,
        sent_command=sent_command,
        intervention_snapshot={"state": "MODEL"},
    )


def _json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_default_blosc_depth_round_trip_is_exact_mm(tmp_path):
    path = tmp_path / "depth.depth.bl2.partial"
    source = np.resize(np.arange(1001, dtype=np.uint16), (48, 64))
    encoder = recorder_module._BloscZstdDepthEncoder(
        path=path,
        fps=10,
        width=64,
        height=48,
    )
    encoder.write(source)
    encoder.write(np.flipud(source).copy())
    encoder.close()
    expected = [
        recorder_module.hashlib.sha256(frame.tobytes()).hexdigest()
        for frame in (source, np.flipud(source).copy())
    ]
    recorder_module._validate_blosc_depth_stream(
        path,
        expected_hashes=expected,
        expected_fps=10,
        expected_shape=(48, 64),
    )


def test_default_encoders_commit_one_complete_rgbd_frame(tmp_path):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        config={"enable_motion": False},
        validate_timing=False,
    )
    _submit(
        recorder,
        frame_sequence=1,
        captured_at=10.0,
        camera_sequence=1,
    )
    final_path = recorder.finalize("unlabeled", reason="encoder_contract_test")
    manifest = json.loads((final_path / "manifest.json").read_text())
    segment = manifest["segments"][0]
    assert set(segment["videos"]) == set(CAMERA_KEYS)
    assert set(segment["depth_videos"]) == set(CAMERA_KEYS)
    assert all((final_path / path).stat().st_size > 0 for path in segment["videos"].values())
    assert all(
        (final_path / path).stat().st_size > 0
        for path in segment["depth_videos"].values()
    )


def test_isolated_recorder_commits_rgbd_in_spawned_process(tmp_path):
    recorder = IsolatedFaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        config={"enable_motion": False},
        validate_timing=False,
    )
    try:
        assert recorder.process_pid != os.getpid()
        _submit(
            recorder,
            frame_sequence=1,
            captured_at=10.0,
            camera_sequence=1,
        )
        final_path = recorder.finalize(
            "unlabeled", reason="isolated_encoder_contract_test"
        )
        manifest = json.loads((final_path / "manifest.json").read_text())
        assert manifest["frame_count"] == 1
        assert set(manifest["segments"][0]["videos"]) == set(CAMERA_KEYS)
        assert set(manifest["segments"][0]["depth_videos"]) == set(CAMERA_KEYS)
    finally:
        recorder.close()


def test_commits_three_camera_episode_without_mislabeling_sent_command_as_action(
    tmp_path,
):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        segment_sec=0.2,
        min_free_gb=0,
        config={"enable_motion": True, "robot": "FA"},
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )

    _submit(
        recorder,
        frame_sequence=1,
        captured_at=10.0,
        camera_sequence=50,
        sent_command=None,
    )
    # Reusing a latest 10 Hz camera frame is valid; per-camera metadata proves
    # that it was reused instead of fabricating a fresh camera sample.
    _submit(
        recorder,
        frame_sequence=2,
        captured_at=10.1,
        camera_sequence=50,
        sent_command=_sent_command(source="model", sequence=7, sent_at=10.05),
        seed=1,
    )
    _submit(
        recorder,
        frame_sequence=3,
        captured_at=10.2,
        camera_sequence=51,
        sent_command=_sent_command(source="human", sequence=8, sent_at=10.15),
        seed=2,
    )
    recorder.record_event("intervention_entered", trigger="operator")
    final_path = recorder.finalize(
        "success", intervention_summary={"intervention_count": 1}
    )

    assert final_path.is_dir()
    assert (final_path / "COMMITTED").is_file()
    assert not recorder.staging_path.exists()
    assert list((tmp_path / ".in_progress").iterdir()) == []

    rows = _json_rows(final_path / "frames.jsonl")
    assert len(rows) == 3
    assert all("action" not in row for row in rows)
    assert len(rows[0]["state"]) == 16
    assert rows[0]["sent_command"] is None
    assert len(rows[1]["sent_command"]) == 16
    assert rows[1]["sent_command_source"] == "model"
    assert rows[2]["sent_command_source"] == "human"
    assert rows[1]["sent_command_receipt"]["complete"] is True
    assert rows[0]["camera_metadata"][CAMERA_KEYS[0]]["sequence"] == 50
    assert rows[1]["camera_metadata"][CAMERA_KEYS[0]]["sequence"] == 50

    manifest = json.loads((final_path / "manifest.json").read_text())
    assert manifest["frame_count"] == 3
    assert manifest["transition_count"] == 2
    assert manifest["training_export_contract"]["action"] == "frames[t+1].state"
    assert (
        manifest["training_export_contract"]["sent_command_is_not_training_action"]
        is True
    )
    assert [segment["frame_count"] for segment in manifest["segments"]] == [2, 1]
    for segment in manifest["segments"]:
        assert set(segment["videos"]) == set(CAMERA_KEYS)
        assert set(segment["depth_videos"]) == set(CAMERA_KEYS)
        for relative_path in segment["videos"].values():
            assert (final_path / relative_path).stat().st_size > 0
        for relative_path in segment["depth_videos"].values():
            assert (final_path / relative_path).stat().st_size > 0

    metadata = json.loads((final_path / "metadata.json").read_text())
    assert metadata["state_shape"] == [16]
    assert metadata["sent_command_shape"] == [16]
    assert metadata["depth_dtype"] == "uint16"
    assert metadata["depth_unit"] == "millimetres"
    assert (
        metadata["training_export_contract"]["last_frame_has_training_transition"]
        is False
    )
    assert metadata["camera_sampling_contract"]["latest_frame_reuse_allowed"] is True
    # ROS publish return is not physical execution feedback.  The recorder
    # keeps command-publication evidence, while motion_executed remains null.
    assert metadata["motion_executed"] is None
    assert metadata["command_publication"] == {
        "complete_task_command_observed": True,
        "successful_publish_notification_count": 0,
        "partial_task_command_count": 0,
        "task_command_sources": ["human", "model"],
        "task_output_may_have_changed": True,
        "physical_execution_verified": False,
    }
    assert (
        _json_rows(tmp_path / "episodes.jsonl")[0]["episode_uuid"]
        == recorder.episode_uuid
    )


@pytest.mark.parametrize(
    ("source", "outcome", "kwargs", "expected_success"),
    [
        ("hold", "failure", {"failure_reason": "drawer did not open"}, False),
        ("model", "aborted", {"reason": "operator stopped run"}, None),
        ("human", "error", {"reason": "external controller fault"}, None),
        ("model", "unlabeled", {"reason": "label timeout"}, None),
    ],
)
def test_explicit_terminal_outcomes_are_preserved(
    tmp_path, source, outcome, kwargs, expected_success
):
    root = tmp_path / outcome
    recorder = FaEpisodeRecorder(
        root,
        fps=10,
        min_free_gb=0,
        validate_timing=False,
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    _submit(
        recorder,
        frame_sequence=1,
        captured_at=10.0,
        camera_sequence=1,
        sent_command=_sent_command(source=source, sent_at=9.9),
    )
    final_path = recorder.finalize(outcome, **kwargs)
    document = json.loads((final_path / "outcome.json").read_text())
    assert document["outcome"] == outcome
    assert document["success"] is expected_success
    assert document["failure_reason"] == kwargs.get("failure_reason")
    assert document["outcome_reason"] == kwargs.get("reason")


def test_partial_or_future_sent_command_fails_closed(tmp_path):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    command = _sent_command(sent_at=10.1)
    command["receipt"]["right_hand_published"] = False
    command["receipt"]["complete"] = False

    with pytest.raises(RecorderValidationError, match="complete"):
        _submit(
            recorder,
            frame_sequence=1,
            captured_at=10.0,
            camera_sequence=1,
            sent_command=command,
        )
    with pytest.raises(RecorderFailedError):
        recorder.raise_if_failed()
    staging = recorder.abort("invalid command", outcome="error")
    assert staging.is_dir()
    assert not (staging / "COMMITTED").exists()
    assert json.loads((staging / "outcome.json").read_text())["outcome"] == "error"

    future_root = tmp_path / "future"
    future = FaEpisodeRecorder(
        future_root,
        fps=10,
        min_free_gb=0,
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    with pytest.raises(RecorderValidationError, match="newer than the observation"):
        _submit(
            future,
            frame_sequence=1,
            captured_at=10.0,
            camera_sequence=1,
            sent_command=_sent_command(sent_at=10.01),
        )
    future.abort("causality violation", outcome="error")


def test_partial_publish_is_unverified_motion_evidence(tmp_path):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        validate_timing=False,
        config={"enable_motion": True},
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    recorder.note_partial_command(
        {
            "source": "human",
            "arm_published": True,
            "left_hand_published": True,
            "right_hand_published": False,
            "left_hand_changed": True,
            "right_hand_changed": False,
            "complete": False,
        }
    )
    _submit(
        recorder,
        frame_sequence=1,
        captured_at=10.0,
        camera_sequence=1,
    )
    final_path = recorder.finalize("error", reason="partial ROS publish")
    outcome = json.loads((final_path / "outcome.json").read_text())

    assert outcome["motion_executed"] is None
    assert outcome["command_publication"] == {
        "complete_task_command_observed": False,
        "successful_publish_notification_count": 0,
        "partial_task_command_count": 1,
        "task_command_sources": ["human"],
        "task_output_may_have_changed": True,
        "physical_execution_verified": False,
    }


def test_failed_abort_then_close_preserves_first_failure_evidence(tmp_path):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    images = _images()
    images["head"] = images["head"][:-1]
    with pytest.raises(RecorderValidationError):
        recorder.submit_observation(
            images=images,
            state=np.zeros(16, dtype=np.float32),
            camera_metadata=_camera_metadata(1, 10.0),
            state_metadata=None,
            sync_metadata=None,
            sequence=1,
            captured_at=10.0,
            sent_command=None,
        )
    staging = recorder.abort(
        "client preserved reason",
        outcome="error",
        intervention_summary={
            "reported_outcome": "success",
            "reported_reason": "operator_labeled_success",
        },
    )
    before = (staging / "outcome.json").read_bytes()

    assert recorder.close() == staging
    assert (staging / "outcome.json").read_bytes() == before
    outcome = json.loads(before)
    assert "images['head'] expected" in outcome["outcome_reason"]
    summary = outcome["intervention_summary"]
    assert summary["reported_outcome"] == "success"
    assert summary["effective_outcome"] == "error"
    assert "images['head'] expected" in summary["effective_reason"]


def test_invalid_frame_shape_poison_episode_and_cannot_commit(tmp_path):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    images = _images()
    images["head"] = images["head"][:-1]
    with pytest.raises(RecorderValidationError, match="head"):
        recorder.submit_observation(
            images=images,
            state=np.zeros(16, dtype=np.float32),
            camera_metadata=_camera_metadata(1, 10.0),
            state_metadata=None,
            sync_metadata=None,
            sequence=1,
            captured_at=10.0,
            sent_command=None,
        )
    with pytest.raises(RecorderFailedError):
        recorder.raise_if_failed()
    staging = recorder.abort("bad camera frame", outcome="error")
    assert staging.parent.name == ".in_progress"
    assert not (staging / "COMMITTED").exists()


class _BlockingEncoder(_FakeEncoder):
    entered = threading.Event()
    release = threading.Event()

    def write(self, frame):
        self.entered.set()
        assert self.release.wait(timeout=5)
        super().write(frame)


def test_bounded_queue_overflow_is_fail_closed_and_drops_nothing_silently(tmp_path):
    _BlockingEncoder.entered.clear()
    _BlockingEncoder.release.clear()
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        queue_size=1,
        min_free_gb=0,
        encoder_factory=_BlockingEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    _submit(
        recorder,
        frame_sequence=1,
        captured_at=10.0,
        camera_sequence=1,
    )
    assert _BlockingEncoder.entered.wait(timeout=5)
    _submit(
        recorder,
        frame_sequence=2,
        captured_at=10.1,
        camera_sequence=1,
    )
    with pytest.raises(RecorderFailedError, match="queue full"):
        _submit(
            recorder,
            frame_sequence=3,
            captured_at=10.2,
            camera_sequence=2,
        )
    _BlockingEncoder.release.set()
    staging = recorder.abort("writer backpressure", outcome="error")
    assert staging.is_dir()
    assert not (staging / "COMMITTED").exists()
    outcome = json.loads((staging / "outcome.json").read_text())
    assert outcome["submitted_frame_count"] == 2


class _FailingEncoder(_FakeEncoder):
    def write(self, frame):
        raise OSError("injected encoder failure")


def test_background_encoder_failure_is_observable_and_cannot_commit(tmp_path):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        encoder_factory=_FailingEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    _submit(
        recorder,
        frame_sequence=1,
        captured_at=10.0,
        camera_sequence=1,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            recorder.raise_if_failed()
        except RecorderFailedError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("background encoder failure was not exposed")
    staging = recorder.abort("encoder failure", outcome="error")
    assert staging.is_dir()
    assert not (staging / "COMMITTED").exists()


def test_background_disk_failure_is_observable_and_cannot_commit(tmp_path, monkeypatch):
    real_check = recorder_module._ensure_free_space
    calls = 0

    def fail_after_initialization(path, min_free_bytes):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("injected disk failure")
        return real_check(path, min_free_bytes)

    monkeypatch.setattr(
        recorder_module, "_ensure_free_space", fail_after_initialization
    )
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    _submit(
        recorder,
        frame_sequence=1,
        captured_at=10.0,
        camera_sequence=1,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            recorder.raise_if_failed()
        except RecorderFailedError as exc:
            assert "injected disk failure" in str(exc)
            break
        time.sleep(0.01)
    else:
        pytest.fail("background disk failure was not exposed")
    staging = recorder.abort("disk failure", outcome="error")
    assert staging.is_dir()
    assert not (staging / "COMMITTED").exists()


def test_single_writer_lock_prevents_overlapping_episode_transactions(tmp_path):
    recorder = FaEpisodeRecorder(
        tmp_path,
        fps=10,
        min_free_gb=0,
        encoder_factory=_FakeEncoder,
        depth_encoder_factory=_FakeDepthEncoder,
    )
    with pytest.raises(RecorderBusyError):
        FaEpisodeRecorder(
            tmp_path,
            fps=10,
            min_free_gb=0,
            encoder_factory=_FakeEncoder,
            depth_encoder_factory=_FakeDepthEncoder,
        )
    recorder.abort("test complete")
