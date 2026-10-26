import json
from pathlib import Path

import pytest

from beavr.scripts import summarize_fa_real_rollouts as summary


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _add_episode(
    root: Path,
    episode_uuid: str,
    *,
    outcome_label: str = "success",
    assisted: bool = False,
) -> dict[str, object]:
    episode = root / "episodes" / episode_uuid
    for camera in summary.CAMERA_KEYS:
        camera_dir = episode / "videos" / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        (camera_dir / "segment-000000.mp4").write_bytes(b"not-decoded-in-summary")

    camera_metadata = {
        camera: {
            "sequence": index + 1,
            "captured_at_monotonic_s": 9.9 + index * 0.01,
        }
        for index, camera in enumerate(summary.CAMERA_KEYS)
    }
    video_paths = {
        camera: f"videos/{camera}/segment-000000.mp4" for camera in summary.CAMERA_KEYS
    }
    command = [float(index) / 10.0 for index in range(summary.STATE_DIM)]
    frames = [
        {
            "schema_version": 1,
            "frame_index": 0,
            "sequence": 100,
            "captured_at": 10.0,
            "state": [0.0] * summary.STATE_DIM,
            "sent_command": None,
            "sent_command_source": "none",
            "sent_command_sequence": None,
            "sent_command_chunk": None,
            "sent_command_step": None,
            "sent_command_at_monotonic_s": None,
            "sent_command_at_wall_s": None,
            "sent_command_receipt": None,
            "segment_index": 0,
            "segment_frame_index": 0,
            "videos": video_paths,
            "camera_metadata": camera_metadata,
            "state_metadata": {},
            "sync_metadata": {},
            "intervention": {},
        },
        {
            "schema_version": 1,
            "frame_index": 1,
            "sequence": 101,
            "captured_at": 10.5,
            "state": [0.1] * summary.STATE_DIM,
            "sent_command": command,
            "sent_command_source": "human" if assisted else "model",
            "sent_command_sequence": 1,
            "sent_command_chunk": 0,
            "sent_command_step": 0,
            "sent_command_at_monotonic_s": 10.25,
            "sent_command_at_wall_s": 1000.25,
            "sent_command_receipt": {
                "complete": True,
                "transport": "ros_publish_returned",
            },
            "segment_index": 0,
            "segment_frame_index": 1,
            "videos": video_paths,
            "camera_metadata": camera_metadata,
            "state_metadata": {},
            "sync_metadata": {},
            "intervention": {},
        },
    ]
    events = [
        {
            "schema_version": 1,
            "event_index": 0,
            "type": "takeover" if assisted else "model_command",
            "recorded_at": 1000.3,
            "payload": {},
        }
    ]
    timing = {
        "valid": True,
        "assessed": True,
        "target_fps": 2,
        "frame_count": 2,
        "duration_sec": 0.5,
        "effective_fps": 2.0,
        "relative_fps_error": 0.0,
        "max_gap_sec": 0.5,
        "max_allowed_gap_sec": 1.5,
        "violations": [],
    }
    segments = [
        {
            "segment_index": 0,
            "start_frame": 0,
            "frame_count": 2,
            "videos": video_paths,
        }
    ]
    started_at = "2026-10-19T00:00:00Z"
    ended_at = "2026-10-19T00:00:01Z"
    command_publication = {
        "complete_task_command_observed": True,
        "successful_publish_notification_count": 1,
        "partial_task_command_count": 0,
        "task_command_sources": ["human" if assisted else "model"],
        "task_output_may_have_changed": True,
        "physical_execution_verified": False,
    }
    metadata = {
        "schema_version": 1,
        "episode_uuid": episode_uuid,
        "status": "complete",
        "camera_keys": list(summary.CAMERA_KEYS),
        "state_shape": [summary.STATE_DIM],
        "sent_command_shape": [summary.STATE_DIM],
        "frame_count": 2,
        "transition_count": 1,
        "event_count": 1,
        "motion_executed": None,
        "command_publication": command_publication,
        "capture_timing": timing,
    }
    manifest = {
        "schema_version": 1,
        "episode_uuid": episode_uuid,
        "frame_count": 2,
        "transition_count": 1,
        "event_count": 1,
        "segments": segments,
        "capture_timing": timing,
        "camera_keys": list(summary.CAMERA_KEYS),
    }
    failure_reason = "drawer did not open" if outcome_label == "failure" else None
    outcome_reason = (
        f"episode {outcome_label}"
        if outcome_label in {"aborted", "error", "unlabeled"}
        else None
    )
    outcome = {
        "schema_version": 1,
        "episode_uuid": episode_uuid,
        "outcome": outcome_label,
        "success": (
            True
            if outcome_label == "success"
            else False if outcome_label == "failure" else None
        ),
        "outcome_reason": outcome_reason,
        "failure_reason": failure_reason,
        "intervention_summary": {
            "assisted": assisted,
            "effective_outcome": outcome_label,
        },
        "frame_count": 2,
        "transition_count": 1,
        "event_count": 1,
        "motion_executed": None,
        "command_publication": command_publication,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
    }
    _write_json(episode / "metadata.json", metadata)
    _write_json(episode / "manifest.json", manifest)
    _write_json(episode / "outcome.json", outcome)
    _write_json(
        episode / "COMMITTED",
        {"schema_version": 1, "episode_uuid": episode_uuid},
    )
    _write_jsonl(episode / "frames.jsonl", frames)
    _write_jsonl(episode / "events.jsonl", events)
    return {
        "schema_version": 1,
        "episode_uuid": episode_uuid,
        "path": f"episodes/{episode_uuid}",
        "outcome": outcome_label,
        "failure_reason": failure_reason,
        "frame_count": 2,
        "transition_count": 1,
        "motion_executed": None,
        "command_publication": command_publication,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
    }


def test_summarize_healthy_fa_rollouts_and_does_not_modify_data(tmp_path):
    rows = [
        _add_episode(tmp_path, "autonomous-success", outcome_label="success"),
        _add_episode(
            tmp_path,
            "assisted-failure",
            outcome_label="failure",
            assisted=True,
        ),
    ]
    _write_jsonl(tmp_path / "episodes.jsonl", rows)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    report = summary.summarize(tmp_path)

    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert report["healthy"] is True
    assert report["committed_count"] == 2
    assert report["incomplete_count"] == 0
    assert report["outcome_counts"] == {
        "success": 1,
        "failure": 1,
        "aborted": 0,
        "error": 0,
        "unlabeled": 0,
    }
    assert report["assisted_count"] == 1
    assert report["autonomous_count"] == 1
    assert report["task_output_publish_evidence_count"] == 2
    assert report["physical_execution_verified_count"] == 0
    assert report["frame_count"] == 4
    assert report["transition_count"] == 2
    assert report["duration_sec"] == pytest.approx(1.0)


def test_summarize_accepts_complete_rgbd_manifest(tmp_path):
    episode_uuid = "rgbd-success"
    index_row = _add_episode(tmp_path, episode_uuid)
    episode = tmp_path / "episodes" / episode_uuid
    depth_paths = {}
    for camera in summary.CAMERA_KEYS:
        path = episode / "videos" / f"{camera}_depth" / "segment-000000.mkv"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"ffv1-not-decoded-in-summary")
        depth_paths[camera] = str(path.relative_to(episode))

    metadata = json.loads((episode / "metadata.json").read_text())
    metadata.update(
        depth_camera_keys=list(summary.CAMERA_KEYS),
        depth_dtype="uint16",
        depth_unit="millimetres",
    )
    _write_json(episode / "metadata.json", metadata)
    manifest = json.loads((episode / "manifest.json").read_text())
    manifest["depth_camera_keys"] = list(summary.CAMERA_KEYS)
    manifest["segments"][0]["depth_videos"] = depth_paths
    _write_json(episode / "manifest.json", manifest)
    frames = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]
    for row in frames:
        row["depth_videos"] = depth_paths
        row["depth_sha256"] = {camera: "0" * 64 for camera in summary.CAMERA_KEYS}
    _write_jsonl(episode / "frames.jsonl", frames)
    _write_jsonl(tmp_path / "episodes.jsonl", [index_row])

    report = summary.summarize(tmp_path)

    assert report["healthy"] is True
    assert report["committed_count"] == 1


def test_strict_rejects_duplicate_index_and_staging_path(tmp_path):
    row = _add_episode(tmp_path, "duplicate")
    staged_row = dict(row, path=".in_progress/duplicate")
    _write_jsonl(tmp_path / "episodes.jsonl", [row, staged_row])

    report = summary.summarize(tmp_path)

    assert report["healthy"] is False
    assert any("duplicate episode UUIDs" in issue for issue in report["issues"])
    assert any(
        "must never point into .in_progress" in issue for issue in report["issues"]
    )


def test_strict_rejects_unsafe_or_missing_three_camera_manifest(tmp_path):
    row = _add_episode(tmp_path, "unsafe-video")
    episode = tmp_path / "episodes" / "unsafe-video"
    manifest = summary._load_json(episode / "manifest.json")
    manifest["segments"][0]["videos"][summary.CAMERA_KEYS[0]] = "../outside.mp4"
    _write_json(episode / "manifest.json", manifest)
    _write_jsonl(tmp_path / "episodes.jsonl", [row])

    report = summary.summarize(tmp_path)

    assert report["healthy"] is False
    assert any("unsafe head video path" in issue for issue in report["issues"])
    assert any(
        "video files differ from manifest" in issue for issue in report["issues"]
    )


def test_strict_rejects_bad_counts_uuid_and_frame_command_contract(tmp_path):
    row = _add_episode(tmp_path, "bad-contract", assisted=True)
    episode = tmp_path / "episodes" / "bad-contract"
    metadata = summary._load_json(episode / "metadata.json")
    metadata["episode_uuid"] = "someone-else"
    metadata["frame_count"] = 99
    _write_json(episode / "metadata.json", metadata)
    frames = summary._load_jsonl(episode / "frames.jsonl")
    frames[0]["state"] = [0.0] * 15
    frames[1]["sent_command_source"] = "none"
    frames[1]["sequence"] = frames[0]["sequence"]
    _write_jsonl(episode / "frames.jsonl", frames)
    _write_jsonl(tmp_path / "episodes.jsonl", [row])

    report = summary.summarize(tmp_path)

    assert report["healthy"] is False
    assert any("metadata UUID mismatch" in issue for issue in report["issues"])
    assert any("metadata frame_count=99" in issue for issue in report["issues"])
    assert any(
        "state must be a finite 16-vector" in issue for issue in report["issues"]
    )
    assert any("invalid sent_command_source" in issue for issue in report["issues"])
    assert any(
        "sequence is not strictly increasing" in issue for issue in report["issues"]
    )


def test_in_progress_episode_is_reported_but_not_indexed_or_marked_corrupt(tmp_path):
    staging = tmp_path / ".in_progress" / "recording-now"
    staging.mkdir(parents=True)
    _write_json(staging / "metadata.json", {"status": "recording"})

    report = summary.summarize(tmp_path)

    assert report["healthy"] is True
    assert report["committed_count"] == 0
    assert report["incomplete_count"] == 1
    assert report["incomplete"][0]["location"] == ".in_progress"


def test_cli_json_and_strict_exit_status(tmp_path, capsys):
    row = _add_episode(tmp_path, "cli-episode")
    _write_jsonl(tmp_path / "episodes.jsonl", [row])

    assert summary.main([str(tmp_path), "--json", "--strict"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["healthy"] is True

    (tmp_path / "episodes" / "cli-episode" / "COMMITTED").unlink()
    assert summary.main([str(tmp_path), "--strict"]) == 2
    output = capsys.readouterr().out
    assert "integrity=FAILED" in output
    assert "lacks COMMITTED" in output
