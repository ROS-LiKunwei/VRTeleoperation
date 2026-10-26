"""Read-only summary and integrity checks for FA AHA-WAM real rollouts.

The command never repairs, removes, or rewrites recorder output.  It is safe to
run while inspecting a dataset, although an actively recording episode below
``.in_progress`` is reported only as incomplete and is not treated as corrupt.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
STATE_DIM = 16
CAMERA_KEYS = ("head", "wrist_CV2756100033", "wrist_CV2856D0001C")
VALID_OUTCOMES = ("success", "failure", "aborted", "error", "unlabeled")
VALID_COMMAND_SOURCES = frozenset({"model", "human", "hold"})


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_vector(value: Any, length: int = STATE_DIM) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_finite_number(item) for item in value)
    )


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_nonnegative_int(value: Any) -> bool:
    return value is None or _nonnegative_int(value)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    if not _finite_number(left) or not _finite_number(right):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _episode_bytes(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
    except OSError:
        return total
    return total


def _read_artifact(
    episode: Path,
    name: str,
    issues: list[str],
    *,
    jsonl: bool = False,
) -> Any:
    path = episode / name
    if not path.is_file() or path.is_symlink():
        issues.append(f"{episode.name}: missing or unsafe {name}")
        return [] if jsonl else {}
    try:
        return _load_jsonl(path) if jsonl else _load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        issues.append(f"{episode.name}: cannot parse {name}: {exc}")
        return [] if jsonl else {}


def _safe_video_path(episode: Path, value: Any, camera: str) -> Path | None:
    """Accept only recorder-owned MP4 paths that cannot leave the episode."""

    if not isinstance(value, str):
        return None
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 3
        or relative.parts[:2] != ("videos", camera)
        or relative.suffix != ".mp4"
    ):
        return None
    current = episode
    if current.is_symlink():
        return None
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            return None
    try:
        current.resolve(strict=False).relative_to(episode.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return None
    return current


def _safe_depth_video_path(
    episode: Path, value: Any, camera: str, expected_filename: str
) -> Path | None:
    """Accept only the recorder-owned depth segment path for one camera."""

    if not isinstance(value, str):
        return None
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 3
        or relative.parts[:2] != ("videos", f"{camera}_depth")
        or relative.name != expected_filename
    ):
        return None
    current = episode
    if current.is_symlink():
        return None
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            return None
    try:
        current.resolve(strict=False).relative_to(episode.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return None
    return current


def _validate_identity(
    episode_name: str,
    artifact_name: str,
    value: dict[str, Any],
    issues: list[str],
) -> None:
    if not value:
        return
    if value.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"{episode_name}: unsupported {artifact_name} schema_version")
    if value.get("episode_uuid") != episode_name:
        issues.append(f"{episode_name}: {artifact_name} UUID mismatch")


def _validate_declared_counts(
    episode_name: str,
    artifact_name: str,
    value: dict[str, Any],
    *,
    frame_count: int,
    transition_count: int,
    event_count: int,
    issues: list[str],
) -> None:
    expected = {
        "frame_count": frame_count,
        "transition_count": transition_count,
        "event_count": event_count,
    }
    for key, actual in expected.items():
        if value.get(key) != actual:
            issues.append(
                f"{episode_name}: {artifact_name} {key}={value.get(key)!r}, "
                f"expected={actual}"
            )


def _validate_command_publication(
    episode_name: str,
    artifact_name: str,
    value: Any,
    issues: list[str],
) -> None:
    prefix = f"{episode_name}: {artifact_name} command_publication"
    if not isinstance(value, dict):
        issues.append(f"{prefix} must be an object")
        return
    complete = value.get("complete_task_command_observed")
    notification_count = value.get("successful_publish_notification_count")
    partial_count = value.get("partial_task_command_count")
    sources = value.get("task_command_sources")
    may_have_changed = value.get("task_output_may_have_changed")
    execution_verified = value.get("physical_execution_verified")
    if not isinstance(complete, bool):
        issues.append(f"{prefix}.complete_task_command_observed must be bool")
    if not _nonnegative_int(notification_count):
        issues.append(
            f"{prefix}.successful_publish_notification_count must be nonnegative int"
        )
    if not _nonnegative_int(partial_count):
        issues.append(f"{prefix}.partial_task_command_count must be nonnegative int")
    if (
        not isinstance(sources, list)
        or not all(isinstance(source, str) for source in sources)
        or len(sources) != len(set(sources))
        or any(source not in {"model", "human"} for source in sources)
    ):
        issues.append(f"{prefix}.task_command_sources is invalid")
    if not isinstance(may_have_changed, bool):
        issues.append(f"{prefix}.task_output_may_have_changed must be bool")
    if execution_verified is not False:
        issues.append(f"{prefix}.physical_execution_verified must be false")
    if (
        _nonnegative_int(notification_count)
        and notification_count > 0
        and complete is not True
    ):
        issues.append(f"{prefix} notification count conflicts with complete=false")


def _validate_outcome(
    episode_name: str, outcome: dict[str, Any], issues: list[str]
) -> None:
    if not outcome:
        return
    label = outcome.get("outcome")
    reason = outcome.get("outcome_reason")
    failure_reason = outcome.get("failure_reason")
    if label not in VALID_OUTCOMES:
        issues.append(f"{episode_name}: invalid outcome={label!r}")
        return
    if label == "success":
        if outcome.get("success") is not True:
            issues.append(f"{episode_name}: success outcome requires success=true")
        if reason is not None or failure_reason is not None:
            issues.append(f"{episode_name}: success outcome must not contain reasons")
    elif label == "failure":
        if outcome.get("success") is not False:
            issues.append(f"{episode_name}: failure outcome requires success=false")
        if reason is not None or not _nonempty_string(failure_reason):
            issues.append(
                f"{episode_name}: failure outcome requires only failure_reason"
            )
    else:
        if outcome.get("success") is not None:
            issues.append(f"{episode_name}: {label} outcome requires success=null")
        if not _nonempty_string(reason) or failure_reason is not None:
            issues.append(
                f"{episode_name}: {label} outcome requires only outcome_reason"
            )
    intervention = outcome.get("intervention_summary")
    if not isinstance(intervention, dict):
        issues.append(f"{episode_name}: intervention_summary must be an object")
    elif "assisted" in intervention and not isinstance(
        intervention.get("assisted"), bool
    ):
        issues.append(f"{episode_name}: intervention_summary.assisted must be bool")
    if isinstance(intervention, dict) and "effective_outcome" in intervention:
        if intervention.get("effective_outcome") != label:
            issues.append(
                f"{episode_name}: intervention effective_outcome differs from outcome"
            )


def _validate_frames(
    episode_name: str,
    frames: list[dict[str, Any]],
    expected_mapping: list[
        tuple[int, int, dict[str, str], dict[str, str] | None]
    ],
) -> tuple[list[str], Counter[str], float]:
    issues: list[str] = []
    sources: Counter[str] = Counter()
    previous_sequence: int | None = None
    previous_capture: float | None = None
    previous_command_sequence: int | None = None
    command_signatures: dict[int, str] = {}
    capture_times: list[float] = []

    actual_indices = [row.get("frame_index") for row in frames]
    if actual_indices != list(range(len(frames))):
        issues.append(f"{episode_name}: frame_index is not contiguous from zero")

    for index, row in enumerate(frames):
        prefix = f"{episode_name}: frame {index}"
        if row.get("schema_version") != SCHEMA_VERSION:
            issues.append(f"{prefix} schema_version mismatch")
        if "action" in row:
            issues.append(f"{prefix} must not contain raw training action")

        sequence = row.get("sequence")
        if not _nonnegative_int(sequence):
            issues.append(f"{prefix} has invalid sequence={sequence!r}")
        elif previous_sequence is not None and sequence <= previous_sequence:
            issues.append(f"{prefix} sequence is not strictly increasing")
        else:
            previous_sequence = sequence

        captured_at = row.get("captured_at")
        if not _finite_number(captured_at) or float(captured_at) < 0:
            issues.append(f"{prefix} has invalid captured_at")
        else:
            captured = float(captured_at)
            capture_times.append(captured)
            if previous_capture is not None and captured <= previous_capture:
                issues.append(f"{prefix} captured_at is not strictly increasing")
            previous_capture = captured

        if not _finite_vector(row.get("state")):
            issues.append(f"{prefix} state must be a finite 16-vector")

        command = row.get("sent_command")
        source = row.get("sent_command_source")
        sources[source if isinstance(source, str) and source else "invalid"] += 1
        command_fields = (
            "sent_command_sequence",
            "sent_command_chunk",
            "sent_command_step",
            "sent_command_at_monotonic_s",
            "sent_command_at_wall_s",
            "sent_command_receipt",
        )
        if command is None:
            if source != "none":
                issues.append(f"{prefix} null sent_command requires source='none'")
            for key in command_fields:
                if row.get(key) is not None:
                    issues.append(f"{prefix} null sent_command requires {key}=null")
        else:
            if not _finite_vector(command):
                issues.append(f"{prefix} sent_command must be a finite 16-vector")
            if source not in VALID_COMMAND_SOURCES:
                issues.append(f"{prefix} has invalid sent_command_source={source!r}")
            command_sequence = row.get("sent_command_sequence")
            if not _nonnegative_int(command_sequence):
                issues.append(f"{prefix} has invalid sent_command_sequence")
            else:
                if (
                    previous_command_sequence is not None
                    and command_sequence < previous_command_sequence
                ):
                    issues.append(f"{prefix} sent_command_sequence moved backwards")
                previous_command_sequence = command_sequence
            for key in ("sent_command_chunk", "sent_command_step"):
                if not _optional_nonnegative_int(row.get(key)):
                    issues.append(f"{prefix} has invalid {key}")
            for key in (
                "sent_command_at_monotonic_s",
                "sent_command_at_wall_s",
            ):
                if not _finite_number(row.get(key)) or float(row.get(key, -1)) < 0:
                    issues.append(f"{prefix} has invalid {key}")
            if (
                _finite_number(captured_at)
                and _finite_number(row.get("sent_command_at_monotonic_s"))
                and float(row["sent_command_at_monotonic_s"]) > float(captured_at)
            ):
                issues.append(f"{prefix} sent command is newer than observation")
            receipt = row.get("sent_command_receipt")
            if not isinstance(receipt, dict) or receipt.get("complete") is not True:
                issues.append(f"{prefix} sent_command_receipt must be complete")

            if _nonnegative_int(command_sequence):
                signature_value = {
                    "command": command,
                    "source": source,
                    "chunk": row.get("sent_command_chunk"),
                    "step": row.get("sent_command_step"),
                    "monotonic": row.get("sent_command_at_monotonic_s"),
                    "wall": row.get("sent_command_at_wall_s"),
                    "receipt": receipt,
                }
                try:
                    signature = json.dumps(
                        signature_value,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError):
                    issues.append(f"{prefix} sent command contains invalid JSON data")
                else:
                    old_signature = command_signatures.setdefault(
                        command_sequence, signature
                    )
                    if old_signature != signature:
                        issues.append(
                            f"{prefix} reuses sent_command_sequence with different data"
                        )

        if index < len(expected_mapping):
            (
                segment_index,
                segment_frame_index,
                videos,
                depth_videos,
            ) = expected_mapping[index]
            expected = {
                "segment_index": segment_index,
                "segment_frame_index": segment_frame_index,
                "videos": videos,
            }
            if depth_videos is not None:
                expected["depth_videos"] = depth_videos
                hashes = row.get("depth_sha256")
                if not isinstance(hashes, dict) or set(hashes) != set(CAMERA_KEYS):
                    issues.append(
                        f"{prefix} depth_sha256 must contain all FA cameras"
                    )
            for key, value in expected.items():
                if row.get(key) != value:
                    issues.append(
                        f"{prefix} {key}={row.get(key)!r}, expected={value!r}"
                    )
        camera_metadata = row.get("camera_metadata")
        if not isinstance(camera_metadata, dict) or set(camera_metadata) != set(
            CAMERA_KEYS
        ):
            issues.append(f"{prefix} camera_metadata must contain all FA cameras")

    duration = capture_times[-1] - capture_times[0] if len(capture_times) >= 2 else 0.0
    return issues, sources, duration


def _validate_events(episode_name: str, events: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if [row.get("event_index") for row in events] != list(range(len(events))):
        issues.append(f"{episode_name}: event_index is not contiguous from zero")
    for index, row in enumerate(events):
        prefix = f"{episode_name}: event {index}"
        if row.get("schema_version") != SCHEMA_VERSION:
            issues.append(f"{prefix} schema_version mismatch")
        if not _nonempty_string(row.get("type")):
            issues.append(f"{prefix} has invalid type")
        if not _finite_number(row.get("recorded_at")):
            issues.append(f"{prefix} has invalid recorded_at")
        if not isinstance(row.get("payload"), dict):
            issues.append(f"{prefix} payload must be an object")
    return issues


def _validate_manifest(
    episode: Path,
    manifest: dict[str, Any],
    frame_count: int,
    issues: list[str],
    *,
    require_depth: bool,
    depth_segment_suffix: str,
) -> list[tuple[int, int, dict[str, str], dict[str, str] | None]]:
    expected_mapping: list[
        tuple[int, int, dict[str, str], dict[str, str] | None]
    ] = []
    segments = manifest.get("segments")
    expected_paths: set[str] = set()
    if not isinstance(segments, list) or not segments:
        issues.append(f"{episode.name}: manifest requires at least one video segment")
        return expected_mapping

    next_start = 0
    for expected_index, segment in enumerate(segments):
        prefix = f"{episode.name}: manifest segment {expected_index}"
        if not isinstance(segment, dict):
            issues.append(f"{prefix} must be an object")
            continue
        segment_index = segment.get("segment_index")
        start_frame = segment.get("start_frame")
        count = segment.get("frame_count")
        videos = segment.get("videos")
        depth_videos = segment.get("depth_videos")
        if segment_index != expected_index:
            issues.append(f"{prefix} segment_index is not contiguous")
        if start_frame != next_start:
            issues.append(
                f"{prefix} start_frame={start_frame!r}, expected={next_start}"
            )
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            issues.append(f"{prefix} has invalid frame_count={count!r}")
            continue
        if not isinstance(videos, dict) or set(videos) != set(CAMERA_KEYS):
            issues.append(f"{prefix} videos must contain exactly three FA cameras")
            next_start += count
            continue

        clean_videos: dict[str, str] = {}
        clean_depth_videos: dict[str, str] = {}
        all_paths_safe = True
        for camera in CAMERA_KEYS:
            expected_value = f"videos/{camera}/segment-{expected_index:06d}.mp4"
            value = videos.get(camera)
            if value != expected_value:
                issues.append(
                    f"{prefix} {camera} path={value!r}, expected={expected_value!r}"
                )
            path = _safe_video_path(episode, value, camera)
            if path is None:
                issues.append(f"{prefix} has unsafe {camera} video path")
                all_paths_safe = False
                continue
            relative = str(path.relative_to(episode))
            if relative in expected_paths:
                issues.append(f"{prefix} reuses video path {relative}")
            expected_paths.add(relative)
            clean_videos[camera] = relative
            if not path.is_file() or path.is_symlink():
                issues.append(f"{prefix} missing or unsafe video {relative}")
            elif path.stat().st_size <= 0:
                issues.append(f"{prefix} empty video {relative}")
            if require_depth:
                depth_filename = (
                    f"segment-{expected_index:06d}{depth_segment_suffix}"
                )
                expected_depth = f"videos/{camera}_depth/{depth_filename}"
                depth_value = (
                    depth_videos.get(camera)
                    if isinstance(depth_videos, dict)
                    else None
                )
                if depth_value != expected_depth:
                    issues.append(
                        f"{prefix} {camera} depth path={depth_value!r}, "
                        f"expected={expected_depth!r}"
                    )
                depth_path = _safe_depth_video_path(
                    episode, depth_value, camera, depth_filename
                )
                if depth_path is None:
                    issues.append(f"{prefix} has unsafe {camera} depth video path")
                    all_paths_safe = False
                    continue
                depth_relative = str(depth_path.relative_to(episode))
                if depth_relative in expected_paths:
                    issues.append(f"{prefix} reuses video path {depth_relative}")
                expected_paths.add(depth_relative)
                clean_depth_videos[camera] = depth_relative
                if not depth_path.is_file() or depth_path.is_symlink():
                    issues.append(
                        f"{prefix} missing or unsafe depth video {depth_relative}"
                    )
                elif depth_path.stat().st_size <= 0:
                    issues.append(f"{prefix} empty depth video {depth_relative}")
        if all_paths_safe and len(clean_videos) == len(CAMERA_KEYS):
            expected_mapping.extend(
                (
                    expected_index,
                    offset,
                    dict(clean_videos),
                    dict(clean_depth_videos) if require_depth else None,
                )
                for offset in range(count)
            )
        next_start += count

    if next_start != frame_count:
        issues.append(
            f"{episode.name}: manifest covers {next_start} frames, expected={frame_count}"
        )

    videos_dir = episode / "videos"
    actual_paths: set[str] = set()
    if videos_dir.is_dir() and not videos_dir.is_symlink():
        for path in (
            *videos_dir.rglob("*.mp4"),
            *videos_dir.rglob("*.mkv"),
            *videos_dir.rglob("*.depth.bl2"),
        ):
            try:
                actual_paths.add(str(path.relative_to(episode)))
            except ValueError:
                issues.append(f"{episode.name}: video path escapes episode: {path}")
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        issues.append(
            f"{episode.name}: video files differ from manifest "
            f"(missing={missing}, extra={extra})"
        )
    partials = [str(path.relative_to(episode)) for path in episode.rglob("*.partial")]
    if partials:
        issues.append(f"{episode.name}: committed episode has partial files {partials}")
    return expected_mapping


def _inspect_committed_episode(path: Path) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if path.is_symlink():
        return (
            {
                "episode_uuid": path.name,
                "path": str(path),
                "committed": False,
                "frame_count": 0,
                "transition_count": 0,
                "duration_sec": 0.0,
                "outcome": "missing",
                "assisted": False,
                "issues": [f"{path.name}: episode directory is a symlink"],
            },
            [f"{path.name}: episode directory is a symlink"],
        )

    metadata = _read_artifact(path, "metadata.json", issues)
    manifest = _read_artifact(path, "manifest.json", issues)
    outcome = _read_artifact(path, "outcome.json", issues)
    committed = _read_artifact(path, "COMMITTED", issues)
    frames = _read_artifact(path, "frames.jsonl", issues, jsonl=True)
    events = _read_artifact(path, "events.jsonl", issues, jsonl=True)

    for artifact_name, artifact in (
        ("metadata", metadata),
        ("manifest", manifest),
        ("outcome", outcome),
        ("COMMITTED", committed),
    ):
        _validate_identity(path.name, artifact_name, artifact, issues)

    frame_count = len(frames)
    transition_count = max(0, frame_count - 1)
    event_count = len(events)
    for artifact_name, artifact in (
        ("metadata", metadata),
        ("manifest", manifest),
        ("outcome", outcome),
    ):
        if artifact:
            _validate_declared_counts(
                path.name,
                artifact_name,
                artifact,
                frame_count=frame_count,
                transition_count=transition_count,
                event_count=event_count,
                issues=issues,
            )

    if metadata:
        if metadata.get("status") != "complete":
            issues.append(
                f"{path.name}: committed metadata status={metadata.get('status')!r}"
            )
        if metadata.get("state_shape") != [STATE_DIM]:
            issues.append(f"{path.name}: metadata state_shape must be [16]")
        if metadata.get("sent_command_shape") != [STATE_DIM]:
            issues.append(f"{path.name}: metadata sent_command_shape must be [16]")
        if metadata.get("camera_keys") != list(CAMERA_KEYS):
            issues.append(f"{path.name}: metadata camera_keys mismatch")
        depth_keys = metadata.get("depth_camera_keys")
        if depth_keys is not None:
            if depth_keys != list(CAMERA_KEYS):
                issues.append(f"{path.name}: metadata depth_camera_keys mismatch")
            if metadata.get("depth_dtype") != "uint16":
                issues.append(f"{path.name}: metadata depth_dtype must be uint16")
            if metadata.get("depth_unit") != "millimetres":
                issues.append(
                    f"{path.name}: metadata depth_unit must be millimetres"
                )
    if manifest and manifest.get("camera_keys") != list(CAMERA_KEYS):
        issues.append(f"{path.name}: manifest camera_keys mismatch")
    _validate_outcome(path.name, outcome, issues)

    require_depth = metadata.get("depth_camera_keys") == list(CAMERA_KEYS)
    depth_codec = metadata.get("depth_video_codec")
    depth_segment_suffix = (
        ".depth.bl2"
        if isinstance(depth_codec, dict)
        and depth_codec.get("encoding") == "uint16_mm_blosc_zstd_v1"
        else ".mkv"
    )
    if require_depth and manifest.get("depth_camera_keys") != list(CAMERA_KEYS):
        issues.append(f"{path.name}: manifest depth_camera_keys mismatch")
    expected_mapping = _validate_manifest(
        path,
        manifest,
        frame_count,
        issues,
        require_depth=require_depth,
        depth_segment_suffix=depth_segment_suffix,
    )
    frame_issues, source_counts, duration = _validate_frames(
        path.name, frames, expected_mapping
    )
    issues.extend(frame_issues)
    issues.extend(_validate_events(path.name, events))

    manifest_timing = manifest.get("capture_timing")
    metadata_timing = metadata.get("capture_timing")
    if manifest_timing != metadata_timing:
        issues.append(f"{path.name}: metadata/manifest capture_timing differ")
    if isinstance(manifest_timing, dict) and frame_count >= 2:
        recorded_duration = manifest_timing.get("duration_sec")
        if not _same_number(recorded_duration, duration):
            issues.append(
                f"{path.name}: capture duration={recorded_duration!r}, "
                f"recomputed={duration}"
            )

    metadata_motion = metadata.get("motion_executed")
    outcome_motion = outcome.get("motion_executed")
    for artifact_name, motion in (
        ("metadata", metadata_motion),
        ("outcome", outcome_motion),
    ):
        if motion is not True and motion is not False and motion is not None:
            issues.append(
                f"{path.name}: {artifact_name} motion_executed must be bool or null"
            )
    if metadata_motion != outcome_motion:
        issues.append(f"{path.name}: metadata/outcome motion_executed differ")

    metadata_publication = metadata.get("command_publication")
    outcome_publication = outcome.get("command_publication")
    _validate_command_publication(
        path.name, "metadata", metadata_publication, issues
    )
    _validate_command_publication(
        path.name, "outcome", outcome_publication, issues
    )
    if metadata_publication != outcome_publication:
        issues.append(f"{path.name}: metadata/outcome command_publication differ")
    if isinstance(outcome_publication, dict):
        task_output_may_have_changed = outcome_publication.get(
            "task_output_may_have_changed"
        )
        if task_output_may_have_changed is True and outcome_motion is not None:
            issues.append(
                f"{path.name}: published task output requires "
                "motion_executed=null without hardware execution ack"
            )

    intervention = outcome.get("intervention_summary")
    declared_assisted = (
        intervention.get("assisted") if isinstance(intervention, dict) else None
    )
    human_command = source_counts.get("human", 0) > 0
    assisted = declared_assisted is True or human_command
    if declared_assisted is False and human_command:
        issues.append(
            f"{path.name}: assisted=false conflicts with human sent_command evidence"
        )

    result = {
        "episode_uuid": path.name,
        "path": str(path),
        "committed": True,
        "outcome": outcome.get("outcome", "missing"),
        "failure_reason": outcome.get("failure_reason"),
        "motion_executed": outcome_motion,
        "command_publication": outcome_publication,
        "assisted": assisted,
        "frame_count": frame_count,
        "transition_count": transition_count,
        "event_count": event_count,
        "duration_sec": duration,
        "sent_command_source_counts": dict(sorted(source_counts.items())),
        "started_at_utc": outcome.get("started_at_utc"),
        "ended_at_utc": outcome.get("ended_at_utc"),
        "bytes": _episode_bytes(path),
        "issues": list(issues),
    }
    return result, issues


def _scan_episode_directories(root: Path) -> tuple[list[Path], list[Path], list[str]]:
    issues: list[str] = []
    episodes_dir = root / "episodes"
    staging_dir = root / ".in_progress"
    if episodes_dir.is_symlink():
        issues.append("episodes directory must not be a symlink")
    if staging_dir.is_symlink():
        issues.append(".in_progress directory must not be a symlink")
    episode_paths = (
        sorted(
            (item for item in episodes_dir.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        )
        if episodes_dir.is_dir() and not episodes_dir.is_symlink()
        else []
    )
    staging_paths = (
        sorted(
            (item for item in staging_dir.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        )
        if staging_dir.is_dir() and not staging_dir.is_symlink()
        else []
    )
    for path in staging_paths:
        if (path / "COMMITTED").exists():
            issues.append(f"{path.name}: COMMITTED episode remains in .in_progress")
    return episode_paths, staging_paths, issues


def summarize(root: str | Path) -> dict[str, Any]:
    """Inspect a rollout root without changing any filesystem content."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"rollout root is not a directory: {root_path}")

    issues: list[str] = []
    index_path = root_path / "episodes.jsonl"
    index_rows: list[dict[str, Any]] = []
    if index_path.exists():
        if not index_path.is_file() or index_path.is_symlink():
            issues.append("episodes.jsonl must be a regular file")
        else:
            try:
                index_rows = _load_jsonl(index_path)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                issues.append(f"cannot parse episodes.jsonl: {exc}")

    index_uuids: list[str] = []
    index_paths: list[str] = []
    valid_index_rows: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(index_rows, 1):
        prefix = f"episodes.jsonl row {row_number}"
        if row.get("schema_version") != SCHEMA_VERSION:
            issues.append(f"{prefix}: unsupported schema_version")
        episode_uuid = row.get("episode_uuid")
        if (
            not _nonempty_string(episode_uuid)
            or Path(str(episode_uuid)).name != episode_uuid
            or episode_uuid in {".", ".."}
        ):
            issues.append(f"{prefix}: invalid episode_uuid={episode_uuid!r}")
            continue
        index_uuids.append(episode_uuid)
        value = row.get("path")
        index_paths.append(value if isinstance(value, str) else repr(value))
        expected_path = f"episodes/{episode_uuid}"
        if value != expected_path:
            issues.append(
                f"{prefix}: path={value!r}, expected={expected_path!r}; "
                "index must never point into .in_progress"
            )
        valid_index_rows[episode_uuid] = row

    duplicate_uuids = sorted(
        value for value, count in Counter(index_uuids).items() if count > 1
    )
    if duplicate_uuids:
        issues.append(f"duplicate episode UUIDs in index: {duplicate_uuids}")
    duplicate_paths = sorted(
        value for value, count in Counter(index_paths).items() if count > 1
    )
    if duplicate_paths:
        issues.append(f"duplicate episode paths in index: {duplicate_paths}")

    episode_paths, staging_paths, directory_issues = _scan_episode_directories(
        root_path
    )
    issues.extend(directory_issues)
    committed_paths: list[Path] = []
    interrupted_paths: list[Path] = []
    for path in episode_paths:
        if (path / "COMMITTED").is_file() and not (path / "COMMITTED").is_symlink():
            committed_paths.append(path)
        else:
            interrupted_paths.append(path)
            issues.append(f"{path.name}: episode directory lacks COMMITTED marker")

    episodes: list[dict[str, Any]] = []
    for path in committed_paths:
        episode, episode_issues = _inspect_committed_episode(path)
        episodes.append(episode)
        issues.extend(episode_issues)

    committed_uuids = {item["episode_uuid"] for item in episodes}
    indexed_uuids = set(index_uuids)
    for episode_uuid in sorted(indexed_uuids - committed_uuids):
        issues.append(f"indexed committed episode is missing: {episode_uuid}")
    for episode_uuid in sorted(committed_uuids - indexed_uuids):
        issues.append(
            f"committed episode is absent from episodes.jsonl: {episode_uuid}"
        )

    for episode in episodes:
        row = valid_index_rows.get(episode["episode_uuid"])
        if row is None:
            continue
        expected = {
            "schema_version": SCHEMA_VERSION,
            "path": f"episodes/{episode['episode_uuid']}",
            "outcome": episode["outcome"],
            "failure_reason": episode["failure_reason"],
            "frame_count": episode["frame_count"],
            "transition_count": episode["transition_count"],
            "motion_executed": episode["motion_executed"],
            "command_publication": episode["command_publication"],
            "started_at_utc": episode["started_at_utc"],
            "ended_at_utc": episode["ended_at_utc"],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                issues.append(
                    f"{episode['episode_uuid']}: episodes.jsonl {key}="
                    f"{row.get(key)!r}, episode={value!r}"
                )

    outcomes = Counter(item["outcome"] for item in episodes)
    outcome_counts = {name: outcomes[name] for name in VALID_OUTCOMES}
    missing_outcomes = sum(
        count for name, count in outcomes.items() if name not in VALID_OUTCOMES
    )
    if missing_outcomes:
        outcome_counts["missing"] = missing_outcomes
    assisted_count = sum(bool(item["assisted"]) for item in episodes)
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root_path),
        "healthy": not issues,
        "issues": issues,
        "index_count": len(index_rows),
        "committed_count": len(episodes),
        "incomplete_count": len(staging_paths) + len(interrupted_paths),
        "outcome_counts": outcome_counts,
        "assisted_count": assisted_count,
        "autonomous_count": len(episodes) - assisted_count,
        "task_output_publish_evidence_count": sum(
            isinstance(item.get("command_publication"), dict)
            and item["command_publication"].get("task_output_may_have_changed")
            is True
            for item in episodes
        ),
        "physical_execution_verified_count": sum(
            isinstance(item.get("command_publication"), dict)
            and item["command_publication"].get("physical_execution_verified")
            is True
            for item in episodes
        ),
        "frame_count": sum(item["frame_count"] for item in episodes),
        "transition_count": sum(item["transition_count"] for item in episodes),
        "duration_sec": sum(item["duration_sec"] for item in episodes),
        "committed_bytes": sum(item["bytes"] for item in episodes),
        "episodes": episodes,
        "incomplete": [
            {
                "episode_uuid": path.name,
                "path": str(path),
                "location": (".in_progress" if path in staging_paths else "episodes"),
                "bytes": _episode_bytes(path),
            }
            for path in [*staging_paths, *interrupted_paths]
        ],
    }


def _format_human(report: dict[str, Any]) -> str:
    outcomes = report["outcome_counts"]
    outcome_text = " ".join(
        f"{name}={outcomes.get(name, 0)}" for name in VALID_OUTCOMES
    )
    lines = [
        f"FA rollout root: {report['root']}",
        (
            f"committed={report['committed_count']} "
            f"incomplete={report['incomplete_count']} "
            f"assisted={report['assisted_count']} "
            f"autonomous={report['autonomous_count']}"
        ),
        f"outcomes: {outcome_text}",
        (
            "task_output_publish_evidence="
            f"{report['task_output_publish_evidence_count']} "
            "physical_execution_verified="
            f"{report['physical_execution_verified_count']}"
        ),
        (
            f"frames={report['frame_count']} "
            f"transitions={report['transition_count']} "
            f"duration_sec={report['duration_sec']:.6f}"
        ),
        f"integrity={'healthy' if report['healthy'] else 'FAILED'}",
    ]
    if report["issues"]:
        lines.append("issues:")
        lines.extend(f"  - {issue}" for issue in report["issues"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="FA real-rollout dataset root")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 2 when committed-data integrity checks fail",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable JSON report",
    )
    args = parser.parse_args(argv)
    try:
        report = summarize(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_human(report), end="")
    return 2 if args.strict and not report["healthy"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CAMERA_KEYS", "SCHEMA_VERSION", "STATE_DIM", "main", "summarize"]
