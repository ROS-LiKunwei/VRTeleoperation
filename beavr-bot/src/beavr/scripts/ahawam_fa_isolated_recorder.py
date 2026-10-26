"""Process-isolated facade for the FA RGB-D episode recorder.

The camera/ROS process copies one synchronized RGB-D observation into a fixed
shared-memory slot and sends only metadata over a pipe.  A spawned child owns
all video encoders, compression threads, filesystem writes, hashing and final
validation.  The synchronous acknowledgement means the slot is never reused
until the child has copied it into its own bounded recorder queue.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import traceback
from collections.abc import Mapping
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Optional

import numpy as np

from beavr.scripts.ahawam_fa_episode_recorder import (
    CAMERA_KEYS,
    CAMERA_SHAPES,
    DEPTH_SHAPES,
    FaEpisodeRecorder,
    RecorderError,
    RecorderFailedError,
)


RPC_TIMEOUT_S = 15.0
FINALIZE_TIMEOUT_S = 180.0


def _layout() -> tuple[dict[str, tuple[int, tuple[int, ...], np.dtype]], int]:
    result: dict[str, tuple[int, tuple[int, ...], np.dtype]] = {}
    offset = 0
    for prefix, shapes, dtype in (
        ("rgb", CAMERA_SHAPES, np.dtype(np.uint8)),
        ("depth", DEPTH_SHAPES, np.dtype(np.uint16)),
    ):
        for key in CAMERA_KEYS:
            shape = tuple(shapes[key])
            if offset % dtype.alignment:
                offset += dtype.alignment - offset % dtype.alignment
            result[f"{prefix}:{key}"] = (offset, shape, dtype)
            offset += int(np.prod(shape)) * dtype.itemsize
    return result, offset


_SHARED_LAYOUT, _SHARED_BYTES = _layout()


def _views(buffer: memoryview) -> dict[str, np.ndarray]:
    return {
        name: np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
        for name, (offset, shape, dtype) in _SHARED_LAYOUT.items()
    }


def _send_ok(connection: Any, value: Any = None) -> None:
    connection.send({"ok": True, "value": value})


def _send_error(connection: Any, exc: BaseException) -> None:
    connection.send(
        {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


def _recorder_process_main(
    connection: Any,
    shared_memory_name: str,
    recorder_kwargs: dict[str, Any],
) -> None:
    shared = shared_memory.SharedMemory(name=shared_memory_name, create=False)
    shared_views = _views(shared.buf)
    recorder: Optional[FaEpisodeRecorder] = None
    try:
        try:
            recorder = FaEpisodeRecorder(**recorder_kwargs)
            _send_ok(connection, {"pid": mp.current_process().pid})
        except BaseException as exc:
            _send_error(connection, exc)
            return

        while True:
            try:
                command, payload = connection.recv()
            except EOFError:
                break
            try:
                if command == "submit_observation":
                    assert recorder is not None
                    value = recorder.submit_observation(
                        images={
                            key: shared_views[f"rgb:{key}"] for key in CAMERA_KEYS
                        },
                        depths={
                            key: shared_views[f"depth:{key}"] for key in CAMERA_KEYS
                        },
                        **payload,
                    )
                    _send_ok(connection, value)
                elif command == "record_event":
                    assert recorder is not None
                    event_type = payload.pop("event_type")
                    _send_ok(connection, recorder.record_event(event_type, **payload))
                elif command == "note_successful_command":
                    assert recorder is not None
                    recorder.note_successful_command(payload["source"])
                    _send_ok(connection)
                elif command == "note_partial_command":
                    assert recorder is not None
                    recorder.note_partial_command(payload["receipt"])
                    _send_ok(connection)
                elif command == "raise_if_failed":
                    assert recorder is not None
                    recorder.raise_if_failed()
                    _send_ok(connection)
                elif command == "finalize":
                    assert recorder is not None
                    path = recorder.finalize(**payload)
                    _send_ok(connection, str(path))
                elif command == "abort":
                    assert recorder is not None
                    path = recorder.abort(**payload)
                    _send_ok(connection, str(path))
                elif command == "close":
                    assert recorder is not None
                    path = recorder.close()
                    _send_ok(connection, str(path))
                    break
                else:
                    raise RecorderError(f"unknown isolated-recorder command: {command}")
            except BaseException as exc:
                _send_error(connection, exc)
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except BaseException:
                pass
        connection.close()
        # Drop NumPy exports before closing the shared-memory mapping.
        shared_views.clear()
        shared.close()


class IsolatedFaEpisodeRecorder:
    """API-compatible recorder proxy with all encoding in a spawned process."""

    def __init__(
        self,
        root: str | Path,
        *,
        fps: int = 30,
        segment_sec: float = 10.0,
        queue_size: int = 128,
        min_free_gb: float = 5.0,
        config: Optional[Mapping[str, Any]] = None,
        validate_timing: bool = True,
        max_relative_fps_error: float = 0.25,
        max_frame_gap_factor: float = 3.0,
    ) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._frame_count = 0
        self._shared = shared_memory.SharedMemory(create=True, size=_SHARED_BYTES)
        self._views = _views(self._shared.buf)
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._connection = parent_connection
        recorder_kwargs = {
            "root": str(Path(root).expanduser().resolve()),
            "fps": fps,
            "segment_sec": segment_sec,
            "queue_size": queue_size,
            "min_free_gb": min_free_gb,
            "config": dict(config or {}),
            "validate_timing": validate_timing,
            "max_relative_fps_error": max_relative_fps_error,
            "max_frame_gap_factor": max_frame_gap_factor,
        }
        self._process = context.Process(
            target=_recorder_process_main,
            args=(child_connection, self._shared.name, recorder_kwargs),
            name="fa-episode-recorder-process",
            daemon=False,
        )
        self._process.start()
        child_connection.close()
        try:
            startup = self._receive(RPC_TIMEOUT_S, "recorder startup")
            self.process_pid = int(startup["pid"])
        except BaseException:
            self._shutdown_process()
            self._release_shared_memory()
            raise

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _receive(self, timeout_s: float, operation: str) -> Any:
        if not self._connection.poll(timeout_s):
            if not self._process.is_alive():
                raise RecorderFailedError(
                    f"isolated recorder exited during {operation} "
                    f"with code {self._process.exitcode}"
                )
            raise RecorderFailedError(f"isolated recorder timed out during {operation}")
        response = self._connection.recv()
        if not isinstance(response, dict) or response.get("ok") is not True:
            if not isinstance(response, dict):
                raise RecorderFailedError(
                    f"invalid isolated recorder response during {operation}"
                )
            raise RecorderFailedError(
                f"isolated recorder {response.get('error_type', 'error')} during "
                f"{operation}: {response.get('error', '')}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("value")

    def _rpc(
        self,
        command: str,
        payload: Optional[dict[str, Any]] = None,
        *,
        timeout_s: float = RPC_TIMEOUT_S,
    ) -> Any:
        with self._lock:
            if self._closed:
                raise RecorderFailedError("isolated recorder is closed")
            if not self._process.is_alive():
                raise RecorderFailedError(
                    f"isolated recorder process exited with code {self._process.exitcode}"
                )
            self._connection.send((command, {} if payload is None else payload))
            return self._receive(timeout_s, command)

    def submit_observation(
        self,
        *,
        images: Mapping[str, np.ndarray],
        depths: Mapping[str, np.ndarray],
        state: np.ndarray,
        camera_metadata: Mapping[str, Any],
        state_metadata: Optional[Mapping[str, Any]],
        sync_metadata: Optional[Mapping[str, Any]],
        sequence: int,
        captured_at: float,
        sent_command: Optional[Any],
        intervention_snapshot: Optional[Any] = None,
    ) -> int:
        with self._lock:
            if self._closed:
                raise RecorderFailedError("isolated recorder is closed")
            for key in CAMERA_KEYS:
                np.copyto(
                    self._views[f"rgb:{key}"],
                    np.asarray(images[key]),
                    casting="no",
                )
                np.copyto(
                    self._views[f"depth:{key}"],
                    np.asarray(depths[key]),
                    casting="no",
                )
            payload = {
                "state": np.asarray(state, dtype=np.float32),
                "camera_metadata": dict(camera_metadata),
                "state_metadata": state_metadata,
                "sync_metadata": sync_metadata,
                "sequence": sequence,
                "captured_at": captured_at,
                "sent_command": sent_command,
                "intervention_snapshot": intervention_snapshot,
            }
            if not self._process.is_alive():
                raise RecorderFailedError(
                    f"isolated recorder process exited with code {self._process.exitcode}"
                )
            self._connection.send(("submit_observation", payload))
            value = self._receive(RPC_TIMEOUT_S, "submit_observation")
            self._frame_count += 1
            return int(value)

    def record_event(self, event_type: str, **payload: Any) -> int:
        return int(self._rpc("record_event", {"event_type": event_type, **payload}))

    def note_successful_command(self, source: str) -> None:
        self._rpc("note_successful_command", {"source": source})

    def note_partial_command(self, receipt: Mapping[str, Any]) -> None:
        self._rpc("note_partial_command", {"receipt": dict(receipt)})

    def raise_if_failed(self) -> None:
        self._rpc("raise_if_failed")

    def finalize(
        self,
        outcome: str,
        *,
        reason: Optional[str] = None,
        failure_reason: Optional[str] = None,
        intervention_summary: Optional[Any] = None,
    ) -> Path:
        value = self._rpc(
            "finalize",
            {
                "outcome": outcome,
                "reason": reason,
                "failure_reason": failure_reason,
                "intervention_summary": intervention_summary,
            },
            timeout_s=FINALIZE_TIMEOUT_S,
        )
        return Path(value)

    def abort(
        self,
        reason: str,
        *,
        outcome: str = "aborted",
        intervention_summary: Optional[Any] = None,
    ) -> Path:
        value = self._rpc(
            "abort",
            {
                "reason": reason,
                "outcome": outcome,
                "intervention_summary": intervention_summary,
            },
            timeout_s=FINALIZE_TIMEOUT_S,
        )
        return Path(value)

    def close(self) -> Path:
        with self._lock:
            if self._closed:
                return Path()
            try:
                if self._process.is_alive():
                    self._connection.send(("close", {}))
                    value = self._receive(FINALIZE_TIMEOUT_S, "close")
                else:
                    value = ""
            finally:
                self._closed = True
                self._shutdown_process()
                self._connection.close()
                self._release_shared_memory()
            return Path(value)

    def _shutdown_process(self) -> None:
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5.0)

    def _release_shared_memory(self) -> None:
        self._views.clear()
        self._shared.close()
        try:
            self._shared.unlink()
        except FileNotFoundError:
            pass
