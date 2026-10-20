"""Hardware-independent control ownership for AHA-WAM robot clients.

Importing this module never starts a thread or opens a device.  Robot clients
must establish a measured-position hold before moving from ``SAFE_HOLD`` to
``HUMAN``.
"""

from __future__ import annotations

import copy
import errno
import fcntl
import os
import sys
import termios
import threading
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional, TextIO


class ControlMode(str, Enum):
    AUTO = "AUTO"
    SAFE_HOLD = "SAFE_HOLD"
    HUMAN = "HUMAN"
    FINALIZED = "FINALIZED"


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"
    ERROR = "error"
    UNLABELED = "unlabeled"


@dataclass(frozen=True)
class InterventionEvent:
    sequence: int
    timestamp: float
    event: str
    previous_mode: ControlMode
    mode: ControlMode
    outcome: EpisodeOutcome
    reason: str
    source: str
    automatic: bool
    chunk: Optional[int]
    step: Optional[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event": self.event,
            "previous_mode": self.previous_mode.value,
            "mode": self.mode.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "source": self.source,
            "automatic": self.automatic,
            "chunk": self.chunk,
            "step": self.step,
        }


@dataclass(frozen=True)
class InterventionSnapshot:
    enabled: bool
    mode: ControlMode
    outcome: EpisodeOutcome
    revision: int
    updated_at: float
    takeover_reason: Optional[str]
    takeover_source: Optional[str]
    takeover_automatic: bool
    takeover_chunk: Optional[int]
    takeover_step: Optional[int]
    final_reason: Optional[str]
    final_source: Optional[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "outcome": self.outcome.value,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "takeover_reason": self.takeover_reason,
            "takeover_source": self.takeover_source,
            "takeover_automatic": self.takeover_automatic,
            "takeover_chunk": self.takeover_chunk,
            "takeover_step": self.takeover_step,
            "final_reason": self.final_reason,
            "final_source": self.final_source,
        }


def _text(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


class OperatorIntervention:
    """Thread-safe exclusive command owner and terminal episode label."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._condition = threading.Condition(threading.RLock())
        self._mode = ControlMode.AUTO
        self._outcome = EpisodeOutcome.UNLABELED
        self._revision = 0
        self._updated_at = time.time()
        self._takeover_reason: Optional[str] = None
        self._takeover_source: Optional[str] = None
        self._takeover_automatic = False
        self._takeover_chunk: Optional[int] = None
        self._takeover_step: Optional[int] = None
        self._final_reason: Optional[str] = None
        self._final_source: Optional[str] = None
        self._events: list[InterventionEvent] = []
        self._guard_owner: Optional[int] = None

    @property
    def mode(self) -> ControlMode:
        with self._condition:
            return self._mode

    @property
    def outcome(self) -> EpisodeOutcome:
        with self._condition:
            return self._outcome

    def request_takeover(
        self,
        reason: str,
        *,
        automatic: bool = False,
        chunk: Optional[int] = None,
        step: Optional[int] = None,
    ) -> InterventionSnapshot:
        reason = _text(reason, "takeover reason")
        with self._condition:
            if not self.enabled or self._mode is ControlMode.FINALIZED:
                return self._snapshot_locked()
            if self._mode is ControlMode.SAFE_HOLD:
                return self._snapshot_locked()
            self._reject_reentrant_locked()
            previous = self._mode
            self._mode = ControlMode.SAFE_HOLD
            self._takeover_reason = reason
            self._takeover_source = "automatic" if automatic else "operator"
            self._takeover_automatic = bool(automatic)
            self._takeover_chunk = None if chunk is None else int(chunk)
            self._takeover_step = None if step is None else int(step)
            self._record_locked(
                "takeover_requested", previous, reason, self._takeover_source
            )
            return self._snapshot_locked()

    def enter_human(self) -> InterventionSnapshot:
        with self._condition:
            if self._mode is ControlMode.HUMAN:
                return self._snapshot_locked()
            if self._mode is not ControlMode.SAFE_HOLD:
                raise RuntimeError(
                    "human control requires SAFE_HOLD, current mode is "
                    f"{self._mode.value}"
                )
            self._reject_reentrant_locked()
            previous = self._mode
            self._mode = ControlMode.HUMAN
            self._record_locked(
                "human_entered",
                previous,
                self._takeover_reason or "operator takeover",
                "operator",
            )
            return self._snapshot_locked()

    def finish(
        self,
        outcome: EpisodeOutcome | str,
        reason: str,
        source: str,
    ) -> InterventionSnapshot:
        normalized = (
            outcome
            if isinstance(outcome, EpisodeOutcome)
            else EpisodeOutcome(str(outcome).strip().lower())
        )
        reason = _text(reason, "final reason")
        source = _text(source, "final source")
        with self._condition:
            if self._mode is ControlMode.FINALIZED:
                if normalized is not self._outcome:
                    raise RuntimeError(
                        f"episode already finalized as {self._outcome.value}"
                    )
                return self._snapshot_locked()
            self._reject_reentrant_locked()
            previous = self._mode
            self._mode = ControlMode.FINALIZED
            self._outcome = normalized
            self._final_reason = reason
            self._final_source = source
            self._record_locked("episode_finalized", previous, reason, source)
            return self._snapshot_locked()

    def snapshot(self) -> InterventionSnapshot:
        with self._condition:
            return self._snapshot_locked()

    def events(self) -> tuple[InterventionEvent, ...]:
        with self._condition:
            return tuple(self._events)

    @contextmanager
    def command_authority(self, source: str) -> Iterator[None]:
        source = str(source).strip().lower()
        allowed = {
            "model": frozenset({ControlMode.AUTO}),
            "human": frozenset({ControlMode.HUMAN}),
            "hold": frozenset(
                {ControlMode.AUTO, ControlMode.SAFE_HOLD, ControlMode.FINALIZED}
            ),
        }
        if source not in allowed:
            raise RuntimeError(f"unknown command source: {source!r}")
        with self._condition:
            if self._mode not in allowed[source]:
                raise RuntimeError(
                    f"{source} command is not authorized in {self._mode.value}"
                )
            if self._guard_owner is not None:
                raise RuntimeError("nested command authority is not allowed")
            self._guard_owner = threading.get_ident()
            try:
                yield
            finally:
                self._guard_owner = None

    def _record_locked(
        self,
        event: str,
        previous: ControlMode,
        reason: str,
        source: str,
    ) -> None:
        self._revision += 1
        self._updated_at = time.time()
        self._events.append(
            InterventionEvent(
                sequence=self._revision,
                timestamp=self._updated_at,
                event=event,
                previous_mode=previous,
                mode=self._mode,
                outcome=self._outcome,
                reason=reason,
                source=source,
                automatic=self._takeover_automatic,
                chunk=self._takeover_chunk,
                step=self._takeover_step,
            )
        )
        self._condition.notify_all()

    def _reject_reentrant_locked(self) -> None:
        if self._guard_owner == threading.get_ident():
            raise RuntimeError("cannot change control mode inside command authority")

    def _snapshot_locked(self) -> InterventionSnapshot:
        return InterventionSnapshot(
            enabled=self.enabled,
            mode=self._mode,
            outcome=self._outcome,
            revision=self._revision,
            updated_at=self._updated_at,
            takeover_reason=self._takeover_reason,
            takeover_source=self._takeover_source,
            takeover_automatic=self._takeover_automatic,
            takeover_chunk=self._takeover_chunk,
            takeover_step=self._takeover_step,
            final_reason=self._final_reason,
            final_source=self._final_source,
        )


class TerminalKeyListener:
    """Context-managed nonblocking key reader which restores the TTY."""

    def __init__(
        self, enabled: bool = True, stream: Optional[TextIO | int] = None
    ) -> None:
        self.enabled = bool(enabled)
        self.stream = sys.stdin if stream is None else stream
        self._lock = threading.RLock()
        self._fd: Optional[int] = None
        self._saved_termios: Optional[list[object]] = None
        self._saved_flags: Optional[int] = None
        self._active = False
        self._entered = False

    @property
    def available(self) -> bool:
        with self._lock:
            return self._active

    def __enter__(self) -> "TerminalKeyListener":
        with self._lock:
            if self._entered:
                return self
            self._entered = True
            if not self.enabled:
                return self
            try:
                fd = int(self.stream) if isinstance(self.stream, int) else self.stream.fileno()
            except (AttributeError, OSError, TypeError, ValueError):
                return self
            if not os.isatty(fd):
                return self
            saved_termios = copy.deepcopy(termios.tcgetattr(fd))
            saved_flags = int(fcntl.fcntl(fd, fcntl.F_GETFL))
            try:
                tty.setcbreak(fd, termios.TCSANOW)
                fcntl.fcntl(fd, fcntl.F_SETFL, saved_flags | os.O_NONBLOCK)
            except BaseException:
                termios.tcsetattr(fd, termios.TCSANOW, saved_termios)
                fcntl.fcntl(fd, fcntl.F_SETFL, saved_flags)
                self._entered = False
                raise
            self._fd = fd
            self._saved_termios = saved_termios
            self._saved_flags = saved_flags
            self._active = True
            return self

    def poll(self) -> list[str]:
        with self._lock:
            if not self._active or self._fd is None:
                return []
            chunks: list[bytes] = []
            while True:
                try:
                    data = os.read(self._fd, 4096)
                except BlockingIOError:
                    break
                except OSError as exc:
                    if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                        break
                    raise
                if not data:
                    break
                chunks.append(data)
            return list(b"".join(chunks).decode("utf-8", errors="ignore"))

    def close(self) -> None:
        with self._lock:
            errors: list[BaseException] = []
            if self._fd is not None and self._saved_termios is not None:
                try:
                    termios.tcsetattr(
                        self._fd, termios.TCSANOW, self._saved_termios
                    )
                except BaseException as exc:
                    errors.append(exc)
            if self._fd is not None and self._saved_flags is not None:
                try:
                    fcntl.fcntl(self._fd, fcntl.F_SETFL, self._saved_flags)
                except BaseException as exc:
                    errors.append(exc)
            self._fd = None
            self._saved_termios = None
            self._saved_flags = None
            self._active = False
            self._entered = False
            if errors:
                raise OSError("failed to restore terminal state") from errors[0]

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except OSError:
            if exc_type is None:
                raise
        return False


__all__ = [
    "ControlMode",
    "EpisodeOutcome",
    "InterventionEvent",
    "InterventionSnapshot",
    "OperatorIntervention",
    "TerminalKeyListener",
]
