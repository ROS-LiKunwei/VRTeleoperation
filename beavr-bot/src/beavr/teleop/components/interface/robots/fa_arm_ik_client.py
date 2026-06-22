"""FA arm IK client backed by the `ik_7dof` pybind module."""

from __future__ import annotations

import abc
import csv
import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from beavr.teleop.components.interface.robots.fa_command_builder import FA_ARM_JOINT_COUNT
from beavr.teleop.components.operator.operator_types import CartesianTarget

_ACCEPTABLE_POSITION_THRESHOLD_M = 0.05
_ACCEPTABLE_ORIENTATION_THRESHOLD_RAD = 0.5


@dataclass(frozen=True)
class FaArmIkConfig:
    urdf_file: str
    srdf_file: str = ""
    module_name: str = "ik_7dof_pybind"
    reference_frame: str = "pelvis"
    max_iters: int = 200
    max_joint_step_rad: float = 1.0
    eps: float = 1e-3
    log_enabled: bool = True
    log_dir: str = ""

    def __post_init__(self):
        if self.reference_frame not in ("pelvis", "arm_base"):
            raise ValueError("reference_frame must be pelvis or arm_base")
        if not str(self.module_name).strip():
            raise ValueError("module_name must not be empty")
        object.__setattr__(self, "max_iters", max(1, int(self.max_iters)))
        object.__setattr__(self, "max_joint_step_rad", max(0.0, float(self.max_joint_step_rad)))
        object.__setattr__(self, "eps", max(1e-9, float(self.eps)))


@dataclass(frozen=True)
class FaArmIkResult:
    success: bool
    q_target: tuple[float, ...]
    has_solution: bool = False
    position_error: float = 0.0
    orientation_error: float = 0.0
    iterations: int = 0
    solve_time_ms: float = 0.0
    message: str = ""

    def __post_init__(self):
        if len(self.q_target) != FA_ARM_JOINT_COUNT:
            raise ValueError(f"FA IK q_target must contain 7 values, got {len(self.q_target)}")
        if not np.all(np.isfinite(np.asarray(self.q_target, dtype=np.float64))):
            raise ValueError("FA IK q_target contains NaN/Inf")


class FaArmIkClientBase(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        hand_side: str,
        target: CartesianTarget,
        current_arm_q: Sequence[float],
    ) -> FaArmIkResult:
        """Return a 7D target for one FA arm."""

    def compute_fk(self, hand_side: str, current_arm_q: Sequence[float]) -> np.ndarray | None:
        return None


class FaPybindIkClient(FaArmIkClientBase):
    """Direct Python binding for `fa_arm_kinematic::IKSolver`."""

    def __init__(self, config: FaArmIkConfig):
        self.config = config
        self._ik_logger = _FaIkCsvLogger(config.log_dir) if config.log_enabled else None
        try:
            module = _import_ik_module(config.module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"FA IK pybind module '{config.module_name}' is unavailable. "
                "Build ik_7dof and either source its install/setup.bash or set "
                "IK_7DOF_INSTALL_PREFIX to the deployed install/ik_7dof directory."
            ) from exc
        self._solver = module.FaIkSolver(config.urdf_file, config.srdf_file)

    def solve(self, hand_side: str, target: CartesianTarget, current_arm_q: Sequence[float]) -> FaArmIkResult:
        current = _as_arm_q(current_arm_q)
        rotation = _quat_xyzw_to_rotation(target.orientation_xyzw)
        out = self._solver.solve_arm_ik(
            np.asarray(target.position_m, dtype=np.float64),
            rotation,
            hand_side,
            [float(v) for v in current],
            self.config.reference_frame,
            self.config.max_iters,
            self.config.eps,
        )
        q_target = out.get("q_target") or ()
        try:
            q_target_array = _as_arm_q(q_target)
            has_usable_q_target = True
        except (TypeError, ValueError):
            q_target_array = current
            has_usable_q_target = False
        q_target_array = _limit_joint_step(current, q_target_array, self.config.max_joint_step_rad)
        position_error = float(out.get("position_error", np.inf))
        orientation_error = float(out.get("orientation_error", np.inf))
        acceptable_solution = (
            position_error <= _ACCEPTABLE_POSITION_THRESHOLD_M
            and orientation_error <= _ACCEPTABLE_ORIENTATION_THRESHOLD_RAD
        )
        has_solution = has_usable_q_target
        success = has_solution
        if success and not acceptable_solution:
            message = "using best approximate IK outside the usable position/orientation threshold"
        elif success:
            message = ""
        else:
            message = "ik_7dof did not return a usable q_target"
        result = FaArmIkResult(
            success=success,
            q_target=tuple(float(v) for v in q_target_array),
            has_solution=has_solution,
            position_error=position_error,
            orientation_error=orientation_error,
            iterations=int(out.get("iterations", 0)),
            solve_time_ms=float(out.get("solve_time_ms", 0.0)),
            message=message,
        )
        if self._ik_logger is not None:
            self._ik_logger.log_solve(
                hand_side=hand_side,
                target=target,
                current_arm_q=current,
                q_target=result.q_target,
                config=self.config,
                result=result,
            )
        return result

    def compute_fk(self, hand_side: str, current_arm_q: Sequence[float]) -> np.ndarray | None:
        out = self._solver.compute_arm_fk(
            [float(v) for v in _as_arm_q(current_arm_q)],
            hand_side,
            self.config.reference_frame,
        )
        homo = np.eye(4, dtype=np.float64)
        homo[:3, :3] = np.asarray(out["rotation"], dtype=np.float64).reshape(3, 3)
        homo[:3, 3] = np.asarray(out["translation"], dtype=np.float64).reshape(3)
        return homo


def _as_arm_q(values: Sequence[float]) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (FA_ARM_JOINT_COUNT,) or not np.all(np.isfinite(q)):
        raise ValueError("FA arm IK input/output must contain 7 finite joints")
    return q


def _limit_joint_step(current: np.ndarray, target: np.ndarray, max_step_rad: float) -> np.ndarray:
    if max_step_rad <= 0.0:
        return target
    delta = target - current
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm <= max_step_rad:
        return target
    return current + delta * (max_step_rad / delta_norm)


def _quat_xyzw_to_rotation(quat_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("FA IK target orientation must be a finite xyzw quaternion")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("FA IK target orientation quaternion must be non-zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _import_ik_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    for candidate in _candidate_ik_module_paths():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return importlib.import_module(module_name)


def _candidate_ik_module_paths() -> list[Path]:
    paths: list[Path] = []
    explicit_path = os.environ.get("IK_7DOF_PYTHONPATH")
    if explicit_path:
        paths.extend(Path(item).expanduser() for item in explicit_path.split(os.pathsep) if item)

    install_prefix = os.environ.get("IK_7DOF_INSTALL_PREFIX")
    if install_prefix:
        paths.extend((Path(install_prefix).expanduser() / "local/lib").glob("python*/dist-packages"))

    paths.extend(Path("/home/likunwei/humanoid_ws/install/ik_7dof/local/lib").glob("python*/dist-packages"))
    return [path for path in paths if path.exists()]


class _FaIkCsvLogger:
    _HEADER = (
        "wall_time_s",
        "wall_time_iso",
        "target_timestamp_s",
        "hand_side",
        "frame_id",
        "hand_command",
        "reference_frame",
        "max_iters",
        "max_joint_step_rad",
        "eps",
        "target_x",
        "target_y",
        "target_z",
        "target_qx",
        "target_qy",
        "target_qz",
        "target_qw",
        "current_q0",
        "current_q1",
        "current_q2",
        "current_q3",
        "current_q4",
        "current_q5",
        "current_q6",
        "success",
        "has_solution",
        "position_error_m",
        "orientation_error_rad",
        "iterations",
        "solve_time_ms",
        "message",
        "q_target0",
        "q_target1",
        "q_target2",
        "q_target3",
        "q_target4",
        "q_target5",
        "q_target6",
    )

    def __init__(self, log_dir: str):
        self._lock = threading.Lock()
        self.log_dir = Path(log_dir).expanduser() if log_dir else _default_ik_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")[:-3]
        self.log_file = self.log_dir / f"fa_ik_{timestamp}_pid{os.getpid()}.csv"
        self._file = self.log_file.open("a", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self._HEADER)
        self._writer.writeheader()

    def log_solve(
        self,
        *,
        hand_side: str,
        target: CartesianTarget,
        current_arm_q: Sequence[float],
        q_target: Sequence[float],
        config: FaArmIkConfig,
        result: FaArmIkResult,
    ) -> None:
        now_s = time.time()
        row = {
            "wall_time_s": f"{now_s:.6f}",
            "wall_time_iso": datetime.fromtimestamp(now_s).isoformat(timespec="milliseconds"),
            "target_timestamp_s": f"{float(getattr(target, 'timestamp_s', 0.0) or 0.0):.6f}",
            "hand_side": hand_side,
            "frame_id": getattr(target, "frame_id", ""),
            "hand_command": getattr(target, "hand_command", ""),
            "reference_frame": config.reference_frame,
            "max_iters": int(config.max_iters),
            "max_joint_step_rad": f"{float(config.max_joint_step_rad):.9g}",
            "eps": f"{float(config.eps):.9g}",
            "success": int(result.success),
            "has_solution": int(result.has_solution),
            "position_error_m": f"{result.position_error:.9g}",
            "orientation_error_rad": f"{result.orientation_error:.9g}",
            "iterations": int(result.iterations),
            "solve_time_ms": f"{result.solve_time_ms:.9g}",
            "message": result.message,
        }
        for idx, value in enumerate(target.position_m):
            row[f"target_{'xyz'[idx]}"] = f"{float(value):.9g}"
        for idx, value in enumerate(target.orientation_xyzw):
            row[f"target_q{'xyzw'[idx]}"] = f"{float(value):.9g}"
        for idx, value in enumerate(current_arm_q):
            row[f"current_q{idx}"] = f"{float(value):.9g}"
        for idx, value in enumerate(q_target):
            row[f"q_target{idx}"] = f"{float(value):.9g}"
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()


def _default_ik_log_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "beavr-bot":
            return parent / "Log" / "ik"
    return Path.cwd() / "Log" / "ik"


__all__ = [
    "FaArmIkConfig",
    "FaArmIkResult",
    "FaArmIkClientBase",
    "FaPybindIkClient",
]
