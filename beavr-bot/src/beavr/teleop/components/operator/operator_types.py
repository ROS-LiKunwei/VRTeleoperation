from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

# Operator → Interface contracts: explicit, minimal, and stable.
# Notes:
# - All fields are explicit (no **kwargs) per user preference.
# - Units: positions in meters, orientations as XYZW unit quaternions,
#   velocities in m/s and rad/s, time is seconds (monotonic).
# - "hand_side" distinguishes left/right streams when bimanual [[memory:2739081]].


HandSide = Literal["left", "right"]


@dataclass(frozen=True)
class CartesianTarget:
    """High-level end-effector target for impedance/pose control.

    - frame_id: reference frame for pose (e.g., "base", "task").
    - position_m: (x, y, z) in meters.
    - orientation_xyzw: unit quaternion (x, y, z, w).
    - linear_velocity_ff: optional feedforward linear velocity (m/s).
    - angular_velocity_ff: optional feedforward angular velocity (rad/s).
    - stiffness and damping can be scalar or per-axis tuples when supported.
    """

    timestamp_s: float
    hand_side: HandSide
    frame_id: str
    position_m: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]
    linear_velocity_ff: Optional[Tuple[float, float, float]] = None
    angular_velocity_ff: Optional[Tuple[float, float, float]] = None
    stiffness: Optional[Tuple[float, float, float]] = None
    damping: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class JointTarget:
    """Joint-space target for position control.

    - joint_positions_rad: desired joint angles (radians).
    - joint_velocities_ff: optional feedforward joint velocities (rad/s).
    - gains: optional per-joint position gains when supported by the interface.
    """

    timestamp_s: float
    hand_side: HandSide
    joint_positions_rad: List[float]
    joint_velocities_ff: Optional[List[float]] = None
    gains: Optional[List[float]] = None


@dataclass(frozen=True)
class GripperCommand:
    """Simple gripper command.

    - width_m: target opening width in meters.
    - speed_mps: optional opening/closing speed (m/s).
    """

    timestamp_s: float
    hand_side: HandSide
    width_m: float
    speed_mps: Optional[float] = None


@dataclass(frozen=True)
class ModeChange:
    """Request the interface to switch control modes.

    Common modes include: "idle", "stop", "cartesian_impedance", "joint_position".
    """

    timestamp_s: float
    hand_side: HandSide
    mode: Literal["idle", "stop", "cartesian_impedance", "joint_position"]
    reason: Optional[str] = None


__all__ = [
    "HandSide",
    "CartesianTarget",
    "JointTarget",
    "GripperCommand",
    "ModeChange",
]
