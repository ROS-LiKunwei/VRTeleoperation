from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Interface telemetry/events flowing upstream to recorders/monitors.


@dataclass(frozen=True)
class RobotState:
    """Minimal robot state for feedback/recording.

    - joint_positions_rad: current joint angles.
    - ee_position_m, ee_orientation_xyzw: current end-effector pose.
    """

    timestamp_s: float
    joint_positions_rad: List[float]
    ee_position_m: Tuple[float, float, float]
    ee_orientation_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ErrorEvent:
    """Error surface communicated by the interface.

    - code: short code; message: human-readable detail.
    """

    timestamp_s: float
    code: str
    message: Optional[str] = None


@dataclass(frozen=True)
class CartesianState:
    """Typed cartesian state wrapper used by interfaces before publishing.

    This is an internal boundary type. When publishing, use ``to_dict()`` to
    preserve the existing wire format expected by downstream consumers
    (e.g. MultiRobotAdapter/BeavrBot), namely::

        {"cartesian_position": [x, y, z], "timestamp": <float>}
    """

    position_m: Optional[Tuple[float, float, float]] = None
    timestamp_s: Optional[float] = None
    h_matrix: Optional[
        Tuple[
            Tuple[float, float, float, float],
            Tuple[float, float, float, float],
            Tuple[float, float, float, float],
            Tuple[float, float, float, float],
        ]
    ] = None

    def to_dict(self) -> dict:
        return {
            "cartesian_position": list(self.position_m) if self.position_m is not None else [0.0, 0.0, 0.0],
            "timestamp": self.timestamp_s if self.timestamp_s is not None else 0.0,
        }


@dataclass(frozen=True)
class CommandedCartesianState:
    """Typed commanded cartesian state wrapper.

    The commanded end-effector target is represented as 7 values in the order
    [x, y, z, qx, qy, qz, qw]. The published wire format is::

        {"commanded_cartesian_position": [..7..], "timestamp": <float>}
    """

    commanded_cartesian_position: Sequence[float]  # length 7
    timestamp_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "commanded_cartesian_position": list(self.commanded_cartesian_position),
            "timestamp": self.timestamp_s,
        }


__all_ = [
    "RobotState",
    "ErrorEvent",
    "CartesianState",
    "CommandedCartesianState",
]
