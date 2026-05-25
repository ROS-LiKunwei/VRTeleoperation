#!/usr/bin/env python3
"""
Offline self-test for SYSMO-32 VR-to-robot relative pose retargeting.

This script does not open ZMQ sockets and does not require MuJoCo. It mirrors the
relative-pose math used by XArmOperator._apply_retargeted_angles() for SYSMO-32:

    Unity/PICO hand pose -> internal right-handed VR pose -> robot-base target

Run:
    uv run python scripts/self_test_sysmo32_retarget.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from beavr.teleop.components.detector.vr.keypoint_transform import (
    UNITY_LEFT_TO_INTERNAL_RIGHT,
)
from beavr.teleop.components.operator.robots.sysmo32_operator import H_R_V_SYSMO32


ATOL_POS = 1e-9
ATOL_ROT = 1e-9


@dataclass(frozen=True)
class PoseCase:
    name: str
    unity_delta_m: np.ndarray
    internal_delta_m: np.ndarray
    expected_robot_delta_m: np.ndarray
    internal_rotation: Rotation = Rotation.identity()


def homo(position_m: np.ndarray, rotation: Rotation | None = None) -> np.ndarray:
    h = np.eye(4)
    h[:3, 3] = position_m
    h[:3, :3] = Rotation.identity().as_matrix() if rotation is None else rotation.as_matrix()
    return h


def unity_pose_to_internal_vr_homo(unity_position_m: np.ndarray, internal_rotation: Rotation) -> np.ndarray:
    internal_position_m = unity_position_m * UNITY_LEFT_TO_INTERNAL_RIGHT
    return homo(internal_position_m, internal_rotation)


def retarget_like_xarm_operator(
    robot_init_h: np.ndarray,
    hand_init_h: np.ndarray,
    hand_moving_h: np.ndarray,
    resolution_scale: float = 1.0,
) -> np.ndarray:
    """Same relative transform as XArmOperator._apply_retargeted_angles()."""
    h_ht_hi = np.linalg.inv(hand_init_h) @ hand_moving_h

    h_robot_to_vr = H_R_V_SYSMO32
    r_robot_to_vr = h_robot_to_vr[:3, :3]
    r_vr_to_robot = np.linalg.inv(r_robot_to_vr)

    relative_affine_in_robot_frame = np.eye(4)
    relative_affine_in_robot_frame[:3, :3] = r_vr_to_robot @ h_ht_hi[:3, :3] @ r_robot_to_vr
    relative_affine_in_robot_frame[:3, 3] = (
        r_vr_to_robot @ (hand_moving_h[:3, 3] - hand_init_h[:3, 3]) * resolution_scale
    )

    target_h = np.eye(4)
    target_h[:3, :3] = relative_affine_in_robot_frame[:3, :3] @ robot_init_h[:3, :3]
    target_h[:3, 3] = robot_init_h[:3, 3] + relative_affine_in_robot_frame[:3, 3]
    return target_h


def expected_robot_rotation_delta(internal_rotation: Rotation) -> np.ndarray:
    r_robot_to_vr = H_R_V_SYSMO32[:3, :3]
    r_vr_to_robot = np.linalg.inv(r_robot_to_vr)
    return r_vr_to_robot @ internal_rotation.as_matrix() @ r_robot_to_vr


def format_pose(h: np.ndarray) -> str:
    quat_xyzw = Rotation.from_matrix(h[:3, :3]).as_quat()
    pos = h[:3, 3]
    return (
        f"pos=({pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}), "
        f"quat_xyzw=({quat_xyzw[0]: .6f}, {quat_xyzw[1]: .6f}, "
        f"{quat_xyzw[2]: .6f}, {quat_xyzw[3]: .6f})"
    )


def check_case(side: str, robot_init_h: np.ndarray, hand_init_h: np.ndarray, case: PoseCase) -> None:
    unity_init = hand_init_h[:3, 3] * UNITY_LEFT_TO_INTERNAL_RIGHT
    unity_moving_position = unity_init + case.unity_delta_m
    hand_moving_h = unity_pose_to_internal_vr_homo(unity_moving_position, case.internal_rotation)

    target_h = retarget_like_xarm_operator(robot_init_h, hand_init_h, hand_moving_h)
    actual_delta = target_h[:3, 3] - robot_init_h[:3, 3]
    expected_position = robot_init_h[:3, 3] + case.expected_robot_delta_m
    expected_r = expected_robot_rotation_delta(case.internal_rotation) @ robot_init_h[:3, :3]

    np.testing.assert_allclose(
        hand_moving_h[:3, 3] - hand_init_h[:3, 3],
        case.internal_delta_m,
        atol=ATOL_POS,
        err_msg=f"{side} {case.name}: Unity->internal VR delta mismatch",
    )
    np.testing.assert_allclose(
        actual_delta,
        case.expected_robot_delta_m,
        atol=ATOL_POS,
        err_msg=f"{side} {case.name}: robot translation mismatch",
    )
    np.testing.assert_allclose(
        target_h[:3, 3],
        expected_position,
        atol=ATOL_POS,
        err_msg=f"{side} {case.name}: robot target position mismatch",
    )
    np.testing.assert_allclose(
        target_h[:3, :3],
        expected_r,
        atol=ATOL_ROT,
        err_msg=f"{side} {case.name}: robot target orientation mismatch",
    )

    print(
        f"[PASS] {side:5s} {case.name:18s} "
        f"unity_delta={case.unity_delta_m} "
        f"internal_delta={case.internal_delta_m} "
        f"robot_delta={actual_delta} "
        f"target {format_pose(target_h)}"
    )


def main() -> None:
    r_robot_to_vr = H_R_V_SYSMO32[:3, :3]
    r_vr_to_robot = np.linalg.inv(r_robot_to_vr)
    assert np.isclose(np.linalg.det(r_robot_to_vr), 1.0)

    # MuJoCo home wrist poses from recent logs.
    robot_init = {
        "right": homo(np.array([0.206, -0.186, 0.475])),
        "left": homo(np.array([0.206, 0.186, 0.475])),
    }

    # Simulated first valid PICO/Unity wrist poses. They are converted once to
    # the internal right-handed VR frame, matching keypoint_transform.py.
    unity_hand_init = {
        "right": np.array([-0.10, 0.86, 0.03]),
        "left": np.array([-0.18, 0.96, 0.05]),
    }
    hand_init = {
        side: unity_pose_to_internal_vr_homo(pos, Rotation.identity())
        for side, pos in unity_hand_init.items()
    }

    cases = [
        PoseCase(
            name="unity_up",
            unity_delta_m=np.array([0.0, 0.10, 0.0]),
            internal_delta_m=np.array([0.0, 0.10, 0.0]),
            expected_robot_delta_m=r_vr_to_robot @ np.array([0.0, 0.10, 0.0]),
        ),
        PoseCase(
            name="unity_right",
            unity_delta_m=np.array([0.10, 0.0, 0.0]),
            internal_delta_m=np.array([0.10, 0.0, 0.0]),
            expected_robot_delta_m=r_vr_to_robot @ np.array([0.10, 0.0, 0.0]),
        ),
        PoseCase(
            name="unity_forward",
            unity_delta_m=np.array([0.0, 0.0, 0.10]),
            internal_delta_m=np.array([0.0, 0.0, -0.10]),
            expected_robot_delta_m=r_vr_to_robot @ np.array([0.0, 0.0, -0.10]),
        ),
        PoseCase(
            name="diagonal",
            unity_delta_m=np.array([0.04, 0.05, -0.06]),
            internal_delta_m=np.array([0.04, 0.05, 0.06]),
            expected_robot_delta_m=r_vr_to_robot @ np.array([0.04, 0.05, 0.06]),
        ),
        PoseCase(
            name="yaw_vr_y_20deg",
            unity_delta_m=np.array([0.02, 0.03, 0.00]),
            internal_delta_m=np.array([0.02, 0.03, 0.00]),
            expected_robot_delta_m=r_vr_to_robot @ np.array([0.02, 0.03, 0.00]),
            internal_rotation=Rotation.from_euler("y", 20.0, degrees=True),
        ),
    ]

    print("SYSMO-32 retarget self-test")
    print(f"H_R_V_SYSMO32 rotation det={np.linalg.det(r_robot_to_vr):.1f}")
    print("Expected direction checks:")
    print(f"  PICO +Y/up      -> Robot {r_vr_to_robot @ np.array([0.0, 1.0, 0.0])}")
    print(f"  PICO +X/right   -> Robot {r_vr_to_robot @ np.array([1.0, 0.0, 0.0])}")
    print(f"  PICO +Z/forward -> Robot {r_vr_to_robot @ np.array([0.0, 0.0, -1.0])}")
    print()

    for side in ("right", "left"):
        for case in cases:
            check_case(side, robot_init[side], hand_init[side], case)

    print("\nAll SYSMO-32 retarget checks passed.")


if __name__ == "__main__":
    main()
