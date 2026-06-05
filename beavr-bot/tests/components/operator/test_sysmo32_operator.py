import numpy as np
from scipy.spatial.transform import Rotation

from beavr.teleop.components.operator.robots.sysmo32_operator import H_R_V_SYSMO32
from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator


def test_sysmo32_vr_motion_axes_map_to_robot_base_axes():
    vr_to_robot = np.linalg.inv(H_R_V_SYSMO32)[:3, :3]

    vr_right = np.array([1.0, 0.0, 0.0])
    vr_up = np.array([0.0, 1.0, 0.0])
    vr_back = np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(vr_to_robot @ vr_right, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vr_to_robot @ vr_up, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(vr_to_robot @ vr_back, [-1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(H_R_V_SYSMO32[:3, :3]), 1.0, atol=1e-6)


def test_sysmo32_orientation_delta_uses_base_frame_not_mirrored_body_frame():
    op = XArmOperator.__new__(XArmOperator)
    op.rotation_delta_frame = "base"

    init_r = Rotation.from_euler("xyz", [25.0, -15.0, 10.0], degrees=True).as_matrix()
    base_delta = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    moving_r = base_delta @ init_r

    op.hand_init_h = np.eye(4)
    op.hand_init_h[:3, :3] = init_r
    op.hand_moving_h = np.eye(4)
    op.hand_moving_h[:3, :3] = moving_r

    body_relative_h = np.eye(4)
    body_relative_h[:3, :3] = init_r.T @ moving_r

    actual_delta = op._hand_rotation_delta_vr(body_relative_h)

    np.testing.assert_allclose(actual_delta, base_delta, atol=1e-6)
    assert not np.allclose(actual_delta, body_relative_h[:3, :3], atol=1e-6)
