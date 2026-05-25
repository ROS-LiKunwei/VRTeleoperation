import numpy as np

from beavr.teleop.components.operator.robots.sysmo32_operator import H_R_V_SYSMO32


def test_sysmo32_vr_motion_axes_map_to_robot_base_axes():
    vr_to_robot = np.linalg.inv(H_R_V_SYSMO32)[:3, :3]

    vr_right = np.array([1.0, 0.0, 0.0])
    vr_up = np.array([0.0, 1.0, 0.0])
    vr_back = np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(vr_to_robot @ vr_right, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vr_to_robot @ vr_up, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(vr_to_robot @ vr_back, [-1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(H_R_V_SYSMO32[:3, :3]), 1.0, atol=1e-6)
