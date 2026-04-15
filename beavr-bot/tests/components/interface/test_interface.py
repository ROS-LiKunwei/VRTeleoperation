import numpy as np

from beavr.teleop.components.interface.interface_types import CartesianState


class MockController:
    """Simple mock controller for testing interface state management."""

    def __init__(self, ip: str):
        self._cartesian_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._joint_pos = np.array([0.0] * 7, dtype=np.float32)

    def get_cartesian_position(self) -> np.ndarray:
        """Return current cartesian position."""
        return self._cartesian_pos

    def get_joint_position(self) -> np.ndarray:
        """Return current joint position."""
        return self._joint_pos

    def move_cartesian(self, target_pos: np.ndarray) -> None:
        """Update cartesian position."""
        self._cartesian_pos = target_pos[:3]  # Only store position part


def test_cartesian_state_handling():
    """Test that interface properly handles cartesian state transformations."""
    # Create mock controller
    controller = MockController("10.0.0.1")

    # Set up test data
    target_pos = np.array([0.5, 0.0, 0.3], dtype=np.float32)  # x, y, z
    target_ori = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # quaternion xyzw

    # Create and verify cartesian state
    state = CartesianState(
        timestamp_s=0.0,
        position_m=tuple(target_pos),  # Use fixed time for testing
    )
    assert isinstance(state, CartesianState)
    assert len(state.position_m) == 3

    # Move to target
    controller.move_cartesian(np.concatenate([target_pos, target_ori]))

    # Verify cartesian state
    current_pos = controller.get_cartesian_position()
    np.testing.assert_allclose(current_pos, target_pos, atol=1e-6)

    # Verify joint state
    joint_pos = controller.get_joint_position()
    assert len(joint_pos) == 7  # 7-DOF arm
