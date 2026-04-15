import logging

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger(__name__)


class CompStateFilter:
    """
    A complementary filter for smoothing pose data (position and orientation).
    It uses linear interpolation for position and SLERP for orientation,
    with adaptive filtering based on velocity.
    """

    def __init__(
        self,
        init_state: np.ndarray,
        pos_ratio: float = 0.6,
        ori_ratio: float = 0.8,
        adaptive: bool = True,
    ):
        """
        Initializes the filter.

        Args:
            init_state: Initial pose [x, y, z, qx, qy, qz, qw].
            pos_ratio: Base filtering ratio for position (0-1). Higher means more smoothing.
            ori_ratio: Base filtering ratio for orientation (0-1). Higher means more smoothing.
            adaptive: Whether to adapt filtering ratios based on speed.
        """
        self.pos_state: np.ndarray = init_state[:3]
        self.ori_state: np.ndarray = init_state[3:7]
        self.pos_ratio: float = pos_ratio
        self.ori_ratio: float = ori_ratio
        self.adaptive: bool = adaptive
        self.prev_pos: np.ndarray = init_state[:3]
        self.velocity: np.ndarray = np.zeros(3)
        self.prev_quat: np.ndarray = init_state[3:7]

    def __call__(self, next_state: np.ndarray) -> np.ndarray:
        """
        Applies the filter to the next state measurement.

        Args:
            next_state: The new pose measurement [x, y, z, qx, qy, qz, qw].

        Returns:
            The filtered pose [x, y, z, qx, qy, qz, qw].
        """
        # Calculate velocity
        current_pos = next_state[:3]
        self.velocity = current_pos - self.prev_pos
        speed = np.linalg.norm(self.velocity)

        # Adaptive filtering for position
        actual_pos_ratio = self.pos_ratio
        if self.adaptive:
            # Increase filtering when movement is small (reduce jitter)
            # Decrease filtering when movement is large (reduce lag)
            threshold = 0.005  # Adjust based on your units
            if speed < threshold:
                actual_pos_ratio = min(0.95, self.pos_ratio + 0.1)  # More filtering for small movements
            else:
                # Less filtering for large movements, capped at a minimum
                actual_pos_ratio = max(0.7, self.pos_ratio - 0.1 * (speed / threshold))

        # Apply position filtering (Linear Interpolation - LERP)
        self.pos_state = self.pos_state * actual_pos_ratio + current_pos * (1 - actual_pos_ratio)

        # Always use stronger filtering for orientation (less affected by adaptive)
        # and ensure orientation ratio is at least as high as position ratio
        actual_ori_ratio = max(actual_pos_ratio, self.ori_ratio)

        # Apply orientation filtering (Spherical Linear Interpolation - SLERP)
        # Ensure quaternions are normalized and handle potential flips for stable SLERP
        current_quat = next_state[3:7]
        if np.dot(self.ori_state, current_quat) < 0:
            current_quat = -current_quat  # Flip if necessary

        # Normalize quaternions before SLERP
        self.ori_state /= np.linalg.norm(self.ori_state)
        current_quat /= np.linalg.norm(current_quat)

        try:
            ori_interp = Slerp([0, 1], Rotation.from_quat([self.ori_state, current_quat]))
            self.ori_state = ori_interp([1 - actual_ori_ratio])[0].as_quat()
            self.ori_state /= np.linalg.norm(self.ori_state)  # Re-normalize after SLERP
        except ValueError as e:
            # Handle potential Slerp errors (e.g., identical quaternions after normalization/flip)
            logger.warning(f"SLERP Warning: {e}. Using previous orientation state.")
            # Keep self.ori_state as is

        self.prev_pos = current_pos
        self.prev_quat = current_quat  # Store the (potentially flipped) current quat for next iteration's dot product check

        return np.concatenate([self.pos_state, self.ori_state])
