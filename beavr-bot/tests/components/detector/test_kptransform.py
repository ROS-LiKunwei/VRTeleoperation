import numpy as np

from beavr.teleop.components.detector.vr.keypoint_transform import (
    TransformHandPositionCoords,
)
from beavr.teleop.configs.constants import robots


def _make_keypoints_for_right_hand_identity_frame():
    # 24 keypoints, wrist at index 0
    pts = np.zeros((robots.OCULUS_NUM_KEYPOINTS, 3), dtype=np.float64)
    knuckles = robots.OCULUS_JOINTS["knuckles"]  # [6, 9, 12, 16]
    # Place three landmark directions from wrist along axes
    pts[knuckles[0]] = np.array([1.0, 0.0, 0.0])  # toward +X
    pts[knuckles[1]] = np.array([0.0, 1.0, 0.0])  # toward +Y
    pts[knuckles[-1]] = np.array([0.0, 0.0, 1.0])  # toward +Z
    return [tuple(map(float, p)) for p in pts.tolist()]


def test_transform_coordinate_frame():
    """Test that hand keypoint transform produces a valid coordinate frame."""
    # Create raw keypoints representing identity frame
    raw_keypoints = _make_keypoints_for_right_hand_identity_frame()

    # Create transform instance (no need for network params since we call directly)
    tf = TransformHandPositionCoords(
        host="",  # Not used in direct calls
        keypoint_sub_port=0,  # Not used in direct calls
        keypoint_transform_pub_port=0,  # Not used in direct calls
        hand_side=robots.RIGHT,
        moving_average_limit=1,  # No smoothing for this test
    )

    # Convert keypoints to numpy array for transform
    hand_coords = np.array(raw_keypoints, dtype=np.float64)

    # Transform the keypoints
    transformed_hand_coords, hand_dir_frame = tf.transform_keypoints(hand_coords)

    # Extract frame vectors
    origin = hand_dir_frame[0]  # Wrist position
    x_vec = hand_dir_frame[1]  # Cross product of palm direction and normal
    y_vec = hand_dir_frame[2]  # Palm normal
    z_vec = hand_dir_frame[3]  # Palm direction

    # Verify frame properties:

    # 1. Origin should be at wrist position
    np.testing.assert_allclose(origin, hand_coords[0], atol=1e-6)

    # 2. Frame vectors should be orthonormal
    # Check orthogonality (dot products should be 0)
    np.testing.assert_allclose(np.dot(x_vec, y_vec), 0, atol=1e-6)
    np.testing.assert_allclose(np.dot(y_vec, z_vec), 0, atol=1e-6)
    np.testing.assert_allclose(np.dot(z_vec, x_vec), 0, atol=1e-6)

    # Check normalization (vectors should have unit length)
    np.testing.assert_allclose(np.linalg.norm(x_vec), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y_vec), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(z_vec), 1.0, atol=1e-6)

    # 3. Frame should follow right-hand rule
    expected_z = np.cross(x_vec, y_vec)
    np.testing.assert_allclose(z_vec, expected_z, atol=1e-6)

    # 4. Transformed keypoints should be relative to wrist
    # First point (wrist) should be at origin
    np.testing.assert_allclose(transformed_hand_coords[0], np.zeros(3), atol=1e-6)

    # Key landmarks should maintain relative distances
    raw_distances = np.linalg.norm(hand_coords[1:] - hand_coords[0], axis=1)
    transformed_distances = np.linalg.norm(transformed_hand_coords[1:] - transformed_hand_coords[0], axis=1)
    np.testing.assert_allclose(transformed_distances, raw_distances, atol=1e-6)
