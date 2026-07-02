import time

import numpy as np

from beavr.teleop.components.detector.detector_types import InputFrame, SessionCommand
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator
from beavr.teleop.configs.constants import robots


def _identity4():
    h = np.eye(4, dtype=np.float64)
    return h


def test_operator_publishes_cartesian_target(bus):
    host = "127.0.0.1"
    transformed_keypoints_port = 5555
    endeff_publish_port = 7777
    endeff_subscribe_port = 6666  # not used by operator in this test

    # Minimal transforms (identity): robot base to VR base and hand tracking to VR base
    h_r_v = _identity4()
    h_t_v = _identity4()

    # Create operator (right hand)
    op = XArmOperator(
        operator_name="xarm7_right_operator",
        host=host,
        transformed_keypoints_port=transformed_keypoints_port,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=endeff_publish_port,
        endeff_subscribe_port=endeff_subscribe_port,
        moving_average_limit=1,
        h_r_v=h_r_v,
        h_t_v=h_t_v,
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=None,
        logging_config={"enabled": False},
        hand_side=robots.RIGHT,
    )

    op.robot_init_h = _identity4()
    op.robot_moving_h = _identity4()
    op.hand_init_h = _identity4()
    op.hand_init_t = np.zeros(3)
    op.is_first_frame = False

    # Provide a hand frame. InputFrame.frame_vectors should be 4 vectors: origin + 3 axes.
    # Here we simulate the right hand at origin with canonical axes; also provide some keypoints (not used for pose when frame_vectors present).
    origin = (0.0, 0.0, 0.0)
    x = (1.0, 0.0, 0.0)
    y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    frame_vectors = (origin, x, y, z)
    keypoints = [(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS

    # Publish hand frame for the right hand topic consumed by operator
    right_frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"

    def publish_hand_frame():
        bus.publish(
            host,
            transformed_keypoints_port,
            right_frame_topic,
            InputFrame(
                timestamp_s=time.time(),
                hand_side=robots.RIGHT,
                keypoints=keypoints,
                is_relative=False,
                frame_vectors=frame_vectors,
            ),
        )

    publish_hand_frame()
    op._apply_retargeted_angles()

    # Read what the operator published
    cmd = bus.recv_latest(endeff_publish_port, "endeff_coords")
    assert isinstance(cmd, CartesianTarget)
    # With identity transforms and zero motion,
    # target should match robot init pose => zero position and identity quat
    pos = np.asarray(cmd.position_m, dtype=np.float64)
    quat = np.asarray(cmd.orientation_xyzw, dtype=np.float64)
    np.testing.assert_allclose(pos, np.zeros(3), atol=1e-6)
    # Unit quaternion with positive w hemisphere
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)
    assert quat[3] >= 0.0


def test_operator_rejects_nan_hand_frame():
    op = XArmOperator(
        operator_name="xarm7_right_operator",
        host="127.0.0.1",
        transformed_keypoints_port=5555,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=7777,
        endeff_subscribe_port=6666,
        moving_average_limit=3,
        h_r_v=_identity4(),
        h_t_v=_identity4(),
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=None,
        logging_config={"enabled": False},
        hand_side=robots.RIGHT,
    )

    frame = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [0.0, np.nan, 0.0],
            [0.0, 0.0, np.nan],
        ],
        dtype=np.float64,
    )

    assert op._sanitize_hand_frame(frame) is None


def test_operator_keeps_hand_init_after_hand_frame_timeout():
    op = XArmOperator.__new__(XArmOperator)
    op.operator_name = "xarm7_right_operator"
    op.hand_frame_timeout_s = 0.01
    op._last_hand_data_time = time.time() - 1.0
    op._last_hand_frame_timestamp_s = time.time() - 1.0
    op.last_valid_hand_frame = np.zeros((4, 3), dtype=np.float64)
    op.latest_hand_command = 1
    op.hand_moving_h = np.eye(4)
    op.comp_filter = object()
    op.is_first_frame = False
    op.hand_init_h = np.eye(4)
    op.hand_init_t = np.zeros(3)
    op._ignore_hand_frames_before_s = 0.0
    op._last_hand_timeout_log_time = 0.0
    op._arm_transformed_keypoint_subscriber = type(
        "EmptyHandFrameSubscriber",
        (),
        {"recv_keypoints": lambda self: None},
    )()

    result = op._get_hand_frame()

    assert result is None
    assert op.is_first_frame is False
    np.testing.assert_allclose(op.hand_init_h, np.eye(4))
    np.testing.assert_allclose(op.hand_init_t, np.zeros(3))
    assert op.last_valid_hand_frame is not None
    assert op.latest_hand_command == 1
    assert op._last_hand_data_time > 0.0
    assert op._last_hand_frame_timestamp_s > 0.0
    assert op._ignore_hand_frames_before_s == 0.0


def test_operator_refreshes_hand_init_after_reset_completion(monkeypatch):
    op = XArmOperator.__new__(XArmOperator)
    op.operator_name = "xarm7_right_operator"
    op._post_reset_hand_rebaseline_after_s = 10.0
    op.comp_filter = object()
    op._last_reset_hand_wait_log_time = 0.0
    op._post_resume_stable_position_epsilon_m = 0.008
    op._post_resume_stable_orientation_epsilon_rad = 0.08
    op._post_resume_stable_dwell_s = 0.0
    op._log_reset_hand_wait = lambda: None
    op._turn_frame_to_homo_mat = XArmOperator._turn_frame_to_homo_mat.__get__(op, XArmOperator)
    op._refresh_robot_init_pose_after_hand_baseline = lambda: True

    frame = np.asarray(
        [
            [0.2, 0.3, 0.4],
            [1.2, 0.3, 0.4],
            [0.2, 1.3, 0.4],
            [0.2, 0.3, 1.4],
        ],
        dtype=np.float64,
    )
    calls = []

    def get_hand_frame(use_cache=True, min_timestamp_s=0.0):
        calls.append((use_cache, min_timestamp_s))
        return frame

    op._get_hand_frame = get_hand_frame

    assert op._rebaseline_hand_after_reset_if_needed()

    assert calls == [(False, 10.0)]
    np.testing.assert_allclose(op.hand_init_h[:3, 3], frame[0])
    np.testing.assert_allclose(op.hand_moving_h, op.hand_init_h)
    assert op.hand_init_t is not None
    assert op.comp_filter is None
    assert op._post_reset_hand_rebaseline_after_s is None


def test_operator_waits_settle_frames_before_post_reset_hand_init(monkeypatch):
    op = XArmOperator.__new__(XArmOperator)
    op.operator_name = "xarm7_right_operator"
    op._post_reset_hand_rebaseline_after_s = 10.0
    op._post_reset_hand_rebaseline_frames = []
    op._reset_hand_settle_frames = 3
    op._post_resume_stable_dwell_s = 0.0
    op.comp_filter = object()
    op.hand_init_h = None
    op.hand_moving_h = None
    op.hand_init_t = None
    op._last_reset_hand_wait_log_time = 0.0
    op._log_reset_hand_wait = lambda: None
    op._turn_frame_to_homo_mat = XArmOperator._turn_frame_to_homo_mat.__get__(op, XArmOperator)
    op._refresh_robot_init_pose_after_hand_baseline = lambda: True

    frames = [
        np.asarray(
            [
                [0.1, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [0.1, 1.0, 0.0],
                [0.1, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.101, 0.0, 0.0],
                [1.101, 0.0, 0.0],
                [0.101, 1.0, 0.0],
                [0.101, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.102, 0.0, 0.0],
                [1.102, 0.0, 0.0],
                [0.102, 1.0, 0.0],
                [0.102, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ]

    def get_hand_frame(use_cache=True, min_timestamp_s=0.0):
        assert not use_cache
        assert min_timestamp_s == 10.0
        return frames.pop(0)

    op._get_hand_frame = get_hand_frame

    assert op._rebaseline_hand_after_reset_if_needed()
    assert op.hand_init_h is None
    assert op._post_reset_hand_rebaseline_after_s == 10.0

    assert op._rebaseline_hand_after_reset_if_needed()
    assert op.hand_init_h is None
    assert op._post_reset_hand_rebaseline_after_s == 10.0

    assert op._rebaseline_hand_after_reset_if_needed()
    np.testing.assert_allclose(op.hand_init_h[:3, 3], np.asarray([0.102, 0.0, 0.0]))
    assert op._post_reset_hand_rebaseline_after_s is None
    assert op._post_reset_hand_rebaseline_frames == []


def test_operator_waits_stable_dwell_before_post_reset_hand_init(monkeypatch):
    op = XArmOperator.__new__(XArmOperator)
    op.operator_name = "xarm7_right_operator"
    op._post_reset_hand_rebaseline_after_s = 10.0
    op._post_reset_hand_rebaseline_frames = []
    op._post_reset_hand_rebaseline_first_frame_time_s = None
    op._reset_hand_settle_frames = 1
    op._post_resume_stable_position_epsilon_m = 0.008
    op._post_resume_stable_orientation_epsilon_rad = 0.08
    op._post_resume_stable_dwell_s = 1.0
    op.comp_filter = object()
    op.hand_init_h = None
    op.hand_moving_h = None
    op.hand_init_t = None
    op._last_reset_hand_wait_log_time = 0.0
    op._log_reset_hand_wait = lambda: None
    op._turn_frame_to_homo_mat = XArmOperator._turn_frame_to_homo_mat.__get__(op, XArmOperator)
    op._refresh_robot_init_pose_after_hand_baseline = lambda: True

    frame = np.asarray(
        [
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.1, 1.0, 0.0],
            [0.1, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    op._get_hand_frame = lambda use_cache=True, min_timestamp_s=0.0: frame

    assert op._rebaseline_hand_after_reset_if_needed()
    assert op.hand_init_h is None
    assert op._post_reset_hand_rebaseline_after_s == 10.0

    op._post_reset_hand_rebaseline_first_frame_time_s = time.time() - 2.0
    assert op._rebaseline_hand_after_reset_if_needed()
    np.testing.assert_allclose(op.hand_init_h[:3, 3], frame[0])
    assert op._post_reset_hand_rebaseline_after_s is None


def test_operator_keeps_waiting_when_post_reset_hand_frames_drift(monkeypatch):
    op = XArmOperator.__new__(XArmOperator)
    op.operator_name = "xarm7_right_operator"
    op._post_reset_hand_rebaseline_after_s = 10.0
    op._post_reset_hand_rebaseline_frames = []
    op._reset_hand_settle_frames = 3
    op._post_resume_stable_position_epsilon_m = 0.008
    op._post_resume_stable_orientation_epsilon_rad = 0.08
    op._post_resume_stable_dwell_s = 0.0
    op.comp_filter = object()
    op.hand_init_h = None
    op.hand_moving_h = None
    op.hand_init_t = None
    op._last_reset_hand_wait_log_time = 0.0
    op._log_reset_hand_wait = lambda: None
    op._turn_frame_to_homo_mat = XArmOperator._turn_frame_to_homo_mat.__get__(op, XArmOperator)
    op._refresh_robot_init_pose_after_hand_baseline = lambda: (_ for _ in ()).throw(AssertionError("must wait"))

    frames = [
        np.asarray(
            [
                [0.1, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [0.1, 1.0, 0.0],
                [0.1, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.2, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.2, 1.0, 0.0],
                [0.2, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.3, 0.0, 0.0],
                [1.3, 0.0, 0.0],
                [0.3, 1.0, 0.0],
                [0.3, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ]
    op._get_hand_frame = lambda use_cache=True, min_timestamp_s=0.0: frames.pop(0)

    assert op._rebaseline_hand_after_reset_if_needed()
    assert op._rebaseline_hand_after_reset_if_needed()
    assert op._rebaseline_hand_after_reset_if_needed()
    assert op.hand_init_h is None
    assert op._post_reset_hand_rebaseline_after_s == 10.0


def test_operator_uses_first_post_resume_hand_frame_for_hand_init(bus):
    host = "127.0.0.1"
    transformed_keypoints_port = 5555
    endeff_publish_port = 7777
    endeff_subscribe_port = 6666
    teleop_state_port = 8888
    hand_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"

    op = XArmOperator(
        operator_name="xarm7_right_operator",
        host=host,
        transformed_keypoints_port=transformed_keypoints_port,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=endeff_publish_port,
        endeff_subscribe_port=endeff_subscribe_port,
        moving_average_limit=1,
        h_r_v=_identity4(),
        h_t_v=_identity4(),
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=teleop_state_port,
        logging_config={"enabled": False},
        hand_side=robots.RIGHT,
        post_resume_stable_dwell_s=0.0,
    )

    stale_frame = (
        (9.0, 9.0, 9.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    post_resume_frame = (
        (0.2, 0.3, 0.4),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    keypoints = [(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS

    # 绿幕/resume 前即使 PICO 已在稳定发送，也不能提前建立 Hand init H。
    bus.publish(
        host,
        transformed_keypoints_port,
        hand_topic,
        InputFrame(
            timestamp_s=time.time(),
            hand_side=robots.RIGHT,
            keypoints=keypoints,
            is_relative=False,
            frame_vectors=stale_frame,
        ),
    )
    op._apply_retargeted_angles()
    assert op.hand_init_h is None

    resume_timestamp_s = time.time()
    bus.publish(
        host,
        teleop_state_port,
        "pause",
        SessionCommand(timestamp_s=resume_timestamp_s, command=robots.RESUME),
    )
    bus.publish(
        host,
        endeff_subscribe_port,
        "endeff_homo",
        CartesianState(
            timestamp_s=time.time(),
            h_matrix=tuple(map(tuple, np.eye(4).tolist())),
        ),
    )
    op._apply_retargeted_angles()
    assert op.robot_init_h is not None
    assert op.hand_init_h is None

    bus.publish(
        host,
        transformed_keypoints_port,
        hand_topic,
        InputFrame(
            timestamp_s=time.time(),
            hand_side=robots.RIGHT,
            keypoints=keypoints,
            is_relative=False,
            frame_vectors=post_resume_frame,
        ),
    )
    bus.publish(
        host,
        endeff_subscribe_port,
        "endeff_homo",
        CartesianState(
            timestamp_s=time.time(),
            h_matrix=tuple(map(tuple, np.eye(4).tolist())),
        ),
    )
    op._apply_retargeted_angles()

    np.testing.assert_allclose(op.hand_init_h[:3, 3], np.asarray(post_resume_frame[0]))
    assert not np.allclose(op.hand_init_h[:3, 3], np.asarray(stale_frame[0]))
