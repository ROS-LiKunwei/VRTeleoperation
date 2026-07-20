import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from beavr.teleop.configs.constants import robots
from beavr.teleop.components.operator.robots.fa_axis_calibration import (
    BimanualWristSample,
    FaAxisCalibrationConfig,
    FaAxisCalibrationSession,
    FaAxisCalibrationState,
    compute_fa_axis_calibration,
)
from beavr.teleop.common.configs.loader import Laterality, load_robot_config
from beavr.teleop.components.operator.robots.fa_operator import FaOperator, H_R_V_FA


def _nominal_vr_to_robot():
    return np.linalg.inv(H_R_V_FA[:3, :3])


def _sample(timestamp_s, center, interhand=0.4):
    center = np.asarray(center, dtype=np.float64)
    lateral = np.asarray([interhand * 0.5, 0.0, 0.0], dtype=np.float64)
    return BimanualWristSample(
        timestamp_s=timestamp_s,
        left=center - lateral,
        right=center + lateral,
        left_timestamp_s=timestamp_s,
        right_timestamp_s=timestamp_s,
    )


def _skewed_sample(timestamp_s, center, skew_s, interhand=0.4):
    sample = _sample(timestamp_s, center, interhand)
    return BimanualWristSample(
        timestamp_s=sample.timestamp_s,
        left=sample.left,
        right=sample.right,
        left_timestamp_s=sample.left_timestamp_s,
        right_timestamp_s=sample.right_timestamp_s + skew_s,
    )


def _window(center, start=0.0, delta=0.1):
    return [_sample(start + i * delta, center) for i in range(5)]


def test_fa_axis_calibration_nominal_axes_return_existing_mapping():
    config = FaAxisCalibrationConfig()
    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        origin_samples=_window([0.0, 0.0, 0.0]),
        forward_samples=_window([0.0, 0.0, -0.2], start=1.0),
        left_samples=_window([-0.2, 0.0, 0.0], start=2.0),
        config=config,
    )

    assert reason is None
    np.testing.assert_allclose(result.r_vr_to_robot, _nominal_vr_to_robot(), atol=1e-6)
    np.testing.assert_allclose(result.r_vr_to_robot @ [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.r_vr_to_robot @ [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-6)
    assert result.rotation_determinant == pytest.approx(1.0)


def test_fa_axis_calibration_compensates_known_vr_rotation():
    config = FaAxisCalibrationConfig()
    nominal = _nominal_vr_to_robot()
    nominal_axes_vr = nominal.T
    vr_bias = Rotation.from_euler("zyx", [25.0, -20.0, 15.0], degrees=True).as_matrix()
    measured_axes_vr = vr_bias @ nominal_axes_vr
    forward_vr = measured_axes_vr[:, 0]
    left_vr = measured_axes_vr[:, 1]

    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=nominal,
        origin_samples=_window([0.0, 0.0, 0.0]),
        forward_samples=_window(0.2 * forward_vr, start=1.0),
        left_samples=_window(0.2 * left_vr, start=2.0),
        config=config,
    )

    assert reason is None
    np.testing.assert_allclose(result.r_vr_to_robot @ forward_vr, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.r_vr_to_robot @ left_vr, [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.r_vr_to_robot.T @ result.r_vr_to_robot, np.eye(3), atol=1e-6)
    assert np.linalg.det(result.r_vr_to_robot) == pytest.approx(1.0)


def test_fa_axis_calibration_rejects_short_motion():
    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        origin_samples=_window([0.0, 0.0, 0.0]),
        forward_samples=_window([0.0, 0.0, -0.02], start=1.0),
        left_samples=_window([-0.2, 0.0, 0.0], start=2.0),
        config=FaAxisCalibrationConfig(),
    )

    assert result is None
    assert reason == "FORWARD_MOTION_TOO_SHORT"


def test_fa_axis_calibration_rejects_collinear_axes():
    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        origin_samples=_window([0.0, 0.0, 0.0]),
        forward_samples=_window([0.0, 0.0, -0.2], start=1.0),
        left_samples=_window([0.0, 0.0, -0.25], start=2.0),
        config=FaAxisCalibrationConfig(),
    )

    assert result is None
    assert reason == "AXES_NEARLY_COLLINEAR"


def test_fa_axis_calibration_tolerates_asymmetric_same_direction_hand_motion():
    origin_samples = _window([0.0, 0.0, 0.0])
    forward_samples = [
        BimanualWristSample(
            timestamp_s=1.0 + i * 0.1,
            left=np.asarray([-0.2, 0.0, -0.16], dtype=np.float64),
            right=np.asarray([0.2, 0.0, -0.24], dtype=np.float64),
            left_timestamp_s=1.0 + i * 0.1,
            right_timestamp_s=1.0 + i * 0.1,
        )
        for i in range(5)
    ]
    left_samples = [
        BimanualWristSample(
            timestamp_s=2.0 + i * 0.1,
            left=np.asarray([-0.32, 0.0, 0.0], dtype=np.float64),
            right=np.asarray([0.02, 0.0, 0.0], dtype=np.float64),
            left_timestamp_s=2.0 + i * 0.1,
            right_timestamp_s=2.0 + i * 0.1,
        )
        for i in range(5)
    ]

    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        origin_samples=origin_samples,
        forward_samples=forward_samples,
        left_samples=left_samples,
        config=FaAxisCalibrationConfig(),
    )

    assert reason is None
    assert result is not None


def test_fa_axis_calibration_rejects_hand_direction_error_over_default_limit():
    origin_samples = _window([0.0, 0.0, 0.0])
    angle_rad = np.deg2rad(25.0)
    lateral_delta = 0.2 * np.sin(angle_rad)
    forward_delta = -0.2 * np.cos(angle_rad)
    forward_samples = [
        BimanualWristSample(
            timestamp_s=1.0 + i * 0.1,
            left=np.asarray([-0.2 - lateral_delta, 0.0, forward_delta]),
            right=np.asarray([0.2 + lateral_delta, 0.0, forward_delta]),
            left_timestamp_s=1.0 + i * 0.1,
            right_timestamp_s=1.0 + i * 0.1,
        )
        for i in range(5)
    ]

    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        origin_samples=origin_samples,
        forward_samples=forward_samples,
        left_samples=_window([-0.2, 0.0, 0.0], start=2.0),
        config=FaAxisCalibrationConfig(),
    )

    assert result is None
    assert reason == "FORWARD_HANDS_MISMATCH"


def test_fa_axis_calibration_rejects_hand_distance_ratio_over_default_limit():
    origin_samples = _window([0.0, 0.0, 0.0])
    forward_samples = [
        BimanualWristSample(
            timestamp_s=1.0 + i * 0.1,
            left=np.asarray([-0.2, 0.0, -0.1]),
            right=np.asarray([0.2, 0.0, -0.2]),
            left_timestamp_s=1.0 + i * 0.1,
            right_timestamp_s=1.0 + i * 0.1,
        )
        for i in range(5)
    ]

    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        origin_samples=origin_samples,
        forward_samples=forward_samples,
        left_samples=_window([-0.2, 0.0, 0.0], start=2.0),
        config=FaAxisCalibrationConfig(),
    )

    assert result is None
    assert reason == "FORWARD_HANDS_MISMATCH"


def test_fa_axis_calibration_rejects_opposite_hand_motion():
    origin_samples = _window([0.0, 0.0, 0.0])
    forward_samples = _window([0.0, 0.0, -0.2], start=1.0)
    left_samples = [
            BimanualWristSample(
                timestamp_s=2.0 + i * 0.1,
                left=np.asarray([-0.6, 0.0, 0.0], dtype=np.float64),
                right=np.asarray([0.3, 0.0, 0.0], dtype=np.float64),
                left_timestamp_s=2.0 + i * 0.1,
                right_timestamp_s=2.0 + i * 0.1,
            )
        for i in range(5)
    ]

    result, reason = compute_fa_axis_calibration(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        origin_samples=origin_samples,
        forward_samples=forward_samples,
        left_samples=left_samples,
        config=FaAxisCalibrationConfig(),
    )

    assert result is None
    assert reason == "LEFT_HANDS_MISMATCH"


def test_fa_axis_calibration_session_reaches_ready_and_detects_origin_jump():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
            tracking_origin_jump_translation_m=0.15,
            tracking_origin_jump_confirm_frames=2,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.CAPTURING_FORWARD

    for sample in [_sample(0.4, [0, 0, -0.2]), _sample(0.5, [0, 0, -0.2]), _sample(0.7, [0, 0, -0.2])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_AFTER_FORWARD

    for sample in [_sample(0.8, [0, 0, 0]), _sample(1.0, [0, 0, 0])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.CAPTURING_LEFT

    for sample in [_sample(1.1, [-0.2, 0, 0]), _sample(1.2, [-0.2, 0, 0]), _sample(1.4, [-0.2, 0, 0])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY
    assert not session.ready

    for sample in [_sample(1.5, [0.035, 0, 0]), _sample(1.8, [0.035, 0, 0])]:
        session.update(sample)
    assert session.ready

    session.update(_sample(1.9, [0.1, 0.2, 0.0]))
    assert session.ready
    session.update(_sample(2.0, [0.3, 0.4, 0.0]))
    assert session.state == FaAxisCalibrationState.INVALIDATED
    assert session.failure_reason == "TRACKING_ORIGIN_JUMP"


def test_fa_axis_calibration_offline_full_prompt_flow_reaches_ready():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
        ),
    )
    observed_states = []

    def update(sample):
        session.update(sample)
        state = session.consume_state_change()
        if state is not None:
            observed_states.append(state)

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        update(sample)
    for sample in [_sample(0.4, [0, 0, -0.07]), _sample(0.5, [0, 0, -0.07]), _sample(0.7, [0, 0, -0.07])]:
        update(sample)
    for sample in [_sample(0.8, [0, 0, 0]), _sample(1.0, [0, 0, 0])]:
        update(sample)
    for sample in [_sample(1.1, [-0.07, 0, 0]), _sample(1.2, [-0.07, 0, 0]), _sample(1.4, [-0.07, 0, 0])]:
        update(sample)
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY
    assert not session.ready
    for sample in [_sample(1.5, [0.035, 0, 0]), _sample(1.8, [0.035, 0, 0])]:
        update(sample)

    assert session.ready
    assert FaAxisCalibrationState.CAPTURING_ORIGIN in observed_states
    assert FaAxisCalibrationState.CAPTURING_FORWARD in observed_states
    assert FaAxisCalibrationState.WAITING_RETURN_AFTER_FORWARD in observed_states
    assert FaAxisCalibrationState.CAPTURING_LEFT in observed_states
    assert FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY in observed_states
    assert observed_states[-1] == FaAxisCalibrationState.READY


def test_fa_axis_calibration_requires_relaxed_origin_return_before_ready():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)
    for sample in [_sample(0.4, [0, 0, -0.07]), _sample(0.5, [0, 0, -0.07]), _sample(0.7, [0, 0, -0.07])]:
        session.update(sample)
    for sample in [_sample(0.8, [0, 0, 0]), _sample(1.0, [0, 0, 0])]:
        session.update(sample)
    for sample in [_sample(1.1, [-0.07, 0, 0]), _sample(1.2, [-0.07, 0, 0]), _sample(1.4, [-0.07, 0, 0])]:
        session.update(sample)

    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY
    assert session.result is not None
    assert not session.ready

    session.update(_sample(1.5, [-0.07, 0, 0]))
    session.update(_sample(1.8, [-0.07, 0, 0]))
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY
    assert not session.ready

    # 双手中点虽然回到原点，但任一手腕超过 4cm 仍不能开始遥操作。
    session.update(_sample(1.9, [0, 0, 0], interhand=0.5))
    session.update(_sample(2.2, [0, 0, 0], interhand=0.5))
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY
    assert not session.ready

    session.update(_sample(2.3, [0.035, 0, 0]))
    session.update(_sample(2.6, [0.035, 0, 0]))
    assert session.ready


def test_fa_axis_calibration_collinear_left_attempt_retries_left_after_return():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)
    for sample in [_sample(0.4, [0, 0, -0.07]), _sample(0.5, [0, 0, -0.07]), _sample(0.7, [0, 0, -0.07])]:
        session.update(sample)
    for sample in [_sample(0.8, [0, 0, 0]), _sample(1.0, [0, 0, 0])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.CAPTURING_LEFT

    for sample in [_sample(1.1, [0, 0, -0.08]), _sample(1.2, [0, 0, -0.08]), _sample(1.4, [0, 0, -0.08])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_LEFT_RETRY
    assert session.failure_reason == "AXES_NEARLY_COLLINEAR"

    for sample in [_sample(1.5, [0, 0, 0]), _sample(1.7, [0, 0, 0])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.CAPTURING_LEFT
    assert session.failure_reason is None

    for sample in [_sample(1.8, [-0.07, 0, 0]), _sample(1.9, [-0.07, 0, 0]), _sample(2.1, [-0.07, 0, 0])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_READY
    assert not session.ready
    for sample in [_sample(2.2, [0.035, 0, 0]), _sample(2.5, [0.035, 0, 0])]:
        session.update(sample)
    assert session.ready


def test_fa_axis_calibration_forward_hands_mismatch_retries_after_return():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)

    angle_rad = np.deg2rad(25.0)
    lateral_delta = 0.2 * np.sin(angle_rad)
    forward_delta = -0.2 * np.cos(angle_rad)
    for timestamp_s in (0.4, 0.5, 0.7):
        session.update(
            BimanualWristSample(
                timestamp_s=timestamp_s,
                left=np.asarray([-0.2 - lateral_delta, 0.0, forward_delta]),
                right=np.asarray([0.2 + lateral_delta, 0.0, forward_delta]),
                left_timestamp_s=timestamp_s,
                right_timestamp_s=timestamp_s,
            )
        )

    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_FORWARD_RETRY
    assert session.failure_reason == "FORWARD_HANDS_MISMATCH"

    session.update(_sample(0.8, [0, 0, 0]))
    session.update(_sample(1.0, [0, 0, 0]))
    assert session.state == FaAxisCalibrationState.CAPTURING_FORWARD
    assert session.failure_reason is None


def test_fa_axis_calibration_left_hands_mismatch_retries_after_return():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)
    for sample in [_sample(0.4, [0, 0, -0.2]), _sample(0.5, [0, 0, -0.2]), _sample(0.7, [0, 0, -0.2])]:
        session.update(sample)
    for sample in [_sample(0.8, [0, 0, 0]), _sample(1.0, [0, 0, 0])]:
        session.update(sample)

    for timestamp_s in (1.1, 1.2, 1.4):
        session.update(
            BimanualWristSample(
                timestamp_s=timestamp_s,
                left=np.asarray([-0.3, 0.0, 0.0]),
                right=np.asarray([0.0, 0.0, 0.0]),
                left_timestamp_s=timestamp_s,
                right_timestamp_s=timestamp_s,
            )
        )

    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_LEFT_RETRY
    assert session.failure_reason == "LEFT_HANDS_MISMATCH"

    session.update(_sample(1.5, [0, 0, 0]))
    session.update(_sample(1.7, [0, 0, 0]))
    assert session.state == FaAxisCalibrationState.CAPTURING_LEFT
    assert session.failure_reason is None


def test_fa_axis_calibration_retries_short_forward_after_return_to_origin():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
            calibration_stable_position_epsilon_m=0.02,
            calibration_min_motion_distance_m=0.12,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.CAPTURING_FORWARD

    for sample in [
        _sample(0.4, [0, 0, -0.06]),
        _sample(0.5, [0, 0, -0.06]),
        _sample(0.7, [0, 0, -0.06]),
    ]:
        session.update(sample)
    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_FORWARD_RETRY
    assert session.failure_reason == "FORWARD_MOTION_TOO_SHORT"

    session.update(_sample(0.8, [0, 0, 0]))
    session.update(_sample(1.0, [0, 0, 0]))
    assert session.state == FaAxisCalibrationState.CAPTURING_FORWARD
    assert session.failure_reason is None


def test_fa_axis_calibration_default_accepts_six_centimeter_forward_motion():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)

    for sample in [_sample(0.4, [0, 0, -0.07]), _sample(0.5, [0, 0, -0.07]), _sample(0.7, [0, 0, -0.07])]:
        session.update(sample)

    assert session.state == FaAxisCalibrationState.WAITING_RETURN_AFTER_FORWARD
    assert session.failure_reason is None


def test_fa_axis_calibration_short_attempt_does_not_wait_silently():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(
            calibration_sample_duration_s=0.2,
            calibration_stable_dwell_s=0.1,
        ),
    )

    for sample in [_sample(0.0, [0, 0, 0]), _sample(0.1, [0, 0, 0]), _sample(0.3, [0, 0, 0])]:
        session.update(sample)

    for sample in [_sample(0.4, [0, 0, -0.04]), _sample(0.5, [0, 0, -0.04]), _sample(0.7, [0, 0, -0.04])]:
        session.update(sample)

    assert session.state == FaAxisCalibrationState.WAITING_RETURN_BEFORE_FORWARD_RETRY
    assert session.failure_reason == "FORWARD_MOTION_TOO_SHORT"


def test_fa_axis_calibration_session_does_not_fail_on_timestamp_skew():
    session = FaAxisCalibrationSession(
        r_vr_to_robot_nominal=_nominal_vr_to_robot(),
        config=FaAxisCalibrationConfig(calibration_max_timestamp_skew_s=0.05),
    )

    session.update(_skewed_sample(0.0, [0, 0, 0], skew_s=0.2))

    assert session.state == FaAxisCalibrationState.CAPTURING_ORIGIN
    assert session.failure_reason is None


def test_fa_operator_builds_calibration_sample_when_timestamps_are_skewed():
    operator = FaOperator.__new__(FaOperator)
    operator._fa_calibration_right_subscriber = None
    operator._fa_calibration_left_subscriber = None
    operator._fa_latest_calibration_frames = {
        robots.RIGHT: (np.asarray([0.2, 0.0, 0.0], dtype=np.float64), 100.3),
        robots.LEFT: (np.asarray([-0.2, 0.0, 0.0], dtype=np.float64), 100.0),
    }
    operator._fa_axis_calibration_config = FaAxisCalibrationConfig(
        calibration_max_timestamp_skew_s=0.05,
        calibration_max_frame_age_s=10_000_000_000.0,
    )
    operator._fa_last_calibration_sync_wait_log_time = float("inf")

    sample = operator._get_fa_calibration_sample()

    assert sample is not None
    assert sample.timestamp_s == pytest.approx(100.3)
    assert operator._fa_last_calibration_sample_unavailable_reason is None


def test_fa_config_enables_axis_calibration_only_for_bimanual():
    bimanual_cfg = load_robot_config("fa", Laterality.BIMANUAL, True, control_backend="mujoco")
    assert len(bimanual_cfg.operators) == 2
    assert all(operator.enable_vr_axis_calibration for operator in bimanual_cfg.operators)
    assert all(operator.require_calibration_each_enable for operator in bimanual_cfg.operators)
    assert all(operator.calibration_max_left_right_direction_error_deg == 20.0 for operator in bimanual_cfg.operators)
    assert all(operator.calibration_max_left_right_distance_ratio_error == 0.35 for operator in bimanual_cfg.operators)
    assert all(operator.calibration_ready_return_position_epsilon_m == 0.04 for operator in bimanual_cfg.operators)
    assert all(operator.calibration_ready_return_dwell_s == 0.20 for operator in bimanual_cfg.operators)
    audio_owners = [operator for operator in bimanual_cfg.operators if operator.calibration_audio_enabled]
    assert len(audio_owners) == 1
    assert audio_owners[0].hand_side == "right"

    right_cfg = load_robot_config("fa", Laterality.RIGHT, True, control_backend="mujoco")
    assert len(right_cfg.operators) == 1
    assert right_cfg.operators[0].hand_side == "right"
    assert not right_cfg.operators[0].enable_vr_axis_calibration
