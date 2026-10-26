import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import beavr.scripts.ahawam_fa_async_client as client
from beavr.lerobot.common.robot_devices.cameras.recording_fault import CameraFault
from beavr.scripts.ahawam_fa_async_client import (
    CAMERA_KEYS,
    FaCommandReceipt,
    FaObservationStream,
    FaRos,
    ObservationStreamError,
    PartialCommandPublishError,
    SentCommandHistory,
    limit_fa_target,
    recovery_takeover_reason,
    safe_measured_hold,
    send_authorized_command,
)
from beavr.scripts.ahawam_intervention import (
    ControlMode,
    OperatorIntervention,
)


def _complete_receipt(source: str, sent_at: float) -> FaCommandReceipt:
    return FaCommandReceipt(
        source=source,
        sent_at_monotonic_s=sent_at,
        arm_published=True,
        left_hand_published=True,
        right_hand_published=True,
        left_hand_changed=False,
        right_hand_changed=False,
    )


class _GatewayRos:
    def __init__(self) -> None:
        self.commands = []

    def publish(self, command, _args, *, source):
        vector = np.asarray(command, dtype=np.float32).copy()
        self.commands.append((source, vector))
        return _complete_receipt(source, time.monotonic())


def test_topology_guard_requires_sole_real_gateway_and_candidate_publishers() -> None:
    args = SimpleNamespace(
        enable_motion=True,
        enable_operator_intervention=True,
        arm_command_topic="/real/arm",
        left_hand_topic="/real/left",
        right_hand_topic="/real/right",
        human_arm_input_topic="/candidate/arm",
        human_left_hand_input_topic="/candidate/left",
        human_right_hand_input_topic="/candidate/right",
        human_rebaseline_request_topic="/candidate/rebaseline_request",
        human_rebaseline_ack_topic="/candidate/rebaseline_ack",
    )
    ros = object.__new__(FaRos)
    counts = {
        args.arm_command_topic: 2,
        args.left_hand_topic: 1,
        args.right_hand_topic: 1,
        args.human_arm_input_topic: 1,
        args.human_left_hand_input_topic: 1,
        args.human_right_hand_input_topic: 1,
        args.human_rebaseline_ack_topic: 1,
    }
    subscriber_counts = {args.human_rebaseline_request_topic: 1}
    ros.node = SimpleNamespace(
        count_publishers=lambda topic: counts[topic],
        count_subscribers=lambda topic: subscriber_counts[topic],
    )

    with pytest.raises(RuntimeError, match="ownership is not exclusive"):
        ros.validate_command_topology(args, timeout_s=0.0)

    counts[args.arm_command_topic] = 1
    counts[args.human_left_hand_input_topic] = 0
    with pytest.raises(RuntimeError, match="candidate/rebaseline endpoint is missing"):
        ros.validate_command_topology(args, timeout_s=0.0)

    counts[args.human_left_hand_input_topic] = 1
    ros.validate_command_topology(args, timeout_s=0.0)

    counts[args.right_hand_topic] = 2
    with pytest.raises(RuntimeError, match="ownership changed after startup"):
        ros.assert_exclusive_real_publishers(args)
    counts[args.right_hand_topic] = 1
    counts[args.human_arm_input_topic] = 2
    with pytest.raises(RuntimeError, match="human candidate ownership changed"):
        ros.assert_runtime_command_topology(args)


def test_command_gateway_rejects_model_after_takeover() -> None:
    ros = _GatewayRos()
    args = SimpleNamespace(enable_motion=True)
    operator = OperatorIntervention(enabled=True)
    history = SentCommandHistory()
    first = np.zeros(16, dtype=np.float32)

    sent = send_authorized_command(
        ros=ros,
        args=args,
        operator=operator,
        history=history,
        command=first,
        source="model",
    )
    assert sent is not None
    assert len(ros.commands) == 1

    operator.request_takeover("operator_pressed_i")
    assert operator.mode is ControlMode.SAFE_HOLD
    with pytest.raises(RuntimeError, match="model command is not authorized"):
        send_authorized_command(
            ros=ros,
            args=args,
            operator=operator,
            history=history,
            command=np.ones(16, dtype=np.float32),
            source="model",
        )
    assert len(ros.commands) == 1
    assert history.latest().sequence == 1


def test_repeated_takeover_key_is_ignored_after_safe_hold() -> None:
    class RepeatedTakeoverKey:
        @staticmethod
        def poll():
            return ["i"]

    operator = OperatorIntervention(enabled=True)
    operator.request_takeover("operator_pressed_i")

    client.poll_operator_controls(
        RepeatedTakeoverKey(), operator, chunk=3, step=4
    )

    assert operator.mode is ControlMode.SAFE_HOLD
    assert operator.snapshot().takeover_reason == "operator_pressed_i"


def test_sent_command_history_never_binds_a_future_command() -> None:
    history = SentCommandHistory()
    earlier = np.zeros(16, dtype=np.float32)
    future = np.ones(16, dtype=np.float32)
    history.mark_sent(earlier, "model", _complete_receipt("model", 10.0))
    history.mark_sent(future, "human", _complete_receipt("human", 20.0))

    assert history.snapshot_at(9.999) is None
    selected = history.snapshot_at(15.0)
    assert selected is not None
    assert selected.source == "model"
    assert selected.sent_at_monotonic_s == 10.0
    assert selected.command == pytest.approx(earlier)

    selected.command[0] = 123.0
    assert history.snapshot_at(15.0).command[0] == 0.0


def _callback_only_ros() -> FaRos:
    ros = object.__new__(FaRos)
    ros._human_condition = threading.Condition()
    ros._human_arm = None
    ros._human_left_gripper = 0.0
    ros._human_right_gripper = 0.0
    ros._human_sequence = 0
    ros._human_arm_sequence = 0
    ros._human_received_at = 0.0
    ros._human_error = None
    ros.left_gripper = 0.0
    ros.right_gripper = 0.0
    return ros


def _human_arm_message(offset: float = 0.0):
    return SimpleNamespace(
        left_arm_target_rad=(np.arange(7, dtype=np.float32) + offset).tolist(),
        right_arm_target_rad=(np.arange(7, dtype=np.float32) + 7 + offset).tolist(),
    )


def test_human_input_requires_new_post_handoff_arm_and_maps_20_21() -> None:
    ros = _callback_only_ros()
    ros._human_arm_callback(_human_arm_message())
    handoff_sequence = ros.human_command_sequence
    fallback_arm = np.linspace(-0.7, 0.7, 14, dtype=np.float32)

    with pytest.raises(TimeoutError, match="fresh post-handoff"):
        ros.wait_human_action(
            newer_than_sequence=handoff_sequence,
            timeout_s=0.001,
            max_age_s=1.0,
            fallback_arm=fallback_arm,
        )

    # A hand-only edge after takeover is forwarded while the arm stays at the
    # last authorized target; the pre-handoff VR arm candidate is never reused.
    ros._human_hand_callback("left", SimpleNamespace(data=20))
    hand_action, hand_metadata = ros.wait_human_action(
        newer_than_sequence=handoff_sequence,
        timeout_s=0.01,
        max_age_s=1.0,
        fallback_arm=fallback_arm,
    )
    assert hand_action[:14] == pytest.approx(fallback_arm)
    assert hand_action[14:].tolist() == [1.0, 0.0]

    ros._human_hand_callback("right", SimpleNamespace(data=21))
    ros._human_arm_callback(_human_arm_message(offset=0.5))
    action, metadata = ros.wait_human_action(
        newer_than_sequence=int(hand_metadata["sequence"]),
        timeout_s=0.01,
        max_age_s=1.0,
        fallback_arm=fallback_arm,
    )

    assert metadata["sequence"] == handoff_sequence + 3
    assert action.shape == (16,)
    assert action[:14] == pytest.approx(np.arange(14) + 0.5)
    assert action[14:].tolist() == [1.0, 0.0]


def test_handoff_seeds_model_gripper_latches_and_discards_old_human_hand() -> None:
    ros = _callback_only_ros()
    ros._human_arm_callback(_human_arm_message())
    ros._human_hand_callback("left", SimpleNamespace(data=21))
    ros._human_hand_callback("right", SimpleNamespace(data=21))

    # The model subsequently closed both hands.  Pre-handoff human cache must
    # not silently reopen them when the first human arm proposal arrives.
    ros.left_gripper = 1.0
    ros.right_gripper = 1.0
    handoff_sequence = ros.begin_human_handoff()
    ros._human_arm_callback(_human_arm_message(offset=0.25))
    action, metadata = ros.wait_human_action(
        newer_than_sequence=handoff_sequence,
        timeout_s=0.01,
        max_age_s=1.0,
        fallback_arm=np.zeros(14, dtype=np.float32),
    )
    assert action[14:].tolist() == [1.0, 1.0]

    # A real post-handoff hand-only edge is then accepted without changing arm.
    ros._human_hand_callback("right", SimpleNamespace(data=21))
    hand_only, _ = ros.wait_human_action(
        newer_than_sequence=int(metadata["sequence"]),
        timeout_s=0.01,
        max_age_s=1.0,
        fallback_arm=action[:14],
    )
    assert hand_only[:14] == pytest.approx(action[:14])
    assert hand_only[14:].tolist() == [1.0, 0.0]


def test_human_arm_callback_rejects_misaligned_6_plus_8_payload() -> None:
    ros = _callback_only_ros()
    ros._human_arm_callback(
        SimpleNamespace(
            left_arm_target_rad=np.zeros(6, dtype=np.float32),
            right_arm_target_rad=np.zeros(8, dtype=np.float32),
        )
    )

    with pytest.raises(RuntimeError, match="invalid FA human command proposal"):
        ros.wait_human_action(
            newer_than_sequence=0,
            timeout_s=0.001,
            max_age_s=1.0,
            fallback_arm=np.zeros(14, dtype=np.float32),
        )


def test_human_successful_send_is_recorded_before_terminal_label() -> None:
    class HumanRos(_GatewayRos):
        def rebaseline_human_control(self, _timeout_s):
            return 7

        def begin_human_handoff(self):
            return 0

        def wait_human_action(self, **_kwargs):
            target = np.linspace(-0.1, 0.1, 16, dtype=np.float32)
            target[14:] = [1.0, 0.0]
            return target, {"sequence": 1, "arm_sequence": 1, "age_ms": 0.0}

        def state(self, _timeout_s, _max_age_s):
            return np.zeros(16, dtype=np.float32)

    class LabelAfterOneSend:
        def __init__(self):
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return [] if self.poll_count == 1 else ["s"]

    class HealthyObservationStream:
        @staticmethod
        def raise_if_failed():
            return None

    class Recorder:
        def __init__(self):
            self.sources = []

        @staticmethod
        def raise_if_failed():
            return None

        def note_successful_command(self, source):
            self.sources.append(source)

    ros = HumanRos()
    recorder = Recorder()
    operator = OperatorIntervention(enabled=True)
    operator.request_takeover("test_handoff", automatic=True)
    args = SimpleNamespace(
        human_control_timeout_s=1.0,
        human_rebaseline_timeout_s=1.0,
        max_human_rebaseline_error_rad=0.2,
        max_human_command_age_ms=200.0,
        state_timeout_s=0.1,
        max_state_age_ms=200.0,
        max_joint_delta_rad=0.03,
        enable_motion=True,
    )

    with pytest.raises(client.EpisodeTerminated, match="operator_labeled_success"):
        client.run_human_control(
            ros=ros,
            args=args,
            operator=operator,
            listener=LabelAfterOneSend(),
            observation_stream=HealthyObservationStream(),
            history=SentCommandHistory(),
            initial_hold=np.zeros(16, dtype=np.float32),
            recorder=recorder,
        )

    assert recorder.sources == ["human"]
    assert [source for source, _ in ros.commands] == ["human"]


def test_first_human_arm_target_must_match_measured_rebaseline_pose() -> None:
    class MismatchedHumanRos(_GatewayRos):
        def rebaseline_human_control(self, _timeout_s):
            return 9

        def begin_human_handoff(self):
            return 10

        def wait_human_action(self, **_kwargs):
            target = np.ones(16, dtype=np.float32)
            return target, {"sequence": 11, "arm_sequence": 11, "age_ms": 0.0}

        def state(self, _timeout_s, _max_age_s):
            return np.zeros(16, dtype=np.float32)

    class NoKeys:
        @staticmethod
        def poll():
            return []

    class HealthyObservationStream:
        @staticmethod
        def raise_if_failed():
            return None

    ros = MismatchedHumanRos()
    operator = OperatorIntervention(enabled=True)
    operator.request_takeover("test_handoff", automatic=True)
    args = SimpleNamespace(
        human_control_timeout_s=1.0,
        human_rebaseline_timeout_s=1.0,
        max_human_rebaseline_error_rad=0.08,
        max_human_command_age_ms=200.0,
        state_timeout_s=0.1,
        max_state_age_ms=200.0,
        max_joint_delta_rad=0.03,
        enable_motion=True,
    )

    with pytest.raises(RuntimeError, match="post-rebaseline FA human target drifted"):
        client.run_human_control(
            ros=ros,
            args=args,
            operator=operator,
            listener=NoKeys(),
            observation_stream=HealthyObservationStream(),
            history=SentCommandHistory(),
            initial_hold=np.zeros(16, dtype=np.float32),
        )

    assert ros.commands == []


def test_slow_post_rebaseline_ramp_fails_before_human_authority() -> None:
    class SlowlyDriftingHumanRos(_GatewayRos):
        def __init__(self):
            super().__init__()
            self.sequence = 10

        def rebaseline_human_control(self, _timeout_s):
            return 19

        def begin_human_handoff(self):
            return 10

        def wait_human_action(self, **_kwargs):
            self.sequence += 1
            target = np.zeros(16, dtype=np.float32)
            target[:14] = 0.03 * (self.sequence - 10)
            return target, {
                "sequence": self.sequence,
                "arm_sequence": self.sequence,
                "age_ms": 0.0,
            }

        def state(self, _timeout_s, _max_age_s):
            return np.zeros(16, dtype=np.float32)

    class NoKeys:
        @staticmethod
        def poll():
            return []

    class HealthyObservationStream:
        @staticmethod
        def raise_if_failed():
            return None

    ros = SlowlyDriftingHumanRos()
    operator = OperatorIntervention(enabled=True)
    operator.request_takeover("test_handoff", automatic=True)
    args = SimpleNamespace(
        human_control_timeout_s=1.0,
        human_rebaseline_timeout_s=1.0,
        human_rebaseline_verify_s=1.0,
        max_human_rebaseline_error_rad=0.08,
        max_human_command_age_ms=200.0,
        state_timeout_s=0.1,
        max_state_age_ms=200.0,
        max_joint_delta_rad=0.03,
        enable_motion=True,
    )

    with pytest.raises(RuntimeError, match="post-rebaseline FA human target drifted"):
        client.run_human_control(
            ros=ros,
            args=args,
            operator=operator,
            listener=NoKeys(),
            observation_stream=HealthyObservationStream(),
            history=SentCommandHistory(),
            initial_hold=np.zeros(16, dtype=np.float32),
        )

    assert ros.commands == []
    assert operator.mode is ControlMode.SAFE_HOLD


class _Message:
    pass


class _Publisher:
    def __init__(self, failure=None) -> None:
        self.failure = failure
        self.messages = []

    def publish(self, message) -> None:
        if self.failure is not None:
            raise self.failure
        self.messages.append(message)


def test_publish_reports_exact_partial_receipt() -> None:
    ros = object.__new__(FaRos)
    ros.MinSnapTarget = _Message
    ros.Int32 = _Message
    ros.arm_pub = _Publisher()
    ros.left_hand_pub = _Publisher()
    ros.right_hand_pub = _Publisher(OSError("right hand link down"))
    ros.left_gripper = 0.0
    ros.right_gripper = 0.0
    args = SimpleNamespace(
        hz=30.0,
        max_velocity_rad_s=0.3,
        max_acceleration_rad_s2=2.0,
    )
    command = np.zeros(16, dtype=np.float32)
    command[14:] = 1.0

    with pytest.raises(PartialCommandPublishError) as caught:
        ros.publish(command, args, source="human")

    receipt = caught.value.receipt
    assert receipt.source == "human"
    assert receipt.arm_published is True
    assert receipt.left_hand_published is True
    assert receipt.right_hand_published is False
    assert receipt.left_hand_changed is True
    assert receipt.right_hand_changed is False
    assert receipt.complete is False
    assert ros.arm_pub.messages[0].left_arm_target_rad == pytest.approx([0.0] * 7)
    assert ros.left_hand_pub.messages[0].data == 20
    assert ros.left_gripper == 1.0
    assert ros.right_gripper == 0.0


class _MeasuredRos(_GatewayRos):
    def __init__(self, measured: np.ndarray) -> None:
        super().__init__()
        self.measured = measured

    def state_snapshot(self, timeout_s, max_age_s):
        assert timeout_s == 2.0
        assert max_age_s == 0.2
        return self.measured.copy(), {"sequence": 8}


def test_measured_hold_uses_fresh_feedback_and_hold_authority() -> None:
    measured = np.linspace(-0.7, 0.7, 16, dtype=np.float32)
    measured[14:] = [1.0, 0.0]
    ros = _MeasuredRos(measured)
    args = SimpleNamespace(
        enable_motion=True,
        state_timeout_s=2.0,
        max_state_age_ms=200.0,
    )
    operator = OperatorIntervention(enabled=True)
    operator.request_takeover("model_failed", automatic=True)
    history = SentCommandHistory()

    hold = safe_measured_hold(
        ros=ros,
        args=args,
        operator=operator,
        history=history,
    )

    assert hold == pytest.approx(measured)
    assert ros.commands[0][0] == "hold"
    assert ros.commands[0][1] == pytest.approx(measured)
    assert history.latest().source == "hold"
    assert history.latest().command == pytest.approx(measured)


def test_policy_recovery_only_triggers_in_active_mode() -> None:
    assert recovery_takeover_reason({}) is None
    assert recovery_takeover_reason(
        {
            "recovery_controller": {
                "operator_intervention_required": True,
                "mode": "diagnostic",
                "reason": "low confidence",
            }
        }
    ) is None
    assert recovery_takeover_reason(
        {
            "recovery_controller": {
                "operator_intervention_required": True,
                "mode": "active",
                "reason": "low confidence",
            }
        }
    ) == "low confidence"
    assert recovery_takeover_reason(
        {
            "recovery_controller": {
                "operator_intervention_required": False,
                "mode": "active",
            }
        }
    ) is None

    with pytest.raises(ValueError, match="invalid recovery mode"):
        recovery_takeover_reason(
            {
                "recovery_controller": {
                    "operator_intervention_required": True,
                    "mode": "unexpected",
                }
            }
        )


def test_limit_fa_target_clips_14d_arm_and_binarizes_grippers() -> None:
    measured = np.zeros(16, dtype=np.float32)
    target = np.concatenate(
        [
            np.linspace(-1.0, 1.0, 14, dtype=np.float32),
            np.array([0.799, 0.8], dtype=np.float32),
        ]
    )
    limited = limit_fa_target(
        target,
        measured,
        previous_command=None,
        max_joint_delta_rad=0.03,
    )

    assert limited.shape == (16,)
    assert float(np.max(np.abs(limited[:14]))) <= 0.030001
    assert limited[14:].tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="16D"):
        limit_fa_target(
            np.zeros(15), measured, None, max_joint_delta_rad=0.03
        )


def _stream_args():
    return SimpleNamespace(
        record_fps=100,
        max_camera_age_ms=300,
        max_camera_skew_ms=150.0,
        state_timeout_s=0.1,
        max_state_age_ms=200.0,
        max_image_state_skew_ms=150.0,
    )


class _StreamRos:
    def assert_runtime_command_topology(self, _args):
        return None

    def state_at_observation(
        self,
        observation_time,
        _timeout_s,
        _max_age_s,
        _max_skew_s,
        *,
        newer_than_sequence,
    ):
        sequence = newer_than_sequence + 1
        return np.zeros(16, dtype=np.float32), {
            "sequence": sequence,
            "sample_at_monotonic_s": observation_time,
        }


def _one_camera_sample(*_args, **_kwargs):
    captured_at = time.monotonic()
    images = {
        key: np.zeros((2, 3, 3), dtype=np.uint8) for key in CAMERA_KEYS
    }
    depths = {key: np.zeros((2, 3), dtype=np.uint16) for key in CAMERA_KEYS}
    metadata = {
        key: {
            "sequence": 1,
            "captured_at_monotonic_s": captured_at,
            "age_ms": 0.0,
        }
        for key in CAMERA_KEYS
    }
    return images, depths, metadata


@pytest.mark.parametrize("failure_origin", ["camera", "recorder"])
def test_observation_stream_camera_or_recorder_fault_is_fail_closed(
    monkeypatch, failure_origin
) -> None:
    if failure_origin == "camera":
        def fail_camera(*_args, **_kwargs):
            raise RuntimeError("camera capture failed")

        monkeypatch.setattr(client, "read_images", fail_camera)
        callback = None
        expected_cause = "camera capture failed"
    else:
        monkeypatch.setattr(client, "read_images", _one_camera_sample)

        def fail_recorder(_observation):
            raise RuntimeError("recorder submission failed")

        callback = fail_recorder
        expected_cause = "recorder submission failed"

    stream = FaObservationStream(
        cameras={},
        ros=_StreamRos(),
        args=_stream_args(),
        command_history=SentCommandHistory(),
        on_observation=callback,
    )
    stream.start()
    try:
        with pytest.raises(ObservationStreamError) as caught:
            stream.wait_next(newer_than_sequence=0, timeout_s=0.5)
        assert expected_cause in str(caught.value.__cause__)
        with pytest.raises(ObservationStreamError):
            stream.raise_if_failed()
    finally:
        stream.stop()


def test_observation_stream_retries_transient_stale_frame(monkeypatch) -> None:
    attempts = 0

    def transient_camera(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise CameraFault("head", "stale_rgbd", "temporary pause")
        return _one_camera_sample()

    monkeypatch.setattr(client, "read_images", transient_camera)
    args = _stream_args()
    args.observation_timeout_s = 0.5
    stream = FaObservationStream(
        cameras={},
        ros=_StreamRos(),
        args=args,
        command_history=SentCommandHistory(),
    )
    stream.start()
    try:
        observation = stream.wait_next(newer_than_sequence=0, timeout_s=0.5)
        assert observation.sequence >= 1
        assert attempts >= 3
        stream.raise_if_failed()
    finally:
        stream.stop()


def test_observation_stream_retries_unsynchronized_camera_set(monkeypatch) -> None:
    attempts = 0

    def transient_skew(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        images, depths, metadata = _one_camera_sample()
        if attempts == 1:
            metadata[CAMERA_KEYS[0]]["captured_at_monotonic_s"] -= 0.2
        return images, depths, metadata

    monkeypatch.setattr(client, "read_images", transient_skew)
    args = _stream_args()
    args.observation_timeout_s = 0.5
    stream = FaObservationStream(
        cameras={},
        ros=_StreamRos(),
        args=args,
        command_history=SentCommandHistory(),
    )
    stream.start()
    try:
        observation = stream.wait_next(newer_than_sequence=0, timeout_s=0.5)
        assert observation.sequence >= 1
        assert attempts >= 2
        stream.raise_if_failed()
    finally:
        stream.stop()


def test_observation_stream_retries_aligned_but_stale_camera_set(monkeypatch) -> None:
    attempts = 0

    def transient_stale_set(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        images, depths, metadata = _one_camera_sample()
        if attempts == 1:
            for value in metadata.values():
                value["captured_at_monotonic_s"] -= 1.0
        return images, depths, metadata

    monkeypatch.setattr(client, "read_images", transient_stale_set)
    args = _stream_args()
    args.observation_timeout_s = 0.5
    stream = FaObservationStream(
        cameras={},
        ros=_StreamRos(),
        args=args,
        command_history=SentCommandHistory(),
    )
    stream.start()
    try:
        observation = stream.wait_next(newer_than_sequence=0, timeout_s=0.5)
        assert observation.sequence >= 1
        assert attempts >= 2
        stream.raise_if_failed()
    finally:
        stream.stop()


def test_observation_stream_retries_camera_set_without_matching_joint_state(
    monkeypatch,
) -> None:
    camera_attempts = 0

    def camera_sample(*_args, **_kwargs):
        nonlocal camera_attempts
        camera_attempts += 1
        images, depths, metadata = _one_camera_sample()
        for value in metadata.values():
            value["sequence"] = camera_attempts
        return images, depths, metadata

    class TransientJointSkewRos(_StreamRos):
        def __init__(self):
            self.state_attempts = 0

        def state_at_observation(self, *args, **kwargs):
            self.state_attempts += 1
            if self.state_attempts == 1:
                raise TimeoutError("nearest joint sample skew=222.2 ms")
            return super().state_at_observation(*args, **kwargs)

    monkeypatch.setattr(client, "read_images", camera_sample)
    args = _stream_args()
    args.observation_timeout_s = 0.5
    ros = TransientJointSkewRos()
    stream = FaObservationStream(
        cameras={},
        ros=ros,
        args=args,
        command_history=SentCommandHistory(),
    )
    stream.start()
    try:
        observation = stream.wait_next(newer_than_sequence=0, timeout_s=0.5)
        assert observation.sequence >= 1
        assert ros.state_attempts >= 2
        assert camera_attempts >= 2
        stream.raise_if_failed()
    finally:
        stream.stop()


def test_policy_connection_failure_holds_then_allows_human_label(
    monkeypatch, tmp_path
) -> None:
    events = []

    class FakeListener:
        def __init__(self, enabled):
            assert enabled
            self.available = False
            self._polled = False

        def __enter__(self):
            self.available = True
            events.append("listener_started")
            return self

        def poll(self):
            if self._polled:
                return []
            self._polled = True
            return ["s"]

        def __exit__(self, *_args):
            self.available = False
            events.append("listener_stopped")

    class FakeRos(_GatewayRos):
        def __init__(self, _args):
            super().__init__()
            self.human_command_sequence = 0

        def validate_command_topology(self, _args, _timeout):
            events.append("topology_checked")

        def state_snapshot(self, _timeout, _max_age):
            return np.zeros(16, dtype=np.float32), {"sequence": 1}

        def begin_human_handoff(self):
            return self.human_command_sequence

        def rebaseline_human_control(self, _timeout_s):
            events.append("bimanual_rebaseline")
            return 1

        def close(self):
            events.append("ros_closed")

    class FakeCamera:
        def disconnect(self):
            events.append("camera_closed")

    class FakeStream:
        def __init__(self, _cameras, _ros, _args, _history, on_observation):
            assert on_observation is not None

        def start(self):
            events.append("stream_started")

        def raise_if_failed(self):
            return None

        def stop(self):
            events.append("stream_stopped")

    def fail_policy(*_args, **_kwargs):
        events.append("policy_connect_attempted")
        raise ConnectionRefusedError("brain offline")

    fake_ros_holder = {}

    def make_ros(args):
        fake_ros_holder["value"] = FakeRos(args)
        return fake_ros_holder["value"]

    monkeypatch.setattr(client, "TerminalKeyListener", FakeListener)
    monkeypatch.setattr(client, "FaRos", make_ros)
    monkeypatch.setattr(client, "FaObservationStream", FakeStream)
    monkeypatch.setattr(client, "PolicyConnection", fail_policy)
    monkeypatch.setattr(
        client,
        "open_cameras",
        lambda *_args, **_kwargs: {key: FakeCamera() for key in CAMERA_KEYS},
    )
    args = SimpleNamespace(
        enable_motion=True,
        confirm_motion="RUN",
        require_outcome_label=True,
        enable_operator_intervention=True,
        command_topology_timeout_s=0.1,
        host="127.0.0.1",
        port=10000,
        connect_timeout_s=0.01,
        socket_timeout_s=1.0,
        camera_settings=tmp_path / "camera.json",
        camera_open_retries=1,
        camera_retry_delay_s=0.0,
        record_dir=None,
        record_fps=10,
        prompt="task",
        arm_command_topic="/min_snap/target",
        left_hand_topic="/left_topic_to_hand",
        right_hand_topic="/right_topic_to_hand",
        human_arm_input_topic="/ahawam/fa/human/min_snap_target",
        human_left_hand_input_topic="/ahawam/fa/human/left_hand",
        human_right_hand_input_topic="/ahawam/fa/human/right_hand",
        state_timeout_s=0.1,
        max_state_age_ms=200.0,
        human_control_timeout_s=1.0,
        human_rebaseline_timeout_s=1.0,
        max_human_command_age_ms=200.0,
        max_joint_delta_rad=0.03,
        failure_reason=None,
    )

    client.run(args)

    assert events.index("stream_started") < events.index("policy_connect_attempted")
    # Initial arm, model-failure, and finalization boundaries each issue a
    # fresh measured hold.  The final retry is intentional: a previous hold
    # attempt may have only partially reached the three FA command topics.
    assert [source for source, _ in fake_ros_holder["value"].commands] == [
        "hold",
        "hold",
        "hold",
    ]
    for _, command in fake_ros_holder["value"].commands:
        np.testing.assert_array_equal(command, np.zeros(16, dtype=np.float32))
    assert events[-3:] == ["camera_closed", "ros_closed", "listener_stopped"]


@pytest.mark.parametrize(
    ("failure_point", "expected_reason"),
    [
        ("terminal_hold", "OSError: terminal hold failed"),
        ("ros_cleanup", "OSError: ros close failed"),
    ],
)
def test_terminal_or_cleanup_failure_commits_effective_error_reason(
    monkeypatch, tmp_path, failure_point, expected_reason
) -> None:
    import beavr.scripts.ahawam_fa_isolated_recorder as recorder_module

    finalized = {}

    class FakeListener:
        def __init__(self, enabled):
            assert enabled
            self.available = False

        def __enter__(self):
            self.available = True
            return self

        @staticmethod
        def poll():
            return ["s"]

        def __exit__(self, *_args):
            self.available = False

    class FakeRos(_GatewayRos):
        def __init__(self, _args):
            super().__init__()
            self.publish_count = 0

        @staticmethod
        def validate_command_topology(_args, _timeout):
            return None

        @staticmethod
        def state_snapshot(_timeout, _max_age):
            return np.zeros(16, dtype=np.float32), {"sequence": 1}

        def publish(self, command, args, *, source):
            self.publish_count += 1
            if failure_point == "terminal_hold" and self.publish_count == 1:
                raise OSError("terminal hold failed")
            return super().publish(command, args, source=source)

        def close(self):
            if failure_point == "ros_cleanup":
                raise OSError("ros close failed")

    class FakeCamera:
        @staticmethod
        def disconnect():
            return None

    class FakeStream:
        def __init__(self, _cameras, _ros, _args, _history, on_observation):
            assert on_observation is not None

        @staticmethod
        def start():
            return None

        @staticmethod
        def raise_if_failed():
            return None

        @staticmethod
        def stop():
            return None

    class FakePolicy:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def request(*_args, **_kwargs):
            raise AssertionError("fake executor must not execute policy request")

        @staticmethod
        def close():
            return None

    class PendingFuture:
        @staticmethod
        def done():
            return False

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def submit(*_args, **_kwargs):
            return PendingFuture()

        @staticmethod
        def shutdown(**_kwargs):
            return None

    class FakeRecorder:
        def __init__(self, *_args, **_kwargs):
            self.frame_count = 1

        @staticmethod
        def record_event(*_args, **_kwargs):
            return None

        @staticmethod
        def raise_if_failed():
            return None

        def finalize(self, outcome, **kwargs):
            finalized.update(outcome=outcome, **kwargs)
            return tmp_path / "episode"

        @staticmethod
        def abort(*_args, **_kwargs):
            raise AssertionError("healthy fake recorder must finalize")

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(client, "TerminalKeyListener", FakeListener)
    monkeypatch.setattr(client, "FaRos", FakeRos)
    monkeypatch.setattr(client, "FaObservationStream", FakeStream)
    monkeypatch.setattr(client, "PolicyConnection", FakePolicy)
    monkeypatch.setattr(client, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        recorder_module, "IsolatedFaEpisodeRecorder", FakeRecorder
    )
    monkeypatch.setattr(
        client,
        "open_cameras",
        lambda *_args, **_kwargs: {key: FakeCamera() for key in CAMERA_KEYS},
    )
    args = SimpleNamespace(
        enable_motion=True,
        confirm_motion="RUN",
        require_outcome_label=True,
        enable_operator_intervention=True,
        command_topology_timeout_s=0.0,
        host="127.0.0.1",
        port=10000,
        connect_timeout_s=0.01,
        socket_timeout_s=1.0,
        camera_settings=tmp_path / "camera.json",
        camera_open_retries=1,
        camera_retry_delay_s=0.0,
        record_dir=tmp_path / "recordings",
        record_fps=10,
        record_segment_sec=60.0,
        record_queue_size=8,
        record_min_free_gb=0.0,
        prompt="task",
        arm_command_topic="/min_snap/target",
        left_hand_topic="/left_topic_to_hand",
        right_hand_topic="/right_topic_to_hand",
        human_arm_input_topic="/ahawam/fa/human/min_snap_target",
        human_left_hand_input_topic="/ahawam/fa/human/left_hand",
        human_right_hand_input_topic="/ahawam/fa/human/right_hand",
        state_timeout_s=0.1,
        max_state_age_ms=200.0,
        failure_reason=None,
    )

    with pytest.raises(OSError, match=expected_reason.split(": ", 1)[1]):
        client.run(args)

    assert finalized["outcome"] == "error"
    assert finalized["reason"] == expected_reason
    summary = finalized["intervention_summary"]
    assert summary["reported_outcome"] == "success"
    assert summary["reported_reason"] == "operator_labeled_success"
    assert summary["effective_outcome"] == "error"
    assert summary["effective_reason"] == expected_reason
