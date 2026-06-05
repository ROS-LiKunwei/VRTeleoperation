import logging
from types import SimpleNamespace

import numpy as np
import pytest

from beavr.teleop.common.configs.loader import Laterality, load_robot_config
from beavr.teleop.components.interface.robots.sysmo32_command import (
    SYSMO32_COMMAND_LENGTH,
    SYSMO32_HAND_ACTION_GRASP,
    SYSMO32_HAND_ACTION_RELEASE,
    SYSMO32_LEFT_JOINT_NAMES,
    SYSMO32_RIGHT_JOINT_NAMES,
    Sysmo32ArmCommand,
    Sysmo32ArmSafetyConfig,
    Sysmo32CommandBuilder,
    Sysmo32CommandLimiter,
    Sysmo32HandGestureMapper,
    Sysmo32JointStateCache,
)
from beavr.teleop.components.interface.robots.sysmo32_real_control import (
    SYSMO32_LEFT_HAND_ACTION_TOPIC,
    SYSMO32_RIGHT_HAND_ACTION_TOPIC,
    Sysmo32RealControl,
    Sysmo32RealControlConfig,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.simulation.sysmo32_mujoco_command_sim import Sysmo32MujocoCommandMirror
from beavr.teleop.configs.constants import ports, robots


def _hand_keypoints(distance: float) -> np.ndarray:
    keypoints = np.zeros((26, 3), dtype=np.float64)
    keypoints[5] = np.array([0.0, 0.0, 0.0])
    keypoints[10] = np.array([distance, 0.0, 0.0])
    return keypoints


def test_sysmo32_arm_command_contract_defaults():
    command = Sysmo32CommandBuilder().build([0.1] * 6, [-0.1] * 6, timestamp_s=1.0)

    assert len(command.values) == SYSMO32_COMMAND_LENGTH
    assert command.left_arm == tuple([0.1] * 6)
    assert command.right_arm == tuple([-0.1] * 6)
    assert command.speed_mode == 0.0
    assert command.reserved == (0.0, 0.0, 0.0, 0.0)
    assert command.neck_joint == 0.0


def test_sysmo32_arm_command_rejects_invalid_length_and_nan():
    with pytest.raises(ValueError, match="length"):
        Sysmo32ArmCommand(timestamp_s=1.0, values=tuple([0.0] * 17))

    with pytest.raises(ValueError, match="NaN/Inf"):
        Sysmo32CommandBuilder().build([np.nan] * 6, [0.0] * 6)


def test_sysmo32_speed_mode_4_is_not_allowed_as_default():
    with pytest.raises(ValueError, match="speed_mode=4.0"):
        Sysmo32ArmSafetyConfig(speed_mode=4.0)


def test_sysmo32_limiter_clips_joint_position_and_limits_velocity():
    safety = Sysmo32ArmSafetyConfig(
        joint_lower_limits_rad=tuple([-1.0] * 12),
        joint_upper_limits_rad=tuple([1.0] * 12),
        max_joint_velocity_rad_s=tuple([0.5] * 12),
        max_joint_jump_rad=2.0,
    )
    limiter = Sysmo32CommandLimiter(safety)

    first = Sysmo32CommandBuilder(safety).build([2.0] * 6, [-2.0] * 6, timestamp_s=1.0)
    limited, reason = limiter.limit(first, now_s=1.0)
    assert limited is not None
    assert "joint position clipped" in reason
    assert limited.left_arm == tuple([1.0] * 6)
    assert limited.right_arm == tuple([-1.0] * 6)

    second = Sysmo32CommandBuilder(safety).build([0.0] * 6, [0.0] * 6, timestamp_s=1.1)
    limited, reason = limiter.limit(second, now_s=1.1)
    assert limited is not None
    assert "joint velocity limited" in reason
    assert np.allclose(limited.left_arm, [0.95] * 6)
    assert np.allclose(limited.right_arm, [-0.95] * 6)


def test_sysmo32_limiter_limits_large_jump():
    safety = Sysmo32ArmSafetyConfig(
        max_joint_jump_rad=0.2,
        max_joint_velocity_rad_s=tuple([10.0] * 12),
    )
    limiter = Sysmo32CommandLimiter(safety)
    builder = Sysmo32CommandBuilder(safety)
    first = builder.build([0.0] * 6, [0.0] * 6, timestamp_s=1.0)
    assert limiter.limit(first, now_s=1.0)[0] is not None

    second = builder.build([0.5] * 6, [0.0] * 6, timestamp_s=1.1)
    limited, reason = limiter.limit(second, now_s=1.1)
    assert limited is not None
    assert "joint jump limited" in reason
    assert np.allclose(limited.left_arm, [0.2] * 6)
    assert np.allclose(limited.right_arm, [0.0] * 6)


def test_sysmo32_cartesian_target_jump_is_clamped_not_rejected():
    safety = Sysmo32ArmSafetyConfig(max_translation_step_m=0.02, max_rotation_step_rad=0.08)
    controller = Sysmo32RealControl.__new__(Sysmo32RealControl)
    controller.config = Sysmo32RealControlConfig(arm=safety)
    controller._last_safety_log_time = {}
    controller._last_accepted_targets = {
        robots.LEFT: None,
        robots.RIGHT: CartesianTarget(
            timestamp_s=1.0,
            hand_side=robots.RIGHT,
            frame_id="base",
            position_m=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    }
    raw_target = CartesianTarget(
        timestamp_s=1.1,
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.10, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    clamped = controller._sanitize_cartesian_target(robots.RIGHT, raw_target)

    assert clamped is not None
    assert np.allclose(clamped.position_m, (0.02, 0.0, 0.0))


def test_hand_gesture_mapper_hysteresis_pause_and_timeout_release():
    mapper = Sysmo32HandGestureMapper(confirm_frames=3, hand_frame_timeout_s=0.3)
    now = 10.0

    assert mapper.update_from_keypoints(robots.RIGHT, _hand_keypoints(0.02), now_s=now) == 1
    assert mapper.update_from_keypoints(robots.RIGHT, _hand_keypoints(0.02), now_s=now + 0.01) == 1
    assert mapper.update_from_keypoints(robots.RIGHT, _hand_keypoints(0.02), now_s=now + 0.02) == 2
    assert mapper.update_from_keypoints(robots.RIGHT, _hand_keypoints(0.04), now_s=now + 0.03) == 2
    assert mapper.update_from_keypoints(robots.RIGHT, _hand_keypoints(0.06), now_s=now + 0.04) == 1

    mapper.update_from_keypoints(robots.LEFT, _hand_keypoints(0.02), now_s=now)
    mapper.update_from_keypoints(robots.LEFT, _hand_keypoints(0.02), now_s=now + 0.01)
    mapper.update_from_keypoints(robots.LEFT, _hand_keypoints(0.02), now_s=now + 0.02)
    assert mapper.force_release(robots.LEFT) == SYSMO32_HAND_ACTION_RELEASE
    assert mapper.action_for(robots.LEFT, now_s=now + 1.0) == SYSMO32_HAND_ACTION_RELEASE
    assert not mapper.has_fresh_frame(robots.LEFT, now_s=now + 1.0)


def test_joint_state_cache_parses_and_rejects_stale_state():
    cache = Sysmo32JointStateCache(joint_state_timeout_s=0.5)
    names = list(SYSMO32_LEFT_JOINT_NAMES) + list(SYSMO32_RIGHT_JOINT_NAMES)
    positions = list(range(12))
    msg = SimpleNamespace(name=names, position=positions)

    snapshot = cache.update_from_joint_state_msg(msg, now_s=10.0)

    assert snapshot.left_arm == tuple(float(v) for v in range(6))
    assert snapshot.right_arm == tuple(float(v) for v in range(6, 12))
    assert cache.is_fresh(now_s=10.4)
    assert not cache.is_fresh(now_s=10.6)


def test_pause_immediately_publishes_release_without_new_hand_frame(monkeypatch, bus):
    import beavr.teleop.components.interface.robots.sysmo32_real_control as real_mod

    class FakeSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def recv_keypoints(self):
            return None

        def stop(self):
            return None

    class FakeKinematics:
        available = False

        def __init__(self, urdf_path):
            self.urdf_path = urdf_path

        def placeholder_ik(self, hand_side, target, current_joints):
            return np.zeros(6)

    monkeypatch.setattr(real_mod, "ZMQSubscriber", FakeSubscriber)
    monkeypatch.setattr(real_mod, "Sysmo32MujocoKinematics", FakeKinematics)
    monkeypatch.setattr(real_mod, "cleanup_zmq_resources", lambda: None)

    controller = Sysmo32RealControl(
        host="127.0.0.1",
        control_backend="mujoco",
        right_target_port=10011,
        left_target_port=10013,
        right_state_publish_port=10012,
        left_state_publish_port=10014,
        teleoperation_state_port=ports.KEYPOINT_STREAM_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
    )

    controller._enter_pause("unit-test pause")

    left_action = bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_LEFT_HAND_ACTION_TOPIC)
    right_action = bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_RIGHT_HAND_ACTION_TOPIC)
    assert left_action.action_id == SYSMO32_HAND_ACTION_RELEASE
    assert right_action.action_id == SYSMO32_HAND_ACTION_RELEASE
    assert not controller._teleop_active
    controller.cleanup()


def test_mujoco_hand_action_callback_prints_only(caplog):
    mirror = Sysmo32MujocoCommandMirror.__new__(Sysmo32MujocoCommandMirror)

    with caplog.at_level(logging.INFO):
        mirror.on_left_hand_action(SYSMO32_HAND_ACTION_RELEASE)
        mirror.on_right_hand_action(SYSMO32_HAND_ACTION_GRASP)

    assert "left action=1, print only, no execution" in caplog.text
    assert "right action=2, print only, no execution" in caplog.text


def test_sysmo32_config_routes_backends():
    mujoco_cfg = load_robot_config("sysmo32", Laterality.BIMANUAL, True, control_backend="mujoco")
    assert mujoco_cfg.control_backend == "mujoco"
    assert len(mujoco_cfg.robots) == 1
    assert len(mujoco_cfg.environment) == 1
    assert mujoco_cfg.camera_streamers == []

    real_cfg = load_robot_config("sysmo32", Laterality.BIMANUAL, False, control_backend="real")
    assert real_cfg.control_backend == "real"
    assert len(real_cfg.robots) == 1
    assert real_cfg.environment == []
    assert real_cfg.operators[0].teleoperation_state_port == ports.KEYPOINT_STREAM_PORT
    assert real_cfg.operators[0].hand_frame_timeout_s == 0.3
    assert real_cfg.operators[0].rotation_delta_frame == "base"
