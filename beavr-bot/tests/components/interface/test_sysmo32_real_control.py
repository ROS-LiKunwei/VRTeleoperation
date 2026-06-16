import time
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
    SYSMO32_ARM_COMMAND_TOPIC,
    SYSMO32_LEFT_HAND_ACTION_TOPIC,
    SYSMO32_RIGHT_HAND_ACTION_TOPIC,
    Sysmo32RealControl,
    Sysmo32RealControlConfig,
)
from beavr.teleop.components.interface.robots.sysmo32_kinematics import (
    SYSMO32_LEFT_ELBOW_INDEX,
    SYSMO32_LEFT_WRIST_YAW_INDEX,
    SYSMO32_RIGHT_ELBOW_INDEX,
    SYSMO32_RIGHT_WRIST_YAW_INDEX,
    Sysmo32MujocoKinematics,
    _damped_least_squares_delta,
    _default_nullspace_reference_joints,
    _elbow_sign_satisfied,
    _select_elbow_preferred_result,
    _wrist_yaw_reference_error,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.simulation.sysmo32_mujoco_command_sim import (
    Sysmo32MujocoCommandMirror,
    _quintic_blend,
)
from beavr.teleop.configs.constants import ports, robots


def _hand_keypoints(distance: float) -> np.ndarray:
    keypoints = np.zeros((26, 3), dtype=np.float64)
    keypoints[5] = np.array([0.0, 0.0, 0.0])
    keypoints[10] = np.array([distance, 0.0, 0.0])
    return keypoints


def _curled_hand_keypoints() -> np.ndarray:
    keypoints = np.zeros((26, 3), dtype=np.float64)
    keypoints[0] = np.array([0.0, 0.0, 0.0])
    for finger in ("index", "middle", "ring", "pinky"):
        chain = robots.OCULUS_JOINTS[finger]
        keypoints[chain[0]] = np.array([0.05, 0.0, 0.0])
        keypoints[chain[-1]] = np.array([0.04, 0.0, 0.0])
    keypoints[robots.OCULUS_JOINTS["thumb"][-1]] = np.array([0.0, 0.08, 0.0])
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


def test_sysmo32_speed_mode_4_is_allowed():
    command = Sysmo32CommandBuilder(Sysmo32ArmSafetyConfig(speed_mode=4.0)).build(
        [0.1] * 6,
        [-0.1] * 6,
        timestamp_s=1.0,
    )

    assert command.speed_mode == 4.0


def test_sysmo32_real_control_config_normalizes_arm_smoother_mode():
    config = Sysmo32RealControlConfig(arm_trajectory_smoother="servo")

    assert config.arm_trajectory_smoother == "jerk_limited_servo"

    with pytest.raises(ValueError, match="arm_trajectory_smoother"):
        Sysmo32RealControlConfig(arm_trajectory_smoother="bad_mode")


def test_sysmo32_ik_nullspace_delta_moves_only_unconstrained_joint_toward_reference():
    jacobian = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    current_qpos = np.array([0.25, -0.25, 1.0], dtype=np.float64)
    reference_qpos = np.array([-1.0, 1.0, -1.0], dtype=np.float64)

    delta = _damped_least_squares_delta(
        jacobian,
        np.zeros(2, dtype=np.float64),
        current_qpos,
        reference_qpos,
        damping=1e-6,
        nullspace_gain=0.2,
        nullspace_step_limit_rad=0.3,
    )

    np.testing.assert_allclose(delta[:2], np.zeros(2), atol=1e-8)
    assert delta[2] < 0.0
    assert np.linalg.norm(delta) <= 0.3 + 1e-9


def test_sysmo32_default_nullspace_reference_uses_human_elbow_signs():
    reference = _default_nullspace_reference_joints()

    assert reference[SYSMO32_LEFT_ELBOW_INDEX] < 0.0
    assert reference[SYSMO32_RIGHT_ELBOW_INDEX] > 0.0
    assert reference[SYSMO32_LEFT_WRIST_YAW_INDEX] == 0.0
    assert reference[SYSMO32_RIGHT_WRIST_YAW_INDEX] == 0.0
    assert _elbow_sign_satisfied(robots.LEFT, reference[:6])
    assert _elbow_sign_satisfied(robots.RIGHT, reference[6:12])
    assert _wrist_yaw_reference_error(robots.LEFT, reference[:6], reference[:6]) == 0.0
    assert _wrist_yaw_reference_error(robots.RIGHT, reference[6:12], reference[6:12]) == 0.0


def test_sysmo32_nullspace_config_keeps_human_elbow_signs_for_zero_reference():
    kinematics = Sysmo32MujocoKinematics.__new__(Sysmo32MujocoKinematics)
    kinematics._nullspace_reference_joints = np.zeros(12, dtype=np.float64)
    kinematics._nullspace_gain = 0.0
    kinematics._nullspace_step_limit_rad = 0.0

    kinematics.configure_nullspace([0.0] * 12, gain=0.03, step_limit_rad=0.015)

    assert kinematics._nullspace_reference_joints[SYSMO32_LEFT_ELBOW_INDEX] < 0.0
    assert kinematics._nullspace_reference_joints[SYSMO32_RIGHT_ELBOW_INDEX] > 0.0


def test_sysmo32_nullspace_weights_pull_wrist_yaw_harder():
    jacobian = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    current_qpos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    reference_qpos = np.zeros(3, dtype=np.float64)

    unweighted = _damped_least_squares_delta(
        jacobian,
        np.zeros(2, dtype=np.float64),
        current_qpos,
        reference_qpos,
        damping=1e-6,
        nullspace_gain=0.1,
        nullspace_step_limit_rad=1.0,
    )
    weighted = _damped_least_squares_delta(
        jacobian,
        np.zeros(2, dtype=np.float64),
        current_qpos,
        reference_qpos,
        damping=1e-6,
        nullspace_gain=0.1,
        nullspace_step_limit_rad=1.0,
        nullspace_weights=np.array([1.0, 1.0, 4.0], dtype=np.float64),
    )

    assert weighted[2] < unweighted[2]
    np.testing.assert_allclose(weighted[:2], np.zeros(2), atol=1e-8)


def test_sysmo32_elbow_preference_does_not_override_large_task_error():
    tracking_best = np.zeros(6, dtype=np.float64)
    elbow_valid = np.ones(6, dtype=np.float64)

    selected = _select_elbow_preferred_result(
        tracking_best,
        best_task_error=0.005,
        best_valid_qpos=elbow_valid,
        best_valid_task_error=0.04,
    )

    np.testing.assert_array_equal(selected, tracking_best)

    selected = _select_elbow_preferred_result(
        tracking_best,
        best_task_error=0.005,
        best_valid_qpos=elbow_valid,
        best_valid_task_error=0.015,
    )

    np.testing.assert_array_equal(selected, elbow_valid)


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


def test_sysmo32_stale_cartesian_target_is_dropped_after_hold():
    controller = Sysmo32RealControl.__new__(Sysmo32RealControl)
    controller.control_backend = "mujoco"
    controller.config = Sysmo32RealControlConfig(hand_frame_timeout_s=0.1)
    controller._teleop_active = True
    controller._latest_targets = {
        robots.LEFT: CartesianTarget(
            timestamp_s=1.0,
            hand_side=robots.LEFT,
            frame_id="base",
            position_m=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        robots.RIGHT: None,
    }
    controller._last_safety_log_time = {}
    controller._real_joint_state_fresh = lambda: True
    controller._current_joint_snapshot = lambda: SimpleNamespace(all_joints=np.zeros(12))
    warnings = []
    controller._warn_safety = lambda key, message: warnings.append((key, message))

    controller._publish_arm_command_if_safe()

    assert controller._latest_targets[robots.LEFT] is None
    assert warnings == [(f"{robots.LEFT}_target_stale", f"{robots.LEFT} arm held: CartesianTarget stale")]


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


def test_hand_gesture_mapper_recognizes_curled_keypoints():
    mapper = Sysmo32HandGestureMapper(confirm_frames=2, hand_frame_timeout_s=0.3)
    now = 20.0

    assert (
        mapper.update_from_keypoints(robots.RIGHT, _curled_hand_keypoints(), now_s=now)
        == SYSMO32_HAND_ACTION_RELEASE
    )
    assert (
        mapper.update_from_keypoints(robots.RIGHT, _curled_hand_keypoints(), now_s=now + 0.01)
        == SYSMO32_HAND_ACTION_GRASP
    )


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


def test_real_control_reset_does_not_replace_ik_elbow_reference_with_current_joints(monkeypatch, bus):
    import beavr.teleop.components.interface.robots.sysmo32_real_control as real_mod

    class FakeSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def recv_keypoints(self):
            return None

        def stop(self):
            return None

    kinematics_instances = []

    class FakeKinematics:
        available = True

        def __init__(self, urdf_path):
            self.urdf_path = urdf_path
            self.nullspace_calls = []
            kinematics_instances.append(self)

        def configure_nullspace(
            self,
            reference_joints_rad=None,
            gain=None,
            step_limit_rad=None,
            orientation_weight=None,
            max_joint_step_rad=None,
        ):
            self.nullspace_calls.append(
                {
                    "reference": None
                    if reference_joints_rad is None
                    else tuple(float(v) for v in reference_joints_rad),
                    "gain": gain,
                    "step_limit_rad": step_limit_rad,
                    "orientation_weight": orientation_weight,
                    "max_joint_step_rad": max_joint_step_rad,
                }
            )

        def fk(self, hand_side, all_joints):
            return np.eye(4, dtype=np.float64)

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
        teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
        config=Sysmo32RealControlConfig(
            control_backend="mujoco",
            ik_nullspace_gain=0.07,
            ik_nullspace_step_limit_rad=0.02,
            ik_orientation_weight=0.18,
            ik_max_joint_step_rad=0.06,
        ),
    )
    kinematics = kinematics_instances[0]
    expected_nullspace_calls = [
        {
            "reference": None,
            "gain": 0.07,
            "step_limit_rad": 0.02,
            "orientation_weight": 0.18,
            "max_joint_step_rad": 0.06,
        }
    ]
    assert kinematics.nullspace_calls == expected_nullspace_calls

    controller._dry_joint_cache.update(range(6), range(6, 12), now_s=time.time())

    controller._publish_current_endeff_homo(robots.RIGHT, publish_port=10012)

    assert kinematics.nullspace_calls == expected_nullspace_calls
    assert bus.recv_latest(10012, "endeff_homo") is not None
    controller.cleanup()


def test_real_control_publishes_lerobot_joint_states_at_configured_rate(monkeypatch, bus):
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

    monkeypatch.setattr(real_mod, "ZMQSubscriber", FakeSubscriber)
    monkeypatch.setattr(real_mod, "Sysmo32MujocoKinematics", FakeKinematics)
    monkeypatch.setattr(real_mod, "cleanup_zmq_resources", lambda: None)

    controller = Sysmo32RealControl(
        host="127.0.0.1",
        control_backend="mujoco",
        right_target_port=10011,
        left_target_port=10013,
        right_endeff_publish_port=10012,
        left_endeff_publish_port=10014,
        right_state_publish_port=10018,
        left_state_publish_port=10020,
        teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
        config=Sysmo32RealControlConfig(control_backend="mujoco", state_publish_fps=30.0),
    )
    controller._dry_joint_cache.update([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], now_s=time.time())
    controller._latest_targets[robots.RIGHT] = CartesianTarget(
        timestamp_s=123.0,
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    controller._publish_lerobot_joint_states()

    right_state = bus.recv_latest(10018, "sysmo32_right")
    left_state = bus.recv_latest(10020, "sysmo32_left")
    assert right_state["joint_states"]["joint_position"] == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    assert left_state["joint_states"]["joint_position"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert right_state["joint_angles_rad"] == right_state["joint_states"]["joint_position"]
    assert right_state["commanded_cartesian_state"]["commanded_cartesian_position"] == [
        0.1,
        0.2,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert right_state["commanded_cartesian_state"]["timestamp_s"] == 123.0
    assert "commanded_cartesian_state" not in left_state
    controller.cleanup()


def test_hand_action_not_published_before_first_hand_frame(monkeypatch, bus):
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
        teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
    )

    controller._publish_hand_actions_for_current_state()
    controller._enter_pause("unit-test pause")

    assert bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_LEFT_HAND_ACTION_TOPIC) is None
    assert bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_RIGHT_HAND_ACTION_TOPIC) is None
    assert not controller._teleop_active
    controller.cleanup()


def test_hand_action_publishes_after_first_hand_frame(monkeypatch, bus):
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
        teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
    )

    controller._hand_mapper.update_from_keypoints(robots.RIGHT, _hand_keypoints(0.02), now_s=time.time())
    controller._hand_frame_started[robots.RIGHT] = True
    controller._publish_hand_actions_for_current_state()

    assert bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_LEFT_HAND_ACTION_TOPIC) is None
    right_action = bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_RIGHT_HAND_ACTION_TOPIC)
    assert right_action.action_id == SYSMO32_HAND_ACTION_RELEASE
    controller.cleanup()


def test_pause_publishes_release_after_hand_frame_started(monkeypatch, bus):
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
        teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
    )
    controller._hand_frame_started[robots.LEFT] = True
    controller._hand_frame_started[robots.RIGHT] = True
    controller._enter_pause("unit-test pause")

    left_action = bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_LEFT_HAND_ACTION_TOPIC)
    right_action = bus.recv_latest(ports.SYSMO32_HAND_ACTION_MIRROR_PORT, SYSMO32_RIGHT_HAND_ACTION_TOPIC)
    assert left_action.action_id == SYSMO32_HAND_ACTION_RELEASE
    assert right_action.action_id == SYSMO32_HAND_ACTION_RELEASE
    assert not controller._teleop_active
    controller.cleanup()


def test_real_with_mujoco_pause_publishes_hold_from_real_joint_state(monkeypatch, bus):
    import beavr.teleop.components.interface.robots.sysmo32_real_control as real_mod

    class FakeSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def recv_keypoints(self):
            return None

        def stop(self):
            return None

    class FakeRos2Bridge:
        def __init__(self, topics, require_ros):
            self.available = True
            self.require_ros = require_ros
            self.joint_cache = Sysmo32JointStateCache(topics.joint_state_timeout_s)
            self.joint_cache.update([0.10] * 6, [-0.20] * 6, now_s=time.time())
            self.published_arm_commands = []

        def spin_once(self):
            return None

        def publish_arm_command(self, command):
            self.published_arm_commands.append(command)
            return True

        def publish_hand_action(self, hand_side, action_id):
            return True

        def close(self):
            return None

    class FakeKinematics:
        available = False

        def __init__(self, urdf_path):
            self.urdf_path = urdf_path

    monkeypatch.setattr(real_mod, "ZMQSubscriber", FakeSubscriber)
    monkeypatch.setattr(real_mod, "Sysmo32Ros2Bridge", FakeRos2Bridge)
    monkeypatch.setattr(real_mod, "Sysmo32MujocoKinematics", FakeKinematics)
    monkeypatch.setattr(real_mod, "cleanup_zmq_resources", lambda: None)

    controller = Sysmo32RealControl(
        host="127.0.0.1",
        control_backend="real_with_mujoco",
        right_target_port=10011,
        left_target_port=10013,
        right_state_publish_port=10012,
        left_state_publish_port=10014,
        teleoperation_state_port=ports.KEYPOINT_STREAM_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
        config=Sysmo32RealControlConfig(allow_mujoco_mirror_without_joint_state=False),
    )
    controller._dry_joint_cache.update([1.0] * 6, [1.0] * 6, now_s=time.time())

    controller._enter_pause("unit-test pause")

    assert len(controller._ros2.published_arm_commands) == 1
    hold_command = controller._ros2.published_arm_commands[0]
    assert np.allclose(hold_command.left_arm, [0.10] * 6)
    assert np.allclose(hold_command.right_arm, [-0.20] * 6)
    assert bus.recv_latest(ports.SYSMO32_ARM_COMMAND_MIRROR_PORT, SYSMO32_ARM_COMMAND_TOPIC) == hold_command
    assert controller._pause_hold_command == hold_command
    assert not controller._teleop_active
    controller.cleanup()


def test_pause_hold_heartbeat_republishes_last_hold_command(monkeypatch):
    import beavr.teleop.components.interface.robots.sysmo32_real_control as real_mod

    class FakeSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def recv_keypoints(self):
            return None

        def stop(self):
            return None

    class FakeRos2Bridge:
        def __init__(self, topics, require_ros):
            self.available = True
            self.joint_cache = Sysmo32JointStateCache(topics.joint_state_timeout_s)
            self.joint_cache.update([0.30] * 6, [-0.40] * 6, now_s=time.time())
            self.published_arm_commands = []

        def spin_once(self):
            return None

        def publish_arm_command(self, command):
            self.published_arm_commands.append(command)
            return True

        def publish_hand_action(self, hand_side, action_id):
            return True

        def close(self):
            return None

    class FakeKinematics:
        available = False

        def __init__(self, urdf_path):
            self.urdf_path = urdf_path

    monkeypatch.setattr(real_mod, "ZMQSubscriber", FakeSubscriber)
    monkeypatch.setattr(real_mod, "Sysmo32Ros2Bridge", FakeRos2Bridge)
    monkeypatch.setattr(real_mod, "Sysmo32MujocoKinematics", FakeKinematics)
    monkeypatch.setattr(real_mod, "cleanup_zmq_resources", lambda: None)

    controller = Sysmo32RealControl(
        host="127.0.0.1",
        control_backend="real",
        right_target_port=10011,
        left_target_port=10013,
        right_state_publish_port=10012,
        left_state_publish_port=10014,
        teleoperation_state_port=ports.KEYPOINT_STREAM_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
    )

    controller._enter_pause("unit-test pause")
    assert len(controller._ros2.published_arm_commands) == 1

    controller._ros2.joint_cache.snapshot.timestamp_s = 0.0
    controller._last_pause_hold_publish_time = 0.0
    controller._publish_pause_hold_if_needed()

    assert len(controller._ros2.published_arm_commands) == 2
    assert np.allclose(controller._ros2.published_arm_commands[1].left_arm, [0.30] * 6)
    assert np.allclose(controller._ros2.published_arm_commands[1].right_arm, [-0.40] * 6)
    controller.cleanup()


def test_mujoco_hand_action_callback_prints_only(monkeypatch):
    import beavr.teleop.components.simulation.sysmo32_mujoco_command_sim as sim_mod

    mirror = Sysmo32MujocoCommandMirror.__new__(Sysmo32MujocoCommandMirror)
    messages = []

    def fake_info(message, *args):
        messages.append(message % args)

    monkeypatch.setattr(sim_mod.logger, "info", fake_info)
    mirror.on_left_hand_action(SYSMO32_HAND_ACTION_RELEASE)
    mirror.on_right_hand_action(SYSMO32_HAND_ACTION_GRASP)

    assert any("left action=1, print only, no execution" in message for message in messages)
    assert any("right action=2, print only, no execution" in message for message in messages)


def test_mujoco_command_mirror_accepts_ros2_real_command_topic():
    mirror = Sysmo32MujocoCommandMirror.__new__(Sysmo32MujocoCommandMirror)
    applied = []
    mirror.apply_arm_command = applied.append
    data = [0.1] * 6 + [-0.2] * 6 + [0.0] * 6

    mirror._on_ros_arm_command(SimpleNamespace(data=data))

    assert len(applied) == 1
    assert applied[0].values == tuple(data)


def test_mujoco_command_mirror_uses_quintic_interpolation():
    mirror = Sysmo32MujocoCommandMirror.__new__(Sysmo32MujocoCommandMirror)
    mirror.control_dt = 0.1
    mirror.arm_command_interpolation_steps = 5
    mirror._arm_joint_ids = list(range(12))
    mirror._arm_qpos_addrs = list(range(12))
    mirror._hold_joint_positions = np.zeros(12, dtype=np.float64)
    mirror._trajectory_start_positions = None
    mirror._trajectory_target_positions = None
    mirror._trajectory_start_time_s = None
    mirror._log_applied_arm_command = lambda values: None
    model = SimpleNamespace(jnt_range=np.asarray([[-10.0, 10.0]] * 12, dtype=np.float64))
    data = SimpleNamespace(qpos=np.zeros(12, dtype=np.float64))
    mirror._kinematics = SimpleNamespace(available=True, model=model, data=data)

    command = Sysmo32ArmCommand(timestamp_s=1.0, values=tuple([1.0] * 12 + [0.0] * 6))
    mirror.apply_arm_command(command)
    start_s = mirror._trajectory_start_time_s

    assert np.allclose(mirror._hold_joint_positions, np.zeros(12))
    mirror._update_interpolated_hold(start_s + 0.25)
    assert np.allclose(mirror._hold_joint_positions, np.full(12, _quintic_blend(0.5)))

    mirror._update_interpolated_hold(start_s + 0.5)
    assert np.allclose(mirror._hold_joint_positions, np.ones(12))
    assert mirror._trajectory_target_positions is None


def test_mujoco_backend_publishes_same_command_to_ros_and_mirror(monkeypatch, bus):
    import beavr.teleop.components.interface.robots.sysmo32_real_control as real_mod

    class FakeSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def recv_keypoints(self):
            return None

        def stop(self):
            return None

    class FakeRos2Bridge:
        def __init__(self, topics, require_ros):
            self.available = True
            self.require_ros = require_ros
            self.joint_cache = Sysmo32JointStateCache(topics.joint_state_timeout_s)
            self.published_arm_commands = []

        def spin_once(self):
            return None

        def publish_arm_command(self, command):
            self.published_arm_commands.append(command)
            return True

        def publish_hand_action(self, hand_side, action_id):
            return True

        def close(self):
            return None

    class FakeKinematics:
        available = True

        def __init__(self, urdf_path):
            self.urdf_path = urdf_path

        def solve_ik(self, hand_side, target, current_joints):
            return np.full(6, 0.25)

    monkeypatch.setattr(real_mod, "ZMQSubscriber", FakeSubscriber)
    monkeypatch.setattr(real_mod, "Sysmo32Ros2Bridge", FakeRos2Bridge)
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
        config=Sysmo32RealControlConfig(
            allow_placeholder_ik_for_mujoco=True,
            publish_arm_command_topic_in_mujoco=True,
        ),
    )
    controller._latest_targets[robots.RIGHT] = CartesianTarget(
        timestamp_s=time.time(),
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    controller._publish_arm_command_if_safe()

    assert controller._ros2.require_ros
    assert len(controller._ros2.published_arm_commands) == 1
    mirror_command = bus.recv_latest(ports.SYSMO32_ARM_COMMAND_MIRROR_PORT, SYSMO32_ARM_COMMAND_TOPIC)
    assert mirror_command == controller._ros2.published_arm_commands[0]
    controller.cleanup()


def test_real_with_mujoco_holds_mirror_until_real_reset(monkeypatch, bus):
    import beavr.teleop.components.interface.robots.sysmo32_real_control as real_mod

    class FakeSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def recv_keypoints(self):
            return None

        def stop(self):
            return None

    class FakeRos2Bridge:
        def __init__(self, topics, require_ros):
            self.available = True
            self.joint_cache = Sysmo32JointStateCache(topics.joint_state_timeout_s)
            self.published_arm_commands = []

        def spin_once(self):
            return None

        def publish_arm_command(self, command):
            self.published_arm_commands.append(command)
            return True

        def publish_hand_action(self, hand_side, action_id):
            return True

        def close(self):
            return None

    class FakeKinematics:
        available = True

        def __init__(self, urdf_path):
            self.urdf_path = urdf_path

        def solve_ik(self, hand_side, target, current_joints):
            return np.ones(6)

    monkeypatch.setattr(real_mod, "ZMQSubscriber", FakeSubscriber)
    monkeypatch.setattr(real_mod, "Sysmo32Ros2Bridge", FakeRos2Bridge)
    monkeypatch.setattr(real_mod, "Sysmo32MujocoKinematics", FakeKinematics)
    monkeypatch.setattr(real_mod, "cleanup_zmq_resources", lambda: None)

    controller = Sysmo32RealControl(
        host="127.0.0.1",
        control_backend="real_with_mujoco",
        right_target_port=10011,
        left_target_port=10013,
        right_state_publish_port=10012,
        left_state_publish_port=10014,
        teleoperation_state_port=ports.KEYPOINT_STREAM_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
        config=Sysmo32RealControlConfig(allow_mujoco_mirror_without_joint_state=False),
    )
    controller._latest_targets[robots.RIGHT] = CartesianTarget(
        timestamp_s=time.time(),
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    controller._publish_arm_command_if_safe()

    assert controller._ros2.published_arm_commands == []
    assert bus.recv_latest(ports.SYSMO32_ARM_COMMAND_MIRROR_PORT, SYSMO32_ARM_COMMAND_TOPIC) is None
    controller.cleanup()


def test_real_with_mujoco_mirrors_after_real_publish_gate(monkeypatch, bus):
    import beavr.teleop.components.interface.robots.sysmo32_real_control as real_mod

    class FakeSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def recv_keypoints(self):
            return None

        def stop(self):
            return None

    class FakeRos2Bridge:
        def __init__(self, topics, require_ros):
            self.available = True
            self.joint_cache = Sysmo32JointStateCache(topics.joint_state_timeout_s)
            self.joint_cache.update([0.0] * 6, [0.0] * 6, now_s=time.time())
            self.published_arm_commands = []

        def spin_once(self):
            return None

        def publish_arm_command(self, command):
            self.published_arm_commands.append(command)
            return True

        def publish_hand_action(self, hand_side, action_id):
            return True

        def close(self):
            return None

    class FakeKinematics:
        available = True

        def __init__(self, urdf_path):
            self.urdf_path = urdf_path

        def solve_ik(self, hand_side, target, current_joints):
            return np.full(6, 0.25)

    monkeypatch.setattr(real_mod, "ZMQSubscriber", FakeSubscriber)
    monkeypatch.setattr(real_mod, "Sysmo32Ros2Bridge", FakeRos2Bridge)
    monkeypatch.setattr(real_mod, "Sysmo32MujocoKinematics", FakeKinematics)
    monkeypatch.setattr(real_mod, "cleanup_zmq_resources", lambda: None)

    controller = Sysmo32RealControl(
        host="127.0.0.1",
        control_backend="real_with_mujoco",
        right_target_port=10011,
        left_target_port=10013,
        right_state_publish_port=10012,
        left_state_publish_port=10014,
        teleoperation_state_port=ports.KEYPOINT_STREAM_PORT,
        transformed_right_port=ports.KEYPOINT_TRANSFORM_PORT,
        transformed_left_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
        config=Sysmo32RealControlConfig(allow_mujoco_mirror_without_joint_state=False),
    )
    controller._real_reset_ready = True
    controller._latest_targets[robots.RIGHT] = CartesianTarget(
        timestamp_s=time.time(),
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    controller._publish_arm_command_if_safe()

    assert len(controller._ros2.published_arm_commands) == 1
    mirror_command = bus.recv_latest(ports.SYSMO32_ARM_COMMAND_MIRROR_PORT, SYSMO32_ARM_COMMAND_TOPIC)
    assert mirror_command == controller._ros2.published_arm_commands[0]
    controller.cleanup()


def test_sysmo32_config_routes_backends():
    mujoco_cfg = load_robot_config("sysmo32", Laterality.BIMANUAL, True, control_backend="mujoco")
    assert mujoco_cfg.control_backend == "mujoco"
    assert len(mujoco_cfg.robots) == 1
    assert len(mujoco_cfg.environment) == 1
    assert mujoco_cfg.camera_streamers == []
    assert mujoco_cfg.robots[0].config.publish_arm_command_topic_in_mujoco
    assert mujoco_cfg.environment[0].arm_command_source == "ros2"
    assert mujoco_cfg.environment[0].ros_arm_command_topic == "/sysmo_left_arm_controller/commands"
    assert mujoco_cfg.environment[0].publish_joint_states
    assert mujoco_cfg.environment[0].joint_state_topic == "/joint_states"
    assert mujoco_cfg.environment[0].arm_command_interpolation_steps == 5

    real_cfg = load_robot_config("sysmo32", Laterality.BIMANUAL, False, control_backend="real")
    assert real_cfg.control_backend == "real"
    assert len(real_cfg.robots) == 1
    assert real_cfg.environment == []
    robot_cfg = real_cfg.robots[0]
    assert robot_cfg.right_target_port == ports.XARM_ENDEFF_SUBSCRIBE_PORT + 2
    assert robot_cfg.left_target_port == ports.XARM_ENDEFF_SUBSCRIBE_PORT + 4
    assert robot_cfg.right_endeff_publish_port == ports.XARM_ENDEFF_PUBLISH_PORT + 2
    assert robot_cfg.left_endeff_publish_port == ports.XARM_ENDEFF_PUBLISH_PORT + 4
    assert robot_cfg.right_state_publish_port == ports.XARM_STATE_PUBLISH_PORT + 2
    assert robot_cfg.left_state_publish_port == ports.XARM_STATE_PUBLISH_PORT + 4
    assert robot_cfg.config.ros2.joint_state_timeout_s == 1.0
    assert robot_cfg.config.hand_frame_timeout_s == 1.0
    assert real_cfg.robots[0].teleoperation_state_port == ports.XARM_TELEOPERATION_STATE_PORT
    assert real_cfg.operators[0].endeff_publish_port == robot_cfg.right_target_port
    assert real_cfg.operators[0].endeff_subscribe_port == robot_cfg.right_endeff_publish_port
    assert real_cfg.operators[1].endeff_publish_port == robot_cfg.left_target_port
    assert real_cfg.operators[1].endeff_subscribe_port == robot_cfg.left_endeff_publish_port
    assert real_cfg.operators[0].teleoperation_state_port == ports.XARM_TELEOPERATION_STATE_PORT
    assert real_cfg.operators[0].hand_frame_timeout_s == 1.0
    assert real_cfg.operators[0].rotation_delta_frame == "base"

    real_with_mujoco_cfg = load_robot_config(
        "sysmo32", Laterality.BIMANUAL, True, control_backend="real_with_mujoco"
    )
    assert real_with_mujoco_cfg.control_backend == "real_with_mujoco"
    assert len(real_with_mujoco_cfg.robots) == 1
    assert len(real_with_mujoco_cfg.environment) == 1
    assert not real_with_mujoco_cfg.robots[0].config.allow_mujoco_mirror_without_joint_state
    assert real_with_mujoco_cfg.environment[0].arm_command_source == "ros2"
    assert not real_with_mujoco_cfg.environment[0].publish_joint_states
