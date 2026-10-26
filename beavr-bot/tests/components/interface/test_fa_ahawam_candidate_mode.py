import builtins
from types import SimpleNamespace

import draccus
import numpy as np
import pytest

from beavr.teleop.common.configs.loader import Laterality, load_robot_config
from beavr.teleop.components.interface.robots import fa_real_control as fa_real_control_module
from beavr.teleop.components.interface.robots.fa_real_control import (
    FA_AHAWAM_HUMAN_LEFT_HAND_TOPIC,
    FA_AHAWAM_HUMAN_MIN_SNAP_TARGET_TOPIC,
    FA_AHAWAM_HUMAN_RIGHT_HAND_TOPIC,
    FA_AHAWAM_REBASELINE_ACK_TOPIC,
    FA_AHAWAM_REBASELINE_REQUEST_TOPIC,
    FA_REAL_LEFT_HAND_TOPIC,
    FA_REAL_MIN_SNAP_TARGET_TOPIC,
    FA_REAL_RIGHT_HAND_TOPIC,
    FaRealControlConfig,
    FaRos2Topics,
)
from beavr.teleop.components.interface.robots.fa_command_builder import (
    FaJointStateSnapshot,
)
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator
from beavr.teleop.configs.constants import robots
from beavr.teleop.configs.robots.fa_config import FaRealControlCfg
from beavr.teleop.main import MainConfig


def test_fa_standalone_teleop_defaults_are_unchanged():
    cfg = FaRealControlCfg(control_backend="real")

    assert not cfg.ahawam_intervention_mode
    assert not cfg.config.ahawam_intervention_mode
    assert cfg.config.ros2.min_snap_target_topic == FA_REAL_MIN_SNAP_TARGET_TOPIC
    assert cfg.config.ros2.left_hand_topic == FA_REAL_LEFT_HAND_TOPIC
    assert cfg.config.ros2.right_hand_topic == FA_REAL_RIGHT_HAND_TOPIC
    assert cfg.config.initial_pose_enabled
    assert cfg.config.initial_hand_ros_action == 21


def test_fa_ahawam_mode_routes_only_human_candidates_and_disables_startup_actions():
    standalone = FaRealControlCfg(control_backend="real")
    candidate = FaRealControlCfg(
        control_backend="real",
        ahawam_intervention_mode=True,
    )

    assert candidate.config.ahawam_intervention_mode
    assert candidate.config.ros2.min_snap_target_topic == FA_AHAWAM_HUMAN_MIN_SNAP_TARGET_TOPIC
    assert candidate.config.ros2.left_hand_topic == FA_AHAWAM_HUMAN_LEFT_HAND_TOPIC
    assert candidate.config.ros2.right_hand_topic == FA_AHAWAM_HUMAN_RIGHT_HAND_TOPIC
    assert not candidate.config.initial_pose_enabled
    assert candidate.config.initial_hand_ros_action is None

    # Candidate routing does not bypass or replace the existing IK/limiter
    # configuration.
    assert candidate.config.ik == standalone.config.ik
    assert candidate.config.upper == standalone.config.upper
    assert candidate.config.max_ik_solution_jump_rad == standalone.config.max_ik_solution_jump_rad


@pytest.mark.parametrize(
    ("override", "direct_topic"),
    [
        ({"ahawam_human_min_snap_target_topic": FA_REAL_MIN_SNAP_TARGET_TOPIC}, FA_REAL_MIN_SNAP_TARGET_TOPIC),
        ({"ahawam_human_left_hand_topic": FA_REAL_LEFT_HAND_TOPIC}, FA_REAL_LEFT_HAND_TOPIC),
        ({"ahawam_human_right_hand_topic": FA_REAL_RIGHT_HAND_TOPIC}, FA_REAL_RIGHT_HAND_TOPIC),
    ],
)
def test_fa_ahawam_candidate_config_rejects_direct_hardware_topics(override, direct_topic):
    with pytest.raises(ValueError, match=f"direct FA hardware topic: {direct_topic}"):
        FaRealControlConfig(
            ahawam_intervention_mode=True,
            ros2=FaRos2Topics(**override),
        )


def test_fa_ahawam_candidate_config_rejects_topic_aliasing():
    with pytest.raises(ValueError, match="candidate topics must be distinct"):
        FaRealControlConfig(
            ahawam_intervention_mode=True,
            ros2=FaRos2Topics(
                ahawam_human_left_hand_topic="/ahawam/fa/human/shared_hand",
                ahawam_human_right_hand_topic="/ahawam/fa/human/shared_hand",
            ),
        )


def test_fa_ahawam_loader_dataclass_override_reaches_real_control_config():
    cfg = load_robot_config(
        "fa",
        Laterality.BIMANUAL,
        simulation_mode=False,
        control_backend="real",
        ahawam_intervention_mode=True,
    )

    assert cfg.ahawam_intervention_mode
    assert cfg.robots[0].ahawam_intervention_mode
    assert cfg.robots[0].config.ahawam_intervention_mode
    assert cfg.robots[0].config.ros2.min_snap_target_topic == FA_AHAWAM_HUMAN_MIN_SNAP_TARGET_TOPIC


def test_fa_ahawam_mode_is_available_as_a_top_level_cli_override():
    cfg = draccus.parse(
        MainConfig,
        args=[
            "--robot_name=fa",
            "--laterality=bimanual",
            "--control_backend=real",
            "--ahawam_intervention_mode=true",
        ],
    )

    assert cfg.ahawam_intervention_mode
    assert cfg.robot.robots[0].config.ahawam_intervention_mode
    assert cfg.robot.robots[0].config.ros2.min_snap_target_topic == FA_AHAWAM_HUMAN_MIN_SNAP_TARGET_TOPIC


def test_fa_ros_bridge_constructs_candidate_publishers_without_real_output_topics(monkeypatch):
    min_snap_topics = []
    hand_topics = []
    subscriptions = []
    node = SimpleNamespace(
        create_publisher=lambda msg, topic, queue: hand_topics.append(topic) or object(),
        create_subscription=lambda msg, topic, callback, queue: subscriptions.append(topic),
    )
    modules = {
        "rclpy": SimpleNamespace(ok=lambda: True, create_node=lambda name: node),
        "min_snap.msg": SimpleNamespace(MinSnapTarget=object),
        "sensor_msgs.msg": SimpleNamespace(JointState=object),
        "std_msgs.msg": SimpleNamespace(Int32=object),
    }
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in modules:
            return modules[name]
        return original_import(name, *args, **kwargs)

    def fake_min_snap_publisher(**kwargs):
        min_snap_topics.append(kwargs["topic"])
        return object()

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        fa_real_control_module,
        "MinSnapTargetPublisher",
        fake_min_snap_publisher,
    )
    config = FaRealControlConfig(ahawam_intervention_mode=True)

    bridge = fa_real_control_module.FaRos2Bridge(config.ros2, require_ros=True)

    assert bridge.available
    assert min_snap_topics == [FA_AHAWAM_HUMAN_MIN_SNAP_TARGET_TOPIC]
    assert hand_topics == [
        FA_AHAWAM_HUMAN_LEFT_HAND_TOPIC,
        FA_AHAWAM_HUMAN_RIGHT_HAND_TOPIC,
        FA_AHAWAM_REBASELINE_ACK_TOPIC,
    ]
    assert FA_REAL_MIN_SNAP_TARGET_TOPIC not in min_snap_topics
    assert FA_REAL_LEFT_HAND_TOPIC not in hand_topics
    assert FA_REAL_RIGHT_HAND_TOPIC not in hand_topics
    assert subscriptions == [FA_AHAWAM_REBASELINE_REQUEST_TOPIC, "/joint_states"]


def test_ahawam_rebaseline_request_waits_for_both_hand_init_resets():
    class FakeRos:
        def __init__(self):
            self.request = 41
            self.acks = []

        def take_rebaseline_request(self):
            request, self.request = self.request, None
            return request

        def publish_rebaseline_ack(self, request_id):
            self.acks.append(request_id)
            return True

    class FakePublisherManager:
        def __init__(self):
            self.messages = []

        def publish(self, host, port, topic, message):
            self.messages.append((host, port, topic, message))

    controller = fa_real_control_module.FaRealControl.__new__(
        fa_real_control_module.FaRealControl
    )
    controller.config = FaRealControlConfig(ahawam_intervention_mode=True)
    controller._ros2 = FakeRos()
    controller._publisher_manager = FakePublisherManager()
    controller.host = "127.0.0.1"
    controller._left_endeff_publish_port = 10022
    controller._right_endeff_publish_port = 10012
    controller._hand_init_ready = {robots.LEFT: True, robots.RIGHT: True}
    controller._latest_targets = {robots.LEFT: object(), robots.RIGHT: object()}
    controller._latest_target_keys = {robots.LEFT: object(), robots.RIGHT: object()}
    controller._active_arm_goals = {robots.LEFT: object(), robots.RIGHT: object()}
    controller._arm_goal_dirty = {robots.LEFT: True, robots.RIGHT: True}
    controller._accept_cartesian_targets_after_s = {robots.LEFT: 0.0, robots.RIGHT: 0.0}
    controller._reset_approx_ik_cache = lambda: None
    controller._reset_ik_recovery_state = lambda: None
    controller._last_safety_log_time = {}
    controller._pending_ahawam_rebaseline_id = None
    controller._ahawam_rebaseline_target_ready = {
        robots.LEFT: True,
        robots.RIGHT: True,
    }

    controller._handle_ahawam_rebaseline_request()

    assert controller._pending_ahawam_rebaseline_id == 41
    assert controller._hand_init_ready == {robots.LEFT: False, robots.RIGHT: False}
    assert controller._ahawam_rebaseline_target_ready == {
        robots.LEFT: False,
        robots.RIGHT: False,
    }
    assert [message[2] for message in controller._publisher_manager.messages] == [
        "ahawam_rebaseline",
        "ahawam_rebaseline",
    ]
    controller._hand_init_ready[robots.LEFT] = True
    controller._ack_ahawam_rebaseline_if_ready()
    assert controller._ros2.acks == []
    controller._hand_init_ready[robots.RIGHT] = True
    controller._ack_ahawam_rebaseline_if_ready()
    assert controller._ros2.acks == []
    controller._ahawam_rebaseline_target_ready = {
        robots.LEFT: True,
        robots.RIGHT: True,
    }
    controller._ack_ahawam_rebaseline_if_ready()
    assert controller._ros2.acks == [41]
    assert controller._pending_ahawam_rebaseline_id is None


def test_ahawam_neutral_targets_anchor_measured_joints_without_ik():
    left_target = CartesianTarget(
        timestamp_s=11.0,
        hand_side=robots.LEFT,
        frame_id="base",
        position_m=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    right_target = CartesianTarget(
        timestamp_s=12.0,
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.4, 0.5, 0.6),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    class OneShotSubscriber:
        def __init__(self, target):
            self.target = target

        def recv_keypoints(self):
            target, self.target = self.target, None
            return target

    snapshot = FaJointStateSnapshot(
        timestamp_s=10.0,
        left_arm=tuple(np.linspace(-0.7, -0.1, 7)),
        right_arm=tuple(np.linspace(0.1, 0.7, 7)),
        neck=(0.0, 0.0),
    )
    controller = fa_real_control_module.FaRealControl.__new__(
        fa_real_control_module.FaRealControl
    )
    controller._left_target_subscriber = OneShotSubscriber(left_target)
    controller._right_target_subscriber = OneShotSubscriber(right_target)
    controller._hand_init_ready = {robots.LEFT: True, robots.RIGHT: True}
    controller._accept_cartesian_targets_after_s = {robots.LEFT: 1.0, robots.RIGHT: 1.0}
    controller._pending_ahawam_rebaseline_id = 99
    controller._ahawam_rebaseline_target_ready = {robots.LEFT: False, robots.RIGHT: False}
    controller._ahawam_neutral_candidate_streaming = {robots.LEFT: False, robots.RIGHT: False}
    controller._latest_targets = {robots.LEFT: None, robots.RIGHT: None}
    controller._latest_target_keys = {robots.LEFT: None, robots.RIGHT: None}
    controller._last_ik_cartesian_targets = {robots.LEFT: None, robots.RIGHT: None}
    controller._last_safe_arm_targets = {robots.LEFT: None, robots.RIGHT: None}
    controller._active_arm_goals = {robots.LEFT: None, robots.RIGHT: None}
    controller._arm_goal_dirty = {robots.LEFT: True, robots.RIGHT: True}
    controller._arm_ik_retry_required = {robots.LEFT: True, robots.RIGHT: True}
    controller._resume_hold_until_target = True
    controller._current_joint_snapshot = lambda: snapshot
    controller._real_joint_state_fresh = lambda: True
    controller._warn_safety = lambda *_args, **_kwargs: None
    controller._publish_hand_command_on_edge = lambda *_args, **_kwargs: pytest.fail(
        "neutral hand command must not be forwarded during arming"
    )

    controller._receive_cartesian_targets()

    assert controller._ahawam_rebaseline_target_ready == {
        robots.LEFT: True,
        robots.RIGHT: True,
    }
    assert controller._last_ik_cartesian_targets == {
        robots.LEFT: left_target,
        robots.RIGHT: right_target,
    }
    assert controller._active_arm_goals[robots.LEFT] == pytest.approx(snapshot.left_arm)
    assert controller._active_arm_goals[robots.RIGHT] == pytest.approx(snapshot.right_arm)
    assert controller._arm_goal_dirty == {robots.LEFT: False, robots.RIGHT: False}
    assert controller._arm_ik_retry_required == {robots.LEFT: False, robots.RIGHT: False}
    assert controller._ahawam_neutral_candidate_streaming == {
        robots.LEFT: True,
        robots.RIGHT: True,
    }


def test_operator_ahawam_rebaseline_becomes_fresh_resume_edge(monkeypatch):
    operator = XArmOperator.__new__(XArmOperator)
    operator.operator_name = "test_left_operator"
    operator._ahawam_rebaseline_subscriber = SimpleNamespace(
        recv_keypoints=lambda: SessionCommand(timestamp_s=10.0, command=robots.RESUME)
    )
    operator._last_teleop_command = None
    operator._last_teleop_command_timestamp_s = 0.0
    operator.is_first_frame = False
    operator._ignore_hand_frames_before_s = 0.0
    cleared = []
    operator._clear_hand_tracking_cache = lambda: cleared.append(True)
    monkeypatch.setattr(
        "beavr.teleop.components.operator.robots.xarm7_operator.time.time",
        lambda: 20.0,
    )

    assert operator._consume_ahawam_rebaseline()
    # Intervention must not masquerade as a new RESUME: FaOperator interprets
    # RESUME as a request to repeat the full VR axis calibration.
    assert operator._last_teleop_command is None
    assert operator._last_teleop_command_timestamp_s == 0.0
    assert operator._ignore_hand_frames_before_s == 20.0
    assert operator.is_first_frame
    assert operator._post_reset_hand_rebaseline_dwell_override_s == 2.0
    assert cleared == [True]
