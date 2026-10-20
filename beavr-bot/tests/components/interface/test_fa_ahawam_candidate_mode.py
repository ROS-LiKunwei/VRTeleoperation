import builtins
from types import SimpleNamespace

import draccus
import pytest

from beavr.teleop.common.configs.loader import Laterality, load_robot_config
from beavr.teleop.components.interface.robots import fa_real_control as fa_real_control_module
from beavr.teleop.components.interface.robots.fa_real_control import (
    FA_AHAWAM_HUMAN_LEFT_HAND_TOPIC,
    FA_AHAWAM_HUMAN_MIN_SNAP_TARGET_TOPIC,
    FA_AHAWAM_HUMAN_RIGHT_HAND_TOPIC,
    FA_REAL_LEFT_HAND_TOPIC,
    FA_REAL_MIN_SNAP_TARGET_TOPIC,
    FA_REAL_RIGHT_HAND_TOPIC,
    FaRealControlConfig,
    FaRos2Topics,
)
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
    ]
    assert FA_REAL_MIN_SNAP_TARGET_TOPIC not in min_snap_topics
    assert FA_REAL_LEFT_HAND_TOPIC not in hand_topics
    assert FA_REAL_RIGHT_HAND_TOPIC not in hand_topics
    assert subscriptions == ["/joint_states"]
