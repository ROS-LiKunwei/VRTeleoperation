import pytest
import torch

from beavr.lerobot.common.robot_devices.control_utils import control_loop
from beavr.lerobot.common.robot_devices.robots.configs import Sysmo32AdapterConfig
from beavr.lerobot.common.robot_devices.robots.utils import make_robot_from_config
from beavr.teleop.configs.constants import ports


def test_sysmo32_adapter_can_disable_action_recording():
    cfg = Sysmo32AdapterConfig(cameras={}, record_actions=False)
    robot = make_robot_from_config(cfg)

    assert "observation.state" in robot.features
    assert "action" not in robot.features


def test_sysmo32_adapter_defaults_record_images_and_action():
    cfg = Sysmo32AdapterConfig()
    robot = make_robot_from_config(cfg)

    assert "observation.state" in robot.features
    assert "observation.images.front" in robot.features
    assert "action" in robot.features


def test_sysmo32_adapter_receives_and_builds_bimanual_action(monkeypatch, bus):
    from tests.conftest import FakeZMQSubscriber

    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.robots.beavrbot.ZMQSubscriber",
        FakeZMQSubscriber,
    )
    cfg = Sysmo32AdapterConfig(cameras={})
    robot = make_robot_from_config(cfg)
    robot.connect()

    right_action = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    left_action = [-0.1, -0.2, 0.4, 0.0, 0.0, 0.707, 0.707]
    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 2,
        "sysmo32_right",
        {
            "joint_states": {"joint_position": [1, 2, 3, 4, 5, 6]},
            "commanded_cartesian_state": {"commanded_cartesian_position": right_action},
        },
    )
    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 4,
        "sysmo32_left",
        {
            "joint_states": {"joint_position": [7, 8, 9, 10, 11, 12]},
            "commanded_cartesian_state": {"commanded_cartesian_position": left_action},
        },
    )

    observation, action = robot.teleop_step(record_data=True)

    assert observation is not None
    assert observation["observation.state"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert action is not None
    assert action["action"].tolist() == pytest.approx(right_action + left_action)


def test_control_loop_records_observation_only_dataset(monkeypatch):
    class FakeRobot:
        is_connected = True
        robot_type = "sysmo32"

        def teleop_step(self, record_data=False):
            return {"observation.state": torch.zeros(12)}, None

    class FakeDataset:
        fps = 30
        features = {"observation.state": {"shape": (12,), "dtype": "float32"}}

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

    dataset = FakeDataset()
    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.control_utils.log_control_info",
        lambda *args, **kwargs: None,
    )

    control_loop(
        robot=FakeRobot(),
        control_time_s=0.01,
        teleoperate=True,
        dataset=dataset,
        events={"exit_early": False},
        fps=None,
        single_task="record joints",
    )

    assert dataset.frames
    assert "observation.state" in dataset.frames[0]
    assert "action" not in dataset.frames[0]
    assert dataset.frames[0]["task"] == "record joints"


def test_control_loop_honors_exit_when_required_action_is_missing(monkeypatch):
    class FakeRobot:
        is_connected = True
        robot_type = "sysmo32"

        def teleop_step(self, record_data=False):
            return {"observation.state": torch.zeros(12)}, None

    class FakeDataset:
        fps = 30
        features = {
            "observation.state": {"shape": (12,), "dtype": "float32"},
            "action": {"shape": (14,), "dtype": "float32"},
        }

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

    dataset = FakeDataset()
    events = {"exit_early": True}
    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.control_utils.log_control_info",
        lambda *args, **kwargs: None,
    )

    control_loop(
        robot=FakeRobot(),
        control_time_s=10.0,
        teleoperate=True,
        dataset=dataset,
        events=events,
        fps=None,
        single_task="record joints",
    )

    assert dataset.frames == []
    assert events["exit_early"] is False
