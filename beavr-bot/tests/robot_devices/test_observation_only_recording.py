import pytest
import torch

from beavr.lerobot.common.robot_devices.control_utils import control_loop
from beavr.lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig, OpenCVCameraConfig
from beavr.lerobot.common.robot_devices.robots.configs import FaAdapterConfig, Sysmo32AdapterConfig
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


def test_fa_adapter_records_16d_state_with_o6_switches():
    cfg = FaAdapterConfig(cameras={})
    robot = make_robot_from_config(cfg)

    assert robot.features["observation.state"]["shape"] == (16,)
    assert robot.features["observation.state"]["names"][-2:] == [
        "left_hand_gripper_state",
        "right_hand_gripper_state",
    ]
    assert robot.features["action"]["shape"] == (16,)


def test_fa_adapter_default_camera_name_matches_existing_dataset_info():
    cfg = FaAdapterConfig()
    robot = make_robot_from_config(cfg)

    assert list(robot.camera_features) == ["observation.images.front"]


def test_fa_adapter_accepts_up_to_three_recording_cameras():
    cfg = FaAdapterConfig(
        cameras={
            "front": OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=480),
            "left_wrist": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480),
            "right_left": IntelRealSenseCameraConfig(serial_number=123456, fps=30, width=640, height=480),
        }
    )
    robot = make_robot_from_config(cfg)

    assert list(robot.camera_features) == [
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_left",
    ]


def test_fa_adapter_rejects_more_than_three_recording_cameras():
    cfg = FaAdapterConfig(
        cameras={
            f"cam_{idx}": OpenCVCameraConfig(camera_index=idx, fps=30, width=640, height=480)
            for idx in range(4)
        }
    )

    with pytest.raises(ValueError, match="at most 3 cameras"):
        make_robot_from_config(cfg)


def test_sysmo32_adapter_receives_and_builds_bimanual_action(monkeypatch, bus):
    from tests.conftest import FakeZMQSubscriber

    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.robots.beavrbot.ZMQSubscriber",
        FakeZMQSubscriber,
    )
    cfg = Sysmo32AdapterConfig(cameras={})
    robot = make_robot_from_config(cfg)
    robot.connect()

    right_state_0 = [1, 2, 3, 4, 5, 6]
    left_state_0 = [7, 8, 9, 10, 11, 12]
    right_state_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    left_state_1 = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 2,
        "sysmo32_right",
        {
            "joint_states": {"joint_position": right_state_0},
        },
    )
    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 4,
        "sysmo32_left",
        {
            "joint_states": {"joint_position": left_state_0},
        },
    )

    observation, action = robot.teleop_step(record_data=True)

    assert observation is not None
    assert observation["observation.state"].tolist() == left_state_0 + right_state_0
    assert action is None

    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 2,
        "sysmo32_right",
        {
            "joint_states": {"joint_position": right_state_1},
        },
    )
    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 4,
        "sysmo32_left",
        {
            "joint_states": {"joint_position": left_state_1},
        },
    )

    observation, action = robot.teleop_step(record_data=True)

    assert observation is not None
    assert observation["observation.state"].tolist() == left_state_0 + right_state_0
    assert action is not None
    assert action["action"].tolist() == pytest.approx(left_state_1 + right_state_1)


def test_fa_adapter_records_bimanual_o6_gripper_state(monkeypatch, bus):
    from tests.conftest import FakeZMQSubscriber

    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.robots.beavrbot.ZMQSubscriber",
        FakeZMQSubscriber,
    )
    cfg = FaAdapterConfig(cameras={}, record_actions=False)
    robot = make_robot_from_config(cfg)
    robot.connect()

    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 4,
        "fa_left",
        {
            "joint_states": {"joint_position": [0, 0, 0, 0, 0, 0, 0]},
            "hand_command": 2,
        },
    )
    bus.publish(
        "127.0.0.1",
        ports.XARM_STATE_PUBLISH_PORT + 2,
        "fa_right",
        {
            "joint_states": {"joint_position": [0, 0, 0, 0, 0, 0, 0]},
            "hand_command": 1,
        },
    )

    observation = robot.capture_observation()

    assert robot.features["observation.state"]["names"][-2:] == [
        "left_hand_gripper_state",
        "right_hand_gripper_state",
    ]
    assert observation["observation.state"].tolist()[-2:] == [1.0, 0.0]


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


def test_orbbec_depth_uses_one_rgbd_pair_and_image_storage(monkeypatch):
    import numpy as np
    from beavr.lerobot.common.datasets.utils import get_features_from_robot
    from beavr.lerobot.common.robot_devices.cameras.configs import OrbbecCameraConfig

    robot = make_robot_from_config(FaAdapterConfig(cameras={
        'camera_0': OrbbecCameraConfig(serial_number='first'),
        'camera_1': OrbbecCameraConfig(serial_number='second'),
    }))
    features = get_features_from_robot(robot, use_videos=True)
    for name, camera in robot.cameras.items():
        assert features[f'observation.images.{name}']['dtype'] == 'video'
        assert features[f'observation.images.{name}_depth']['dtype'] == 'image'
        assert features[f'observation.images.{name}_depth']['shape'] == (480, 640, 1)
        rgb = np.full((480, 640, 3), 100, dtype=np.uint8)
        depth = np.full((480, 640), 1234, dtype=np.uint16)
        monkeypatch.setattr(camera, 'read_latest_rgbd', lambda max_age_ms: (rgb, depth))
        monkeypatch.setattr(camera, 'async_read', lambda: pytest.fail('Must fetch RGB-D atomically'))
    monkeypatch.setattr(robot, '_is_connected', True)
    observation = robot.capture_observation()
    for name in robot.cameras:
        assert observation[f'observation.images.{name}_depth'].numpy()[0, 0, 0] == 1234


def test_read_only_recorder_never_constructs_control_publishers(monkeypatch):
    from beavr.lerobot.common.robot_devices.robots import beavrbot

    def forbidden(*args, **kwargs):
        pytest.fail('A recording process must not create control publishers or handshakes')

    monkeypatch.setattr(beavrbot.ZMQPublisherManager, 'get_instance', forbidden)
    monkeypatch.setattr(beavrbot.HandshakeCoordinator, 'get_instance', forbidden)
    robot = make_robot_from_config(Sysmo32AdapterConfig(cameras={}, read_only=True))
    robot.connect()
    robot.manage_teleop_state = True  # read_only must override even an accidental toggle.
    robot.teleop_resume()
    robot.teleop_stop()
    with pytest.raises(RuntimeError, match='cannot send'):
        robot.send_action([0] * 12)
    robot.disconnect()
    assert robot.pub_manager is None
    assert robot._publish_pool is None


def test_read_only_rejects_missing_and_stale_states(monkeypatch):
    import time
    robot = make_robot_from_config(Sysmo32AdapterConfig(cameras={}, read_only=True))
    robot.connect()
    with pytest.raises(RuntimeError, match='missing/stale'):
        robot.capture_observation()
    with pytest.raises(TimeoutError, match='No fresh'):
        robot.wait_for_recording_states(timeout_s=0)
    stale = {c['name']: {'timestamp': time.time() - 10,
                        'joint_states': {'joint_position': [0] * 6}}
             for c in robot.robot_configs}
    monkeypatch.setattr(robot, '_fetch_robot_states', lambda: stale)
    with pytest.raises(RuntimeError, match='missing/stale'):
        robot.capture_observation()
    robot.disconnect()


def test_read_only_next_state_action_and_episode_boundary(monkeypatch):
    import time
    robot = make_robot_from_config(Sysmo32AdapterConfig(cameras={}, read_only=True))
    robot.connect()
    states = {c['name']: {'timestamp': time.time(), 'joint_states': {'joint_position': [0.1] * 6}}
              for c in robot.robot_configs}
    monkeypatch.setattr(robot, '_fetch_robot_states', lambda: states)
    assert robot.teleop_step(record_data=True)[1] is None
    for value in states.values():
        value['joint_states']['joint_position'] = [0.2] * 6
    observation, action = robot.teleop_step(record_data=True)
    assert torch.allclose(observation['observation.state'], torch.full((12,), 0.1))
    assert (action['action'] == pytest.approx([0.2] * 12))
    robot.reset_recording()
    assert robot.teleop_step(record_data=True)[1] is None
    robot.disconnect()
