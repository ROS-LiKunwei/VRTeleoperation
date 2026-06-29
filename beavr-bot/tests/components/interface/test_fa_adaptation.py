import csv
import numpy as np
import pytest
import sys
from types import SimpleNamespace

from beavr.lerobot.common.robot_devices.robots.configs import FaAdapterConfig
from beavr.lerobot.common.robot_devices.robots.utils import make_robot_config
from beavr.teleop.common.configs.loader import Laterality, load_robot_config
from beavr.teleop.components.interface.robots.fa_arm_ik_client import (
    FaArmIkConfig,
    FaArmIkResult,
    FaPybindIkClient,
)
from beavr.teleop.components.interface.robots.fa_command_builder import (
    FA_LEFT_ARM_JOINT_NAMES,
    FA_NECK_JOINT_NAMES,
    FA_RIGHT_ARM_JOINT_NAMES,
    FA_UPPER_COMMAND_LENGTH,
    FaJointStateCache,
    FaJointStateSnapshot,
    FaUpperPositionCommand,
    FaUpperPositionCommandBuilder,
)
from beavr.teleop.components.interface.robots.fa_mujoco_kinematics import (
    FaKinematicsConfig,
    FaMujocoKinematics,
)
from beavr.teleop.components.interface.robots.fa_real_control import FaRealControl, FaRealControlConfig
from beavr.teleop.components.interface.robots import fa_real_control as fa_real_control_module
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.operator.robots.fa_operator import H_R_V_FA
from beavr.teleop.components.operator.robots.sysmo32_operator import H_R_V_SYSMO32
from beavr.teleop.components.simulation.sysmo32_mujoco_command_sim import (
    Sysmo32MujocoCommandMirror,
)
from beavr.teleop.configs.constants import ports, robots


def test_fa_config_routes_backends():
    mujoco_cfg = load_robot_config("fa", Laterality.BIMANUAL, True, control_backend="mujoco")
    assert mujoco_cfg.robot_name == "fa"
    assert mujoco_cfg.control_backend == "mujoco"
    assert mujoco_cfg.model_root.endswith("robots/fa_description")
    assert mujoco_cfg.left_arm_joint_count == 7
    assert mujoco_cfg.right_arm_joint_count == 7
    assert len(mujoco_cfg.robots) == 1
    assert len(mujoco_cfg.operators) == 2
    assert len(mujoco_cfg.environment) == 1
    robot_cfg = mujoco_cfg.robots[0]
    assert isinstance(robot_cfg.config, FaRealControlConfig)
    assert robot_cfg.config.ros2.joint_state_topic == "/joint_states"
    assert robot_cfg.config.ros2.upper_position_command_topic == "/upper_position_controller/commands"
    assert mujoco_cfg.upper_position_command_topic == "/upper_position_controller/commands"
    assert mujoco_cfg.upper_position_command_size == 16
    assert robot_cfg.config.command_publish_hz == pytest.approx(1000.0)
    assert robot_cfg.config.upper.max_joint_velocity_rad_s == pytest.approx(tuple([1.2] * 16))
    assert robot_cfg.config.ik.module_name == "ik_7dof_pybind"
    assert robot_cfg.config.ik.max_iters == 200
    assert robot_cfg.config.ik.eps == pytest.approx(1e-3)
    assert mujoco_cfg.environment[0].ros_arm_command_topic == "/upper_position_controller/commands"
    assert mujoco_cfg.environment[0].arm_command_source == "ros2"
    assert mujoco_cfg.environment[0].kinematics_type == "fa"
    assert mujoco_cfg.environment[0].interpolation_profile == "linear"
    assert mujoco_cfg.environment[0].publish_joint_states
    assert mujoco_cfg.environment[0].expected_command_length == 16
    assert mujoco_cfg.environment[0].joint_state_joint_names == FA_LEFT_ARM_JOINT_NAMES + FA_RIGHT_ARM_JOINT_NAMES

    real_cfg = load_robot_config("fa", Laterality.BIMANUAL, False, control_backend="real")
    assert real_cfg.control_backend == "real"
    assert real_cfg.environment == []
    assert real_cfg.robots[0].right_target_port == ports.XARM_ENDEFF_SUBSCRIBE_PORT + 2
    assert real_cfg.robots[0].left_target_port == ports.XARM_ENDEFF_SUBSCRIBE_PORT + 4
    assert real_cfg.operators[0].hand_side == robots.RIGHT
    assert real_cfg.operators[1].hand_side == robots.LEFT

    mirror_cfg = load_robot_config("fa", Laterality.BIMANUAL, True, control_backend="real_with_mujoco")
    assert not mirror_cfg.robots[0].config.allow_mujoco_mirror_without_joint_state
    assert mirror_cfg.environment[0].arm_command_source == "ros2"
    assert mirror_cfg.environment[0].kinematics_type == "fa"
    assert mirror_cfg.environment[0].interpolation_profile == "linear"
    assert not mirror_cfg.environment[0].publish_joint_states


def test_fa_mujoco_initial_pose_waits_for_joint_state():
    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig(
        initial_pose_enabled=True,
        initial_left_arm_positions_rad=tuple([0.1] * 7),
        initial_right_arm_positions_rad=tuple([0.2] * 7),
    )
    controller.control_backend = "mujoco"
    controller._startup_initial_pose_armed = True
    controller._initial_pose_done = False
    controller._initial_pose_started_at_s = None
    controller._real_joint_state_fresh = lambda: False
    warnings = []
    controller._warn_safety = lambda key, message: warnings.append((key, message))
    controller._current_joint_snapshot = lambda: (_ for _ in ()).throw(AssertionError("snapshot should not be read"))

    controller._publish_initial_pose_if_needed()

    assert controller._initial_pose_started_at_s is None
    assert warnings == [("initial_pose_joint_state_stale", "FA initial pose held until fresh /joint_states")]


def test_fa_real_with_mujoco_initial_pose_publishes_min_snap_target():
    class FakeRos2:
        def __init__(self):
            self.min_snap_commands = []

        def publish_upper_position_command(self, command):
            raise AssertionError("FA modes must not publish native upper commands directly")

        def publish_min_snap_target(self, command, expected_duration_s, max_velocity_rad_s, max_acceleration_rad_s2):
            self.min_snap_commands.append(
                (command, expected_duration_s, max_velocity_rad_s, max_acceleration_rad_s2)
            )
            return True

    class ResetRecorder:
        def __init__(self):
            self.values = []

        def reset(self, value):
            self.values.append(tuple(value))

    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig(
        control_backend="real_with_mujoco",
        initial_pose_enabled=True,
        initial_left_arm_positions_rad=tuple([0.1] * 7),
        initial_right_arm_positions_rad=tuple([0.2] * 7),
    )
    controller.control_backend = "real_with_mujoco"
    controller._startup_initial_pose_armed = True
    controller._initial_pose_done = False
    controller._initial_pose_started_at_s = None
    controller._ros2 = FakeRos2()
    controller._builder = FaUpperPositionCommandBuilder()
    controller._limiter = ResetRecorder()
    controller._real_joint_state_fresh = lambda: True
    controller._warn_safety = lambda key, message: (_ for _ in ()).throw(AssertionError(message))
    snapshot = FaJointStateSnapshot(
        timestamp_s=1.0,
        left_arm=tuple([0.0] * 7),
        right_arm=tuple([0.0] * 7),
        neck=(0.3, 0.4),
    )
    controller._current_joint_snapshot = lambda: snapshot

    controller._publish_initial_pose_if_needed()

    assert len(controller._ros2.min_snap_commands) == 1
    command, duration, max_velocity, max_acceleration = controller._ros2.min_snap_commands[0]
    assert command.left_arm == pytest.approx(tuple([0.1] * 7))
    assert command.right_arm == pytest.approx(tuple([0.2] * 7))
    assert command.neck == pytest.approx((0.3, 0.4))
    assert duration == pytest.approx(controller.config.initial_pose_duration_s)
    assert max_velocity == pytest.approx(controller.config.initial_pose_max_velocity_rad_s)
    assert max_acceleration == pytest.approx(controller.config.initial_pose_max_acceleration_rad_s2)
    assert controller._last_published_upper_command is command
    assert controller._initial_pose_started_at_s is not None
    assert controller._active_arm_goals[robots.LEFT] == pytest.approx(np.asarray([0.1] * 7))
    assert controller._active_arm_goals[robots.RIGHT] == pytest.approx(np.asarray([0.2] * 7))


def test_fa_real_with_mujoco_initial_pose_republishes_during_startup_window(monkeypatch):
    class FakeRos2:
        def __init__(self):
            self.min_snap_commands = []

        def publish_upper_position_command(self, command):
            raise AssertionError("FA modes must not publish native upper commands directly")

        def publish_min_snap_target(self, command, expected_duration_s, max_velocity_rad_s, max_acceleration_rad_s2):
            self.min_snap_commands.append(
                (command, expected_duration_s, max_velocity_rad_s, max_acceleration_rad_s2)
            )
            return True

    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig(
        control_backend="real_with_mujoco",
        initial_pose_enabled=True,
        initial_left_arm_positions_rad=tuple([0.1] * 7),
        initial_right_arm_positions_rad=tuple([0.2] * 7),
        pause_hold_heartbeat_hz=20.0,
    )
    controller.control_backend = "real_with_mujoco"
    controller._startup_initial_pose_armed = True
    controller._initial_pose_done = False
    controller._initial_pose_started_at_s = 10.0
    controller._last_initial_pose_publish_time_s = 10.0
    controller._last_published_upper_command = FaUpperPositionCommand(
        timestamp_s=10.0,
        values=tuple([0.1] * 7 + [0.2] * 7 + [0.3, 0.4]),
    )
    controller._ros2 = FakeRos2()
    controller._warn_safety = lambda key, message: (_ for _ in ()).throw(AssertionError(message))

    monkeypatch.setattr(fa_real_control_module.time, "time", lambda: 10.02)
    controller._publish_initial_pose_if_needed()
    assert controller._ros2.min_snap_commands == []

    monkeypatch.setattr(fa_real_control_module.time, "time", lambda: 10.06)
    controller._publish_initial_pose_if_needed()

    assert len(controller._ros2.min_snap_commands) == 1
    command = controller._ros2.min_snap_commands[0][0]
    assert command.left_arm == pytest.approx(tuple([0.1] * 7))
    assert command.right_arm == pytest.approx(tuple([0.2] * 7))


def test_fa_initial_pose_ready_logs_teleop_prompt_once(monkeypatch):
    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig(initial_pose_duration_s=5.0)
    controller._initial_pose_done = False
    controller._initial_pose_started_at_s = 10.0
    controller._teleop_ready_prompt_logged = False
    messages = []
    monkeypatch.setattr(
        fa_real_control_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )

    assert controller._initial_pose_ready(now_s=15.0)
    assert controller._initial_pose_ready(now_s=16.0)

    assert messages.count("FA 已到达准备位置，可以开始遥操作。") == 1


def test_fa_upper_position_command_builder_maps_7d_arms_and_neck_to_16d_payload():
    builder = FaUpperPositionCommandBuilder()
    command = builder.build(
        left_arm_joints_rad=[0, 1, 2, 3, 4, 5, 6],
        right_arm_joints_rad=[10, 11, 12, 13, 14, 15, 16],
        neck_joints_rad=[20, 21],
        timestamp_s=1.0,
    )
    assert len(command.values) == FA_UPPER_COMMAND_LENGTH
    assert command.left_arm == (0, 1, 2, 3, 4, 5, 6)
    assert command.right_arm == (10, 11, 12, 13, 14, 15, 16)
    assert command.neck == (20, 21)
    assert command.values[0:7] == command.left_arm
    assert command.values[7:14] == command.right_arm
    assert command.values[14:16] == command.neck

    with pytest.raises(ValueError, match="7 finite joint values"):
        builder.build([0] * 6, [0] * 7, [0] * 2)
    with pytest.raises(ValueError, match="length"):
        FaUpperPositionCommand(timestamp_s=1.0, values=tuple([0.0] * 15))


def test_fa_joint_state_cache_parses_by_name_not_index():
    names = list(FA_NECK_JOINT_NAMES + FA_RIGHT_ARM_JOINT_NAMES + FA_LEFT_ARM_JOINT_NAMES)
    positions = [100.0, 101.0] + list(range(10, 17)) + list(range(0, 7))
    msg = SimpleNamespace(name=names, position=positions)
    cache = FaJointStateCache(joint_state_timeout_s=1.0)
    snapshot = cache.update_from_joint_state_msg(msg, now_s=10.0)
    assert snapshot.left_arm == tuple(range(0, 7))
    assert snapshot.right_arm == tuple(range(10, 17))
    assert snapshot.neck == (100.0, 101.0)
    assert cache.is_fresh(now_s=10.5)
    assert not cache.is_fresh(now_s=12.0)

    missing = SimpleNamespace(name=names[:-1], position=positions[:-1])
    assert cache.update_from_joint_state_msg(missing, now_s=13.0) is snapshot
    assert cache.last_missing_joint is not None


def test_fa_ik_result_requires_7d_and_pybind_client_calls_solver(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="7 values"):
        FaArmIkResult(success=True, q_target=tuple([0.0] * 6))

    class FakeSolver:
        def __init__(self, urdf_file, srdf_file=""):
            self.urdf_file = urdf_file
            self.srdf_file = srdf_file

        def solve_arm_ik(self, translation, rotation, arm_side, initial_q, reference_frame, max_iters, eps):
            assert np.allclose(translation, [0.1, 0.2, 0.3])
            assert np.allclose(rotation, np.eye(3))
            assert arm_side == robots.LEFT
            assert initial_q == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            assert reference_frame == "pelvis"
            assert max_iters == 9
            assert eps == pytest.approx(1e-4)
            return {
                "success": True,
                "has_solution": True,
                "q_target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "position_error": 0.001,
                "orientation_error": 0.002,
                "iterations": 3,
                "solve_time_ms": 0.4,
            }

        def compute_arm_fk(self, q, arm_side, reference_frame):
            assert q == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            assert arm_side == robots.LEFT
            assert reference_frame == "pelvis"
            return {"translation": np.array([0.1, 0.2, 0.3]), "rotation": np.eye(3)}

    monkeypatch.setitem(sys.modules, "fake_ik_7dof_pybind", SimpleNamespace(FaIkSolver=FakeSolver))
    client = FaPybindIkClient(
        FaArmIkConfig(
            urdf_file="/home/likunwei/dataCollection/beavr-bot/robots/fa_description/urdf/fa_robot.urdf",
            module_name="fake_ik_7dof_pybind",
            max_iters=9,
            eps=1e-4,
            log_dir=str(tmp_path),
        )
    )
    target = CartesianTarget(
        timestamp_s=1.0,
        hand_side=robots.LEFT,
        frame_id="base",
        position_m=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    result = client.solve(robots.LEFT, target, [1, 2, 3, 4, 5, 6, 7])
    assert result.success
    assert result.has_solution
    assert result.q_target == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    log_files = list(tmp_path.glob("fa_ik_*_pid*.csv"))
    assert len(log_files) == 1
    with log_files[0].open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["hand_side"] == robots.LEFT
    assert rows[0]["frame_id"] == "base"
    assert rows[0]["success"] == "1"
    assert rows[0]["has_solution"] == "1"
    assert rows[0]["position_error_m"] == "0.001"
    assert rows[0]["orientation_error_rad"] == "0.002"
    assert rows[0]["q_target6"] == "0.7"
    fk = client.compute_fk(robots.LEFT, [1, 2, 3, 4, 5, 6, 7])
    assert np.allclose(fk[:3, 3], [0.1, 0.2, 0.3])


def test_fa_ik_accepts_approximate_solution_when_position_and_orientation_within_threshold(monkeypatch):
    class FakeSolver:
        def __init__(self, urdf_file, srdf_file=""):
            pass

        def solve_arm_ik(self, translation, rotation, arm_side, initial_q, reference_frame, max_iters, eps):
            return {
                "success": False,
                "has_solution": True,
                "q_target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "position_error": 0.049,
                "orientation_error": 0.49,
                "iterations": 200,
                "solve_time_ms": 5.0,
            }

    monkeypatch.setitem(sys.modules, "fake_approx_ik_7dof_pybind", SimpleNamespace(FaIkSolver=FakeSolver))
    client = FaPybindIkClient(
        FaArmIkConfig(
            urdf_file="/tmp/fa.urdf",
            module_name="fake_approx_ik_7dof_pybind",
            log_enabled=False,
        )
    )
    target = CartesianTarget(
        timestamp_s=1.0,
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    result = client.solve(robots.RIGHT, target, [0, 0, 0, 0, 0, 0, 0])

    assert result.success
    assert result.has_solution
    assert result.message == ""
    assert result.q_target == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))


@pytest.mark.parametrize(
    ("position_error", "orientation_error"),
    [
        (0.051, 0.49),
        (0.049, 0.51),
    ],
)
def test_fa_ik_uses_best_approximation_when_position_or_orientation_exceeds_threshold(
    monkeypatch, position_error, orientation_error
):
    class FakeSolver:
        def __init__(self, urdf_file, srdf_file=""):
            pass

        def solve_arm_ik(self, translation, rotation, arm_side, initial_q, reference_frame, max_iters, eps):
            return {
                "success": False,
                "has_solution": True,
                "q_target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "position_error": position_error,
                "orientation_error": orientation_error,
                "iterations": 200,
                "solve_time_ms": 5.0,
            }

    monkeypatch.setitem(sys.modules, "fake_reject_ik_7dof_pybind", SimpleNamespace(FaIkSolver=FakeSolver))
    client = FaPybindIkClient(
        FaArmIkConfig(
            urdf_file="/tmp/fa.urdf",
            module_name="fake_reject_ik_7dof_pybind",
            log_enabled=False,
        )
    )
    target = CartesianTarget(
        timestamp_s=1.0,
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    result = client.solve(robots.RIGHT, target, [0, 0, 0, 0, 0, 0, 0])

    assert result.success
    assert result.has_solution
    assert result.message == "using best approximate IK outside the usable position/orientation threshold"
    assert result.q_target == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))


def test_fa_ik_rejects_when_no_usable_q_target(monkeypatch):
    class FakeSolver:
        def __init__(self, urdf_file, srdf_file=""):
            pass

        def solve_arm_ik(self, translation, rotation, arm_side, initial_q, reference_frame, max_iters, eps):
            return {
                "success": False,
                "has_solution": False,
                "q_target": [],
                "position_error": 0.011,
                "orientation_error": 0.49,
                "iterations": 200,
                "solve_time_ms": 5.0,
            }

    monkeypatch.setitem(sys.modules, "fake_reject_ik_7dof_pybind", SimpleNamespace(FaIkSolver=FakeSolver))
    client = FaPybindIkClient(
        FaArmIkConfig(
            urdf_file="/tmp/fa.urdf",
            module_name="fake_reject_ik_7dof_pybind",
            log_enabled=False,
        )
    )
    target = CartesianTarget(
        timestamp_s=1.0,
        hand_side=robots.RIGHT,
        frame_id="base",
        position_m=(0.1, 0.2, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    result = client.solve(robots.RIGHT, target, [0, 0, 0, 0, 0, 0, 0])

    assert not result.success
    assert not result.has_solution
    assert result.message == "ik_7dof did not return a usable q_target"


def test_fa_pelvis_uses_same_vr_mapping_as_sysmo32_base():
    np.testing.assert_allclose(H_R_V_FA, H_R_V_SYSMO32)


def test_fa_adapter_config_is_available_for_lerobot_recording():
    cfg = make_robot_config("fa_adapter", cameras={})
    assert isinstance(cfg, FaAdapterConfig)
    assert cfg.robot_type == "fa"
    assert [item["state_topic"] for item in cfg.robot_configs] == ["fa_left", "fa_right"]
    assert [item["joint_count"] for item in cfg.robot_configs] == [7, 7]


def test_fa_reset_fk_uses_single_arm_joints_for_pybind_client():
    class FakeIkClient:
        def __init__(self):
            self.calls = []

        def compute_fk(self, hand_side, current_arm_q):
            self.calls.append((hand_side, tuple(current_arm_q)))
            assert len(current_arm_q) == 7
            return np.eye(4)

    class FakePublisherManager:
        def __init__(self):
            self.messages = []

        def publish(self, host, port, topic, message):
            self.messages.append((host, port, topic, message))

    class ResetRecorder:
        def __init__(self):
            self.values = []

        def reset(self, value):
            self.values.append(value)

    controller = FaRealControl.__new__(FaRealControl)
    controller.host = "127.0.0.1"
    controller.control_backend = "mujoco"
    controller._ik_client = FakeIkClient()
    controller._kinematics = SimpleNamespace(available=False)
    controller._publisher_manager = FakePublisherManager()
    controller._limiter = ResetRecorder()
    controller._needs_reset = True
    controller._real_reset_ready = False
    snapshot = FaJointStateSnapshot(
        timestamp_s=1.0,
        left_arm=tuple(range(7)),
        right_arm=tuple(range(10, 17)),
        neck=(0.0, 0.0),
    )
    controller._current_joint_snapshot = lambda: snapshot
    controller._real_joint_state_fresh = lambda: False
    controller._warn_safety = lambda key, message: (_ for _ in ()).throw(AssertionError(message))

    controller._publish_current_endeff_homo(robots.LEFT, publish_port=1234)
    controller._publish_current_endeff_homo(robots.RIGHT, publish_port=1236)

    assert controller._ik_client.calls == [
        (robots.LEFT, tuple(range(7))),
        (robots.RIGHT, tuple(range(10, 17))),
    ]
    assert [item[2] for item in controller._publisher_manager.messages] == ["endeff_homo", "endeff_homo"]


def test_fa_real_control_ignores_duplicate_session_state_commands():
    class FakePauseSubscriber:
        def __init__(self, messages):
            self.messages = list(messages)

        def recv_keypoints(self):
            if not self.messages:
                return None
            return self.messages.pop(0)

    controller = FaRealControl.__new__(FaRealControl)
    controller.control_backend = "mujoco"
    controller._teleop_active = True
    controller._needs_reset = False
    controller._real_reset_ready = True
    controller._latest_targets = {robots.LEFT: "left_target", robots.RIGHT: "right_target"}
    controller._pause_hold_command = "hold"
    controller._enter_pause = lambda reason: (_ for _ in ()).throw(AssertionError(reason))
    controller._pause_subscriber = FakePauseSubscriber([SimpleNamespace(command=robots.RESUME)])

    controller._handle_session_command()

    assert controller._teleop_active
    assert not controller._needs_reset
    assert controller._latest_targets == {robots.LEFT: "left_target", robots.RIGHT: "right_target"}
    assert controller._pause_hold_command == "hold"

    entered_pause = []
    controller._teleop_active = False
    controller._enter_pause = lambda reason: entered_pause.append(reason)
    controller._pause_subscriber = FakePauseSubscriber([SimpleNamespace(command=robots.PAUSE)])

    controller._handle_session_command()

    assert entered_pause == []


def test_fa_pause_syncs_current_joints_as_next_ik_reference():
    class ResetRecorder:
        def __init__(self):
            self.values = []

        def reset(self, value):
            self.values.append(tuple(value))

    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig(safety_hold_arm_on_pause=True)
    controller.control_backend = "real_with_mujoco"
    controller._teleop_active = True
    controller._needs_reset = False
    controller._real_reset_ready = True
    controller._latest_targets = {robots.LEFT: object(), robots.RIGHT: object()}
    controller._latest_target_keys = {robots.LEFT: ("old",), robots.RIGHT: ("old",)}
    controller._active_arm_goals = {robots.LEFT: np.asarray([9.0] * 7), robots.RIGHT: np.asarray([9.0] * 7)}
    controller._arm_goal_dirty = {robots.LEFT: True, robots.RIGHT: True}
    controller._last_safe_arm_targets = {
        robots.LEFT: np.asarray([8.0] * 7),
        robots.RIGHT: np.asarray([8.0] * 7),
    }
    controller._last_ik_cartesian_targets = {robots.LEFT: object(), robots.RIGHT: object()}
    controller._builder = FaUpperPositionCommandBuilder()
    controller._limiter = ResetRecorder()
    controller._resume_hold_until_target = True
    controller._last_pause_hold_publish_time = 0.0
    published = []
    controller._publish_upper_command_outputs = (
        lambda command, require_real_reset=False, allow_stale_real_hold=False, **kwargs: published.append(command) or True
    )
    snapshot = FaJointStateSnapshot(
        timestamp_s=1.0,
        left_arm=tuple([0.1] * 7),
        right_arm=tuple([0.2] * 7),
        neck=(0.3, 0.4),
    )
    controller._real_joint_state_fresh = lambda: True
    controller._current_joint_snapshot = lambda: snapshot

    controller._enter_pause("unit-test pause")

    assert not controller._teleop_active
    assert controller._last_safe_arm_targets[robots.LEFT] == pytest.approx(np.asarray([0.1] * 7))
    assert controller._last_safe_arm_targets[robots.RIGHT] == pytest.approx(np.asarray([0.2] * 7))
    assert controller._last_ik_cartesian_targets == {robots.LEFT: None, robots.RIGHT: None}
    assert len(controller._limiter.values) == 1
    assert np.allclose(controller._limiter.values[0], snapshot.upper_joints)
    assert controller._pause_hold_command.left_arm == pytest.approx(tuple([0.1] * 7))
    assert controller._pause_hold_command.right_arm == pytest.approx(tuple([0.2] * 7))
    assert published
    assert not controller._resume_hold_until_target


def test_fa_resume_holds_pause_pose_until_first_valid_target():
    class FakePauseSubscriber:
        def __init__(self, messages):
            self.messages = list(messages)

        def recv_keypoints(self):
            if not self.messages:
                return None
            return self.messages.pop(0)

    class FakeTargetSubscriber:
        def __init__(self, message=None):
            self.message = message

        def recv_keypoints(self):
            message = self.message
            self.message = None
            return message

    class ResetRecorder:
        def __init__(self):
            self.values = []

        def reset(self, value):
            self.values.append(tuple(value))

    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig(safety_hold_arm_on_pause=True)
    controller.control_backend = "real_with_mujoco"
    controller._teleop_active = False
    controller._needs_reset = False
    controller._real_reset_ready = False
    controller._latest_targets = {robots.LEFT: object(), robots.RIGHT: object()}
    controller._latest_target_keys = {robots.LEFT: ("old",), robots.RIGHT: ("old",)}
    controller._active_arm_goals = {robots.LEFT: np.asarray([9.0] * 7), robots.RIGHT: np.asarray([9.0] * 7)}
    controller._arm_goal_dirty = {robots.LEFT: True, robots.RIGHT: True}
    controller._last_safe_arm_targets = {
        robots.LEFT: np.asarray([8.0] * 7),
        robots.RIGHT: np.asarray([8.0] * 7),
    }
    controller._last_ik_cartesian_targets = {robots.LEFT: object(), robots.RIGHT: object()}
    controller._builder = FaUpperPositionCommandBuilder()
    controller._limiter = ResetRecorder()
    controller._last_pause_hold_publish_time = 0.0
    controller._pause_subscriber = FakePauseSubscriber([SimpleNamespace(command=robots.RESUME)])
    controller._warn_safety = lambda key, message: (_ for _ in ()).throw(AssertionError(message))
    snapshot = FaJointStateSnapshot(
        timestamp_s=1.0,
        left_arm=tuple([0.3] * 7),
        right_arm=tuple([0.4] * 7),
        neck=(0.5, 0.6),
    )
    controller._real_joint_state_fresh = lambda: True
    controller._current_joint_snapshot = lambda: snapshot

    controller._handle_session_command()

    assert controller._teleop_active
    assert controller._needs_reset
    assert not controller._real_reset_ready
    assert controller._latest_targets == {robots.LEFT: None, robots.RIGHT: None}
    assert controller._latest_target_keys == {robots.LEFT: None, robots.RIGHT: None}
    assert controller._active_arm_goals == {robots.LEFT: None, robots.RIGHT: None}
    assert controller._arm_goal_dirty == {robots.LEFT: False, robots.RIGHT: False}
    assert controller._last_safe_arm_targets[robots.LEFT] == pytest.approx(np.asarray([0.3] * 7))
    assert controller._last_safe_arm_targets[robots.RIGHT] == pytest.approx(np.asarray([0.4] * 7))
    assert controller._last_ik_cartesian_targets == {robots.LEFT: None, robots.RIGHT: None}
    assert controller._pause_hold_command.left_arm == pytest.approx(tuple([0.3] * 7))
    assert controller._pause_hold_command.right_arm == pytest.approx(tuple([0.4] * 7))
    assert controller._resume_hold_until_target

    published = []
    controller._publish_upper_command_outputs = (
        lambda command, require_real_reset=False, allow_stale_real_hold=False, **kwargs: published.append(
            (command, require_real_reset, allow_stale_real_hold, kwargs)
        )
        or True
    )
    controller._publish_pause_hold_if_needed(force=True)

    assert published
    assert published[-1][0].left_arm == pytest.approx(tuple([0.3] * 7))
    assert published[-1][1] is False
    assert published[-1][2] is True
    assert published[-1][3]["force_min_snap_publish"] is True

    target = SimpleNamespace(hand_side=robots.LEFT, hand_command=None)
    controller._left_target_subscriber = FakeTargetSubscriber(target)
    controller._right_target_subscriber = FakeTargetSubscriber()
    controller._publish_hand_command_on_edge = lambda hand_side, hand_command: None

    controller._receive_cartesian_targets()

    assert not controller._resume_hold_until_target
    assert controller._latest_targets[robots.LEFT] is target


def test_fa_real_control_holds_large_ik_solution_jump():
    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig(max_ik_solution_jump_rad=1.0)
    controller._teleop_active = True
    controller.control_backend = "mujoco"
    controller._real_joint_state_fresh = lambda: False
    controller._warn_safety_messages = []
    controller._warn_safety = lambda key, message: controller._warn_safety_messages.append((key, message))
    snapshot = FaJointStateSnapshot(
        timestamp_s=1.0,
        left_arm=tuple([0.0] * 7),
        right_arm=tuple([0.0] * 7),
        neck=(0.0, 0.0),
    )
    controller._current_joint_snapshot = lambda: snapshot
    controller._latest_targets = {robots.LEFT: None, robots.RIGHT: object()}
    controller._last_safe_arm_targets = {
        robots.LEFT: None,
        robots.RIGHT: np.asarray([0.1] * 7, dtype=np.float64),
    }
    controller._ik_client = SimpleNamespace(
        solve=lambda hand_side, target, current_arm: FaArmIkResult(
            success=True,
            q_target=tuple([2.5] * 7),
            has_solution=True,
        )
    )
    published = []
    controller._builder = FaUpperPositionCommandBuilder()
    controller._limiter = SimpleNamespace(limit=lambda command, now_s=None: (command, ""))
    controller._publish_upper_command_outputs = lambda command, require_real_reset: published.append(command) or True

    controller._publish_upper_command_if_safe()

    assert published
    assert published[0].right_arm == pytest.approx(tuple([0.1] * 7))
    assert controller._last_safe_arm_targets[robots.RIGHT] == pytest.approx(np.asarray([0.1] * 7))
    assert controller._warn_safety_messages[0][0] == "right_ik_solution_jump"


def test_fa_command_publish_reuses_active_goal_without_repeating_ik_for_same_target():
    controller = FaRealControl.__new__(FaRealControl)
    controller.config = FaRealControlConfig()
    controller._teleop_active = True
    controller.control_backend = "mujoco"
    controller._real_joint_state_fresh = lambda: False
    controller._warn_safety = lambda key, message: None
    snapshot = FaJointStateSnapshot(
        timestamp_s=1.0,
        left_arm=tuple([0.0] * 7),
        right_arm=tuple([0.0] * 7),
        neck=(0.0, 0.0),
    )
    controller._current_joint_snapshot = lambda: snapshot
    target = SimpleNamespace(hand_side=robots.RIGHT, timestamp_s=123.0)
    controller._latest_targets = {robots.LEFT: None, robots.RIGHT: target}
    controller._latest_target_keys = {robots.LEFT: None, robots.RIGHT: None}
    controller._active_arm_goals = {robots.LEFT: None, robots.RIGHT: None}
    controller._last_safe_arm_targets = {robots.LEFT: None, robots.RIGHT: None}
    solve_calls = []

    def solve(hand_side, target, current_arm):
        solve_calls.append((hand_side, target.timestamp_s))
        return FaArmIkResult(success=True, q_target=tuple([0.2] * 7), has_solution=True)

    controller._ik_client = SimpleNamespace(solve=solve)
    controller._builder = FaUpperPositionCommandBuilder()
    controller._limiter = SimpleNamespace(limit=lambda command, now_s=None: (command, ""))
    published = []
    controller._publish_upper_command_outputs = lambda command, require_real_reset: published.append(command) or True

    controller._publish_upper_command_if_safe()
    controller._publish_upper_command_if_safe()

    assert solve_calls == [(robots.RIGHT, 123.0)]
    assert len(published) == 2
    assert published[-1].right_arm == pytest.approx(tuple([0.2] * 7))


def test_fa_mujoco_mirror_accepts_16d_command_and_uses_linear_interpolation():
    mirror = Sysmo32MujocoCommandMirror.__new__(Sysmo32MujocoCommandMirror)
    mirror.control_dt = 0.1
    mirror.arm_command_interpolation_steps = 5
    mirror.interpolation_profile = "linear"
    mirror.expected_command_length = FA_UPPER_COMMAND_LENGTH
    mirror._arm_joint_ids = list(range(14))
    mirror._arm_qpos_addrs = list(range(14))
    mirror._hold_joint_positions = np.zeros(14, dtype=np.float64)
    mirror._trajectory_start_positions = None
    mirror._trajectory_target_positions = None
    mirror._trajectory_start_time_s = None
    mirror._log_applied_arm_command = lambda values: None
    model = SimpleNamespace(jnt_range=np.asarray([[-10.0, 10.0]] * 14, dtype=np.float64))
    data = SimpleNamespace(qpos=np.zeros(14, dtype=np.float64))
    mirror._kinematics = SimpleNamespace(available=True, model=model, data=data)

    command = FaUpperPositionCommand(timestamp_s=1.0, values=tuple([1.0] * FA_UPPER_COMMAND_LENGTH))
    mirror.apply_arm_command(command)
    start_s = mirror._trajectory_start_time_s

    assert np.allclose(mirror._hold_joint_positions, np.zeros(14))
    mirror._update_interpolated_hold(start_s + 0.125)
    assert np.allclose(mirror._hold_joint_positions, np.full(14, 0.25))

    mirror._update_interpolated_hold(start_s + 0.5)
    assert np.allclose(mirror._hold_joint_positions, np.ones(14))
    assert mirror._trajectory_target_positions is None


def test_fa_mujoco_mirror_subscribes_to_fa_command_topic(monkeypatch):
    created_subscribers = []

    class FakeSubscriber:
        def __init__(self, host, port, topic, message_type=None):
            self.host = host
            self.port = port
            self.topic = topic
            self.message_type = message_type
            created_subscribers.append(self)

        def recv_keypoints(self):
            return None

    monkeypatch.setattr(
        "beavr.teleop.components.simulation.sysmo32_mujoco_command_sim.ZMQSubscriber",
        FakeSubscriber,
    )

    mirror = Sysmo32MujocoCommandMirror(
        host="127.0.0.1",
        arm_command_port=12040,
        hand_action_port=12041,
        urdf_path="/tmp/fa.urdf",
        load_model=False,
        kinematics_type="fa",
        expected_command_length=FA_UPPER_COMMAND_LENGTH,
    )

    assert mirror._arm_command_topic == "fa_upper_position_command"
    assert mirror._arm_command_type is FaUpperPositionCommand
    assert created_subscribers[0].topic == "fa_upper_position_command"
    assert created_subscribers[0].message_type is FaUpperPositionCommand


def test_fa_kinematics_reports_model_status_and_joint_metadata():
    kin = FaMujocoKinematics(
        FaKinematicsConfig(
            model_path="/home/likunwei/dataCollection/beavr-bot/robots/fa_description/urdf/fa_robot.urdf",
            left_joint_names=FA_LEFT_ARM_JOINT_NAMES,
            right_joint_names=FA_RIGHT_ARM_JOINT_NAMES,
            max_iter=2,
        )
    )
    if not kin.available:
        assert kin.load_error
        assert any(token in kin.load_error for token in ("FA", "XML", "site", "joint", "model", "resource"))
        return

    assert len(kin.left_joint_ids) == len(FA_LEFT_ARM_JOINT_NAMES)
    assert len(kin.right_joint_ids) == len(FA_RIGHT_ARM_JOINT_NAMES)
    assert kin.left_site_id >= 0 or kin.left_body_id >= 0
    assert kin.right_site_id >= 0 or kin.right_body_id >= 0

    current = np.zeros(14, dtype=np.float64)
    left_fk = kin.fk(robots.LEFT, current)
    assert left_fk is not None
    target = CartesianTarget(
        timestamp_s=1.0,
        hand_side=robots.LEFT,
        frame_id="base",
        position_m=tuple(left_fk[:3, 3] + np.array([0.005, 0.0, 0.0])),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    solved = kin.solve_ik(robots.LEFT, target, current, max_iter=2)
    assert solved is not None
    assert solved.shape == (7,)
    assert np.all(np.isfinite(solved))


def test_fa_kinematics_can_load_arm_model_without_endeff_for_mujoco_mirror():
    kin = FaMujocoKinematics(
        FaKinematicsConfig(
            model_path="/home/likunwei/dataCollection/beavr-bot/robots/fa_description/urdf/fa_robot.urdf",
            left_joint_names=FA_LEFT_ARM_JOINT_NAMES,
            right_joint_names=FA_RIGHT_ARM_JOINT_NAMES,
            require_endeff=False,
        )
    )
    if not kin.available:
        assert kin.load_error
        return

    assert len(kin.left_joint_ids) == len(FA_LEFT_ARM_JOINT_NAMES)
    assert len(kin.right_joint_ids) == len(FA_RIGHT_ARM_JOINT_NAMES)


def test_fa_kinematics_missing_model_has_clear_error(tmp_path):
    missing = tmp_path / "missing.urdf"
    kin = FaMujocoKinematics(FaKinematicsConfig(model_path=str(missing)))
    assert not kin.available
    assert str(missing) in kin.load_error
