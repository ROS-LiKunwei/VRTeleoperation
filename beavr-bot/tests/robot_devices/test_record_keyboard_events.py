from beavr.lerobot.common.robot_devices.control_utils import (
    _apply_recording_key_event,
    _terminal_sequence_to_recording_key,
    reset_environment,
)


def _events():
    return {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }


def test_terminal_arrow_sequences_map_to_recording_keys():
    assert _terminal_sequence_to_recording_key(b"\x1b[D") == "left"
    assert _terminal_sequence_to_recording_key(b"\x1b[C") == "right"
    assert _terminal_sequence_to_recording_key(b"\x1b") == "esc"
    assert _terminal_sequence_to_recording_key(b"a") is None


def test_left_arrow_requests_rerecord():
    events = _events()

    assert _apply_recording_key_event("left", events) is True

    assert events["rerecord_episode"] is True
    assert events["exit_early"] is True
    assert events["stop_recording"] is False


def test_right_arrow_requests_next_recording_step():
    events = _events()

    assert _apply_recording_key_event("right", events) is True

    assert events["rerecord_episode"] is False
    assert events["exit_early"] is True
    assert events["stop_recording"] is False


def test_escape_stops_recording():
    events = _events()

    assert _apply_recording_key_event("esc", events) is True

    assert events["rerecord_episode"] is False
    assert events["exit_early"] is True
    assert events["stop_recording"] is True


def test_reset_environment_clears_stale_exit_event(monkeypatch):
    events = _events()
    events["exit_early"] = True

    class RobotWithoutTeleopStop:
        pass

    reset_environment(RobotWithoutTeleopStop(), events, reset_time_s=0.0, fps=30)

    assert events["exit_early"] is False


def test_reset_environment_ignores_rerecord_event(monkeypatch):
    events = _events()
    events["rerecord_episode"] = True
    events["exit_early"] = True

    class RobotWithoutTeleopStop:
        pass

    reset_environment(RobotWithoutTeleopStop(), events, reset_time_s=1.0, fps=30)

    assert events["rerecord_episode"] is False
    assert events["exit_early"] is False


def test_reset_environment_is_skippable_without_observation_capture(monkeypatch):
    events = _events()
    clock = {"t": 0.0}

    class Robot:
        teleop_stop_called = False

        def teleop_stop(self):
            self.teleop_stop_called = True

        def capture_observation(self):
            raise AssertionError("reset_environment must not capture observations")

    def fake_sleep(_duration):
        events["exit_early"] = True
        clock["t"] += 0.001

    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.control_utils.time.perf_counter",
        lambda: clock["t"],
    )
    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.control_utils.time.sleep",
        fake_sleep,
    )

    robot = Robot()
    reset_environment(robot, events, reset_time_s=10.0, fps=30)

    assert robot.teleop_stop_called is True
    assert events["exit_early"] is False
