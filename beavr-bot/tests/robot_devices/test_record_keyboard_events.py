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
    seen_events = {}

    class RobotWithoutTeleopStop:
        pass

    def fake_control_loop(*, events, **kwargs):
        seen_events["exit_early_at_reset_start"] = events["exit_early"]

    monkeypatch.setattr(
        "beavr.lerobot.common.robot_devices.control_utils.control_loop",
        fake_control_loop,
    )

    reset_environment(RobotWithoutTeleopStop(), events, reset_time_s=5, fps=30)

    assert seen_events["exit_early_at_reset_start"] is False
