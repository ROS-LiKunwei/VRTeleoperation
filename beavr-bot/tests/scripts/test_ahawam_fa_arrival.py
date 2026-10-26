import numpy as np

import beavr.scripts.ahawam_fa_arrival as arrival


def test_arrival_requires_stable_fresh_feedback(monkeypatch):
    ticks = iter((0.0, 0.0, 0.0, 0.1, 0.1, 0.21, 0.21, 0.21))
    monkeypatch.setattr(arrival.time, "monotonic", lambda: next(ticks))

    class Ros:
        def __init__(self):
            self.sequence = 0

        def state_snapshot(self, _timeout, _max_age, *, newer_than_sequence):
            self.sequence += 1
            assert newer_than_sequence in (None, self.sequence - 1)
            return np.zeros(16, dtype=np.float32), {
                "sequence": self.sequence,
                "source_age_ms": 0.0,
            }

    report = arrival.wait_for_robot_arrival(
        Ros(),
        np.zeros(16, dtype=np.float32),
        stable_seconds=0.2,
        timeout_s=1.0,
    )

    assert report["joint_state_sequence"] == 3
    assert report["mode"] == "target_reached"
    assert report["max_error_rad"] == 0.0
    assert report["spread_rad"] == 0.0
