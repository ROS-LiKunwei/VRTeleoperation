import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from beavr.lerobot.common.robot_devices.cameras.configs import OrbbecCameraConfig
from beavr.lerobot.common.robot_devices.cameras.orbbec import OrbbecCamera
from beavr.lerobot.common.robot_devices.cameras.recording_fault import CameraFault, handle_recording_fault


def test_reject_stale_and_failed_capture():
    cam = OrbbecCamera(OrbbecCameraConfig(serial_number='test'))
    cam._latest_color = np.zeros((2, 2, 3), dtype=np.uint8)
    cam._latest_depth = np.zeros((2, 2), dtype=np.uint16)
    cam._latest_timestamp = time.monotonic() - .4
    with pytest.raises(CameraFault, match='stale_rgbd'):
        cam.read_latest_rgbd(2000)
    cam._latest_timestamp = time.monotonic()
    cam.read_latest_rgbd(2000)
    cam._capture_error = 'SDK failed'
    with pytest.raises(CameraFault, match='capture_thread_failed'):
        cam.read_latest_rgbd(2000)


@pytest.mark.parametrize('succeed_on', [2, 99])
def test_bounded_recovery_marks_prefix_and_stops(tmp_path, succeed_on):
    calls = []
    def connect(warmup):
        assert warmup
        calls.append('connect')
        assert list((tmp_path/'meta/recording_faults').glob('*.json'))
        if calls.count('connect') < succeed_on:
            raise RuntimeError('unplugged')
    cam = SimpleNamespace(config=SimpleNamespace(serial_number='test', reconnect_attempts=3, max_frame_age_ms=300),
                          disconnect=lambda: calls.append('disconnect'), connect=connect,
                          read_latest_rgbd=lambda age: None)
    events = {}
    handle_recording_fault(SimpleNamespace(root=tmp_path, num_episodes=6), events, CameraFault(cam, 'stale', 'test'))
    report = json.loads(next((tmp_path/'meta/recording_faults').glob('*.json')).read_text())
    assert events.get('stop_recording', False) == (succeed_on > 3)
    assert events['camera_recovered'] == (succeed_on <= 3)
    assert not events['rerecord_episode']
    assert len(report['attempts']) == min(succeed_on, 3)
    assert report['recovered'] == (succeed_on <= 3)
    assert report['episode_index'] == 6


def test_operator_stop_is_not_overridden(tmp_path):
    events = {'stop_recording': False}
    def connect(warmup):
        events['stop_recording'] = True
    cam = SimpleNamespace(config=SimpleNamespace(serial_number='test', reconnect_attempts=3, max_frame_age_ms=300),
                          disconnect=lambda: None, connect=connect, read_latest_rgbd=lambda age: None)
    handle_recording_fault(SimpleNamespace(root=tmp_path, num_episodes=0), events, CameraFault(cam, 'stale', 'test'))
    assert events['stop_recording'] and not events['camera_recovered']
