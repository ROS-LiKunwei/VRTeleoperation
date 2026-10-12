import json
from types import SimpleNamespace

import pytest

from beavr.lerobot.common.robot_devices.cameras.manual_controls import (
    ORBBEC_OFF, ORBBEC_FIXED, apply_orbbec_manual_controls,
    apply_realsense_manual_controls, load_camera_settings, save_camera_settings,
)
from beavr.lerobot.common.robot_devices.cameras.configs import OrbbecCameraConfig


class FakeOrbbec:
    def __init__(self, reject=None):
        self.values = {v: True for v in ORBBEC_OFF.values()}
        self.reject = reject

    def is_property_supported(self, prop, permission):
        return True

    def get_int_property_range(self, prop):
        return SimpleNamespace(min=0, max=10000, step=1, default_value=100)

    def set_bool_property(self, prop, value):
        if prop != self.reject:
            self.values[prop] = value

    set_int_property = set_bool_property

    def get_bool_property(self, prop):
        return self.values[prop]

    get_int_property = get_bool_property


def sdk():
    names = list(ORBBEC_OFF.values()) + [x[0] for x in ORBBEC_FIXED.values()]
    return SimpleNamespace(OBPropertyID=SimpleNamespace(**dict(zip(names, names))),
                           OBPermissionType=SimpleNamespace(PERMISSION_READ=1, PERMISSION_WRITE=2))


def test_orbbec_disables_automatic_modes_and_freezes_values():
    d = FakeOrbbec()
    result = apply_orbbec_manual_controls(d, {'color_exposure': 123, 'depth_exposure': 222}, sdk())
    assert all(result['values'][name] is False for name in ORBBEC_OFF)
    assert result['values']['color_exposure'] == 123
    assert result['values']['color_white_balance'] == 100
    assert result['values']['ir_exposure'] == 222
    assert result['values']['color_power_line_frequency'] == 0


def test_rejected_auto_disable_fails_instead_of_silent_fallback():
    with pytest.raises(RuntimeError, match='read back'):
        apply_orbbec_manual_controls(FakeOrbbec('OB_PROP_COLOR_AUTO_EXPOSURE_BOOL'), {}, sdk())


@pytest.mark.parametrize('values', [{'color_exposure': 10001}, {'typo': 1},
                                   {'depth_gain': 1, 'ir_gain': 2}, {'color_power_line_frequency': 3}])
def test_invalid_controls_fail(values):
    with pytest.raises(ValueError):
        apply_orbbec_manual_controls(FakeOrbbec(), values, sdk())


def test_settings_reuse_and_resume_guard(tmp_path):
    camera = SimpleNamespace(config=OrbbecCameraConfig(serial_number='test'),
                             effective_controls=apply_orbbec_manual_controls(FakeOrbbec(), {}, sdk()),
                             stream_profiles={})
    path = tmp_path / 'camera_settings.json'
    save_camera_settings({'wrist': camera}, path)
    cfg = load_camera_settings(path)['wrist']
    assert cfg.manual_mode
    assert cfg.manual_controls['color_exposure'] == 100
    camera.config = cfg
    save_camera_settings({'wrist': camera}, path)
    camera.config.width = 1280
    with pytest.raises(ValueError, match='differ'):
        save_camera_settings({'wrist': camera}, path)


def test_realsense_color_and_depth_manual_controls():
    from beavr.lerobot.common.robot_devices.cameras.manual_controls import RS_OFF, RS_FIXED
    names = (*RS_OFF, *RS_FIXED)
    class Sensor:
        def __init__(self):
            self.values = {}
        def supports(self, option): return True
        def is_option_read_only(self, option): return False
        def get_option_range(self, option): return SimpleNamespace(min=0, max=10000, step=1, default=100)
        def set_option(self, option, value): self.values[option] = value
        def get_option(self, option): return self.values[option]
    color, depth = Sensor(), Sensor()
    device = SimpleNamespace(first_color_sensor=lambda: color, first_depth_sensor=lambda: depth)
    rs = SimpleNamespace(option=SimpleNamespace(**dict(zip(names, names))))
    result = apply_realsense_manual_controls(device, {'color_exposure': 200}, rs, use_depth=True)
    assert result['values']['color_exposure'] == 200
    assert color.values['enable_auto_white_balance'] == 0
    assert color.values['enable_auto_exposure'] == 0
    assert depth.values['enable_auto_exposure'] == 0
