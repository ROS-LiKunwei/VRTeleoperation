"""Fixed SDK image controls shared by recording and inference camera backends.

This disables exposed photometric auto controls, not the sensor ISP or the
stereo depth algorithm. Numeric controls use SDK defaults unless overridden;
resolved values are exported so later runs need not rely on firmware defaults.
"""
import json
import math
from dataclasses import asdict
from pathlib import Path


# Order matters: turn off automatic modes before writing manual values.
ORBBEC_OFF = {
    'color_auto_exposure': 'OB_PROP_COLOR_AUTO_EXPOSURE_BOOL',
    'color_auto_white_balance': 'OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL',
    'color_hdr': 'OB_PROP_COLOR_HDR_BOOL',
    'depth_auto_exposure': 'OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL',
    'ir_auto_exposure': 'OB_PROP_IR_AUTO_EXPOSURE_BOOL',
    'hdr_merge': 'OB_PROP_HDR_MERGE_BOOL',
    'frame_interleave': 'OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL',
}
ORBBEC_FIXED = {
    'color_auto_exposure_priority': ('OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT', 0),
    'depth_auto_exposure_priority': ('OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT', 0),
    'color_backlight_compensation': ('OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT', 0),
    'color_power_line_frequency': ('OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT', 0),
    'color_exposure': ('OB_PROP_COLOR_EXPOSURE_INT', None),
    'color_gain': ('OB_PROP_COLOR_GAIN_INT', None),
    'color_white_balance': ('OB_PROP_COLOR_WHITE_BALANCE_INT', None),
    'color_brightness': ('OB_PROP_COLOR_BRIGHTNESS_INT', None),
    'color_contrast': ('OB_PROP_COLOR_CONTRAST_INT', None),
    'color_saturation': ('OB_PROP_COLOR_SATURATION_INT', None),
    'color_gamma': ('OB_PROP_COLOR_GAMMA_INT', None),
    'color_sharpness': ('OB_PROP_COLOR_SHARPNESS_INT', 0),
    'color_denoising_level': ('OB_PROP_COLOR_DENOISING_LEVEL_INT', 0),
    'depth_exposure': ('OB_PROP_DEPTH_EXPOSURE_INT', None),
    'depth_gain': ('OB_PROP_DEPTH_GAIN_INT', None),
    'ir_exposure': ('OB_PROP_IR_EXPOSURE_INT', None),
    'ir_gain': ('OB_PROP_IR_GAIN_INT', None),
}


def _check_value(name, value, limits, default_attr):
    value = getattr(limits, default_attr) if value is None else value
    if not math.isfinite(value) or not limits.min <= value <= limits.max:
        raise ValueError(f'{name}={value} outside [{limits.min}, {limits.max}]')
    if limits.step and not math.isclose((value - limits.min) / limits.step,
                                        round((value - limits.min) / limits.step), abs_tol=1e-5):
        raise ValueError(f'{name}={value} does not match SDK step {limits.step}')
    return value


def apply_orbbec_manual_controls(device, overrides, sdk):
    unknown = set(overrides) - set(ORBBEC_FIXED)
    if unknown:
        raise ValueError(f'Unknown Orbbec manual controls: {sorted(unknown)}')
    # Depth and IR exposure/gain may alias the same hardware control.
    for suffix in ('exposure', 'gain'):
        d, ir = f'depth_{suffix}', f'ir_{suffix}'
        if d in overrides and ir in overrides and overrides[d] != overrides[ir]:
            raise ValueError(f'{d} and {ir} must agree')
    requested = dict(overrides)
    for suffix in ('exposure', 'gain'):
        for src, dst in ((f'depth_{suffix}', f'ir_{suffix}'), (f'ir_{suffix}', f'depth_{suffix}')):
            if src in requested:
                requested.setdefault(dst, requested[src])
    result = {'values': {}, 'unsupported': []}
    getters = {}
    required = {'color_auto_exposure', 'color_auto_white_balance', 'color_exposure',
                'color_gain', 'color_white_balance'}
    for name, (prop_name, value) in {
        **{k: (v, False) for k, v in ORBBEC_OFF.items()}, **ORBBEC_FIXED,
    }.items():
        prop = getattr(sdk.OBPropertyID, prop_name, None)
        readable = prop is not None and device.is_property_supported(prop, sdk.OBPermissionType.PERMISSION_READ)
        if not readable:
            if name in required or name in overrides:
                raise RuntimeError(f'Required camera control is unavailable: {prop_name}')
            result['unsupported'].append(name)
            continue
        is_bool = name in ORBBEC_OFF
        getter = device.get_bool_property if is_bool else device.get_int_property
        setter = device.set_bool_property if is_bool else device.set_int_property
        if not is_bool:
            limits = device.get_int_property_range(prop)
            if name in requested and value is not None and requested[name] != value:
                raise ValueError(f'Manual mode requires {name}={value}')
            value = _check_value(name, requested.get(name, value), limits, 'default_value')
            if int(value) != value:
                raise ValueError(f'{name} must be an integer in SDK units')
            value = int(value)
        if device.is_property_supported(prop, sdk.OBPermissionType.PERMISSION_WRITE):
            setter(prop, value)
        elif getter(prop) != value:
            raise RuntimeError(f'Cannot set read-only camera control: {prop_name}')
        result['values'][name] = value
        getters[name] = (getter, prop)
    # Verify after ALL writes, since one SDK control can affect another.
    for name, (getter, prop) in getters.items():
        actual = getter(prop)
        if actual != result['values'][name]:
            raise RuntimeError(f'Camera rejected {name}={result["values"][name]}: read back {actual}')
    return result


RS_OFF = ('hdr_enabled', 'enable_auto_exposure', 'enable_auto_white_balance',
          'auto_exposure_priority', 'backlight_compensation', 'power_line_frequency',
          'rgb_tnr_enabled')
RS_FIXED = ('exposure', 'gain', 'white_balance', 'brightness', 'contrast',
            'saturation', 'gamma', 'hue', 'sharpness')


def apply_realsense_manual_controls(device, overrides, sdk, use_depth=False):
    result = {'values': {}, 'unsupported': []}
    getters = {}
    used = set()
    for kind, sensor in [('color', device.first_color_sensor()),
                         *([('depth', device.first_depth_sensor())] if use_depth else [])]:
        for name in (*RS_OFF, *RS_FIXED):
            key = f'{kind}_{name}'
            option = getattr(sdk.option, name, None)
            if option is None or not sensor.supports(option):
                if key in overrides or name in ('enable_auto_exposure', 'exposure', 'gain') or (
                    kind == 'color' and name in ('enable_auto_white_balance', 'white_balance')
                ):
                    raise RuntimeError(f'Required camera control is unavailable: {key}')
                result['unsupported'].append(key)
                continue
            forced = name in RS_OFF or name == 'sharpness'
            if forced and overrides.get(key, 0) != 0:
                raise ValueError(f'Manual mode requires {key}=0')
            value = _check_value(key, 0 if forced else overrides.get(key),
                                 sensor.get_option_range(option), 'default')
            if sensor.is_option_read_only(option):
                if not math.isclose(sensor.get_option(option), value, abs_tol=1e-5):
                    raise RuntimeError(f'Cannot set read-only camera control: {key}')
            else:
                sensor.set_option(option, value)
            result['values'][key] = value
            getters[key] = (sensor, option)
            used.add(key)
    if set(overrides) - used:
        raise ValueError(f'Unknown RealSense manual controls: {sorted(set(overrides) - used)}')
    for key, (sensor, option) in getters.items():
        actual = sensor.get_option(option)
        if not math.isclose(actual, result['values'][key], abs_tol=1e-5):
            raise RuntimeError(f'Camera rejected {key}={result["values"][key]}: read back {actual}')
    return result


def camera_settings_document(cameras):
    entries = {}
    for name, camera in cameras.items():
        config = asdict(camera.config)
        config['type'] = camera.config.type
        effective = getattr(camera, 'effective_controls', None)
        if config['type'] == 'orbbec':
            config['expected_stream_profiles'] = getattr(camera, 'stream_profiles', {})
        if effective:
            # Only writable numeric overrides; disabled boolean modes are backend invariants.
            values = effective['values']
            config['manual_controls'] = ({k: v for k, v in values.items() if k in ORBBEC_FIXED}
                                         if config['type'] == 'orbbec' else values)
        if getattr(camera, 'serial_number', None) is not None:
            config['serial_number'] = camera.serial_number
            config['name'] = None
        entries[name] = {'config': config, 'effective_controls': effective,
                         'stream_profiles': getattr(camera, 'stream_profiles', None)}
    return {'schema_version': 1, 'cameras': entries}


def save_camera_settings(cameras, path):
    path = Path(path)
    document = camera_settings_document(cameras)
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError(f'Camera settings differ from the existing dataset: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + '\n')


def load_camera_settings(path):
    """Use this in both recording and inference, then make_cameras_from_configs()."""
    import draccus
    from .configs import CameraConfig
    document = json.loads(Path(path).read_text())
    if document.get('schema_version') != 1:
        raise ValueError('Unsupported camera settings schema')
    return {name: draccus.decode(CameraConfig, entry['config'])
            for name, entry in document['cameras'].items()}
