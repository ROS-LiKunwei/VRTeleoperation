"""Fail closed at camera faults; never bridge a recording across reconnects."""
import json
import logging
import time
from pathlib import Path


class CameraFault(TimeoutError):
    def __init__(self, camera, reason, detail):
        self.camera, self.reason = camera, reason
        super().__init__(f"{camera}: {reason}: {detail}")


def handle_recording_fault(dataset, events, fault):
    # Returning from control_loop lets record() commit the valid prefix and
    # finalize normally. The marker is written BEFORE attempting reconnection.
    events.update(exit_early=True, rerecord_episode=False, camera_recovered=False)
    camera = fault.camera
    folder = Path(dataset.root) / "meta" / "recording_faults"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"episode_{dataset.num_episodes:06d}_{time.time_ns()}.json"
    buffer = getattr(dataset, "episode_buffer", None) or {}
    report = dict(episode_index=dataset.num_episodes, error=str(fault),
                  buffered_frames=buffer.get("size", 0),
                  serial_number=camera.config.serial_number,
                  disposition="interrupted_prefix_requires_review",
                  reason=fault.reason, attempts=[], recovered=False)

    def persist():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(path)

    persist()
    logging.error("Camera recording fault: %s; marker=%s", fault, path)
    for attempt in range(camera.config.reconnect_attempts):
        result = dict(attempt=attempt + 1)
        try:
            camera.disconnect()
            camera.connect(warmup=True)
            camera.read_latest_rgbd(camera.config.max_frame_age_ms)
            report['recovered'] = True
        except Exception as exc:
            result['error'] = repr(exc)
            logging.exception("Camera reconnect attempt %s failed", attempt + 1)
        report['attempts'].append(result)
        persist()
        if report['recovered']:
            break
    if report['recovered']:
        # Preserve an Escape/stop request arriving while connect() was running.
        events['camera_recovered'] = not events.get('stop_recording', False)
    else:
        events['stop_recording'] = True
    logging.warning("Camera recovery completed: recovered=%s, continue_new_episode=%s",
                    report['recovered'], events['camera_recovered'])
