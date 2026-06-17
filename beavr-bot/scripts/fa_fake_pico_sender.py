#!/usr/bin/env python3
"""Send synthetic PICO4 hand frames for FA teleop bringup.

This script connects to BeaVR's PICO input ports and publishes two stable,
slowly moving 26-keypoint hands plus an optional resume signal.
"""

from __future__ import annotations

import argparse
import math
import signal
import time
from dataclasses import dataclass

import zmq


RIGHT_HAND_PORT = 8087
LEFT_HAND_PORT = 8110
PAUSE_PORT = 8100
BUTTON_PORT = 8095


@dataclass(frozen=True)
class HandProfile:
    x_offset: float
    y_offset: float
    z_offset: float
    mirror_x: float


def _make_socket(context: zmq.Context, host: str, port: int) -> zmq.Socket:
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.connect(f"tcp://{host}:{port}")
    return socket


def _hand_keypoints(profile: HandProfile, t: float) -> list[tuple[float, float, float]]:
    wrist = (
        profile.x_offset + 0.05 * math.sin(0.7 * t),
        profile.y_offset + 0.035 * math.sin(0.5 * t),
        profile.z_offset + 0.04 * math.cos(0.45 * t),
    )
    roll = 0.28 * math.sin(0.35 * t)
    pitch = 0.20 * math.cos(0.28 * t)

    def rot(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        x *= profile.mirror_x
        cy, sy = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        x, z = cy * x + sy * z, -sy * x + cy * z
        y, z = cr * y - sr * z, sr * y + cr * z
        return wrist[0] + x, wrist[1] + y, wrist[2] + z

    local = [(0.0, 0.0, 0.0)] * 26
    local[0] = (0.0, 0.0, 0.0)
    local[1] = (0.0, 0.045, 0.0)

    chains = {
        "thumb": ([-0.055, -0.075, -0.09, -0.105], [0.035, 0.06, 0.083, 0.105], [0.012, 0.016, 0.018, 0.02], [2, 3, 4, 5]),
        "index": ([-0.035, -0.035, -0.034, -0.033], [0.075, 0.115, 0.15, 0.18], [0.008, 0.01, 0.011, 0.012], [7, 8, 9, 10]),
        "middle": ([0.0, 0.0, 0.0, 0.0], [0.085, 0.13, 0.17, 0.205], [0.0, 0.002, 0.003, 0.004], [12, 13, 14, 15]),
        "ring": ([0.035, 0.035, 0.034, 0.033], [0.078, 0.116, 0.15, 0.18], [-0.006, -0.008, -0.009, -0.01], [17, 18, 19, 20]),
        "pinky": ([0.065, 0.066, 0.065, 0.064], [0.068, 0.098, 0.124, 0.148], [-0.012, -0.014, -0.015, -0.016], [22, 23, 24, 25]),
    }
    metacarpals = {
        6: (-0.035, 0.055, 0.006),
        11: (0.0, 0.06, 0.0),
        16: (0.035, 0.055, -0.006),
        21: (0.065, 0.048, -0.012),
    }
    for idx, point in metacarpals.items():
        local[idx] = point
    for xs, ys, zs, indices in chains.values():
        for idx, x, y, z in zip(indices, xs, ys, zs):
            local[idx] = (x, y, z)

    return [rot(point) for point in local]


def _payload(points: list[tuple[float, float, float]], hand_command: int) -> bytes:
    coords = "|".join(f"{x:.7f},{y:.7f},{z:.7f}" for x, y, z in points)
    return f"absolute:{coords}:hand_command={hand_command}".encode("utf-8")


def _try_send(socket: zmq.Socket, data: bytes) -> bool:
    try:
        socket.send(data, flags=zmq.NOBLOCK)
    except zmq.Again:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.134", help="BeaVR PICO bind host from dev.yaml")
    parser.add_argument("--rate", type=float, default=60.0, help="Frame rate in Hz")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever")
    parser.add_argument("--no-pause", action="store_true", help="Do not send stable resume on port 8100")
    parser.add_argument("--button-high", action="store_true", help="Send High on button port 8095")
    args = parser.parse_args()

    context = zmq.Context.instance()
    right = _make_socket(context, args.host, RIGHT_HAND_PORT)
    left = _make_socket(context, args.host, LEFT_HAND_PORT)
    pause = None if args.no_pause else _make_socket(context, args.host, PAUSE_PORT)
    button = _make_socket(context, args.host, BUTTON_PORT) if args.button_high else None

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    right_profile = HandProfile(x_offset=0.12, y_offset=1.45, z_offset=0.20, mirror_x=1.0)
    left_profile = HandProfile(x_offset=-0.12, y_offset=1.45, z_offset=0.20, mirror_x=-1.0)
    period = 1.0 / args.rate
    start = time.monotonic()
    next_tick = start
    frames = 0

    print(
        f"Sending fake PICO data to {args.host}: right={RIGHT_HAND_PORT}, left={LEFT_HAND_PORT}, "
        f"pause={'off' if args.no_pause else PAUSE_PORT}, rate={args.rate:.1f}Hz"
    )
    while running:
        now = time.monotonic()
        if args.duration > 0 and now - start >= args.duration:
            break
        t = now - start
        hand_command = 2 if int(t // 2.0) % 2 else 1
        _try_send(right, _payload(_hand_keypoints(right_profile, t), hand_command))
        _try_send(left, _payload(_hand_keypoints(left_profile, t + 0.35), hand_command))
        if pause is not None:
            _try_send(pause, b"High")
        if button is not None:
            _try_send(button, b"High")
        frames += 1
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))

    print(f"Stopped after {frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
