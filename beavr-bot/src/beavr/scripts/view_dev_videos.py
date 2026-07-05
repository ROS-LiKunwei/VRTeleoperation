#!/usr/bin/env python3
"""Serve a headless web preview for one /dev/video camera."""

from __future__ import annotations

import argparse
import html
import platform
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2


DEFAULT_CAMERA = "16"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


@dataclass
class SharedFrame:
    label: str
    jpeg: bytes | None = None
    frame_count: int = 0
    fps: float = 0.0
    last_error: str | None = None
    stopped: bool = False

    def __post_init__(self) -> None:
        self.lock = threading.Lock()


def resolve_linux_video_source(raw_source: str) -> tuple[str, int | str]:
    path = Path(raw_source)
    if path.is_absolute():
        return str(path), str(path)

    try:
        index = int(raw_source)
    except ValueError:
        return raw_source, raw_source

    if platform.system() == "Linux":
        device_path = f"/dev/video{abs(index)}"
        return device_path, device_path

    return f"index {index}", index


def open_capture(source: int | str, args: argparse.Namespace) -> cv2.VideoCapture:
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)

    if args.width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps is not None:
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    if args.mjpg:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open camera source {source!r}")

    return capture


def capture_loop(shared: SharedFrame, source: int | str, args: argparse.Namespace) -> None:
    capture = open_capture(source, args)
    fps_window_start_s = time.monotonic()
    fps_window_frames = 0

    try:
        while True:
            with shared.lock:
                if shared.stopped:
                    return

            ok, frame = capture.read()
            if not ok or frame is None:
                with shared.lock:
                    shared.last_error = "camera read failed"
                time.sleep(0.05)
                continue

            if args.stream_width is not None or args.stream_height is not None:
                height, width = frame.shape[:2]
                target_width = args.stream_width or width
                target_height = args.stream_height or height
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok:
                with shared.lock:
                    shared.last_error = "jpeg encode failed"
                continue

            now_s = time.monotonic()
            fps_window_frames += 1
            elapsed_s = now_s - fps_window_start_s
            fps = shared.fps
            if elapsed_s >= 1.0:
                fps = fps_window_frames / elapsed_s
                fps_window_frames = 0
                fps_window_start_s = now_s

            with shared.lock:
                shared.jpeg = encoded.tobytes()
                shared.frame_count += 1
                shared.fps = fps
                shared.last_error = None
    finally:
        capture.release()


class CameraPreviewHandler(BaseHTTPRequestHandler):
    server_version = "BeavrCameraPreview/1.0"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_index()
        elif self.path == "/stream.mjpg":
            self._send_stream()
        elif self.path == "/status":
            self._send_status()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    @property
    def shared(self) -> SharedFrame:
        return self.server.shared_frame  # type: ignore[attr-defined]

    def _send_index(self) -> None:
        label = html.escape(self.shared.label)
        body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{label} preview</title>
  <style>
    html, body {{
      margin: 0;
      min-height: 100%;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
    }}
    header {{
      padding: 12px 16px;
      background: #1f1f1f;
      border-bottom: 1px solid #333;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }}
    main {{
      padding: 16px;
    }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      background: #000;
    }}
    code {{
      color: #9fd;
    }}
  </style>
</head>
<body>
  <header>
    <strong>{label}</strong>
    <span id="status">connecting</span>
  </header>
  <main>
    <img src="/stream.mjpg" alt="{label} stream">
  </main>
  <script>
    async function refreshStatus() {{
      try {{
        const response = await fetch('/status', {{cache: 'no-store'}});
        document.getElementById('status').textContent = await response.text();
      }} catch (error) {{
        document.getElementById('status').textContent = 'status unavailable';
      }}
    }}
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_status(self) -> None:
        with self.shared.lock:
            if self.shared.last_error:
                status = self.shared.last_error
            else:
                status = f"{self.shared.frame_count} frames, {self.shared.fps:.1f} fps"
        encoded = status.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            with self.shared.lock:
                jpeg = self.shared.jpeg
                stopped = self.shared.stopped
            if stopped:
                return
            if jpeg is None:
                time.sleep(0.05)
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.001)
            except (BrokenPipeError, ConnectionResetError):
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser preview for one /dev/video camera.")
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Camera index or path. Default: {DEFAULT_CAMERA} -> /dev/video16 on Linux.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"HTTP bind host. Default: {DEFAULT_HOST}.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port. Default: {DEFAULT_PORT}.")
    parser.add_argument("--width", type=int, default=None, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height.")
    parser.add_argument("--fps", type=float, default=None, help="Requested capture FPS.")
    parser.add_argument("--stream-width", type=int, default=None, help="Resize streamed frames to this width.")
    parser.add_argument("--stream-height", type=int, default=None, help="Resize streamed frames to this height.")
    parser.add_argument("--jpeg-quality", type=int, default=80, choices=range(1, 101), metavar="[1-100]")
    parser.add_argument("--mjpg", action="store_true", help="Request MJPG capture format from the camera.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label, source = resolve_linux_video_source(args.camera)
    shared = SharedFrame(label=label)

    capture_thread = threading.Thread(target=capture_loop, args=(shared, source, args), daemon=True)
    capture_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), CameraPreviewHandler)
    server.shared_frame = shared  # type: ignore[attr-defined]

    print(f"Opened {args.camera!r} as {label}")
    print(f"Web preview listening on http://{args.host}:{args.port}/")
    if args.host == "0.0.0.0":
        print("Open http://<board-ip>:8080/ from another computer on the same network.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping preview server.")
    finally:
        with shared.lock:
            shared.stopped = True
        server.shutdown()
        server.server_close()
        capture_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
