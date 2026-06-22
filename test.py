#!/usr/bin/env python3
import cv2
import glob
import math
import re
import time
import argparse
import numpy as np


def video_index_from_path(path: str) -> int:
    """
    从 /dev/video0 这种路径中提取 0。
    """
    match = re.search(r"/dev/video(\d+)$", path)
    return int(match.group(1)) if match else -1


def natural_sort_video_devices(devices):
    """
    按 /dev/video0, /dev/video1, /dev/video2 ... 排序。
    """
    return sorted(devices, key=video_index_from_path)


def try_open_camera(device_path, width=None, height=None, fps=None):
    """
    尝试打开摄像头，并读取一帧。
    能读到图像才认为是有效摄像头。
    """
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

    if not cap.isOpened():
        return None

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)

    # 给摄像头一点初始化时间
    time.sleep(0.1)

    ok = False
    frame = None

    # 多读几次，避免第一帧失败
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
        time.sleep(0.05)

    if not ok:
        cap.release()
        return None

    return cap


def scan_cameras(width=None, height=None, fps=None):
    """
    扫描 /dev/video*，返回可用摄像头列表。
    """
    devices = natural_sort_video_devices(glob.glob("/dev/video*"))

    cameras = []

    print("检测到的 video 设备：")
    for dev in devices:
        print(f"  {dev}")

    print("\n正在尝试打开可用摄像头...")

    for dev in devices:
        cap = try_open_camera(dev, width=width, height=height, fps=fps)

        if cap is not None:
            idx = video_index_from_path(dev)
            cameras.append(
                {
                    "device": dev,
                    "index": idx,
                    "cap": cap,
                }
            )
            print(f"  [OK] {dev}, index={idx}")
        else:
            print(f"  [跳过] {dev} 无法读取图像，可能不是摄像头或已被占用")

    return cameras


def make_tile(frame, label, tile_w, tile_h):
    """
    将单个摄像头画面缩放成统一大小，并加文字标签。
    """
    if frame is None:
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        cv2.putText(
            tile,
            "No Frame",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return tile

    tile = cv2.resize(frame, (tile_w, tile_h))

    # 半透明黑底，方便看清文字
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile_w, 45), (0, 0, 0), -1)
    tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)

    cv2.putText(
        tile,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return tile


def compose_grid(tiles, tile_w, tile_h):
    """
    把多个摄像头画面拼成网格。
    """
    n = len(tiles)

    if n == 0:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    grid_rows = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            i = r * cols + c
            if i < n:
                row_tiles.append(tiles[i])
            else:
                row_tiles.append(blank.copy())
        grid_rows.append(np.hstack(row_tiles))

    return np.vstack(grid_rows)


def main():
    parser = argparse.ArgumentParser(
        description="自动检测并显示 /dev/video* 摄像头画面"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="摄像头采集宽度，默认 640"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="摄像头采集高度，默认 480"
    )
    parser.add_argument("--fps", type=int, default=30, help="摄像头采集帧率，默认 30")
    parser.add_argument(
        "--tile-width", type=int, default=480, help="每个显示窗口块的宽度，默认 480"
    )
    parser.add_argument(
        "--tile-height", type=int, default=360, help="每个显示窗口块的高度，默认 360"
    )
    args = parser.parse_args()

    cameras = scan_cameras(width=args.width, height=args.height, fps=args.fps)

    if not cameras:
        print("\n没有找到可用摄像头。")
        print("可以先用下面命令检查：")
        print("  ls /dev/video*")
        print("  v4l2-ctl --list-devices")
        return

    print(f"\n共打开 {len(cameras)} 个摄像头。")
    print("按 q 退出。")

    window_name = "All Cameras"

    while True:
        tiles = []

        for cam_id, cam in enumerate(cameras):
            cap = cam["cap"]
            dev = cam["device"]
            dev_index = cam["index"]

            ret, frame = cap.read()

            if not ret or frame is None:
                frame = None

            label = f"display={cam_id}  index={dev_index}  {dev}"
            tile = make_tile(frame, label, args.tile_width, args.tile_height)
            tiles.append(tile)

        grid = compose_grid(tiles, args.tile_width, args.tile_height)

        cv2.imshow(window_name, grid)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    for cam in cameras:
        cam["cap"].release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
