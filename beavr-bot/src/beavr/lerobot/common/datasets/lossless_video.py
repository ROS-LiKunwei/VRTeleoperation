"""Exact RGB24 and uint16-mm video storage. Requires software PyAV codecs.

Gray depth v1: FFV1 gray16le MKV, stored=round(mm*65535/1000).

Depth packing v1: R=low byte, G=high byte, B=0. Never decode via YUV420.
PNG files are staging inputs only and may be deleted after verified encoding.
"""
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

RGB_ENCODING = 'rgb24_h264_lossless_v1'
DEPTH_ENCODING = 'uint16_rg_h264_lossless_v1'
GRAY_DEPTH_ENCODING = 'uint16_gray_ffv1_0_1000mm_v1'

def depth_to_gray16(depth):
    if depth.dtype != np.uint16 or np.any(depth > 1000):
        raise ValueError('Gray depth requires uint16 millimetres in 0..1000')
    return np.rint(depth.astype(np.float64) * 65535 / 1000).astype(np.uint16)

def gray16_to_depth(gray):
    return np.rint(gray.astype(np.float64) * 1000 / 65535).astype(np.uint16)

def feature_video_path(path, feature):
    if (feature.get('info') or {}).get('lossless_encoding') == GRAY_DEPTH_ENCODING:
        return Path(path).with_suffix('.mkv')
    return Path(path)



def pack_depth(depth):
    depth = np.asarray(depth)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Depth must be HxW uint16')
    return np.stack((depth & 255, depth >> 8, np.zeros_like(depth)), axis=-1).astype(np.uint8)


def unpack_depth(rgb):
    if np.any(rgb[..., 2]):
        raise ValueError('Invalid packed depth: nonzero reserved blue channel')
    return rgb[..., 0].astype(np.uint16) | (rgb[..., 1].astype(np.uint16) << 8)


def _read_source(path, depth):
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint16) if depth else np.asarray(image.convert('RGB'))


def encode_lossless_video(img_dir, video_path, fps, depth=False, gray_depth=False):
    """Verify every decoded pixel before publishing video or deleting any staging input."""
    paths = sorted(Path(img_dir).glob('frame_*.png'))
    if not paths:
        raise ValueError(f'No frames in {img_dir}')
    target = Path(video_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial' + target.suffix)
    depth = depth or gray_depth
    h, w = _read_source(paths[0], depth).shape[:2]
    try:
        with av.open(str(temporary), 'w') as output:
            stream = output.add_stream('ffv1' if gray_depth else 'libx264rgb', rate=fps,
                options={'level': '3'} if gray_depth else {'crf': '0', 'preset': 'fast', 'g': str(fps), 'bf': '0'})
            pixel_format = 'gray16le' if gray_depth else 'rgb24'
            stream.width, stream.height, stream.pix_fmt = w, h, pixel_format
            stream.codec_context.thread_count = 2
            for path in paths:
                source = _read_source(path, depth)
                pixels = depth_to_gray16(source) if gray_depth else (pack_depth(source) if depth else source)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format=pixel_format)):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        count = 0
        with av.open(str(temporary)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for count, frame in enumerate(container.decode(video=0), 1):
                if count > len(paths):
                    raise ValueError('Encoded video has excess frames')
                pixels = frame.to_ndarray(format='gray16le' if gray_depth else 'rgb24')
                actual = gray16_to_depth(pixels) if gray_depth else (unpack_depth(pixels) if depth else pixels)
                source = _read_source(paths[count-1], depth)
                if gray_depth and not np.array_equal(pixels, depth_to_gray16(source)):
                    raise ValueError(f'Lossless gray16 verification failed at frame {count-1}')
                if not np.array_equal(actual, source):
                    raise ValueError(f'Lossless video verification failed at frame {count-1}')
        if count != len(paths):
            raise ValueError('Encoded video frame count mismatch')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def decode_lossless_video(video_path, timestamps, tolerance_s, encoding):
    """Return RGB float32 [0,1] or exact depth float32 mm, shape T,C,H,W."""
    if encoding not in (RGB_ENCODING, DEPTH_ENCODING, GRAY_DEPTH_ENCODING):
        raise ValueError(f'Unknown lossless encoding: {encoding}')
    if not timestamps:
        raise ValueError('timestamps must not be empty')
    # Matroska timestamps are quantized to 1 ms, unlike MP4 time bases.
    if encoding == GRAY_DEPTH_ENCODING:
        tolerance_s = max(tolerance_s, 0.000501)
    found = [None] * len(timestamps)
    distances = [float('inf')] * len(timestamps)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        container.seek(max(0, int((min(timestamps)-tolerance_s) / stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * frame.time_base)
            if t > max(timestamps) + tolerance_s:
                break
            indices = [i for i, q in enumerate(timestamps) if abs(t-q) <= tolerance_s and abs(t-q) < distances[i]]
            if indices:
                if encoding == GRAY_DEPTH_ENCODING:
                    array = gray16_to_depth(frame.to_ndarray(format='gray16le'))[None].astype(np.float32)
                else:
                    rgb = frame.to_ndarray(format='rgb24')
                    array = unpack_depth(rgb)[None].astype(np.float32) if encoding == DEPTH_ENCODING else rgb.transpose(2,0,1).astype(np.float32) / 255.0
                for i in indices:
                    found[i], distances[i] = torch.from_numpy(array.copy()), abs(t-timestamps[i])
    if any(frame is None for frame in found):
        raise ValueError(f'Video frame missing within tolerance: {video_path}, {timestamps}')
    return torch.stack(found)


def read_v30_lossless_video(root, episode_index, key, timestamps):
    """Read this recorder's v3 MP4 via episode metadata, including time offsets."""
    import pyarrow.parquet as pq
    root = Path(root)
    info = json.loads((root/'meta/info.json').read_text())
    rows = (row for path in sorted((root/'meta/episodes').rglob('*.parquet')) for row in pq.read_table(path).to_pylist())
    row = next(row for row in rows if row['episode_index'] == episode_index)
    prefix = f'videos/{key}'
    path = root / info['video_path'].format(video_key=key, chunk_index=row[prefix+'/chunk_index'], file_index=row[prefix+'/file_index'])
    path = feature_video_path(path, info['features'][key])
    offset = row[prefix+'/from_timestamp']
    feature_info = info['features'][key].get('info') or {}
    encoding = feature_info.get('lossless_encoding')
    if encoding is None and not feature_info.get('is_depth_map') and not feature_info.get('video.is_depth_map'):
        encoding = RGB_ENCODING  # standard RGB video also uses the explicit software RGB decode
    return decode_lossless_video(path, [t+offset for t in timestamps], 0.5/info['fps'] + 1e-6, encoding)
