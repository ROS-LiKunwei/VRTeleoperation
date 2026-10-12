"""Export display-only MP4 copies; never alter lossless training videos."""
import argparse
from pathlib import Path
import json
import av
import cv2
import numpy as np
from beavr.lerobot.common.datasets.lossless_video import DEPTH_ENCODING, RGB_ENCODING, GRAY_DEPTH_ENCODING, unpack_depth, gray16_to_depth


def export_previews(root, output, max_frames=0, keys=None):
    root, output = Path(root), Path(output)
    info = json.loads((root/'meta/info.json').read_text())
    if output.resolve().is_relative_to((root/'videos').resolve()):
        raise ValueError('Preview output must not be inside original videos')
    destination = output
    if destination.exists():
        raise FileExistsError(destination)
    output = destination.with_name(destination.name + '.partial')
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    (output/'README.txt').write_text('DISPLAY ONLY: lossy YUV420 RGB and depth visualization. Do not use for training or recover metric depth. Originals are unchanged.\n')
    for key, feature in info['features'].items():
        if keys is not None and key not in keys:
            continue
        encoding = (feature.get('info') or {}).get('lossless_encoding')
        if encoding not in (RGB_ENCODING, DEPTH_ENCODING, GRAY_DEPTH_ENCODING):
            continue
        for source in sorted((root/'videos'/key).rglob('*')):
            if source.suffix not in ('.mp4', '.mkv'):
                continue
            target = (output/source.relative_to(root/'videos')).with_suffix('.mp4')
            target.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(source)) as inp, av.open(str(target), 'w') as out:
                stream = inp.streams.video[0]
                stream.codec_context.thread_count = 2
                enc = out.add_stream('libx264',rate=stream.average_rate,options={'crf':'18','preset':'fast'})
                enc.width,enc.height,enc.pix_fmt=stream.width,stream.height,'yuv420p'
                enc.codec_context.thread_count=2
                count = 0
                for i,frame in enumerate(inp.decode(video=0)):
                    if max_frames and i >= max_frames:
                        break
                    rgb=frame.to_ndarray(format='rgb24') if encoding != GRAY_DEPTH_ENCODING else None
                    if encoding in (DEPTH_ENCODING, GRAY_DEPTH_ENCODING):
                        depth=gray16_to_depth(frame.to_ndarray(format='gray16le')) if encoding == GRAY_DEPTH_ENCODING else unpack_depth(rgb)
                        # Fixed 0..1000 mm display scale; zero/invalid is black.
                        gray=np.rint(np.clip(depth,0,1000).astype(np.float32)*255/1000).astype(np.uint8)
                        rgb=np.repeat(gray[...,None],3,axis=2)
                        rgb[depth==0]=0
                    count += 1
                    for packet in enc.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                        out.mux(packet)
                for packet in enc.encode(): out.mux(packet)
            with av.open(str(target)) as check:
                check.streams.video[0].codec_context.thread_count = 2
                decoded = sum(1 for _ in check.decode(video=0))
                if decoded != count or check.streams.video[0].pix_fmt != 'yuv420p':
                    raise ValueError(f'Preview validation failed: {target}')
            print('PREVIEW_VERIFIED', target, count, flush=True)
            report[str(target.relative_to(output))] = {'frames':count,'source':str(source.relative_to(root)),
                'depth_display_mm':[0,1000] if encoding != RGB_ENCODING else None}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    output.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args()
    print(export_previews(args.root,args.output,args.max_frames))


if __name__ == '__main__':
    main()
