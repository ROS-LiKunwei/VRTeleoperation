"""data_100 RGB encoding with unchanged native geometry and gray16 depth.

Creates a separate finalized v3 dataset. The source is never overwritten.
"""
import json
import shutil
from pathlib import Path

import av

from .lossless_video import RGB_ENCODING, GRAY_DEPTH_ENCODING
from .video_utils import get_video_info


def export_data100(source, target):
    source, target = Path(source), Path(target)
    staging = target.with_name(target.name + '.partial')
    if target.exists() or staging.exists() or target.resolve().is_relative_to(source.resolve()):
        raise ValueError('Output must be a new directory outside the source dataset')
    info = json.loads((source/'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0':
        raise ValueError('Export requires finalized v3.0 lossless source')
    video_keys = [key for key,ft in info['features'].items() if ft['dtype']=='video']
    for key in video_keys:
        if (info['features'][key].get('info') or {}).get('lossless_encoding') not in (RGB_ENCODING,GRAY_DEPTH_ENCODING):
            raise ValueError(f'{key}: expected lossless RGB or current gray16 MKV')
    staging.mkdir(parents=True)
    for folder in ('meta','data'):
        shutil.copytree(source/folder,staging/folder)
    counts = {}
    for key in video_keys:
        depth=info['features'][key]['info']['lossless_encoding']==GRAY_DEPTH_ENCODING
        paths=sorted((source/'videos'/key).rglob('*.mkv' if depth else '*.mp4'))
        if len(paths)!=info['total_episodes']:
            raise ValueError(f'{key}: expected one video file per episode')
        for path in paths:
            dest=staging/path.relative_to(source)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if depth:
                shutil.copy2(path,dest)
                continue
            with av.open(str(path)) as inp, av.open(str(dest),'w') as out:
                src=inp.streams.video[0]
                src.codec_context.thread_count=2
                dst=out.add_stream('libsvtav1',rate=src.average_rate,options={'crf':'30','preset':'8','g':'2'})
                dst.width,dst.height,dst.pix_fmt=src.width,src.height,'yuv420p'
                dst.codec_context.thread_count=2
                count=0
                for frame in inp.decode(video=0):
                    # Explicit software RGB decode: do not use browser-rendered frames.
                    rgb=frame.to_ndarray(format='rgb24')
                    for packet in dst.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                        out.mux(packet)
                    count+=1
                for packet in dst.encode(): out.mux(packet)
            with av.open(str(dest)) as verify:
                verify.streams.video[0].codec_context.thread_count=2
                actual=sum(1 for _ in verify.decode(video=0))
            if actual!=count:
                raise ValueError(f'Preview frame count mismatch: {path}')
            counts[str(path.relative_to(source))]=count
        if not depth:
            info['features'][key]['info']=get_video_info(staging/paths[0].relative_to(source))
    (staging/'meta/info.json').write_text(json.dumps(info,indent=2)+'\n')
    report={'source_lossless_dataset':str(source.resolve()),'source_unchanged':True,
        'rgb':'AV1/YUV420 MP4; lossy, like data_100; software rgb24 decode before encoding',
        'depth':'Unchanged FFV1 gray16 MKV; copied byte-for-byte; reversible 0..1 m mapping',
        'geometry':'native resolution, alignment and serial camera names unchanged',
        'frame_counts':counts,
        'stats':'Source pre-encoding statistics retained, as in the recorder; RGB compression can change pixels'}
    (staging/'meta/export_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    staging.rename(target)
    return target
