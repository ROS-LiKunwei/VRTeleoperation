"""Resume interrupted local RGB-D recording without robot/camera access.

All PNGs are retained until videos, v3 metadata, and previews have passed checks.
Uncommitted episodes remain in a sibling recovery directory.
"""
import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import av
import numpy as np
import pyarrow.parquet as pq

from beavr.lerobot.common.datasets.compute_stats import compute_episode_stats
from beavr.lerobot.common.datasets.lossless_video import encode_lossless_video, feature_video_path, GRAY_DEPTH_ENCODING
from beavr.lerobot.common.datasets.utils import serialize_dict
from beavr.lerobot.common.datasets.video_utils import get_video_info
from beavr.lerobot.common.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset_v21_to_v30
from beavr.scripts.preview_lossless_rgbd import export_previews


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix('.writing')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def finalize(root, workers=3):
    root=Path(root).resolve()
    archive=root.with_name(root.name+'_recovery')
    archive.mkdir(exist_ok=True)
    backup=archive/'original_meta'
    if not backup.exists():shutil.copytree(root/'meta',backup)
    info=json.loads((backup/'info.json').read_text())
    if info['codebase_version']!='v2.1':raise ValueError('Recovery requires original v2.1 metadata')
    episodes=[json.loads(s) for s in (backup/'episodes.jsonl').read_text().splitlines()]
    if sum(e['length'] for e in episodes)!=info['total_frames']:raise ValueError('Frame count mismatch')
    keys=[k for k,f in info['features'].items() if f['dtype']=='video']
    progress_path=archive/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'videos':{},'phase':'encoding'}
    images=root/'images' if (root/'images').exists() else archive/'images'
    phase=json.loads((root/'meta/info.json').read_text())['codebase_version']
    if phase=='v2.1':
        def work(ep,key):
            epidx=ep['episode_index']; token=f'{epidx}:{key}'
            folder=images/key/f'episode_{epidx:06d}'
            files=sorted(folder.glob('frame_*.png'))
            if [p.name for p in files]!=[f'frame_{i:06d}.png' for i in range(ep['length'])]:
                raise ValueError(f'Incomplete PNG sequence: {folder}')
            path=root/feature_video_path(info['video_path'].format(episode_chunk=epidx//info['chunks_size'],episode_index=epidx,video_key=key),info['features'][key])
            saved=progress['videos'].get(token)
            if saved and path.exists() and saved['sha256']==digest(path):return token,saved
            stats=compute_episode_stats({key:[str(p) for p in files]}, {key:info['features'][key]})
            gray=info['features'][key]['info']['lossless_encoding']==GRAY_DEPTH_ENCODING
            encode_lossless_video(folder,path,info['fps'],gray_depth=gray)
            return token,{'path':str(path.relative_to(root)),'frames':len(files),'sha256':digest(path),'stats':serialize_dict(stats)}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(work,ep,k) for ep in episodes for k in keys]
            for future in as_completed(futures):
                token,result=future.result();progress['videos'][token]=result
                atomic_json(progress_path,progress)
                print('VERIFIED',len(progress['videos']),len(futures),token,flush=True)
        stats_rows=[]; data_hashes={}
        for ep in episodes:
            idx=ep['episode_index'];path=root/info['data_path'].format(episode_chunk=idx//info['chunks_size'],episode_index=idx)
            table=pq.read_table(path)
            if len(table)!=ep['length']:raise ValueError(f'Parquet length mismatch {idx}')
            nonvisual={k:np.asarray(table[k].to_pylist()) for k in table.column_names}
            for key in ('observation.state','action'):
                if not np.isfinite(nonvisual[key]).all():raise ValueError('Non-finite state/action')
            stats=serialize_dict(compute_episode_stats(nonvisual,info['features']))
            for k in keys:stats.update(progress['videos'][f'{idx}:{k}']['stats'])
            stats_rows.append({'episode_index':idx,'stats':stats})
            data_hashes[str(idx)]=digest(path)
        (root/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in stats_rows))
        for k in keys:
            path=root/progress['videos'][f"{episodes[0]['episode_index']}:{k}"]['path']
            info['features'][k]['info']={**get_video_info(path),**info['features'][k]['info']}
            if info['features'][k]['info'].get('is_depth_map'):info['features'][k]['info']['video.is_depth_map']=True
        atomic_json(root/'meta/info.json',info)
        progress.update(phase='converting',data_sha256=data_hashes);atomic_json(progress_path,progress)
        # Moving staging images avoids duplicating 87 GB in the v3 converter.
        if images==root/'images':images.rename(archive/'images')
        convert_dataset_v21_to_v30(root)
    current=json.loads((root/'meta/info.json').read_text())
    if current['codebase_version']!='v3.0':raise ValueError('v3 conversion incomplete')
    for ep in episodes:
        idx=ep['episode_index']
        p=root/current['data_path'].format(chunk_index=idx//info['chunks_size'],file_index=idx%info['chunks_size'])
        if digest(p)!=progress['data_sha256'][str(idx)]:raise ValueError('Parquet changed during conversion')
        for k in keys:
            p=root/feature_video_path(current['video_path'].format(video_key=k,chunk_index=idx//info['chunks_size'],file_index=idx%info['chunks_size']),current['features'][k])
            if digest(p)!=progress['videos'][f'{idx}:{k}']['sha256']:raise ValueError('Video changed during conversion')
    progress['phase']='previews';atomic_json(progress_path,progress)
    parts=archive/'preview_parts';parts.mkdir(exist_ok=True)
    def preview(key):
        out=parts/key
        if not (out/'manifest.json').exists():
            # A failed preview is derived data; it can be recreated from verified originals.
            partial=out.with_name(out.name+'.partial')
            if partial.exists():shutil.rmtree(partial)
            export_previews(root,out,keys=[key])
        report=json.loads((out/'manifest.json').read_text())
        if len(report)!=len(episodes):raise ValueError('Preview episode count mismatch')
        return key,report
    reports={}
    if not (root/'previews/manifest.json').exists():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(preview,k) for k in keys]
            for future in as_completed(futures):
                key,report=future.result();reports.update(report);print('PREVIEW_CAMERA_DONE',key,flush=True)
        dest=root/'previews.partial';dest.mkdir(exist_ok=True)
        for k in keys:
            src=parts/k/k
            if src.exists():src.rename(dest/k)
        (dest/'README.txt').write_text('Display-only RGB and grayscale depth MP4. Use videos/ for lossless training data.\n')
        atomic_json(dest/'manifest.json',reports)
        dest.rename(root/'previews')
    report=json.loads((root/'previews/manifest.json').read_text())
    if len(report)!=len(episodes)*len(keys):raise ValueError('Incomplete previews')
    for record in report.values():
        filename=Path(record['source']).stem
        idx=int(filename.split('-')[-1])
        if record['frames']!=episodes[idx]['length']:raise ValueError('Preview frame count mismatch')
    # Final durable proof is saved before removing only verified committed PNG episodes.
    progress['phase']='verified_cleanup';atomic_json(progress_path,progress)
    completed={f"episode_{e['episode_index']:06d}" for e in episodes}
    for k in keys:
        for name in completed:
            p=archive/'images'/k/name
            if p.exists():shutil.rmtree(p)
    progress.update(phase='complete',completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),episodes=len(episodes),frames=info['total_frames'],original_videos=len(keys)*len(episodes),preview_videos=len(report))
    atomic_json(progress_path,progress)
    shutil.copy2(progress_path,root/'meta/finalization_report.json')
    print('FINALIZATION_COMPLETE',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    finalize(args.root,args.workers)
