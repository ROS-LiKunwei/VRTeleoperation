import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from beavr.lerobot.common.datasets.lossless_video import RGB_ENCODING,GRAY_DEPTH_ENCODING,read_v30_lossless_video
from beavr.lerobot.common.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset_v21_to_v30
from beavr.lerobot.common.datasets.data100_export import export_data100


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def test_data100_keeps_original_and_depth(tmp_path):
    rgb_key,depth_key='observation.images.head','observation.images.head_depth'
    rgb=np.zeros((128,128,3),dtype=np.uint8)
    rgb[:64,:64]=[220,20,30];rgb[64:,:64]=[20,200,30];rgb[:,64:]=[20,30,210]
    depth=np.resize(np.arange(1001,dtype=np.uint16),(128,128,1))
    features={rgb_key:{'dtype':'video','shape':rgb.shape,'names':['height','width','channels'],'info':{'lossless_encoding':RGB_ENCODING}},
        depth_key:{'dtype':'video','shape':depth.shape,'names':['height','width','channels'],'info':{'is_depth_map':True,'lossless_encoding':GRAY_DEPTH_ENCODING}},
        'observation.state':{'dtype':'float32','shape':(2,),'names':['a','b']},
        'action':{'dtype':'float32','shape':(2,),'names':['a','b']}}
    source=tmp_path/'original'
    ds=LeRobotDataset.create('test/original',fps=30,root=source,features=features)
    try:
        for ep in range(2):
            for i in range(3):ds.add_frame({rgb_key:rgb,depth_key:depth,'observation.state':np.array([ep,i],dtype=np.float32),'action':np.array([ep,i+1],dtype=np.float32),'task':'test'})
            ds.save_episode()
    finally:ds.stop_image_writer()
    convert_dataset_v21_to_v30(source)
    before=hashes(source)
    output=export_data100(source,tmp_path/'view')
    assert before==hashes(source)
    info=json.loads((output/'meta/info.json').read_text())
    assert info['features'][rgb_key]['info']['video.codec']=='av1'
    assert info['features'][rgb_key]['info']['video.pix_fmt']=='yuv420p'
    assert 'lossless_encoding' not in info['features'][rgb_key]['info']
    assert info['features'][depth_key]['info']['lossless_encoding']==GRAY_DEPTH_ENCODING
    for rel,sha in before.items():
        if rel.endswith('.mkv') or rel.startswith('data/'):
            assert hashlib.sha256((output/rel).read_bytes()).hexdigest()==sha
    assert not list(output.rglob('*.png'))
    for ep in range(2):
        result=read_v30_lossless_video(output,ep,rgb_key,[0,1/30,2/30]).numpy()
        assert np.abs(result*255-rgb.transpose(2,0,1)[None]).mean()<5
        result=read_v30_lossless_video(output,ep,depth_key,[0,1/30,2/30]).numpy()
        np.testing.assert_array_equal(result,np.repeat(depth.transpose(2,0,1)[None],3,axis=0))

    from beavr.scripts.preview_lossless_rgbd import export_previews
    import av
    before = hashes(source)
    preview = export_previews(source, source/'previews')
    for rel, sha in before.items():
        assert hashlib.sha256((source/rel).read_bytes()).hexdigest() == sha
    assert len(list(preview.rglob('*.mp4'))) == 4
    assert not list(source.rglob('*.png'))
    for video in preview.rglob('*.mp4'):
        with av.open(str(video)) as container:
            frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
            assert len(frames) == 3
            assert container.streams.video[0].pix_fmt == 'yuv420p'
            if '_depth' in str(video):
                expected = np.rint(depth[...,0].astype(float)*255/1000)
                assert np.abs(frames[0][...,0].astype(float)-expected).mean() < 5
            else:
                assert np.abs(frames[0].astype(float)-rgb).mean() < 5
