import json
import numpy as np
import pyarrow.parquet as pq
import pytest
from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from beavr.lerobot.common.datasets.lossless_video import RGB_ENCODING, DEPTH_ENCODING, GRAY_DEPTH_ENCODING, read_v30_lossless_video
from beavr.lerobot.common.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset_v21_to_v30

@pytest.mark.parametrize('async_encoding', [False, True])
@pytest.mark.parametrize('encoding', [DEPTH_ENCODING, GRAY_DEPTH_ENCODING])
def test_exact_mp4_dataset(tmp_path, async_encoding, encoding):
    # Exercise every uint16 value, including byte boundaries and invalid zero.
    depth=np.arange(65536,dtype=np.uint16).reshape(256,256,1)
    if encoding == GRAY_DEPTH_ENCODING:
        depth = depth % 1001
    rgb=np.random.default_rng(42).integers(0,256,(256,256,3),dtype=np.uint8)
    keys=['observation.images.head','observation.images.head_depth']
    features={keys[0]:{'dtype':'video','shape':rgb.shape,'names':['height','width','channels'],'info':{'lossless_encoding':RGB_ENCODING}},
              keys[1]:{'dtype':'video','shape':depth.shape,'names':['height','width','channels'],'info':{'is_depth_map':True,'depth_unit':'mm','lossless_encoding':encoding}}}
    ds=LeRobotDataset.create('test/lossless',fps=30,root=tmp_path/'data',features=features,image_writer_threads=2,keep_image_files=False)
    ds.set_async_video_encoding(async_encoding)
    try:
        for ep in range(2):
            for i in range(3): ds.add_frame({keys[0]:np.roll(rgb,i,axis=0),keys[1]:np.roll(depth,i,axis=0),'task':'test'})
            ds.save_episode()
        ds.wait_for_async_video_encoding(shutdown=True)
        for ep in range(2):
            frames=ds._query_videos({k:[0,1/30,2/30] for k in keys},ep)
            np.testing.assert_array_equal((frames[keys[0]].numpy()*255).round().astype(np.uint8),np.stack([np.roll(rgb,i,axis=0).transpose(2,0,1) for i in range(3)]))
            np.testing.assert_array_equal(frames[keys[1]].numpy(),np.stack([np.roll(depth,i,axis=0).transpose(2,0,1) for i in range(3)]))
        assert not list(ds.root.rglob('*.png'))
        assert ds.meta.stats[keys[1]]['max'].item()==int(depth.max())
        convert_dataset_v21_to_v30(ds.root)
        assert len(list(ds.root.rglob('*.mp4')))==(2 if encoding == GRAY_DEPTH_ENCODING else 4)
        assert len(list(ds.root.rglob('*.mkv')))==(2 if encoding == GRAY_DEPTH_ENCODING else 0)
        assert not list(ds.root.rglob('*.png'))
        table=pq.read_table(next((ds.root/'data').rglob('*.parquet')))
        assert not any(k in table.column_names for k in keys)
        for ep in range(2):
            for k,source in zip(keys,[rgb,depth]):
                result=read_v30_lossless_video(ds.root,ep,k,[2/30,0,1/30]).numpy()
                if k==keys[0]: result=(result*255).round().astype(np.uint8)
                np.testing.assert_array_equal(result,np.stack([np.roll(source,i,axis=0).transpose(2,0,1) for i in [2,0,1]]))
    finally: ds.stop_image_writer()


def test_failed_verification_preserves_source(tmp_path, monkeypatch):
    from PIL import Image
    from beavr.lerobot.common.datasets import lossless_video
    source=tmp_path/'frame_000000.png'
    Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(source)
    monkeypatch.setattr(lossless_video.np, 'array_equal', lambda *args: False)
    with pytest.raises(ValueError, match='verification failed'):
        lossless_video.encode_lossless_video(tmp_path, tmp_path/'video.mp4',30)
    assert source.exists()
    assert not (tmp_path/'video.mp4').exists()
    assert not (tmp_path/'video.partial.mp4').exists()
