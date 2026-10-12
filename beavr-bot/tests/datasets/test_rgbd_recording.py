"""Check that depth values survive all dataset storage paths without RGB conversion."""
import io

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from beavr.lerobot.common.datasets.utils import hf_transform_to_torch
from beavr.lerobot.common.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset_v21_to_v30


@pytest.mark.parametrize('async_encoding', [False, True])
@pytest.mark.parametrize('keep_images', [False, True])
def test_depth_roundtrip(tmp_path, async_encoding, keep_images):
    key = 'observation.images.front_depth'
    rgb_key = 'observation.images.front'
    features = {
        key: {'dtype': 'image', 'shape': (64, 64, 1), 'names': ['height', 'width', 'channels'],
              'info': {'is_depth_map': True, 'depth_unit': 'mm'}},
        rgb_key: {'dtype': 'video', 'shape': (64, 64, 3), 'names': ['height', 'width', 'channels']},
    }
    ds = LeRobotDataset.create('test/rgbd', fps=10, root=tmp_path / 'dataset', features=features,
                              keep_image_files=keep_images, depth_video=True, image_writer_threads=2)
    ds.set_async_video_encoding(async_encoding)
    depth = np.resize(np.array([0, 1, 255, 256, 1000, 32768, 65535], dtype=np.uint16), (64, 64, 1))
    try:
        for episode in range(2):
            for _ in range(3):
                ds.add_frame({key: depth.copy(), rgb_key: np.zeros((64, 64, 3), dtype=np.uint8), 'task': 'test'})
            ds.save_episode()
        ds.wait_for_async_video_encoding(shutdown=True)
    finally:
        ds.stop_image_writer()
    pngs = list((ds.root / 'images').rglob('*.png'))
    assert bool(pngs) == keep_images
    for png in pngs:
        if png.parent.parent.name.endswith('_depth'):
            np.testing.assert_array_equal(np.array(Image.open(png)), depth[..., 0])
    for episode in range(2):
        table = pq.read_table(ds.root / ds.meta.get_data_file_path(episode))
        for item in table[key].to_pylist():
            image = Image.open(io.BytesIO(item['bytes']))
            np.testing.assert_array_equal(np.array(image), depth[..., 0])
            loaded = hf_transform_to_torch({key: [image]})[key][0].numpy()
            np.testing.assert_array_equal(loaded[0], depth[..., 0])
        with av.open(str(ds.root / 'videos' / key / f'episode-{episode:06d}.mp4')) as video:
            assert len(list(video.decode(video=0))) == 3
    assert ds.meta.stats[key]['max'].item() == 65535
    assert ds.meta.stats[key]['mean'].shape == (1, 1, 1)
    settings_path = ds.root / "meta" / "camera_settings.json"
    settings_path.write_text('{"schema_version":1,"cameras":{}}')
    convert_dataset_v21_to_v30(ds.root)
    assert settings_path.read_text() == '{"schema_version":1,"cameras":{}}'
    assert bool(list((ds.root / 'images').rglob('*.png'))) == keep_images
    assert len(list((ds.root / 'videos' / key).glob('*.mp4'))) == 2
    table = pq.read_table(next((ds.root / 'data').rglob('*.parquet')))
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(table[key][0].as_py()['bytes']))), depth[..., 0])


def test_rerecord_removes_pending_images(tmp_path):
    key = 'observation.images.front_depth'
    ds = LeRobotDataset.create('test/rerecord', fps=10, root=tmp_path / 'dataset',
        features={key: {'dtype': 'image', 'shape': (64, 64, 1), 'names': ['height', 'width', 'channels']}},
        keep_image_files=True, image_writer_threads=2)
    try:
        ds.add_frame({key: np.ones((64, 64, 1), dtype=np.uint16), 'task': 'test'})
        ds.clear_episode_buffer()
        assert not list((ds.root / 'images').rglob('*.png'))
    finally:
        ds.stop_image_writer()
