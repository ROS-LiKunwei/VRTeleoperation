import json
import shutil
import numpy as np
from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from beavr.lerobot.common.datasets.lossless_video import RGB_ENCODING,GRAY_DEPTH_ENCODING,read_v30_lossless_video
from beavr.scripts.finalize_rgbd_dataset import finalize


def test_recover_deferred_recording(tmp_path):
    root=tmp_path/'interrupted'
    rgb=np.full((64,64,3),[40,80,120],dtype=np.uint8)
    depth=np.resize(np.arange(1001,dtype=np.uint16),(64,64,1))
    rk,dk='observation.images.head','observation.images.head_depth'
    f={rk:{'dtype':'video','shape':rgb.shape,'names':['height','width','channels'],'info':{'lossless_encoding':RGB_ENCODING}},dk:{'dtype':'video','shape':depth.shape,'names':['height','width','channels'],'info':{'is_depth_map':True,'lossless_encoding':GRAY_DEPTH_ENCODING}},'observation.state':{'dtype':'float32','shape':(2,),'names':['a','b']},'action':{'dtype':'float32','shape':(2,),'names':['a','b']}}
    ds=LeRobotDataset.create('test/recovery',fps=30,root=root,features=f,image_writer_threads=1)
    ds.set_async_video_encoding(True)
    for _ in range(3):ds.add_frame({rk:rgb,dk:depth,'observation.state':np.zeros(2,dtype=np.float32),'action':np.ones(2,dtype=np.float32),'task':'test'})
    ds.save_episode();ds.stop_image_writer()
    pending=root/'images'/rk/'episode_000001';pending.mkdir()
    shutil.copy2(root/'images'/rk/'episode_000000/frame_000000.png',pending/'frame_000000.png')
    marker=root/"meta/recording_faults/episode_000000.json"
    marker.parent.mkdir()
    marker.write_text('{"disposition":"interrupted_prefix_requires_review"}')
    finalize(root,1)
    assert json.loads(marker.read_text())["disposition"]=="interrupted_prefix_requires_review"
    assert json.loads((root/'meta/info.json').read_text())['codebase_version']=='v3.0'
    assert len(list((root/'previews').rglob('*.mp4')))==2
    assert not (root/'images').exists()
    archive=root.with_name(root.name+'_recovery')
    assert (archive/'images'/rk/'episode_000001/frame_000000.png').exists()
    assert not (archive/'images'/rk/'episode_000000').exists()
    np.testing.assert_array_equal(read_v30_lossless_video(root,0,dk,[0])[0].numpy(),depth.transpose(2,0,1))
    assert json.loads((root/'meta/finalization_report.json').read_text())['phase']=='complete'
    finalize(root,1)  # Completed job can be rerun without corrupting the dataset.
