eavr-bot) likunwei@likunwei:~/dataCollection/beavr-bot$ PYTHONPATH=src python src/beavr/s(beavr-bot) likunwei@likunwei:~/dataCollection/beavr-bot$ PYTHONPATH=src python src/beavr/scripts/control_robot.py \ \
  --robot.type=fa_adapter \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=local/fa_test \
  --control.root=datasets \
  --control.episode=0
INFO 2026-06-24 18:13:13 ol_robot.py:650 {'control': {'episode': 0,
             'fps': 30,
             'play_sounds': False,
             'repo_id': 'local/fa_test',
             'root': 'datasets'},
 'robot': {'cameras': {'front': {'camera_index': 0,
                                 'channels': 3,
                                 'color_mode': 'rgb',
                                 'fps': 30,
                                 'height': 480,
                                 'mock': False,
                                 'rotation': 180,
                                 'width': 640}},
           'mock': False,
           'record_actions': True,
           'record_bimanual_gripper_state': True,
           'record_next_joint_state_action': True,
           'robot_configs': [{'action_key': 'left_arm_next_joint_action',
                              'command_state_path': ['commanded_cartesian_state',
                                                     'commanded_cartesian_position'],
                              'command_topic': 'endeff_coords',
                              'endeff_publish_port': 10013,
                              'home_subscribe_port': 10007,
                              'host': '192.168.220.157',
                              'joint_count': 7,
                              'joint_state_path': ['joint_states',
                                                   'joint_position'],
                              'name': 'fa_left',
                              'observation_key': 'left_arm_state',
                              'robot_type': 'arm',
                              'state_port': 10020,
                              'state_topic': 'fa_left',
                              'teleop_port': 8089},
                             {'action_key': 'right_arm_next_joint_action',
                              'command_state_path': ['commanded_cartesian_state',
                                                     'commanded_cartesian_position'],
                              'command_topic': 'endeff_coords',
                              'endeff_publish_port': 10011,
                              'home_subscribe_port': 10007,
                              'host': '192.168.220.157',
                              'joint_count': 7,
                              'joint_state_path': ['joint_states',
                                                   'joint_position'],
                              'name': 'fa_right',
                              'observation_key': 'right_arm_state',
                              'robot_type': 'arm',
                              'state_port': 10018,
                              'state_topic': 'fa_right',
                              'teleop_port': 8089}],
           'robot_type': 'fa'},
 'teleop': {'laterality': None,
            'operate': None,
            'robot_name': None,
            'start': True}}
INFO 2026-06-24 18:13:13 beavrbot.py:281 Connected to camera: front
INFO 2026-06-24 18:13:13 beavrbot.py:283 Started async reading for camera: front
INFO 2026-06-24 18:13:13 beavrbot.py:290 MultiRobotAdapter (fa) connected successfully
WARNING 2026-06-24 18:13:13 ts/utils.py:333 
The dataset you requested (local/fa_test) is in 3.0 format.
While current version of LeRobot is backward-compatible with it, the version of your dataset still uses global
stats instead of per-episode stats. Update your dataset stats to the new format using this command:
```
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py --repo-id=local/fa_test
```

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).

INFO 2026-06-24 18:13:19 beavrbot.py:301 Disconnected from camera: front
INFO 2026-06-24 18:13:19 beavrbot.py:306 MultiRobotAdapter (fa) disconnected successfully
INFO 2026-06-24 18:13:19 beavrbot.py:561 Teleop stop acknowledged by subscribers
INFO 2026-06-24 18:13:19 ol_robot.py:693 Starting cleanup...
INFO 2026-06-24 18:13:19 ol_robot.py:699 Cleanup complete. Exiting.
Traceback (most recent call last):
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/lerobot_dataset.py", line 99, in __init__
    self.load_metadata()
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/lerobot_dataset.py", line 111, in load_metadata
    self.tasks, self.task_to_task_index = load_tasks(self.root)
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/utils.py", line 229, in load_tasks
    tasks = load_jsonlines(local_dir / TASKS_PATH)
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/utils.py", line 176, in load_jsonlines
    with jsonlines.open(fpath, "r") as reader:
  File "/home/likunwei/dataCollection/beavr-bot/.venv/lib/python3.10/site-packages/jsonlines/jsonlines.py", line 643, in open
    fp = builtins.open(file, mode=mode + "t", encoding=encoding)
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/meta/tasks.jsonl'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/likunwei/dataCollection/beavr-bot/.venv/lib/python3.10/site-packages/huggingface_hub/utils/_http.py", line 402, in hf_raise_for_status
    response.raise_for_status()
  File "/home/likunwei/dataCollection/beavr-bot/.venv/lib/python3.10/site-packages/requests/models.py", line 1026, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/datasets/local/fa_test/refs

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/scripts/control_robot.py", line 703, in <module>
    control_robot()
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/configs/parser.py", line 229, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/scripts/control_robot.py", line 687, in control_robot
    replay(robot, cfg.control)
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/robot_devices/utils.py", line 42, in wrapper
    raise e
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/robot_devices/utils.py", line 38, in wrapper
    return func(robot, *args, **kwargs)
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/scripts/control_robot.py", line 609, in replay
    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/lerobot_dataset.py", line 500, in __init__
    self.meta = LeRobotDatasetMetadata(
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/lerobot_dataset.py", line 102, in __init__
    self.revision = get_safe_version(self.repo_id, self.revision)
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/utils.py", line 357, in get_safe_version
    hub_versions = get_repo_versions(repo_id)
  File "/home/likunwei/dataCollection/beavr-bot/src/beavr/lerobot/common/datasets/utils.py", line 339, in get_repo_versions
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
  File "/home/likunwei/dataCollection/beavr-bot/.venv/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
  File "/home/likunwei/dataCollection/beavr-bot/.venv/lib/python3.10/site-packages/huggingface_hub/hf_api.py", line 3251, in list_repo_refs
    hf_raise_for_status(response)
  File "/home/likunwei/dataCollection/beavr-bot/.venv/lib/python3.10/site-packages/huggingface_hub/utils/_http.py", line 452, in hf_raise_for_status
    raise _format(RepositoryNotFoundError, message, response) from e
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-6a3b3d3f-692043496cce6660329b13f1;e3d16b8e-8562-4686-87e5-b03a392e2297)

Repository Not Found for url: https://huggingface.co/api/datasets/local/fa_test/refs.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.
