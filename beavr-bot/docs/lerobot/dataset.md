# Dataset Creation

Datasets are recorded with `control_robot.py`. The script can teleoperate a robot live, record demonstrations and push them to the Hugging Face Hub.

Basic usage:

```bash
python beavr/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.repo_id=$USER/my_dataset \
  --control.num_episodes=50
```

During recording you can use the keyboard shortcuts defined in the script to restart or stop episodes. Recorded data is stored locally and can optionally be uploaded.
