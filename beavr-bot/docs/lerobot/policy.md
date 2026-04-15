# Policy Training and Evaluation

Training is performed using the datasets produced by `control_robot.py`. Policies are implemented inside `beavr.lerobot.common.policies`.

Example training invocation:

```bash
python beavr/scripts/train.py \
  --dataset.repo_id=$USER/my_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_my_dataset
```

To evaluate a trained policy, provide the `--control.policy.path` argument when running `control_robot.py`:

```bash
python beavr/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.policy.path=outputs/train/act_my_dataset/checkpoints/080000/pretrained_model
```
