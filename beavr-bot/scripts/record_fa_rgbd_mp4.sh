#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec .venv/bin/python src/beavr/scripts/control_robot.py \
  --robot.type=fa_adapter \
  --control.type=record \
  --control.read_only=true \
  --control.manage_teleop_state=false \
  --teleop.start=false \
  --control.camera_settings_path=configs/cameras/sysmo32_rgbd_aligned_native.json \
  --control.save_depth=true \
  --control.keep_image_files=false \
  --control.video=true \
  --control.depth_video=false \
  --control.lossless_video=true \
  --control.export_previews=true \
  --control.export_data100=false \
  --control.depth_video_format=gray16_mkv \
  --control.fps=30 \
  --control.num_episodes=1 \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=5 \
  --control.root="./datasets/fa_rgbd_mp4_$(date +%Y%m%d_%H%M%S)" \
  --control.repo_id=local/fa_rgbd_mp4 \
  --control.single_task="teleoperate FA" \
  --control.push_to_hub=false \
  --control.play_sounds=true \
  --control.dataset_format=v3.0 \
  "$@"
