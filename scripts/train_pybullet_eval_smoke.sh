#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
STATS_PATH="${2:-data/raw_data/realdex_bspline_stats_free8.npz}"
RUN_DIR="${3:-data/outputs/pybullet_eval_smoke}"

cd "$(dirname "$0")/../3D-Diffusion-Policy"

python train.py \
  --config-name=simple_dp3.yaml \
  task=realdex_transition \
  task.dataset.zarr_path=data/realdex_bspline_free8.zarr \
  task.dataset.val_ratio=0.04 \
  horizon=8 \
  n_obs_steps=1 \
  n_action_steps=8 \
  training.pybullet_eval.num_control_points=12 \
  training.debug=False \
  training.resume=False \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.max_val_steps=1 \
  training.rollout_every=999999 \
  training.sample_every=999999 \
  training.checkpoint_every=999999 \
  training.device="${DEVICE}" \
  training.use_ema=False \
  training.pybullet_eval.enabled=True \
  training.pybullet_eval.stats_path="${STATS_PATH}" \
  training.pybullet_eval.stats_mode=bspline \
  training.pybullet_eval.max_episodes=1 \
  training.pybullet_eval.robot_surface_points_per_link=64 \
  training.pybullet_eval.log_legacy_pybullet_metrics=False \
  logging.mode=offline \
  exp_name=pybullet-eval-smoke \
  dataloader.batch_size=1 \
  dataloader.num_workers=0 \
  val_dataloader.batch_size=1 \
  val_dataloader.num_workers=0 \
  checkpoint.save_ckpt=False \
  checkpoint.generalization_gap.enabled=False \
  training.early_stop.enabled=False \
  hydra.run.dir="${RUN_DIR}"
