#!/usr/bin/env python3
"""Standalone PyBullet validation over all validation-set trajectories.

Loads a trained checkpoint, iterates over the validation split of a B-spline
zarr dataset, runs inference + joint-trajectory reconstruction + PyBullet
collision checking for every episode, and writes per-trajectory metrics to JSON
with a terminal summary.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve the 3D-Diffusion-Policy package root so that imports work from
# any working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _PROJECT_ROOT / "3D-Diffusion-Policy"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from train import TrainDP3Workspace  # noqa: E402
from diffusion_policy_3d.common.pybullet_validation import (  # noqa: E402
    PyBulletValidationConfig,
    PyBulletValidationRunner,
    _episode_bounds,
)
from diffusion_policy_3d.dataset.transition_dataset import (  # noqa: E402
    TransitionTrajectoryDataset,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate all trajectories in a dataset split with PyBullet collision detection."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to a trained .ckpt file.",
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="data/realdex_bspline_free10.zarr",
        help="Path to the B-spline zarr dataset.",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="data/raw_data/realdex_bspline_stats.npz",
        help="Path to the B-spline delta_w statistics .npz.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit validation episodes (for smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: analysis_outputs/validation/<timestamp>.",
    )
    parser.add_argument(
        "--num-control-points",
        type=int,
        default=16,
        help="Number of B-spline control points for reconstruction.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override dataset horizon (default: from checkpoint cfg.task.dataset.horizon or cfg.horizon).",
    )
    parser.add_argument(
        "--jobs-root",
        type=str,
        default=None,
        help=(
            "Override the regular workpiece STL root used by PyBullet validation. "
            "Expected layout: <jobs-root>/job_xxx/workpiece.stl"
        ),
    )
    parser.add_argument(
        "--simple-jobs-root",
        type=str,
        default=None,
        help=(
            "Override the simple workpiece STL root used by PyBullet validation. "
            "Expected layout: <simple-jobs-root>/job_xxx/workpiece.stl"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_horizon(workspace: TrainDP3Workspace, cli_horizon: Optional[int]) -> int:
    if cli_horizon is not None:
        return cli_horizon
    # Try OmegaConf select
    from omegaconf import OmegaConf
    dataset_horizon = OmegaConf.select(workspace.cfg, "task.dataset.horizon", default=None)
    if dataset_horizon is not None:
        return int(dataset_horizon)
    return int(workspace.cfg.horizon)


def _build_val_dataset(
    zarr_path: str,
    horizon: int,
    val_ratio: float,
    workspace: TrainDP3Workspace,
) -> TransitionTrajectoryDataset:
    """Instantiate the full dataset and return its validation copy."""
    from omegaconf import OmegaConf

    # Pull dataset-construction args from the checkpoint config when available,
    # falling back to sensible defaults.
    ds_cfg = OmegaConf.select(workspace.cfg, "task.dataset", default={}) or {}
    dataset = TransitionTrajectoryDataset(
        zarr_path=str(zarr_path),
        horizon=horizon,
        pad_before=int(ds_cfg.get("pad_before", 0)),
        pad_after=int(ds_cfg.get("pad_after", 0)),
        seed=int(ds_cfg.get("seed", 42)),
        val_ratio=val_ratio,
        max_train_episodes=ds_cfg.get("max_train_episodes"),
        point_cloud_key=str(ds_cfg.get("point_cloud_key", "point_cloud")),
        obs_keys=tuple(ds_cfg.get("obs_keys", TransitionTrajectoryDataset.DEFAULT_OBS_KEYS)),
        split_by_workpiece=bool(ds_cfg.get("split_by_workpiece", True)),
        stratify_workpiece_split=bool(ds_cfg.get("stratify_workpiece_split", True)),
        simple_workpiece_id_offset=int(ds_cfg.get("simple_workpiece_id_offset", 1000)),
        workpiece_split_strategy=str(ds_cfg.get("workpiece_split_strategy", "tail")),
    )
    return dataset.get_validation_dataset()


def _build_pybullet_config(
    workspace: TrainDP3Workspace,
    stats_path: str,
    num_control_points: int,
    jobs_root: Optional[str] = None,
    simple_jobs_root: Optional[str] = None,
) -> PyBulletValidationConfig:
    """Build PyBulletValidationConfig, seeding from the checkpoint and overriding with CLI."""
    from omegaconf import OmegaConf

    pyb_cfg_raw = OmegaConf.select(workspace.cfg, "training.pybullet_eval", default={}) or {}
    return PyBulletValidationConfig(
        enabled=True,
        stats_path=stats_path,
        stats_mode=str(pyb_cfg_raw.get("stats_mode", "auto")),
        jobs_root=str(jobs_root if jobs_root is not None else pyb_cfg_raw.get("jobs_root", "data/raw_data/jobs")),
        simple_jobs_root=(
            simple_jobs_root
            if simple_jobs_root is not None
            else pyb_cfg_raw.get("simple_jobs_root", "data/raw_data/simple_jobs")
        ),
        simple_workpiece_id_offset=int(pyb_cfg_raw.get("simple_workpiece_id_offset", 1000)),
        job_name_template=str(pyb_cfg_raw.get("job_name_template", "job_{workpiece_id:03d}")),
        workpiece_filename=str(pyb_cfg_raw.get("workpiece_filename", "workpiece.stl")),
        urdf_path=pyb_cfg_raw.get("urdf_path"),
        urdf_package_roots=tuple(pyb_cfg_raw.get("urdf_package_roots", ["config/robot-model"])),
        tcp_link_name=str(pyb_cfg_raw.get("tcp_link_name", "tool0")),
        stl_x_offset_m=float(pyb_cfg_raw.get("stl_x_offset_m", 0.5)),
        collision_distance_threshold=float(pyb_cfg_raw.get("collision_distance_threshold", 0.0)),
        interpolate_for_collision=bool(pyb_cfg_raw.get("interpolate_for_collision", True)),
        max_joint_step_rad=float(pyb_cfg_raw.get("max_joint_step_rad", 0.01)),
        min_interpolated_steps_per_segment=int(pyb_cfg_raw.get("min_interpolated_steps_per_segment", 1)),
        goal_position_norm_m=float(pyb_cfg_raw.get("goal_position_norm_m", 0.1)),
        goal_tolerance_m=float(pyb_cfg_raw.get("goal_tolerance_m", 0.01)),
        num_control_points=num_control_points,
        spline_degree=int(pyb_cfg_raw.get("spline_degree", 5)),
        target_steps=int(pyb_cfg_raw.get("target_steps", 64)),
        max_episodes=None,  # we control this ourselves via --max-episodes
        sdf_filename=str(pyb_cfg_raw.get("sdf_filename", "workpiece_sdf.npz")),
        sdf_required=bool(pyb_cfg_raw.get("sdf_required", True)),
        robot_surface_points_per_link=int(pyb_cfg_raw.get("robot_surface_points_per_link", 256)),
        sdf_out_of_bounds_value_m=pyb_cfg_raw.get("sdf_out_of_bounds_value_m"),
        log_legacy_pybullet_metrics=bool(pyb_cfg_raw.get("log_legacy_pybullet_metrics", True)),
    )


def _build_obs_batch(
    replay_buffer,
    episode_idx: int,
    obs_keys: tuple[str, ...],
    n_obs_steps: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Mirrors PyBulletValidationRunner._build_obs_batch."""
    episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
    start_idx, end_idx = _episode_bounds(episode_ends, episode_idx)
    episode_length = end_idx - start_idx
    if episode_length <= 0:
        raise ValueError(f"Episode {episode_idx} is empty.")

    obs_batch: dict[str, torch.Tensor] = {}
    raw_obs: dict[str, np.ndarray] = {}
    for key in obs_keys:
        value = np.asarray(replay_buffer[key][start_idx:end_idx], dtype=np.float32)
        value = value[:n_obs_steps]
        if value.shape[0] < n_obs_steps:
            pad_count = n_obs_steps - value.shape[0]
            pad = np.repeat(value[-1:], pad_count, axis=0)
            value = np.concatenate([value, pad], axis=0)
        raw_obs[key] = value.copy()
        obs_batch[key] = torch.from_numpy(value[None]).to(device)
    return obs_batch, raw_obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # ---- output directory ----
    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "analysis_outputs" / "validation" / ts
    else:
        output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ---- resolve paths relative to project root ----
    checkpoint_path = pathlib.Path(args.checkpoint_path)
    zarr_path = _PROJECT_ROOT / args.zarr_path
    stats_path = _PROJECT_ROOT / args.stats_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # ---- load checkpoint ----
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device(args.device)
    workspace = TrainDP3Workspace.create_from_checkpoint(
        str(checkpoint_path),
    )
    policy = workspace.model
    policy.to(device)
    policy.eval()

    horizon = _resolve_horizon(workspace, args.horizon)
    n_obs_steps = int(workspace.cfg.n_obs_steps)
    print(f"Checkpoint config: horizon={horizon}, n_obs_steps={n_obs_steps}")

    # ---- dataset ----
    print(f"Loading validation split from: {zarr_path}")
    print(f"  val_ratio={args.val_ratio}")
    val_dataset = _build_val_dataset(
        zarr_path=str(zarr_path),
        horizon=horizon,
        val_ratio=args.val_ratio,
        workspace=workspace,
    )
    val_mask = val_dataset.val_mask
    val_episode_indices = np.flatnonzero(np.asarray(val_mask, dtype=bool))
    total_val = len(val_episode_indices)
    print(f"Validation episodes: {total_val}")

    if args.max_episodes is not None and args.max_episodes < total_val:
        val_episode_indices = val_episode_indices[: args.max_episodes]
        print(f"  (limited to {args.max_episodes} by --max-episodes)")

    replay_buffer = val_dataset.replay_buffer
    obs_keys = val_dataset.obs_keys

    if "workpiece_ids" not in replay_buffer.meta:
        raise KeyError(
            "PyBullet validation requires `meta/workpiece_ids` in the zarr dataset. "
            "Rebuild the dataset with workpiece metadata."
        )
    workpiece_ids = np.asarray(replay_buffer.meta["workpiece_ids"][:], dtype=np.int64)

    # ---- PyBullet ----
    print("Initialising PyBullet validator …")
    pyb_cfg = _build_pybullet_config(
        workspace=workspace,
        stats_path=str(stats_path),
        num_control_points=args.num_control_points,
        jobs_root=args.jobs_root,
        simple_jobs_root=args.simple_jobs_root,
    )
    print(f"  num_control_points={pyb_cfg.num_control_points}")
    print(f"  stats_mode={pyb_cfg.stats_mode}")
    print(f"  jobs_root={pyb_cfg.jobs_root}")
    print(f"  simple_jobs_root={pyb_cfg.simple_jobs_root}")
    runner = PyBulletValidationRunner(pyb_cfg)
    validator = runner.validator

    # ---- per-trajectory loop ----
    per_traj_metrics: list[dict] = []
    collision_count = 0
    total_segment_collision_steps = 0
    total_segment_steps = 0
    sdf_distances = []
    goal_errors = []

    print(f"\nRunning validation on {len(val_episode_indices)} episodes …")
    with torch.no_grad():
        for idx, ep_idx in enumerate(val_episode_indices.tolist()):
            wid = int(workpiece_ids[ep_idx])

            # Build observation batch
            obs_dict, raw_obs = _build_obs_batch(
                replay_buffer=replay_buffer,
                episode_idx=ep_idx,
                obs_keys=obs_keys,
                n_obs_steps=n_obs_steps,
                device=device,
            )

            # Inference
            result = policy.predict_action(obs_dict)
            pred_action_horizon = (
                result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
            )

            # Debug: print expected vs actual
            if idx == 0:
                from diffusion_policy_3d.common.bspline import _resolve_free_control_point_slice
                free_slice = _resolve_free_control_point_slice(pyb_cfg.num_control_points)
                expected_free = free_slice.stop - free_slice.start
                print(f"  [debug] num_control_points={pyb_cfg.num_control_points}")
                print(f"  [debug] expected free CPs: {expected_free}")
                print(f"  [debug] pred_action_horizon shape: {pred_action_horizon.shape}")

            # Reconstruct joint trajectory
            joint_trajectory = validator.reconstruct_joint_trajectory(
                pred_action_horizon=pred_action_horizon,
                start_joint_normalized=raw_obs["first_joint_angles_normalized"][0],
                end_joint_normalized=raw_obs["last_joint_angles_normalized"][0],
            )

            # Evaluate
            metric = validator.evaluate_trajectory(
                workpiece_id=wid,
                joint_trajectory=joint_trajectory,
                start_joint_state=validator._unnormalize_joint_state(
                    raw_obs["first_joint_angles_normalized"][0]
                ),
                goal_position_normalized=raw_obs["goal_position"][0],
            )

            collision_steps = float(metric["segment_collision_steps"])
            total_steps = float(metric["segment_steps"])
            has_collision = bool(metric["has_collision"])

            traj_entry = {
                "episode_idx": int(ep_idx),
                "workpiece_id": wid,
                "has_collision": has_collision,
                "collision_steps": collision_steps,
                "total_steps": total_steps,
                "collision_rate": collision_steps / total_steps if total_steps > 0 else 0.0,
                "min_sdf_distance_m": float(metric["min_sdf_distance_m"]),
                "goal_error_m": float(metric["goal_error_m"]),
                "goal_reached": bool(metric["goal_reached"]),
                "success": bool(metric["success"]),
            }
            per_traj_metrics.append(traj_entry)

            if has_collision:
                collision_count += 1
            total_segment_collision_steps += collision_steps
            total_segment_steps += total_steps
            if not np.isnan(metric["min_sdf_distance_m"]):
                sdf_distances.append(float(metric["min_sdf_distance_m"]))
            goal_errors.append(float(metric["goal_error_m"]))

            # Progress
            if (idx + 1) % max(1, len(val_episode_indices) // 10) == 0 or idx == 0:
                print(f"  [{idx + 1}/{len(val_episode_indices)}] "
                      f"collision_rate_so_far={collision_count / (idx + 1):.3f}")

    # ---- cleanup ----
    runner.close()

    # ---- summary ----
    total = float(len(per_traj_metrics))
    traj_collision_rate = collision_count / total if total > 0 else 0.0
    collision_free_rate = 1.0 - traj_collision_rate
    overall_segment_collision_rate = (
        total_segment_collision_steps / total_segment_steps
        if total_segment_steps > 0
        else 0.0
    )
    mean_min_sdf = float(np.mean(sdf_distances)) if sdf_distances else float("nan")
    sdf_valid_rate = len(sdf_distances) / total if total > 0 else 0.0
    goal_reached_count = sum(1 for t in per_traj_metrics if t["goal_reached"])
    mean_goal_error = float(np.mean(goal_errors)) if goal_errors else float("nan")

    # ---- save JSON ----
    output_json = output_dir / "per_trajectory_metrics.json"
    summary = {
        "config": {
            "checkpoint_path": str(checkpoint_path),
            "zarr_path": str(zarr_path),
            "stats_path": str(stats_path),
            "num_control_points": args.num_control_points,
            "val_ratio": args.val_ratio,
            "horizon": horizon,
            "n_obs_steps": n_obs_steps,
        },
        "summary": {
            "total_validation_episodes": int(total),
            "trajectories_with_collision": int(collision_count),
            "trajectory_collision_rate": float(traj_collision_rate),
            "collision_free_trajectory_rate": float(collision_free_rate),
            "overall_segment_collision_rate": float(overall_segment_collision_rate),
            "mean_min_sdf_distance_m": float(mean_min_sdf),
            "sdf_valid_rate": float(sdf_valid_rate),
            "goal_reached_count": int(goal_reached_count),
            "goal_reached_rate": goal_reached_count / total if total > 0 else 0.0,
            "mean_goal_error_m": float(mean_goal_error),
        },
        "per_trajectory": per_traj_metrics,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPer-trajectory metrics saved to: {output_json}")

    # ---- terminal summary ----
    print()
    print("=" * 56)
    print("  **Validation Summary**")
    print("=" * 56)
    print(f"  Total validation episodes:        {int(total)}")
    print(f"  Collision trajectories:           {int(collision_count)} "
          f"({traj_collision_rate * 100:.1f}%)")
    print(f"  Collision-free trajectories:      {int(total - collision_count)} "
          f"({collision_free_rate * 100:.1f}%)")
    print(f"  Overall segment collision rate:    "
          f"{int(total_segment_collision_steps)}/{int(total_segment_steps)} "
          f"({overall_segment_collision_rate * 100:.1f}%)")
    print(f"  Mean min SDF distance:            {mean_min_sdf:.4f} m")
    print(f"  SDF valid rate:                   {sdf_valid_rate * 100:.1f}%")
    print(f"  Goal reached:                     {goal_reached_count} "
          f"({goal_reached_count / total * 100:.1f}%)")
    print(f"  Mean goal error:                  {mean_goal_error:.4f} m")
    print("=" * 56)


if __name__ == "__main__":
    main()
