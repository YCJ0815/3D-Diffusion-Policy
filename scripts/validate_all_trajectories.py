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
import time
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
from train_cspace import TrainDP3CSpaceWorkspace  # noqa: E402
from diffusion_policy_3d.common.pybullet_validation import (  # noqa: E402
    PyBulletCollisionValidator,
    PyBulletValidationConfig,
    PyBulletValidationRunner,
    _episode_bounds,
    _select_lowest_candidate_score_index,
)
from diffusion_policy_3d.dataset.transition_dataset import (  # noqa: E402
    TransitionTrajectoryDataset,
)
from diffusion_policy_3d.dataset.transition_cspace_dataset import (  # noqa: E402
    TransitionTrajectoryCSpaceDataset,
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
        "--target-steps",
        type=int,
        default=None,
        help="Number of output trajectory steps for B-spline reconstruction. "
             "Auto-detected from checkpoint config when omitted.",
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
    parser.add_argument(
        "--cspace-feature-dir",
        type=str,
        default=None,
        help=(
            "Optional override for the C-space feature directory. "
            "If omitted, the script uses checkpoint cfg.task.dataset.cspace_feature_dir when needed."
        ),
    )
    parser.add_argument(
        "--cspace-feature-filename",
        type=str,
        default=None,
        help="Optional override for the C-space feature npy filename.",
    )
    parser.add_argument(
        "--cspace-workpiece-ids-filename",
        type=str,
        default=None,
        help="Optional override for the C-space workpiece IDs npy filename.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help=(
            "Override the number of candidate trajectories generated per episode. "
            "Default: use checkpoint cfg.training.pybullet_eval.num_candidates."
        ),
    )
    parser.add_argument(
        "--candidate-pool",
        choices=("on", "off"),
        default="on",
        help=(
            "Toggle the multi-path candidate pool during inference. "
            "`on` = sample multiple candidates then select the safest one; "
            "`off` = run a single-path inference without candidate-pool selection."
        ),
    )
    parser.add_argument(
        "--single-episode-index",
        type=int,
        default=None,
        help=(
            "Validate exactly one trajectory from the validation subset. "
            "This is the index inside the validation subset after applying --max-episodes, not the raw replay-buffer episode id."
        ),
    )
    parser.add_argument(
        "--measure-inference-time",
        action="store_true",
        help=(
            "Measure elapsed time from candidate inference start to final selected output. "
            "Useful with --single-episode-index."
        ),
    )
    parser.add_argument(
        "--regular-jobs-only",
        action="store_true",
        help=(
            "Restrict validation to regular jobs only. Episodes whose workpiece_id is "
            "below simple_workpiece_id_offset are kept; simple jobs are excluded."
        ),
    )
    parser.add_argument(
        "--enable-surface-cbf-qp-guidance",
        action="store_true",
        help=(
            "Use the late-stage surface-sample CBF-QP guided denoising path during "
            "validation inference instead of the standard predict_action/candidate-pool path."
        ),
    )
    parser.add_argument(
        "--guidance-steps",
        type=int,
        default=2,
        help="Number of final denoising steps that run surface-sample CBF-QP guidance.",
    )
    parser.add_argument(
        "--guidance-qp-candidates",
        type=int,
        default=4,
        help="How many near-collision candidates to project with QP per guidance step.",
    )
    parser.add_argument(
        "--guidance-active-constraints",
        type=int,
        default=16,
        help="Maximum number of active surface constraints kept in each guidance QP.",
    )
    parser.add_argument(
        "--guidance-check-steps",
        type=int,
        default=64,
        help="Dense B-spline evaluation steps used during per-step guidance collision checks.",
    )
    parser.add_argument(
        "--guidance-cert-steps",
        type=int,
        default=256,
        help="Dense B-spline evaluation steps used for the final certificate check.",
    )
    parser.add_argument(
        "--guidance-cert-swept-intermediate",
        type=int,
        default=3,
        help="Number of swept interpolation points inserted between adjacent certificate waypoints.",
    )
    parser.add_argument("--guidance-d-safe", type=float, default=0.03)
    parser.add_argument("--guidance-d-trigger", type=float, default=0.06)
    parser.add_argument("--guidance-d-cert", type=float, default=0.01)
    parser.add_argument("--guidance-eps-deep", type=float, default=0.03)
    parser.add_argument("--guidance-delta-max", type=float, default=0.05)
    parser.add_argument("--guidance-lambda-s", type=float, default=0.25)
    parser.add_argument("--guidance-rho", type=float, default=1.0e5)
    parser.add_argument(
        "--guidance-ddim-eta",
        type=float,
        default=0.0,
        help="Deterministic DDIM eta used during the guided denoising tail.",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_horizon(
    workspace: TrainDP3Workspace | TrainDP3CSpaceWorkspace,
    cli_horizon: Optional[int],
) -> int:
    if cli_horizon is not None:
        return cli_horizon
    from omegaconf import OmegaConf

    dataset_horizon = OmegaConf.select(
        workspace.cfg, "task.dataset.horizon", default=None
    )
    if dataset_horizon is not None:
        return int(dataset_horizon)
    return int(workspace.cfg.horizon)


def _build_val_dataset(
    zarr_path: str,
    horizon: int,
    val_ratio: float,
    workspace: TrainDP3Workspace | TrainDP3CSpaceWorkspace,
) -> TransitionTrajectoryDataset:
    """Instantiate the full dataset and return its validation copy."""
    from omegaconf import OmegaConf

    ds_cfg = OmegaConf.select(workspace.cfg, "task.dataset", default={}) or {}
    common_kwargs = dict(
        zarr_path=str(zarr_path),
        horizon=horizon,
        pad_before=int(ds_cfg.get("pad_before", 0)),
        pad_after=int(ds_cfg.get("pad_after", 0)),
        seed=int(ds_cfg.get("seed", 42)),
        val_ratio=val_ratio,
        max_train_episodes=ds_cfg.get("max_train_episodes"),
        point_cloud_key=str(ds_cfg.get("point_cloud_key", "point_cloud")),
        obs_keys=tuple(
            ds_cfg.get("obs_keys", TransitionTrajectoryDataset.DEFAULT_OBS_KEYS)
        ),
        split_by_workpiece=bool(ds_cfg.get("split_by_workpiece", True)),
        stratify_workpiece_split=bool(ds_cfg.get("stratify_workpiece_split", True)),
        simple_workpiece_id_offset=int(ds_cfg.get("simple_workpiece_id_offset", 1000)),
        workpiece_split_strategy=str(ds_cfg.get("workpiece_split_strategy", "tail")),
    )
    dataset_target = str(ds_cfg.get("_target_", ""))
    if dataset_target.endswith("TransitionTrajectoryCSpaceDataset"):
        dataset = TransitionTrajectoryCSpaceDataset(
            cspace_feature_dir=str(ds_cfg.get("cspace_feature_dir")),
            cspace_feature_filename=str(
                ds_cfg.get(
                    "cspace_feature_filename", "workpiece_key_config_features.npy"
                )
            ),
            cspace_workpiece_ids_filename=str(
                ds_cfg.get("cspace_workpiece_ids_filename", "workpiece_ids.npy")
            ),
            **common_kwargs,
        )
    else:
        dataset = TransitionTrajectoryDataset(**common_kwargs)
    return dataset.get_validation_dataset()


def _build_pybullet_config(
    workspace: TrainDP3Workspace | TrainDP3CSpaceWorkspace,
    stats_path: str,
    num_control_points: int,
    collision_log_path: Optional[str] = None,
    jobs_root: Optional[str] = None,
    simple_jobs_root: Optional[str] = None,
    target_steps: Optional[int] = None,
) -> PyBulletValidationConfig:
    """Build PyBulletValidationConfig, seeding from the checkpoint and overriding with CLI."""
    from omegaconf import OmegaConf

    pyb_cfg_raw = OmegaConf.select(workspace.cfg, "training.pybullet_eval", default={}) or {}
    return PyBulletValidationConfig(
        enabled=True,
        stats_path=stats_path,
        stats_mode=str(pyb_cfg_raw.get("stats_mode", "auto")),
        include_regular_jobs=bool(pyb_cfg_raw.get("include_regular_jobs", True)),
        include_simple_jobs=bool(pyb_cfg_raw.get("include_simple_jobs", True)),
        jobs_root=str(
            jobs_root if jobs_root is not None else pyb_cfg_raw.get("jobs_root", "data/raw_data/jobs")
        ),
        simple_jobs_root=(
            simple_jobs_root
            if simple_jobs_root is not None
            else pyb_cfg_raw.get("simple_jobs_root", "data/raw_data/simple_jobs")
        ),
        simple_workpiece_id_offset=int(
            pyb_cfg_raw.get("simple_workpiece_id_offset", 1000)
        ),
        job_name_template=str(
            pyb_cfg_raw.get("job_name_template", "job_{workpiece_id:03d}")
        ),
        workpiece_filename=str(pyb_cfg_raw.get("workpiece_filename", "workpiece.stl")),
        urdf_path=pyb_cfg_raw.get("urdf_path"),
        urdf_package_roots=tuple(
            pyb_cfg_raw.get("urdf_package_roots", ["config/robot-model"])
        ),
        tcp_link_name=str(pyb_cfg_raw.get("tcp_link_name", "tool0")),
        stl_x_offset_m=float(pyb_cfg_raw.get("stl_x_offset_m", 0.5)),
        collision_distance_threshold=float(
            pyb_cfg_raw.get("collision_distance_threshold", 0.0)
        ),
        interpolate_for_collision=bool(
            pyb_cfg_raw.get("interpolate_for_collision", False)
        ),
        max_joint_step_rad=float(pyb_cfg_raw.get("max_joint_step_rad", 0.01)),
        min_interpolated_steps_per_segment=int(
            pyb_cfg_raw.get("min_interpolated_steps_per_segment", 1)
        ),
        goal_position_norm_m=float(pyb_cfg_raw.get("goal_position_norm_m", 0.1)),
        goal_tolerance_m=float(pyb_cfg_raw.get("goal_tolerance_m", 0.01)),
        num_control_points=num_control_points,
        spline_degree=int(pyb_cfg_raw.get("spline_degree", 5)),
        target_steps=(
            target_steps
            if target_steps is not None
            else int(pyb_cfg_raw.get("target_steps", 64))
        ),
        max_episodes=None,
        random_sample_episodes=bool(pyb_cfg_raw.get("random_sample_episodes", False)),
        random_seed=int(pyb_cfg_raw.get("random_seed", 42)),
        diffusion_sampling_seed=(
            None
            if pyb_cfg_raw.get("diffusion_sampling_seed", None) is None
            else int(pyb_cfg_raw.get("diffusion_sampling_seed"))
        ),
        inference_num_steps=(
            None
            if pyb_cfg_raw.get("inference_num_steps", None) is None
            else int(pyb_cfg_raw.get("inference_num_steps"))
        ),
        num_candidates=int(pyb_cfg_raw.get("num_candidates", 16)),
        candidate_scheduler_eta=(
            None
            if pyb_cfg_raw.get("candidate_scheduler_eta", 1.0) is None
            else float(pyb_cfg_raw.get("candidate_scheduler_eta", 1.0))
        ),
        candidate_action_noise_std=float(
            pyb_cfg_raw.get("candidate_action_noise_std", 0.0)
        ),
        candidate_action_noise_clip=(
            None
            if pyb_cfg_raw.get("candidate_action_noise_clip", None) is None
            else float(pyb_cfg_raw.get("candidate_action_noise_clip"))
        ),
        candidate_selection=str(pyb_cfg_raw.get("candidate_selection", "weighted_sdf")),
        selection_topk=int(pyb_cfg_raw.get("selection_topk", 128)),
        selection_d_safe=float(pyb_cfg_raw.get("selection_d_safe", 0.005)),
        selection_d_pen=float(pyb_cfg_raw.get("selection_d_pen", 0.005)),
        selection_margin_weight=float(pyb_cfg_raw.get("selection_margin_weight", 1.0)),
        selection_penetration_weight=float(
            pyb_cfg_raw.get("selection_penetration_weight", 2.0)
        ),
        selection_smooth_weight=float(
            pyb_cfg_raw.get("selection_smooth_weight", 0.01)
        ),
        selection_length_weight=float(
            pyb_cfg_raw.get("selection_length_weight", 0.005)
        ),
        sdf_filename=str(pyb_cfg_raw.get("sdf_filename", "workpiece_sdf.npz")),
        sdf_required=bool(pyb_cfg_raw.get("sdf_required", True)),
        robot_surface_points_per_link=(
            pyb_cfg_raw.get("robot_surface_points_per_link", {"pen_link": 80, "wrist3": 16})
        ),
        sdf_out_of_bounds_value_m=pyb_cfg_raw.get("sdf_out_of_bounds_value_m"),
        log_legacy_pybullet_metrics=bool(
            pyb_cfg_raw.get("log_legacy_pybullet_metrics", True)
        ),
        collision_log_path=collision_log_path,
        progress_mininterval_sec=float(pyb_cfg_raw.get("progress_mininterval_sec", 1.0)),
        num_workers=int(pyb_cfg_raw.get("num_workers", 1)),
        inference_batch_size=int(pyb_cfg_raw.get("inference_batch_size", 32)),
        worker_start_method=str(pyb_cfg_raw.get("worker_start_method", "spawn")),
        worker_chunksize=int(pyb_cfg_raw.get("worker_chunksize", 1)),
    )


def _append_collision_events(log_path: pathlib.Path, collision_events: list[dict]) -> None:
    if not collision_events:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for event in collision_events:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _build_obs_batch(
    replay_buffer,
    episode_idx: int,
    obs_keys: tuple[str, ...],
    n_obs_steps: int,
    device: torch.device,
    workpiece_id: Optional[int] = None,
    dataset=None,
    policy=None,
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
    cspace_feature_key = getattr(policy, "cspace_feature_key", None)
    if cspace_feature_key is not None:
        if (
            dataset is None
            or workpiece_id is None
            or not hasattr(dataset, "get_cspace_feature_by_workpiece_id")
        ):
            raise KeyError(
                "This checkpoint expects C-space features, but the validation dataset "
                "cannot provide them. Pass a C-space dataset or override --cspace-feature-dir."
            )
        cspace_feature = dataset.get_cspace_feature_by_workpiece_id(int(workpiece_id))
        raw_obs[cspace_feature_key] = cspace_feature.copy()
        obs_batch[cspace_feature_key] = torch.from_numpy(cspace_feature[None]).to(device)
    return obs_batch, raw_obs


def _effective_num_candidates(
    pyb_cfg: PyBulletValidationConfig,
    cli_num_candidates: Optional[int],
    candidate_pool_enabled: bool,
) -> int:
    if not candidate_pool_enabled:
        return 1
    if cli_num_candidates is not None:
        if int(cli_num_candidates) <= 0:
            raise ValueError(
                f"--num-candidates must be >= 1, got {cli_num_candidates}"
            )
        return int(cli_num_candidates)
    return int(pyb_cfg.num_candidates)


def _candidate_pool_enabled(candidate_pool_arg: str) -> bool:
    return str(candidate_pool_arg).lower() == "on"


def _filter_episode_indices_by_job_type(
    episode_indices: np.ndarray,
    workpiece_ids: np.ndarray,
    *,
    simple_workpiece_id_offset: int,
    regular_jobs_only: bool,
) -> np.ndarray:
    if not regular_jobs_only:
        return np.asarray(episode_indices, dtype=np.int64)

    filtered_episode_indices = [
        int(episode_idx)
        for episode_idx in np.asarray(episode_indices, dtype=np.int64).tolist()
        if int(workpiece_ids[int(episode_idx)]) < int(simple_workpiece_id_offset)
    ]
    return np.asarray(filtered_episode_indices, dtype=np.int64)


def _predict_surface_cbf_qp_guided(
    *,
    policy,
    validator,
    pyb_cfg: PyBulletValidationConfig,
    obs_dict: dict[str, torch.Tensor],
    raw_obs: dict[str, np.ndarray],
    workpiece_id: int,
    device: torch.device,
    args,
    batch_start: int = 0,
    num_candidates_override: Optional[int] = None,
    measure_inference_time: bool = False,
) -> dict[str, object]:
    from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
        PyBulletSurfaceEnvironmentAdapter,
        SurfaceCBFQPGuidanceConfig,
        SurfaceCBFQPGuidanceRunner,
    )

    if validator.stats_mode != "bspline":
        raise ValueError(
            "Surface CBF-QP guidance currently requires bspline stats mode, "
            f"got {validator.stats_mode!r}."
        )

    num_candidates = (
        int(num_candidates_override)
        if num_candidates_override is not None
        else int(pyb_cfg.num_candidates)
    )
    if num_candidates <= 0:
        raise ValueError(f"num_candidates must be positive for guidance, got {num_candidates}")

    base_seed = int((pyb_cfg.diffusion_sampling_seed or 0) + int(batch_start))
    generator = torch.Generator(device=device)
    generator.manual_seed(base_seed)

    environment = PyBulletSurfaceEnvironmentAdapter(
        validator=validator,
        workpiece_id=int(workpiece_id),
        joint_lower_limits=np.asarray(validator.joint_lower_limits, dtype=np.float32),
        joint_upper_limits=np.asarray(validator.joint_upper_limits, dtype=np.float32),
    )
    guidance_config = SurfaceCBFQPGuidanceConfig(
        enabled=True,
        num_candidates=int(num_candidates),
        guidance_steps=int(args.guidance_steps),
        qp_candidates=int(args.guidance_qp_candidates),
        active_constraints=int(args.guidance_active_constraints),
        check_steps=int(args.guidance_check_steps),
        cert_steps=int(args.guidance_cert_steps),
        cert_swept_intermediate=int(args.guidance_cert_swept_intermediate),
        d_safe=float(args.guidance_d_safe),
        d_trigger=float(args.guidance_d_trigger),
        d_cert=float(args.guidance_d_cert),
        eps_deep=float(args.guidance_eps_deep),
        delta_max=float(args.guidance_delta_max),
        lambda_s=float(args.guidance_lambda_s),
        rho=float(args.guidance_rho),
        ddim_eta=float(args.guidance_ddim_eta),
    )
    guidance_runner = SurfaceCBFQPGuidanceRunner(
        config=guidance_config,
        environment=environment,
    )

    start_joint_normalized = raw_obs["first_joint_angles_normalized"][0].astype(np.float32)
    end_joint_normalized = raw_obs["last_joint_angles_normalized"][0].astype(np.float32)
    scheduler_step_kwargs = {"eta": float(args.guidance_ddim_eta)}

    start_time = time.perf_counter() if measure_inference_time else None
    result = policy.sample_with_surface_cbf_qp_guidance(
        obs_dict,
        q_start_normalized=start_joint_normalized,
        q_goal_normalized=end_joint_normalized,
        delta_w_mean=np.asarray(validator.stats_mean, dtype=np.float32),
        delta_w_std=np.asarray(validator.stats_std, dtype=np.float32),
        num_control_points=int(pyb_cfg.num_control_points),
        spline_degree=int(pyb_cfg.spline_degree),
        guidance_runner=guidance_runner,
        generator=generator,
        num_inference_steps=pyb_cfg.inference_num_steps,
        scheduler_step_kwargs=scheduler_step_kwargs,
    )
    selected_action_horizon = (
        result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
    )
    selected_result = validator.reconstruct_candidate(
        pred_action_horizon=selected_action_horizon,
        start_joint_normalized=start_joint_normalized,
        end_joint_normalized=end_joint_normalized,
    )
    selected_score_details = validator.score_candidate(
        workpiece_id=workpiece_id,
        normalized_control_points=selected_result["normalized_control_points"],
        joint_trajectory=selected_result["joint_trajectory"],
    )
    end_time = time.perf_counter() if measure_inference_time else None
    inference_elapsed_sec = (
        float(end_time - start_time)
        if measure_inference_time and start_time is not None and end_time is not None
        else None
    )
    guidance_log = result.get("guidance_log", {})
    return {
        "candidate_pool_enabled": False,
        "surface_cbf_qp_guidance_enabled": True,
        "selected_candidate_idx": int(guidance_log.get("best_candidate_index", 0) or 0),
        "selected_candidate_seed": int(base_seed),
        "selected_action_horizon": np.asarray(selected_action_horizon, dtype=np.float32),
        "selected_joint_trajectory": np.asarray(selected_result["joint_trajectory"], dtype=np.float32),
        "selected_score_details": selected_score_details,
        "candidate_score_details": [selected_score_details],
        "candidate_seeds": [int(base_seed)],
        "num_candidates": int(num_candidates),
        "inference_elapsed_sec": inference_elapsed_sec,
        "guidance_log": guidance_log,
        "guidance_candidate_count": len(result.get("guidance_candidates", [])),
    }


def _predict_select_candidate(
    *,
    policy,
    validator,
    pyb_cfg: PyBulletValidationConfig,
    obs_dict: dict[str, torch.Tensor],
    raw_obs: dict[str, np.ndarray],
    workpiece_id: int,
    device: torch.device,
    batch_start: int = 0,
    num_candidates_override: Optional[int] = None,
    candidate_pool_enabled: bool = True,
    measure_inference_time: bool = False,
) -> dict[str, object]:
    num_candidates = _effective_num_candidates(
        pyb_cfg,
        num_candidates_override,
        candidate_pool_enabled=candidate_pool_enabled,
    )
    if candidate_pool_enabled and pyb_cfg.diffusion_sampling_seed is None:
        raise ValueError(
            "PyBullet validation requires diffusion_sampling_seed to be set when using candidate selection."
        )

    scheduler_step_kwargs = {}
    if pyb_cfg.candidate_scheduler_eta is not None:
        scheduler_step_kwargs["eta"] = float(pyb_cfg.candidate_scheduler_eta)

    start_joint_normalized = raw_obs["first_joint_angles_normalized"][0].astype(np.float32)
    end_joint_normalized = raw_obs["last_joint_angles_normalized"][0].astype(np.float32)

    start_time = time.perf_counter() if measure_inference_time else None

    if not candidate_pool_enabled:
        single_candidate_seed = int((pyb_cfg.diffusion_sampling_seed or 0) + int(batch_start))
        generator = torch.Generator(device=device)
        generator.manual_seed(single_candidate_seed)
        result = policy.predict_action(
            obs_dict,
            generator=generator,
            num_inference_steps=pyb_cfg.inference_num_steps,
            scheduler_step_kwargs=scheduler_step_kwargs,
        )
        selected_action_horizon = (
            result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
        )
        selected_result = validator.reconstruct_candidate(
            pred_action_horizon=selected_action_horizon,
            start_joint_normalized=start_joint_normalized,
            end_joint_normalized=end_joint_normalized,
        )
        selected_score_details = validator.score_candidate(
            workpiece_id=workpiece_id,
            normalized_control_points=selected_result["normalized_control_points"],
            joint_trajectory=selected_result["joint_trajectory"],
        )
        end_time = time.perf_counter() if measure_inference_time else None
        inference_elapsed_sec = (
            float(end_time - start_time)
            if measure_inference_time and start_time is not None and end_time is not None
            else None
        )
        return {
            "candidate_pool_enabled": False,
            "selected_candidate_idx": 0,
            "selected_candidate_seed": int(single_candidate_seed),
            "selected_action_horizon": np.asarray(
                selected_action_horizon, dtype=np.float32
            ),
            "selected_joint_trajectory": np.asarray(
                selected_result["joint_trajectory"], dtype=np.float32
            ),
            "selected_score_details": selected_score_details,
            "candidate_score_details": [selected_score_details],
            "candidate_seeds": [int(single_candidate_seed)],
            "num_candidates": 1,
            "inference_elapsed_sec": inference_elapsed_sec,
        }

    candidate_results: list[dict[str, object]] = []
    candidate_score_details: list[dict[str, object]] = []
    candidate_seeds: list[int] = []
    candidate_action_horizons: list[np.ndarray] = []

    for candidate_idx in range(num_candidates):
        candidate_seed = (
            int(pyb_cfg.diffusion_sampling_seed)
            + candidate_idx * 1_000_003
            + int(batch_start)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(candidate_seed)
        result = policy.predict_action(
            obs_dict,
            generator=generator,
            num_inference_steps=pyb_cfg.inference_num_steps,
            scheduler_step_kwargs=scheduler_step_kwargs,
        )
        candidate_action = (
            result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
        )
        if candidate_idx > 0 and float(pyb_cfg.candidate_action_noise_std) > 0.0:
            rng = np.random.default_rng(candidate_seed)
            candidate_noise = rng.normal(
                loc=0.0,
                scale=float(pyb_cfg.candidate_action_noise_std),
                size=candidate_action.shape,
            ).astype(np.float32)
            if pyb_cfg.candidate_action_noise_clip is not None:
                candidate_noise = np.clip(
                    candidate_noise,
                    -float(pyb_cfg.candidate_action_noise_clip),
                    float(pyb_cfg.candidate_action_noise_clip),
                ).astype(np.float32)
            candidate_action = (candidate_action + candidate_noise).astype(np.float32)
        candidate_action_horizons.append(candidate_action)
        candidate_seeds.append(int(candidate_seed))

        candidate_result = validator.reconstruct_candidate(
            pred_action_horizon=candidate_action,
            start_joint_normalized=start_joint_normalized,
            end_joint_normalized=end_joint_normalized,
        )
        candidate_results.append(candidate_result)
        candidate_score_details.append(
            validator.score_candidate(
                workpiece_id=workpiece_id,
                normalized_control_points=candidate_result["normalized_control_points"],
                joint_trajectory=candidate_result["joint_trajectory"],
            )
        )

    candidate_score_keys_array = np.asarray(
        [
            [
                score_details["has_pen"],
                score_details["pen_step_count"],
                score_details["pen_point_count"],
                score_details["neg_min_sdf"],
                score_details["neg_worstk_mean"],
                score_details["margin_violation"],
            ]
            for score_details in candidate_score_details
        ],
        dtype=np.float32,
    )
    selected_candidate_idx = _select_lowest_candidate_score_index(
        candidate_score_keys_array
    )
    selected_result = candidate_results[selected_candidate_idx]
    selected_score_details = candidate_score_details[selected_candidate_idx]

    end_time = time.perf_counter() if measure_inference_time else None
    inference_elapsed_sec = (
        float(end_time - start_time)
        if measure_inference_time and start_time is not None and end_time is not None
        else None
    )

    return {
        "candidate_pool_enabled": True,
        "selected_candidate_idx": int(selected_candidate_idx),
        "selected_candidate_seed": int(candidate_seeds[selected_candidate_idx]),
        "selected_action_horizon": np.asarray(
            candidate_action_horizons[selected_candidate_idx], dtype=np.float32
        ),
        "selected_joint_trajectory": np.asarray(
            selected_result["joint_trajectory"], dtype=np.float32
        ),
        "selected_score_details": selected_score_details,
        "candidate_score_details": candidate_score_details,
        "candidate_seeds": candidate_seeds,
        "num_candidates": int(num_candidates),
        "inference_elapsed_sec": inference_elapsed_sec,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "analysis_outputs" / "validation" / ts
    else:
        output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    zarr_path = _PROJECT_ROOT / args.zarr_path
    stats_path = _PROJECT_ROOT / args.stats_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device(args.device)
    workspace = TrainDP3CSpaceWorkspace.create_from_checkpoint(
        str(checkpoint_path),
    )
    from omegaconf import OmegaConf

    if args.cspace_feature_dir is not None:
        OmegaConf.update(
            workspace.cfg,
            "task.dataset.cspace_feature_dir",
            args.cspace_feature_dir,
            force_add=True,
        )
    if args.cspace_feature_filename is not None:
        OmegaConf.update(
            workspace.cfg,
            "task.dataset.cspace_feature_filename",
            args.cspace_feature_filename,
            force_add=True,
        )
    if args.cspace_workpiece_ids_filename is not None:
        OmegaConf.update(
            workspace.cfg,
            "task.dataset.cspace_workpiece_ids_filename",
            args.cspace_workpiece_ids_filename,
            force_add=True,
        )
    policy = workspace.model
    policy.to(device)
    policy.eval()

    horizon = _resolve_horizon(workspace, args.horizon)
    n_obs_steps = int(workspace.cfg.n_obs_steps)
    print(f"Checkpoint config: horizon={horizon}, n_obs_steps={n_obs_steps}")

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

    replay_buffer = val_dataset.replay_buffer
    obs_keys = val_dataset.obs_keys

    if "workpiece_ids" not in replay_buffer.meta:
        raise KeyError(
            "PyBullet validation requires `meta/workpiece_ids` in the zarr dataset. "
            "Rebuild the dataset with workpiece metadata."
        )
    workpiece_ids = np.asarray(replay_buffer.meta["workpiece_ids"][:], dtype=np.int64)

    print("Initialising PyBullet validator …")
    resolved_target_steps = args.target_steps
    if resolved_target_steps is None:
        pyb_cfg_raw = OmegaConf.select(workspace.cfg, "training.pybullet_eval", default={}) or {}
        resolved_target_steps = int(pyb_cfg_raw.get("target_steps", 64))
    pyb_cfg = _build_pybullet_config(
        workspace=workspace,
        stats_path=str(stats_path),
        num_control_points=args.num_control_points,
        collision_log_path=str(output_dir / "pybullet_collision_events.jsonl"),
        jobs_root=args.jobs_root,
        simple_jobs_root=args.simple_jobs_root,
        target_steps=resolved_target_steps,
    )
    candidate_pool_enabled = _candidate_pool_enabled(args.candidate_pool)
    guidance_enabled = bool(args.enable_surface_cbf_qp_guidance)
    effective_num_candidates = (
        int(args.num_candidates)
        if args.num_candidates is not None
        else (
            int(pyb_cfg.num_candidates)
            if guidance_enabled
            else _effective_num_candidates(
                pyb_cfg,
                args.num_candidates,
                candidate_pool_enabled=candidate_pool_enabled,
            )
        )
    )
    if effective_num_candidates <= 0:
        raise ValueError(f"effective_num_candidates must be positive, got {effective_num_candidates}")
    print(f"  num_control_points={pyb_cfg.num_control_points}")
    print(f"  stats_mode={pyb_cfg.stats_mode}")
    print(f"  jobs_root={pyb_cfg.jobs_root}")
    print(f"  simple_jobs_root={pyb_cfg.simple_jobs_root}")
    print(f"  effective_num_candidates={effective_num_candidates}")
    print(f"  candidate_selection={pyb_cfg.candidate_selection}")
    print(f"  inference_num_steps={pyb_cfg.inference_num_steps}")
    print(f"  candidate_action_noise_std={pyb_cfg.candidate_action_noise_std}")
    print(f"  candidate_action_noise_clip={pyb_cfg.candidate_action_noise_clip}")
    print(f"  candidate_pool={args.candidate_pool}")
    print(f"  surface_cbf_qp_guidance={guidance_enabled}")
    if guidance_enabled:
        print(f"  guidance_steps={args.guidance_steps}")
        print(f"  guidance_cert_steps={args.guidance_cert_steps}")
        print(f"  guidance_ddim_eta={args.guidance_ddim_eta}")
    if candidate_pool_enabled:
        runner = PyBulletValidationRunner(pyb_cfg)
        validator = runner.validator
    else:
        runner = None
        validator = PyBulletCollisionValidator(pyb_cfg)

    val_episode_indices = _filter_episode_indices_by_job_type(
        val_episode_indices,
        workpiece_ids,
        simple_workpiece_id_offset=int(pyb_cfg.simple_workpiece_id_offset),
        regular_jobs_only=bool(args.regular_jobs_only),
    )
    if args.regular_jobs_only:
        print(
            "  regular_jobs_only=true -> filtered validation episodes: "
            f"{len(val_episode_indices)}"
        )

    if args.max_episodes is not None and args.max_episodes < len(val_episode_indices):
        val_episode_indices = val_episode_indices[: args.max_episodes]
        print(f"  (limited to {args.max_episodes} by --max-episodes)")

    if args.single_episode_index is not None:
        if args.single_episode_index < 0 or args.single_episode_index >= len(
            val_episode_indices
        ):
            raise IndexError(
                f"--single-episode-index {args.single_episode_index} is out of range "
                f"for {len(val_episode_indices)} validation episodes."
            )
        val_episode_indices = np.asarray(
            [int(val_episode_indices[int(args.single_episode_index)])],
            dtype=np.int64,
        )
        print(
            "  (single-episode mode: validation subset index "
            f"{args.single_episode_index})"
        )

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

            obs_dict, raw_obs = _build_obs_batch(
                replay_buffer=replay_buffer,
                episode_idx=ep_idx,
                obs_keys=obs_keys,
                n_obs_steps=n_obs_steps,
                device=device,
                workpiece_id=wid,
                dataset=val_dataset,
                policy=policy,
            )

            if args.enable_surface_cbf_qp_guidance:
                selection = _predict_surface_cbf_qp_guided(
                    policy=policy,
                    validator=validator,
                    pyb_cfg=pyb_cfg,
                    obs_dict=obs_dict,
                    raw_obs=raw_obs,
                    workpiece_id=wid,
                    device=device,
                    batch_start=idx,
                    num_candidates_override=args.num_candidates,
                    measure_inference_time=bool(args.measure_inference_time),
                    args=args,
                )
            else:
                selection = _predict_select_candidate(
                    policy=policy,
                    validator=validator,
                    pyb_cfg=pyb_cfg,
                    obs_dict=obs_dict,
                    raw_obs=raw_obs,
                    workpiece_id=wid,
                    device=device,
                    batch_start=idx,
                    num_candidates_override=args.num_candidates,
                    candidate_pool_enabled=candidate_pool_enabled,
                    measure_inference_time=bool(args.measure_inference_time),
                )
            joint_trajectory = np.asarray(
                selection["selected_joint_trajectory"], dtype=np.float32
            )

            if idx == 0:
                from diffusion_policy_3d.common.bspline import (
                    _resolve_free_control_point_slice,
                )

                free_slice = _resolve_free_control_point_slice(pyb_cfg.num_control_points)
                expected_free = free_slice.stop - free_slice.start
                selected_action_horizon = np.asarray(
                    selection["selected_action_horizon"], dtype=np.float32
                )
                print(f"  [debug] num_control_points={pyb_cfg.num_control_points}")
                print(f"  [debug] expected free CPs: {expected_free}")
                print(
                    f"  [debug] pred_action_horizon shape: {selected_action_horizon.shape}"
                )

            metric = validator.evaluate_trajectory(
                workpiece_id=wid,
                joint_trajectory=joint_trajectory,
                start_joint_state=validator._unnormalize_joint_state(
                    raw_obs["first_joint_angles_normalized"][0]
                ),
                goal_position_normalized=raw_obs["goal_position"][0],
                episode_idx=int(ep_idx),
            )
            _append_collision_events(
                output_dir / "pybullet_collision_events.jsonl",
                metric.get("collision_events", []),
            )

            collision_steps = float(metric["segment_collision_steps"])
            total_steps = float(metric["segment_steps"])
            has_collision = bool(metric["has_collision"])

            traj_entry = {
                "episode_idx": int(ep_idx),
                "validation_subset_index": int(idx),
                "workpiece_id": wid,
                "has_collision": has_collision,
                "collision_steps": collision_steps,
                "total_steps": total_steps,
                "collision_rate": collision_steps / total_steps if total_steps > 0 else 0.0,
                "min_sdf_distance_m": float(metric["min_sdf_distance_m"]),
                "goal_error_m": float(metric["goal_error_m"]),
                "goal_reached": bool(metric["goal_reached"]),
                "success": bool(metric["success"]),
                "selected_candidate_idx": int(selection["selected_candidate_idx"]),
                "selected_candidate_seed": int(selection["selected_candidate_seed"]),
                "candidate_pool_enabled": bool(selection["candidate_pool_enabled"]),
                "surface_cbf_qp_guidance_enabled": bool(selection.get("surface_cbf_qp_guidance_enabled", False)),
                "num_candidates": int(selection["num_candidates"]),
                "selected_score": selection["selected_score_details"],
                "inference_elapsed_sec": selection["inference_elapsed_sec"],
                "guidance_log": selection.get("guidance_log"),
                "guidance_candidate_count": selection.get("guidance_candidate_count"),
            }
            per_traj_metrics.append(traj_entry)

            if has_collision:
                collision_count += 1
            total_segment_collision_steps += collision_steps
            total_segment_steps += total_steps
            if not np.isnan(metric["min_sdf_distance_m"]):
                sdf_distances.append(float(metric["min_sdf_distance_m"]))
            goal_errors.append(float(metric["goal_error_m"]))

            if selection["inference_elapsed_sec"] is not None:
                print(
                    "  Inference-to-selected-output time: "
                    f"{float(selection['inference_elapsed_sec']):.6f} s"
                )

            if (idx + 1) % max(1, len(val_episode_indices) // 10) == 0 or idx == 0:
                traj_collision_rate_so_far = collision_count / (idx + 1)
                step_collision_rate_so_far = (
                    total_segment_collision_steps / total_segment_steps
                    if total_segment_steps > 0
                    else 0.0
                )
                print(
                    f"  [{idx + 1}/{len(val_episode_indices)}] "
                    f"traj_collision_rate_so_far={traj_collision_rate_so_far:.3f} "
                    f"step_collision_rate_so_far={step_collision_rate_so_far:.3f}"
                )

    if runner is not None:
        runner.close()
    else:
        validator.close()

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
            "candidate_pool": str(args.candidate_pool),
            "surface_cbf_qp_guidance": bool(args.enable_surface_cbf_qp_guidance),
            "effective_num_candidates": int(effective_num_candidates),
            "candidate_selection": str(pyb_cfg.candidate_selection),
            "inference_num_steps": pyb_cfg.inference_num_steps,
            "measure_inference_time": bool(args.measure_inference_time),
            "regular_jobs_only": bool(args.regular_jobs_only),
            "simple_workpiece_id_offset": int(pyb_cfg.simple_workpiece_id_offset),
            "single_episode_mode": args.single_episode_index is not None,
            "single_episode_validation_offset": args.single_episode_index,
            "guidance_steps": int(args.guidance_steps),
            "guidance_qp_candidates": int(args.guidance_qp_candidates),
            "guidance_active_constraints": int(args.guidance_active_constraints),
            "guidance_check_steps": int(args.guidance_check_steps),
            "guidance_cert_steps": int(args.guidance_cert_steps),
            "guidance_cert_swept_intermediate": int(args.guidance_cert_swept_intermediate),
            "guidance_d_safe": float(args.guidance_d_safe),
            "guidance_d_trigger": float(args.guidance_d_trigger),
            "guidance_d_cert": float(args.guidance_d_cert),
            "guidance_eps_deep": float(args.guidance_eps_deep),
            "guidance_delta_max": float(args.guidance_delta_max),
            "guidance_lambda_s": float(args.guidance_lambda_s),
            "guidance_rho": float(args.guidance_rho),
            "guidance_ddim_eta": float(args.guidance_ddim_eta),
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

    print()
    print("=" * 56)
    print("  **Validation Summary**")
    print("=" * 56)
    print(f"  Total validation episodes:        {int(total)}")
    print(
        f"  Collision trajectories:           {int(collision_count)} "
        f"({traj_collision_rate * 100:.1f}%)"
    )
    print(
        f"  Collision-free trajectories:      {int(total - collision_count)} "
        f"({collision_free_rate * 100:.1f}%)"
    )
    print(
        f"  Overall segment collision rate:    "
        f"{int(total_segment_collision_steps)}/{int(total_segment_steps)} "
        f"({overall_segment_collision_rate * 100:.1f}%)"
    )
    print(f"  Mean min SDF distance:            {mean_min_sdf:.4f} m")
    print(f"  SDF valid rate:                   {sdf_valid_rate * 100:.1f}%")
    print(
        f"  Goal reached:                     {goal_reached_count} "
        f"({goal_reached_count / total * 100:.1f}%)"
    )
    print(f"  Mean goal error:                  {mean_goal_error:.4f} m")
    print("=" * 56)


if __name__ == "__main__":
    main()