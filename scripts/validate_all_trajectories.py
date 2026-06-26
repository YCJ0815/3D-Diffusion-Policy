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
import shutil
import sys
import time
from typing import Optional
import xml.etree.ElementTree as ET

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
from diffusion_policy_3d.common.input_data import load_bspline_planning_input_data  # noqa: E402
from guidance_config import (  # noqa: E402
    add_surface_cbf_qp_guidance_parser_args,
    apply_surface_cbf_qp_guidance_config,
)


def _qp_guided_surface_points_per_link(args) -> dict[str, int]:
    return {
        "pen_link": int(args.guidance_pen_link_points),
        "wrist_3_link": int(args.guidance_wrist3_points),
    }


def _format_device(value) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.device)
    return str(value)


def _print_runtime_device_report(
    *,
    prefix: str,
    requested_device,
    policy,
    obs_dict: dict[str, torch.Tensor] | None = None,
    planner_mode: str | None = None,
    late_stage_geometry_backend: str | None = None,
) -> None:
    param_device = "no-parameters"
    try:
        first_param = next(policy.parameters())
        param_device = str(first_param.device)
    except StopIteration:
        pass
    obs_devices = sorted(
        {
            str(value.device)
            for value in (obs_dict or {}).values()
            if isinstance(value, torch.Tensor)
        }
    )
    print(f"[{prefix}] requested_device={requested_device}")
    print(
        f"[{prefix}] torch.cuda.is_available={torch.cuda.is_available()} "
        f"cuda.device_count={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        print(
            f"[{prefix}] torch.cuda.current_device={torch.cuda.current_device()} "
            f"torch.cuda.device_name={torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    print(
        f"[{prefix}] policy_parameter_device={param_device} "
        f"policy_on_requested_device={param_device == str(requested_device)}"
    )
    if obs_dict is not None:
        print(
            f"[{prefix}] obs_tensor_devices={obs_devices} "
            f"obs_on_requested_device={all(device == str(requested_device) for device in obs_devices)}"
        )
    if planner_mode is not None:
        print(f"[{prefix}] planner_mode={planner_mode}")
    if late_stage_geometry_backend is not None:
        print(f"[{prefix}] late_stage_qp_geometry_backend={late_stage_geometry_backend}")
        print(f"[{prefix}] final_qp_solver_backend=cpu")


def _format_qp_skip_reason(reason: str | None) -> str:
    reason_map = {
        None: "unknown",
        "surface_cbf_qp_guidance_disabled": "surface CBF-QP guidance disabled",
        "no_sdf_surface_samples": "no robot surface SDF samples were collected",
        "all_surface_samples_outside_sdf": "all robot surface samples are outside the SDF grid",
        "no_worst_timesteps": "no worst trajectory timesteps found",
        "no_risk_segments": "no risk segments found below the trigger threshold",
        "no_topk_constraints": "no valid risk-window CBF constraints built",
        "deep_penetration_unrepairable": "candidate is too deeply in collision for local SCP-QP repair",
        "safe_candidate_no_repair": "candidate clearance is already above the repair trigger",
        "solver_failure": "SCP-QP solver failed before certificate",
        "local_waypoint_qp_not_applicable": "local waypoint QP was not applicable for this certificate failure",
        "no_local_collision_windows": "no local collision window was found from certificate failures",
        "no_local_waypoint_constraints": "no valid local waypoint constraints were built",
        "local_waypoint_solver_failure": "local waypoint QP solver failed",
        "local_waypoint_certificate_failure": "local waypoint QP did not pass the re-certificate",
        "non_finite_h_min_before": "pre-guidance minimum margin is non-finite",
        "unknown_skip_condition": "unknown guidance skip condition",
    }
    return reason_map.get(reason, str(reason))


def _resample_joint_trajectory_to_steps(joint_trajectory: np.ndarray, num_steps: int) -> np.ndarray:
    joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
    if joint_trajectory.ndim != 2:
        raise ValueError(f"joint_trajectory must be rank-2 [T, J], got {joint_trajectory.shape}")
    if joint_trajectory.shape[0] == int(num_steps):
        return joint_trajectory.astype(np.float32)
    if joint_trajectory.shape[0] <= 1:
        return np.repeat(joint_trajectory.astype(np.float32), int(num_steps), axis=0)
    source_axis = np.linspace(0.0, 1.0, joint_trajectory.shape[0], dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float64)
    resampled = np.stack(
        [np.interp(target_axis, source_axis, joint_trajectory[:, joint_index]) for joint_index in range(joint_trajectory.shape[1])],
        axis=1,
    )
    return resampled.astype(np.float32)


def _summarize_qp_status_from_selection(selection: dict[str, object]) -> dict[str, object]:
    guidance_candidates = list(selection.get("guidance_candidates", []) or [])
    guidance_log = dict(selection.get("guidance_log", {}) or {})
    guidance_enabled = bool(selection.get("surface_cbf_qp_guidance_enabled", False))
    late_stage_enabled = bool(selection.get("late_stage_qp_guided_diffusion_enabled", False))
    selected_candidate_idx = int(selection.get("selected_candidate_idx", 0))
    selected_candidate_info = None
    for candidate_info in guidance_candidates:
        if int(candidate_info.get("candidate_index", -1)) == selected_candidate_idx:
            selected_candidate_info = candidate_info
            break

    if late_stage_enabled:
        attempted_count = int(guidance_log.get("guidance_steps_applied", 0) or 0)
        success_count = int(1 if bool(selection.get("planning_success", False)) else 0)
    else:
        attempted_count = sum(bool(candidate_info.get("qp_attempted", False)) for candidate_info in guidance_candidates)
        success_count = sum(bool(candidate_info.get("qp_success", False)) for candidate_info in guidance_candidates)
    qp_attempted = attempted_count > 0
    if late_stage_enabled:
        qp_skip_reason = None if qp_attempted else "no_topk_constraints"
    elif not guidance_enabled:
        qp_skip_reason = "surface_cbf_qp_guidance_disabled"
    elif selected_candidate_info is not None and not bool(selected_candidate_info.get("qp_attempted", False)):
        qp_skip_reason = selected_candidate_info.get("qp_skip_reason")
    elif not qp_attempted:
        qp_skip_reason = "no_topk_constraints"
    else:
        qp_skip_reason = None

    return {
        "qp_attempted": bool(qp_attempted),
        "qp_attempted_count": int(attempted_count),
        "qp_success_count": int(success_count),
        "qp_skip_reason": qp_skip_reason,
        "qp_skip_reason_text": _format_qp_skip_reason(qp_skip_reason),
        "selected_candidate_qp_attempted": bool(
            selected_candidate_info is not None and selected_candidate_info.get("qp_attempted", False)
        ),
        "selected_candidate_qp_success": bool(
            selected_candidate_info is not None and selected_candidate_info.get("qp_success", False)
        ),
        "selected_candidate_qp_skip_reason": None if selected_candidate_info is None else selected_candidate_info.get("qp_skip_reason"),
        "selected_candidate_sdf_value_count": int(
            0 if selected_candidate_info is None else selected_candidate_info.get("sdf_value_count", 0)
        ),
        "selected_candidate_finite_sdf_value_count": int(
            0 if selected_candidate_info is None else selected_candidate_info.get("finite_sdf_value_count", 0)
        ),
        "selected_candidate_finite_sdf_timestep_count": int(
            0 if selected_candidate_info is None else selected_candidate_info.get("finite_sdf_timestep_count", 0)
        ),
        "guidance_num_qp_called": int(guidance_log.get("num_qp_called", 0) or 0),
        "guidance_num_qp_success": int(guidance_log.get("num_qp_success", 0) or 0),
        "guidance_scp_iterations": int(guidance_log.get("scp_iterations_configured", 0) or 0),
        "guidance_selected_candidate_pass_count": int(guidance_log.get("selected_candidate_pass_count", 0) or 0),
        "guidance_selected_candidate_passes_succeeded": int(
            guidance_log.get("selected_candidate_passes_succeeded", 0) or 0
        ),
        "guidance_final_success_source": str(guidance_log.get("final_success_source", "failure") or "failure"),
        "guidance_repair_attempt_count": int(guidance_log.get("repair_attempt_count", 0) or 0),
        "guidance_repair_attempted_indices": list(guidance_log.get("repair_attempted_candidate_indices", []) or []),
    }


def _compute_episode_singularity_summary(
    *,
    validator: PyBulletCollisionValidator,
    start_joint_state: np.ndarray,
    goal_joint_state: np.ndarray,
    joint_trajectory: np.ndarray,
) -> dict[str, object]:
    start_metrics = validator.compute_joint_state_singularity_metrics(start_joint_state)
    goal_metrics = validator.compute_joint_state_singularity_metrics(goal_joint_state)
    trajectory_metrics = validator.compute_joint_trajectory_singularity_metrics(joint_trajectory)
    return {
        "link_name": str(start_metrics.get("link_name", trajectory_metrics.get("link_name", validator.cfg.tcp_link_name))),
        "start": start_metrics,
        "goal": goal_metrics,
        "trajectory": trajectory_metrics,
    }


def _finite_group_mean(entries: list[dict[str, object]], extractor) -> float:
    values = []
    for entry in entries:
        value = extractor(entry)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _build_singularity_group_summary(per_traj_metrics: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    def _scp_used(entry: dict[str, object]) -> bool:
        return bool(entry.get("surface_cbf_qp_guidance_enabled", False)) and int(
            entry.get("guidance_selected_candidate_pass_count", 0) or 0
        ) > 0

    groups = {
        "scp_qp_failed": [
            entry for entry in per_traj_metrics
            if _scp_used(entry) and bool(entry.get("has_collision", False))
        ],
        "scp_qp_collision_free": [
            entry for entry in per_traj_metrics
            if _scp_used(entry) and not bool(entry.get("has_collision", False))
        ],
        "no_qp_collision_free": [
            entry for entry in per_traj_metrics
            if bool(entry.get("surface_cbf_qp_guidance_enabled", False))
            and int(entry.get("guidance_selected_candidate_pass_count", 0) or 0) == 0
            and not bool(entry.get("has_collision", False))
        ],
    }

    summary: dict[str, dict[str, object]] = {}
    for group_name, entries in groups.items():
        summary[group_name] = {
            "count": int(len(entries)),
            "start_sigma_min_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("start") or {}).get("sigma_min")),
            ),
            "start_reciprocal_condition_number_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("start") or {}).get("reciprocal_condition_number")),
            ),
            "start_manipulability_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("start") or {}).get("manipulability")),
            ),
            "goal_sigma_min_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("goal") or {}).get("sigma_min")),
            ),
            "goal_reciprocal_condition_number_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("goal") or {}).get("reciprocal_condition_number")),
            ),
            "goal_manipulability_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("goal") or {}).get("manipulability")),
            ),
            "trajectory_sigma_min_min_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("trajectory") or {}).get("sigma_min_min")),
            ),
            "trajectory_sigma_min_mean_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("trajectory") or {}).get("sigma_min_mean")),
            ),
            "trajectory_reciprocal_condition_number_min_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("trajectory") or {}).get("reciprocal_condition_number_min")),
            ),
            "trajectory_manipulability_min_mean": _finite_group_mean(
                entries,
                lambda entry: (((entry.get("singularity") or {}).get("trajectory") or {}).get("manipulability_min")),
            ),
        }
    return summary


def _print_validation_progress(
    *,
    index: int,
    total: int,
    episode_idx: int,
    workpiece_id: int,
    selection: dict[str, object],
    metric: dict[str, object],
) -> None:
    qp_summary = _summarize_qp_status_from_selection(selection)
    scp_part = (
        f"SCP={qp_summary['guidance_selected_candidate_passes_succeeded']}/"
        f"{qp_summary['guidance_selected_candidate_pass_count']}"
    )
    qp_part = (
        f"{scp_part} QP=yes passes={qp_summary['guidance_num_qp_success']}/{qp_summary['guidance_num_qp_called']}"
        if qp_summary["qp_attempted"]
        else (
            f"{scp_part} QP=no reason={qp_summary['qp_skip_reason_text']} "
            f"finite_sdf={qp_summary['selected_candidate_finite_sdf_value_count']}/"
            f"{qp_summary['selected_candidate_sdf_value_count']} "
            f"finite_timesteps={qp_summary['selected_candidate_finite_sdf_timestep_count']}"
        )
    )
    print(
        f"[{index}/{total}] episode={episode_idx} workpiece_id={workpiece_id} "
        f"collision={bool(metric['has_collision'])} min_sdf={float(metric['min_sdf_distance_m']):.6f} "
        f"goal_error={float(metric['goal_error_m']):.6f} {qp_part}"
    )


def _truncate_status_text(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _build_validation_status_lines(
    *,
    index: int,
    total: int,
    episode_idx: int,
    workpiece_id: int,
    selection: dict[str, object],
    metric: dict[str, object],
    collision_count: int,
    total_segment_collision_steps: float,
    total_segment_steps: float,
) -> list[str]:
    qp_summary = _summarize_qp_status_from_selection(selection)
    traj_collision_rate_so_far = collision_count / index if index > 0 else 0.0
    step_collision_rate_so_far = (
        total_segment_collision_steps / total_segment_steps
        if total_segment_steps > 0
        else 0.0
    )
    inference_elapsed_sec = selection.get("inference_elapsed_sec")
    inference_text = (
        f" infer={float(inference_elapsed_sec):.3f}s"
        if inference_elapsed_sec is not None
        else ""
    )
    progress_line = (
        f"[{index}/{total}] episode={episode_idx} workpiece_id={workpiece_id} "
        f"collision={bool(metric['has_collision'])} "
        f"min_sdf={float(metric['min_sdf_distance_m']):.6f} "
        f"goal_error={float(metric['goal_error_m']):.6f}{inference_text}"
    )
    status_line = (
        f"traj_collision_rate={traj_collision_rate_so_far:.3f} "
        f"step_collision_rate={step_collision_rate_so_far:.3f} "
        f"SCP={qp_summary['guidance_selected_candidate_passes_succeeded']}/"
        f"{qp_summary['guidance_selected_candidate_pass_count']} "
        f"QP={qp_summary['guidance_num_qp_success']}/{qp_summary['guidance_num_qp_called']} "
        f"repair={qp_summary['guidance_repair_attempt_count']} "
        f"final={qp_summary['guidance_final_success_source']} "
        f"selected_candidate={int(selection.get('selected_candidate_idx', 0))}"
    )
    return [progress_line, status_line]


class _ValidationStatusDisplay:
    def __init__(self) -> None:
        self._enabled = sys.stdout.isatty()
        self._last_line_count = 0

    def render(self, lines: list[str]) -> None:
        if not lines:
            return
        width = shutil.get_terminal_size(fallback=(120, 20)).columns
        clipped_lines = [_truncate_status_text(line, width) for line in lines]
        if not self._enabled:
            print(" | ".join(clipped_lines))
            return
        if self._last_line_count > 0:
            sys.stdout.write(f"\x1b[{self._last_line_count}F")
        for line in clipped_lines:
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(line)
            sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_line_count = len(clipped_lines)


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
        "--random-sample-episodes",
        type=int,
        default=None,
        help=(
            "Randomly sample this many validation episodes after job-type and "
            "trajopt-success filtering. Use with --random-sample-seed for reproducibility."
        ),
    )
    parser.add_argument(
        "--random-sample-seed",
        type=int,
        default=42,
        help="Seed for --random-sample-episodes.",
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
        "--num-inference-steps",
        type=int,
        default=10,
        help="DDIM denoising steps for validation inference.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation split ratio. When omitted, uses the checkpoint config value.",
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
    add_surface_cbf_qp_guidance_parser_args(
        parser,
        include_num_candidates=True,
        include_candidate_inference_steps=False,
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
        "--trajopt-success-only",
        action="store_true",
        help=(
            "Only run validation/inference on episodes whose trajopt/planning success flag is true. "
            "The script will look for a per-episode success field in the zarr dataset."
        ),
    )
    parser.add_argument(
        "--trajopt-success-key",
        type=str,
        default=None,
        help=(
            "Optional explicit zarr field name for the per-episode trajopt/planning success flag. "
            "When omitted, the script will try common candidates automatically."
        ),
    )
    parser.add_argument(
        "--trajopt-success-results-dir",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional planning-results root to read trajopt_success directly from "
            "transition_*.json files. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--trajopt-success-match-tol",
        type=float,
        default=1e-5,
        help=(
            "Tolerance used when matching zarr episodes to planning-result JSON/NPZ "
            "via normalized start/goal joint vectors."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file/per-episode warnings from trajopt-success JSON matching.",
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
    val_ratio: Optional[float],
    workspace: TrainDP3Workspace | TrainDP3CSpaceWorkspace,
) -> TransitionTrajectoryDataset:
    """Instantiate the full dataset and return its validation copy."""
    from omegaconf import OmegaConf

    ds_cfg = OmegaConf.select(workspace.cfg, "task.dataset", default={}) or {}
    effective_val_ratio = val_ratio if val_ratio is not None else float(ds_cfg.get("val_ratio", 0.1))
    common_kwargs = dict(
        zarr_path=str(zarr_path),
        horizon=horizon,
        pad_before=int(ds_cfg.get("pad_before", 0)),
        pad_after=int(ds_cfg.get("pad_after", 0)),
        seed=int(ds_cfg.get("seed", 42)),
        val_ratio=effective_val_ratio,
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
            pyb_cfg_raw.get("robot_surface_points_per_link", {"pen_link": 80, "wrist_3_link": 16})
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


def _to_episode_bool_flags_from_array(
    *,
    values: np.ndarray,
    replay_buffer,
    field_name: str,
) -> np.ndarray | None:
    array = np.asarray(values)
    if array.ndim == 0:
        return None
    if array.shape[0] == replay_buffer.n_episodes:
        return np.asarray(array, dtype=bool).reshape(-1)
    if array.shape[0] == replay_buffer.n_steps:
        episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
        flags: list[bool] = []
        for episode_idx in range(replay_buffer.n_episodes):
            start_idx, end_idx = _episode_bounds(episode_ends, episode_idx)
            episode_values = np.asarray(array[start_idx:end_idx]).reshape(-1)
            if episode_values.size == 0:
                raise ValueError(f"Episode {episode_idx} is empty when reading `{field_name}`.")
            first_value = bool(episode_values[0])
            if not np.all(np.asarray(episode_values, dtype=bool) == first_value):
                raise ValueError(
                    f"Field `{field_name}` varies within episode {episode_idx}; "
                    "expected a stable per-episode success flag."
                )
            flags.append(first_value)
        return np.asarray(flags, dtype=bool)
    return None


def _resolve_trajopt_success_flags(
    *,
    replay_buffer,
    explicit_key: str | None,
) -> tuple[np.ndarray, str]:
    candidate_keys: list[str] = []
    if explicit_key is not None:
        candidate_keys.append(str(explicit_key))
    candidate_keys.extend(
        [
            "trajopt_success",
            "plan_success",
            "motion_gen_success",
            "planning_success",
            "success",
        ]
    )

    checked_keys: list[str] = []
    seen_keys: set[str] = set()
    for key in candidate_keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        checked_keys.append(key)

        if key in replay_buffer.meta:
            flags = _to_episode_bool_flags_from_array(
                values=replay_buffer.meta[key][:],
                replay_buffer=replay_buffer,
                field_name=f"meta/{key}",
            )
            if flags is not None:
                return flags, f"meta/{key}"

        if key in replay_buffer:
            flags = _to_episode_bool_flags_from_array(
                values=replay_buffer[key][:],
                replay_buffer=replay_buffer,
                field_name=f"data/{key}",
            )
            if flags is not None:
                return flags, f"data/{key}"

    raise KeyError(
        "Unable to find a per-episode trajopt/planning success flag in the zarr dataset. "
        f"Checked keys: {checked_keys}. "
        "Pass --trajopt-success-key <field> if your dataset uses a different name."
    )


def _filter_episode_indices_by_trajopt_success(
    episode_indices: np.ndarray,
    trajopt_success_flags: np.ndarray,
) -> np.ndarray:
    filtered_episode_indices = [
        int(episode_idx)
        for episode_idx in np.asarray(episode_indices, dtype=np.int64).tolist()
        if bool(trajopt_success_flags[int(episode_idx)])
    ]
    return np.asarray(filtered_episode_indices, dtype=np.int64)


def _default_urdf_path() -> pathlib.Path:
    return _PROJECT_ROOT / "config" / "ur5e_with_pen.urdf"


def _load_joint_limits_from_urdf(urdf_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    resolved_urdf_path = pathlib.Path(
        urdf_path if urdf_path is not None else str(_default_urdf_path())
    ).expanduser().resolve()
    root = ET.parse(resolved_urdf_path).getroot()
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        lower_limits.append(float(lower))
        upper_limits.append(float(upper))
    if not lower_limits:
        raise ValueError(f"No revolute joint limits found in URDF: {resolved_urdf_path}")
    return (
        np.asarray(lower_limits, dtype=np.float32),
        np.asarray(upper_limits, dtype=np.float32),
    )


def _denormalize_joint_angles(
    normalized_joint_angles: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    normalized_joint_angles = np.asarray(normalized_joint_angles, dtype=np.float32).reshape(-1)
    normalized_01 = (normalized_joint_angles + 1.0) * 0.5
    return (lower_limits + normalized_01 * (upper_limits - lower_limits)).astype(np.float32)


def _resolve_episode_joint_vectors(
    replay_buffer,
    field_name: str,
) -> np.ndarray:
    if field_name not in replay_buffer:
        raise KeyError(
            f"Expected zarr data/{field_name} for trajopt-success JSON matching, but it is missing."
        )
    array = np.asarray(replay_buffer[field_name][:], dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Expected zarr data/{field_name} to have shape [T, J], got {array.shape}."
        )
    episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
    vectors: list[np.ndarray] = []
    for episode_idx in range(replay_buffer.n_episodes):
        start_idx, end_idx = _episode_bounds(episode_ends, episode_idx)
        episode_values = np.asarray(array[start_idx:end_idx], dtype=np.float32)
        if episode_values.shape[0] == 0:
            raise ValueError(f"Episode {episode_idx} is empty when reading data/{field_name}.")
        first_value = np.asarray(episode_values[0], dtype=np.float32).reshape(-1)
        if not np.allclose(episode_values, first_value[None, :], atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Field data/{field_name} varies within episode {episode_idx}; "
                "expected a stable per-episode vector."
            )
        vectors.append(first_value)
    return np.asarray(vectors, dtype=np.float32)


def _resolve_episode_reversed_flags(replay_buffer) -> np.ndarray:
    if "is_reversed_episode" not in replay_buffer.meta:
        return np.zeros(replay_buffer.n_episodes, dtype=bool)
    values = np.asarray(replay_buffer.meta["is_reversed_episode"][:], dtype=np.int64).reshape(-1)
    if values.shape != (replay_buffer.n_episodes,):
        raise ValueError(
            "`meta/is_reversed_episode` must have shape "
            f"({replay_buffer.n_episodes},), got {values.shape}"
        )
    return values.astype(bool)


def _resolve_job_name_from_path(path: pathlib.Path) -> str | None:
    for parent in path.parents:
        if parent.name.startswith("job_"):
            return parent.name
    return None


def _infer_source_kind_from_results_dirs(
    path: pathlib.Path,
    results_dirs: list[pathlib.Path],
) -> str:
    resolved_path = path.expanduser().resolve()
    matches: list[pathlib.Path] = []
    for results_dir in results_dirs:
        resolved_dir = results_dir.expanduser().resolve()
        try:
            resolved_path.relative_to(resolved_dir)
            matches.append(resolved_dir)
        except ValueError:
            continue
    if matches:
        matched_root = max(matches, key=lambda current: len(current.parts))
        return "simple" if "simple" in matched_root.name.lower() else "regular"
    for parent in resolved_path.parents:
        if "simple" in parent.name.lower():
            return "simple"
    return "regular"


def _encode_workpiece_id(local_workpiece_id: int, source_kind: str) -> int:
    return 1000 + int(local_workpiece_id) if source_kind == "simple" else int(local_workpiece_id)


def _episode_signature_key(
    workpiece_id: int,
    first_joint_angles: np.ndarray,
    last_joint_angles: np.ndarray,
    tol: float,
) -> tuple[int, tuple[int, ...]]:
    if tol <= 0:
        raise ValueError(f"trajopt-success match tolerance must be positive, got {tol}")
    signature = np.concatenate(
        [
            np.asarray(first_joint_angles, dtype=np.float32).reshape(-1),
            np.asarray(last_joint_angles, dtype=np.float32).reshape(-1),
        ]
    )
    quantized = np.rint(signature / float(tol)).astype(np.int64)
    return int(workpiece_id), tuple(int(value) for value in quantized.tolist())


def _resolve_trajopt_success_flags_from_results_json(
    *,
    replay_buffer,
    workpiece_ids: np.ndarray,
    goal_position_norm_m: float,
    urdf_path: str | None,
    results_dirs: list[str],
    match_tol: float,
    quiet: bool = False,
) -> tuple[np.ndarray, str]:
    resolved_results_dirs = [
        pathlib.Path(path).expanduser().resolve() for path in results_dirs
    ]
    json_paths: list[pathlib.Path] = []
    for results_dir in resolved_results_dirs:
        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"--trajopt-success-results-dir does not exist or is not a directory: {results_dir}"
            )
        json_paths.extend(sorted(results_dir.rglob("transition_*.json")))
    if not json_paths:
        raise FileNotFoundError(
            "No transition_*.json files found under trajopt-success results dirs: "
            f"{[str(path) for path in resolved_results_dirs]}"
        )

    signature_to_success: dict[tuple[int, tuple[int, ...]], bool] = {}
    for json_path in json_paths:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        array_file = metadata.get("array_file")
        if not array_file:
            raise KeyError(f"Missing `array_file` in planning metadata: {json_path}")
        npz_path = (json_path.parent / str(array_file)).resolve()
        if not npz_path.is_file():
            if not quiet:
                print(f"  [trajopt-success-json] Warning: Planning metadata {json_path} references missing NPZ file: {npz_path}. Skipping.")
            continue
        job_name = _resolve_job_name_from_path(npz_path)
        if job_name is None:
            raise ValueError(f"Unable to infer job name from planning NPZ path: {npz_path}")
        try:
            local_workpiece_id = int(job_name.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid job name format for planning NPZ path: {npz_path}") from exc
        source_kind = _infer_source_kind_from_results_dirs(npz_path, resolved_results_dirs)
        workpiece_id = _encode_workpiece_id(local_workpiece_id, source_kind)
        planning_input = load_bspline_planning_input_data(
            npz_path=str(npz_path),
            norm=float(goal_position_norm_m),
            urdf_path=urdf_path,
        )
        success_value = bool(metadata.get("trajopt_success", False))
        forward_key = _episode_signature_key(
            workpiece_id=workpiece_id,
            first_joint_angles=planning_input.first_joint_angles_normalized,
            last_joint_angles=planning_input.last_joint_angles_normalized,
            tol=match_tol,
        )
        reversed_key = _episode_signature_key(
            workpiece_id=workpiece_id,
            first_joint_angles=planning_input.last_joint_angles_normalized,
            last_joint_angles=planning_input.first_joint_angles_normalized,
            tol=match_tol,
        )
        for key in (forward_key, reversed_key):
            existing = signature_to_success.get(key)
            if existing is not None and bool(existing) != success_value:
                raise ValueError(
                    "Conflicting trajopt_success values found for the same trajectory signature "
                    f"while indexing planning results: {json_path}"
                )
            signature_to_success[key] = success_value

    episode_first_joint_angles = _resolve_episode_joint_vectors(
        replay_buffer, "first_joint_angles_normalized"
    )
    episode_last_joint_angles = _resolve_episode_joint_vectors(
        replay_buffer, "last_joint_angles_normalized"
    )
    episode_reversed_flags = _resolve_episode_reversed_flags(replay_buffer)

    flags: list[bool] = []
    unmatched_episodes: list[int] = []
    lower_limits, upper_limits = _load_joint_limits_from_urdf(urdf_path)
    for episode_idx in range(replay_buffer.n_episodes):
        first_joint_angles = episode_first_joint_angles[episode_idx]
        last_joint_angles = episode_last_joint_angles[episode_idx]
        signature_key = _episode_signature_key(
            workpiece_id=int(workpiece_ids[episode_idx]),
            first_joint_angles=first_joint_angles,
            last_joint_angles=last_joint_angles,
            tol=match_tol,
        )
        success_flag = signature_to_success.get(signature_key)
        if success_flag is None:
            unmatched_episodes.append(int(episode_idx))
            if len(unmatched_episodes) <= 5 and not quiet:
                start_joint_angles = _denormalize_joint_angles(
                    first_joint_angles, lower_limits, upper_limits
                )
                end_joint_angles = _denormalize_joint_angles(
                    last_joint_angles, lower_limits, upper_limits
                )
                print(
                    "  [trajopt-success-json] unmatched episode "
                    f"{episode_idx}: workpiece_id={int(workpiece_ids[episode_idx])} "
                    f"is_reversed={bool(episode_reversed_flags[episode_idx])} "
                    f"q_start={np.array2string(start_joint_angles, precision=4)} "
                    f"q_goal={np.array2string(end_joint_angles, precision=4)}"
                )
            success_flag = False
        flags.append(bool(success_flag))

    if unmatched_episodes:
        print(
            "  [trajopt-success-json] warning: "
            f"{len(unmatched_episodes)} / {replay_buffer.n_episodes} episodes could not be matched "
            "to planning-result JSON and will be treated as trajopt_success=false."
        )

    return (
        np.asarray(flags, dtype=bool),
        "planning_results_json:" + ",".join(str(path) for path in resolved_results_dirs),
    )


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
        surface_points_per_link_override=_qp_guided_surface_points_per_link(args),
    )
    guidance_config = SurfaceCBFQPGuidanceConfig(
        enabled=True,
        num_candidates=int(num_candidates),
        guidance_steps=int(args.guidance_steps),
        max_risk_segments=int(args.guidance_max_risk_segments),
        window_radius=int(args.guidance_window_radius),
        points_per_segment=int(args.guidance_points_per_segment),
        min_constraints_per_segment=int(args.guidance_min_constraints_per_segment),
        active_constraints=int(args.guidance_active_constraints),
        check_steps=int(args.guidance_check_steps),
        cert_steps=int(args.guidance_cert_steps),
        cert_swept_intermediate=int(args.guidance_cert_swept_intermediate),
        d_safe=float(args.guidance_d_safe),
        d_trigger=float(args.guidance_d_trigger),
        d_cert=float(args.guidance_d_cert),
        eps_deep=float(args.guidance_eps_deep),
        delta_max=float(args.guidance_delta_max),
        scp_iterations=int(args.guidance_scp_iterations),
        delta_max_total=float(args.guidance_delta_max_total),
        delta_max_pass1=float(args.guidance_delta_max_pass1),
        delta_max_pass2=float(args.guidance_delta_max_pass2),
        d_trigger_pass2_offset=float(args.guidance_d_trigger_pass2_offset),
        margin_buffer=float(args.guidance_margin_buffer),
        enable_local_waypoint_qp_after_certificate=bool(args.enable_local_waypoint_qp_after_certificate),
        local_waypoint_qp_window_radius=int(args.local_waypoint_qp_window_radius),
        local_waypoint_qp_max_collision_segments=int(args.local_waypoint_qp_max_collision_segments),
        local_waypoint_qp_min_clearance_trigger=float(args.local_waypoint_qp_min_clearance_trigger),
        local_waypoint_qp_target_buffer=float(args.local_waypoint_qp_target_buffer),
        local_waypoint_qp_lambda_s=float(args.local_waypoint_qp_lambda_s),
        local_waypoint_qp_delta_max=float(args.local_waypoint_qp_delta_max),
        local_waypoint_qp_max_velocity_step=float(args.local_waypoint_qp_max_velocity_step),
        local_waypoint_qp_max_acceleration_step=float(args.local_waypoint_qp_max_acceleration_step),
        local_waypoint_qp_maxiter=int(args.local_waypoint_qp_maxiter),
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
    guided_joint_trajectory = result.get("guided_joint_trajectory")
    if guided_joint_trajectory is not None:
        selected_result["joint_trajectory"] = _resample_joint_trajectory_to_steps(
            np.asarray(guided_joint_trajectory, dtype=np.float32),
            int(pyb_cfg.target_steps),
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
        "selected_candidate_idx": int(guidance_log.get("selected_candidate_index", guidance_log.get("best_candidate_index", 0)) or 0),
        "selected_candidate_seed": int(base_seed),
        "selected_action_horizon": np.asarray(selected_action_horizon, dtype=np.float32),
        "selected_joint_trajectory": np.asarray(selected_result["joint_trajectory"], dtype=np.float32),
        "selected_score_details": selected_score_details,
        "candidate_score_details": [selected_score_details],
        "candidate_seeds": [int(base_seed)],
        "num_candidates": int(num_candidates),
        "inference_elapsed_sec": inference_elapsed_sec,
        "guidance_log": guidance_log,
        "guidance_candidates": list(result.get("guidance_candidates", [])),
        "guidance_candidate_count": int(guidance_log.get("num_candidates_total", len(result.get("guidance_candidates", []))) or 0),
        "guidance_repair_attempt_count": int(guidance_log.get("repair_attempt_count", 0) or 0),
        "guidance_repair_attempted_indices": list(guidance_log.get("repair_attempted_candidate_indices", []) or []),
        "final_success_source": str(guidance_log.get("final_success_source", "failure") or "failure"),
    }


def _predict_late_stage_qp_guided_diffusion(
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
    skip_final_certification: bool = False,
) -> dict[str, object]:
    from diffusion_policy_3d.common.late_stage_qp_guided_ddim import (
        LateStageQPGuidedDDIMConfig,
        LateStageQPGuidedDDIMRunner,
    )
    from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
        PyBulletSurfaceEnvironmentAdapter,
        SurfaceCBFQPGuidanceConfig,
    )

    if validator.stats_mode != "bspline":
        raise ValueError(
            "Late-stage QP-guided diffusion currently requires bspline stats mode, "
            f"got {validator.stats_mode!r}."
        )

    num_candidates = (
        int(num_candidates_override)
        if num_candidates_override is not None
        else int(args.num_candidates if args.num_candidates is not None else pyb_cfg.num_candidates)
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
        surface_points_per_link_override=_qp_guided_surface_points_per_link(args),
    )
    scp_config = SurfaceCBFQPGuidanceConfig(
        enabled=True,
        num_candidates=int(num_candidates),
        guidance_steps=int(args.guidance_steps),
        max_risk_segments=int(args.guidance_max_risk_segments),
        window_radius=int(args.guidance_window_radius),
        points_per_segment=int(args.guidance_points_per_segment),
        min_constraints_per_segment=int(args.guidance_min_constraints_per_segment),
        active_constraints=int(args.guidance_active_constraints),
        check_steps=int(args.coarse_check_steps),
        cert_steps=int(args.guidance_cert_steps),
        cert_swept_intermediate=int(args.guidance_cert_swept_intermediate),
        d_safe=float(args.guidance_safe_distance),
        d_trigger=float(args.guidance_trigger_distance),
        d_cert=float(args.guidance_d_cert),
        eps_deep=float(args.guidance_eps_deep),
        delta_max=float(args.guidance_delta_max),
        scp_iterations=int(args.qp_inner_scp_rounds),
        delta_max_total=float(args.trust_region_end),
        delta_max_pass1=float(args.trust_region_start),
        delta_max_pass2=float(args.trust_region_end),
        d_trigger_pass2_offset=float(args.guidance_d_trigger_pass2_offset),
        margin_buffer=float(args.guidance_margin_buffer),
        enable_local_waypoint_qp_after_certificate=bool(args.enable_local_waypoint_qp_after_certificate),
        local_waypoint_qp_window_radius=int(args.local_waypoint_qp_window_radius),
        local_waypoint_qp_max_collision_segments=int(args.local_waypoint_qp_max_collision_segments),
        local_waypoint_qp_min_clearance_trigger=float(args.local_waypoint_qp_min_clearance_trigger),
        local_waypoint_qp_target_buffer=float(args.local_waypoint_qp_target_buffer),
        local_waypoint_qp_lambda_s=float(args.local_waypoint_qp_lambda_s),
        local_waypoint_qp_delta_max=float(args.local_waypoint_qp_delta_max),
        local_waypoint_qp_max_velocity_step=float(args.local_waypoint_qp_max_velocity_step),
        local_waypoint_qp_max_acceleration_step=float(args.local_waypoint_qp_max_acceleration_step),
        local_waypoint_qp_maxiter=int(args.local_waypoint_qp_maxiter),
        lambda_s=float(args.guidance_lambda_s),
        rho=float(args.guidance_rho),
        ddim_eta=float(args.guidance_ddim_eta),
        joint_limit_steps=int(args.guidance_joint_limit_steps),
        fallback_to_terminal_cbf=False,
    )
    guidance_config = LateStageQPGuidedDDIMConfig(
        enabled=True,
        num_candidates=int(num_candidates),
        guidance_steps=int(args.guidance_steps),
        guidance_timesteps=tuple(int(v) for v in (args.guidance_timesteps or [])),
        qp_candidates=int(args.qp_candidates),
        qp_inner_scp_rounds=int(args.qp_inner_scp_rounds),
        coarse_check_steps=int(args.coarse_check_steps),
        guidance_trigger_distance=float(args.guidance_trigger_distance),
        guidance_safe_distance=float(args.guidance_safe_distance),
        trust_region_start=float(args.trust_region_start),
        trust_region_end=float(args.trust_region_end),
        blend_weights=tuple(float(v) for v in args.blend_weights),
        repair_score_weights=tuple(float(v) for v in args.repair_score_weights),
        ddim_eta=float(args.guidance_ddim_eta),
        skip_final_certification=bool(skip_final_certification),
        scp_config=scp_config,
    )
    guidance_runner = LateStageQPGuidedDDIMRunner(config=guidance_config, environment=environment)
    if not getattr(_predict_late_stage_qp_guided_diffusion, "_printed_device_report", False):
        late_stage_geometry_backend = (
            "cuda"
            if environment.torch_available() and torch.cuda.is_available()
            else ("cpu" if environment.torch_available() else "disabled")
        )
        _print_runtime_device_report(
            prefix="runtime-device",
            requested_device=device,
            policy=policy,
            obs_dict=obs_dict,
            planner_mode="qp_guided_diffusion",
            late_stage_geometry_backend=late_stage_geometry_backend,
        )
        _predict_late_stage_qp_guided_diffusion._printed_device_report = True

    start_joint_normalized = raw_obs["first_joint_angles_normalized"][0].astype(np.float32)
    end_joint_normalized = raw_obs["last_joint_angles_normalized"][0].astype(np.float32)
    scheduler_step_kwargs = {"eta": float(args.guidance_ddim_eta)}

    start_time = time.perf_counter() if measure_inference_time else None
    result = policy.sample_with_late_stage_qp_guided_diffusion(
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
    selected_action_horizon = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
    guided_joint_trajectory = result.get("guided_joint_trajectory")
    planning_success = bool(result.get("planning_success", False))
    if guided_joint_trajectory is not None and np.asarray(guided_joint_trajectory).size > 0:
        joint_trajectory = _resample_joint_trajectory_to_steps(
            np.asarray(guided_joint_trajectory, dtype=np.float32),
            int(pyb_cfg.target_steps),
        )
    else:
        joint_trajectory = np.empty((0, len(validator.revolute_joint_indices)), dtype=np.float32)
    selected_score_details = (
        validator.score_candidate(
            workpiece_id=workpiece_id,
            normalized_control_points=np.asarray(result.get("guided_control_points_normalized"), dtype=np.float32),
            joint_trajectory=joint_trajectory,
        )
        if planning_success
        else {}
    )
    end_time = time.perf_counter() if measure_inference_time else None
    guidance_log = dict(result.get("guidance_log", {}) or {})
    inference_elapsed_sec = (
        float(end_time - start_time)
        if measure_inference_time and start_time is not None and end_time is not None
        else None
    )
    return {
        "planner_mode": "qp_guided_diffusion",
        "planning_success": planning_success,
        "candidate_pool_enabled": False,
        "surface_cbf_qp_guidance_enabled": False,
        "late_stage_qp_guided_diffusion_enabled": True,
        "selected_candidate_idx": int(guidance_log.get("selected_candidate_index", -1)),
        "selected_candidate_seed": int(base_seed),
        "selected_action_horizon": np.asarray(selected_action_horizon, dtype=np.float32),
        "selected_joint_trajectory": np.asarray(joint_trajectory, dtype=np.float32),
        "selected_score_details": selected_score_details,
        "candidate_score_details": [selected_score_details] if selected_score_details else [],
        "candidate_seeds": [int(base_seed)],
        "num_candidates": int(num_candidates),
        "inference_elapsed_sec": inference_elapsed_sec,
        "guidance_log": guidance_log,
        "guidance_candidates": list(result.get("guidance_candidates", [])),
        "guidance_candidate_count": int(guidance_log.get("num_candidates_guided", num_candidates) or num_candidates),
        "final_success_source": "qp_guided_diffusion" if planning_success else "planning_failure",
    }


def _select_late_stage_topk_residuals_for_post_qp(
    *,
    late_stage_selection: dict[str, object],
    top_k: int,
) -> tuple[list[np.ndarray], list[int]]:
    final_candidates = list(late_stage_selection.get("guidance_candidates", []) or [])
    final_by_index = {
        int(candidate_info.get("candidate_index", -1)): candidate_info
        for candidate_info in final_candidates
    }
    guidance_log = dict(late_stage_selection.get("guidance_log", {}) or {})
    last_step_infos: list[dict[str, object]] = []
    for step_info in reversed(list(guidance_log.get("guidance_step_infos", []) or [])):
        candidate_infos = list(step_info.get("candidate_infos", []) or [])
        if candidate_infos:
            last_step_infos = candidate_infos
            break

    if last_step_infos:
        ordered_indices = [
            int(info.get("candidate_index", -1))
            for info in sorted(
                last_step_infos,
                key=lambda info: (
                    float("inf")
                    if not np.isfinite(float(info.get("repairability_score", float("inf"))))
                    else float(info.get("repairability_score", float("inf"))),
                    int(info.get("candidate_index", -1)),
                ),
            )
        ]
    else:
        ordered_indices = [
            int(candidate_info.get("candidate_index", -1))
            for candidate_info in final_candidates
        ]

    selected_residuals: list[np.ndarray] = []
    selected_indices: list[int] = []
    for candidate_index in ordered_indices:
        if candidate_index not in final_by_index:
            continue
        residual = np.asarray(
            final_by_index[candidate_index].get("normalized_free_residual", np.empty((0, 0))),
            dtype=np.float32,
        )
        if residual.size == 0:
            continue
        selected_residuals.append(residual)
        selected_indices.append(int(candidate_index))
        if len(selected_residuals) >= int(top_k):
            break
    return selected_residuals, selected_indices


def _predict_qp_guided_diffusion_then_post_qp(
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

    late_stage_selection = _predict_late_stage_qp_guided_diffusion(
        policy=policy,
        validator=validator,
        pyb_cfg=pyb_cfg,
        obs_dict=obs_dict,
        raw_obs=raw_obs,
        workpiece_id=workpiece_id,
        device=device,
        args=args,
        batch_start=batch_start,
        num_candidates_override=num_candidates_override,
        measure_inference_time=measure_inference_time,
        skip_final_certification=True,
    )
    candidate_residuals, selected_late_stage_indices = _select_late_stage_topk_residuals_for_post_qp(
        late_stage_selection=late_stage_selection,
        top_k=max(1, int(args.final_post_qp_candidates) + int(args.final_backup_candidates)),
    )
    if not candidate_residuals:
        late_stage_selection["planner_mode"] = "qp_guided_diffusion_post_qp"
        late_stage_selection["surface_cbf_qp_guidance_enabled"] = True
        late_stage_selection["late_stage_qp_guided_diffusion_enabled"] = True
        late_stage_selection["final_success_source"] = "late_stage_no_candidate_residuals"
        return late_stage_selection

    num_candidates = int(len(candidate_residuals))
    environment = PyBulletSurfaceEnvironmentAdapter(
        validator=validator,
        workpiece_id=int(workpiece_id),
        joint_lower_limits=np.asarray(validator.joint_lower_limits, dtype=np.float32),
        joint_upper_limits=np.asarray(validator.joint_upper_limits, dtype=np.float32),
        surface_points_per_link_override=_qp_guided_surface_points_per_link(args),
    )
    guidance_config = SurfaceCBFQPGuidanceConfig(
        enabled=True,
        num_candidates=int(num_candidates),
        guidance_steps=int(args.guidance_steps),
        max_risk_segments=int(args.guidance_max_risk_segments),
        window_radius=int(args.guidance_window_radius),
        points_per_segment=int(args.guidance_points_per_segment),
        min_constraints_per_segment=int(args.guidance_min_constraints_per_segment),
        active_constraints=int(args.guidance_active_constraints),
        check_steps=int(args.guidance_check_steps),
        cert_steps=int(args.guidance_cert_steps),
        cert_swept_intermediate=int(args.guidance_cert_swept_intermediate),
        d_safe=float(args.guidance_d_safe),
        d_trigger=float(args.guidance_d_trigger),
        d_cert=float(args.guidance_d_cert),
        eps_deep=float(args.guidance_eps_deep),
        delta_max=float(args.guidance_delta_max),
        scp_iterations=int(args.final_post_qp_rounds),
        delta_max_total=float(args.guidance_delta_max_total),
        delta_max_pass1=float(args.guidance_delta_max_pass1),
        delta_max_pass2=float(args.guidance_delta_max_pass2),
        d_trigger_pass2_offset=float(args.guidance_d_trigger_pass2_offset),
        margin_buffer=float(args.guidance_margin_buffer),
        enable_local_waypoint_qp_after_certificate=bool(args.enable_local_waypoint_qp_after_certificate),
        local_waypoint_qp_window_radius=int(args.local_waypoint_qp_window_radius),
        local_waypoint_qp_max_collision_segments=int(args.local_waypoint_qp_max_collision_segments),
        local_waypoint_qp_min_clearance_trigger=float(args.local_waypoint_qp_min_clearance_trigger),
        local_waypoint_qp_target_buffer=float(args.local_waypoint_qp_target_buffer),
        local_waypoint_qp_lambda_s=float(args.local_waypoint_qp_lambda_s),
        local_waypoint_qp_delta_max=float(args.local_waypoint_qp_delta_max),
        local_waypoint_qp_max_velocity_step=float(args.local_waypoint_qp_max_velocity_step),
        local_waypoint_qp_max_acceleration_step=float(args.local_waypoint_qp_max_acceleration_step),
        local_waypoint_qp_maxiter=int(args.local_waypoint_qp_maxiter),
        lambda_s=float(args.guidance_lambda_s),
        rho=float(args.guidance_rho),
        ddim_eta=float(args.guidance_ddim_eta),
        joint_limit_steps=int(args.guidance_joint_limit_steps),
        fallback_to_terminal_cbf=False,
    )
    guidance_runner = SurfaceCBFQPGuidanceRunner(
        config=guidance_config,
        environment=environment,
    )
    start_joint_normalized = raw_obs["first_joint_angles_normalized"][0].astype(np.float32)
    end_joint_normalized = raw_obs["last_joint_angles_normalized"][0].astype(np.float32)
    post_result = guidance_runner.run(
        candidate_residuals=np.stack(candidate_residuals, axis=0).astype(np.float32),
        q_start_normalized=start_joint_normalized,
        q_goal_normalized=end_joint_normalized,
        delta_w_mean=np.asarray(validator.stats_mean, dtype=np.float32),
        delta_w_std=np.asarray(validator.stats_std, dtype=np.float32),
        num_control_points=int(pyb_cfg.num_control_points),
        spline_degree=int(pyb_cfg.spline_degree),
    )
    selected_action_horizon = (
        policy.normalizer["action"]
        .unnormalize(
            torch.from_numpy(np.asarray(post_result.best_normalized_free_residual, dtype=np.float32))
            .to(device=policy.device, dtype=policy.dtype)
            .unsqueeze(0)
        )[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    selected_joint_trajectory = _resample_joint_trajectory_to_steps(
        np.asarray(post_result.best_joint_trajectory, dtype=np.float32),
        int(pyb_cfg.target_steps),
    )
    selected_score_details = validator.score_candidate(
        workpiece_id=workpiece_id,
        normalized_control_points=np.asarray(post_result.best_control_points_normalized, dtype=np.float32),
        joint_trajectory=selected_joint_trajectory,
    )
    post_guidance_log = post_result.log.to_dict()
    combined_guidance_log = dict(post_guidance_log)
    combined_guidance_log["planner_mode"] = "qp_guided_diffusion_post_qp"
    combined_guidance_log["late_stage_guidance_log"] = dict(late_stage_selection.get("guidance_log", {}) or {})
    combined_guidance_log["late_stage_planning_success"] = bool(late_stage_selection.get("planning_success", False))
    combined_guidance_log["combined_post_qp_candidate_indices"] = [int(v) for v in selected_late_stage_indices]
    combined_guidance_log["combined_post_qp_candidate_count"] = int(num_candidates)
    return {
        "planner_mode": "qp_guided_diffusion_post_qp",
        "planning_success": True,
        "candidate_pool_enabled": False,
        "surface_cbf_qp_guidance_enabled": True,
        "late_stage_qp_guided_diffusion_enabled": True,
        "selected_candidate_idx": int(post_guidance_log.get("selected_candidate_index", post_guidance_log.get("best_candidate_index", 0)) or 0),
        "selected_candidate_seed": int(late_stage_selection.get("selected_candidate_seed", 0)),
        "selected_action_horizon": np.asarray(selected_action_horizon, dtype=np.float32),
        "selected_joint_trajectory": np.asarray(selected_joint_trajectory, dtype=np.float32),
        "selected_score_details": selected_score_details,
        "candidate_score_details": [selected_score_details],
        "candidate_seeds": list(late_stage_selection.get("candidate_seeds", [])),
        "num_candidates": int(num_candidates),
        "inference_elapsed_sec": late_stage_selection.get("inference_elapsed_sec"),
        "guidance_log": combined_guidance_log,
        "guidance_candidates": list(post_result.candidate_infos),
        "guidance_candidate_count": int(len(post_result.candidate_infos)),
        "guidance_repair_attempt_count": int(post_guidance_log.get("repair_attempt_count", 0) or 0),
        "guidance_repair_attempted_indices": list(post_guidance_log.get("repair_attempted_candidate_indices", []) or []),
        "final_success_source": str(post_guidance_log.get("final_success_source", "failure") or "failure"),
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
            "surface_cbf_qp_guidance_enabled": False,
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
            "guidance_log": {},
            "guidance_candidates": [],
            "guidance_candidate_count": 0,
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
        "surface_cbf_qp_guidance_enabled": False,
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
        "guidance_log": {},
        "guidance_candidates": [],
        "guidance_candidate_count": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    apply_surface_cbf_qp_guidance_config(
        args,
        include_num_candidates=True,
        include_candidate_inference_steps=False,
    )
    if args.qp_candidates <= 0:
        raise ValueError(f"qp-candidates must be positive, got {args.qp_candidates}")
    if args.qp_inner_scp_rounds <= 0:
        raise ValueError(f"qp-inner-scp-rounds must be positive, got {args.qp_inner_scp_rounds}")
    if args.coarse_check_steps <= 1:
        raise ValueError(f"coarse-check-steps must be greater than 1, got {args.coarse_check_steps}")
    planner_mode = str(getattr(args, "planner_mode", "baseline"))
    uses_late_stage_guidance = planner_mode in {"qp_guided_diffusion", "qp_guided_diffusion_post_qp"}
    guidance_timesteps = list(args.guidance_timesteps or [])
    if uses_late_stage_guidance and guidance_timesteps and any(int(v) <= 0 for v in guidance_timesteps):
        raise ValueError(f"guidance-timesteps must contain positive integers, got {args.guidance_timesteps}")
    if int(args.num_inference_steps) <= 0:
        raise ValueError(f"num-inference-steps must be positive, got {args.num_inference_steps}")
    if uses_late_stage_guidance and guidance_timesteps and max(int(v) for v in guidance_timesteps) > int(args.num_inference_steps):
        raise ValueError(
            "guidance-timesteps cannot exceed num-inference-steps, "
            f"got {args.guidance_timesteps} vs {args.num_inference_steps}"
        )
    if args.guidance_pen_link_points <= 0 or args.guidance_wrist3_points <= 0:
        raise ValueError("guidance-pen-link-points and guidance-wrist3-points must be positive")
    if args.final_post_qp_candidates <= 0:
        raise ValueError(f"final-post-qp-candidates must be positive, got {args.final_post_qp_candidates}")
    if args.final_backup_candidates < 0:
        raise ValueError(f"final-backup-candidates must be non-negative, got {args.final_backup_candidates}")
    if args.final_post_qp_rounds <= 0:
        raise ValueError(f"final-post-qp-rounds must be positive, got {args.final_post_qp_rounds}")
    if args.trust_region_start <= 0.0 or args.trust_region_end <= 0.0:
        raise ValueError("trust-region-start/end must be positive")
    if args.trust_region_start > args.trust_region_end:
        raise ValueError("trust-region-start must be <= trust-region-end")
    expected_blend_count = len(guidance_timesteps) if guidance_timesteps else int(args.guidance_steps)
    if uses_late_stage_guidance and len(args.blend_weights) != expected_blend_count:
        raise ValueError(
            "blend-weights length must match explicit guidance-timesteps, or guidance-steps when timesteps are empty, "
            f"got {len(args.blend_weights)} vs {expected_blend_count}"
        )
    if len(args.repair_score_weights) != 3:
        raise ValueError("repair-score-weights must contain exactly 3 values")

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
    from omegaconf import OmegaConf

    ds_cfg = OmegaConf.select(workspace.cfg, "task.dataset", default={}) or {}
    effective_val_ratio = args.val_ratio if args.val_ratio is not None else float(ds_cfg.get("val_ratio", 0.1))
    print(f"  val_ratio={effective_val_ratio}" + (" (from checkpoint config)" if args.val_ratio is None else " (CLI override)"))
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
    trajopt_success_flags = None
    trajopt_success_source = None
    if args.trajopt_success_only:
        if not args.trajopt_success_results_dir:
            trajopt_success_flags, trajopt_success_source = _resolve_trajopt_success_flags(
                replay_buffer=replay_buffer,
                explicit_key=args.trajopt_success_key,
            )
            print(
                "  trajopt_success_only=true -> using success flag from "
                f"{trajopt_success_source}"
            )

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
    pyb_cfg.inference_num_steps = int(args.num_inference_steps)
    if args.trajopt_success_only and args.trajopt_success_results_dir:
        trajopt_success_flags, trajopt_success_source = _resolve_trajopt_success_flags_from_results_json(
            replay_buffer=replay_buffer,
            workpiece_ids=workpiece_ids,
            goal_position_norm_m=float(pyb_cfg.goal_position_norm_m),
            urdf_path=pyb_cfg.urdf_path,
            results_dirs=args.trajopt_success_results_dir,
            match_tol=float(args.trajopt_success_match_tol),
            quiet=bool(args.quiet),
        )
        print(
            "  trajopt_success_only=true -> using success flag from "
            f"{trajopt_success_source}"
        )
    candidate_pool_enabled = _candidate_pool_enabled(args.candidate_pool)
    planner_mode = str(args.planner_mode)
    guidance_enabled = planner_mode in {"post_qp", "qp_guided_diffusion_post_qp"}
    qp_guided_diffusion_enabled = planner_mode in {"qp_guided_diffusion", "qp_guided_diffusion_post_qp"}
    effective_num_candidates = (
        int(args.num_candidates)
        if args.num_candidates is not None
        else (
            int(pyb_cfg.num_candidates)
            if guidance_enabled or qp_guided_diffusion_enabled
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
    print(f"  planner_mode={planner_mode}")
    print(f"  surface_cbf_qp_guidance={guidance_enabled}")
    print(f"  late_stage_qp_guided_diffusion={qp_guided_diffusion_enabled}")
    if guidance_enabled or qp_guided_diffusion_enabled:
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
    if args.trajopt_success_only:
        before_trajopt_filter = len(val_episode_indices)
        val_episode_indices = _filter_episode_indices_by_trajopt_success(
            val_episode_indices,
            np.asarray(trajopt_success_flags, dtype=bool),
        )
        print(
            "  trajopt_success_only=true -> filtered validation episodes: "
            f"{len(val_episode_indices)} / {before_trajopt_filter}"
        )

    if args.random_sample_episodes is not None:
        if int(args.random_sample_episodes) <= 0:
            raise ValueError(
                f"--random-sample-episodes must be positive, got {args.random_sample_episodes}"
            )
        before_random_sample = len(val_episode_indices)
        sample_count = min(int(args.random_sample_episodes), before_random_sample)
        rng = np.random.default_rng(int(args.random_sample_seed))
        if sample_count < before_random_sample:
            selected_positions = np.sort(
                rng.choice(before_random_sample, size=sample_count, replace=False)
            )
            val_episode_indices = np.asarray(val_episode_indices[selected_positions], dtype=np.int64)
        print(
            "  random_sample_episodes=true -> sampled validation episodes: "
            f"{len(val_episode_indices)} / {before_random_sample} "
            f"(seed={int(args.random_sample_seed)})"
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

    if len(val_episode_indices) == 0:
        raise ValueError(
            "No validation episodes remain after applying the requested filters. "
            "Check --trajopt-success-key / --trajopt-success-results-dir / "
            "--regular-jobs-only / --max-episodes."
        )

    per_traj_metrics: list[dict] = []
    collision_count = 0
    total_segment_collision_steps = 0
    total_segment_steps = 0
    sdf_distances = []
    goal_errors = []
    status_display = _ValidationStatusDisplay()

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
            if not getattr(main, "_printed_device_report", False):
                _print_runtime_device_report(
                    prefix="runtime-device",
                    requested_device=device,
                    policy=policy,
                    obs_dict=obs_dict,
                    planner_mode=planner_mode,
                )
                main._printed_device_report = True

            if planner_mode == "post_qp":
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
                selection["planner_mode"] = "post_qp"
                selection["planning_success"] = True
            elif planner_mode == "qp_guided_diffusion":
                selection = _predict_late_stage_qp_guided_diffusion(
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
            elif planner_mode == "qp_guided_diffusion_post_qp":
                selection = _predict_qp_guided_diffusion_then_post_qp(
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
                selection["planner_mode"] = "baseline"
                selection["planning_success"] = True
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

            planning_success = bool(selection.get("planning_success", True))
            if not planning_success or joint_trajectory.size == 0:
                failed_segment_steps = int(pyb_cfg.target_steps)
                metric = {
                    "has_collision": True,
                    "segment_collision_steps": failed_segment_steps,
                    "segment_steps": failed_segment_steps,
                    "collision_step_count": failed_segment_steps,
                    "collision_point_count": 0,
                    "min_sdf_distance_m": float("nan"),
                    "goal_error_m": float("nan"),
                    "goal_reached": False,
                    "success": False,
                    "pybullet_pass": False,
                    "planning_failure_imputed_collision": True,
                }
            else:
                metric = validator.evaluate_trajectory(
                    workpiece_id=wid,
                    joint_trajectory=joint_trajectory,
                    start_joint_state=validator._unnormalize_joint_state(
                        raw_obs["first_joint_angles_normalized"][0]
                    ),
                    goal_position_normalized=raw_obs["goal_position"][0],
                    episode_idx=int(ep_idx),
                )
                metric["pybullet_pass"] = not bool(metric.get("has_collision", False))
                metric["planning_failure_imputed_collision"] = False
            start_joint_state = validator._unnormalize_joint_state(
                raw_obs["first_joint_angles_normalized"][0]
            )
            goal_joint_state = validator._unnormalize_joint_state(
                raw_obs["last_joint_angles_normalized"][0]
            )
            singularity_summary = (
                _compute_episode_singularity_summary(
                    validator=validator,
                    start_joint_state=start_joint_state,
                    goal_joint_state=goal_joint_state,
                    joint_trajectory=joint_trajectory,
                )
                if joint_trajectory.size > 0
                else {}
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
                "planner_mode": str(selection.get("planner_mode", planner_mode)),
                "planning_success": bool(selection.get("planning_success", True)),
                "pybullet_pass": bool(metric.get("pybullet_pass", False)),
                "planning_failure_imputed_collision": bool(
                    metric.get("planning_failure_imputed_collision", False)
                ),
                "candidate_pool_enabled": bool(selection["candidate_pool_enabled"]),
                "surface_cbf_qp_guidance_enabled": bool(selection.get("surface_cbf_qp_guidance_enabled", False)),
                "late_stage_qp_guided_diffusion_enabled": bool(selection.get("late_stage_qp_guided_diffusion_enabled", False)),
                "num_candidates": int(selection["num_candidates"]),
                "selected_score": selection["selected_score_details"],
                "inference_elapsed_sec": selection["inference_elapsed_sec"],
                "guidance_log": selection.get("guidance_log"),
                "guidance_candidates": selection.get("guidance_candidates", []),
                "guidance_candidate_count": selection.get("guidance_candidate_count"),
                "guidance_repair_attempt_count": int(selection.get("guidance_repair_attempt_count", 0) or 0),
                "guidance_repair_attempted_indices": list(selection.get("guidance_repair_attempted_indices", []) or []),
                "final_success_source": str(selection.get("final_success_source", "failure") or "failure"),
                "singularity": singularity_summary,
            }
            guidance_log = dict(selection.get("guidance_log", {}) or {})
            for key in (
                "num_candidates_guided",
                "repairability_score",
                "qp_status",
                "qp_slack_sum",
                "qp_delta_norm",
                "guidance_steps_applied",
                "min_clearance",
                "min_sdf",
                "num_penetration",
                "max_penetration_depth",
                "certified_before_waypoint_qp",
                "recovered_by_waypoint_qp",
                "selected_candidate_index",
                "diffusion_time",
                "guided_qp_time",
                "certification_time",
                "waypoint_fallback_time",
                "total_planning_time",
            ):
                if key in guidance_log:
                    traj_entry[key] = guidance_log[key]
            traj_entry.update(_summarize_qp_status_from_selection(selection))
            per_traj_metrics.append(traj_entry)

            if has_collision:
                collision_count += 1
            total_segment_collision_steps += collision_steps
            total_segment_steps += total_steps
            if not np.isnan(metric["min_sdf_distance_m"]):
                sdf_distances.append(float(metric["min_sdf_distance_m"]))
            goal_errors.append(float(metric["goal_error_m"]))

            status_display.render(
                _build_validation_status_lines(
                    index=idx + 1,
                    total=len(val_episode_indices),
                    episode_idx=int(ep_idx),
                    workpiece_id=wid,
                    selection=selection,
                    metric=metric,
                    collision_count=collision_count,
                    total_segment_collision_steps=total_segment_collision_steps,
                    total_segment_steps=total_segment_steps,
                )
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
    planning_failure_count = sum(1 for t in per_traj_metrics if not t["planning_success"])
    planning_failure_imputed_collision_count = sum(
        1 for t in per_traj_metrics if t["planning_failure_imputed_collision"]
    )
    mean_goal_error = float(np.mean(goal_errors)) if goal_errors else float("nan")
    singularity_group_summary = _build_singularity_group_summary(per_traj_metrics)

    output_json = output_dir / "per_trajectory_metrics.json"
    summary = {
        "config": {
            "checkpoint_path": str(checkpoint_path),
            "zarr_path": str(zarr_path),
            "stats_path": str(stats_path),
            "num_control_points": args.num_control_points,
            "val_ratio": effective_val_ratio,
            "val_ratio_source": "checkpoint_config" if args.val_ratio is None else "cli_override",
            "horizon": horizon,
            "n_obs_steps": n_obs_steps,
            "candidate_pool": str(args.candidate_pool),
            "planner_mode": str(planner_mode),
            "surface_cbf_qp_guidance": bool(args.enable_surface_cbf_qp_guidance),
            "late_stage_qp_guided_diffusion": bool(qp_guided_diffusion_enabled),
            "effective_num_candidates": int(effective_num_candidates),
            "candidate_selection": str(pyb_cfg.candidate_selection),
            "inference_num_steps": pyb_cfg.inference_num_steps,
            "measure_inference_time": bool(args.measure_inference_time),
            "random_sample_episodes": args.random_sample_episodes,
            "random_sample_seed": int(args.random_sample_seed),
            "regular_jobs_only": bool(args.regular_jobs_only),
            "trajopt_success_only": bool(args.trajopt_success_only),
            "trajopt_success_key": args.trajopt_success_key,
            "trajopt_success_results_dir": list(args.trajopt_success_results_dir or []),
            "trajopt_success_match_tol": float(args.trajopt_success_match_tol),
            "trajopt_success_source": trajopt_success_source,
            "simple_workpiece_id_offset": int(pyb_cfg.simple_workpiece_id_offset),
            "single_episode_mode": args.single_episode_index is not None,
            "single_episode_validation_offset": args.single_episode_index,
            "guidance_steps": int(args.guidance_steps),
            "guidance_timesteps": [int(v) for v in (args.guidance_timesteps or [])],
            "qp_candidates": int(args.qp_candidates),
            "qp_inner_scp_rounds": int(args.qp_inner_scp_rounds),
            "coarse_check_steps": int(args.coarse_check_steps),
            "guidance_pen_link_points": int(args.guidance_pen_link_points),
            "guidance_wrist3_points": int(args.guidance_wrist3_points),
            "final_post_qp_candidates": int(args.final_post_qp_candidates),
            "final_backup_candidates": int(args.final_backup_candidates),
            "final_post_qp_rounds": int(args.final_post_qp_rounds),
            "guidance_trigger_distance": float(args.guidance_trigger_distance),
            "guidance_safe_distance": float(args.guidance_safe_distance),
            "trust_region_start": float(args.trust_region_start),
            "trust_region_end": float(args.trust_region_end),
            "blend_weights": [float(v) for v in args.blend_weights],
            "repair_score_weights": [float(v) for v in args.repair_score_weights],
            "guidance_max_risk_segments": int(args.guidance_max_risk_segments),
            "guidance_window_radius": int(args.guidance_window_radius),
            "guidance_points_per_segment": int(args.guidance_points_per_segment),
            "guidance_min_constraints_per_segment": int(args.guidance_min_constraints_per_segment),
            "guidance_active_constraints": int(args.guidance_active_constraints),
            "guidance_check_steps": int(args.guidance_check_steps),
            "guidance_cert_steps": int(args.guidance_cert_steps),
            "guidance_cert_swept_intermediate": int(args.guidance_cert_swept_intermediate),
            "guidance_d_safe": float(args.guidance_d_safe),
            "guidance_d_trigger": float(args.guidance_d_trigger),
            "guidance_d_cert": float(args.guidance_d_cert),
            "guidance_eps_deep": float(args.guidance_eps_deep),
            "guidance_delta_max": float(args.guidance_delta_max),
            "guidance_scp_iterations": int(args.guidance_scp_iterations),
            "guidance_delta_max_total": float(args.guidance_delta_max_total),
            "guidance_delta_max_pass1": float(args.guidance_delta_max_pass1),
            "guidance_delta_max_pass2": float(args.guidance_delta_max_pass2),
            "guidance_d_trigger_pass2_offset": float(args.guidance_d_trigger_pass2_offset),
            "guidance_margin_buffer": float(args.guidance_margin_buffer),
            "enable_local_waypoint_qp_after_certificate": bool(args.enable_local_waypoint_qp_after_certificate),
            "local_waypoint_qp_window_radius": int(args.local_waypoint_qp_window_radius),
            "local_waypoint_qp_max_collision_segments": int(args.local_waypoint_qp_max_collision_segments),
            "local_waypoint_qp_min_clearance_trigger": float(args.local_waypoint_qp_min_clearance_trigger),
            "local_waypoint_qp_target_buffer": float(args.local_waypoint_qp_target_buffer),
            "local_waypoint_qp_lambda_s": float(args.local_waypoint_qp_lambda_s),
            "local_waypoint_qp_delta_max": float(args.local_waypoint_qp_delta_max),
            "local_waypoint_qp_max_velocity_step": float(args.local_waypoint_qp_max_velocity_step),
            "local_waypoint_qp_max_acceleration_step": float(args.local_waypoint_qp_max_acceleration_step),
            "local_waypoint_qp_maxiter": int(args.local_waypoint_qp_maxiter),
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
            "planning_failure_count": int(planning_failure_count),
            "planning_failure_rate": planning_failure_count / total if total > 0 else 0.0,
            "planning_failure_imputed_collision_count": int(planning_failure_imputed_collision_count),
            "mean_min_sdf_distance_m": float(mean_min_sdf),
            "sdf_valid_rate": float(sdf_valid_rate),
            "goal_reached_count": int(goal_reached_count),
            "goal_reached_rate": goal_reached_count / total if total > 0 else 0.0,
            "mean_goal_error_m": float(mean_goal_error),
            "singularity_groups": singularity_group_summary,
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
    if planning_failure_count > 0:
        print(
            f"  Planning failures:                 {int(planning_failure_count)} "
            f"({planning_failure_count / total * 100:.1f}%, imputed as full collision segments)"
        )
    print(f"  Mean min SDF distance:            {mean_min_sdf:.4f} m")
    print(f"  SDF valid rate:                   {sdf_valid_rate * 100:.1f}%")
    print(
        f"  Goal reached:                     {goal_reached_count} "
        f"({goal_reached_count / total * 100:.1f}%)"
    )
    print(f"  Mean goal error:                  {mean_goal_error:.4f} m")
    print("  Singularity groups:")
    for group_name, group_metrics in singularity_group_summary.items():
        print(
            f"    {group_name}: count={group_metrics['count']} "
            f"start_sigma_min={group_metrics['start_sigma_min_mean']:.6f} "
            f"goal_sigma_min={group_metrics['goal_sigma_min_mean']:.6f} "
            f"traj_sigma_min_min={group_metrics['trajectory_sigma_min_min_mean']:.6f}"
        )
    print("=" * 56)


if __name__ == "__main__":
    main()
