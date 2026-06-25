#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train import TrainDP3Workspace
from diffusion_policy_3d.common.bspline import (
    _resolve_free_control_point_slice,
    fit_quintic_bspline_to_npz_trajectory,
    load_delta_w_stats,
    reconstruct_trajectory_from_normalized_free_residual,
    unnormalize_joint_trajectory_with_urdf_limits,
)
from diffusion_policy_3d.common.input_data import load_bspline_planning_input_data
from infer_bspline_trajectory import build_obs_dict, ensure_dir, save_joint_plot
from guidance_config import (
    add_surface_cbf_qp_guidance_parser_args,
    apply_surface_cbf_qp_guidance_config,
    build_surface_cbf_qp_parameter_summary,
)


def qp_guided_surface_points_per_link(args) -> dict[str, int]:
    return {
        "pen_link": int(args.guidance_pen_link_points),
        "wrist_3_link": int(args.guidance_wrist3_points),
    }


def _format_qp_skip_reason(reason: str | None) -> str:
    reason_map = {
        None: "unknown",
        "surface_cbf_qp_guidance_disabled": "surface CBF-QP guidance disabled",
        "compare_mode_disables_surface_cbf_qp_guidance": "compare mode disables surface CBF-QP guidance",
        "candidate_mode_does_not_use_surface_cbf_qp_guidance": "candidate mode does not use surface CBF-QP guidance",
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


def _resample_joint_trajectory_to_steps(joint_trajectory, num_steps: int):
    joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
    if joint_trajectory.ndim != 2:
        raise ValueError(f"joint_trajectory must be rank-2 [T, J], got {joint_trajectory.shape}")
    if joint_trajectory.shape[0] == int(num_steps):
        return joint_trajectory.astype(np.float32)
    if joint_trajectory.shape[0] <= 1:
        return np.repeat(joint_trajectory.astype(np.float32), int(num_steps), axis=0)
    source_axis = np.linspace(0.0, 1.0, joint_trajectory.shape[0], dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float64)
    return np.stack(
        [np.interp(target_axis, source_axis, joint_trajectory[:, joint_index]) for joint_index in range(joint_trajectory.shape[1])],
        axis=1,
    ).astype(np.float32)


def summarize_qp_status(
    *,
    mode: str,
    compare_mode: bool,
    guidance_enabled: bool,
    guidance_payload: dict | None,
) -> dict[str, object]:
    guidance_log = {} if guidance_payload is None else dict(guidance_payload.get("guidance_log", {}) or {})
    guidance_candidates = [] if guidance_payload is None else list(guidance_payload.get("guidance_candidates", []) or [])
    selected_candidate_idx = int(guidance_log.get("best_candidate_index", 0) or 0)
    selected_candidate_info = None
    for candidate_info in guidance_candidates:
        if int(candidate_info.get("candidate_index", -1)) == selected_candidate_idx:
            selected_candidate_info = candidate_info
            break

    attempted_count = sum(bool(candidate_info.get("qp_attempted", False)) for candidate_info in guidance_candidates)
    success_count = sum(bool(candidate_info.get("qp_success", False)) for candidate_info in guidance_candidates)
    qp_attempted = attempted_count > 0
    if mode != "baseline":
        qp_skip_reason = "candidate_mode_does_not_use_surface_cbf_qp_guidance"
    elif compare_mode:
        qp_skip_reason = "compare_mode_disables_surface_cbf_qp_guidance"
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
        "guidance_candidate_count": int(len(guidance_candidates)),
    }


def print_inference_progress(
    *,
    sample_index: int,
    total_samples: int,
    mode: str,
    npz_path: pathlib.Path,
    summary: dict,
) -> None:
    scp_part = (
        f"SCP={summary['guidance_selected_candidate_passes_succeeded']}/"
        f"{summary['guidance_selected_candidate_pass_count']}"
    )
    qp_part = (
        f"{scp_part} QP=yes passes={summary['guidance_num_qp_success']}/{summary['guidance_num_qp_called']}"
        if summary["qp_attempted"]
        else (
            f"{scp_part} QP=no reason={summary['qp_skip_reason_text']} "
            f"finite_sdf={summary['selected_candidate_finite_sdf_value_count']}/"
            f"{summary['selected_candidate_sdf_value_count']} "
            f"finite_timesteps={summary['selected_candidate_finite_sdf_timestep_count']}"
        )
    )
    min_sdf_value = summary.get("min_sdf_distance_m")
    min_sdf_text = "n/a" if min_sdf_value is None else f"{float(min_sdf_value):.6f}"
    print(
        f"[{sample_index}/{total_samples}] mode={mode} npz={npz_path.name} "
        f"selected_candidate={int(summary.get('selected_candidate_index', 0))} "
        f"min_sdf={min_sdf_text} final={summary.get('guidance_final_success_source', 'failure')} {qp_part}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch inference for B-spline diffusion policy over transition NPZ files. "
            "Samples trajectories, reconstructs predictions, and optionally compares "
            "baseline inference against multi-candidate inference on the same samples."
        )
    )
    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories to scan recursively for transition_*.npz files.",
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to a trained checkpoint (.ckpt).")
    parser.add_argument(
        "--stats-path",
        type=str,
        required=True,
        help="Path to the B-spline delta_w statistics (.npz) used during training.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory for per-trajectory inference outputs.",
    )
    parser.add_argument("--jobs-root", type=str, default=None, help="Root directory for regular job STL/SDF files.")
    parser.add_argument("--simple-jobs-root", type=str, default=None, help="Root directory for simple job STL/SDF files.")
    parser.add_argument("--fallback-stl-path", type=str, default=None, help="Fallback STL path when job matching fails.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--norm-m", type=float, default=0.1)
    parser.add_argument("--radius-m", type=float, default=0.1)
    parser.add_argument("--height-m", type=float, default=0.1)
    parser.add_argument("--num-output-points", type=int, default=512)
    parser.add_argument("--num-mesh-sample-points", type=int, default=100000)
    parser.add_argument("--stl-x-offset-mm", type=float, default=500.0)
    parser.add_argument("--urdf-path", type=str, default=None)
    parser.add_argument("--trajectory-key", type=str, default="q_plan")
    parser.add_argument("--target-steps", type=int, default=64)
    parser.add_argument("--num-control-points", type=int, default=16)
    parser.add_argument("--spline-degree", type=int, default=5)
    parser.add_argument("--use-poisson-disk", action="store_true")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on raw discovered NPZ files before source filtering and random sampling.",
    )
    parser.add_argument(
        "--sample-source",
        type=str,
        choices=["regular", "simple", "all"],
        default="regular",
        help="Which source pool to sample trajectories from.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of trajectories to randomly sample for inference.",
    )
    parser.add_argument("--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic random trajectory sampling.",
    )
    parser.add_argument(
        "--val-split",
        action="store_true",
        help="Filter NPZ files to only those belonging to validation workpiece IDs "
             "(uses the same split logic as training). Requires --zarr-path.",
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default=None,
        help="Path to zarr dataset for resolving validation workpiece IDs (required with --val-split).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for workpiece split when --val-split is enabled (default: 0.1, matching training config).",
    )
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=42,
        help="Seed for deterministic validation split (default: 42).",
    )
    parser.add_argument(
        "--no-split-by-workpiece",
        action="store_true",
        help="Disable split-by-workpiece when --val-split is enabled (default: split by workpiece).",
    )
    parser.add_argument(
        "--no-stratify-workpiece-split",
        action="store_true",
        help="Disable stratified split across regular/simple workpiece IDs (default: stratified, matching training config).",
    )
    parser.add_argument(
        "--workpiece-split-strategy",
        type=str,
        choices=["random", "tail"],
        default="tail",
        help="Strategy for selecting validation workpiece IDs: random or tail (default: tail, matching training config).",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["baseline", "candidate", "compare"],
        default="baseline",
        help="baseline=single prediction, candidate=best from candidate pool, compare=run both on same sampled trajectories.",
    )
    parser.add_argument(
        "--enable-candidate-pool",
        action="store_true",
        help="Shortcut that promotes baseline mode to candidate mode.",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=42,
        help="Base seed for deterministic candidate sampling.",
    )
    parser.add_argument(
        "--candidate-scheduler-eta",
        type=float,
        default=1.0,
        help="Optional DDIM eta passed into scheduler_step_kwargs during candidate sampling.",
    )
    parser.add_argument(
        "--candidate-action-noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise added to candidate action horizons after the first candidate.",
    )
    parser.add_argument(
        "--candidate-action-noise-clip",
        type=float,
        default=None,
        help="Optional clip bound applied to candidate action noise.",
    )
    parser.add_argument(
        "--candidate-selection",
        type=str,
        choices=["weighted_sdf", "first"],
        default="weighted_sdf",
        help="How to choose the final trajectory from the candidate pool.",
    )
    parser.add_argument(
        "--simple-workpiece-id-offset",
        type=int,
        default=1000,
        help="Offset applied when mapping simple job IDs into workpiece IDs for candidate scoring.",
    )
    parser.add_argument(
        "--cspace-feature-dir",
        type=str,
        default=None,
        help="Directory containing C-space inference features for C-space checkpoints.",
    )
    parser.add_argument(
        "--cspace-feature-filename",
        type=str,
        default="workpiece_key_config_features.npy",
        help="Filename of the C-space feature array inside --cspace-feature-dir.",
    )
    parser.add_argument(
        "--cspace-workpiece-ids-filename",
        type=str,
        default="workpiece_ids.npy",
        help="Filename of the workpiece ID array aligned with the C-space features.",
    )
    add_surface_cbf_qp_guidance_parser_args(
        parser,
        include_num_candidates=True,
        include_candidate_inference_steps=True,
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose summary.json already exists.")
    return parser


def infer_source_kind(npz_path: pathlib.Path, input_dirs: list[pathlib.Path]) -> str:
    npz_parts = set(npz_path.parts)
    for input_dir in input_dirs:
        name = input_dir.name.lower()
        if str(npz_path).startswith(str(input_dir.resolve())):
            if "simple" in name:
                return "simple"
            return "regular"
    if "simple_results" in npz_parts or "simple" in str(npz_path).lower():
        return "simple"
    return "regular"


def resolve_sampling_mode(args) -> str:
    if args.sampling_mode == "compare":
        return "compare"
    if args.enable_candidate_pool:
        return "candidate"
    return args.sampling_mode


def validate_args(args) -> None:
    if args.sample_count <= 0:
        raise ValueError(f"sample-count must be positive, got {args.sample_count}")
    if args.num_candidates <= 0:
        raise ValueError(f"num-candidates must be positive, got {args.num_candidates}")
    if args.candidate_action_noise_std < 0.0:
        raise ValueError(
            f"candidate-action-noise-std must be non-negative, got {args.candidate_action_noise_std}"
        )
    if args.candidate_action_noise_clip is not None and args.candidate_action_noise_clip <= 0.0:
        raise ValueError(
            f"candidate-action-noise-clip must be positive when provided, got {args.candidate_action_noise_clip}"
        )
    if args.guidance_steps <= 0:
        raise ValueError(f"guidance-steps must be positive, got {args.guidance_steps}")
    if args.qp_candidates <= 0:
        raise ValueError(f"qp-candidates must be positive, got {args.qp_candidates}")
    if args.qp_inner_scp_rounds <= 0:
        raise ValueError(f"qp-inner-scp-rounds must be positive, got {args.qp_inner_scp_rounds}")
    if args.coarse_check_steps <= 1:
        raise ValueError(f"coarse-check-steps must be greater than 1, got {args.coarse_check_steps}")
    if args.trust_region_start <= 0.0 or args.trust_region_end <= 0.0:
        raise ValueError("trust-region-start/end must be positive")
    if args.trust_region_start > args.trust_region_end:
        raise ValueError("trust-region-start must be <= trust-region-end")
    if len(args.blend_weights) != args.guidance_steps:
        raise ValueError(
            "blend-weights length must match guidance-steps, "
            f"got {len(args.blend_weights)} vs {args.guidance_steps}"
        )
    if len(args.repair_score_weights) != 3:
        raise ValueError("repair-score-weights must contain exactly 3 values")
    if args.guidance_max_risk_segments <= 0:
        raise ValueError(
            "guidance-max-risk-segments must be positive, "
            f"got {args.guidance_max_risk_segments}"
        )
    if args.guidance_window_radius < 0:
        raise ValueError(
            f"guidance-window-radius must be non-negative, got {args.guidance_window_radius}"
        )
    if args.guidance_points_per_segment <= 0:
        raise ValueError(
            "guidance-points-per-segment must be positive, "
            f"got {args.guidance_points_per_segment}"
        )
    if args.guidance_min_constraints_per_segment <= 0:
        raise ValueError(
            "guidance-min-constraints-per-segment must be positive, "
            f"got {args.guidance_min_constraints_per_segment}"
        )
    if args.guidance_active_constraints <= 0:
        raise ValueError(
            "guidance-active-constraints must be positive, "
            f"got {args.guidance_active_constraints}"
        )
    if args.guidance_scp_iterations != 2:
        raise ValueError(
            f"guidance-scp-iterations must be 2 for the current SCP implementation, got {args.guidance_scp_iterations}"
        )
    if args.guidance_delta_max_total <= 0.0:
        raise ValueError(
            f"guidance-delta-max-total must be positive, got {args.guidance_delta_max_total}"
        )
    if args.guidance_delta_max_pass1 <= 0.0:
        raise ValueError(
            f"guidance-delta-max-pass1 must be positive, got {args.guidance_delta_max_pass1}"
        )
    if args.guidance_delta_max_pass2 <= 0.0:
        raise ValueError(
            f"guidance-delta-max-pass2 must be positive, got {args.guidance_delta_max_pass2}"
        )
    if args.guidance_d_trigger_pass2_offset < 0.0:
        raise ValueError(
            "guidance-d-trigger-pass2-offset must be non-negative, "
            f"got {args.guidance_d_trigger_pass2_offset}"
        )
    if args.guidance_margin_buffer < 0.0:
        raise ValueError(
            f"guidance-margin-buffer must be non-negative, got {args.guidance_margin_buffer}"
        )
    if args.local_waypoint_qp_window_radius < 0:
        raise ValueError(
            f"local-waypoint-qp-window-radius must be non-negative, got {args.local_waypoint_qp_window_radius}"
        )
    if args.local_waypoint_qp_max_collision_segments <= 0:
        raise ValueError(
            "local-waypoint-qp-max-collision-segments must be positive, "
            f"got {args.local_waypoint_qp_max_collision_segments}"
        )
    if args.local_waypoint_qp_target_buffer < 0.0:
        raise ValueError(
            f"local-waypoint-qp-target-buffer must be non-negative, got {args.local_waypoint_qp_target_buffer}"
        )
    if args.local_waypoint_qp_delta_max <= 0.0:
        raise ValueError(
            f"local-waypoint-qp-delta-max must be positive, got {args.local_waypoint_qp_delta_max}"
        )
    if args.local_waypoint_qp_max_velocity_step <= 0.0:
        raise ValueError(
            "local-waypoint-qp-max-velocity-step must be positive, "
            f"got {args.local_waypoint_qp_max_velocity_step}"
        )
    if args.local_waypoint_qp_max_acceleration_step <= 0.0:
        raise ValueError(
            "local-waypoint-qp-max-acceleration-step must be positive, "
            f"got {args.local_waypoint_qp_max_acceleration_step}"
        )
    if args.local_waypoint_qp_maxiter <= 0:
        raise ValueError(
            f"local-waypoint-qp-maxiter must be positive, got {args.local_waypoint_qp_maxiter}"
        )
    if resolve_sampling_mode(args) in {"candidate", "compare"} and args.candidate_selection == "weighted_sdf":
        if args.jobs_root is None:
            raise ValueError("--jobs-root is required when candidate-selection=weighted_sdf")



def resolve_job_name_from_npz(npz_path: pathlib.Path) -> str | None:
    for parent in npz_path.parents:
        if parent.name.startswith("job_"):
            return parent.name
    return None


def resolve_workpiece_id_from_npz(
    npz_path: pathlib.Path,
    input_dirs: list[pathlib.Path],
    simple_workpiece_id_offset: int,
) -> int:
    job_name = resolve_job_name_from_npz(npz_path)
    if job_name is None:
        raise ValueError(f"Unable to resolve job name from NPZ path: {npz_path}")
    suffix = job_name.removeprefix("job_")
    try:
        workpiece_id = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Job name does not contain an integer workpiece ID: {job_name}") from exc
    if infer_source_kind(npz_path=npz_path, input_dirs=input_dirs) == "simple":
        return int(simple_workpiece_id_offset) + workpiece_id
    return workpiece_id


def policy_requires_cspace_feature(policy) -> bool:
    cspace_feature_key = getattr(policy, "cspace_feature_key", None)
    return isinstance(cspace_feature_key, str) and len(cspace_feature_key) > 0


class CSpaceFeatureProvider:
    def __init__(
        self,
        feature_dir: pathlib.Path,
        features: np.ndarray,
        workpiece_ids: np.ndarray,
    ):
        features = np.asarray(features, dtype=np.float32)
        workpiece_ids = np.asarray(workpiece_ids, dtype=np.int64)
        if features.ndim != 3:
            raise ValueError(f"C-space features must be rank-3 [N, 128, C], got shape {features.shape}")
        if workpiece_ids.ndim != 1:
            raise ValueError(f"C-space workpiece IDs must be rank-1, got shape {workpiece_ids.shape}")
        if workpiece_ids.shape[0] != features.shape[0]:
            raise ValueError(
                "C-space workpiece IDs must align with features, got "
                f"{workpiece_ids.shape[0]} IDs for {features.shape[0]} feature rows."
            )
        unique_ids, counts = np.unique(workpiece_ids, return_counts=True)
        duplicate_ids = unique_ids[counts > 1]
        if duplicate_ids.size > 0:
            raise ValueError(
                "C-space workpiece IDs must be unique; duplicates: "
                f"{duplicate_ids.tolist()}"
            )
        self.feature_dir = pathlib.Path(feature_dir)
        self.features = np.ascontiguousarray(features)
        self.workpiece_ids = np.ascontiguousarray(workpiece_ids)
        self.row_by_workpiece_id = {
            int(workpiece_id): int(row_index)
            for row_index, workpiece_id in enumerate(self.workpiece_ids.tolist())
        }

    @classmethod
    def from_files(
        cls,
        feature_dir: str,
        feature_filename: str,
        workpiece_ids_filename: str,
    ) -> "CSpaceFeatureProvider":
        resolved_feature_dir = pathlib.Path(feature_dir).expanduser().resolve()
        feature_path = resolved_feature_dir / feature_filename
        workpiece_ids_path = resolved_feature_dir / workpiece_ids_filename
        missing_paths = [str(path) for path in (feature_path, workpiece_ids_path) if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing C-space feature artifacts: {missing_paths}")
        return cls(
            feature_dir=resolved_feature_dir,
            features=np.load(feature_path),
            workpiece_ids=np.load(workpiece_ids_path),
        )

    def get_feature(self, workpiece_id: int) -> np.ndarray:
        workpiece_id = int(workpiece_id)
        if workpiece_id not in self.row_by_workpiece_id:
            raise KeyError(
                f"C-space feature is missing for workpiece_id={workpiece_id} in {self.feature_dir}."
            )
        feature_row = self.row_by_workpiece_id[workpiece_id]
        return np.asarray(self.features[feature_row], dtype=np.float32)


def build_cspace_feature_provider(args, policy) -> CSpaceFeatureProvider | None:
    if not policy_requires_cspace_feature(policy):
        return None
    if args.cspace_feature_dir is None:
        raise ValueError(
            "This checkpoint requires C-space features. Please provide --cspace-feature-dir "
            "(and optionally --cspace-feature-filename / --cspace-workpiece-ids-filename)."
        )
    return CSpaceFeatureProvider.from_files(
        feature_dir=args.cspace_feature_dir,
        feature_filename=args.cspace_feature_filename,
        workpiece_ids_filename=args.cspace_workpiece_ids_filename,
    )


def inject_cspace_feature(
    *,
    obs_dict: dict,
    raw_obs: dict,
    cspace_feature: np.ndarray,
    n_obs_steps: int,
    device: torch.device,
) -> None:
    cspace_feature = np.asarray(cspace_feature, dtype=np.float32)
    raw_obs["cspace_feature"] = cspace_feature.copy()
    obs_value = np.expand_dims(cspace_feature, axis=0)
    obs_dict["cspace_feature"] = torch.from_numpy(obs_value).to(device)


def prepare_obs_inputs(
    *,
    npz_path: pathlib.Path,
    stl_path: pathlib.Path,
    input_dirs: list[pathlib.Path],
    policy,
    workspace: TrainDP3Workspace,
    device: torch.device,
    args,
    cspace_feature_provider: CSpaceFeatureProvider | None,
) -> tuple[dict, dict, int | None]:
    obs_dict, raw_obs = build_obs_dict(
        stl_path=str(stl_path),
        npz_path=str(npz_path),
        norm_m=args.norm_m,
        radius_m=args.radius_m,
        height_m=args.height_m,
        num_output_points=args.num_output_points,
        num_mesh_sample_points=args.num_mesh_sample_points,
        stl_x_offset_mm=args.stl_x_offset_mm,
        urdf_path=args.urdf_path,
        use_poisson_disk=args.use_poisson_disk,
        n_obs_steps=workspace.cfg.n_obs_steps,
        device=device,
    )

    workpiece_id = None
    if policy_requires_cspace_feature(policy):
        if cspace_feature_provider is None:
            raise ValueError("C-space checkpoint requires --cspace-feature-dir so cspace_feature can be injected.")
        workpiece_id = resolve_workpiece_id_from_npz(
            npz_path=npz_path,
            input_dirs=input_dirs,
            simple_workpiece_id_offset=args.simple_workpiece_id_offset,
        )
        inject_cspace_feature(
            obs_dict=obs_dict,
            raw_obs=raw_obs,
            cspace_feature=cspace_feature_provider.get_feature(workpiece_id),
            n_obs_steps=workspace.cfg.n_obs_steps,
            device=device,
        )
    return obs_dict, raw_obs, workpiece_id


def infer_jobs_dir_from_results_dir(results_dir: pathlib.Path) -> pathlib.Path:
    name = results_dir.name
    if name == "results":
        return results_dir.parent / "jobs"
    if name == "simple_results":
        return results_dir.parent / "simple_jobs"
    if name.startswith("results_"):
        return results_dir.parent / name.replace("results_", "jobs_", 1)
    if name.startswith("simple_results_"):
        return results_dir.parent / name.replace("simple_results_", "simple_jobs_", 1)
    if "results" in name:
        return results_dir.parent / name.replace("results", "jobs", 1)
    return results_dir.parent / "jobs"


def resolve_matching_stl(
    npz_path: pathlib.Path,
    input_dirs: list[pathlib.Path],
    jobs_root: str | None,
    simple_jobs_root: str | None,
    fallback_stl_path: str | None,
) -> pathlib.Path:
    job_name = resolve_job_name_from_npz(npz_path)
    source_kind = infer_source_kind(npz_path=npz_path, input_dirs=input_dirs)
    candidate_paths: list[pathlib.Path] = []

    if job_name is not None:
        if source_kind == "simple" and simple_jobs_root is not None:
            candidate_paths.append(pathlib.Path(simple_jobs_root).expanduser().resolve() / job_name / "workpiece.stl")
        elif source_kind != "simple" and jobs_root is not None:
            candidate_paths.append(pathlib.Path(jobs_root).expanduser().resolve() / job_name / "workpiece.stl")

        for parent in npz_path.parents:
            if parent.name == job_name and "results" in parent.parent.name:
                candidate_paths.append(
                    infer_jobs_dir_from_results_dir(parent.parent.resolve()) / job_name / "workpiece.stl"
                )
                break

        for candidate_path in candidate_paths:
            if candidate_path.is_file():
                return candidate_path

    if fallback_stl_path is not None:
        candidate_path = pathlib.Path(fallback_stl_path).expanduser().resolve()
        if candidate_path.is_file():
            return candidate_path
        raise FileNotFoundError(f"Fallback STL path does not exist: {candidate_path}")

    if candidate_paths:
        raise FileNotFoundError(
            f"Unable to resolve the matching STL for NPZ {npz_path}. Tried: {[str(path) for path in candidate_paths]}"
        )

    raise FileNotFoundError(f"Unable to resolve STL for NPZ {npz_path}.")


def collect_npz_files(input_dirs: list[pathlib.Path], max_files: int | None) -> list[pathlib.Path]:
    npz_files: list[pathlib.Path] = []
    for input_dir in input_dirs:
        npz_files.extend(sorted(input_dir.rglob("transition_*.npz")))
    unique_npz_files = sorted({path.resolve() for path in npz_files})
    if not unique_npz_files:
        raise FileNotFoundError(f"No transition_*.npz files found under: {[str(path) for path in input_dirs]}")
    if max_files is not None:
        unique_npz_files = unique_npz_files[:max_files]
    return unique_npz_files


def filter_npz_files_by_source(
    npz_files: list[pathlib.Path],
    input_dirs: list[pathlib.Path],
    sample_source: str,
) -> list[pathlib.Path]:
    if sample_source == "all":
        return list(npz_files)
    filtered = [
        npz_path
        for npz_path in npz_files
        if infer_source_kind(npz_path=npz_path, input_dirs=input_dirs) == sample_source
    ]
    if not filtered:
        raise FileNotFoundError(f"No {sample_source} transition_*.npz files found in the discovered inputs.")
    return filtered


def sample_npz_files(
    npz_files: list[pathlib.Path],
    sample_count: int,
    sample_seed: int,
) -> list[pathlib.Path]:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if len(npz_files) < sample_count:
        raise ValueError(f"Requested {sample_count} samples, but only found {len(npz_files)} eligible NPZ files.")
    rng = np.random.default_rng(sample_seed)
    sampled_indices = np.sort(rng.choice(len(npz_files), size=sample_count, replace=False))
    return [npz_files[int(index)] for index in sampled_indices]


def resolve_val_workpiece_ids_from_zarr(
    *,
    zarr_path: str,
    val_ratio: float,
    val_split_seed: int,
    split_by_workpiece: bool,
    stratify_workpiece_split: bool,
    workpiece_split_strategy: str,
) -> set[int]:
    """Extract validation workpiece IDs from a zarr dataset using the same split logic as training."""
    from diffusion_policy_3d.dataset.transition_dataset import TransitionTrajectoryDataset
    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

    replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)
    _, val_mask = TransitionTrajectoryDataset._resolve_ratio_split_masks(
        replay_buffer=replay_buffer,
        val_ratio=val_ratio,
        seed=val_split_seed,
        split_by_workpiece=split_by_workpiece,
        stratify_workpiece_split=stratify_workpiece_split,
        simple_workpiece_id_offset=1000,
        workpiece_split_strategy=workpiece_split_strategy,
    )
    episode_workpiece_ids = np.asarray(replay_buffer.meta["workpiece_ids"][:], dtype=np.int64)
    val_workpiece_ids = set(int(wid) for wid in np.unique(episode_workpiece_ids[val_mask]))
    print(
        f"[Val Split] zarr={zarr_path}, val_ratio={val_ratio}, seed={val_split_seed}, "
        f"split_by_workpiece={split_by_workpiece} -> {len(val_workpiece_ids)} validation workpiece IDs"
    )
    return val_workpiece_ids


def filter_npz_files_by_workpiece_ids(
    npz_files: list[pathlib.Path],
    input_dirs: list[pathlib.Path],
    val_workpiece_ids: set[int],
    simple_workpiece_id_offset: int = 1000,
) -> list[pathlib.Path]:
    """Keep only NPZ files whose associated workpiece ID is in val_workpiece_ids."""
    filtered = []
    for npz_path in npz_files:
        try:
            wid = resolve_workpiece_id_from_npz(
                npz_path=npz_path,
                input_dirs=input_dirs,
                simple_workpiece_id_offset=simple_workpiece_id_offset,
            )
            if wid in val_workpiece_ids:
                filtered.append(npz_path)
        except (ValueError, KeyError):
            continue
    print(
        f"[Val Split] Filtered {len(npz_files)} NPZ files -> {len(filtered)} matching validation workpiece IDs"
    )
    return filtered


def build_output_dir(output_root: pathlib.Path, npz_path: pathlib.Path, input_dirs: list[pathlib.Path]) -> pathlib.Path:
    for input_dir in input_dirs:
        try:
            rel = npz_path.relative_to(input_dir.resolve())
            return output_root / rel.parent / f"{npz_path.stem}_bspline_inference"
        except ValueError:
            continue
    return output_root / npz_path.parent.name / f"{npz_path.stem}_bspline_inference"


def build_summary_output_dir(base_output_dir: pathlib.Path, mode: str, compare_mode: bool) -> pathlib.Path:
    if compare_mode:
        return base_output_dir / mode
    return base_output_dir


def build_summary_path(base_output_dir: pathlib.Path, mode: str, compare_mode: bool) -> pathlib.Path:
    return build_summary_output_dir(base_output_dir=base_output_dir, mode=mode, compare_mode=compare_mode) / "summary.json"


class CandidateValidatorWrapper:
    def __init__(self, args):
        from diffusion_policy_3d.common.pybullet_validation import (  # local import to keep basic helpers importable
            PyBulletCollisionValidator,
            PyBulletValidationConfig,
            _select_lowest_candidate_score_index,
        )

        cfg = PyBulletValidationConfig(
            enabled=True,
            stats_path=str(args.stats_path),
            stats_mode="auto",
            include_regular_jobs=True,
            include_simple_jobs=True,
            jobs_root=str(args.jobs_root),
            simple_jobs_root=args.simple_jobs_root,
            simple_workpiece_id_offset=int(args.simple_workpiece_id_offset),
            urdf_path=args.urdf_path,
            stl_x_offset_m=float(args.stl_x_offset_mm) / 1000.0,
            num_control_points=int(args.num_control_points),
            spline_degree=int(args.spline_degree),
            target_steps=int(args.target_steps),
            num_candidates=int(args.num_candidates),
            candidate_scheduler_eta=args.candidate_scheduler_eta,
            candidate_action_noise_std=float(args.candidate_action_noise_std),
            candidate_action_noise_clip=args.candidate_action_noise_clip,
            candidate_selection="weighted_sdf",
            robot_surface_points_per_link=args.robot_surface_points_per_link,
        )
        self.validator = PyBulletCollisionValidator(cfg)
        self._select_lowest_candidate_score_index = _select_lowest_candidate_score_index
        self.selection_name = str(args.candidate_selection)

    def close(self) -> None:
        self.validator.close()

    def score_candidate(self, workpiece_id: int, candidate_artifact: dict) -> dict:
        return self.validator.score_candidate(
            workpiece_id=workpiece_id,
            normalized_control_points=np.asarray(candidate_artifact["pred_w_star"], dtype=np.float32),
            joint_trajectory=np.asarray(candidate_artifact["pred_joint_horizon"], dtype=np.float32),
        )

    def select_candidate_index(self, score_details: list[dict]) -> int:
        if self.selection_name == "first":
            return 0
        score_keys = np.asarray(
            [
                [
                    score_detail["has_pen"],
                    score_detail["pen_step_count"],
                    score_detail["pen_point_count"],
                    score_detail["neg_min_sdf"],
                    score_detail["neg_worstk_mean"],
                    score_detail["margin_violation"],
                ]
                for score_detail in score_details
            ],
            dtype=np.float32,
        )
        return int(self._select_lowest_candidate_score_index(score_keys))


def build_candidate_validator(args) -> CandidateValidatorWrapper | None:
    needs_guidance_validator = str(getattr(args, "planner_mode", "baseline")) in {"post_qp", "qp_guided_diffusion", "qp_guided_diffusion_post_qp"}
    needs_candidate_validator = (
        resolve_sampling_mode(args) in {"candidate", "compare"}
        and args.candidate_selection != "first"
    )
    if not (needs_guidance_validator or needs_candidate_validator):
        return None
    return CandidateValidatorWrapper(args)


def predict_action_outputs(
    policy,
    obs_dict: dict,
    *,
    generator=None,
    num_inference_steps: int | None = None,
    scheduler_step_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        result = policy.predict_action(
            obs_dict,
            generator=generator,
            num_inference_steps=num_inference_steps,
            scheduler_step_kwargs=scheduler_step_kwargs,
        )
    return {
        "pred_action_window": result["action"][0].detach().cpu().numpy().astype(np.float32),
        "pred_action_horizon": result["action_pred"][0].detach().cpu().numpy().astype(np.float32),
    }


def predict_surface_cbf_qp_guided_outputs(
    *,
    npz_path: pathlib.Path,
    obs_dict: dict,
    policy,
    args,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    candidate_validator: CandidateValidatorWrapper,
    workpiece_id: int,
    generator=None,
) -> tuple[dict[str, np.ndarray], dict]:
    from diffusion_policy_3d.common.input_data import load_bspline_planning_input_data
    from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
        PyBulletSurfaceEnvironmentAdapter,
        SurfaceCBFQPGuidanceConfig,
        SurfaceCBFQPGuidanceRunner,
    )

    planning_result = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=args.norm_m,
        urdf_path=args.urdf_path,
    )
    environment = PyBulletSurfaceEnvironmentAdapter(
        validator=candidate_validator.validator,
        workpiece_id=workpiece_id,
        joint_lower_limits=planning_result.joint_lower_limits,
        joint_upper_limits=planning_result.joint_upper_limits,
    )
    guidance_config = SurfaceCBFQPGuidanceConfig(
        enabled=True,
        num_candidates=int(args.num_candidates),
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
        joint_limit_steps=int(args.guidance_joint_limit_steps),
        guidance_targets=tuple(float(v) for v in args.guidance_targets),
        fallback_to_terminal_cbf=bool(args.guidance_fallback_to_terminal_cbf),
    )
    guidance_runner = SurfaceCBFQPGuidanceRunner(
        config=guidance_config,
        environment=environment,
    )
    with torch.no_grad():
        result = policy.sample_with_surface_cbf_qp_guidance(
            obs_dict,
            q_start_normalized=planning_result.first_joint_angles_normalized,
            q_goal_normalized=planning_result.last_joint_angles_normalized,
            delta_w_mean=stats_mean,
            delta_w_std=stats_std,
            num_control_points=int(args.num_control_points),
            spline_degree=int(args.spline_degree),
            guidance_runner=guidance_runner,
            generator=generator,
            num_inference_steps=args.candidate_inference_steps,
            scheduler_step_kwargs={"eta": float(args.guidance_ddim_eta)},
        )
    guided_joint_trajectory = result.get("guided_joint_trajectory")
    if guided_joint_trajectory is not None:
        guided_joint_trajectory = _resample_joint_trajectory_to_steps(
            guided_joint_trajectory,
            int(candidate_validator.validator.cfg.target_steps),
        )
    return {
        "pred_action_window": result["action"][0].detach().cpu().numpy().astype(np.float32),
        "pred_action_horizon": result["action_pred"][0].detach().cpu().numpy().astype(np.float32),
    }, {
        "planning_result": planning_result,
        "guidance_log": result.get("guidance_log", {}),
        "guided_joint_trajectory": guided_joint_trajectory,
        "guided_control_points_normalized": result.get("guided_control_points_normalized"),
        "guidance_candidates": result.get("guidance_candidates", []),
    }


def predict_late_stage_qp_guided_outputs(
    *,
    npz_path: pathlib.Path,
    obs_dict: dict,
    policy,
    args,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    candidate_validator: CandidateValidatorWrapper,
    workpiece_id: int,
    generator=None,
) -> tuple[dict[str, np.ndarray], dict]:
    from diffusion_policy_3d.common.input_data import load_bspline_planning_input_data
    from diffusion_policy_3d.common.late_stage_qp_guided_ddim import (
        LateStageQPGuidedDDIMConfig,
        LateStageQPGuidedDDIMRunner,
    )
    from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
        PyBulletSurfaceEnvironmentAdapter,
        SurfaceCBFQPGuidanceConfig,
    )

    planning_result = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=args.norm_m,
        urdf_path=args.urdf_path,
    )
    environment = PyBulletSurfaceEnvironmentAdapter(
        validator=candidate_validator.validator,
        workpiece_id=workpiece_id,
        joint_lower_limits=planning_result.joint_lower_limits,
        joint_upper_limits=planning_result.joint_upper_limits,
        surface_points_per_link_override=qp_guided_surface_points_per_link(args),
    )
    scp_config = SurfaceCBFQPGuidanceConfig(
        enabled=True,
        num_candidates=int(args.num_candidates),
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
        guidance_targets=tuple(float(v) for v in args.guidance_targets),
        fallback_to_terminal_cbf=False,
    )
    guidance_config = LateStageQPGuidedDDIMConfig(
        enabled=True,
        num_candidates=int(args.num_candidates),
        guidance_steps=int(args.guidance_steps),
        guidance_timesteps=tuple(int(v) for v in args.guidance_timesteps),
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
        scp_config=scp_config,
    )
    guidance_runner = LateStageQPGuidedDDIMRunner(config=guidance_config, environment=environment)
    with torch.no_grad():
        result = policy.sample_with_late_stage_qp_guided_diffusion(
            obs_dict,
            q_start_normalized=planning_result.first_joint_angles_normalized,
            q_goal_normalized=planning_result.last_joint_angles_normalized,
            delta_w_mean=stats_mean,
            delta_w_std=stats_std,
            num_control_points=int(args.num_control_points),
            spline_degree=int(args.spline_degree),
            guidance_runner=guidance_runner,
            generator=generator,
            num_inference_steps=args.candidate_inference_steps,
            scheduler_step_kwargs={"eta": float(args.guidance_ddim_eta)},
        )
    guided_joint_trajectory = result.get("guided_joint_trajectory")
    if guided_joint_trajectory is not None and np.asarray(guided_joint_trajectory).size > 0:
        guided_joint_trajectory = _resample_joint_trajectory_to_steps(
            guided_joint_trajectory,
            int(candidate_validator.validator.cfg.target_steps),
        )
    return {
        "pred_action_window": result["action"][0].detach().cpu().numpy().astype(np.float32),
        "pred_action_horizon": result["action_pred"][0].detach().cpu().numpy().astype(np.float32),
    }, {
        "planning_result": planning_result,
        "planning_success": bool(result.get("planning_success", False)),
        "guidance_log": result.get("guidance_log", {}),
        "guided_joint_trajectory": guided_joint_trajectory,
        "guided_control_points_normalized": result.get("guided_control_points_normalized"),
        "guidance_candidates": result.get("guidance_candidates", []),
    }


def select_late_stage_topk_residuals_for_post_qp(
    *,
    guidance_payload: dict,
    top_k: int,
) -> tuple[list[np.ndarray], list[int]]:
    final_candidates = list(guidance_payload.get("guidance_candidates", []) or [])
    final_by_index = {
        int(candidate_info.get("candidate_index", -1)): candidate_info
        for candidate_info in final_candidates
    }
    guidance_log = dict(guidance_payload.get("guidance_log", {}) or {})
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


def predict_qp_guided_diffusion_then_post_qp_outputs(
    *,
    npz_path: pathlib.Path,
    obs_dict: dict,
    policy,
    args,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    candidate_validator: CandidateValidatorWrapper,
    workpiece_id: int,
    generator=None,
) -> tuple[dict[str, np.ndarray], dict]:
    from diffusion_policy_3d.common.input_data import load_bspline_planning_input_data
    from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
        PyBulletSurfaceEnvironmentAdapter,
        SurfaceCBFQPGuidanceConfig,
        SurfaceCBFQPGuidanceRunner,
    )

    guided_outputs, guidance_payload = predict_late_stage_qp_guided_outputs(
        npz_path=npz_path,
        obs_dict=obs_dict,
        policy=policy,
        args=args,
        stats_mean=stats_mean,
        stats_std=stats_std,
        candidate_validator=candidate_validator,
        workpiece_id=workpiece_id,
        generator=generator,
    )
    candidate_residuals, selected_late_stage_indices = select_late_stage_topk_residuals_for_post_qp(
        guidance_payload=guidance_payload,
        top_k=max(1, int(args.final_post_qp_candidates) + int(args.final_backup_candidates)),
    )
    if not candidate_residuals:
        return guided_outputs, guidance_payload

    planning_result = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=args.norm_m,
        urdf_path=args.urdf_path,
    )
    environment = PyBulletSurfaceEnvironmentAdapter(
        validator=candidate_validator.validator,
        workpiece_id=workpiece_id,
        joint_lower_limits=planning_result.joint_lower_limits,
        joint_upper_limits=planning_result.joint_upper_limits,
        surface_points_per_link_override=qp_guided_surface_points_per_link(args),
    )
    guidance_config = SurfaceCBFQPGuidanceConfig(
        enabled=True,
        num_candidates=int(len(candidate_residuals)),
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
        guidance_targets=tuple(float(v) for v in args.guidance_targets),
        fallback_to_terminal_cbf=bool(args.guidance_fallback_to_terminal_cbf),
    )
    guidance_runner = SurfaceCBFQPGuidanceRunner(
        config=guidance_config,
        environment=environment,
    )
    post_result = guidance_runner.run(
        candidate_residuals=np.stack(candidate_residuals, axis=0).astype(np.float32),
        q_start_normalized=planning_result.first_joint_angles_normalized,
        q_goal_normalized=planning_result.last_joint_angles_normalized,
        delta_w_mean=stats_mean,
        delta_w_std=stats_std,
        num_control_points=int(args.num_control_points),
        spline_degree=int(args.spline_degree),
    )
    action_pred = (
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
    guided_joint_trajectory = _resample_joint_trajectory_to_steps(
        np.asarray(post_result.best_joint_trajectory, dtype=np.float32),
        int(candidate_validator.validator.cfg.target_steps),
    )
    post_log = post_result.log.to_dict()
    combined_payload = {
        "planning_result": planning_result,
        "planning_success": True,
        "guidance_log": {
            **post_log,
            "planner_mode": "qp_guided_diffusion_post_qp",
            "late_stage_guidance_log": dict(guidance_payload.get("guidance_log", {}) or {}),
            "late_stage_planning_success": bool(guidance_payload.get("planning_success", False)),
            "combined_post_qp_candidate_indices": [int(v) for v in selected_late_stage_indices],
            "combined_post_qp_candidate_count": int(len(candidate_residuals)),
        },
        "guided_joint_trajectory": guided_joint_trajectory,
        "guided_control_points_normalized": post_result.best_control_points_normalized,
        "guidance_candidates": post_result.candidate_infos,
    }
    return {
        "pred_action_window": guided_outputs["pred_action_window"],
        "pred_action_horizon": action_pred,
    }, combined_payload


def reconstruct_prediction_artifacts(
    npz_path: pathlib.Path,
    pred_action_window: np.ndarray,
    pred_action_horizon: np.ndarray,
    args,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
) -> dict:
    free_slice = _resolve_free_control_point_slice(args.num_control_points)
    expected_action_shape = (free_slice.stop - free_slice.start, 6)
    if pred_action_horizon.shape != expected_action_shape:
        raise ValueError(
            "Predicted normalized free control-point residual has incompatible shape. "
            f"Expected {expected_action_shape}, got {pred_action_horizon.shape}."
        )

    planning_result = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=args.norm_m,
        urdf_path=args.urdf_path,
    )
    recon_result = reconstruct_trajectory_from_normalized_free_residual(
        normalized_free_delta_w=pred_action_horizon,
        start_state=planning_result.first_joint_angles_normalized,
        end_state=planning_result.last_joint_angles_normalized,
        mean=stats_mean,
        std=stats_std,
        num_control_points=args.num_control_points,
        num_steps=args.target_steps,
        degree=args.spline_degree,
    )
    pred_joint_horizon_normalized = recon_result["fitted_trajectory"].astype(np.float32)
    pred_joint_horizon = unnormalize_joint_trajectory_with_urdf_limits(
        normalized_trajectory=pred_joint_horizon_normalized,
        lower_limits=planning_result.joint_lower_limits,
        upper_limits=planning_result.joint_upper_limits,
    )

    gt_fit_result = None
    gt_joint_traj = None
    npz_data = np.load(npz_path)
    if planning_result.trajectory_key in npz_data.files:
        gt_joint_traj = np.asarray(npz_data[planning_result.trajectory_key], dtype=np.float32)
        gt_fit_result = fit_quintic_bspline_to_npz_trajectory(
            npz_path=str(npz_path),
            trajectory_key=args.trajectory_key,
            target_steps=args.target_steps,
            urdf_path=args.urdf_path,
            num_control_points=args.num_control_points,
            degree=args.spline_degree,
        )

    return {
        "planning_result": planning_result,
        "pred_action_window": pred_action_window,
        "pred_action_horizon": pred_action_horizon,
        "pred_delta_w": recon_result["delta_w"],
        "pred_w_line": recon_result["w_line"],
        "pred_w_star": recon_result["w_star"],
        "pred_joint_horizon_normalized": pred_joint_horizon_normalized,
        "pred_joint_horizon": pred_joint_horizon,
        "gt_fit_result": gt_fit_result,
        "gt_joint_traj": gt_joint_traj,
    }


def save_prediction_artifacts(
    output_dir: pathlib.Path,
    raw_obs: dict,
    artifact: dict,
    metadata: dict,
    candidate_scores: list[dict] | None = None,
) -> dict:
    ensure_dir(output_dir)

    np.save(output_dir / "pred_action_window_normalized.npy", artifact["pred_action_window"])
    np.save(output_dir / "pred_action_horizon_normalized.npy", artifact["pred_action_horizon"])
    np.save(output_dir / "pred_delta_w.npy", artifact["pred_delta_w"])
    np.save(output_dir / "pred_w_line.npy", artifact["pred_w_line"])
    np.save(output_dir / "pred_w_star.npy", artifact["pred_w_star"])
    np.save(output_dir / "pred_joint_horizon_normalized.npy", artifact["pred_joint_horizon_normalized"])
    np.save(output_dir / "pred_joint_horizon.npy", artifact["pred_joint_horizon"])
    np.save(output_dir / "point_cloud.npy", raw_obs["point_cloud"])
    if "cspace_feature" in raw_obs:
        np.save(output_dir / "cspace_feature.npy", raw_obs["cspace_feature"])

    gt_fit_result = artifact["gt_fit_result"]
    planning_result = artifact["planning_result"]
    gt_joint_traj = artifact["gt_joint_traj"]
    if gt_fit_result is not None:
        np.save(output_dir / "gt_w_star.npy", gt_fit_result["w_star"].astype(np.float32))
        np.save(output_dir / "gt_delta_w.npy", gt_fit_result["delta_w"].astype(np.float32))
        np.save(output_dir / "gt_joint_horizon_normalized.npy", gt_fit_result["normalized_trajectory"].astype(np.float32))
        np.save(
            output_dir / "gt_joint_horizon.npy",
            unnormalize_joint_trajectory_with_urdf_limits(
                normalized_trajectory=gt_fit_result["normalized_trajectory"],
                lower_limits=planning_result.joint_lower_limits,
                upper_limits=planning_result.joint_upper_limits,
            ),
        )

    save_joint_plot(
        pred_joint_traj=artifact["pred_joint_horizon"],
        gt_joint_traj=gt_joint_traj,
        output_path=output_dir / "pred_joint_horizon.png",
    )

    summary = {
        **metadata,
        "output_dir": str(output_dir),
        "pred_action_window_shape": list(artifact["pred_action_window"].shape),
        "pred_action_horizon_shape": list(artifact["pred_action_horizon"].shape),
        "pred_joint_horizon_shape": list(artifact["pred_joint_horizon"].shape),
        "trajectory_key": planning_result.trajectory_key,
        "has_ground_truth_trajectory": bool(gt_joint_traj is not None),
        "has_cspace_feature": bool("cspace_feature" in raw_obs),
    }
    if candidate_scores is not None:
        with open(output_dir / "candidate_scores.json", "w", encoding="utf-8") as f:
            json.dump(candidate_scores, f, indent=2)
        summary["candidate_scores_path"] = str(output_dir / "candidate_scores.json")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def score_artifact_if_possible(
    candidate_validator: CandidateValidatorWrapper | None,
    workpiece_id: int | None,
    artifact: dict,
) -> dict | None:
    if candidate_validator is None or workpiece_id is None:
        return None
    return candidate_validator.score_candidate(workpiece_id=workpiece_id, candidate_artifact=artifact)


def predict_candidate_artifacts(
    npz_path: pathlib.Path,
    sample_index: int,
    obs_dict: dict,
    policy,
    device: torch.device,
    args,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
) -> list[dict]:
    scheduler_step_kwargs = {}
    if args.candidate_scheduler_eta is not None:
        scheduler_step_kwargs["eta"] = float(args.candidate_scheduler_eta)

    candidate_artifacts = []
    for candidate_idx in range(int(args.num_candidates)):
        candidate_seed = int(args.candidate_seed) + candidate_idx * 1_000_003 + int(sample_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(candidate_seed)
        predicted = predict_action_outputs(
            policy=policy,
            obs_dict=obs_dict,
            generator=generator,
            num_inference_steps=args.candidate_inference_steps,
            scheduler_step_kwargs=scheduler_step_kwargs,
        )
        if candidate_idx > 0 and args.candidate_action_noise_std > 0.0:
            rng = np.random.default_rng(candidate_seed)
            noise = rng.normal(
                loc=0.0,
                scale=float(args.candidate_action_noise_std),
                size=predicted["pred_action_horizon"].shape,
            ).astype(np.float32)
            if args.candidate_action_noise_clip is not None:
                noise = np.clip(
                    noise,
                    -float(args.candidate_action_noise_clip),
                    float(args.candidate_action_noise_clip),
                ).astype(np.float32)
            predicted["pred_action_horizon"] = (predicted["pred_action_horizon"] + noise).astype(np.float32)
        artifact = reconstruct_prediction_artifacts(
            npz_path=npz_path,
            pred_action_window=predicted["pred_action_window"],
            pred_action_horizon=predicted["pred_action_horizon"],
            args=args,
            stats_mean=stats_mean,
            stats_std=stats_std,
        )
        artifact["candidate_index"] = int(candidate_idx)
        artifact["candidate_seed"] = int(candidate_seed)
        candidate_artifacts.append(artifact)
    return candidate_artifacts


def build_candidate_score_record(candidate_artifact: dict, score_detail: dict | None) -> dict:
    record: dict[str, object] = {
        "candidate_index": int(candidate_artifact["candidate_index"]),
        "candidate_seed": int(candidate_artifact["candidate_seed"]),
    }
    if score_detail is None:
        record["selection_mode"] = "first"
        return record
    record.update({
        "selection_mode": "weighted_sdf",
        "has_pen": float(score_detail["has_pen"]),
        "pen_step_count": float(score_detail["pen_step_count"]),
        "pen_point_count": float(score_detail["pen_point_count"]),
        "num_pen": float(score_detail["num_pen"]),
        "neg_min_sdf": float(score_detail["neg_min_sdf"]),
        "neg_worstk_mean": float(score_detail["neg_worstk_mean"]),
        "margin_violation": float(score_detail["margin_violation"]),
        "min_sdf_distance_m": float(score_detail["min_sdf_distance_m"]),
        "sdf_finite_ratio": float(score_detail["sdf_finite_ratio"]),
        "penetrating_link_names": list(score_detail.get("penetrating_link_names", [])),
    })
    return record


def run_mode_inference(
    *,
    mode: str,
    npz_path: pathlib.Path,
    stl_path: pathlib.Path,
    base_output_dir: pathlib.Path,
    workspace: TrainDP3Workspace,
    policy,
    device: torch.device,
    args,
    input_dirs: list[pathlib.Path],
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    sample_index: int,
    total_samples: int,
    compare_mode: bool,
    candidate_validator: CandidateValidatorWrapper | None,
    cspace_feature_provider: CSpaceFeatureProvider | None,
) -> dict:
    obs_dict, raw_obs, prepared_workpiece_id = prepare_obs_inputs(
        npz_path=npz_path,
        stl_path=stl_path,
        input_dirs=input_dirs,
        policy=policy,
        workspace=workspace,
        device=device,
        args=args,
        cspace_feature_provider=cspace_feature_provider,
    )
    output_dir = build_summary_output_dir(base_output_dir=base_output_dir, mode=mode, compare_mode=compare_mode)
    workpiece_id = prepared_workpiece_id
    candidate_scores: list[dict] | None = None
    selected_score: dict | None = None
    guidance_payload: dict | None = None

    planner_mode = str(getattr(args, "planner_mode", "baseline"))
    if mode == "baseline":
        if planner_mode == "post_qp" and not compare_mode:
            if candidate_validator is None:
                raise ValueError(
                    "surface CBF-QP guidance requires a PyBullet candidate validator / geometry backend"
                )
            workpiece_id = resolve_workpiece_id_from_npz(
                npz_path=npz_path,
                input_dirs=input_dirs,
                simple_workpiece_id_offset=args.simple_workpiece_id_offset,
            )
            guided_outputs, guidance_payload = predict_surface_cbf_qp_guided_outputs(
                npz_path=npz_path,
                obs_dict=obs_dict,
                policy=policy,
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
                candidate_validator=candidate_validator,
                workpiece_id=workpiece_id,
            )
            artifact = reconstruct_prediction_artifacts(
                npz_path=npz_path,
                pred_action_window=guided_outputs["pred_action_window"],
                pred_action_horizon=guided_outputs["pred_action_horizon"],
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
            )
            artifact["candidate_index"] = int(guidance_payload.get("guidance_log", {}).get("best_candidate_index", 0) or 0)
            artifact["candidate_seed"] = None
            selected_score = score_artifact_if_possible(
                candidate_validator=candidate_validator,
                workpiece_id=workpiece_id,
                artifact=artifact,
            )
        elif planner_mode == "qp_guided_diffusion" and not compare_mode:
            if candidate_validator is None:
                raise ValueError(
                    "qp_guided_diffusion requires a PyBullet candidate validator / geometry backend"
                )
            workpiece_id = resolve_workpiece_id_from_npz(
                npz_path=npz_path,
                input_dirs=input_dirs,
                simple_workpiece_id_offset=args.simple_workpiece_id_offset,
            )
            guided_outputs, guidance_payload = predict_late_stage_qp_guided_outputs(
                npz_path=npz_path,
                obs_dict=obs_dict,
                policy=policy,
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
                candidate_validator=candidate_validator,
                workpiece_id=workpiece_id,
            )
            artifact = reconstruct_prediction_artifacts(
                npz_path=npz_path,
                pred_action_window=guided_outputs["pred_action_window"],
                pred_action_horizon=guided_outputs["pred_action_horizon"],
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
            )
            if guidance_payload.get("guided_joint_trajectory") is not None:
                artifact["pred_joint_horizon"] = np.asarray(
                    guidance_payload["guided_joint_trajectory"], dtype=np.float32
                )
            artifact["candidate_index"] = int(guidance_payload.get("guidance_log", {}).get("selected_candidate_index", -1) or -1)
            artifact["candidate_seed"] = None
            if bool(guidance_payload.get("planning_success", False)):
                selected_score = score_artifact_if_possible(
                    candidate_validator=candidate_validator,
                    workpiece_id=workpiece_id,
                    artifact=artifact,
                )
        elif planner_mode == "qp_guided_diffusion_post_qp" and not compare_mode:
            if candidate_validator is None:
                raise ValueError(
                    "qp_guided_diffusion_post_qp requires a PyBullet candidate validator / geometry backend"
                )
            workpiece_id = resolve_workpiece_id_from_npz(
                npz_path=npz_path,
                input_dirs=input_dirs,
                simple_workpiece_id_offset=args.simple_workpiece_id_offset,
            )
            guided_outputs, guidance_payload = predict_qp_guided_diffusion_then_post_qp_outputs(
                npz_path=npz_path,
                obs_dict=obs_dict,
                policy=policy,
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
                candidate_validator=candidate_validator,
                workpiece_id=workpiece_id,
            )
            artifact = reconstruct_prediction_artifacts(
                npz_path=npz_path,
                pred_action_window=guided_outputs["pred_action_window"],
                pred_action_horizon=guided_outputs["pred_action_horizon"],
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
            )
            if guidance_payload.get("guided_joint_trajectory") is not None:
                artifact["pred_joint_horizon"] = np.asarray(
                    guidance_payload["guided_joint_trajectory"], dtype=np.float32
                )
            artifact["candidate_index"] = int(guidance_payload.get("guidance_log", {}).get("selected_candidate_index", -1) or -1)
            artifact["candidate_seed"] = None
            if bool(guidance_payload.get("planning_success", False)):
                selected_score = score_artifact_if_possible(
                    candidate_validator=candidate_validator,
                    workpiece_id=workpiece_id,
                    artifact=artifact,
                )
            else:
                selected_score = None
        else:
            predicted = predict_action_outputs(policy=policy, obs_dict=obs_dict)
            artifact = reconstruct_prediction_artifacts(
                npz_path=npz_path,
                pred_action_window=predicted["pred_action_window"],
                pred_action_horizon=predicted["pred_action_horizon"],
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
            )
            artifact["candidate_index"] = 0
            artifact["candidate_seed"] = None
            workpiece_id = resolve_workpiece_id_from_npz(
                npz_path=npz_path,
                input_dirs=input_dirs,
                simple_workpiece_id_offset=args.simple_workpiece_id_offset,
            ) if candidate_validator is not None else None
            selected_score = score_artifact_if_possible(candidate_validator=candidate_validator, workpiece_id=workpiece_id, artifact=artifact)
    else:
        workpiece_id = resolve_workpiece_id_from_npz(
            npz_path=npz_path,
            input_dirs=input_dirs,
            simple_workpiece_id_offset=args.simple_workpiece_id_offset,
        ) if (candidate_validator is not None or args.candidate_selection == "first") else None
        candidate_artifacts = predict_candidate_artifacts(
            npz_path=npz_path,
            sample_index=sample_index,
            obs_dict=obs_dict,
            policy=policy,
            device=device,
            args=args,
            stats_mean=stats_mean,
            stats_std=stats_std,
        )
        if candidate_validator is not None:
            if workpiece_id is None:
                raise ValueError("workpiece_id is required for weighted_sdf candidate selection")
            score_details = [
                candidate_validator.score_candidate(workpiece_id=workpiece_id, candidate_artifact=candidate_artifact)
                for candidate_artifact in candidate_artifacts
            ]
            candidate_scores = [
                build_candidate_score_record(candidate_artifact, score_detail)
                for candidate_artifact, score_detail in zip(candidate_artifacts, score_details)
            ]
            selected_candidate_index = candidate_validator.select_candidate_index(score_details)
            selected_score = score_details[selected_candidate_index]
        else:
            candidate_scores = [build_candidate_score_record(candidate_artifact, None) for candidate_artifact in candidate_artifacts]
            selected_candidate_index = 0
            selected_score = None
        artifact = candidate_artifacts[selected_candidate_index]

    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "npz_path": str(npz_path),
        "stl_path": str(stl_path),
        "stats_path": str(args.stats_path),
        "mode": mode,
        "planner_mode": planner_mode,
        "sampling_mode": resolve_sampling_mode(args),
        "candidate_pool_enabled": bool(mode == "candidate"),
        "surface_cbf_qp_guidance_enabled": bool(mode == "baseline" and planner_mode in {"post_qp", "qp_guided_diffusion_post_qp"} and not compare_mode),
        "late_stage_qp_guided_diffusion_enabled": bool(mode == "baseline" and planner_mode in {"qp_guided_diffusion", "qp_guided_diffusion_post_qp"} and not compare_mode),
        "planning_success": bool(True if guidance_payload is None else guidance_payload.get("planning_success", True)),
        "sample_index": int(sample_index),
        "sample_source": str(args.sample_source),
        "sample_seed": int(args.sample_seed),
        "sample_source_kind": infer_source_kind(npz_path=npz_path, input_dirs=input_dirs),
        "workpiece_id": workpiece_id,
        "cspace_feature_dir": args.cspace_feature_dir,
        "uses_cspace_feature": bool("cspace_feature" in raw_obs),
        "n_obs_steps": int(workspace.cfg.n_obs_steps),
        "n_action_steps": int(workspace.cfg.n_action_steps),
        "policy_horizon": int(workspace.cfg.horizon),
        "target_steps": int(args.target_steps),
        "num_control_points": int(args.num_control_points),
        "spline_degree": int(args.spline_degree),
        "candidate_selection": str(args.candidate_selection if mode == "candidate" else "baseline_single"),
        "num_candidates": int(args.num_candidates if (mode == "candidate" or planner_mode in {"post_qp", "qp_guided_diffusion", "qp_guided_diffusion_post_qp"}) else 1),
        "selected_candidate_index": int(artifact.get("candidate_index", 0)),
        "selected_candidate_seed": artifact.get("candidate_seed"),
    }
    metadata["surface_cbf_qp_guidance_params"] = build_surface_cbf_qp_parameter_summary(
        args,
        include_num_candidates=True,
        include_candidate_inference_steps=True,
    )
    if guidance_payload is not None:
        metadata["surface_cbf_qp_guidance"] = guidance_payload.get("guidance_log", {})
        metadata["guidance_candidate_count"] = len(guidance_payload.get("guidance_candidates", []))
        guidance_log = dict(guidance_payload.get("guidance_log", {}) or {})
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
                metadata[key] = guidance_log[key]
    if selected_score is not None:
        metadata["selected_candidate_score_key"] = [
            float(selected_score["has_pen"]),
            float(selected_score["pen_step_count"]),
            float(selected_score["pen_point_count"]),
            float(selected_score["neg_min_sdf"]),
            float(selected_score["neg_worstk_mean"]),
            float(selected_score["margin_violation"]),
        ]
        metadata["min_sdf_distance_m"] = float(selected_score["min_sdf_distance_m"])
        metadata["has_pen"] = float(selected_score["has_pen"])
    metadata.update(
        summarize_qp_status(
            mode=mode,
            compare_mode=compare_mode,
            guidance_enabled=bool(planner_mode in {"post_qp", "qp_guided_diffusion_post_qp"}),
            guidance_payload=guidance_payload,
        )
    )
    summary = save_prediction_artifacts(
        output_dir=output_dir,
        raw_obs=raw_obs,
        artifact=artifact,
        metadata=metadata,
        candidate_scores=candidate_scores,
    )
    print_inference_progress(
        sample_index=sample_index + 1,
        total_samples=total_samples,
        mode=mode,
        npz_path=npz_path,
        summary=summary,
    )
    return summary


def build_compare_summary(npz_path: pathlib.Path, baseline_summary: dict, candidate_summary: dict) -> dict:
    baseline_min_sdf = baseline_summary.get("min_sdf_distance_m")
    candidate_min_sdf = candidate_summary.get("min_sdf_distance_m")
    min_sdf_gain = None
    if baseline_min_sdf is not None and candidate_min_sdf is not None:
        min_sdf_gain = float(candidate_min_sdf) - float(baseline_min_sdf)
    return {
        "npz_path": str(npz_path),
        "baseline_output_dir": baseline_summary["output_dir"],
        "candidate_output_dir": candidate_summary["output_dir"],
        "baseline_selected_candidate_index": baseline_summary.get("selected_candidate_index", 0),
        "candidate_selected_candidate_index": candidate_summary.get("selected_candidate_index", 0),
        "baseline_min_sdf_distance_m": baseline_min_sdf,
        "candidate_min_sdf_distance_m": candidate_min_sdf,
        "min_sdf_gain_m": min_sdf_gain,
        "sample_index": baseline_summary["sample_index"],
        "sample_seed": baseline_summary["sample_seed"],
        "candidate_seed": candidate_summary.get("selected_candidate_seed"),
    }


def main() -> None:
    inference_start_time = time.perf_counter()
    args = build_parser().parse_args()
    apply_surface_cbf_qp_guidance_config(
        args,
        include_num_candidates=True,
        include_candidate_inference_steps=True,
    )
    validate_args(args)

    checkpoint_path = pathlib.Path(args.checkpoint_path).expanduser().resolve()
    stats_path = pathlib.Path(args.stats_path).expanduser().resolve()
    output_root = ensure_dir(pathlib.Path(args.output_root).expanduser().resolve())
    input_dirs = [pathlib.Path(path).expanduser().resolve() for path in args.input_dirs]
    cspace_feature_dir = None if args.cspace_feature_dir is None else pathlib.Path(args.cspace_feature_dir).expanduser().resolve()

    args.checkpoint_path = str(checkpoint_path)
    args.stats_path = str(stats_path)
    args.output_root = str(output_root)
    args.input_dirs = [str(path) for path in input_dirs]
    args.cspace_feature_dir = None if cspace_feature_dir is None else str(cspace_feature_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"delta_w stats file not found: {stats_path}")
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

    discovered_npz_files = collect_npz_files(input_dirs=input_dirs, max_files=args.max_files)
    eligible_npz_files = filter_npz_files_by_source(
        npz_files=discovered_npz_files,
        input_dirs=input_dirs,
        sample_source=args.sample_source,
    )

    val_split_enabled = bool(args.val_split)
    if val_split_enabled:
        if args.zarr_path is None:
            raise ValueError("--zarr-path is required when --val-split is set")
        zarr_path = pathlib.Path(args.zarr_path).expanduser().resolve()
        if not zarr_path.is_dir():
            raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")
        val_workpiece_ids = resolve_val_workpiece_ids_from_zarr(
            zarr_path=str(zarr_path),
            val_ratio=args.val_ratio,
            val_split_seed=args.val_split_seed,
            split_by_workpiece=not args.no_split_by_workpiece,
            stratify_workpiece_split=not args.no_stratify_workpiece_split,
            workpiece_split_strategy=args.workpiece_split_strategy,
        )
        eligible_npz_files = filter_npz_files_by_workpiece_ids(
            npz_files=eligible_npz_files,
            input_dirs=input_dirs,
            val_workpiece_ids=val_workpiece_ids,
            simple_workpiece_id_offset=args.simple_workpiece_id_offset,
        )
        sampled_npz_files = sample_npz_files(
            npz_files=eligible_npz_files,
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
        )
    else:
        sampled_npz_files = sample_npz_files(
            npz_files=eligible_npz_files,
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
        )

    sampled_manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "sample_source": str(args.sample_source),
        "sample_count": int(len(sampled_npz_files)),
        "sample_seed": int(args.sample_seed),
        "raw_discovered_count": len(discovered_npz_files),
        "eligible_count": len(eligible_npz_files),
        "sampled_npz_paths": [str(path) for path in sampled_npz_files],
        "val_split_enabled": val_split_enabled,
        "val_ratio": float(args.val_ratio) if val_split_enabled else None,
        "val_split_seed": int(args.val_split_seed) if val_split_enabled else None,
    }
    with open(output_root / "sampled_npz_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sampled_manifest, f, indent=2)

    device = torch.device(args.device)
    workspace = TrainDP3Workspace.create_from_checkpoint(str(checkpoint_path))
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    policy = policy.to(device)
    policy.eval()
    stats_mean, stats_std = load_delta_w_stats(str(stats_path))
    effective_mode = resolve_sampling_mode(args)
    compare_mode = effective_mode == "compare"
    cspace_feature_provider = build_cspace_feature_provider(args, policy)
    candidate_validator = build_candidate_validator(args)

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "stats_path": str(stats_path),
        "output_root": str(output_root),
        "sample_source": str(args.sample_source),
        "sample_count": int(len(sampled_npz_files)),
        "sample_seed": int(args.sample_seed),
        "sampling_mode": effective_mode,
        "planner_mode": str(args.planner_mode),
        "val_split_enabled": val_split_enabled,
        "val_ratio": float(args.val_ratio) if val_split_enabled else None,
        "val_split_seed": int(args.val_split_seed) if val_split_enabled else None,
        "candidate_pool_enabled": bool(effective_mode in {"candidate", "compare"}),
        "candidate_selection": str(args.candidate_selection),
        "num_candidates": int(args.num_candidates if effective_mode in {"candidate", "compare"} or str(args.planner_mode) in {"post_qp", "qp_guided_diffusion", "qp_guided_diffusion_post_qp"} else 1),
        "candidate_seed": int(args.candidate_seed),
        "cspace_feature_dir": args.cspace_feature_dir,
        "uses_cspace_feature": bool(cspace_feature_provider is not None),
        "processed": [],
        "failed": [],
    }
    compare_summaries: list[dict] = []

    print(
        f"Discovered {len(discovered_npz_files)} NPZ files, "
        f"eligible {len(eligible_npz_files)}, sampled {len(sampled_npz_files)}."
    )
    try:
        for idx, npz_path in enumerate(sampled_npz_files, start=1):
            base_output_dir = build_output_dir(output_root=output_root, npz_path=npz_path, input_dirs=input_dirs)
            summary_path = build_summary_path(
                base_output_dir=base_output_dir,
                mode="compare" if compare_mode else effective_mode,
                compare_mode=compare_mode,
            )
            if args.skip_existing and summary_path.is_file():
                print(f"[{idx}/{len(sampled_npz_files)}] skip existing: {npz_path}")
                manifest["processed"].append({
                    "npz_path": str(npz_path),
                    "output_dir": str(base_output_dir),
                    "skipped": True,
                })
                continue

            stl_path = None
            try:
                ensure_dir(base_output_dir)
                stl_path = resolve_matching_stl(
                    npz_path=npz_path,
                    input_dirs=input_dirs,
                    jobs_root=args.jobs_root,
                    simple_jobs_root=args.simple_jobs_root,
                    fallback_stl_path=args.fallback_stl_path,
                )
                if compare_mode:
                    baseline_summary = run_mode_inference(
                        mode="baseline",
                        npz_path=npz_path,
                        stl_path=stl_path,
                        base_output_dir=base_output_dir,
                        workspace=workspace,
                        policy=policy,
                        device=device,
                        args=args,
                        input_dirs=input_dirs,
                        stats_mean=stats_mean,
                        stats_std=stats_std,
                        sample_index=idx - 1,
                        total_samples=len(sampled_npz_files),
                        compare_mode=True,
                        candidate_validator=candidate_validator,
                        cspace_feature_provider=cspace_feature_provider,
                    )
                    candidate_summary = run_mode_inference(
                        mode="candidate",
                        npz_path=npz_path,
                        stl_path=stl_path,
                        base_output_dir=base_output_dir,
                        workspace=workspace,
                        policy=policy,
                        device=device,
                        args=args,
                        input_dirs=input_dirs,
                        stats_mean=stats_mean,
                        stats_std=stats_std,
                        sample_index=idx - 1,
                        total_samples=len(sampled_npz_files),
                        compare_mode=True,
                        candidate_validator=candidate_validator,
                        cspace_feature_provider=cspace_feature_provider,
                    )
                    compare_summary = build_compare_summary(
                        npz_path=npz_path,
                        baseline_summary=baseline_summary,
                        candidate_summary=candidate_summary,
                    )
                    compare_dir = ensure_dir(base_output_dir / "compare")
                    with open(compare_dir / "summary.json", "w", encoding="utf-8") as f:
                        json.dump(compare_summary, f, indent=2)
                    manifest["processed"].append({
                        "npz_path": str(npz_path),
                        "output_dir": str(base_output_dir),
                        "baseline_summary_path": str(base_output_dir / "baseline" / "summary.json"),
                        "candidate_summary_path": str(base_output_dir / "candidate" / "summary.json"),
                        "compare_summary_path": str(compare_dir / "summary.json"),
                    })
                    compare_summaries.append(compare_summary)
                else:
                    summary = run_mode_inference(
                        mode=effective_mode,
                        npz_path=npz_path,
                        stl_path=stl_path,
                        base_output_dir=base_output_dir,
                        workspace=workspace,
                        policy=policy,
                        device=device,
                        args=args,
                        input_dirs=input_dirs,
                        stats_mean=stats_mean,
                        stats_std=stats_std,
                        sample_index=idx - 1,
                        total_samples=len(sampled_npz_files),
                        compare_mode=False,
                        candidate_validator=candidate_validator,
                        cspace_feature_provider=cspace_feature_provider,
                    )
                    manifest["processed"].append(summary)
                print(f"[{idx}/{len(sampled_npz_files)}] done: {npz_path}")
            except Exception as exc:
                manifest["failed"].append({
                    "npz_path": str(npz_path),
                    "output_dir": str(base_output_dir),
                    "stl_path": None if stl_path is None else str(stl_path),
                    "sampling_mode": effective_mode,
                    "candidate_pool_enabled": bool(effective_mode in {"candidate", "compare"}),
                    "sample_index": idx - 1,
                    "error": str(exc),
                })
                print(f"[{idx}/{len(sampled_npz_files)}] failed: {npz_path}")
                print(f"  error: {exc}")
    finally:
        if candidate_validator is not None:
            candidate_validator.close()

    if compare_mode:
        compare_summary_payload = {
            "sample_count": len(compare_summaries),
            "baseline_vs_candidate": compare_summaries,
        }
        min_sdf_gains = [
            float(item["min_sdf_gain_m"]) for item in compare_summaries if item.get("min_sdf_gain_m") is not None
        ]
        if min_sdf_gains:
            compare_summary_payload["candidate_better_count"] = int(sum(gain > 0.0 for gain in min_sdf_gains))
            compare_summary_payload["mean_min_sdf_gain_m"] = float(np.mean(np.asarray(min_sdf_gains, dtype=np.float32)))
            compare_summary_payload["median_min_sdf_gain_m"] = float(np.median(np.asarray(min_sdf_gains, dtype=np.float32)))
        with open(output_root / "compare_summary.json", "w", encoding="utf-8") as f:
            json.dump(compare_summary_payload, f, indent=2)

    total_inference_time_sec = time.perf_counter() - inference_start_time
    manifest["total_inference_time_sec"] = float(total_inference_time_sec)

    manifest_path = output_root / "batch_inference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"sampled manifest: {output_root / 'sampled_npz_manifest.json'}")
    if compare_mode:
        print(f"compare summary: {output_root / 'compare_summary.json'}")
    print(f"manifest: {manifest_path}")
    print(f"processed: {len(manifest['processed'])}")
    print(f"failed: {len(manifest['failed'])}")
    print(f"total inference time (sec): {total_inference_time_sec:.3f}")


if __name__ == "__main__":
    main()
