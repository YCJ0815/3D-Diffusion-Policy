#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

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
    parser.add_argument("--num-control-points", type=int, default=12)
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
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic random trajectory sampling.",
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
        "--num-candidates",
        type=int,
        default=32,
        help="Candidate pool size when candidate sampling is enabled.",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=42,
        help="Base seed for deterministic candidate sampling.",
    )
    parser.add_argument(
        "--candidate-inference-steps",
        type=int,
        default=None,
        help="Optional override for diffusion inference steps during candidate sampling.",
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
    if resolve_sampling_mode(args) not in {"candidate", "compare"}:
        return None
    if args.candidate_selection == "first":
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

    if mode == "baseline":
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
        "sampling_mode": resolve_sampling_mode(args),
        "candidate_pool_enabled": bool(mode == "candidate"),
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
        "num_candidates": int(args.num_candidates if mode == "candidate" else 1),
        "selected_candidate_index": int(artifact.get("candidate_index", 0)),
        "selected_candidate_seed": artifact.get("candidate_seed"),
    }
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
    summary = save_prediction_artifacts(
        output_dir=output_dir,
        raw_obs=raw_obs,
        artifact=artifact,
        metadata=metadata,
        candidate_scores=candidate_scores,
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
    args = build_parser().parse_args()
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
    sampled_npz_files = sample_npz_files(
        npz_files=eligible_npz_files,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
    )
    sampled_manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "sample_source": str(args.sample_source),
        "sample_count": int(args.sample_count),
        "sample_seed": int(args.sample_seed),
        "raw_discovered_count": len(discovered_npz_files),
        "eligible_count": len(eligible_npz_files),
        "sampled_npz_paths": [str(path) for path in sampled_npz_files],
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
        "sample_count": int(args.sample_count),
        "sample_seed": int(args.sample_seed),
        "sampling_mode": effective_mode,
        "candidate_pool_enabled": bool(effective_mode in {"candidate", "compare"}),
        "candidate_selection": str(args.candidate_selection),
        "num_candidates": int(args.num_candidates if effective_mode in {"candidate", "compare"} else 1),
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

    manifest_path = output_root / "batch_inference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"sampled manifest: {output_root / 'sampled_npz_manifest.json'}")
    if compare_mode:
        print(f"compare summary: {output_root / 'compare_summary.json'}")
    print(f"manifest: {manifest_path}")
    print(f"processed: {len(manifest['processed'])}")
    print(f"failed: {len(manifest['failed'])}")


if __name__ == "__main__":
    main()
