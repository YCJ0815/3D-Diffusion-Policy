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
            "For each NPZ, reconstruct and save the predicted replay trajectory."
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
    parser.add_argument("--jobs-root", type=str, default=None, help="Root directory for regular job STL files.")
    parser.add_argument("--simple-jobs-root", type=str, default=None, help="Root directory for simple job STL files.")
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
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of NPZ files to process.")
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


def resolve_job_name_from_npz(npz_path: pathlib.Path) -> str | None:
    for parent in npz_path.parents:
        if parent.name.startswith("job_"):
            return parent.name
    return None


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


def build_output_dir(output_root: pathlib.Path, npz_path: pathlib.Path, input_dirs: list[pathlib.Path]) -> pathlib.Path:
    for input_dir in input_dirs:
        try:
            rel = npz_path.relative_to(input_dir.resolve())
            return output_root / rel.parent / f"{npz_path.stem}_bspline_inference"
        except ValueError:
            continue
    return output_root / npz_path.parent.name / f"{npz_path.stem}_bspline_inference"


def run_single_inference(
    npz_path: pathlib.Path,
    stl_path: pathlib.Path,
    output_dir: pathlib.Path,
    workspace: TrainDP3Workspace,
    policy,
    device: torch.device,
    args,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
) -> dict:
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

    with torch.no_grad():
        result = policy.predict_action(obs_dict)

    pred_action_window = result["action"][0].detach().cpu().numpy().astype(np.float32)
    pred_action_horizon = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
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

    np.save(output_dir / "pred_action_window_normalized.npy", pred_action_window)
    np.save(output_dir / "pred_action_horizon_normalized.npy", pred_action_horizon)
    np.save(output_dir / "pred_delta_w.npy", recon_result["delta_w"])
    np.save(output_dir / "pred_w_line.npy", recon_result["w_line"])
    np.save(output_dir / "pred_w_star.npy", recon_result["w_star"])
    np.save(output_dir / "pred_joint_horizon_normalized.npy", pred_joint_horizon_normalized)
    np.save(output_dir / "pred_joint_horizon.npy", pred_joint_horizon)
    np.save(output_dir / "point_cloud.npy", raw_obs["point_cloud"])

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
        pred_joint_traj=pred_joint_horizon,
        gt_joint_traj=gt_joint_traj,
        output_path=output_dir / "pred_joint_horizon.png",
    )

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "npz_path": str(npz_path),
        "stl_path": str(stl_path),
        "stats_path": str(args.stats_path),
        "output_dir": str(output_dir),
        "n_obs_steps": int(workspace.cfg.n_obs_steps),
        "n_action_steps": int(workspace.cfg.n_action_steps),
        "policy_horizon": int(workspace.cfg.horizon),
        "target_steps": int(args.target_steps),
        "num_control_points": int(args.num_control_points),
        "spline_degree": int(args.spline_degree),
        "pred_action_window_shape": list(pred_action_window.shape),
        "pred_action_horizon_shape": list(pred_action_horizon.shape),
        "pred_joint_horizon_shape": list(pred_joint_horizon.shape),
        "trajectory_key": planning_result.trajectory_key,
        "has_ground_truth_trajectory": bool(gt_joint_traj is not None),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint_path).expanduser().resolve()
    stats_path = pathlib.Path(args.stats_path).expanduser().resolve()
    output_root = ensure_dir(pathlib.Path(args.output_root).expanduser().resolve())
    input_dirs = [pathlib.Path(path).expanduser().resolve() for path in args.input_dirs]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"delta_w stats file not found: {stats_path}")
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

    npz_files = collect_npz_files(input_dirs=input_dirs, max_files=args.max_files)
    device = torch.device(args.device)
    workspace = TrainDP3Workspace.create_from_checkpoint(str(checkpoint_path))
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    policy = policy.to(device)
    policy.eval()
    stats_mean, stats_std = load_delta_w_stats(str(stats_path))

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "stats_path": str(stats_path),
        "output_root": str(output_root),
        "processed": [],
        "failed": [],
    }

    print(f"Found {len(npz_files)} NPZ files.")
    for idx, npz_path in enumerate(npz_files, start=1):
        output_dir = build_output_dir(output_root=output_root, npz_path=npz_path, input_dirs=input_dirs)
        summary_path = output_dir / "summary.json"
        if args.skip_existing and summary_path.is_file():
            print(f"[{idx}/{len(npz_files)}] skip existing: {npz_path}")
            manifest["processed"].append({
                "npz_path": str(npz_path),
                "output_dir": str(output_dir),
                "skipped": True,
            })
            continue

        try:
            ensure_dir(output_dir)
            stl_path = resolve_matching_stl(
                npz_path=npz_path,
                input_dirs=input_dirs,
                jobs_root=args.jobs_root,
                simple_jobs_root=args.simple_jobs_root,
                fallback_stl_path=args.fallback_stl_path,
            )
            summary = run_single_inference(
                npz_path=npz_path,
                stl_path=stl_path,
                output_dir=output_dir,
                workspace=workspace,
                policy=policy,
                device=device,
                args=args,
                stats_mean=stats_mean,
                stats_std=stats_std,
            )
            manifest["processed"].append(summary)
            print(f"[{idx}/{len(npz_files)}] done: {npz_path}")
        except Exception as exc:
            manifest["failed"].append({
                "npz_path": str(npz_path),
                "output_dir": str(output_dir),
                "error": str(exc),
            })
            print(f"[{idx}/{len(npz_files)}] failed: {npz_path}")
            print(f"  error: {exc}")

    manifest_path = output_root / "batch_inference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {manifest_path}")
    print(f"processed: {len(manifest['processed'])}")
    print(f"failed: {len(manifest['failed'])}")


if __name__ == "__main__":
    main()
