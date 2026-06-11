import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.increment import resample_joint_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a joint-configuration pool from transition trajectories by "
            "resampling each trajectory to 10 steps and extracting the 8 inner states."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw_data",
        help=(
            "Legacy root directory containing transition_*.npz files. "
            "Ignored when any explicit results directory arguments are provided."
        ),
    )
    parser.add_argument(
        "--jobs-dir",
        type=str,
        default="data/raw_data/jobs",
        help="Directory containing regular jobs metadata/assets.",
    )
    parser.add_argument(
        "--simple-jobs-dir",
        type=str,
        default="data/raw_data/simple_jobs",
        help="Directory containing simple jobs metadata/assets.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/raw_data/results",
        help="Directory containing regular transition_*.npz results.",
    )
    parser.add_argument(
        "--simple-results-dir",
        type=str,
        default="data/raw_data/simple_results",
        help="Directory containing simple transition_*.npz results.",
    )
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="q_plan",
        help="Trajectory key to extract from each transition NPZ.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs/joint_configuration_pool",
        help="Directory to save the extracted pool artifacts.",
    )
    parser.add_argument(
        "--resample-steps",
        type=int,
        default=10,
        help="Trajectory length after uniform resampling.",
    )
    parser.add_argument(
        "--num-inner-samples",
        type=int,
        default=8,
        help="Number of inner joint configurations kept after dropping endpoints.",
    )
    parser.add_argument(
        "--range-expand-ratio",
        type=float,
        default=0.1,
        help="Expand min/max range by this ratio before normalization.",
    )
    return parser


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_scan_roots(args: argparse.Namespace) -> list[pathlib.Path]:
    explicit_results_roots = [
        pathlib.Path(path).resolve()
        for path in (args.results_dir, args.simple_results_dir)
        if path is not None
    ]
    if explicit_results_roots:
        return explicit_results_roots
    return [pathlib.Path(args.input_dir).resolve()]


def collect_npz_paths(scan_roots: list[pathlib.Path]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        paths.extend(sorted(root.rglob("transition_*.npz")))
    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise FileNotFoundError(
            f"No transition_*.npz files found under: {[str(root) for root in scan_roots]}"
        )
    return unique_paths


def serialize_npz_path(npz_path: pathlib.Path, scan_roots: list[pathlib.Path]) -> str:
    npz_path = npz_path.resolve()
    for root in scan_roots:
        try:
            relative = npz_path.relative_to(root.resolve())
            return str(relative)
        except ValueError:
            continue
    try:
        return str(npz_path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(npz_path)


def load_joint_trajectory(npz_path: pathlib.Path, trajectory_key: str) -> tuple[np.ndarray, np.lib.npyio.NpzFile]:
    data = np.load(npz_path)
    if trajectory_key not in data.files:
        raise KeyError(
            f"Trajectory key `{trajectory_key}` not found in {npz_path}. "
            f"Available keys: {data.files}"
        )
    trajectory = np.asarray(data[trajectory_key], dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError(
            f"Trajectory `{trajectory_key}` must have shape [T, 6], got {trajectory.shape} in {npz_path}"
        )
    if trajectory.shape[0] < 2:
        raise ValueError(
            f"Trajectory `{trajectory_key}` must contain at least 2 frames, got {trajectory.shape[0]} in {npz_path}"
        )
    return trajectory, data


def apply_endpoint_correction(
    resampled_trajectory: np.ndarray,
    data: np.lib.npyio.NpzFile,
    npz_path: pathlib.Path,
) -> np.ndarray:
    corrected = np.asarray(resampled_trajectory, dtype=np.float32).copy()
    has_start = "q_start" in data.files
    has_goal = "q_goal" in data.files
    if has_start != has_goal:
        raise KeyError(f"{npz_path} must contain both `q_start` and `q_goal`, or neither.")
    if has_start and has_goal:
        q_start = np.asarray(data["q_start"], dtype=np.float32)
        q_goal = np.asarray(data["q_goal"], dtype=np.float32)
        if q_start.shape != (6,) or q_goal.shape != (6,):
            raise ValueError(
                f"`q_start` and `q_goal` must both have shape (6,), got {q_start.shape} and {q_goal.shape} in {npz_path}"
            )
        corrected[0] = q_start
        corrected[-1] = q_goal
    return corrected.astype(np.float32)


def extract_inner_joint_configurations(
    trajectory: np.ndarray,
    data: np.lib.npyio.NpzFile,
    npz_path: pathlib.Path,
    resample_steps: int,
    num_inner_samples: int,
) -> np.ndarray:
    if resample_steps != num_inner_samples + 2:
        raise ValueError(
            f"resample_steps must equal num_inner_samples + 2, got {resample_steps} and {num_inner_samples}"
        )
    resampled = resample_joint_trajectory(trajectory=trajectory, target_steps=resample_steps)
    corrected = apply_endpoint_correction(
        resampled_trajectory=resampled,
        data=data,
        npz_path=npz_path,
    )
    inner = corrected[1:-1]
    if inner.shape != (num_inner_samples, 6):
        raise ValueError(
            f"Expected extracted inner trajectory to have shape ({num_inner_samples}, 6), got {inner.shape}"
        )
    return inner.astype(np.float32)


def build_ragged_trajectory_archive(
    trajectories: list[np.ndarray],
    paths: list[str],
) -> dict[str, np.ndarray]:
    if len(trajectories) != len(paths):
        raise ValueError("trajectories and paths must have the same length.")
    lengths = np.asarray([traj.shape[0] for traj in trajectories], dtype=np.int32)
    values_concat = np.concatenate(trajectories, axis=0).astype(np.float32)
    return {
        "values_concat": values_concat,
        "lengths": lengths,
        "paths": np.asarray(paths, dtype="<U512"),
    }


def compute_normalization(
    joint_pool_raw: np.ndarray,
    range_expand_ratio: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[int]]:
    if joint_pool_raw.ndim != 2 or joint_pool_raw.shape[1] != 6:
        raise ValueError(f"joint_pool_raw must have shape [K, 6], got {joint_pool_raw.shape}")
    joint_min_raw = joint_pool_raw.min(axis=0).astype(np.float32)
    joint_max_raw = joint_pool_raw.max(axis=0).astype(np.float32)
    joint_span_raw = (joint_max_raw - joint_min_raw).astype(np.float32)
    expanded_min = (joint_min_raw - range_expand_ratio * joint_span_raw).astype(np.float32)
    expanded_max = (joint_max_raw + range_expand_ratio * joint_span_raw).astype(np.float32)
    expanded_span = (expanded_max - expanded_min).astype(np.float32)
    zero_span_mask = joint_span_raw == 0
    safe_span = expanded_span.copy()
    safe_span[zero_span_mask] = 1.0
    normalized_01 = (joint_pool_raw - expanded_min.reshape(1, 6)) / safe_span.reshape(1, 6)
    normalized = (normalized_01 * 2.0 - 1.0).astype(np.float32)
    range_payload = {
        "joint_min_raw": joint_min_raw,
        "joint_max_raw": joint_max_raw,
        "joint_min_expanded": expanded_min,
        "joint_max_expanded": expanded_max,
        "joint_span_raw": joint_span_raw,
        "expand_ratio": np.asarray(range_expand_ratio, dtype=np.float32),
    }
    zero_span_joints = np.flatnonzero(zero_span_mask).astype(np.int32).tolist()
    return normalized, range_payload, zero_span_joints


def main() -> None:
    args = build_parser().parse_args()
    if args.resample_steps <= 2:
        raise ValueError(f"resample_steps must be greater than 2, got {args.resample_steps}")
    if args.num_inner_samples <= 0:
        raise ValueError(f"num_inner_samples must be positive, got {args.num_inner_samples}")
    if args.resample_steps != args.num_inner_samples + 2:
        raise ValueError(
            f"resample_steps must equal num_inner_samples + 2, got {args.resample_steps} and {args.num_inner_samples}"
        )
    if args.range_expand_ratio < 0:
        raise ValueError(f"range_expand_ratio must be non-negative, got {args.range_expand_ratio}")

    scan_roots = resolve_scan_roots(args)
    npz_paths = collect_npz_paths(scan_roots)
    output_dir = pathlib.Path(args.output_dir)
    ensure_dir(output_dir)

    raw_trajectories: list[np.ndarray] = []
    extracted_trajectories: list[np.ndarray] = []
    serialized_paths: list[str] = []

    for npz_path in npz_paths:
        trajectory, data = load_joint_trajectory(
            npz_path=npz_path,
            trajectory_key=args.trajectory_key,
        )
        extracted = extract_inner_joint_configurations(
            trajectory=trajectory,
            data=data,
            npz_path=npz_path,
            resample_steps=args.resample_steps,
            num_inner_samples=args.num_inner_samples,
        )
        raw_trajectories.append(trajectory.astype(np.float32))
        extracted_trajectories.append(extracted.astype(np.float32))
        serialized_paths.append(serialize_npz_path(npz_path=npz_path, scan_roots=scan_roots))
        data.close()

    trajectories_after_extraction = np.stack(extracted_trajectories, axis=0).astype(np.float32)
    joint_configuration_pool_raw = trajectories_after_extraction.reshape(-1, 6).astype(np.float32)
    joint_configuration_pool_normalized, normalization_range, zero_span_joints = compute_normalization(
        joint_pool_raw=joint_configuration_pool_raw,
        range_expand_ratio=float(args.range_expand_ratio),
    )

    raw_archive = build_ragged_trajectory_archive(
        trajectories=raw_trajectories,
        paths=serialized_paths,
    )

    np.savez(output_dir / "trajectories_before_extraction.npz", **raw_archive)
    np.save(output_dir / "trajectories_after_extraction.npy", trajectories_after_extraction)
    np.save(output_dir / "joint_configuration_pool_raw.npy", joint_configuration_pool_raw)
    np.save(output_dir / "joint_configuration_pool_normalized.npy", joint_configuration_pool_normalized)
    np.savez(output_dir / "normalization_range.npz", **normalization_range)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": args.input_dir,
        "jobs_dir": args.jobs_dir,
        "simple_jobs_dir": args.simple_jobs_dir,
        "results_dir": args.results_dir,
        "simple_results_dir": args.simple_results_dir,
        "scan_roots": [str(path) for path in scan_roots],
        "trajectory_key": args.trajectory_key,
        "output_dir": str(output_dir),
        "num_trajectories": len(raw_trajectories),
        "resample_steps": int(args.resample_steps),
        "num_inner_samples": int(args.num_inner_samples),
        "range_expand_ratio": float(args.range_expand_ratio),
        "trajectory_shape_after_extraction": list(trajectories_after_extraction.shape),
        "joint_configuration_pool_raw_shape": list(joint_configuration_pool_raw.shape),
        "joint_configuration_pool_normalized_shape": list(joint_configuration_pool_normalized.shape),
        "ragged_values_concat_shape": list(raw_archive["values_concat"].shape),
        "ragged_lengths_shape": list(raw_archive["lengths"].shape),
        "zero_span_joints": zero_span_joints,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"num_trajectories: {len(raw_trajectories)}")
    print(f"trajectories_after_extraction: {trajectories_after_extraction.shape}")
    print(f"joint_configuration_pool_raw: {joint_configuration_pool_raw.shape}")
    print(f"joint_configuration_pool_normalized: {joint_configuration_pool_normalized.shape}")
    print(f"zero_span_joints: {zero_span_joints}")
    print(f"saved_output_dir: {output_dir}")


if __name__ == "__main__":
    main()
