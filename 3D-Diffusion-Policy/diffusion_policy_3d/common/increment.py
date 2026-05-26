from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class IncrementDatasetStats:
    mean: np.ndarray
    std: np.ndarray
    sample_count: int
    frame_count: int


def load_increment_stats(stats_path: str | Path) -> IncrementDatasetStats:
    stats_data = np.load(stats_path)
    required_keys = ("joint_mean", "joint_std", "sample_count", "frame_count")
    missing = [key for key in required_keys if key not in stats_data.files]
    if missing:
        raise KeyError(f"Missing increment stats keys in {stats_path}: {missing}")
    return IncrementDatasetStats(
        mean=np.asarray(stats_data["joint_mean"], dtype=np.float32),
        std=np.asarray(stats_data["joint_std"], dtype=np.float32),
        sample_count=int(np.asarray(stats_data["sample_count"]).item()),
        frame_count=int(np.asarray(stats_data["frame_count"]).item()),
    )


def load_joint_trajectory(npz_path: str | Path, trajectory_key: str = "q_plan") -> np.ndarray:
    data = np.load(npz_path)
    if trajectory_key not in data.files:
        raise KeyError(
            f"Trajectory key `{trajectory_key}` not found in {npz_path}. "
            f"Available keys: {data.files}"
        )

    trajectory = np.asarray(data[trajectory_key], dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError(
            f"Trajectory `{trajectory_key}` must have shape [T, 6], got {trajectory.shape}"
        )
    if trajectory.shape[0] < 2:
        raise ValueError(
            f"Trajectory `{trajectory_key}` must contain at least 2 frames for resampling, "
            f"got {trajectory.shape[0]}"
        )
    return trajectory


def load_endpoint_joint_angles(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    required_keys = ("q_start", "q_goal")
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise KeyError(f"Missing endpoint joint angle keys in {npz_path}: {missing}")

    q_start = np.asarray(data["q_start"], dtype=np.float32)
    q_goal = np.asarray(data["q_goal"], dtype=np.float32)
    if q_start.shape != (6,) or q_goal.shape != (6,):
        raise ValueError(
            f"`q_start` and `q_goal` must both have shape (6,), got {q_start.shape} and {q_goal.shape}"
        )
    return q_start, q_goal


def resample_joint_trajectory(trajectory: np.ndarray, target_steps: int = 65) -> np.ndarray:
    if target_steps <= 1:
        raise ValueError(f"target_steps must be greater than 1, got {target_steps}")

    source_steps = trajectory.shape[0]
    source_grid = np.linspace(0.0, 1.0, source_steps, dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, target_steps, dtype=np.float32)

    resampled = np.empty((target_steps, trajectory.shape[1]), dtype=np.float32)
    for joint_idx in range(trajectory.shape[1]):
        resampled[:, joint_idx] = np.interp(
            target_grid,
            source_grid,
            trajectory[:, joint_idx],
        )
    return resampled.astype(np.float32)


def convert_to_delta_trajectory(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError(f"trajectory must have shape [T, 6], got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError(
            f"trajectory must contain at least 2 frames to compute deltas, got {trajectory.shape[0]}"
        )
    return (trajectory[1:] - trajectory[:-1]).astype(np.float32)


def build_increment_trajectory(
    npz_path: str | Path,
    trajectory_key: str = "q_plan",
    target_steps: int = 65,
) -> np.ndarray:
    trajectory = load_joint_trajectory(npz_path, trajectory_key)
    q_start, q_goal = load_endpoint_joint_angles(npz_path)
    resampled_trajectory = resample_joint_trajectory(trajectory, target_steps)
    resampled_trajectory[0] = q_start
    resampled_trajectory[-1] = q_goal
    return convert_to_delta_trajectory(resampled_trajectory)


def build_normalized_increment_trajectory(
    npz_path: str | Path,
    stats_path: str | Path,
    trajectory_key: str = "q_plan",
    target_steps: int = 65,
) -> np.ndarray:
    increment_trajectory = build_increment_trajectory(
        npz_path=npz_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
    )
    stats = load_increment_stats(stats_path)
    return normalize_increment_trajectory(
        increment_trajectory=increment_trajectory,
        mean=stats.mean,
        std=stats.std,
    )


def iter_npz_files(input_dir: str | Path) -> list[Path]:
    input_path = Path(input_dir)
    npz_files = sorted(input_path.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {input_path}")
    return npz_files


def save_increment_trajectory(
    increment_trajectory: np.ndarray,
    source_npz_path: Path,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_npz_path.stem}_delta.npy"
    np.save(output_path, increment_trajectory.astype(np.float32))
    return output_path


def compute_joint_mean_std(increment_trajectories: Iterable[np.ndarray]) -> IncrementDatasetStats:
    flattened = []
    sample_count = 0
    frame_count = 0

    for trajectory in increment_trajectories:
        trajectory = np.asarray(trajectory, dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[1] != 6:
            raise ValueError(f"Each increment trajectory must have shape [T, 6], got {trajectory.shape}")
        flattened.append(trajectory)
        sample_count += 1
        frame_count += trajectory.shape[0]

    if not flattened:
        raise ValueError("No increment trajectories provided for statistics.")

    stacked = np.concatenate(flattened, axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    return IncrementDatasetStats(
        mean=mean,
        std=std,
        sample_count=sample_count,
        frame_count=frame_count,
    )


def normalize_increment_trajectory(
    increment_trajectory: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    increment_trajectory = np.asarray(increment_trajectory, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if increment_trajectory.ndim != 2 or increment_trajectory.shape[1] != 6:
        raise ValueError(
            f"increment_trajectory must have shape [T, 6], got {increment_trajectory.shape}"
        )
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(f"mean/std must both have shape (6,), got {mean.shape} and {std.shape}")
    if np.any(std <= 0):
        raise ValueError(f"std must be positive for all joints, got {std}")
    return ((increment_trajectory - mean.reshape(1, 6)) / std.reshape(1, 6)).astype(np.float32)


def process_increment_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    trajectory_key: str = "q_plan",
    target_steps: int = 65,
    stats_output_path: str | Path | None = None,
) -> IncrementDatasetStats:
    npz_files = iter_npz_files(input_dir)
    increment_output_dir = Path(output_dir)
    increment_output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = increment_output_dir / "raw"
    normalized_output_dir = increment_output_dir / "normalized"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    normalized_output_dir.mkdir(parents=True, exist_ok=True)

    increment_trajectories: list[np.ndarray] = []
    npz_to_increment: list[tuple[Path, np.ndarray]] = []

    for npz_file in npz_files:
        increment_trajectory = build_increment_trajectory(
            npz_path=npz_file,
            trajectory_key=trajectory_key,
            target_steps=target_steps,
        )
        save_increment_trajectory(
            increment_trajectory=increment_trajectory,
            source_npz_path=npz_file,
            output_dir=raw_output_dir,
        )
        increment_trajectories.append(increment_trajectory)
        npz_to_increment.append((npz_file, increment_trajectory))

    stats = compute_joint_mean_std(increment_trajectories)

    for npz_file, increment_trajectory in npz_to_increment:
        normalized_increment = normalize_increment_trajectory(
            increment_trajectory=increment_trajectory,
            mean=stats.mean,
            std=stats.std,
        )
        save_increment_trajectory(
            increment_trajectory=normalized_increment,
            source_npz_path=npz_file,
            output_dir=normalized_output_dir,
        )

    if stats_output_path is not None:
        stats_output_path = Path(stats_output_path)
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            stats_output_path,
            joint_mean=stats.mean.astype(np.float32),
            joint_std=stats.std.astype(np.float32),
            sample_count=np.asarray(stats.sample_count, dtype=np.int64),
            frame_count=np.asarray(stats.frame_count, dtype=np.int64),
        )

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build increment trajectories for all transition NPZ files, recompute per-joint mean/std, and save normalized increments."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw_data/results/job_000",
        help="Directory containing transition NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_data/results/job_000_increment",
        help="Directory to save per-sample increment trajectories under raw/ and normalized/.",
    )
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="q_plan",
        help="Trajectory key in each NPZ file. Default: q_plan.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=65,
        help="Absolute trajectory length before delta conversion. Default: 65.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="data/raw_data/results/job_000_increment_stats.npz",
        help="Path to save joint mean/std statistics as .npz.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = process_increment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
        stats_output_path=args.stats_output,
    )
    print(f"joint_mean: {stats.mean}")
    print(f"joint_std: {stats.std}")
    print(f"sample_count: {stats.sample_count}")
    print(f"frame_count: {stats.frame_count}")
    print(f"output_dir: {args.output_dir}")
    print(f"raw_output_dir: {Path(args.output_dir) / 'raw'}")
    print(f"normalized_output_dir: {Path(args.output_dir) / 'normalized'}")
    print(f"stats_output: {args.stats_output}")


if __name__ == "__main__":
    main()
