import argparse
import pathlib

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resample a joint trajectory and export it as a delta trajectory."
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        required=True,
        help="Path to the source transition NPZ file.",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        required=True,
        help="Path to save the resampled delta joint trajectory as .npy.",
    )
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="q_plan",
        help="Trajectory key in the NPZ to resample. Default: q_plan.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=65,
        help="Target trajectory length after resampling. Default: 65.",
    )
    return parser


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_joint_trajectory(npz_path: str, trajectory_key: str) -> np.ndarray:
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


def load_endpoint_joint_angles(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
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
    npz_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 65,
) -> np.ndarray:
    trajectory = load_joint_trajectory(npz_path, trajectory_key)
    q_start, q_goal = load_endpoint_joint_angles(npz_path)
    resampled_trajectory = resample_joint_trajectory(trajectory, target_steps)
    resampled_trajectory[0] = q_start
    resampled_trajectory[-1] = q_goal
    return convert_to_delta_trajectory(resampled_trajectory)


def _first_existing_array(
    stats: np.lib.npyio.NpzFile,
    candidate_keys: tuple[str, ...],
    stats_path: str,
) -> np.ndarray:
    for key in candidate_keys:
        if key in stats.files:
            return np.asarray(stats[key], dtype=np.float32)
    raise KeyError(
        f"Missing required statistics in {stats_path}. "
        f"Expected one of {candidate_keys}, available keys: {stats.files}"
    )


def load_increment_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(stats_path)
    mean = _first_existing_array(
        stats,
        ("mean", "delta_mean", "increment_mean", "action_mean"),
        stats_path,
    )
    if "std" in stats.files:
        std = np.asarray(stats["std"], dtype=np.float32)
    elif "delta_std" in stats.files:
        std = np.asarray(stats["delta_std"], dtype=np.float32)
    elif "increment_std" in stats.files:
        std = np.asarray(stats["increment_std"], dtype=np.float32)
    elif "action_std" in stats.files:
        std = np.asarray(stats["action_std"], dtype=np.float32)
    elif "var" in stats.files:
        std = np.sqrt(np.asarray(stats["var"], dtype=np.float32))
    elif "delta_var" in stats.files:
        std = np.sqrt(np.asarray(stats["delta_var"], dtype=np.float32))
    elif "increment_var" in stats.files:
        std = np.sqrt(np.asarray(stats["increment_var"], dtype=np.float32))
    elif "action_var" in stats.files:
        std = np.sqrt(np.asarray(stats["action_var"], dtype=np.float32))
    else:
        raise KeyError(
            f"Missing standard deviation or variance in {stats_path}. "
            f"Available keys: {stats.files}"
        )

    mean = mean.reshape(-1).astype(np.float32)
    std = std.reshape(-1).astype(np.float32)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(
            f"Increment stats must both have shape (6,), got mean {mean.shape} and std {std.shape}"
        )
    if np.any(std <= 0):
        raise ValueError(f"Increment std must be positive for all joints, got {std}")
    return mean, std


def normalize_increment_trajectory(
    delta_trajectory: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    delta_trajectory = np.asarray(delta_trajectory, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 6)
    std = np.asarray(std, dtype=np.float32).reshape(1, 6)
    if delta_trajectory.ndim != 2 or delta_trajectory.shape[1] != 6:
        raise ValueError(f"delta_trajectory must have shape [T, 6], got {delta_trajectory.shape}")
    return ((delta_trajectory - mean) / std).astype(np.float32)


def build_increment_stats_from_paths(
    npz_paths: list[str],
    trajectory_key: str = "q_plan",
    target_steps: int = 65,
    std_eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    if not npz_paths:
        raise ValueError("npz_paths must contain at least one transition file.")
    if std_eps <= 0:
        raise ValueError(f"std_eps must be positive, got {std_eps}")

    delta_trajectories = [
        build_increment_trajectory(
            npz_path=npz_path,
            trajectory_key=trajectory_key,
            target_steps=target_steps,
        )
        for npz_path in npz_paths
    ]
    all_deltas = np.concatenate(delta_trajectories, axis=0).astype(np.float32)
    mean = all_deltas.mean(axis=0).astype(np.float32)
    std = all_deltas.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(std_eps)).astype(np.float32)
    var = (std ** 2).astype(np.float32)
    return {
        "mean": mean,
        "std": std,
        "var": var,
        "count": np.asarray(all_deltas.shape[0], dtype=np.int64),
    }


def save_increment_stats(
    npz_paths: list[str],
    output_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 65,
    std_eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    stats = build_increment_stats_from_paths(
        npz_paths=npz_paths,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        std_eps=std_eps,
    )
    output = pathlib.Path(output_path)
    ensure_parent(output)
    np.savez(output, **stats)
    return stats


def build_normalized_increment_trajectory(
    npz_path: str,
    stats_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 65,
) -> np.ndarray:
    delta_trajectory = build_increment_trajectory(
        npz_path=npz_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
    )
    mean, std = load_increment_stats(stats_path)
    return normalize_increment_trajectory(delta_trajectory, mean, std)


def main() -> None:
    args = build_parser().parse_args()

    trajectory = load_joint_trajectory(args.npz_path, args.trajectory_key)
    q_start, q_goal = load_endpoint_joint_angles(args.npz_path)
    delta_trajectory = build_increment_trajectory(
        npz_path=args.npz_path,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
    )

    output_npy = pathlib.Path(args.output_npy)
    ensure_parent(output_npy)
    np.save(output_npy, delta_trajectory.astype(np.float32))

    print(f"trajectory_key: {args.trajectory_key}")
    print(f"input_trajectory: {trajectory.shape}")
    print(f"resampled_absolute_trajectory: {resampled_trajectory.shape}")
    print(f"delta_trajectory: {delta_trajectory.shape}")
    print(f"start_joint_angles: {q_start.shape}")
    print(f"goal_joint_angles: {q_goal.shape}")
    print(f"saved_npy: {output_npy}")


if __name__ == "__main__":
    main()
