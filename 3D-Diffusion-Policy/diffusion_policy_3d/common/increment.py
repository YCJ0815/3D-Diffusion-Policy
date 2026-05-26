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


def main() -> None:
    args = build_parser().parse_args()

    trajectory = load_joint_trajectory(args.npz_path, args.trajectory_key)
    q_start, q_goal = load_endpoint_joint_angles(args.npz_path)
    resampled_trajectory = resample_joint_trajectory(trajectory, args.target_steps)
    resampled_trajectory[0] = q_start
    resampled_trajectory[-1] = q_goal
    delta_trajectory = convert_to_delta_trajectory(resampled_trajectory)

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
