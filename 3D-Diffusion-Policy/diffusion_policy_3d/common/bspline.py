from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from diffusion_policy_3d.common.increment import load_joint_trajectory, resample_joint_trajectory
from diffusion_policy_3d.common.input_data import (
    _default_urdf_path,
    _load_joint_limits_from_urdf,
    _normalize_joint_angles,
)


FREE_CONTROL_POINT_SLICE = slice(2, 14)


def _progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a joint trajectory from a transition NPZ file, resample it to [64, 6], "
            "and normalize it using URDF joint limits."
        )
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
        help="Path to save the normalized resampled joint trajectory as .npy.",
    )
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="q_plan",
        help="Trajectory key in the NPZ to process. Default: q_plan.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=64,
        help="Target trajectory length after resampling. Default: 64.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Path to the robot URDF. Default: config/ur5e_with_pen.urdf.",
    )
    parser.add_argument(
        "--num-control-points",
        type=int,
        default=16,
        help="Number of control points used for quintic B-spline fitting.",
    )
    return parser


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_joint_trajectory_with_endpoints(
    npz_path: str,
    trajectory_key: str = "q_plan",
) -> np.ndarray:
    data = np.load(npz_path)
    trajectory = load_joint_trajectory(npz_path=npz_path, trajectory_key=trajectory_key)

    if "q_start" not in data.files or "q_goal" not in data.files:
        raise KeyError(f"`q_start` and `q_goal` are required in {npz_path} for endpoint correction.")

    q_start = np.asarray(data["q_start"], dtype=np.float32)
    q_goal = np.asarray(data["q_goal"], dtype=np.float32)
    if q_start.shape != (6,) or q_goal.shape != (6,):
        raise ValueError(
            f"`q_start` and `q_goal` must both have shape (6,), got {q_start.shape} and {q_goal.shape}"
        )

    corrected_trajectory = np.asarray(trajectory, dtype=np.float32).copy()
    corrected_trajectory[0] = q_start
    corrected_trajectory[-1] = q_goal
    return corrected_trajectory.astype(np.float32)


def resample_npz_joint_trajectory(
    npz_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 64,
) -> np.ndarray:
    trajectory = load_joint_trajectory_with_endpoints(
        npz_path=npz_path,
        trajectory_key=trajectory_key,
    )
    resampled_trajectory = resample_joint_trajectory(
        trajectory=trajectory,
        target_steps=target_steps,
    )

    # Preserve the exact start/end joint states after interpolation.
    resampled_trajectory[0] = trajectory[0]
    resampled_trajectory[-1] = trajectory[-1]
    return resampled_trajectory.astype(np.float32)


def normalize_joint_trajectory_with_urdf_limits(
    trajectory: np.ndarray,
    urdf_path: str | None = None,
) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError(f"trajectory must have shape [T, 6], got {trajectory.shape}")

    resolved_urdf_path = urdf_path if urdf_path is not None else str(_default_urdf_path())
    _, joint_lower_limits, joint_upper_limits = _load_joint_limits_from_urdf(resolved_urdf_path)
    normalized_trajectory = np.stack(
        [
            _normalize_joint_angles(step, joint_lower_limits, joint_upper_limits)
            for step in trajectory
        ],
        axis=0,
    )
    return normalized_trajectory.astype(np.float32)


def build_normalized_resampled_joint_trajectory(
    npz_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 64,
    urdf_path: str | None = None,
) -> np.ndarray:
    resampled_trajectory = resample_npz_joint_trajectory(
        npz_path=npz_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
    )
    return normalize_joint_trajectory_with_urdf_limits(
        trajectory=resampled_trajectory,
        urdf_path=urdf_path,
    )


def build_open_uniform_knot_vector(
    num_control_points: int,
    degree: int = 5,
) -> np.ndarray:
    if degree != 5:
        raise ValueError(f"This pipeline currently expects a quintic B-spline, got degree={degree}")
    if num_control_points <= degree:
        raise ValueError(
            f"num_control_points must be greater than degree for B-spline fitting, "
            f"got num_control_points={num_control_points}, degree={degree}"
        )

    num_knots = num_control_points + degree + 1
    knot_vector = np.zeros(num_knots, dtype=np.float64)
    knot_vector[-(degree + 1):] = 1.0

    num_internal_knots = num_knots - 2 * (degree + 1)
    if num_internal_knots > 0:
        internal_knots = np.linspace(
            0.0,
            1.0,
            num_internal_knots + 2,
            dtype=np.float64,
        )[1:-1]
        knot_vector[degree + 1: degree + 1 + num_internal_knots] = internal_knots
    return knot_vector


def build_bspline_basis_matrix(
    sample_parameters: np.ndarray,
    num_control_points: int,
    degree: int = 5,
) -> np.ndarray:
    sample_parameters = np.asarray(sample_parameters, dtype=np.float64).reshape(-1)
    knot_vector = build_open_uniform_knot_vector(
        num_control_points=num_control_points,
        degree=degree,
    )
    basis = np.empty((sample_parameters.shape[0], num_control_points), dtype=np.float64)
    for control_idx in range(num_control_points):
        coeffs = np.zeros(num_control_points, dtype=np.float64)
        coeffs[control_idx] = 1.0
        basis[:, control_idx] = BSpline(
            knot_vector,
            coeffs,
            degree,
            extrapolate=False,
        )(sample_parameters)
    return basis


def fit_quintic_bspline_control_points(
    normalized_trajectory: np.ndarray,
    num_control_points: int = 16,
    degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_trajectory = np.asarray(normalized_trajectory, dtype=np.float32)
    if normalized_trajectory.ndim != 2 or normalized_trajectory.shape[1] != 6:
        raise ValueError(
            f"normalized_trajectory must have shape [T, 6], got {normalized_trajectory.shape}"
        )

    sample_parameters = np.linspace(
        0.0,
        1.0,
        normalized_trajectory.shape[0],
        dtype=np.float64,
    )
    basis = build_bspline_basis_matrix(
        sample_parameters=sample_parameters,
        num_control_points=num_control_points,
        degree=degree,
    )
    if num_control_points < 4:
        raise ValueError(
            f"num_control_points must be at least 4 to pin the first/last two control points, "
            f"got {num_control_points}"
        )

    start_state = normalized_trajectory[0].astype(np.float64)
    end_state = normalized_trajectory[-1].astype(np.float64)

    control_points = np.empty((num_control_points, normalized_trajectory.shape[1]), dtype=np.float64)
    control_points[:2] = start_state[None, :]
    control_points[-2:] = end_state[None, :]

    num_free_control_points = num_control_points - 4
    if num_free_control_points > 0:
        fixed_basis = np.concatenate(
            [
                basis[:, :2].sum(axis=1, keepdims=True),
                basis[:, -2:].sum(axis=1, keepdims=True),
            ],
            axis=1,
        )
        fixed_contribution = (
            fixed_basis[:, :1] * start_state[None, :]
            + fixed_basis[:, 1:] * end_state[None, :]
        )
        residual_targets = normalized_trajectory.astype(np.float64) - fixed_contribution
        free_basis = basis[:, 2:-2]
        free_control_points, _, _, _ = np.linalg.lstsq(
            free_basis,
            residual_targets,
            rcond=None,
        )
        control_points[2:-2] = free_control_points
    return control_points.astype(np.float32), build_open_uniform_knot_vector(
        num_control_points=num_control_points,
        degree=degree,
    ).astype(np.float32)


def build_linear_control_points(
    start_state: np.ndarray,
    end_state: np.ndarray,
    num_control_points: int = 16,
) -> np.ndarray:
    start_state = np.asarray(start_state, dtype=np.float32).reshape(-1)
    end_state = np.asarray(end_state, dtype=np.float32).reshape(-1)
    if start_state.shape != (6,) or end_state.shape != (6,):
        raise ValueError(
            f"start_state and end_state must both have shape (6,), got {start_state.shape} and {end_state.shape}"
        )
    if num_control_points < 4:
        raise ValueError(
            f"num_control_points must be at least 4 to pin the first/last two control points, "
            f"got {num_control_points}"
        )

    control_points = np.empty((num_control_points, 6), dtype=np.float32)
    control_points[:2] = start_state[None, :]
    control_points[-2:] = end_state[None, :]

    num_free_control_points = num_control_points - 4
    if num_free_control_points > 0:
        interpolation_weights = np.linspace(
            0.0,
            1.0,
            num_free_control_points + 2,
            dtype=np.float32,
        )[1:-1]
        control_points[2:-2] = (
            (1.0 - interpolation_weights[:, None]) * start_state[None, :]
            + interpolation_weights[:, None] * end_state[None, :]
        )
    return control_points.astype(np.float32)


def evaluate_quintic_bspline(
    control_points: np.ndarray,
    num_steps: int,
    degree: int = 5,
    knot_vector: np.ndarray | None = None,
) -> np.ndarray:
    control_points = np.asarray(control_points, dtype=np.float32)
    if control_points.ndim != 2 or control_points.shape[1] != 6:
        raise ValueError(f"control_points must have shape [K, 6], got {control_points.shape}")
    if num_steps <= 1:
        raise ValueError(f"num_steps must be greater than 1, got {num_steps}")

    if knot_vector is None:
        knot_vector = build_open_uniform_knot_vector(
            num_control_points=control_points.shape[0],
            degree=degree,
        )
    else:
        knot_vector = np.asarray(knot_vector, dtype=np.float64).reshape(-1)

    sample_parameters = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    spline = BSpline(
        knot_vector,
        control_points.astype(np.float64),
        degree,
        axis=0,
        extrapolate=False,
    )
    fitted_trajectory = spline(sample_parameters)
    return np.asarray(fitted_trajectory, dtype=np.float32)


def fit_quintic_bspline_to_npz_trajectory(
    npz_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 64,
    urdf_path: str | None = None,
    num_control_points: int = 16,
    degree: int = 5,
) -> dict[str, np.ndarray]:
    normalized_trajectory = build_normalized_resampled_joint_trajectory(
        npz_path=npz_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        urdf_path=urdf_path,
    )
    control_points, knot_vector = fit_quintic_bspline_control_points(
        normalized_trajectory=normalized_trajectory,
        num_control_points=num_control_points,
        degree=degree,
    )
    sample_parameters = np.linspace(
        0.0,
        1.0,
        normalized_trajectory.shape[0],
        dtype=np.float64,
    )
    basis_matrix = build_bspline_basis_matrix(
        sample_parameters=sample_parameters,
        num_control_points=num_control_points,
        degree=degree,
    )
    linear_control_points = build_linear_control_points(
        start_state=normalized_trajectory[0],
        end_state=normalized_trajectory[-1],
        num_control_points=num_control_points,
    )
    delta_w = control_points.astype(np.float32) - linear_control_points.astype(np.float32)
    fitted_trajectory = evaluate_quintic_bspline(
        control_points=control_points,
        num_steps=normalized_trajectory.shape[0],
        degree=degree,
        knot_vector=knot_vector,
    )
    return {
        "normalized_trajectory": normalized_trajectory.astype(np.float32),
        "control_points": control_points.astype(np.float32),
        "w_star": control_points.astype(np.float32),
        "w_line": linear_control_points.astype(np.float32),
        "delta_w": delta_w.astype(np.float32),
        "basis_matrix": basis_matrix.astype(np.float32),
        "fitted_trajectory": fitted_trajectory.astype(np.float32),
        "knot_vector": knot_vector.astype(np.float32),
    }


def extract_free_delta_w(delta_w: np.ndarray) -> np.ndarray:
    delta_w = np.asarray(delta_w, dtype=np.float32)
    if delta_w.shape != (16, 6):
        raise ValueError(f"delta_w must have shape (16, 6), got {delta_w.shape}")
    return delta_w[FREE_CONTROL_POINT_SLICE].astype(np.float32)


def build_delta_w_stats_from_paths(
    npz_paths: list[str],
    trajectory_key: str = "q_plan",
    target_steps: int = 64,
    urdf_path: str | None = None,
    num_control_points: int = 16,
    degree: int = 5,
    std_eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    if not npz_paths:
        raise ValueError("npz_paths must contain at least one transition file.")
    if std_eps <= 0:
        raise ValueError(f"std_eps must be positive, got {std_eps}")

    free_delta_w_list = []
    basis_matrix = None
    knot_vector = None
    for npz_path in _progress(npz_paths, desc="fit bspline stats", unit="file"):
        fit_result = fit_quintic_bspline_to_npz_trajectory(
            npz_path=npz_path,
            trajectory_key=trajectory_key,
            target_steps=target_steps,
            urdf_path=urdf_path,
            num_control_points=num_control_points,
            degree=degree,
        )
        free_delta_w_list.append(extract_free_delta_w(fit_result["delta_w"]))
        if basis_matrix is None:
            basis_matrix = fit_result["basis_matrix"].astype(np.float32)
        if knot_vector is None:
            knot_vector = fit_result["knot_vector"].astype(np.float32)

    free_delta_w = np.concatenate(free_delta_w_list, axis=0).astype(np.float32)
    mean = free_delta_w.mean(axis=0).astype(np.float32)
    std = free_delta_w.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(std_eps)).astype(np.float32)
    var = (std ** 2).astype(np.float32)

    return {
        "mean": mean,
        "std": std,
        "var": var,
        "count": np.asarray(free_delta_w.shape[0], dtype=np.int64),
        "basis_matrix": basis_matrix.astype(np.float32),
        "knot_vector": knot_vector.astype(np.float32),
    }


def save_delta_w_stats(
    npz_paths: list[str],
    output_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 64,
    urdf_path: str | None = None,
    num_control_points: int = 16,
    degree: int = 5,
    std_eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    stats = build_delta_w_stats_from_paths(
        npz_paths=npz_paths,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        urdf_path=urdf_path,
        num_control_points=num_control_points,
        degree=degree,
        std_eps=std_eps,
    )
    output = Path(output_path)
    ensure_parent(output)
    np.savez(output, **stats)
    return stats


def load_delta_w_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(stats_path)
    if "mean" not in stats.files or "std" not in stats.files:
        raise KeyError(
            f"Missing `mean` or `std` in {stats_path}. Available keys: {stats.files}"
        )
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(
            f"delta_w stats must both have shape (6,), got mean {mean.shape} and std {std.shape}"
        )
    if np.any(std <= 0):
        raise ValueError(f"delta_w std must be positive for all entries, got {std}")
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_delta_w(
    delta_w: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    delta_w = np.asarray(delta_w, dtype=np.float32)
    if delta_w.shape != (16, 6):
        raise ValueError(f"delta_w must have shape (16, 6), got {delta_w.shape}")

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(
            f"mean and std must both have shape (6,), got {mean.shape} and {std.shape}"
        )

    normalized = np.zeros_like(delta_w, dtype=np.float32)
    normalized[FREE_CONTROL_POINT_SLICE] = (
        (delta_w[FREE_CONTROL_POINT_SLICE] - mean.reshape(1, 6)) / std.reshape(1, 6)
    ).astype(np.float32)
    return normalized.astype(np.float32)


def unnormalize_joint_trajectory_with_urdf_limits(
    normalized_trajectory: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    normalized_trajectory = np.asarray(normalized_trajectory, dtype=np.float32)
    lower_limits = np.asarray(lower_limits, dtype=np.float32).reshape(1, -1)
    upper_limits = np.asarray(upper_limits, dtype=np.float32).reshape(1, -1)
    if normalized_trajectory.ndim != 2 or normalized_trajectory.shape[1] != lower_limits.shape[1]:
        raise ValueError(
            "normalized_trajectory must have shape [T, J] matching joint limits, "
            f"got trajectory {normalized_trajectory.shape}, limits {lower_limits.shape}"
        )
    spans = upper_limits - lower_limits
    if np.any(spans <= 0):
        raise ValueError("Invalid URDF joint limits: upper limits must be greater than lower limits.")
    normalized_01 = (normalized_trajectory + 1.0) * 0.5
    return (lower_limits + normalized_01 * spans).astype(np.float32)


def _resolve_free_control_point_slice(num_control_points: int) -> slice:
    if num_control_points < 4:
        raise ValueError(
            f"num_control_points must be at least 4 to pin the first/last two control points, "
            f"got {num_control_points}"
        )
    return slice(2, num_control_points - 2)


def reconstruct_delta_w_from_normalized_free_residual(
    normalized_free_delta_w: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    num_control_points: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    free_slice = _resolve_free_control_point_slice(num_control_points)
    normalized_free_delta_w = np.asarray(normalized_free_delta_w, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 6)
    std = np.asarray(std, dtype=np.float32).reshape(1, 6)

    expected_shape = (free_slice.stop - free_slice.start, 6)
    if normalized_free_delta_w.shape != expected_shape:
        raise ValueError(
            f"normalized_free_delta_w must have shape {expected_shape}, got {normalized_free_delta_w.shape}"
        )

    normalized_delta_w = np.zeros((num_control_points, 6), dtype=np.float32)
    normalized_delta_w[free_slice] = normalized_free_delta_w.astype(np.float32)

    delta_w = np.zeros_like(normalized_delta_w, dtype=np.float32)
    delta_w[free_slice] = (normalized_free_delta_w * std + mean).astype(np.float32)
    return normalized_delta_w.astype(np.float32), delta_w.astype(np.float32)


def reconstruct_control_points_from_normalized_free_residual(
    normalized_free_delta_w: np.ndarray,
    start_state: np.ndarray,
    end_state: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    num_control_points: int = 16,
) -> dict[str, np.ndarray]:
    normalized_delta_w, delta_w = reconstruct_delta_w_from_normalized_free_residual(
        normalized_free_delta_w=normalized_free_delta_w,
        mean=mean,
        std=std,
        num_control_points=num_control_points,
    )
    w_line = build_linear_control_points(
        start_state=start_state,
        end_state=end_state,
        num_control_points=num_control_points,
    )
    w_star = (w_line + delta_w).astype(np.float32)
    return {
        "normalized_delta_w": normalized_delta_w.astype(np.float32),
        "delta_w": delta_w.astype(np.float32),
        "w_line": w_line.astype(np.float32),
        "w_star": w_star.astype(np.float32),
        "control_points": w_star.astype(np.float32),
    }


def reconstruct_trajectory_from_normalized_free_residual(
    normalized_free_delta_w: np.ndarray,
    start_state: np.ndarray,
    end_state: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    num_control_points: int = 16,
    num_steps: int = 64,
    degree: int = 5,
    knot_vector: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    control_point_result = reconstruct_control_points_from_normalized_free_residual(
        normalized_free_delta_w=normalized_free_delta_w,
        start_state=start_state,
        end_state=end_state,
        mean=mean,
        std=std,
        num_control_points=num_control_points,
    )
    fitted_trajectory = evaluate_quintic_bspline(
        control_points=control_point_result["w_star"],
        num_steps=num_steps,
        degree=degree,
        knot_vector=knot_vector,
    )
    control_point_result["fitted_trajectory"] = fitted_trajectory.astype(np.float32)
    return control_point_result


def build_normalized_delta_w_from_npz(
    npz_path: str,
    stats_path: str,
    trajectory_key: str = "q_plan",
    target_steps: int = 64,
    urdf_path: str | None = None,
    num_control_points: int = 16,
    degree: int = 5,
) -> dict[str, np.ndarray]:
    fit_result = fit_quintic_bspline_to_npz_trajectory(
        npz_path=npz_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        urdf_path=urdf_path,
        num_control_points=num_control_points,
        degree=degree,
    )
    mean, std = load_delta_w_stats(stats_path)
    normalized_delta_w = normalize_delta_w(
        delta_w=fit_result["delta_w"],
        mean=mean,
        std=std,
    )
    fit_result["normalized_delta_w"] = normalized_delta_w.astype(np.float32)
    return fit_result


def main() -> None:
    args = build_parser().parse_args()

    fit_result = fit_quintic_bspline_to_npz_trajectory(
        npz_path=args.npz_path,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
        urdf_path=args.urdf_path,
        num_control_points=args.num_control_points,
    )

    output_npy = Path(args.output_npy)
    ensure_parent(output_npy)
    np.save(output_npy, fit_result["control_points"].astype(np.float32))

    print(f"trajectory_key: {args.trajectory_key}")
    print(f"normalized_resampled_trajectory: {fit_result['normalized_trajectory'].shape}")
    print(f"control_points: {fit_result['control_points'].shape}")
    print(f"w_line: {fit_result['w_line'].shape}")
    print(f"delta_w: {fit_result['delta_w'].shape}")
    print(f"basis_matrix: {fit_result['basis_matrix'].shape}")
    print(f"fitted_trajectory: {fit_result['fitted_trajectory'].shape}")
    print(f"target_steps: {args.target_steps}")
    print("spline_degree: 5")
    print(f"num_control_points: {args.num_control_points}")
    print(f"saved_npy: {output_npy}")


if __name__ == "__main__":
    main()
