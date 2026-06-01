import argparse
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.bspline import (
    FREE_CONTROL_POINT_SLICE,
    build_normalized_delta_w_from_npz,
    save_delta_w_stats,
)
from diffusion_policy_3d.common.input_data import load_bspline_planning_input_data
from diffusion_policy_3d.common.pointcloud_roi import (
    extract_normalized_xy_radius_height_roi_from_stl_and_npz,
)


def progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert transition NPZ files into a zarr training dataset for "
            "B-spline control-point residual prediction."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw_data/results/job_000",
        help="Directory containing transition NPZ files.",
    )
    parser.add_argument(
        "--stl-path",
        type=str,
        default=None,
        help=(
            "Fallback STL path used when automatic per-job STL resolution is unavailable. "
            "When NPZ files come from multiple job directories, the script will prefer "
            "data/raw_data/jobs/<job_name>/workpiece.stl for each NPZ."
        ),
    )
    parser.add_argument(
        "--output-zarr",
        type=str,
        required=True,
        help="Path to the output zarr dataset.",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="data/raw_data/results/job_000_bspline_stats.npz",
        help="Path to the B-spline control-point residual statistics file.",
    )
    parser.add_argument(
        "--stats-std-eps",
        type=float,
        default=1e-6,
        help="Minimum std used when computing delta_W statistics.",
    )
    parser.add_argument(
        "--norm-m",
        type=float,
        default=0.1,
        help="Normalization divisor used by point cloud and goal position preprocessing.",
    )
    parser.add_argument("--radius-m", type=float, default=0.1)
    parser.add_argument("--height-m", type=float, default=0.1)
    parser.add_argument("--num-output-points", type=int, default=1024)
    parser.add_argument("--num-mesh-sample-points", type=int, default=100000)
    parser.add_argument("--stl-x-offset-mm", type=float, default=500.0)
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="q_plan",
        help="Trajectory key in each NPZ used for B-spline fitting.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=64,
        help="Resampled trajectory length used by the B-spline fitting stage.",
    )
    parser.add_argument(
        "--spline-degree",
        type=int,
        default=5,
        help="B-spline degree. The target method uses quintic B-splines by default.",
    )
    parser.add_argument(
        "--num-control-points",
        type=int,
        default=16,
        help="Number of B-spline control points to predict residuals for.",
    )
    parser.add_argument("--urdf-path", type=str, default=None)
    parser.add_argument("--use-poisson-disk", action="store_true")
    return parser


def resolve_job_name_from_npz(npz_path: pathlib.Path) -> str | None:
    for parent in npz_path.parents:
        if parent.name.startswith("job_"):
            return parent.name
    return None


def resolve_stl_path_for_npz(
    npz_path: pathlib.Path,
    input_dir: pathlib.Path,
    fallback_stl_path: str | None,
) -> pathlib.Path:
    job_name = resolve_job_name_from_npz(npz_path)
    candidate_paths: list[pathlib.Path] = []

    if job_name is not None:
        for parent in npz_path.parents:
            if parent.name == job_name and parent.parent.name == "results":
                candidate_paths.append(parent.parent.parent / "jobs" / job_name / "workpiece.stl")
                break

        if not candidate_paths:
            input_dir = input_dir.resolve()
            candidate_roots = [
                input_dir.parent if input_dir.name.startswith("job_") else input_dir,
                input_dir.parents[1] if len(input_dir.parents) >= 2 else input_dir,
                PROJECT_ROOT / "data" / "raw_data" / "results",
            ]
            seen_roots: set[pathlib.Path] = set()
            for results_root in candidate_roots:
                resolved_root = results_root.resolve()
                if resolved_root in seen_roots:
                    continue
                seen_roots.add(resolved_root)
                jobs_root = resolved_root.parent / "jobs"
                candidate_paths.append(jobs_root / job_name / "workpiece.stl")

        for candidate_path in candidate_paths:
            if candidate_path.is_file():
                return candidate_path

        raise FileNotFoundError(
            f"Unable to resolve the matching STL for NPZ {npz_path}. "
            f"Expected job-specific STL under job `{job_name}`. "
            f"Tried: {[str(path) for path in candidate_paths]}"
        )

    if fallback_stl_path is not None:
        candidate_path = pathlib.Path(fallback_stl_path)
        if candidate_path.is_file():
            return candidate_path
        raise FileNotFoundError(f"Fallback STL path does not exist: {fallback_stl_path}")

    raise FileNotFoundError(
        f"Unable to resolve STL for NPZ {npz_path}. "
        "The NPZ path does not expose a `job_xxx` parent and no valid fallback `--stl-path` was provided."
    )


def validate_stl_npz_mapping(
    npz_files: list[pathlib.Path],
    input_dir: pathlib.Path,
    fallback_stl_path: str | None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for npz_path in npz_files:
        stl_path = resolve_stl_path_for_npz(
            npz_path=npz_path,
            input_dir=input_dir,
            fallback_stl_path=fallback_stl_path,
        )
        mapping[str(npz_path)] = str(stl_path)
    return mapping


def build_bspline_control_point_residuals(
    npz_path: pathlib.Path,
    stats_path: str,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
) -> np.ndarray:
    fit_result = build_normalized_delta_w_from_npz(
        npz_path=str(npz_path),
        stats_path=stats_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        urdf_path=urdf_path,
        num_control_points=num_control_points,
        degree=spline_degree,
    )
    return fit_result["normalized_delta_w"][FREE_CONTROL_POINT_SLICE].astype(np.float32)


def build_sample(
    npz_path: pathlib.Path,
    stl_path: str,
    stats_path: str,
    norm_m: float,
    radius_m: float,
    height_m: float,
    num_output_points: int,
    num_mesh_sample_points: int,
    stl_x_offset_mm: float,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
    use_poisson_disk: bool,
) -> dict[str, np.ndarray]:
    pointcloud_result = extract_normalized_xy_radius_height_roi_from_stl_and_npz(
        stl_path=stl_path,
        npz_path=str(npz_path),
        radius_m=radius_m,
        height_m=height_m,
        norm_m=norm_m,
        num_output_points=num_output_points,
        num_mesh_sample_points=num_mesh_sample_points,
        use_poisson_disk=use_poisson_disk,
        stl_x_offset_mm=stl_x_offset_mm,
    )
    planning_result = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=norm_m,
        urdf_path=urdf_path,
    )
    action = build_bspline_control_point_residuals(
        npz_path=npz_path,
        stats_path=stats_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        spline_degree=spline_degree,
        num_control_points=num_control_points,
        urdf_path=urdf_path,
    )
    episode_length = action.shape[0]

    def repeat_obs(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(1, *value.shape)
        return np.repeat(value, episode_length, axis=0).astype(np.float32)

    return {
        "point_cloud": np.repeat(
            pointcloud_result.point_cloud[None].astype(np.float32),
            episode_length,
            axis=0,
        ),
        "goal_position": repeat_obs(planning_result.goal_position),
        "goal_direction": repeat_obs(planning_result.goal_direction),
        "first_joint_angles_normalized": repeat_obs(planning_result.first_joint_angles_normalized),
        "last_joint_angles_normalized": repeat_obs(planning_result.last_joint_angles_normalized),
        "action": action.astype(np.float32),
    }


def main() -> None:
    args = build_parser().parse_args()

    input_dir = pathlib.Path(args.input_dir)
    npz_files = sorted(input_dir.rglob("transition_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {args.input_dir}")
    stl_mapping = validate_stl_npz_mapping(
        npz_files=npz_files,
        input_dir=input_dir,
        fallback_stl_path=args.stl_path,
    )

    stats = save_delta_w_stats(
        npz_paths=[str(path) for path in npz_files],
        output_path=args.stats_path,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
        urdf_path=args.urdf_path,
        num_control_points=args.num_control_points,
        degree=args.spline_degree,
        std_eps=args.stats_std_eps,
    )

    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer.create_empty_numpy()
    for npz_path in progress(npz_files, desc="build zarr episodes", unit="file"):
        sample = build_sample(
            npz_path=npz_path,
            stl_path=stl_mapping[str(npz_path)],
            stats_path=args.stats_path,
            norm_m=args.norm_m,
            radius_m=args.radius_m,
            height_m=args.height_m,
            num_output_points=args.num_output_points,
            num_mesh_sample_points=args.num_mesh_sample_points,
            stl_x_offset_mm=args.stl_x_offset_mm,
            trajectory_key=args.trajectory_key,
            target_steps=args.target_steps,
            spline_degree=args.spline_degree,
            num_control_points=args.num_control_points,
            urdf_path=args.urdf_path,
            use_poisson_disk=args.use_poisson_disk,
        )
        buffer.add_episode(sample)

    output_zarr = pathlib.Path(args.output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    buffer.save_to_path(str(output_zarr), if_exists="replace")

    print(f"npz_count: {len(npz_files)}")
    print(f"episodes: {buffer.n_episodes}")
    print(f"steps: {buffer.n_steps}")
    for key, value in buffer.items():
        print(f"{key}: {value.shape}")
    print(f"stats_count: {int(stats['count'])}")
    print(f"stats_mean: {stats['mean'].shape}")
    print(f"stats_std: {stats['std'].shape}")
    print(f"basis_matrix: {stats['basis_matrix'].shape}")
    print(f"stl_mode: strict per-job auto-resolve")
    print(f"resolved_stl_jobs: {len(set(stl_mapping.values()))}")
    print(f"stats_path: {args.stats_path}")
    print(f"saved_zarr: {output_zarr}")


if __name__ == "__main__":
    main()
