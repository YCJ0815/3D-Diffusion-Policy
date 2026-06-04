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
    build_linear_control_points,
    build_normalized_resampled_joint_trajectory,
    build_normalized_delta_w_from_npz,
    fit_quintic_bspline_control_points,
    load_delta_w_stats,
    normalize_delta_w,
    save_delta_w_stats,
)
from diffusion_policy_3d.common.input_data import PlanningInputData, load_bspline_planning_input_data
from diffusion_policy_3d.common.pointcloud_roi import (
    canonicalize_axis_symmetric_tcp_transform,
    convert_points_mm_to_m,
    crop_xy_radius_height_point_cloud,
    load_stl_mesh,
    offset_points,
    sample_mesh_surface,
    sample_point_cloud_to_fixed_size,
    world_to_local_point,
    world_to_local_points,
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
        "--input-dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of directories containing transition NPZ files. "
            "When provided, all directories are scanned and merged."
        ),
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
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=1,
        help=(
            "Number of dataset episodes to generate per transition NPZ. "
            "The first copy uses the base radius/height; extra copies randomize crop parameters."
        ),
    )
    parser.add_argument(
        "--radius-m-min",
        type=float,
        default=None,
        help="Optional minimum random crop radius in meters for augmented copies.",
    )
    parser.add_argument(
        "--radius-m-max",
        type=float,
        default=None,
        help="Optional maximum random crop radius in meters for augmented copies.",
    )
    parser.add_argument(
        "--height-m-min",
        type=float,
        default=None,
        help="Optional minimum random crop height in meters for augmented copies.",
    )
    parser.add_argument(
        "--height-m-max",
        type=float,
        default=None,
        help="Optional maximum random crop height in meters for augmented copies.",
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        default=42,
        help="Random seed used when sampling radius/height for augmented copies.",
    )
    parser.add_argument(
        "--num-output-points",
        type=int,
        default=512,
        help="Final point count after ROI crop and farthest-point sampling.",
    )
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
    parser.add_argument(
        "--add-reversed-copy",
        action="store_true",
        help=(
            "Generate an additional reversed sample for each episode by swapping start/goal, "
            "re-expressing inputs in the original goal TCP frame, and refitting the reversed B-spline."
        ),
    )
    return parser


def collect_npz_files(input_dir: str, input_dirs: list[str] | None) -> list[pathlib.Path]:
    search_dirs = list(input_dirs) if input_dirs else [input_dir]
    npz_files: list[pathlib.Path] = []
    for directory in search_dirs:
        npz_files.extend(sorted(pathlib.Path(directory).rglob("transition_*.npz")))
    return sorted({path.resolve() for path in npz_files})


def resolve_augmentation_bounds(
    base_value: float,
    min_value: float | None,
    max_value: float | None,
    name: str,
) -> tuple[float, float]:
    low = base_value if min_value is None else min_value
    high = base_value if max_value is None else max_value
    if low <= 0 or high <= 0:
        raise ValueError(f"{name} bounds must be positive, got {low} and {high}")
    if low > high:
        raise ValueError(f"{name} minimum cannot exceed maximum, got {low} > {high}")
    return float(low), float(high)


def sample_crop_parameters(
    rng: np.random.Generator,
    copy_idx: int,
    base_radius_m: float,
    base_height_m: float,
    radius_bounds: tuple[float, float],
    height_bounds: tuple[float, float],
) -> tuple[float, float]:
    if copy_idx == 0:
        return float(base_radius_m), float(base_height_m)

    radius_low, radius_high = radius_bounds
    height_low, height_high = height_bounds
    radius_m = float(rng.uniform(radius_low, radius_high))
    height_m = float(rng.uniform(height_low, height_high))
    return radius_m, height_m


def build_raw_mesh_points_world_m(
    stl_path: str,
    num_mesh_sample_points: int,
    stl_x_offset_mm: float,
    use_poisson_disk: bool,
) -> np.ndarray:
    mesh = load_stl_mesh(stl_path)
    raw_mesh_points_mm = sample_mesh_surface(
        mesh=mesh,
        num_points=num_mesh_sample_points,
        use_poisson_disk=use_poisson_disk,
    )
    stl_offset_mm = np.array([stl_x_offset_mm, 0.0, 0.0], dtype=np.float32)
    raw_mesh_points_mm = offset_points(raw_mesh_points_mm, stl_offset_mm)
    return convert_points_mm_to_m(raw_mesh_points_mm).astype(np.float32)


def build_normalized_point_cloud_from_geometry(
    raw_mesh_points_world_m: np.ndarray,
    start_tf_m: np.ndarray,
    goal_xyz_world_m: np.ndarray,
    norm_m: float,
    radius_m: float,
    height_m: float,
    num_output_points: int,
) -> np.ndarray:
    cropped_points_world_m = crop_xy_radius_height_point_cloud(
        points=raw_mesh_points_world_m,
        start=start_tf_m[:3, 3],
        goal=goal_xyz_world_m,
        radius=radius_m,
        height=height_m,
    )
    cropped_points_start_tcp_m = world_to_local_points(
        cropped_points_world_m,
        start_tf_m,
    )
    normalized_points = sample_point_cloud_to_fixed_size(
        cropped_points_start_tcp_m,
        num_output_points,
    ) / norm_m
    return normalized_points.astype(np.float32)


def assemble_sample(
    point_cloud: np.ndarray,
    planning_result: PlanningInputData,
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    episode_length = action.shape[0]

    def repeat_obs(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(1, *value.shape)
        return np.repeat(value, episode_length, axis=0).astype(np.float32)

    return {
        "point_cloud": np.repeat(point_cloud[None].astype(np.float32), episode_length, axis=0),
        "goal_position": repeat_obs(planning_result.goal_position),
        "goal_direction": repeat_obs(planning_result.goal_direction),
        "first_joint_angles_normalized": repeat_obs(planning_result.first_joint_angles_normalized),
        "last_joint_angles_normalized": repeat_obs(planning_result.last_joint_angles_normalized),
        "action": action.astype(np.float32),
    }


def build_reversed_planning_data(
    npz_path: pathlib.Path,
    norm_m: float,
    urdf_path: str | None,
) -> PlanningInputData:
    forward = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=norm_m,
        urdf_path=urdf_path,
    )
    data = np.load(npz_path)
    start_tf = canonicalize_axis_symmetric_tcp_transform(np.asarray(data["start_tf"], dtype=np.float32))
    goal_tf = canonicalize_axis_symmetric_tcp_transform(np.asarray(data["goal_tf"], dtype=np.float32))

    reversed_start_tf = goal_tf
    reversed_goal_position_world = start_tf[:3, 3].astype(np.float32)
    reversed_goal_position_start_tcp_frame = world_to_local_point(
        reversed_goal_position_world,
        reversed_start_tf,
    ).astype(np.float32)
    reversed_goal_position = (reversed_goal_position_start_tcp_frame / norm_m).astype(np.float32)
    reversed_goal_rotation = start_tf[:3, :3].astype(np.float32)
    reversed_goal_direction_world = reversed_goal_rotation[:, 2].astype(np.float32)
    reversed_goal_direction = (
        reversed_start_tf[:3, :3].T @ reversed_goal_rotation
    )[:, :2].reshape(-1).astype(np.float32)

    return PlanningInputData(
        goal_position_world=reversed_goal_position_world,
        goal_position_start_tcp_frame=reversed_goal_position_start_tcp_frame,
        goal_position=reversed_goal_position,
        goal_rotation=reversed_goal_rotation,
        goal_direction_world=reversed_goal_direction_world,
        goal_direction=reversed_goal_direction,
        joint_names=forward.joint_names,
        joint_lower_limits=forward.joint_lower_limits,
        joint_upper_limits=forward.joint_upper_limits,
        first_joint_angles=forward.last_joint_angles.astype(np.float32),
        last_joint_angles=forward.first_joint_angles.astype(np.float32),
        first_joint_angles_normalized=forward.last_joint_angles_normalized.astype(np.float32),
        last_joint_angles_normalized=forward.first_joint_angles_normalized.astype(np.float32),
        trajectory_key=forward.trajectory_key,
    )


def build_reversed_bspline_control_point_residuals(
    npz_path: pathlib.Path,
    stats_path: str,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
) -> np.ndarray:
    normalized_trajectory = build_normalized_resampled_joint_trajectory(
        npz_path=str(npz_path),
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        urdf_path=urdf_path,
    )[::-1].copy()
    control_points, _ = fit_quintic_bspline_control_points(
        normalized_trajectory=normalized_trajectory,
        num_control_points=num_control_points,
        degree=spline_degree,
    )
    linear_control_points = build_linear_control_points(
        start_state=normalized_trajectory[0],
        end_state=normalized_trajectory[-1],
        num_control_points=num_control_points,
    )
    delta_w = control_points.astype(np.float32) - linear_control_points.astype(np.float32)
    mean, std = load_delta_w_stats(stats_path)
    normalized_delta_w = normalize_delta_w(
        delta_w=delta_w,
        mean=mean,
        std=std,
    )
    return normalized_delta_w[FREE_CONTROL_POINT_SLICE].astype(np.float32)


def resolve_job_name_from_npz(npz_path: pathlib.Path) -> str | None:
    for parent in npz_path.parents:
        if parent.name.startswith("job_"):
            return parent.name
    return None


def resolve_results_dir_name_from_npz(npz_path: pathlib.Path) -> str | None:
    for parent in npz_path.parents:
        if "results" in parent.name:
            return parent.name
    return None


def encode_workpiece_id(local_workpiece_id: int, results_dir_name: str | None) -> int:
    if results_dir_name == "simple_results":
        return 1000 + int(local_workpiece_id)
    return int(local_workpiece_id)


def resolve_workpiece_metadata_from_npz(npz_path: pathlib.Path) -> tuple[int, int, int]:
    job_name = resolve_job_name_from_npz(npz_path)
    if job_name is None:
        raise ValueError(f"Unable to infer workpiece id from NPZ path: {npz_path}")
    try:
        local_workpiece_id = int(job_name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid job/workpiece name format: {job_name}") from exc
    results_dir_name = resolve_results_dir_name_from_npz(npz_path)
    encoded_workpiece_id = encode_workpiece_id(
        local_workpiece_id=local_workpiece_id,
        results_dir_name=results_dir_name,
    )
    workpiece_source_id = 1 if results_dir_name == "simple_results" else 0
    return encoded_workpiece_id, local_workpiece_id, workpiece_source_id


def resolve_stl_path_for_npz(
    npz_path: pathlib.Path,
    input_dirs: list[pathlib.Path],
    fallback_stl_path: str | None,
) -> pathlib.Path:
    job_name = resolve_job_name_from_npz(npz_path)
    candidate_paths: list[pathlib.Path] = []

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

    if job_name is not None:
        for parent in npz_path.parents:
            if parent.name == job_name and "results" in parent.parent.name:
                candidate_paths.append(
                    infer_jobs_dir_from_results_dir(parent.parent) / job_name / "workpiece.stl"
                )
                break

        if not candidate_paths:
            candidate_roots: list[pathlib.Path] = []
            for input_dir in input_dirs:
                resolved_input_dir = input_dir.resolve()
                candidate_roots.extend(
                    [
                        resolved_input_dir.parent
                        if resolved_input_dir.name.startswith("job_")
                        else resolved_input_dir,
                        resolved_input_dir.parents[1]
                        if len(resolved_input_dir.parents) >= 2
                        else resolved_input_dir,
                    ]
                )
            candidate_roots.extend(
                [
                    PROJECT_ROOT / "data" / "raw_data" / "results",
                    PROJECT_ROOT / "data" / "raw_data" / "simple_results",
                ]
            )
            seen_roots: set[pathlib.Path] = set()
            for results_root in candidate_roots:
                resolved_root = results_root.resolve()
                if resolved_root in seen_roots:
                    continue
                seen_roots.add(resolved_root)
                jobs_root = infer_jobs_dir_from_results_dir(resolved_root)
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
    input_dirs: list[pathlib.Path],
    fallback_stl_path: str | None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for npz_path in npz_files:
        stl_path = resolve_stl_path_for_npz(
            npz_path=npz_path,
            input_dirs=input_dirs,
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


def build_forward_invariants(
    npz_path: pathlib.Path,
    stats_path: str,
    norm_m: float,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
) -> tuple[PlanningInputData, np.ndarray]:
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
    return planning_result, action.astype(np.float32)


def build_reversed_invariants(
    npz_path: pathlib.Path,
    stats_path: str,
    norm_m: float,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
) -> tuple[PlanningInputData, np.ndarray]:
    planning_result = build_reversed_planning_data(
        npz_path=npz_path,
        norm_m=norm_m,
        urdf_path=urdf_path,
    )
    action = build_reversed_bspline_control_point_residuals(
        npz_path=npz_path,
        stats_path=stats_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        spline_degree=spline_degree,
        num_control_points=num_control_points,
        urdf_path=urdf_path,
    )
    return planning_result, action.astype(np.float32)


def main() -> None:
    args = build_parser().parse_args()
    if args.augment_copies <= 0:
        raise ValueError(f"--augment-copies must be positive, got {args.augment_copies}")

    search_dirs = [pathlib.Path(path) for path in (args.input_dirs if args.input_dirs else [args.input_dir])]
    npz_files = collect_npz_files(
        input_dir=args.input_dir,
        input_dirs=args.input_dirs,
    )
    if not npz_files:
        raise FileNotFoundError(f"No transition_*.npz files found under: {search_dirs}")
    stl_mapping = validate_stl_npz_mapping(
        npz_files=npz_files,
        input_dirs=search_dirs,
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

    radius_bounds = resolve_augmentation_bounds(
        base_value=args.radius_m,
        min_value=args.radius_m_min,
        max_value=args.radius_m_max,
        name="radius_m",
    )
    height_bounds = resolve_augmentation_bounds(
        base_value=args.height_m,
        min_value=args.height_m_min,
        max_value=args.height_m_max,
        name="height_m",
    )
    rng = np.random.default_rng(args.augment_seed)

    buffer = ReplayBuffer.create_empty_numpy()
    raw_mesh_points_cache: dict[str, np.ndarray] = {}
    workpiece_ids = []
    workpiece_local_ids = []
    workpiece_source_ids = []
    is_reversed_episode = []
    for npz_path in progress(npz_files, desc="build zarr episodes", unit="file"):
        stl_path = stl_mapping[str(npz_path)]
        workpiece_id, local_workpiece_id, workpiece_source_id = resolve_workpiece_metadata_from_npz(npz_path)
        if stl_path not in raw_mesh_points_cache:
            raw_mesh_points_cache[stl_path] = build_raw_mesh_points_world_m(
                stl_path=stl_path,
                num_mesh_sample_points=args.num_mesh_sample_points,
                stl_x_offset_mm=args.stl_x_offset_mm,
                use_poisson_disk=args.use_poisson_disk,
            )

        raw_mesh_points_world_m = raw_mesh_points_cache[stl_path]
        npz_data = np.load(npz_path)
        forward_start_tf = canonicalize_axis_symmetric_tcp_transform(
            np.asarray(npz_data["start_tf"], dtype=np.float32)
        )
        forward_goal_xyz_world = np.asarray(npz_data["end_xyz"], dtype=np.float32)
        forward_planning_result, forward_action = build_forward_invariants(
            npz_path=npz_path,
            stats_path=args.stats_path,
            norm_m=args.norm_m,
            trajectory_key=args.trajectory_key,
            target_steps=args.target_steps,
            spline_degree=args.spline_degree,
            num_control_points=args.num_control_points,
            urdf_path=args.urdf_path,
        )

        reversed_planning_result = None
        reversed_action = None
        reversed_start_tf = None
        reversed_goal_xyz_world = None
        if args.add_reversed_copy:
            reversed_start_tf = canonicalize_axis_symmetric_tcp_transform(
                np.asarray(npz_data["goal_tf"], dtype=np.float32)
            )
            reversed_goal_xyz_world = forward_start_tf[:3, 3].astype(np.float32)
            reversed_planning_result, reversed_action = build_reversed_invariants(
                npz_path=npz_path,
                stats_path=args.stats_path,
                norm_m=args.norm_m,
                trajectory_key=args.trajectory_key,
                target_steps=args.target_steps,
                spline_degree=args.spline_degree,
                num_control_points=args.num_control_points,
                urdf_path=args.urdf_path,
            )
        for copy_idx in range(args.augment_copies):
            radius_m, height_m = sample_crop_parameters(
                rng=rng,
                copy_idx=copy_idx,
                base_radius_m=args.radius_m,
                base_height_m=args.height_m,
                radius_bounds=radius_bounds,
                height_bounds=height_bounds,
            )
            point_cloud = build_normalized_point_cloud_from_geometry(
                raw_mesh_points_world_m=raw_mesh_points_world_m,
                start_tf_m=forward_start_tf,
                goal_xyz_world_m=forward_goal_xyz_world,
                norm_m=args.norm_m,
                radius_m=radius_m,
                height_m=height_m,
                num_output_points=args.num_output_points,
            )
            sample = assemble_sample(
                point_cloud=point_cloud,
                planning_result=forward_planning_result,
                action=forward_action,
            )
            buffer.add_episode(sample)
            workpiece_ids.append(workpiece_id)
            workpiece_local_ids.append(local_workpiece_id)
            workpiece_source_ids.append(workpiece_source_id)
            is_reversed_episode.append(0)
            if args.add_reversed_copy:
                reversed_point_cloud = build_normalized_point_cloud_from_geometry(
                    raw_mesh_points_world_m=raw_mesh_points_world_m,
                    start_tf_m=reversed_start_tf,
                    goal_xyz_world_m=reversed_goal_xyz_world,
                    norm_m=args.norm_m,
                    radius_m=radius_m,
                    height_m=height_m,
                    num_output_points=args.num_output_points,
                )
                reversed_sample = assemble_sample(
                    point_cloud=reversed_point_cloud,
                    planning_result=reversed_planning_result,
                    action=reversed_action,
                )
                buffer.add_episode(reversed_sample)
                workpiece_ids.append(workpiece_id)
                workpiece_local_ids.append(local_workpiece_id)
                workpiece_source_ids.append(workpiece_source_id)
                is_reversed_episode.append(1)

    buffer.update_meta(
        {
            "workpiece_ids": np.asarray(workpiece_ids, dtype=np.int64),
            "workpiece_local_ids": np.asarray(workpiece_local_ids, dtype=np.int64),
            "workpiece_source_ids": np.asarray(workpiece_source_ids, dtype=np.int64),
            "is_reversed_episode": np.asarray(is_reversed_episode, dtype=np.int64),
        }
    )

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
    print(f"workpiece_id_encoding: results=0-999, simple_results=1000+local_id")
    print(f"cached_meshes: {len(raw_mesh_points_cache)}")
    print(f"augment_copies: {args.augment_copies}")
    print(f"radius_bounds_m: {radius_bounds}")
    print(f"height_bounds_m: {height_bounds}")
    print(f"augment_seed: {args.augment_seed}")
    print(f"add_reversed_copy: {args.add_reversed_copy}")
    print(f"input_dirs: {[str(path) for path in search_dirs]}")
    print(f"stats_path: {args.stats_path}")
    print(f"saved_zarr: {output_zarr}")


if __name__ == "__main__":
    main()
