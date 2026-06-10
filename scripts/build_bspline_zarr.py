import argparse
import hashlib
import json
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
    fit_quintic_bspline_to_npz_trajectory,
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
        "--jobs-root",
        type=str,
        default=None,
        help=(
            "Optional explicit root directory for regular workpiece STLs. "
            "When provided, `results/job_xxx/*.npz` will prefer "
            "`<jobs-root>/job_xxx/workpiece.stl`."
        ),
    )
    parser.add_argument(
        "--simple-jobs-root",
        type=str,
        default=None,
        help=(
            "Optional explicit root directory for simple workpiece STLs. "
            "When provided, `simple_results/job_xxx/*.npz` will prefer "
            "`<simple-jobs-root>/job_xxx/workpiece.stl`."
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
        "--reuse-stats-if-exists",
        action="store_true",
        help=(
            "Reuse --stats-path when it exists and its embedded build metadata "
            "matches the current inputs and B-spline parameters."
        ),
    )
    parser.add_argument(
        "--mesh-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for cached sampled STL point clouds. "
            "Cache keys include STL path, file stat, mesh sample count, offset, and Poisson setting."
        ),
    )
    parser.add_argument(
        "--bspline-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for cached per-NPZ B-spline fit and planning artifacts. "
            "This avoids re-fitting unchanged transition files across rebuilds."
        ),
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
        default=12,
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


CACHE_SCHEMA_VERSION = 1
STATS_METADATA_KEY = "build_metadata_json"
ARTIFACT_METADATA_KEY = "artifact_metadata_json"


def stable_json_dumps(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def hash_payload(value: dict) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_record(path: pathlib.Path) -> dict:
    resolved_path = path.expanduser().resolve()
    stat = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def optional_resolved_path(path: str | None) -> str | None:
    if path is None:
        return None
    return str(pathlib.Path(path).expanduser().resolve())


def metadata_from_npz(npz_data: np.lib.npyio.NpzFile, key: str) -> dict | None:
    if key not in npz_data.files:
        return None
    try:
        return json.loads(str(np.asarray(npz_data[key]).item()))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def cache_file_path(cache_dir: str | None, namespace: str, metadata: dict) -> pathlib.Path | None:
    if cache_dir is None:
        return None
    digest = hash_payload(metadata)
    path = pathlib.Path(cache_dir).expanduser().resolve() / namespace / f"{digest}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def mesh_cache_file_path(cache_dir: str | None, metadata: dict) -> pathlib.Path | None:
    if cache_dir is None:
        return None
    digest = hash_payload(metadata)
    path = pathlib.Path(cache_dir).expanduser().resolve() / "mesh_points" / f"{digest}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_mesh_cache_metadata(
    stl_path: str,
    num_mesh_sample_points: int,
    stl_x_offset_mm: float,
    use_poisson_disk: bool,
) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "mesh_points_world_m",
        "stl": file_record(pathlib.Path(stl_path)),
        "num_mesh_sample_points": int(num_mesh_sample_points),
        "stl_x_offset_mm": float(stl_x_offset_mm),
        "use_poisson_disk": bool(use_poisson_disk),
    }


def load_or_build_raw_mesh_points_world_m(
    stl_path: str,
    num_mesh_sample_points: int,
    stl_x_offset_mm: float,
    use_poisson_disk: bool,
    mesh_cache_dir: str | None,
    cache_counters: dict[str, int],
) -> np.ndarray:
    metadata = build_mesh_cache_metadata(
        stl_path=stl_path,
        num_mesh_sample_points=num_mesh_sample_points,
        stl_x_offset_mm=stl_x_offset_mm,
        use_poisson_disk=use_poisson_disk,
    )
    cache_path = mesh_cache_file_path(mesh_cache_dir, metadata)
    if cache_path is not None and cache_path.is_file():
        cache_counters["mesh_hits"] += 1
        return np.load(cache_path).astype(np.float32)

    cache_counters["mesh_misses"] += 1
    raw_mesh_points_world_m = build_raw_mesh_points_world_m(
        stl_path=stl_path,
        num_mesh_sample_points=num_mesh_sample_points,
        stl_x_offset_mm=stl_x_offset_mm,
        use_poisson_disk=use_poisson_disk,
    )
    if cache_path is not None:
        np.save(cache_path, raw_mesh_points_world_m.astype(np.float32))
    return raw_mesh_points_world_m.astype(np.float32)


def build_artifact_cache_metadata(
    npz_path: pathlib.Path,
    norm_m: float,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "bspline_forward_artifacts",
        "npz": file_record(npz_path),
        "norm_m": float(norm_m),
        "trajectory_key": str(trajectory_key),
        "target_steps": int(target_steps),
        "spline_degree": int(spline_degree),
        "num_control_points": int(num_control_points),
        "urdf_path": optional_resolved_path(urdf_path),
    }


def planning_to_cache_arrays(planning: PlanningInputData) -> dict[str, np.ndarray]:
    return {
        "planning_goal_position_world": planning.goal_position_world.astype(np.float32),
        "planning_goal_position_start_tcp_frame": planning.goal_position_start_tcp_frame.astype(np.float32),
        "planning_goal_position": planning.goal_position.astype(np.float32),
        "planning_goal_rotation": planning.goal_rotation.astype(np.float32),
        "planning_goal_direction_world": planning.goal_direction_world.astype(np.float32),
        "planning_goal_direction": planning.goal_direction.astype(np.float32),
        "planning_joint_names": np.asarray(planning.joint_names, dtype=str),
        "planning_joint_lower_limits": planning.joint_lower_limits.astype(np.float32),
        "planning_joint_upper_limits": planning.joint_upper_limits.astype(np.float32),
        "planning_first_joint_angles": planning.first_joint_angles.astype(np.float32),
        "planning_last_joint_angles": planning.last_joint_angles.astype(np.float32),
        "planning_first_joint_angles_normalized": planning.first_joint_angles_normalized.astype(np.float32),
        "planning_last_joint_angles_normalized": planning.last_joint_angles_normalized.astype(np.float32),
        "planning_trajectory_key": np.asarray(planning.trajectory_key, dtype=str),
    }


def planning_from_cache_arrays(cache_data: np.lib.npyio.NpzFile) -> PlanningInputData:
    return PlanningInputData(
        goal_position_world=np.asarray(cache_data["planning_goal_position_world"], dtype=np.float32),
        goal_position_start_tcp_frame=np.asarray(
            cache_data["planning_goal_position_start_tcp_frame"], dtype=np.float32
        ),
        goal_position=np.asarray(cache_data["planning_goal_position"], dtype=np.float32),
        goal_rotation=np.asarray(cache_data["planning_goal_rotation"], dtype=np.float32),
        goal_direction_world=np.asarray(cache_data["planning_goal_direction_world"], dtype=np.float32),
        goal_direction=np.asarray(cache_data["planning_goal_direction"], dtype=np.float32),
        joint_names=tuple(str(name) for name in np.asarray(cache_data["planning_joint_names"]).tolist()),
        joint_lower_limits=np.asarray(cache_data["planning_joint_lower_limits"], dtype=np.float32),
        joint_upper_limits=np.asarray(cache_data["planning_joint_upper_limits"], dtype=np.float32),
        first_joint_angles=np.asarray(cache_data["planning_first_joint_angles"], dtype=np.float32),
        last_joint_angles=np.asarray(cache_data["planning_last_joint_angles"], dtype=np.float32),
        first_joint_angles_normalized=np.asarray(
            cache_data["planning_first_joint_angles_normalized"], dtype=np.float32
        ),
        last_joint_angles_normalized=np.asarray(
            cache_data["planning_last_joint_angles_normalized"], dtype=np.float32
        ),
        trajectory_key=str(np.asarray(cache_data["planning_trajectory_key"]).item()),
    )


def load_or_build_bspline_artifacts(
    npz_path: pathlib.Path,
    bspline_cache_dir: str | None,
    norm_m: float,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
    cache_counters: dict[str, int],
) -> dict[str, np.ndarray | PlanningInputData]:
    metadata = build_artifact_cache_metadata(
        npz_path=npz_path,
        norm_m=norm_m,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        spline_degree=spline_degree,
        num_control_points=num_control_points,
        urdf_path=urdf_path,
    )
    cache_path = cache_file_path(bspline_cache_dir, "bspline_artifacts", metadata)
    if cache_path is not None and cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if metadata_from_npz(cached, ARTIFACT_METADATA_KEY) == metadata:
                    cache_counters["bspline_hits"] += 1
                    return {
                        "normalized_trajectory": np.asarray(cached["normalized_trajectory"], dtype=np.float32),
                        "control_points": np.asarray(cached["control_points"], dtype=np.float32),
                        "w_line": np.asarray(cached["w_line"], dtype=np.float32),
                        "delta_w": np.asarray(cached["delta_w"], dtype=np.float32),
                        "basis_matrix": np.asarray(cached["basis_matrix"], dtype=np.float32),
                        "knot_vector": np.asarray(cached["knot_vector"], dtype=np.float32),
                        "fitted_trajectory": np.asarray(cached["fitted_trajectory"], dtype=np.float32),
                        "start_tf": np.asarray(cached["start_tf"], dtype=np.float32),
                        "goal_tf": np.asarray(cached["goal_tf"], dtype=np.float32),
                        "end_xyz": np.asarray(cached["end_xyz"], dtype=np.float32),
                        "planning_result": planning_from_cache_arrays(cached),
                    }
        except (OSError, KeyError, ValueError):
            pass

    cache_counters["bspline_misses"] += 1
    fit_result = fit_quintic_bspline_to_npz_trajectory(
        npz_path=str(npz_path),
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        urdf_path=urdf_path,
        num_control_points=num_control_points,
        degree=spline_degree,
    )
    planning_result = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=norm_m,
        urdf_path=urdf_path,
    )
    npz_data = np.load(npz_path)
    artifacts: dict[str, np.ndarray | PlanningInputData] = {
        "normalized_trajectory": fit_result["normalized_trajectory"].astype(np.float32),
        "control_points": fit_result["control_points"].astype(np.float32),
        "w_line": fit_result["w_line"].astype(np.float32),
        "delta_w": fit_result["delta_w"].astype(np.float32),
        "basis_matrix": fit_result["basis_matrix"].astype(np.float32),
        "knot_vector": fit_result["knot_vector"].astype(np.float32),
        "fitted_trajectory": fit_result["fitted_trajectory"].astype(np.float32),
        "start_tf": canonicalize_axis_symmetric_tcp_transform(
            np.asarray(npz_data["start_tf"], dtype=np.float32)
        ),
        "goal_tf": canonicalize_axis_symmetric_tcp_transform(
            np.asarray(npz_data["goal_tf"], dtype=np.float32)
        ),
        "end_xyz": np.asarray(npz_data["end_xyz"], dtype=np.float32),
        "planning_result": planning_result,
    }
    if cache_path is not None:
        cache_arrays = {
            ARTIFACT_METADATA_KEY: np.asarray(stable_json_dumps(metadata), dtype=str),
            "normalized_trajectory": artifacts["normalized_trajectory"],
            "control_points": artifacts["control_points"],
            "w_line": artifacts["w_line"],
            "delta_w": artifacts["delta_w"],
            "basis_matrix": artifacts["basis_matrix"],
            "knot_vector": artifacts["knot_vector"],
            "fitted_trajectory": artifacts["fitted_trajectory"],
            "start_tf": artifacts["start_tf"],
            "goal_tf": artifacts["goal_tf"],
            "end_xyz": artifacts["end_xyz"],
            **planning_to_cache_arrays(planning_result),
        }
        np.savez(cache_path, **cache_arrays)
    return artifacts


def build_stats_metadata(
    npz_files: list[pathlib.Path],
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
    stats_std_eps: float,
) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "bspline_delta_w_stats",
        "npz_files": [file_record(path) for path in npz_files],
        "trajectory_key": str(trajectory_key),
        "target_steps": int(target_steps),
        "spline_degree": int(spline_degree),
        "num_control_points": int(num_control_points),
        "urdf_path": optional_resolved_path(urdf_path),
        "stats_std_eps": float(stats_std_eps),
    }


def load_delta_w_stats_file(stats_path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(stats_path, allow_pickle=False) as data:
        return {
            "mean": np.asarray(data["mean"], dtype=np.float32),
            "std": np.asarray(data["std"], dtype=np.float32),
            "var": np.asarray(data["var"], dtype=np.float32),
            "count": np.asarray(data["count"], dtype=np.int64),
            "basis_matrix": np.asarray(data["basis_matrix"], dtype=np.float32),
            "knot_vector": np.asarray(data["knot_vector"], dtype=np.float32),
        }


def ensure_delta_w_stats(
    npz_files: list[pathlib.Path],
    stats_path: str,
    reuse_stats_if_exists: bool,
    bspline_cache_dir: str | None,
    norm_m: float,
    trajectory_key: str,
    target_steps: int,
    spline_degree: int,
    num_control_points: int,
    urdf_path: str | None,
    stats_std_eps: float,
    cache_counters: dict[str, int],
) -> dict[str, np.ndarray]:
    output_path = pathlib.Path(stats_path)
    metadata = build_stats_metadata(
        npz_files=npz_files,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        spline_degree=spline_degree,
        num_control_points=num_control_points,
        urdf_path=urdf_path,
        stats_std_eps=stats_std_eps,
    )
    if reuse_stats_if_exists and output_path.is_file():
        with np.load(output_path, allow_pickle=False) as existing:
            if metadata_from_npz(existing, STATS_METADATA_KEY) == metadata:
                print(f"reused_stats: {output_path}")
                return load_delta_w_stats_file(output_path)

    free_delta_w_list = []
    basis_matrix = None
    knot_vector = None
    for npz_path in progress(npz_files, desc="fit bspline stats", unit="file"):
        artifacts = load_or_build_bspline_artifacts(
            npz_path=npz_path,
            bspline_cache_dir=bspline_cache_dir,
            norm_m=norm_m,
            trajectory_key=trajectory_key,
            target_steps=target_steps,
            spline_degree=spline_degree,
            num_control_points=num_control_points,
            urdf_path=urdf_path,
            cache_counters=cache_counters,
        )
        free_delta_w_list.append(
            np.asarray(artifacts["delta_w"], dtype=np.float32)[FREE_CONTROL_POINT_SLICE]
        )
        if basis_matrix is None:
            basis_matrix = np.asarray(artifacts["basis_matrix"], dtype=np.float32)
        if knot_vector is None:
            knot_vector = np.asarray(artifacts["knot_vector"], dtype=np.float32)

    free_delta_w = np.concatenate(free_delta_w_list, axis=0).astype(np.float32)
    std = free_delta_w.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(stats_std_eps)).astype(np.float32)
    stats = {
        "mean": free_delta_w.mean(axis=0).astype(np.float32),
        "std": std,
        "var": (std ** 2).astype(np.float32),
        "count": np.asarray(free_delta_w.shape[0], dtype=np.int64),
        "basis_matrix": np.asarray(basis_matrix, dtype=np.float32),
        "knot_vector": np.asarray(knot_vector, dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **stats,
        **{STATS_METADATA_KEY: np.asarray(stable_json_dumps(metadata), dtype=str)},
    )
    print(f"built_stats: {output_path}")
    return stats


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


def build_reversed_planning_data_from_forward(
    forward: PlanningInputData,
    start_tf: np.ndarray,
    goal_tf: np.ndarray,
    norm_m: float,
) -> PlanningInputData:
    reversed_start_tf = canonicalize_axis_symmetric_tcp_transform(np.asarray(goal_tf, dtype=np.float32))
    forward_start_tf = canonicalize_axis_symmetric_tcp_transform(np.asarray(start_tf, dtype=np.float32))
    reversed_goal_position_world = forward_start_tf[:3, 3].astype(np.float32)
    reversed_goal_position_start_tcp_frame = world_to_local_point(
        reversed_goal_position_world,
        reversed_start_tf,
    ).astype(np.float32)
    reversed_goal_position = (reversed_goal_position_start_tcp_frame / norm_m).astype(np.float32)
    reversed_goal_rotation = forward_start_tf[:3, :3].astype(np.float32)
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


def build_reversed_bspline_control_point_residuals_from_trajectory(
    normalized_trajectory: np.ndarray,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    spline_degree: int,
    num_control_points: int,
) -> np.ndarray:
    reversed_trajectory = np.asarray(normalized_trajectory, dtype=np.float32)[::-1].copy()
    control_points, _ = fit_quintic_bspline_control_points(
        normalized_trajectory=reversed_trajectory,
        num_control_points=num_control_points,
        degree=spline_degree,
    )
    linear_control_points = build_linear_control_points(
        start_state=reversed_trajectory[0],
        end_state=reversed_trajectory[-1],
        num_control_points=num_control_points,
    )
    delta_w = control_points.astype(np.float32) - linear_control_points.astype(np.float32)
    normalized_delta_w = normalize_delta_w(
        delta_w=delta_w,
        mean=stats_mean,
        std=stats_std,
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


def infer_source_kind_from_input_dirs(
    npz_path: pathlib.Path,
    input_dirs: list[pathlib.Path] | None = None,
) -> str:
    resolved_npz_path = npz_path.expanduser().resolve()
    if input_dirs:
        matches: list[pathlib.Path] = []
        for input_dir in input_dirs:
            resolved_input_dir = input_dir.expanduser().resolve()
            try:
                resolved_npz_path.relative_to(resolved_input_dir)
                matches.append(resolved_input_dir)
            except ValueError:
                continue
        if matches:
            matched_root = max(matches, key=lambda path: len(path.parts))
            if "simple" in matched_root.name.lower():
                return "simple"
            return "regular"

    for parent in resolved_npz_path.parents:
        if "simple" in parent.name.lower():
            return "simple"
    return "regular"


def encode_workpiece_id(local_workpiece_id: int, source_kind: str) -> int:
    if source_kind == "simple":
        return 1000 + int(local_workpiece_id)
    return int(local_workpiece_id)


def resolve_workpiece_metadata_from_npz(
    npz_path: pathlib.Path,
    input_dirs: list[pathlib.Path] | None = None,
) -> tuple[int, int, int]:
    job_name = resolve_job_name_from_npz(npz_path)
    if job_name is None:
        raise ValueError(f"Unable to infer workpiece id from NPZ path: {npz_path}")
    try:
        local_workpiece_id = int(job_name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid job/workpiece name format: {job_name}") from exc
    source_kind = infer_source_kind_from_input_dirs(npz_path=npz_path, input_dirs=input_dirs)
    encoded_workpiece_id = encode_workpiece_id(
        local_workpiece_id=local_workpiece_id,
        source_kind=source_kind,
    )
    workpiece_source_id = 1 if source_kind == "simple" else 0
    return encoded_workpiece_id, local_workpiece_id, workpiece_source_id


def resolve_stl_path_for_npz(
    npz_path: pathlib.Path,
    input_dirs: list[pathlib.Path],
    jobs_root: str | None,
    simple_jobs_root: str | None,
    fallback_stl_path: str | None,
) -> pathlib.Path:
    job_name = resolve_job_name_from_npz(npz_path)
    source_kind = infer_source_kind_from_input_dirs(npz_path=npz_path, input_dirs=input_dirs)
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
        if source_kind == "simple" and simple_jobs_root is not None:
            candidate_paths.append(pathlib.Path(simple_jobs_root).expanduser().resolve() / job_name / "workpiece.stl")
        elif source_kind != "simple" and jobs_root is not None:
            candidate_paths.append(pathlib.Path(jobs_root).expanduser().resolve() / job_name / "workpiece.stl")

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
    jobs_root: str | None,
    simple_jobs_root: str | None,
    fallback_stl_path: str | None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for npz_path in npz_files:
        stl_path = resolve_stl_path_for_npz(
            npz_path=npz_path,
            input_dirs=input_dirs,
            jobs_root=jobs_root,
            simple_jobs_root=simple_jobs_root,
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
        jobs_root=args.jobs_root,
        simple_jobs_root=args.simple_jobs_root,
        fallback_stl_path=args.stl_path,
    )

    cache_counters = {
        "mesh_hits": 0,
        "mesh_misses": 0,
        "bspline_hits": 0,
        "bspline_misses": 0,
    }
    stats = ensure_delta_w_stats(
        npz_files=npz_files,
        stats_path=args.stats_path,
        reuse_stats_if_exists=args.reuse_stats_if_exists,
        bspline_cache_dir=args.bspline_cache_dir,
        norm_m=args.norm_m,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
        spline_degree=args.spline_degree,
        num_control_points=args.num_control_points,
        urdf_path=args.urdf_path,
        stats_std_eps=args.stats_std_eps,
        cache_counters=cache_counters,
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
        workpiece_id, local_workpiece_id, workpiece_source_id = resolve_workpiece_metadata_from_npz(
            npz_path=npz_path,
            input_dirs=search_dirs,
        )
        if stl_path not in raw_mesh_points_cache:
            raw_mesh_points_cache[stl_path] = load_or_build_raw_mesh_points_world_m(
                stl_path=stl_path,
                num_mesh_sample_points=args.num_mesh_sample_points,
                stl_x_offset_mm=args.stl_x_offset_mm,
                use_poisson_disk=args.use_poisson_disk,
                mesh_cache_dir=args.mesh_cache_dir,
                cache_counters=cache_counters,
            )

        raw_mesh_points_world_m = raw_mesh_points_cache[stl_path]
        artifacts = load_or_build_bspline_artifacts(
            npz_path=npz_path,
            bspline_cache_dir=args.bspline_cache_dir,
            norm_m=args.norm_m,
            trajectory_key=args.trajectory_key,
            target_steps=args.target_steps,
            spline_degree=args.spline_degree,
            num_control_points=args.num_control_points,
            urdf_path=args.urdf_path,
            cache_counters=cache_counters,
        )
        forward_start_tf = np.asarray(artifacts["start_tf"], dtype=np.float32)
        forward_goal_xyz_world = np.asarray(artifacts["end_xyz"], dtype=np.float32)
        forward_planning_result = artifacts["planning_result"]
        forward_action = normalize_delta_w(
            delta_w=np.asarray(artifacts["delta_w"], dtype=np.float32),
            mean=np.asarray(stats["mean"], dtype=np.float32),
            std=np.asarray(stats["std"], dtype=np.float32),
        )[FREE_CONTROL_POINT_SLICE].astype(np.float32)

        reversed_planning_result = None
        reversed_action = None
        reversed_start_tf = None
        reversed_goal_xyz_world = None
        if args.add_reversed_copy:
            reversed_start_tf = np.asarray(artifacts["goal_tf"], dtype=np.float32)
            reversed_goal_xyz_world = forward_start_tf[:3, 3].astype(np.float32)
            reversed_planning_result = build_reversed_planning_data_from_forward(
                forward=forward_planning_result,
                start_tf=forward_start_tf,
                goal_tf=reversed_start_tf,
                norm_m=args.norm_m,
            )
            reversed_action = build_reversed_bspline_control_point_residuals_from_trajectory(
                normalized_trajectory=np.asarray(artifacts["normalized_trajectory"], dtype=np.float32),
                stats_mean=np.asarray(stats["mean"], dtype=np.float32),
                stats_std=np.asarray(stats["std"], dtype=np.float32),
                spline_degree=args.spline_degree,
                num_control_points=args.num_control_points,
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
    print(f"jobs_root_override: {args.jobs_root}")
    print(f"simple_jobs_root_override: {args.simple_jobs_root}")
    print(f"mesh_cache_dir: {args.mesh_cache_dir}")
    print(f"mesh_cache_hits: {cache_counters['mesh_hits']}")
    print(f"mesh_cache_misses: {cache_counters['mesh_misses']}")
    print(f"bspline_cache_dir: {args.bspline_cache_dir}")
    print(f"bspline_cache_hits: {cache_counters['bspline_hits']}")
    print(f"bspline_cache_misses: {cache_counters['bspline_misses']}")
    print(f"reuse_stats_if_exists: {args.reuse_stats_if_exists}")
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
