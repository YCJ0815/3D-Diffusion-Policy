import argparse
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.increment import (
    build_normalized_increment_trajectory,
    save_increment_stats,
)
from diffusion_policy_3d.common.input_data import load_planning_input_data
from diffusion_policy_3d.common.pointcloud_roi import (
    extract_normalized_xy_radius_height_roi_from_stl_and_npz,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert transition NPZ files into a zarr training dataset."
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
        required=True,
        help="Path to the STL file used to generate point clouds.",
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
        default="data/raw_data/results/job_000_increment_stats.npz",
        help="Path to the increment mean/std statistics file. Rebuilt on every run.",
    )
    parser.add_argument(
        "--force-rebuild-stats",
        action="store_true",
        help="Deprecated: increment statistics are rebuilt on every run.",
    )
    parser.add_argument(
        "--stats-std-eps",
        type=float,
        default=1e-6,
        help="Minimum std used when computing increment statistics.",
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
    parser.add_argument("--trajectory-key", type=str, default="q_plan")
    parser.add_argument("--target-steps", type=int, default=65)
    parser.add_argument("--urdf-path", type=str, default=None)
    parser.add_argument("--use-poisson-disk", action="store_true")
    return parser


def ensure_increment_stats(
    npz_files: list[pathlib.Path],
    stats_path: str,
    trajectory_key: str,
    target_steps: int,
    std_eps: float,
) -> pathlib.Path:
    resolved_stats_path = pathlib.Path(stats_path)
    stats = save_increment_stats(
        npz_paths=[str(path) for path in npz_files],
        output_path=str(resolved_stats_path),
        trajectory_key=trajectory_key,
        target_steps=target_steps,
        std_eps=std_eps,
    )
    print(f"built_stats: {resolved_stats_path}")
    print(f"stats_delta_count: {int(stats['count'])}")
    print(f"stats_mean: {stats['mean']}")
    print(f"stats_std: {stats['std']}")
    return resolved_stats_path


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
    planning_result = load_planning_input_data(
        npz_path=str(npz_path),
        norm=norm_m,
        urdf_path=urdf_path,
    )
    action = build_normalized_increment_trajectory(
        npz_path=str(npz_path),
        stats_path=stats_path,
        trajectory_key=trajectory_key,
        target_steps=target_steps,
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

    npz_files = sorted(pathlib.Path(args.input_dir).rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {args.input_dir}")

    stats_path = ensure_increment_stats(
        npz_files=npz_files,
        stats_path=args.stats_path,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
        std_eps=args.stats_std_eps,
    )

    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer.create_empty_numpy()
    for npz_path in npz_files:
        sample = build_sample(
            npz_path=npz_path,
            stl_path=args.stl_path,
            stats_path=str(stats_path),
            norm_m=args.norm_m,
            radius_m=args.radius_m,
            height_m=args.height_m,
            num_output_points=args.num_output_points,
            num_mesh_sample_points=args.num_mesh_sample_points,
            stl_x_offset_mm=args.stl_x_offset_mm,
            trajectory_key=args.trajectory_key,
            target_steps=args.target_steps,
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
    print(f"stats_path: {stats_path}")
    print(f"saved_zarr: {output_zarr}")


if __name__ == "__main__":
    main()
