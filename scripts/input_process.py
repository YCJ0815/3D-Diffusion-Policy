import argparse
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.input_data import load_planning_input_data
from diffusion_policy_3d.common.pointcloud_roi import (
    extract_normalized_xy_radius_height_roi_from_stl_and_npz,
    save_point_cloud_ply,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrate normalized point cloud and planning data from STL + transition NPZ."
    )
    parser.add_argument("--stl-path", type=str, required=True, help="Path to the STL file.")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to the transition NPZ file.")
    parser.add_argument(
        "--output-npz",
        type=str,
        required=True,
        help="Path to save the integrated normalized input data as .npz.",
    )
    parser.add_argument(
        "--output-pointcloud-ply",
        type=str,
        default=None,
        help="Optional path to save the normalized point cloud as .ply.",
    )
    parser.add_argument("--radius-m", type=float, default=0.1, help="XY ROI radius in meters.")
    parser.add_argument("--height-m", type=float, default=0.1, help="ROI height in meters.")
    parser.add_argument(
        "--norm-m",
        type=float,
        required=True,
        help="Normalization divisor for both point cloud and goal position.",
    )
    parser.add_argument("--num-output-points", type=int, default=1024, help="Final point count.")
    parser.add_argument(
        "--num-mesh-sample-points",
        type=int,
        default=100000,
        help="Dense mesh sampling count before ROI cropping.",
    )
    parser.add_argument(
        "--stl-x-offset-mm",
        type=float,
        default=500.0,
        help="X offset applied to STL points before mm-to-m conversion.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Optional URDF path used to normalize joint angles.",
    )
    parser.add_argument(
        "--use-poisson-disk",
        action="store_true",
        help="Use poisson-disk mesh sampling instead of uniform sampling.",
    )
    parser.add_argument(
        "--output-meta-npz",
        type=str,
        default=None,
        help="Optional path to save extra intermediate arrays for debugging.",
    )
    return parser


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = build_parser().parse_args()

    pointcloud_result = extract_normalized_xy_radius_height_roi_from_stl_and_npz(
        stl_path=args.stl_path,
        npz_path=args.npz_path,
        radius_m=args.radius_m,
        height_m=args.height_m,
        norm_m=args.norm_m,
        num_output_points=args.num_output_points,
        num_mesh_sample_points=args.num_mesh_sample_points,
        use_poisson_disk=args.use_poisson_disk,
        stl_x_offset_mm=args.stl_x_offset_mm,
    )
    planning_result = load_planning_input_data(
        npz_path=args.npz_path,
        norm=args.norm_m,
        urdf_path=args.urdf_path,
    )

    output_npz = pathlib.Path(args.output_npz)
    ensure_parent(output_npz)
    np.savez(
        output_npz,
        point_cloud=pointcloud_result.point_cloud.astype(np.float32),
        goal_position=planning_result.goal_position.astype(np.float32),
        goal_direction=planning_result.goal_direction.astype(np.float32),
        first_joint_angles_normalized=planning_result.first_joint_angles_normalized.astype(np.float32),
        last_joint_angles_normalized=planning_result.last_joint_angles_normalized.astype(np.float32),
        goal_position_world=planning_result.goal_position_world.astype(np.float32),
        goal_position_start_tcp_frame=planning_result.goal_position_start_tcp_frame.astype(np.float32),
        goal_direction_world=planning_result.goal_direction_world.astype(np.float32),
        goal_rotation=planning_result.goal_rotation.astype(np.float32),
        joint_names=np.asarray(planning_result.joint_names, dtype="<U64"),
        joint_lower_limits=planning_result.joint_lower_limits.astype(np.float32),
        joint_upper_limits=planning_result.joint_upper_limits.astype(np.float32),
        trajectory_key=np.asarray(planning_result.trajectory_key),
    )

    if args.output_pointcloud_ply is not None:
        output_ply = pathlib.Path(args.output_pointcloud_ply)
        ensure_parent(output_ply)
        save_point_cloud_ply(pointcloud_result.point_cloud, str(output_ply))

    if args.output_meta_npz is not None:
        output_meta = pathlib.Path(args.output_meta_npz)
        ensure_parent(output_meta)
        np.savez(
            output_meta,
            raw_mesh_points_world_m=pointcloud_result.raw_mesh_points_world_m.astype(np.float32),
            cropped_points_world_m=pointcloud_result.cropped_points_world_m.astype(np.float32),
            cropped_points_start_tcp_m=pointcloud_result.cropped_points_start_tcp_m.astype(np.float32),
            start_tcp_transform_m=pointcloud_result.start_tcp_transform_m.astype(np.float32),
            goal_xyz_world_m=pointcloud_result.goal_xyz_world_m.astype(np.float32),
        )

    print(f"point_cloud: {pointcloud_result.point_cloud.shape}")
    print(f"goal_position: {planning_result.goal_position.shape}")
    print(f"goal_direction: {planning_result.goal_direction.shape}")
    print(f"first_joint_angles_normalized: {planning_result.first_joint_angles_normalized.shape}")
    print(f"last_joint_angles_normalized: {planning_result.last_joint_angles_normalized.shape}")
    print(f"saved_npz: {output_npz}")
    if args.output_pointcloud_ply is not None:
        print(f"saved_pointcloud_ply: {args.output_pointcloud_ply}")
    if args.output_meta_npz is not None:
        print(f"saved_meta_npz: {args.output_meta_npz}")


if __name__ == "__main__":
    main()
