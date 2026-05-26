import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.pointcloud_roi import (
    extract_capsule_roi_from_stl,
    extract_xy_radius_height_roi_from_stl,
    parse_xyz_triplet,
    save_point_cloud_npy,
    save_point_cloud_ply,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract an ROI point cloud from an STL mesh."
    )
    parser.add_argument("--stl-path", type=str, required=True, help="Path to the STL file.")
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Start point in world coordinates.",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Goal point in world coordinates.",
    )
    parser.add_argument("--radius", type=float, required=True, help="Capsule radius.")
    parser.add_argument(
        "--roi-mode",
        type=str,
        default="capsule",
        choices=("capsule", "xy_radius_height"),
        help="ROI extraction mode.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="ROI height used by xy_radius_height mode.",
    )
    parser.add_argument(
        "--num-output-points",
        type=int,
        default=2048,
        help="Final point count after downsampling.",
    )
    parser.add_argument(
        "--num-mesh-sample-points",
        type=int,
        default=100000,
        help="Number of dense mesh surface samples before cropping.",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        required=True,
        help="Path to save the final Nx3 point cloud as .npy.",
    )
    parser.add_argument(
        "--output-ply",
        type=str,
        default=None,
        help="Optional path to save the final Nx3 point cloud as .ply.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the output into the start-centered local frame and divide by radius.",
    )
    parser.add_argument(
        "--use-poisson-disk",
        action="store_true",
        help="Use poisson disk sampling on the STL mesh instead of uniform surface sampling.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_xyz_triplet(tuple(args.start))
    goal = parse_xyz_triplet(tuple(args.goal))

    if args.roi_mode == "capsule":
        result = extract_capsule_roi_from_stl(
            stl_path=args.stl_path,
            start=start,
            goal=goal,
            radius=args.radius,
            num_output_points=args.num_output_points,
            num_mesh_sample_points=args.num_mesh_sample_points,
            use_poisson_disk=args.use_poisson_disk,
            normalize=args.normalize,
        )
    else:
        if args.height is None:
            parser.error("--height is required when --roi-mode xy_radius_height is used.")
        result = extract_xy_radius_height_roi_from_stl(
            stl_path=args.stl_path,
            start=start,
            goal=goal,
            radius=args.radius,
            height=args.height,
            num_output_points=args.num_output_points,
            num_mesh_sample_points=args.num_mesh_sample_points,
            use_poisson_disk=args.use_poisson_disk,
            normalize=args.normalize,
        )

    output_npy = pathlib.Path(args.output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    save_point_cloud_npy(result.point_cloud, str(output_npy))

    if args.output_ply is not None:
        output_ply = pathlib.Path(args.output_ply)
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        save_point_cloud_ply(result.point_cloud, str(output_ply))

    print(f"raw_mesh_points: {result.raw_mesh_points.shape}")
    print(f"cropped_points: {result.cropped_points.shape}")
    print(f"final_point_cloud: {result.point_cloud.shape}")
    print(f"saved_npy: {output_npy}")
    if args.output_ply is not None:
        print(f"saved_ply: {args.output_ply}")


if __name__ == "__main__":
    main()
