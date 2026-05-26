import argparse
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.pointcloud_roi import _require_open3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize an STL mesh together with an ROI point cloud."
    )
    parser.add_argument("--stl-path", type=str, required=True, help="Path to the STL file.")
    parser.add_argument("--roi-npy", type=str, required=True, help="Path to the ROI point cloud .npy file.")
    parser.add_argument(
        "--stl-x-offset",
        type=float,
        default=0.0,
        help="Offset applied to STL vertices along x before visualization.",
    )
    parser.add_argument(
        "--stl-y-offset",
        type=float,
        default=0.0,
        help="Offset applied to STL vertices along y before visualization.",
    )
    parser.add_argument(
        "--stl-z-offset",
        type=float,
        default=0.0,
        help="Offset applied to STL vertices along z before visualization.",
    )
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional start point to render.",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional goal point to render.",
    )
    parser.add_argument(
        "--start-goal-unit-scale",
        type=float,
        default=1.0,
        help="Scale applied to start/goal coordinates before rendering.",
    )
    return parser


def build_colored_sphere(o3d, center: np.ndarray, radius: float, color: tuple[float, float, float]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center.astype(np.float64))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere


def main() -> None:
    args = build_parser().parse_args()
    o3d = _require_open3d()

    mesh = o3d.io.read_triangle_mesh(args.stl_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {args.stl_path}")
    mesh.compute_vertex_normals()
    mesh.translate((args.stl_x_offset, args.stl_y_offset, args.stl_z_offset))
    mesh.paint_uniform_color((0.7, 0.7, 0.7))

    roi_points = np.load(args.roi_npy).astype(np.float64)
    if roi_points.ndim != 2 or roi_points.shape[1] != 3:
        raise ValueError(f"ROI point cloud must have shape [N, 3], got {roi_points.shape}")

    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    roi_pcd.paint_uniform_color((0.9, 0.2, 0.2))

    geometries = [mesh, roi_pcd]

    if args.start is not None and args.goal is not None:
        start = np.asarray(args.start, dtype=np.float64) * args.start_goal_unit_scale
        goal = np.asarray(args.goal, dtype=np.float64) * args.start_goal_unit_scale
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.stack([start, goal], axis=0)),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        line_set.colors = o3d.utility.Vector3dVector([[0.1, 0.7, 0.1]])
        span = np.linalg.norm(roi_points.max(axis=0) - roi_points.min(axis=0))
        marker_radius = max(span * 0.01, 2.0)
        start_marker = build_colored_sphere(o3d, start, marker_radius, (0.1, 0.8, 0.1))
        goal_marker = build_colored_sphere(o3d, goal, marker_radius, (0.1, 0.3, 0.9))
        geometries.extend([line_set, start_marker, goal_marker])

    o3d.visualization.draw_geometries(
        geometries,
        window_name="STL ROI Visualization",
        width=1440,
        height=960,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
