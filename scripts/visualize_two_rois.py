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
        description="Visualize an STL mesh together with two ROI point clouds."
    )
    parser.add_argument("--stl-path", type=str, required=True)
    parser.add_argument("--roi-a-npy", type=str, required=True)
    parser.add_argument("--roi-b-npy", type=str, required=True)
    parser.add_argument("--stl-x-offset", type=float, default=0.0)
    parser.add_argument("--stl-y-offset", type=float, default=0.0)
    parser.add_argument("--stl-z-offset", type=float, default=0.0)
    parser.add_argument("--start-a", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--goal-a", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--start-b", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--goal-b", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--start-goal-unit-scale", type=float, default=1.0)
    return parser


def build_sphere(o3d, center: np.ndarray, radius: float, color: tuple[float, float, float]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center.astype(np.float64))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere


def build_line_set(o3d, start: np.ndarray, goal: np.ndarray, color: tuple[float, float, float]):
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.stack([start, goal], axis=0)),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    line_set.colors = o3d.utility.Vector3dVector([list(color)])
    return line_set


def main() -> None:
    args = build_parser().parse_args()
    o3d = _require_open3d()

    mesh = o3d.io.read_triangle_mesh(args.stl_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {args.stl_path}")
    mesh.compute_vertex_normals()
    mesh.translate((args.stl_x_offset, args.stl_y_offset, args.stl_z_offset))
    mesh.paint_uniform_color((0.75, 0.75, 0.75))

    roi_a = np.load(args.roi_a_npy).astype(np.float64)
    roi_b = np.load(args.roi_b_npy).astype(np.float64)

    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(roi_a)
    pcd_a.paint_uniform_color((0.9, 0.15, 0.15))

    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(roi_b)
    pcd_b.paint_uniform_color((0.15, 0.35, 0.95))

    scale = args.start_goal_unit_scale
    start_a = np.asarray(args.start_a, dtype=np.float64) * scale
    goal_a = np.asarray(args.goal_a, dtype=np.float64) * scale
    start_b = np.asarray(args.start_b, dtype=np.float64) * scale
    goal_b = np.asarray(args.goal_b, dtype=np.float64) * scale

    combined = np.concatenate([roi_a, roi_b], axis=0)
    span = np.linalg.norm(combined.max(axis=0) - combined.min(axis=0))
    marker_radius = max(span * 0.008, 2.0)

    geometries = [
        mesh,
        pcd_a,
        pcd_b,
        build_line_set(o3d, start_a, goal_a, (0.8, 0.1, 0.1)),
        build_line_set(o3d, start_b, goal_b, (0.1, 0.3, 0.85)),
        build_sphere(o3d, start_a, marker_radius, (0.0, 0.8, 0.0)),
        build_sphere(o3d, goal_a, marker_radius, (0.8, 0.4, 0.0)),
        build_sphere(o3d, start_b, marker_radius, (0.0, 0.7, 0.7)),
        build_sphere(o3d, goal_b, marker_radius, (0.45, 0.0, 0.85)),
    ]

    o3d.visualization.draw_geometries(
        geometries,
        window_name="STL With Two ROI Results",
        width=1600,
        height=1000,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
