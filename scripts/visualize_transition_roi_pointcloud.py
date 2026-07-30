import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.pointcloud_roi import (
    canonicalize_axis_symmetric_tcp_transform,
    convert_points_mm_to_m,
    crop_xy_radius_height_point_cloud,
    offset_points,
    sample_point_cloud_to_fixed_size,
    world_to_local_points,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render transition ROI selection and corresponding point cloud."
    )
    parser.add_argument("--npz-path", type=str, default=None)
    parser.add_argument("--stl-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--radius-m", type=float, default=0.1)
    parser.add_argument("--height-m", type=float, default=0.1)
    parser.add_argument("--norm-m", type=float, default=0.1)
    parser.add_argument("--num-mesh-sample-points", type=int, default=80000)
    parser.add_argument("--num-output-points", type=int, default=1024)
    parser.add_argument("--stl-x-offset-mm", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--view-elev", type=float, default=26.0)
    parser.add_argument("--view-azim", type=float, default=-54.0)
    parser.add_argument("--canvas-pixels", type=int, default=3840)
    parser.add_argument("--mesh-feature-edge-alpha", type=float, default=0.42)
    parser.add_argument("--mesh-feature-edge-width", type=float, default=0.45)
    parser.add_argument("--mesh-feature-angle-deg", type=float, default=20.0)
    parser.add_argument("--mesh-shading-strength", type=float, default=0.32)
    parser.add_argument("--mesh-only", action="store_true")
    return parser


def load_sampled_mesh_points(stl_path: str, count: int, stl_x_offset_mm: float) -> np.ndarray:
    mesh = trimesh.load_mesh(stl_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {stl_path}")
    points_mm, _ = trimesh.sample.sample_surface(mesh, count)
    points_mm = offset_points(
        points_mm.astype(np.float32),
        np.array([stl_x_offset_mm, 0.0, 0.0], dtype=np.float32),
    )
    return convert_points_mm_to_m(points_mm)


def load_mesh_world_m(stl_path: str, stl_x_offset_mm: float) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(stl_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {stl_path}")
    mesh = mesh.copy()
    mesh.vertices = offset_points(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.array([stl_x_offset_mm, 0.0, 0.0], dtype=np.float32),
    ) / 1000.0
    return mesh


def point_to_segment_distance_2d(points_xy: np.ndarray, start_xy: np.ndarray, goal_xy: np.ndarray):
    segment = goal_xy - start_xy
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return np.linalg.norm(points_xy - start_xy[None, :], axis=1)
    rel = points_xy - start_xy[None, :]
    t = np.clip(np.sum(rel * segment[None, :], axis=1) / denom, 0.0, 1.0)
    projection = start_xy[None, :] + t[:, None] * segment[None, :]
    return np.linalg.norm(points_xy - projection, axis=1)


def build_capsule_xy(start: np.ndarray, goal: np.ndarray, radius: float, num_arc_points: int = 96) -> np.ndarray:
    start_xy = start[:2]
    goal_xy = goal[:2]
    direction = goal_xy - start_xy
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        angles = np.linspace(0, 2 * np.pi, num_arc_points * 2, endpoint=False)
        return start_xy[None, :] + radius * np.c_[np.cos(angles), np.sin(angles)]

    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    normal_angle = np.arctan2(normal[1], normal[0])
    theta_goal = np.linspace(normal_angle, normal_angle - np.pi, num_arc_points)
    theta_start = np.linspace(normal_angle - np.pi, normal_angle - 2.0 * np.pi, num_arc_points)
    arc_goal = goal_xy[None, :] + radius * np.c_[np.cos(theta_goal), np.sin(theta_goal)]
    arc_start = start_xy[None, :] + radius * np.c_[np.cos(theta_start), np.sin(theta_start)]
    return np.vstack([arc_goal, arc_start])


def draw_capsule_prism(ax, start: np.ndarray, goal: np.ndarray, radius: float, z_min: float, z_max: float):
    xy = build_capsule_xy(start, goal, radius)
    bottom = np.c_[xy, np.full(xy.shape[0], z_min)]
    top = np.c_[xy, np.full(xy.shape[0], z_max)]

    side_faces = [
        [bottom[i], bottom[(i + 1) % len(bottom)], top[(i + 1) % len(top)], top[i]]
        for i in range(len(bottom))
    ]
    faces = [bottom[::-1], top] + side_faces
    fill_color = "#f59e0b"
    line_color = "#b45309"
    collection = Poly3DCollection(
        faces,
        facecolors=fill_color,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.16,
        zsort="average",
        zorder=20,
    )
    collection.set_sort_zpos(1e6)
    ax.add_collection3d(collection)

    closed = np.vstack([xy, xy[:1]])
    for z in (z_min, z_max):
        line = ax.plot(
            closed[:, 0],
            closed[:, 1],
            np.full(closed.shape[0], z),
            color=line_color,
            linewidth=1.7,
            alpha=0.9,
            zorder=30,
        )
        line[0].set_zorder(30)
    for idx in np.linspace(0, xy.shape[0] - 1, 24, dtype=int):
        line = ax.plot(
            [xy[idx, 0], xy[idx, 0]],
            [xy[idx, 1], xy[idx, 1]],
            [z_min, z_max],
            color=line_color,
            linewidth=0.9,
            alpha=0.62,
            zorder=30,
        )
        line[0].set_zorder(30)


def get_feature_edge_segments(mesh: trimesh.Trimesh, feature_angle_deg: float) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    segments = []

    if hasattr(mesh, "edges_unique") and hasattr(mesh, "edges_unique_inverse"):
        counts = np.bincount(mesh.edges_unique_inverse)
        boundary_edges = mesh.edges_unique[counts == 1]
        if len(boundary_edges) > 0:
            segments.append(vertices[boundary_edges])

    if hasattr(mesh, "face_adjacency_edges") and hasattr(mesh, "face_adjacency_angles"):
        threshold = np.deg2rad(feature_angle_deg)
        feature_edges = mesh.face_adjacency_edges[np.asarray(mesh.face_adjacency_angles) >= threshold]
        if len(feature_edges) > 0:
            segments.append(vertices[feature_edges])

    if not segments:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.concatenate(segments, axis=0).astype(np.float32)


def draw_transparent_mesh(
    ax,
    mesh: trimesh.Trimesh,
    feature_edge_alpha: float,
    feature_edge_width: float,
    feature_angle_deg: float,
    shading_strength: float,
):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    face_vertices = vertices[faces]
    normals = np.asarray(mesh.face_normals, dtype=np.float32)
    light = np.array([-0.35, -0.45, 0.82], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(normals @ light, 0.0, 1.0)
    base = np.array([0.62, 0.67, 0.73], dtype=np.float32)
    shade = 1.0 - shading_strength + shading_strength * diffuse
    facecolors = np.c_[np.clip(base[None, :] * shade[:, None], 0.0, 1.0), np.ones(len(faces))]
    collection = Poly3DCollection(
        face_vertices,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        alpha=1,
        zsort="average",
        zorder=0,
    )
    collection.set_sort_zpos(-1e6)
    ax.add_collection3d(collection)

    edge_segments = get_feature_edge_segments(mesh, feature_angle_deg)
    if len(edge_segments) > 0:
        edges = Line3DCollection(
            edge_segments,
            colors=[(0.18, 0.23, 0.28, feature_edge_alpha)],
            linewidths=feature_edge_width,
            zorder=5,
        )
        edges.set_sort_zpos(-9e5)
        ax.add_collection3d(edges)


def clean_transparent_axis(ax):
    ax.set_axis_off()
    ax.grid(False)
    ax.set_facecolor((1, 1, 1, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))


def make_canvas_transparent(fig):
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor((1, 1, 1, 0))


def render_pure_scene(
    output_path: pathlib.Path,
    mesh: trimesh.Trimesh,
    point_cloud: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
    view_elev: float,
    view_azim: float,
    canvas_pixels: int,
    mesh_feature_edge_alpha: float,
    mesh_feature_edge_width: float,
    mesh_feature_angle_deg: float,
    mesh_shading_strength: float,
):
    fig = plt.figure(figsize=(8, 8), dpi=max(1, int(canvas_pixels / 8)))
    make_canvas_transparent(fig)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    clean_transparent_axis(ax)
    ax.set_position([0, 0, 1, 1])

    draw_transparent_mesh(
        ax,
        mesh,
        feature_edge_alpha=mesh_feature_edge_alpha,
        feature_edge_width=mesh_feature_edge_width,
        feature_angle_deg=mesh_feature_angle_deg,
        shading_strength=mesh_shading_strength,
    )
    draw_capsule_prism(ax, start, goal, radius, z_min, z_max)

    point_show = point_cloud
    if point_show.shape[0] > 9000:
        point_show = point_show[np.random.choice(point_show.shape[0], 9000, replace=False)]
    point_artist = ax.scatter(
        point_show[:, 0],
        point_show[:, 1],
        point_show[:, 2],
        s=4.5,
        c="#dc2626",
        alpha=0.86,
        depthshade=False,
        zorder=40,
    )
    point_artist.set_zorder(40)
    point_artist.set_sort_zpos(1e6)

    scene_points = np.vstack(
        [
            np.asarray(mesh.vertices),
            point_show,
            np.c_[build_capsule_xy(start, goal, radius), np.full(192, z_min)],
            np.c_[build_capsule_xy(start, goal, radius), np.full(192, z_max)],
        ]
    )
    set_equal_axes(ax, scene_points, pad=0.10)
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_proj_type("ortho")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, pad_inches=0)
    plt.close(fig)


def render_mesh_only_scene(
    output_path: pathlib.Path,
    mesh: trimesh.Trimesh,
    view_elev: float,
    view_azim: float,
    canvas_pixels: int,
    mesh_feature_edge_alpha: float,
    mesh_feature_edge_width: float,
    mesh_feature_angle_deg: float,
    mesh_shading_strength: float,
):
    fig = plt.figure(figsize=(8, 8), dpi=max(1, int(canvas_pixels / 8)))
    make_canvas_transparent(fig)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    clean_transparent_axis(ax)
    ax.set_position([0, 0, 1, 1])
    draw_transparent_mesh(
        ax,
        mesh,
        feature_edge_alpha=mesh_feature_edge_alpha,
        feature_edge_width=mesh_feature_edge_width,
        feature_angle_deg=mesh_feature_angle_deg,
        shading_strength=mesh_shading_strength,
    )
    set_equal_axes(ax, np.asarray(mesh.vertices), pad=0.10)
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_proj_type("ortho")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, pad_inches=0)
    plt.close(fig)


def set_equal_axes(ax, points: np.ndarray, pad: float = 0.04):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = float(np.max(maxs - mins))
    span = max(span, 1e-3) * (1.0 + pad)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - span / 2, c + span / 2)


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    mesh_world = load_mesh_world_m(args.stl_path, args.stl_x_offset_mm)
    output_path = pathlib.Path(args.output)
    if args.mesh_only:
        render_mesh_only_scene(
            output_path=output_path,
            mesh=mesh_world,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
            canvas_pixels=args.canvas_pixels,
            mesh_feature_edge_alpha=args.mesh_feature_edge_alpha,
            mesh_feature_edge_width=args.mesh_feature_edge_width,
            mesh_feature_angle_deg=args.mesh_feature_angle_deg,
            mesh_shading_strength=args.mesh_shading_strength,
        )
        print(output_path)
        print("mesh_only=True")
        return

    if args.npz_path is None:
        raise ValueError("--npz-path is required unless --mesh-only is set.")

    transition = np.load(args.npz_path)
    start_tf = canonicalize_axis_symmetric_tcp_transform(transition["start_tf"].astype(np.float32))
    start = start_tf[:3, 3]
    goal = transition["end_xyz"].astype(np.float32)
    tcp_points = transition["tcp_points"].astype(np.float32)

    raw_world = load_sampled_mesh_points(
        stl_path=args.stl_path,
        count=args.num_mesh_sample_points,
        stl_x_offset_mm=args.stl_x_offset_mm,
    )
    cropped_world = crop_xy_radius_height_point_cloud(
        points=raw_world,
        start=start,
        goal=goal,
        radius=args.radius_m,
        height=args.height_m,
    )
    cropped_local = world_to_local_points(cropped_world, start_tf)
    sampled_local = sample_point_cloud_to_fixed_size(cropped_local, args.num_output_points)
    normalized_local = sampled_local / args.norm_m

    z_min = float(raw_world[:, 2].min())
    z_max = z_min + args.height_m

    render_pure_scene(
        output_path=output_path,
        mesh=mesh_world,
        point_cloud=cropped_world,
        start=start,
        goal=goal,
        radius=args.radius_m,
        z_min=z_min,
        z_max=z_max,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        canvas_pixels=args.canvas_pixels,
        mesh_feature_edge_alpha=args.mesh_feature_edge_alpha,
        mesh_feature_edge_width=args.mesh_feature_edge_width,
        mesh_feature_angle_deg=args.mesh_feature_angle_deg,
        mesh_shading_strength=args.mesh_shading_strength,
    )
    print(output_path)
    print(f"raw_points={raw_world.shape[0]} cropped_roi_points={cropped_world.shape[0]}")


if __name__ == "__main__":
    main()
