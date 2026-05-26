from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

DEFAULT_POINT_CLOUD_SIZE = 1024


@dataclass
class CapsuleCropResult:
    point_cloud: np.ndarray
    raw_mesh_points: np.ndarray
    cropped_points: np.ndarray


@dataclass
class TransitionPointCloudROI:
    point_cloud: np.ndarray
    cropped_local_points: np.ndarray
    cropped_world_points: np.ndarray
    goal_in_start_tcp_frame: np.ndarray
    start_in_start_tcp_frame: np.ndarray


@dataclass
class XYRadiusHeightCropResult:
    point_cloud: np.ndarray
    raw_mesh_points: np.ndarray
    cropped_points: np.ndarray


@dataclass
class NormalizedNPZROIResult:
    point_cloud: np.ndarray
    raw_mesh_points_world_m: np.ndarray
    cropped_points_world_m: np.ndarray
    cropped_points_start_tcp_m: np.ndarray
    start_tcp_transform_m: np.ndarray
    goal_xyz_world_m: np.ndarray


def _require_open3d() -> Any:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "open3d is required for STL mesh loading and point-cloud export. "
            "Install it with `pip install open3d` in your project environment."
        ) from exc
    return o3d


def load_stl_mesh(stl_path: str) -> Any:
    o3d = _require_open3d()
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {stl_path}")
    mesh.compute_vertex_normals()
    return mesh


def sample_mesh_surface(
    mesh: Any,
    num_points: int,
    use_poisson_disk: bool = False,
) -> np.ndarray:
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    if use_poisson_disk:
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points, dtype=np.float32)


def offset_points(points: np.ndarray, offset: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    return (points + offset).astype(np.float32)


def offset_transform(transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32).reshape(3)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape [4, 4], got {transform.shape}")
    shifted = transform.copy()
    shifted[:3, 3] += offset
    return shifted.astype(np.float32)


def point_to_segment_distance(points: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    segment = goal - start
    segment_norm_sq = np.dot(segment, segment)
    if segment_norm_sq < 1e-12:
        return np.linalg.norm(points - start[None, :], axis=1)

    rel_points = points - start[None, :]
    t = np.sum(rel_points * segment[None, :], axis=1) / segment_norm_sq
    t = np.clip(t, 0.0, 1.0)
    projection = start[None, :] + t[:, None] * segment[None, :]
    return np.linalg.norm(points - projection, axis=1)


def point_to_segment_distance_2d(points_xy: np.ndarray, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    segment = goal_xy - start_xy
    segment_norm_sq = np.dot(segment, segment)
    if segment_norm_sq < 1e-12:
        return np.linalg.norm(points_xy - start_xy[None, :], axis=1)

    rel_points = points_xy - start_xy[None, :]
    t = np.sum(rel_points * segment[None, :], axis=1) / segment_norm_sq
    t = np.clip(t, 0.0, 1.0)
    projection = start_xy[None, :] + t[:, None] * segment[None, :]
    return np.linalg.norm(points_xy - projection, axis=1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape [4, 4], got {transform.shape}")

    homogeneous_points = np.concatenate(
        [points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = homogeneous_points @ transform.T
    return transformed[:, :3].astype(np.float32)


def transform_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float32).reshape(1, 3)
    return transform_points(point, transform)[0]


def world_to_local_points(points: np.ndarray, tcp_transform: np.ndarray) -> np.ndarray:
    tcp_transform = np.asarray(tcp_transform, dtype=np.float32)
    tcp_inv = np.linalg.inv(tcp_transform)
    return transform_points(points, tcp_inv)


def world_to_local_point(point: np.ndarray, tcp_transform: np.ndarray) -> np.ndarray:
    tcp_transform = np.asarray(tcp_transform, dtype=np.float32)
    tcp_inv = np.linalg.inv(tcp_transform)
    return transform_point(point, tcp_inv)


def convert_points_mm_to_m(points_mm: np.ndarray) -> np.ndarray:
    points_mm = np.asarray(points_mm, dtype=np.float32)
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise ValueError(f"points_mm must have shape [N, 3], got {points_mm.shape}")
    return (points_mm / 1000.0).astype(np.float32)


def convert_point_mm_to_m(point_mm: np.ndarray) -> np.ndarray:
    point_mm = np.asarray(point_mm, dtype=np.float32).reshape(3)
    return (point_mm / 1000.0).astype(np.float32)


def convert_transform_mm_to_m(transform_mm: np.ndarray) -> np.ndarray:
    transform_mm = np.asarray(transform_mm, dtype=np.float32)
    if transform_mm.shape != (4, 4):
        raise ValueError(f"transform_mm must have shape [4, 4], got {transform_mm.shape}")
    transform_m = transform_mm.copy()
    transform_m[:3, 3] /= 1000.0
    return transform_m.astype(np.float32)


def load_transition_data_from_npz(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    required_keys = ("start_tf", "goal_tf", "start_xyz", "end_xyz")
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise KeyError(f"Missing required keys in npz file {npz_path}: {missing}")
    return {
        "start_tf": np.asarray(data["start_tf"], dtype=np.float32),
        "goal_tf": np.asarray(data["goal_tf"], dtype=np.float32),
        "start_xyz": np.asarray(data["start_xyz"], dtype=np.float32),
        "end_xyz": np.asarray(data["end_xyz"], dtype=np.float32),
    }


def crop_transition_roi_in_start_tcp_frame(
    points_world: np.ndarray,
    start_tcp_transform: np.ndarray,
    goal_xyz_world: np.ndarray,
    radius: float = 0.1,
    height_min: float = 0.0,
    height_max: float = 0.06,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"points_world must have shape [N, 3], got {points_world.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if height_max <= height_min:
        raise ValueError(
            f"height_max must be larger than height_min, got {height_min} and {height_max}"
        )

    local_points = world_to_local_points(points_world, start_tcp_transform)
    goal_local = world_to_local_point(goal_xyz_world, start_tcp_transform)
    start_local = np.zeros(3, dtype=np.float32)

    points_xy = local_points[:, :2]
    start_xy = start_local[:2]
    goal_xy = goal_local[:2]

    start_disk_dist = np.linalg.norm(points_xy - start_xy[None, :], axis=1)
    goal_disk_dist = np.linalg.norm(points_xy - goal_xy[None, :], axis=1)
    segment_dist = point_to_segment_distance_2d(points_xy, start_xy, goal_xy)
    height_mask = (local_points[:, 2] >= height_min) & (local_points[:, 2] <= height_max)

    planar_mask = (
        (start_disk_dist <= radius)
        | (goal_disk_dist <= radius)
        | (segment_dist <= radius)
    )
    roi_mask = planar_mask & height_mask

    cropped_local_points = local_points[roi_mask]
    cropped_world_points = points_world[roi_mask]
    if cropped_local_points.shape[0] == 0:
        raise ValueError(
            "Transition ROI contains zero points after applying XY capsule and height filters. "
            "Increase radius/height range or check the TCP transform."
        )
    return (
        cropped_local_points.astype(np.float32),
        cropped_world_points.astype(np.float32),
        goal_local.astype(np.float32),
    )


def extract_transition_point_cloud_roi(
    points_world: np.ndarray,
    start_tcp_transform: np.ndarray,
    goal_xyz_world: np.ndarray,
    radius: float = 0.1,
    height_min: float = 0.0,
    height_max: float = 0.06,
    num_output_points: int = DEFAULT_POINT_CLOUD_SIZE,
) -> TransitionPointCloudROI:
    cropped_local_points, cropped_world_points, goal_local = crop_transition_roi_in_start_tcp_frame(
        points_world=points_world,
        start_tcp_transform=start_tcp_transform,
        goal_xyz_world=goal_xyz_world,
        radius=radius,
        height_min=height_min,
        height_max=height_max,
    )
    sampled_local_points = sample_point_cloud_to_fixed_size(cropped_local_points, num_output_points)
    return TransitionPointCloudROI(
        point_cloud=sampled_local_points.astype(np.float32),
        cropped_local_points=cropped_local_points.astype(np.float32),
        cropped_world_points=cropped_world_points.astype(np.float32),
        goal_in_start_tcp_frame=goal_local.astype(np.float32),
        start_in_start_tcp_frame=np.zeros(3, dtype=np.float32),
    )


def crop_capsule_point_cloud(
    points: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    start = np.asarray(start, dtype=np.float32).reshape(3)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)
    distances = point_to_segment_distance(points, start, goal)
    mask = distances <= radius
    cropped = points[mask]
    if cropped.shape[0] == 0:
        raise ValueError(
            "Capsule ROI contains zero points. Increase the sampling density or radius."
        )
    return cropped.astype(np.float32)


def crop_xy_radius_height_point_cloud(
    points: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
    height: float,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")

    start = np.asarray(start, dtype=np.float32).reshape(3)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)

    points_xy = points[:, :2].astype(np.float32)
    start_xy = start[:2]
    goal_xy = goal[:2]
    planar_distances = point_to_segment_distance_2d(points_xy, start_xy, goal_xy)

    z_min = float(np.min(points[:, 2]))
    z_max = z_min + float(height)
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    xy_mask = planar_distances <= radius
    cropped = points[xy_mask & z_mask]
    if cropped.shape[0] == 0:
        raise ValueError(
            "XY-radius/height ROI contains zero points. "
            "Increase radius/height or check the mesh transform and coordinate units."
        )
    return cropped.astype(np.float32)


def farthest_point_sampling_numpy(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    n_points = points.shape[0]
    if n_points == 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    if n_points == 1:
        return np.repeat(points, num_points, axis=0)
    if n_points <= num_points:
        repeat_count = num_points - n_points
        if repeat_count == 0:
            return points.astype(np.float32)
        repeat_indices = np.random.choice(n_points, size=repeat_count, replace=True)
        padded = np.concatenate([points, points[repeat_indices]], axis=0)
        return padded.astype(np.float32)

    selected_indices = np.zeros(num_points, dtype=np.int64)
    distances = np.full(n_points, np.inf, dtype=np.float32)
    selected_indices[0] = np.random.randint(0, n_points)

    for i in range(1, num_points):
        last_point = points[selected_indices[i - 1]]
        current_dist = np.sum((points - last_point[None, :]) ** 2, axis=1)
        distances = np.minimum(distances, current_dist)
        selected_indices[i] = int(np.argmax(distances))

    return points[selected_indices].astype(np.float32)


def sample_point_cloud_to_fixed_size(
    points: np.ndarray,
    num_points: int = DEFAULT_POINT_CLOUD_SIZE,
) -> np.ndarray:
    return farthest_point_sampling_numpy(points, num_points)


def normalize_to_start_frame(
    points: np.ndarray,
    start: np.ndarray,
    rotation: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
) -> np.ndarray:
    centered = points - np.asarray(start, dtype=np.float32).reshape(1, 3)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        centered = centered @ rotation
    if radius is not None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        centered = centered / radius
    return centered.astype(np.float32)


def extract_capsule_roi_from_stl(
    stl_path: str,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
    num_output_points: int = DEFAULT_POINT_CLOUD_SIZE,
    num_mesh_sample_points: int = 100000,
    use_poisson_disk: bool = False,
    normalize: bool = False,
    rotation: Optional[np.ndarray] = None,
) -> CapsuleCropResult:
    mesh = load_stl_mesh(stl_path)
    raw_mesh_points = sample_mesh_surface(
        mesh=mesh,
        num_points=num_mesh_sample_points,
        use_poisson_disk=use_poisson_disk,
    )
    cropped_points = crop_capsule_point_cloud(
        points=raw_mesh_points,
        start=start,
        goal=goal,
        radius=radius,
    )
    sampled_points = sample_point_cloud_to_fixed_size(cropped_points, num_output_points)
    if normalize:
        sampled_points = normalize_to_start_frame(
            sampled_points,
            start=start,
            rotation=rotation,
            radius=radius,
        )

    return CapsuleCropResult(
        point_cloud=sampled_points.astype(np.float32),
        raw_mesh_points=raw_mesh_points.astype(np.float32),
        cropped_points=cropped_points.astype(np.float32),
    )


def extract_xy_radius_height_roi_from_stl(
    stl_path: str,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
    height: float,
    num_output_points: int = DEFAULT_POINT_CLOUD_SIZE,
    num_mesh_sample_points: int = 100000,
    use_poisson_disk: bool = False,
    normalize: bool = False,
    rotation: Optional[np.ndarray] = None,
) -> XYRadiusHeightCropResult:
    mesh = load_stl_mesh(stl_path)
    raw_mesh_points = sample_mesh_surface(
        mesh=mesh,
        num_points=num_mesh_sample_points,
        use_poisson_disk=use_poisson_disk,
    )
    cropped_points = crop_xy_radius_height_point_cloud(
        points=raw_mesh_points,
        start=start,
        goal=goal,
        radius=radius,
        height=height,
    )
    sampled_points = sample_point_cloud_to_fixed_size(cropped_points, num_output_points)
    if normalize:
        sampled_points = normalize_to_start_frame(
            sampled_points,
            start=start,
            rotation=rotation,
            radius=radius,
        )

    return XYRadiusHeightCropResult(
        point_cloud=sampled_points.astype(np.float32),
        raw_mesh_points=raw_mesh_points.astype(np.float32),
        cropped_points=cropped_points.astype(np.float32),
    )


def extract_normalized_xy_radius_height_roi_from_stl_and_npz(
    stl_path: str,
    npz_path: str,
    radius_m: float = 0.1,
    height_m: float = 0.1,
    norm_m: Optional[float] = None,
    num_output_points: int = DEFAULT_POINT_CLOUD_SIZE,
    num_mesh_sample_points: int = 100000,
    use_poisson_disk: bool = False,
    stl_x_offset_mm: float = 500.0,
) -> NormalizedNPZROIResult:
    if radius_m <= 0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")
    if height_m <= 0:
        raise ValueError(f"height_m must be positive, got {height_m}")
    if norm_m is None or norm_m <= 0:
        raise ValueError(f"norm_m must be provided by the caller and be positive, got {norm_m}")

    transition = load_transition_data_from_npz(npz_path)
    mesh = load_stl_mesh(stl_path)
    raw_mesh_points_mm = sample_mesh_surface(
        mesh=mesh,
        num_points=num_mesh_sample_points,
        use_poisson_disk=use_poisson_disk,
    )

    stl_offset_mm = np.array([stl_x_offset_mm, 0.0, 0.0], dtype=np.float32)
    raw_mesh_points_mm = offset_points(raw_mesh_points_mm, stl_offset_mm)

    raw_mesh_points_world_m = convert_points_mm_to_m(raw_mesh_points_mm)
    start_tcp_transform_m = transition["start_tf"].astype(np.float32)
    goal_xyz_world_m = transition["end_xyz"].astype(np.float32)

    cropped_points_world_m = crop_xy_radius_height_point_cloud(
        points=raw_mesh_points_world_m,
        start=start_tcp_transform_m[:3, 3],
        goal=goal_xyz_world_m,
        radius=radius_m,
        height=height_m,
    )
    cropped_points_start_tcp_m = world_to_local_points(
        cropped_points_world_m,
        start_tcp_transform_m,
    )
    normalized_points = sample_point_cloud_to_fixed_size(
        cropped_points_start_tcp_m,
        num_output_points,
    ) / norm_m

    return NormalizedNPZROIResult(
        point_cloud=normalized_points.astype(np.float32),
        raw_mesh_points_world_m=raw_mesh_points_world_m.astype(np.float32),
        cropped_points_world_m=cropped_points_world_m.astype(np.float32),
        cropped_points_start_tcp_m=cropped_points_start_tcp_m.astype(np.float32),
        start_tcp_transform_m=start_tcp_transform_m.astype(np.float32),
        goal_xyz_world_m=goal_xyz_world_m.astype(np.float32),
    )


def save_point_cloud_npy(points: np.ndarray, output_path: str) -> None:
    np.save(output_path, points.astype(np.float32))


def save_point_cloud_ply(points: np.ndarray, output_path: str) -> None:
    o3d = _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if not o3d.io.write_point_cloud(output_path, pcd):
        raise ValueError(f"Failed to save point cloud to {output_path}")


def parse_xyz_triplet(values: Tuple[float, float, float]) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"Expected 3 values for xyz, got {len(values)}")
    return np.asarray(values, dtype=np.float32)
