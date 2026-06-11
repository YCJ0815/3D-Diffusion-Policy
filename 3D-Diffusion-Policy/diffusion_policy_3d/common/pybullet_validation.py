from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from diffusion_policy_3d.common.bspline import (
    load_delta_w_stats,
    reconstruct_trajectory_from_normalized_free_residual,
    unnormalize_joint_trajectory_with_urdf_limits,
)
from diffusion_policy_3d.common.increment import load_increment_stats
from diffusion_policy_3d.common.input_data import (
    _default_urdf_path,
    _load_joint_limits_from_urdf,
)


def _episode_bounds(episode_ends: np.ndarray, episode_idx: int) -> tuple[int, int]:
    end_idx = int(episode_ends[episode_idx])
    start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    return start_idx, end_idx


def _resolve_stats_mode(stats_path: str) -> str:
    stats = np.load(stats_path)
    if "basis_matrix" in stats.files or "knot_vector" in stats.files:
        return "bspline"
    return "increment"


def _resolve_workpiece_stl_path(
    workpiece_id: int,
    jobs_root: str,
    simple_jobs_root: str | None,
    simple_workpiece_id_offset: int,
    job_name_template: str,
    workpiece_filename: str,
) -> Path:
    return _resolve_workpiece_file_path(
        workpiece_id=workpiece_id,
        jobs_root=jobs_root,
        simple_jobs_root=simple_jobs_root,
        simple_workpiece_id_offset=simple_workpiece_id_offset,
        job_name_template=job_name_template,
        filename=workpiece_filename,
        file_label="STL",
    )


def _resolve_workpiece_file_path(
    workpiece_id: int,
    jobs_root: str,
    simple_jobs_root: str | None,
    simple_workpiece_id_offset: int,
    job_name_template: str,
    filename: str,
    file_label: str,
) -> Path:
    workpiece_id = int(workpiece_id)
    resolved_jobs_root = Path(jobs_root).expanduser().resolve()
    local_workpiece_id = workpiece_id
    if simple_jobs_root is not None and workpiece_id >= int(simple_workpiece_id_offset):
        resolved_jobs_root = Path(simple_jobs_root).expanduser().resolve()
        local_workpiece_id = workpiece_id - int(simple_workpiece_id_offset)
    job_name = job_name_template.format(workpiece_id=int(local_workpiece_id))
    file_path = resolved_jobs_root / job_name / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Workpiece {file_label} not found for workpiece_id={workpiece_id}: {file_path}")
    return file_path


def _resolve_mesh_filename(filename: str, package_roots: list[Path]) -> str:
    if not filename.startswith("package://"):
        return filename

    package_rel = filename[len("package://"):]
    package_name, _, rel_path = package_rel.partition("/")
    for root in package_roots:
        candidate = root / package_name / rel_path
        if candidate.is_file():
            return str(candidate)
        candidate = root / rel_path
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Unable to resolve URDF mesh URI `{filename}` with package roots: "
        f"{[str(root) for root in package_roots]}"
    )


def _expand_urdf_package_roots(urdf_path: str, package_roots: list[str]) -> list[Path]:
    urdf_parent = Path(urdf_path).expanduser().resolve().parent
    candidates: list[Path] = []
    for root in package_roots:
        root_path = Path(root).expanduser()
        if root_path.is_absolute():
            candidates.append(root_path.resolve())
        else:
            candidates.extend([
                root_path.resolve(),
                (urdf_parent / root_path).resolve(),
                (urdf_parent.parent / root_path).resolve(),
            ])
    candidates.extend([
        urdf_parent.resolve(),
        (urdf_parent / "robot-model").resolve(),
        urdf_parent.parent.resolve(),
    ])

    unique_roots = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_roots.append(candidate)
    return unique_roots


def _rewrite_urdf_package_uris(urdf_path: str, package_roots: list[str]) -> str:
    roots = _expand_urdf_package_roots(urdf_path, package_roots)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None or not filename.startswith("package://"):
            continue
        mesh.set("filename", _resolve_mesh_filename(filename, roots))
        changed = True

    if not changed:
        return urdf_path

    tmp_dir = Path(tempfile.mkdtemp(prefix="dp3_pybullet_urdf_"))
    resolved_path = tmp_dir / Path(urdf_path).name
    tree.write(resolved_path, encoding="utf-8", xml_declaration=True)
    return str(resolved_path)


def _parse_float_vector(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    parsed = [float(item) for item in value.split()]
    return np.asarray(parsed, dtype=np.float32)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(item) for item in rpy.reshape(3)]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _apply_origin_transform(points: np.ndarray, origin_elem) -> np.ndarray:
    if origin_elem is None:
        return points.astype(np.float32)
    xyz = _parse_float_vector(origin_elem.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_float_vector(origin_elem.get("rpy"), (0.0, 0.0, 0.0))
    rotation = _rpy_to_matrix(rpy)
    return (points @ rotation.T + xyz.reshape(1, 3)).astype(np.float32)


def _select_deterministic_surface_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices].astype(np.float32)


@dataclass
class SDFGrid:
    sdf: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    out_of_bounds_value_m: float | None = None

    @classmethod
    def load(cls, path: Path, out_of_bounds_value_m: float | None = None) -> "SDFGrid":
        data = np.load(path)
        missing = [key for key in ("sdf", "x", "y", "z") if key not in data.files]
        if missing:
            raise KeyError(f"SDF file {path} is missing required keys: {missing}")
        return cls(
            sdf=np.asarray(data["sdf"], dtype=np.float32),
            x=np.asarray(data["x"], dtype=np.float32),
            y=np.asarray(data["y"], dtype=np.float32),
            z=np.asarray(data["z"], dtype=np.float32),
            out_of_bounds_value_m=out_of_bounds_value_m,
        )

    def query(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if self.sdf.shape != (self.x.shape[0], self.y.shape[0], self.z.shape[0]):
            raise ValueError(
                "SDF grid shape must match x/y/z axes, got "
                f"sdf={self.sdf.shape}, axes={(self.x.shape[0], self.y.shape[0], self.z.shape[0])}"
            )
        valid = (
            (points[:, 0] >= self.x[0]) & (points[:, 0] <= self.x[-1]) &
            (points[:, 1] >= self.y[0]) & (points[:, 1] <= self.y[-1]) &
            (points[:, 2] >= self.z[0]) & (points[:, 2] <= self.z[-1])
        )
        values = np.full(points.shape[0], np.nan, dtype=np.float32)
        if self.out_of_bounds_value_m is not None:
            values[~valid] = float(self.out_of_bounds_value_m)
        if not np.any(valid):
            return values

        valid_points = points[valid]
        ix0, ix1, tx = self._axis_indices(self.x, valid_points[:, 0])
        iy0, iy1, ty = self._axis_indices(self.y, valid_points[:, 1])
        iz0, iz1, tz = self._axis_indices(self.z, valid_points[:, 2])

        c000 = self.sdf[ix0, iy0, iz0]
        c100 = self.sdf[ix1, iy0, iz0]
        c010 = self.sdf[ix0, iy1, iz0]
        c110 = self.sdf[ix1, iy1, iz0]
        c001 = self.sdf[ix0, iy0, iz1]
        c101 = self.sdf[ix1, iy0, iz1]
        c011 = self.sdf[ix0, iy1, iz1]
        c111 = self.sdf[ix1, iy1, iz1]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        values[valid] = (c0 * (1.0 - tz) + c1 * tz).astype(np.float32)
        return values

    @staticmethod
    def _axis_indices(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        upper = np.searchsorted(axis, values, side="right")
        upper = np.clip(upper, 1, axis.shape[0] - 1)
        lower = upper - 1
        span = axis[upper] - axis[lower]
        weights = np.divide(
            values - axis[lower],
            span,
            out=np.zeros_like(values, dtype=np.float32),
            where=span > 0,
        )
        return lower.astype(np.int64), upper.astype(np.int64), weights.astype(np.float32)


@dataclass
class PyBulletValidationConfig:
    enabled: bool = False
    stats_path: str | None = None
    stats_mode: str = "auto"
    jobs_root: str = "data/raw_data/jobs"
    simple_jobs_root: str | None = "data/raw_data/simple_jobs"
    simple_workpiece_id_offset: int = 1000
    job_name_template: str = "job_{workpiece_id:03d}"
    workpiece_filename: str = "workpiece.stl"
    urdf_path: str | None = None
    urdf_package_roots: tuple[str, ...] = ("config/robot-model",)
    tcp_link_name: str = "tool0"
    stl_x_offset_m: float = 0.5
    collision_distance_threshold: float = 0.0
    interpolate_for_collision: bool = True
    max_joint_step_rad: float = 0.01
    min_interpolated_steps_per_segment: int = 1
    goal_position_norm_m: float = 0.1
    goal_tolerance_m: float = 0.01
    num_control_points: int = 12
    spline_degree: int = 5
    target_steps: int = 64
    max_episodes: int | None = None
    sdf_filename: str = "workpiece_sdf.npz"
    sdf_required: bool = True
    robot_surface_points_per_link: int = 256
    sdf_out_of_bounds_value_m: float | None = None
    log_legacy_pybullet_metrics: bool = True

    @classmethod
    def from_omegaconf(cls, cfg) -> "PyBulletValidationConfig":
        sdf_out_of_bounds_value_m = cfg.get("sdf_out_of_bounds_value_m", None)
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            stats_path=cfg.get("stats_path"),
            stats_mode=str(cfg.get("stats_mode", "auto")),
            jobs_root=str(cfg.get("jobs_root", "data/raw_data/jobs")),
            simple_jobs_root=cfg.get("simple_jobs_root", "data/raw_data/simple_jobs"),
            simple_workpiece_id_offset=int(cfg.get("simple_workpiece_id_offset", 1000)),
            job_name_template=str(cfg.get("job_name_template", "job_{workpiece_id:03d}")),
            workpiece_filename=str(cfg.get("workpiece_filename", "workpiece.stl")),
            urdf_path=cfg.get("urdf_path"),
            urdf_package_roots=tuple(cfg.get("urdf_package_roots", ["config/robot-model"])),
            tcp_link_name=str(cfg.get("tcp_link_name", "tool0")),
            stl_x_offset_m=float(cfg.get("stl_x_offset_m", 0.5)),
            collision_distance_threshold=float(cfg.get("collision_distance_threshold", 0.0)),
            interpolate_for_collision=bool(cfg.get("interpolate_for_collision", True)),
            max_joint_step_rad=float(cfg.get("max_joint_step_rad", 0.01)),
            min_interpolated_steps_per_segment=int(cfg.get("min_interpolated_steps_per_segment", 1)),
            goal_position_norm_m=float(cfg.get("goal_position_norm_m", 0.1)),
            goal_tolerance_m=float(cfg.get("goal_tolerance_m", 0.01)),
            num_control_points=int(cfg.get("num_control_points", 12)),
            spline_degree=int(cfg.get("spline_degree", 5)),
            target_steps=int(cfg.get("target_steps", 64)),
            max_episodes=cfg.get("max_episodes"),
            sdf_filename=str(cfg.get("sdf_filename", "workpiece_sdf.npz")),
            sdf_required=bool(cfg.get("sdf_required", True)),
            robot_surface_points_per_link=int(cfg.get("robot_surface_points_per_link", 256)),
            sdf_out_of_bounds_value_m=(
                None if sdf_out_of_bounds_value_m is None else float(sdf_out_of_bounds_value_m)
            ),
            log_legacy_pybullet_metrics=bool(cfg.get("log_legacy_pybullet_metrics", True)),
        )


class PyBulletCollisionValidator:
    def __init__(self, cfg: PyBulletValidationConfig):
        self.cfg = cfg
        if not self.cfg.enabled:
            raise ValueError("PyBulletCollisionValidator should only be created when enabled=True.")
        if not self.cfg.stats_path:
            raise ValueError("training.pybullet_eval.stats_path is required when pybullet validation is enabled.")

        try:
            import pybullet as pb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pybullet validation was enabled, but `pybullet` is not installed in the current environment."
            ) from exc

        self.pb = pb
        self.client_id = pb.connect(pb.DIRECT)
        resolved_urdf_path = self.cfg.urdf_path if self.cfg.urdf_path is not None else str(_default_urdf_path())
        resolved_urdf_path = _rewrite_urdf_package_uris(
            urdf_path=resolved_urdf_path,
            package_roots=list(self.cfg.urdf_package_roots),
        )
        self.resolved_urdf_path = resolved_urdf_path
        self.robot_id = pb.loadURDF(
            resolved_urdf_path,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.joint_names, self.joint_lower_limits, self.joint_upper_limits = _load_joint_limits_from_urdf(
            self.cfg.urdf_path if self.cfg.urdf_path is not None else str(_default_urdf_path())
        )
        self.revolute_joint_indices = []
        self.link_name_to_index = {}
        for joint_idx in range(pb.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_info = pb.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client_id)
            joint_type = joint_info[2]
            child_link_name = joint_info[12].decode("utf-8")
            self.link_name_to_index[child_link_name] = joint_idx
            if joint_type == pb.JOINT_REVOLUTE:
                self.revolute_joint_indices.append(joint_idx)
        if len(self.revolute_joint_indices) != len(self.joint_names):
            raise ValueError(
                f"URDF revolute joint count mismatch: parsed {len(self.joint_names)} from XML, "
                f"loaded {len(self.revolute_joint_indices)} in pybullet."
            )
        if self.cfg.tcp_link_name not in self.link_name_to_index:
            raise KeyError(
                f"TCP link `{self.cfg.tcp_link_name}` not found in URDF. "
                f"Available links: {sorted(self.link_name_to_index.keys())}"
            )
        self.tcp_link_index = self.link_name_to_index[self.cfg.tcp_link_name]
        self.stats_mode = (
            _resolve_stats_mode(self.cfg.stats_path)
            if self.cfg.stats_mode == "auto"
            else self.cfg.stats_mode
        )
        if self.stats_mode not in ("increment", "bspline"):
            raise ValueError(f"Unsupported pybullet stats mode: {self.stats_mode}")
        if self.stats_mode == "bspline":
            self.stats_mean, self.stats_std = load_delta_w_stats(self.cfg.stats_path)
        else:
            self.stats_mean, self.stats_std = load_increment_stats(self.cfg.stats_path)
        if self.cfg.robot_surface_points_per_link <= 0:
            raise ValueError(
                "training.pybullet_eval.robot_surface_points_per_link must be positive, "
                f"got {self.cfg.robot_surface_points_per_link}"
            )
        self.workpiece_cache: dict[int, int] = {}
        self.sdf_cache: dict[int, SDFGrid] = {}
        self.robot_surface_points_by_link = self._build_robot_collision_surface_points(
            resolved_urdf_path=self.resolved_urdf_path,
            points_per_link=self.cfg.robot_surface_points_per_link,
        )
        if self.cfg.max_joint_step_rad <= 0:
            raise ValueError(
                f"training.pybullet_eval.max_joint_step_rad must be positive, got {self.cfg.max_joint_step_rad}"
            )
        if self.cfg.min_interpolated_steps_per_segment <= 0:
            raise ValueError(
                "training.pybullet_eval.min_interpolated_steps_per_segment must be positive, "
                f"got {self.cfg.min_interpolated_steps_per_segment}"
            )

    def close(self) -> None:
        if getattr(self, "client_id", None) is not None:
            self.pb.disconnect(self.client_id)
            self.client_id = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _unnormalize_joint_state(self, normalized_joint_state: np.ndarray) -> np.ndarray:
        return unnormalize_joint_trajectory_with_urdf_limits(
            normalized_trajectory=np.asarray(normalized_joint_state, dtype=np.float32).reshape(1, -1),
            lower_limits=self.joint_lower_limits,
            upper_limits=self.joint_upper_limits,
        )[0]

    def _load_workpiece_body(self, workpiece_id: int) -> int:
        workpiece_id = int(workpiece_id)
        if workpiece_id in self.workpiece_cache:
            return self.workpiece_cache[workpiece_id]
        stl_path = _resolve_workpiece_stl_path(
            workpiece_id=workpiece_id,
            jobs_root=self.cfg.jobs_root,
            simple_jobs_root=self.cfg.simple_jobs_root,
            simple_workpiece_id_offset=self.cfg.simple_workpiece_id_offset,
            job_name_template=self.cfg.job_name_template,
            workpiece_filename=self.cfg.workpiece_filename,
        )
        collision_shape = self.pb.createCollisionShape(
            shapeType=self.pb.GEOM_MESH,
            fileName=str(stl_path),
            meshScale=[0.001, 0.001, 0.001],
            flags=self.pb.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=self.client_id,
        )
        body_id = self.pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=[self.cfg.stl_x_offset_m, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        self.workpiece_cache[workpiece_id] = body_id
        return body_id

    def _load_workpiece_sdf(self, workpiece_id: int) -> SDFGrid | None:
        workpiece_id = int(workpiece_id)
        if workpiece_id in self.sdf_cache:
            return self.sdf_cache[workpiece_id]
        try:
            sdf_path = _resolve_workpiece_file_path(
                workpiece_id=workpiece_id,
                jobs_root=self.cfg.jobs_root,
                simple_jobs_root=self.cfg.simple_jobs_root,
                simple_workpiece_id_offset=self.cfg.simple_workpiece_id_offset,
                job_name_template=self.cfg.job_name_template,
                filename=self.cfg.sdf_filename,
                file_label="SDF",
            )
        except FileNotFoundError:
            if self.cfg.sdf_required:
                raise
            return None
        sdf_grid = SDFGrid.load(
            sdf_path,
            out_of_bounds_value_m=self.cfg.sdf_out_of_bounds_value_m,
        )
        self.sdf_cache[workpiece_id] = sdf_grid
        return sdf_grid

    def _build_robot_collision_surface_points(
        self,
        resolved_urdf_path: str,
        points_per_link: int,
    ) -> dict[int, np.ndarray]:
        try:
            import trimesh
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SDF validation requires `trimesh` to sample robot collision geometry."
            ) from exc

        package_roots = [Path(root).expanduser().resolve() for root in self.cfg.urdf_package_roots]
        package_roots.append(Path(resolved_urdf_path).expanduser().resolve().parent)
        tree = ET.parse(resolved_urdf_path)
        root = tree.getroot()
        points_by_link: dict[int, list[np.ndarray]] = {}

        for link_elem in root.findall("link"):
            link_name = link_elem.get("name")
            if not link_name:
                continue
            if link_name in self.link_name_to_index:
                link_index = self.link_name_to_index[link_name]
            elif link_name == "base_link":
                link_index = -1
            else:
                continue

            link_candidates = []
            for collision_elem in link_elem.findall("collision"):
                geometry_elem = collision_elem.find("geometry")
                if geometry_elem is None:
                    continue
                mesh_elem = geometry_elem.find("mesh")
                box_elem = geometry_elem.find("box")
                local_points = None
                if mesh_elem is not None:
                    filename = mesh_elem.get("filename")
                    if filename is None:
                        continue
                    mesh_path = _resolve_mesh_filename(filename, package_roots)
                    mesh = trimesh.load_mesh(mesh_path, force="mesh")
                    scale = _parse_float_vector(mesh_elem.get("scale"), (1.0, 1.0, 1.0)).reshape(1, 3)
                    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
                    face_centers = np.asarray(mesh.triangles_center, dtype=np.float32) * scale
                    local_points = np.concatenate([vertices, face_centers], axis=0)
                elif box_elem is not None:
                    size = _parse_float_vector(box_elem.get("size"), (0.0, 0.0, 0.0)).reshape(3)
                    half = size * 0.5
                    corners = np.asarray(
                        [
                            [sx * half[0], sy * half[1], sz * half[2]]
                            for sx in (-1.0, 1.0)
                            for sy in (-1.0, 1.0)
                            for sz in (-1.0, 1.0)
                        ],
                        dtype=np.float32,
                    )
                    face_centers = np.asarray(
                        [
                            [half[0], 0.0, 0.0], [-half[0], 0.0, 0.0],
                            [0.0, half[1], 0.0], [0.0, -half[1], 0.0],
                            [0.0, 0.0, half[2]], [0.0, 0.0, -half[2]],
                        ],
                        dtype=np.float32,
                    )
                    local_points = np.concatenate([corners, face_centers], axis=0)
                if local_points is None or local_points.size == 0:
                    continue
                local_points = _apply_origin_transform(local_points, collision_elem.find("origin"))
                link_candidates.append(local_points.astype(np.float32))

            if link_candidates:
                merged = np.concatenate(link_candidates, axis=0)
                selected = _select_deterministic_surface_points(merged, points_per_link)
                points_by_link.setdefault(link_index, []).append(selected)

        return {
            link_index: np.concatenate(point_chunks, axis=0).astype(np.float32)
            for link_index, point_chunks in points_by_link.items()
        }

    def _get_link_pose(self, link_index: int) -> tuple[np.ndarray, np.ndarray]:
        if link_index == -1:
            position, orientation = self.pb.getBasePositionAndOrientation(
                self.robot_id,
                physicsClientId=self.client_id,
            )
            return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)
        link_state = self.pb.getLinkState(
            self.robot_id,
            link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return np.asarray(link_state[4], dtype=np.float32), np.asarray(link_state[5], dtype=np.float32)

    def _robot_surface_points_world(self) -> np.ndarray:
        world_points = []
        for link_index, local_points in self.robot_surface_points_by_link.items():
            position, orientation = self._get_link_pose(link_index)
            rotation = np.asarray(
                self.pb.getMatrixFromQuaternion(orientation),
                dtype=np.float32,
            ).reshape(3, 3)
            world_points.append(local_points @ rotation.T + position.reshape(1, 3))
        if not world_points:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(world_points, axis=0).astype(np.float32)

    def _min_sdf_distance_for_current_robot_state(self, sdf_grid: SDFGrid | None) -> float:
        if sdf_grid is None:
            return float("nan")
        robot_points = self._robot_surface_points_world()
        if robot_points.size == 0:
            return float("nan")
        sdf_values = sdf_grid.query(robot_points)
        if np.all(np.isnan(sdf_values)):
            return float("nan")
        return float(np.nanmin(sdf_values))

    def _set_robot_joints(self, joint_state: np.ndarray) -> None:
        joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
        if joint_state.shape[0] != len(self.revolute_joint_indices):
            raise ValueError(
                f"Joint state length mismatch, expected {len(self.revolute_joint_indices)}, got {joint_state.shape[0]}"
            )
        for joint_idx, joint_value in zip(self.revolute_joint_indices, joint_state):
            self.pb.resetJointState(
                self.robot_id,
                joint_idx,
                float(joint_value),
                physicsClientId=self.client_id,
            )

    def _get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        link_state = self.pb.getLinkState(
            self.robot_id,
            self.tcp_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        position = np.asarray(link_state[4], dtype=np.float32)
        orientation = np.asarray(link_state[5], dtype=np.float32)
        return position, orientation

    def _target_world_position(
        self,
        start_joint_state: np.ndarray,
        goal_position_normalized: np.ndarray,
    ) -> np.ndarray:
        self._set_robot_joints(start_joint_state)
        start_tcp_position, start_tcp_quat = self._get_tcp_pose()
        goal_position_local_m = np.asarray(goal_position_normalized, dtype=np.float32).reshape(3) * self.cfg.goal_position_norm_m
        rotation_matrix = np.asarray(
            self.pb.getMatrixFromQuaternion(start_tcp_quat),
            dtype=np.float32,
        ).reshape(3, 3)
        return start_tcp_position + rotation_matrix @ goal_position_local_m

    def reconstruct_joint_trajectory(
        self,
        pred_action_horizon: np.ndarray,
        start_joint_normalized: np.ndarray,
        end_joint_normalized: np.ndarray,
    ) -> np.ndarray:
        start_joint_state = self._unnormalize_joint_state(start_joint_normalized)
        end_joint_state = self._unnormalize_joint_state(end_joint_normalized)
        pred_action_horizon = np.asarray(pred_action_horizon, dtype=np.float32)
        if pred_action_horizon.ndim != 2 or pred_action_horizon.shape[1] != 6:
            raise ValueError(
                f"Predicted action horizon must have shape [T, 6] for pybullet validation, got {pred_action_horizon.shape}"
            )
        if self.stats_mode == "bspline":
            recon_result = reconstruct_trajectory_from_normalized_free_residual(
                normalized_free_delta_w=pred_action_horizon,
                start_state=np.asarray(start_joint_normalized, dtype=np.float32),
                end_state=np.asarray(end_joint_normalized, dtype=np.float32),
                mean=self.stats_mean,
                std=self.stats_std,
                num_control_points=self.cfg.num_control_points,
                num_steps=self.cfg.target_steps,
                degree=self.cfg.spline_degree,
            )
            joint_trajectory = unnormalize_joint_trajectory_with_urdf_limits(
                normalized_trajectory=recon_result["fitted_trajectory"],
                lower_limits=self.joint_lower_limits,
                upper_limits=self.joint_upper_limits,
            )
            return np.asarray(joint_trajectory, dtype=np.float32)

        denormalized_deltas = pred_action_horizon * self.stats_std.reshape(1, 6) + self.stats_mean.reshape(1, 6)
        cumulative = start_joint_state.reshape(1, 6) + np.cumsum(denormalized_deltas, axis=0)
        joint_trajectory = np.concatenate([start_joint_state.reshape(1, 6), cumulative], axis=0).astype(np.float32)
        if joint_trajectory.shape[0] >= 2:
            joint_trajectory[-1] = joint_trajectory[-1].astype(np.float32)
        _ = end_joint_state
        return joint_trajectory

    def densify_joint_trajectory(self, joint_trajectory: np.ndarray) -> np.ndarray:
        joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
        if joint_trajectory.ndim != 2 or joint_trajectory.shape[1] != len(self.revolute_joint_indices):
            raise ValueError(
                "joint_trajectory must have shape [T, J] matching robot joints, "
                f"got {joint_trajectory.shape}"
            )
        if (not self.cfg.interpolate_for_collision) or joint_trajectory.shape[0] <= 1:
            return joint_trajectory.astype(np.float32)

        dense_segments = [joint_trajectory[0]]
        for segment_idx in range(joint_trajectory.shape[0] - 1):
            q0 = joint_trajectory[segment_idx]
            q1 = joint_trajectory[segment_idx + 1]
            max_delta = float(np.max(np.abs(q1 - q0)))
            interpolation_steps = max(
                self.cfg.min_interpolated_steps_per_segment,
                int(np.ceil(max_delta / self.cfg.max_joint_step_rad)),
            )
            for step_idx in range(1, interpolation_steps + 1):
                alpha = float(step_idx) / float(interpolation_steps)
                dense_segments.append(((1.0 - alpha) * q0 + alpha * q1).astype(np.float32))
        return np.asarray(dense_segments, dtype=np.float32)

    def evaluate_trajectory(
        self,
        workpiece_id: int,
        joint_trajectory: np.ndarray,
        start_joint_state: np.ndarray,
        goal_position_normalized: np.ndarray,
    ) -> dict[str, float | bool]:
        workpiece_body_id = self._load_workpiece_body(workpiece_id)
        sdf_grid = self._load_workpiece_sdf(workpiece_id)
        target_world_position = self._target_world_position(start_joint_state, goal_position_normalized)
        joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
        if joint_trajectory.ndim != 2 or joint_trajectory.shape[1] != len(self.revolute_joint_indices):
            raise ValueError(
                "joint_trajectory must have shape [T, J] matching robot joints, "
                f"got {joint_trajectory.shape}"
            )
        if joint_trajectory.shape[0] != self.cfg.target_steps:
            raise ValueError(
                f"PyBullet validation expects exactly {self.cfg.target_steps} reconstructed joint states, "
                f"got {joint_trajectory.shape[0]}. Adjust training.pybullet_eval.target_steps or action reconstruction."
            )
        has_collision = False
        segment_collision_steps = 0
        min_sdf_distance_m = float("nan")
        final_tcp_position = None
        for joint_state in joint_trajectory:
            self._set_robot_joints(joint_state)
            self.pb.performCollisionDetection(physicsClientId=self.client_id)
            contacts = self.pb.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=workpiece_body_id,
                distance=self.cfg.collision_distance_threshold,
                physicsClientId=self.client_id,
            )
            if contacts:
                has_collision = True
                segment_collision_steps += 1
            step_min_sdf_distance_m = self._min_sdf_distance_for_current_robot_state(sdf_grid)
            if not np.isnan(step_min_sdf_distance_m):
                if np.isnan(min_sdf_distance_m):
                    min_sdf_distance_m = step_min_sdf_distance_m
                else:
                    min_sdf_distance_m = min(min_sdf_distance_m, step_min_sdf_distance_m)
            final_tcp_position, _ = self._get_tcp_pose()
        if final_tcp_position is None:
            raise ValueError("Empty joint trajectory is not valid for pybullet validation.")

        legacy_collision_steps = float(segment_collision_steps)
        legacy_trajectory_steps = float(joint_trajectory.shape[0])
        if self.cfg.interpolate_for_collision and self.cfg.log_legacy_pybullet_metrics:
            dense_joint_trajectory = self.densify_joint_trajectory(joint_trajectory)
            legacy_collision_steps = 0.0
            for joint_state in dense_joint_trajectory:
                self._set_robot_joints(joint_state)
                self.pb.performCollisionDetection(physicsClientId=self.client_id)
                contacts = self.pb.getClosestPoints(
                    bodyA=self.robot_id,
                    bodyB=workpiece_body_id,
                    distance=self.cfg.collision_distance_threshold,
                    physicsClientId=self.client_id,
                )
                if contacts:
                    legacy_collision_steps += 1.0
            legacy_trajectory_steps = float(dense_joint_trajectory.shape[0])

        goal_error = float(np.linalg.norm(final_tcp_position - target_world_position))
        goal_reached = goal_error <= self.cfg.goal_tolerance_m
        success = bool(goal_reached and not has_collision)
        return {
            "has_collision": bool(has_collision),
            "segment_collision_steps": float(segment_collision_steps),
            "segment_steps": float(joint_trajectory.shape[0]),
            "min_sdf_distance_m": float(min_sdf_distance_m),
            "legacy_collision_steps": float(legacy_collision_steps),
            "legacy_trajectory_steps": float(legacy_trajectory_steps),
            "goal_error_m": goal_error,
            "goal_reached": bool(goal_reached),
            "success": success,
        }


class PyBulletValidationRunner:
    def __init__(self, cfg: PyBulletValidationConfig):
        self.cfg = cfg
        self.validator = PyBulletCollisionValidator(cfg)

    def close(self) -> None:
        self.validator.close()

    def _build_obs_batch(
        self,
        replay_buffer,
        episode_idx: int,
        obs_keys: tuple[str, ...],
        n_obs_steps: int,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        import torch

        episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
        start_idx, end_idx = _episode_bounds(episode_ends, episode_idx)
        episode_length = end_idx - start_idx
        if episode_length <= 0:
            raise ValueError(f"Episode {episode_idx} is empty.")
        obs_batch: dict[str, torch.Tensor] = {}
        raw_obs: dict[str, np.ndarray] = {}
        for key in obs_keys:
            value = np.asarray(replay_buffer[key][start_idx:end_idx], dtype=np.float32)
            value = value[:n_obs_steps]
            if value.shape[0] < n_obs_steps:
                pad_count = n_obs_steps - value.shape[0]
                pad = np.repeat(value[-1:], pad_count, axis=0)
                value = np.concatenate([value, pad], axis=0)
            raw_obs[key] = value.copy()
            obs_batch[key] = torch.from_numpy(value[None]).to(device)
        return obs_batch, raw_obs

    def run(
        self,
        policy,
        replay_buffer,
        episode_mask: np.ndarray,
        obs_keys: tuple[str, ...],
        n_obs_steps: int,
        device: torch.device,
    ) -> dict[str, float]:
        import torch

        if "workpiece_ids" not in replay_buffer.meta:
            raise KeyError(
                "PyBullet validation requires `meta/workpiece_ids` in the zarr dataset. "
                "Rebuild the dataset with workpiece metadata."
            )
        workpiece_ids = np.asarray(replay_buffer.meta["workpiece_ids"][:], dtype=np.int64)
        episode_indices = np.flatnonzero(np.asarray(episode_mask, dtype=bool))
        if self.cfg.max_episodes is not None:
            episode_indices = episode_indices[: int(self.cfg.max_episodes)]
        if episode_indices.size == 0:
            return {}

        sample_metrics = []
        policy.eval()
        with torch.no_grad():
            for episode_idx in episode_indices.tolist():
                obs_dict, raw_obs = self._build_obs_batch(
                    replay_buffer=replay_buffer,
                    episode_idx=episode_idx,
                    obs_keys=obs_keys,
                    n_obs_steps=n_obs_steps,
                    device=device,
                )
                result = policy.predict_action(obs_dict)
                pred_action_horizon = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
                joint_trajectory = self.validator.reconstruct_joint_trajectory(
                    pred_action_horizon=pred_action_horizon,
                    start_joint_normalized=raw_obs["first_joint_angles_normalized"][0],
                    end_joint_normalized=raw_obs["last_joint_angles_normalized"][0],
                )
                metric = self.validator.evaluate_trajectory(
                    workpiece_id=int(workpiece_ids[episode_idx]),
                    joint_trajectory=joint_trajectory,
                    start_joint_state=self.validator._unnormalize_joint_state(raw_obs["first_joint_angles_normalized"][0]),
                    goal_position_normalized=raw_obs["goal_position"][0],
                )
                sample_metrics.append(metric)

        total = float(len(sample_metrics))
        collision_count = sum(1.0 for item in sample_metrics if item["has_collision"])
        success_count = sum(1.0 for item in sample_metrics if item["success"])
        goal_reached_count = sum(1.0 for item in sample_metrics if item["goal_reached"])
        total_segment_collision_steps = sum(float(item["segment_collision_steps"]) for item in sample_metrics)
        total_segment_steps = sum(float(item["segment_steps"]) for item in sample_metrics)
        total_legacy_collision_steps = sum(float(item["legacy_collision_steps"]) for item in sample_metrics)
        total_legacy_steps = sum(float(item["legacy_trajectory_steps"]) for item in sample_metrics)
        mean_goal_error = sum(float(item["goal_error_m"]) for item in sample_metrics) / total
        min_sdf_distances = np.asarray(
            [float(item["min_sdf_distance_m"]) for item in sample_metrics],
            dtype=np.float32,
        )
        valid_sdf_mask = ~np.isnan(min_sdf_distances)
        mean_min_sdf_distance_m = (
            float(np.mean(min_sdf_distances[valid_sdf_mask]))
            if np.any(valid_sdf_mask)
            else float("nan")
        )
        log_data = {
            "val_traj_collision_rate": collision_count / total,
            "val_segment_collision_rate": (
                total_segment_collision_steps / total_segment_steps
                if total_segment_steps > 0
                else 0.0
            ),
            "val_mean_min_sdf_distance_m": mean_min_sdf_distance_m,
            "val_pybullet_eval_episodes": total,
            "val_sdf_valid_rate": float(np.mean(valid_sdf_mask.astype(np.float32))),
        }
        if self.cfg.log_legacy_pybullet_metrics:
            log_data.update({
                "val_pybullet_collision_rate": collision_count / total,
                "val_pybullet_success_rate": success_count / total,
                "val_pybullet_goal_reached_rate": goal_reached_count / total,
                "val_pybullet_collision_step_rate": (
                    total_legacy_collision_steps / total_legacy_steps
                    if total_legacy_steps > 0
                    else 0.0
                ),
                "val_pybullet_mean_goal_error_m": mean_goal_error,
            })
        return log_data
