from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy_3d.common.pybullet_validation import (
    _apply_origin_transform,
    _expand_urdf_package_roots,
    _parse_float_vector,
    _resolve_mesh_filename,
    _rpy_to_matrix,
    _select_deterministic_surface_points,
)


DEFAULT_LINK_SURFACE_POINT_COUNTS = {
    "wrist_2_link": 64,
    "wrist_3_link": 96,
    "pen_link": 128,
}


def _config_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _homogeneous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def _origin_transform(origin_elem) -> np.ndarray:
    if origin_elem is None:
        return np.eye(4, dtype=np.float32)
    xyz = _parse_float_vector(origin_elem.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_float_vector(origin_elem.get("rpy"), (0.0, 0.0, 0.0))
    return _homogeneous_transform(_rpy_to_matrix(rpy), xyz)


class DifferentiableTrajectoryLoss(nn.Module):
    def __init__(self, config, horizon: int, prediction_type: str):
        super().__init__()
        self.enabled = bool(_config_get(config, "enabled", False))
        self.horizon = int(horizon)
        if not self.enabled:
            return

        if str(prediction_type) != "sample":
            raise ValueError(
                "Differentiable trajectory losses require scheduler prediction_type='sample', "
                f"got {prediction_type!r}."
            )

        self.num_control_points = int(_config_get(config, "num_control_points", 16))
        self.trajectory_steps = int(_config_get(config, "trajectory_steps", 32))
        expected_horizon = self.num_control_points - 6
        if self.horizon != expected_horizon:
            raise ValueError(
                "Trajectory loss expects one action per free B-spline control point: "
                f"horizon={self.horizon}, num_control_points={self.num_control_points}, "
                f"expected horizon={expected_horizon}."
            )

        self.topk = int(_config_get(config, "topk", 32))
        self.softmin_beta = float(_config_get(config, "softmin_beta", 75.0))
        self.d_safe = float(_config_get(config, "d_safe", 0.001))
        self.sdf_collision_weight = float(
            _config_get(config, "sdf_collision_weight", 0.2)
        )
        self.trajectory_collision_weight = float(
            _config_get(config, "trajectory_collision_weight", 0.3)
        )
        self.smooth_weight = float(_config_get(config, "smooth_weight", 0.01))
        self.simple_workpiece_id_offset = int(
            _config_get(config, "simple_workpiece_id_offset", 1000)
        )
        self.sdf_filename = str(
            _config_get(config, "sdf_filename", "workpiece_sdf.npz")
        )
        self.job_name_template = str(
            _config_get(config, "job_name_template", "job_{workpiece_id:03d}")
        )
        self.sdf_out_of_bounds_distance_m = float(
            _config_get(config, "sdf_out_of_bounds_distance_m", 1.0)
        )
        self.query_chunk_size = int(
            _config_get(config, "query_chunk_size", 262144)
        )
        self.cpu_cache_size = int(_config_get(config, "cpu_cache_size", 32))
        self.gpu_cache_size = int(_config_get(config, "gpu_cache_size", 8))
        if self.topk <= 0:
            raise ValueError(f"trajectory_loss.topk must be positive, got {self.topk}")
        if self.softmin_beta <= 0:
            raise ValueError(
                f"trajectory_loss.softmin_beta must be positive, got {self.softmin_beta}"
            )
        if self.d_safe <= 0:
            raise ValueError(
                f"trajectory_loss.d_safe must be positive, got {self.d_safe}"
            )
        if self.query_chunk_size <= 0:
            raise ValueError(
                "trajectory_loss.query_chunk_size must be positive, "
                f"got {self.query_chunk_size}"
            )
        if self.cpu_cache_size < 0 or self.gpu_cache_size < 0:
            raise ValueError(
                "trajectory_loss CPU/GPU cache sizes must be non-negative, "
                f"got {self.cpu_cache_size}/{self.gpu_cache_size}"
            )

        self.stats_path = Path(
            str(_config_get(config, "stats_path"))
        ).expanduser().resolve()
        self.jobs_sdf_root = Path(
            str(_config_get(config, "jobs_sdf_root", "data/raw_data/jobs"))
        ).expanduser().resolve()
        self.simple_sdf_root = Path(
            str(_config_get(config, "simple_sdf_root", "data/raw_data/simple_jobs"))
        ).expanduser().resolve()
        self.urdf_path = Path(
            str(
                _config_get(
                    config,
                    "urdf_path",
                    "config/robot-model/ur5e_with_pen.urdf",
                )
            )
        ).expanduser().resolve()
        self.urdf_package_roots = tuple(
            str(value)
            for value in _config_get(
                config,
                "urdf_package_roots",
                ["config/robot-model"],
            )
        )
        for label, path in (
            ("stats_path", self.stats_path),
            ("urdf_path", self.urdf_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"trajectory_loss.{label} does not exist: {path}"
                )

        stats = np.load(self.stats_path)
        missing_stats = [
            key for key in ("mean", "std", "basis_matrix") if key not in stats.files
        ]
        if missing_stats:
            raise KeyError(
                f"Trajectory stats {self.stats_path} are missing keys: {missing_stats}"
            )
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        basis_matrix = np.asarray(stats["basis_matrix"], dtype=np.float32)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError(
                f"Trajectory stats mean/std must have shape (6,), got {mean.shape}/{std.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError(f"Trajectory stats contain non-finite mean/std values: {self.stats_path}")
        if np.any(std <= 0):
            raise ValueError(f"Trajectory stats std must be positive: {std.tolist()}")
        expected_basis_shape = (self.trajectory_steps, self.num_control_points)
        if basis_matrix.shape != expected_basis_shape:
            raise ValueError(
                f"Trajectory basis_matrix must have shape {expected_basis_shape}, "
                f"got {basis_matrix.shape}"
            )
        if not np.all(np.isfinite(basis_matrix)):
            raise ValueError(f"Trajectory basis matrix contains non-finite values: {self.stats_path}")
        self.register_buffer("delta_w_mean", torch.from_numpy(mean))
        self.register_buffer("delta_w_std", torch.from_numpy(std))
        self.register_buffer("basis_matrix", torch.from_numpy(basis_matrix))

        link_point_counts_cfg = _config_get(
            config,
            "link_surface_point_counts",
            DEFAULT_LINK_SURFACE_POINT_COUNTS,
        )
        self.link_surface_point_counts = {
            str(name): int(count)
            for name, count in dict(link_point_counts_cfg).items()
        }
        if self.link_surface_point_counts != DEFAULT_LINK_SURFACE_POINT_COUNTS:
            expected = DEFAULT_LINK_SURFACE_POINT_COUNTS
            raise ValueError(
                "trajectory_loss.link_surface_point_counts must match the configured "
                f"terminal-priority schedule {expected}, got {self.link_surface_point_counts}"
            )

        self._build_robot_geometry()
        self._cpu_sdf_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._gpu_sdf_cache: OrderedDict[
            tuple[int, str, torch.dtype], dict[str, torch.Tensor]
        ] = OrderedDict()

    def _build_robot_geometry(self) -> None:
        try:
            import trimesh
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Differentiable trajectory collision loss requires `trimesh`."
            ) from exc

        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        package_roots = _expand_urdf_package_roots(
            str(self.urdf_path),
            list(self.urdf_package_roots),
        )
        link_by_name = {
            str(link.get("name")): link
            for link in root.findall("link")
            if link.get("name")
        }
        parent_joint_by_child = {}
        revolute_joint_names = []
        joint_limits = {}
        for joint in root.findall("joint"):
            child_elem = joint.find("child")
            parent_elem = joint.find("parent")
            if child_elem is None or parent_elem is None:
                continue
            child_name = child_elem.get("link")
            parent_name = parent_elem.get("link")
            if not child_name or not parent_name:
                continue
            joint_type = str(joint.get("type", "fixed"))
            joint_name = str(joint.get("name", ""))
            record = {
                "name": joint_name,
                "type": joint_type,
                "parent": parent_name,
                "child": child_name,
                "origin": _origin_transform(joint.find("origin")),
                "axis": _parse_float_vector(
                    None if joint.find("axis") is None else joint.find("axis").get("xyz"),
                    (1.0, 0.0, 0.0),
                ),
            }
            parent_joint_by_child[child_name] = record
            if joint_type == "revolute":
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"Revolute joint {joint_name!r} has no limits.")
                revolute_joint_names.append(joint_name)
                joint_limits[joint_name] = (
                    float(limit.get("lower")),
                    float(limit.get("upper")),
                )
        if len(revolute_joint_names) != 6:
            raise ValueError(
                f"Expected 6 revolute joints in {self.urdf_path}, got {revolute_joint_names}"
            )
        self.revolute_joint_names = tuple(revolute_joint_names)
        joint_index_by_name = {
            name: index for index, name in enumerate(self.revolute_joint_names)
        }
        lower_limits = np.asarray(
            [joint_limits[name][0] for name in self.revolute_joint_names],
            dtype=np.float32,
        )
        upper_limits = np.asarray(
            [joint_limits[name][1] for name in self.revolute_joint_names],
            dtype=np.float32,
        )
        self.register_buffer("joint_lower_limits", torch.from_numpy(lower_limits))
        self.register_buffer("joint_upper_limits", torch.from_numpy(upper_limits))

        required_joint_by_child = {}
        for link_name in self.link_surface_point_counts:
            if link_name not in link_by_name:
                raise KeyError(
                    f"Collision-loss link {link_name!r} is missing from {self.urdf_path}"
                )
            current_link = link_name
            while current_link in parent_joint_by_child:
                joint = parent_joint_by_child[current_link]
                required_joint_by_child[current_link] = joint
                current_link = str(joint["parent"])

        def joint_depth(joint):
            depth = 0
            parent = str(joint["parent"])
            while parent in parent_joint_by_child:
                parent = str(parent_joint_by_child[parent]["parent"])
                depth += 1
            return depth

        self.fk_joint_records = []
        for record_idx, joint in enumerate(
            sorted(required_joint_by_child.values(), key=joint_depth)
        ):
            origin_name = f"fk_origin_{record_idx}"
            axis_name = f"fk_axis_{record_idx}"
            self.register_buffer(
                origin_name,
                torch.from_numpy(np.asarray(joint["origin"], dtype=np.float32)),
            )
            axis = np.asarray(joint["axis"], dtype=np.float32)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-12:
                raise ValueError(f"Joint {joint['name']!r} has a zero axis.")
            self.register_buffer(axis_name, torch.from_numpy(axis / axis_norm))
            q_index = joint_index_by_name.get(str(joint["name"]), -1)
            if joint["type"] == "revolute" and q_index < 0:
                raise KeyError(
                    f"Revolute FK joint {joint['name']!r} is missing from the joint order."
                )
            self.fk_joint_records.append(
                {
                    "name": str(joint["name"]),
                    "type": str(joint["type"]),
                    "parent": str(joint["parent"]),
                    "child": str(joint["child"]),
                    "origin_buffer": origin_name,
                    "axis_buffer": axis_name,
                    "q_index": q_index,
                }
            )

        self.link_point_buffer_names = {}
        for link_name, point_count in self.link_surface_point_counts.items():
            link_elem = link_by_name[link_name]
            candidates = []
            for collision_elem in link_elem.findall("collision"):
                geometry_elem = collision_elem.find("geometry")
                if geometry_elem is None:
                    continue
                mesh_elem = geometry_elem.find("mesh")
                if mesh_elem is None:
                    continue
                filename = mesh_elem.get("filename")
                if filename is None:
                    continue
                mesh_path = _resolve_mesh_filename(filename, package_roots)
                mesh = trimesh.load_mesh(mesh_path, force="mesh")
                scale = _parse_float_vector(
                    mesh_elem.get("scale"),
                    (1.0, 1.0, 1.0),
                ).reshape(1, 3)
                vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
                face_centers = (
                    np.asarray(mesh.triangles_center, dtype=np.float32) * scale
                )
                local_points = np.concatenate([vertices, face_centers], axis=0)
                candidates.append(
                    _apply_origin_transform(
                        local_points,
                        collision_elem.find("origin"),
                    )
                )
            if not candidates:
                raise ValueError(
                    f"Link {link_name!r} has no collision mesh available for SDF sampling."
                )
            merged = np.concatenate(candidates, axis=0)
            selected = _select_deterministic_surface_points(merged, point_count)
            if selected.shape != (point_count, 3):
                raise ValueError(
                    f"Expected {point_count} sampled points for {link_name}, got {selected.shape}"
                )
            buffer_name = f"surface_points_{link_name}"
            self.register_buffer(buffer_name, torch.from_numpy(selected))
            self.link_point_buffer_names[link_name] = buffer_name
        self.total_surface_points = sum(self.link_surface_point_counts.values())
        if self.total_surface_points != 288:
            raise ValueError(
                f"Terminal-priority point schedule must total 288, got {self.total_surface_points}"
            )

    @staticmethod
    def _axis_angle_transform(
        axis: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = angles.shape[0]
        axis = axis.to(device=angles.device, dtype=angles.dtype)
        x, y, z = axis.unbind()
        zero = torch.zeros((), device=angles.device, dtype=angles.dtype)
        skew = torch.stack(
            [
                torch.stack([zero, -z, y]),
                torch.stack([z, zero, -x]),
                torch.stack([-y, x, zero]),
            ]
        )
        identity = torch.eye(3, device=angles.device, dtype=angles.dtype)
        sin_angle = torch.sin(angles).reshape(batch_size, 1, 1)
        cos_angle = torch.cos(angles).reshape(batch_size, 1, 1)
        rotation = (
            identity.reshape(1, 3, 3)
            + sin_angle * skew.reshape(1, 3, 3)
            + (1.0 - cos_angle) * (skew @ skew).reshape(1, 3, 3)
        )
        transform = torch.eye(
            4, device=angles.device, dtype=angles.dtype
        ).reshape(1, 4, 4).repeat(batch_size, 1, 1)
        transform[:, :3, :3] = rotation
        return transform

    def reconstruct_control_points_and_trajectory(
        self,
        normalized_free_delta_w: torch.Tensor,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalized_free_delta_w.ndim != 3:
            raise ValueError(
                "Predicted free control points must have shape [B, H, 6], "
                f"got {tuple(normalized_free_delta_w.shape)}"
            )
        expected_shape = (self.horizon, 6)
        if tuple(normalized_free_delta_w.shape[1:]) != expected_shape:
            raise ValueError(
                f"Predicted free control points must have shape [B, {self.horizon}, 6], "
                f"got {tuple(normalized_free_delta_w.shape)}"
            )
        start_state = self._select_single_observation(start_state)
        end_state = self._select_single_observation(end_state)
        batch_size = normalized_free_delta_w.shape[0]
        if start_state.shape != (batch_size, 6) or end_state.shape != (batch_size, 6):
            raise ValueError(
                "Start/end normalized joint states must both have shape [B, 6], "
                f"got {tuple(start_state.shape)} and {tuple(end_state.shape)}"
            )
        free_interpolation = torch.linspace(
            0.0,
            1.0,
            self.horizon + 2,
            device=normalized_free_delta_w.device,
            dtype=normalized_free_delta_w.dtype,
        )[1:-1].reshape(1, self.horizon, 1)
        linear_control_points = torch.empty(
            (
                batch_size,
                self.num_control_points,
                normalized_free_delta_w.shape[-1],
            ),
            device=normalized_free_delta_w.device,
            dtype=normalized_free_delta_w.dtype,
        )
        linear_control_points[:, :3, :] = start_state[:, None, :]
        linear_control_points[:, -3:, :] = end_state[:, None, :]
        linear_control_points[:, 3:-3, :] = (
            start_state[:, None, :] * (1.0 - free_interpolation)
            + end_state[:, None, :] * free_interpolation
        )
        delta_w = torch.zeros_like(linear_control_points)
        delta_w[:, 3:-3, :] = (
            normalized_free_delta_w * self.delta_w_std.reshape(1, 1, 6)
            + self.delta_w_mean.reshape(1, 1, 6)
        )
        control_points = linear_control_points + delta_w
        trajectory = torch.einsum(
            "tn,bnj->btj",
            self.basis_matrix.to(
                device=control_points.device,
                dtype=control_points.dtype,
            ),
            control_points,
        )
        return control_points, trajectory

    @staticmethod
    def _select_single_observation(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            return value[:, 0, :]
        if value.ndim == 2:
            return value
        raise ValueError(
            f"Expected observation shape [B, 6] or [B, To, 6], got {tuple(value.shape)}"
        )

    def normalized_to_joint_angles(
        self,
        normalized_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        lower = self.joint_lower_limits.reshape(1, 1, 6).to(
            device=normalized_trajectory.device,
            dtype=normalized_trajectory.dtype,
        )
        upper = self.joint_upper_limits.reshape(1, 1, 6).to(
            device=normalized_trajectory.device,
            dtype=normalized_trajectory.dtype,
        )
        return lower + (normalized_trajectory + 1.0) * 0.5 * (upper - lower)

    def robot_surface_points_world(
        self,
        joint_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        if joint_trajectory.ndim != 3 or joint_trajectory.shape[-1] != 6:
            raise ValueError(
                f"joint_trajectory must have shape [B, T, 6], got {tuple(joint_trajectory.shape)}"
            )
        batch_size, trajectory_steps, _ = joint_trajectory.shape
        flat_joints = joint_trajectory.reshape(-1, 6)
        flat_size = flat_joints.shape[0]
        identity = torch.eye(
            4,
            device=joint_trajectory.device,
            dtype=joint_trajectory.dtype,
        ).reshape(1, 4, 4).repeat(flat_size, 1, 1)
        transforms = {"world": identity}
        for joint in self.fk_joint_records:
            parent_transform = transforms.get(joint["parent"])
            if parent_transform is None:
                raise RuntimeError(
                    f"FK parent transform {joint['parent']!r} is unavailable before "
                    f"processing joint {joint['name']!r}."
                )
            origin = getattr(self, joint["origin_buffer"]).to(
                device=joint_trajectory.device,
                dtype=joint_trajectory.dtype,
            ).reshape(1, 4, 4)
            transform = parent_transform @ origin
            if joint["type"] == "revolute":
                motion = self._axis_angle_transform(
                    getattr(self, joint["axis_buffer"]),
                    flat_joints[:, int(joint["q_index"])],
                )
                transform = transform @ motion
            transforms[joint["child"]] = transform

        world_point_chunks = []
        for link_name in self.link_surface_point_counts:
            link_transform = transforms[link_name]
            local_points = getattr(
                self, self.link_point_buffer_names[link_name]
            ).to(
                device=joint_trajectory.device,
                dtype=joint_trajectory.dtype,
            )
            rotation = link_transform[:, :3, :3]
            translation = link_transform[:, :3, 3]
            world_points = (
                torch.einsum("bij,pj->bpi", rotation, local_points)
                + translation[:, None, :]
            )
            world_point_chunks.append(world_points)
        flat_world_points = torch.cat(world_point_chunks, dim=1)
        return flat_world_points.reshape(
            batch_size,
            trajectory_steps,
            self.total_surface_points,
            3,
        )

    def _resolve_sdf_path(self, workpiece_id: int) -> Path:
        workpiece_id = int(workpiece_id)
        sdf_root = self.jobs_sdf_root
        local_id = workpiece_id
        if workpiece_id >= self.simple_workpiece_id_offset:
            sdf_root = self.simple_sdf_root
            local_id = workpiece_id - self.simple_workpiece_id_offset
        job_name = self.job_name_template.format(workpiece_id=local_id)
        sdf_path = sdf_root / job_name / self.sdf_filename
        if not sdf_path.is_file():
            raise FileNotFoundError(
                f"Missing trajectory-loss SDF for workpiece_id={workpiece_id}: {sdf_path}"
            )
        return sdf_path

    def _load_sdf_cpu(self, workpiece_id: int) -> dict[str, np.ndarray]:
        workpiece_id = int(workpiece_id)
        if workpiece_id in self._cpu_sdf_cache:
            value = self._cpu_sdf_cache.pop(workpiece_id)
            self._cpu_sdf_cache[workpiece_id] = value
            return value
        sdf_path = self._resolve_sdf_path(workpiece_id)
        with np.load(sdf_path) as data:
            missing = [key for key in ("sdf", "x", "y", "z") if key not in data.files]
            if missing:
                raise KeyError(f"SDF file {sdf_path} is missing keys: {missing}")
            value = {
                key: np.ascontiguousarray(np.asarray(data[key], dtype=np.float32))
                for key in ("sdf", "x", "y", "z")
            }
        expected_shape = (
            value["x"].shape[0],
            value["y"].shape[0],
            value["z"].shape[0],
        )
        if value["sdf"].shape != expected_shape:
            raise ValueError(
                f"SDF shape {value['sdf'].shape} does not match axes {expected_shape} "
                f"for {sdf_path}"
            )
        for axis_name in ("x", "y", "z"):
            axis = value[axis_name]
            if axis.ndim != 1 or axis.shape[0] < 2:
                raise ValueError(
                    f"SDF axis {axis_name!r} must be one-dimensional with at least "
                    f"two coordinates, got {axis.shape} for {sdf_path}"
                )
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                raise ValueError(
                    f"SDF axis {axis_name!r} must be finite and strictly increasing: {sdf_path}"
                )
        if not np.all(np.isfinite(value["sdf"])):
            raise ValueError(f"SDF grid contains non-finite values: {sdf_path}")
        self._cpu_sdf_cache[workpiece_id] = value
        while len(self._cpu_sdf_cache) > self.cpu_cache_size:
            self._cpu_sdf_cache.popitem(last=False)
        return value

    def _load_sdf_device(
        self,
        workpiece_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        cache_key = (int(workpiece_id), str(device), dtype)
        if cache_key in self._gpu_sdf_cache:
            value = self._gpu_sdf_cache.pop(cache_key)
            self._gpu_sdf_cache[cache_key] = value
            return value
        cpu_value = self._load_sdf_cpu(workpiece_id)
        value = {
            key: torch.from_numpy(array).to(device=device, dtype=dtype)
            for key, array in cpu_value.items()
        }
        self._gpu_sdf_cache[cache_key] = value
        while len(self._gpu_sdf_cache) > self.gpu_cache_size:
            self._gpu_sdf_cache.popitem(last=False)
        return value

    def _query_sdf_grid(
        self,
        points: torch.Tensor,
        grid: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        outputs = []
        for chunk_start in range(0, flat_points.shape[0], self.query_chunk_size):
            chunk = flat_points[
                chunk_start: chunk_start + self.query_chunk_size
            ]
            outputs.append(self._query_sdf_grid_chunk(chunk, grid))
        return torch.cat(outputs, dim=0).reshape(original_shape)

    def _query_sdf_grid_chunk(
        self,
        points: torch.Tensor,
        grid: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x_axis, y_axis, z_axis = grid["x"], grid["y"], grid["z"]
        valid = (
            (points[:, 0] >= x_axis[0])
            & (points[:, 0] <= x_axis[-1])
            & (points[:, 1] >= y_axis[0])
            & (points[:, 1] <= y_axis[-1])
            & (points[:, 2] >= z_axis[0])
            & (points[:, 2] <= z_axis[-1])
        )

        def axis_indices(axis, values):
            upper = torch.searchsorted(axis, values.contiguous(), right=True)
            upper = upper.clamp(1, axis.shape[0] - 1)
            lower = upper - 1
            span = axis[upper] - axis[lower]
            weight = torch.where(
                span > 0,
                (values - axis[lower]) / span,
                torch.zeros_like(values),
            )
            return lower, upper, weight

        ix0, ix1, tx = axis_indices(x_axis, points[:, 0])
        iy0, iy1, ty = axis_indices(y_axis, points[:, 1])
        iz0, iz1, tz = axis_indices(z_axis, points[:, 2])
        sdf = grid["sdf"]
        ny, nz = sdf.shape[1], sdf.shape[2]
        flat_sdf = sdf.reshape(-1)

        def gather(ix, iy, iz):
            return flat_sdf[ix * ny * nz + iy * nz + iz]

        c000 = gather(ix0, iy0, iz0)
        c100 = gather(ix1, iy0, iz0)
        c010 = gather(ix0, iy1, iz0)
        c110 = gather(ix1, iy1, iz0)
        c001 = gather(ix0, iy0, iz1)
        c101 = gather(ix1, iy0, iz1)
        c011 = gather(ix0, iy1, iz1)
        c111 = gather(ix1, iy1, iz1)
        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        interpolated = c0 * (1.0 - tz) + c1 * tz
        fallback = torch.full_like(
            interpolated,
            self.sdf_out_of_bounds_distance_m,
        )
        return torch.where(valid, interpolated, fallback)

    def query_workpiece_sdf(
        self,
        world_points: torch.Tensor,
        workpiece_ids: torch.Tensor,
    ) -> torch.Tensor:
        if workpiece_ids.ndim != 1 or workpiece_ids.shape[0] != world_points.shape[0]:
            raise ValueError(
                "workpiece_id must have shape [B] matching world points, "
                f"got {tuple(workpiece_ids.shape)} and {tuple(world_points.shape)}"
            )
        result = torch.empty(
            world_points.shape[:-1],
            device=world_points.device,
            dtype=world_points.dtype,
        )
        unique_workpiece_ids = torch.unique(workpiece_ids.detach()).cpu().tolist()
        for workpiece_id in unique_workpiece_ids:
            workpiece_id = int(workpiece_id)
            sample_indices = torch.nonzero(
                workpiece_ids == workpiece_id,
                as_tuple=False,
            ).reshape(-1)
            grid = self._load_sdf_device(
                workpiece_id,
                world_points.device,
                world_points.dtype,
            )
            distances = self._query_sdf_grid(
                world_points.index_select(0, sample_indices),
                grid,
            )
            result.index_copy_(0, sample_indices, distances)
        return result

    def forward(
        self,
        predicted_clean_action: torch.Tensor,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            zero = predicted_clean_action.sum() * 0.0
            return {
                "sdf_collision_loss": zero,
                "trajectory_collision_loss": zero,
                "smooth_loss": zero,
                "weighted_auxiliary_loss": zero,
            }
        if "workpiece_id" not in batch:
            raise KeyError(
                "Trajectory collision loss requires batch['workpiece_id']."
            )
        obs = batch.get("obs", {})
        for key in (
            "first_joint_angles_normalized",
            "last_joint_angles_normalized",
        ):
            if key not in obs:
                raise KeyError(
                    f"Trajectory collision loss requires batch['obs'][{key!r}]."
                )
        predicted_clean_action = predicted_clean_action.float()
        control_points, normalized_trajectory = (
            self.reconstruct_control_points_and_trajectory(
                predicted_clean_action,
                obs["first_joint_angles_normalized"],
                obs["last_joint_angles_normalized"],
            )
        )
        joint_trajectory = self.normalized_to_joint_angles(normalized_trajectory)
        world_points = self.robot_surface_points_world(joint_trajectory)
        distances = self.query_workpiece_sdf(
            world_points,
            batch["workpiece_id"].to(
                device=world_points.device,
                dtype=torch.long,
            ),
        )

        violations = F.relu((self.d_safe - distances) / self.d_safe)
        flattened_violations = violations.reshape(violations.shape[0], -1)
        topk = min(self.topk, flattened_violations.shape[1])
        sdf_collision_loss = (
            torch.topk(flattened_violations, k=topk, dim=1)
            .values.pow(2)
            .mean()
        )
        flattened_distances = distances.reshape(distances.shape[0], -1)
        soft_min_distance = -(
            torch.logsumexp(
                -self.softmin_beta * flattened_distances,
                dim=1,
            )
            - math.log(flattened_distances.shape[1])
        ) / self.softmin_beta
        trajectory_collision_loss = F.relu(
            (self.d_safe - soft_min_distance) / self.d_safe
        ).pow(2).mean()
        second_difference = (
            control_points[:, 2:, :]
            - 2.0 * control_points[:, 1:-1, :]
            + control_points[:, :-2, :]
        )
        smooth_loss = second_difference.pow(2).mean()
        weighted_auxiliary_loss = (
            self.sdf_collision_weight * sdf_collision_loss
            + self.trajectory_collision_weight * trajectory_collision_loss
            + self.smooth_weight * smooth_loss
        )
        return {
            "sdf_collision_loss": sdf_collision_loss,
            "trajectory_collision_loss": trajectory_collision_loss,
            "smooth_loss": smooth_loss,
            "weighted_auxiliary_loss": weighted_auxiliary_loss,
        }


def build_differentiable_trajectory_loss(
    config,
    horizon: int,
    prediction_type: str,
) -> DifferentiableTrajectoryLoss | None:
    if not bool(_config_get(config, "enabled", False)):
        return None
    return DifferentiableTrajectoryLoss(
        config=config,
        horizon=horizon,
        prediction_type=prediction_type,
    )


def combine_diffusion_and_trajectory_losses(
    trajectory_loss_module: DifferentiableTrajectoryLoss | None,
    diffusion_loss: torch.Tensor,
    predicted_clean_action: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    if trajectory_loss_module is None:
        value = float(diffusion_loss.detach().item())
        return diffusion_loss, {
            "bc_loss": value,
            "diffusion_loss": value,
            "sdf_collision_loss": 0.0,
            "trajectory_collision_loss": 0.0,
            "smooth_loss": 0.0,
            "total_loss": value,
        }
    auxiliary = trajectory_loss_module(
        predicted_clean_action=predicted_clean_action,
        batch=batch,
    )
    total_loss = diffusion_loss + auxiliary["weighted_auxiliary_loss"]
    return total_loss, {
        "bc_loss": float(diffusion_loss.detach().item()),
        "diffusion_loss": float(diffusion_loss.detach().item()),
        "sdf_collision_loss": float(
            auxiliary["sdf_collision_loss"].detach().item()
        ),
        "trajectory_collision_loss": float(
            auxiliary["trajectory_collision_loss"].detach().item()
        ),
        "smooth_loss": float(auxiliary["smooth_loss"].detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }
