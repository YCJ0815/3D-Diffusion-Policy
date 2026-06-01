from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np

from diffusion_policy_3d.common.pointcloud_roi import world_to_local_point


@dataclass
class PlanningInputData:
    goal_position_world: np.ndarray
    goal_position_start_tcp_frame: np.ndarray
    goal_position: np.ndarray
    goal_rotation: np.ndarray
    goal_direction_world: np.ndarray
    goal_direction: np.ndarray
    joint_names: tuple[str, ...]
    joint_lower_limits: np.ndarray
    joint_upper_limits: np.ndarray
    first_joint_angles: np.ndarray
    last_joint_angles: np.ndarray
    first_joint_angles_normalized: np.ndarray
    last_joint_angles_normalized: np.ndarray
    trajectory_key: str


def _require_keys(data: np.lib.npyio.NpzFile, keys: Iterable[str], npz_path: str) -> None:
    missing = [key for key in keys if key not in data.files]
    if missing:
        raise KeyError(f"Missing required keys in npz file {npz_path}: {missing}")


def _as_float32_array(value: np.ndarray, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {array.shape}")
    return array


def _load_goal_position(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "end_xyz" in data.files:
        return _as_float32_array(data["end_xyz"], expected_shape=(3,))
    if "goal_tf" in data.files:
        goal_tf = _as_float32_array(data["goal_tf"], expected_shape=(4, 4))
        return goal_tf[:3, 3].astype(np.float32)
    raise KeyError("NPZ must contain either `end_xyz` or `goal_tf`.")


def _load_goal_rotation(data: np.lib.npyio.NpzFile) -> np.ndarray:
    _require_keys(data, ("goal_tf",), "in-memory npz")
    goal_tf = _as_float32_array(data["goal_tf"], expected_shape=(4, 4))
    return goal_tf[:3, :3].astype(np.float32)


def _load_goal_direction(data: np.lib.npyio.NpzFile, goal_rotation: np.ndarray) -> np.ndarray:
    if "end_normal" in data.files:
        return _as_float32_array(data["end_normal"], expected_shape=(3,))
    return goal_rotation[:, 2].astype(np.float32)


def _rotate_direction_to_start_tcp_frame(direction_world: np.ndarray, start_tf: np.ndarray) -> np.ndarray:
    direction_world = _as_float32_array(direction_world, expected_shape=(3,))
    rotation_world_from_start = _as_float32_array(start_tf[:3, :3], expected_shape=(3, 3))
    direction_start_tcp = rotation_world_from_start.T @ direction_world
    norm = float(np.linalg.norm(direction_start_tcp))
    if norm <= 1e-12:
        raise ValueError("Goal direction becomes near-zero after rotation into the start TCP frame.")
    return (direction_start_tcp / norm).astype(np.float32)


def _goal_rotation_first_two_columns_in_start_tcp_frame(
    goal_rotation_world: np.ndarray,
    start_tf: np.ndarray,
) -> np.ndarray:
    goal_rotation_world = _as_float32_array(goal_rotation_world, expected_shape=(3, 3))
    rotation_world_from_start = _as_float32_array(start_tf[:3, :3], expected_shape=(3, 3))
    goal_rotation_start_tcp = rotation_world_from_start.T @ goal_rotation_world
    first_two_columns = goal_rotation_start_tcp[:, :2].reshape(-1)
    if first_two_columns.shape != (6,):
        raise ValueError(
            f"Expected flattened first two rotation columns to have shape (6,), got {first_two_columns.shape}"
        )
    return first_two_columns.astype(np.float32)


def _load_start_transform(data: np.lib.npyio.NpzFile) -> np.ndarray:
    _require_keys(data, ("start_tf",), "in-memory npz")
    return _as_float32_array(data["start_tf"], expected_shape=(4, 4))


def _select_trajectory(data: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray]:
    candidate_keys = (
        "q_plan",
        "q_playback",
        "q_seed_path",
        "q_rrt_playback",
    )
    for key in candidate_keys:
        if key in data.files:
            trajectory = _as_float32_array(data[key])
            if trajectory.ndim != 2:
                raise ValueError(f"Trajectory `{key}` must have shape [T, J], got {trajectory.shape}")
            if trajectory.shape[0] == 0:
                raise ValueError(f"Trajectory `{key}` is empty.")
            return key, trajectory
    raise KeyError(
        "NPZ does not contain a supported trajectory key. "
        "Expected one of `q_plan`, `q_playback`, `q_seed_path`, `q_rrt_playback`."
    )


def _load_endpoint_joint_angles(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    _require_keys(data, ("q_start", "q_goal"), "in-memory npz")
    q_start = _as_float32_array(data["q_start"], expected_shape=(6,))
    q_goal = _as_float32_array(data["q_goal"], expected_shape=(6,))
    return q_start, q_goal


def _default_urdf_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "ur5e_with_pen.urdf"


def _load_joint_limits_from_urdf(urdf_path: str) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    joint_names: list[str] = []
    lower_limits: list[float] = []
    upper_limits: list[float] = []

    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        name = joint.get("name")
        if lower is None or upper is None or name is None:
            continue
        joint_names.append(name)
        lower_limits.append(float(lower))
        upper_limits.append(float(upper))

    if not joint_names:
        raise ValueError(f"No revolute joint limits found in URDF: {urdf_path}")

    return (
        tuple(joint_names),
        np.asarray(lower_limits, dtype=np.float32),
        np.asarray(upper_limits, dtype=np.float32),
    )


def _normalize_joint_angles(
    joint_angles: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    joint_angles = _as_float32_array(joint_angles)
    if joint_angles.shape != lower_limits.shape:
        raise ValueError(
            f"Joint angle shape {joint_angles.shape} does not match URDF limits shape {lower_limits.shape}"
        )
    spans = upper_limits - lower_limits
    if np.any(spans <= 0):
        raise ValueError("Invalid URDF joint limits: upper limits must be greater than lower limits.")
    normalized_01 = (joint_angles - lower_limits) / spans
    return (normalized_01 * 2.0 - 1.0).astype(np.float32)


def load_planning_input_data(
    npz_path: str,
    norm: float,
    urdf_path: str | None = None,
) -> PlanningInputData:
    if norm <= 0:
        raise ValueError(f"norm must be positive, got {norm}")

    data = np.load(npz_path)
    resolved_urdf_path = urdf_path if urdf_path is not None else str(_default_urdf_path())

    goal_position_world = _load_goal_position(data)
    start_tf = _load_start_transform(data)
    goal_position_start_tcp_frame = world_to_local_point(goal_position_world, start_tf)
    goal_position = goal_position_start_tcp_frame / float(norm)
    goal_rotation = _load_goal_rotation(data)
    goal_direction_world = _load_goal_direction(data, goal_rotation)
    goal_direction = _rotate_direction_to_start_tcp_frame(goal_direction_world, start_tf)
    trajectory_key, trajectory = _select_trajectory(data)
    joint_names, joint_lower_limits, joint_upper_limits = _load_joint_limits_from_urdf(resolved_urdf_path)

    first_joint_angles, last_joint_angles = _load_endpoint_joint_angles(data)
    first_joint_angles_normalized = _normalize_joint_angles(
        first_joint_angles,
        joint_lower_limits,
        joint_upper_limits,
    )
    last_joint_angles_normalized = _normalize_joint_angles(
        last_joint_angles,
        joint_lower_limits,
        joint_upper_limits,
    )

    return PlanningInputData(
        goal_position_world=goal_position_world.astype(np.float32),
        goal_position_start_tcp_frame=goal_position_start_tcp_frame.astype(np.float32),
        goal_position=goal_position.astype(np.float32),
        goal_rotation=goal_rotation.astype(np.float32),
        goal_direction_world=goal_direction_world.astype(np.float32),
        goal_direction=goal_direction.astype(np.float32),
        joint_names=joint_names,
        joint_lower_limits=joint_lower_limits.astype(np.float32),
        joint_upper_limits=joint_upper_limits.astype(np.float32),
        first_joint_angles=first_joint_angles.astype(np.float32),
        last_joint_angles=last_joint_angles.astype(np.float32),
        first_joint_angles_normalized=first_joint_angles_normalized.astype(np.float32),
        last_joint_angles_normalized=last_joint_angles_normalized.astype(np.float32),
        trajectory_key=trajectory_key,
    )


def load_bspline_planning_input_data(
    npz_path: str,
    norm: float,
    urdf_path: str | None = None,
) -> PlanningInputData:
    planning_data = load_planning_input_data(
        npz_path=npz_path,
        norm=norm,
        urdf_path=urdf_path,
    )
    data = np.load(npz_path)
    start_tf = _load_start_transform(data)
    goal_direction = _goal_rotation_first_two_columns_in_start_tcp_frame(
        planning_data.goal_rotation,
        start_tf,
    )

    return PlanningInputData(
        goal_position_world=planning_data.goal_position_world,
        goal_position_start_tcp_frame=planning_data.goal_position_start_tcp_frame,
        goal_position=planning_data.goal_position,
        goal_rotation=planning_data.goal_rotation,
        goal_direction_world=planning_data.goal_direction_world,
        goal_direction=goal_direction.astype(np.float32),
        joint_names=planning_data.joint_names,
        joint_lower_limits=planning_data.joint_lower_limits,
        joint_upper_limits=planning_data.joint_upper_limits,
        first_joint_angles=planning_data.first_joint_angles,
        last_joint_angles=planning_data.last_joint_angles,
        first_joint_angles_normalized=planning_data.first_joint_angles_normalized,
        last_joint_angles_normalized=planning_data.last_joint_angles_normalized,
        trajectory_key=planning_data.trajectory_key,
    )
