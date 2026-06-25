from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence
import inspect
import math
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from diffusion_policy_3d.common.bspline import (
    FIXED_CONTROL_POINTS_PER_SIDE,
    build_bspline_basis_matrix,
    build_linear_control_points,
    evaluate_quintic_bspline,
)


DEFAULT_GUIDANCE_TARGETS = (-0.02, 0.0)
MAX_REPAIR_ATTEMPTS = 5


def _filter_scheduler_step_kwargs(scheduler, step_kwargs: dict[str, Any]) -> dict[str, Any]:
    if not step_kwargs:
        return {}
    valid_param_names = set(inspect.signature(scheduler.step).parameters.keys())
    return {
        key: value
        for key, value in step_kwargs.items()
        if key in valid_param_names
    }


@dataclass
class SurfaceCBFQPGuidanceConfig:
    enabled: bool = False
    num_candidates: int = 32
    guidance_steps: int = 2
    max_risk_segments: int = 3
    window_radius: int = 2
    points_per_segment: int = 2
    min_constraints_per_segment: int = 4
    active_constraints: int = 24
    check_steps: int = 64
    cert_steps: int = 256
    cert_swept_intermediate: int = 3
    d_safe: float = 0.03
    d_trigger: float = 0.06
    d_cert: float = 0.01
    eps_deep: float = 0.03
    delta_max: float = 0.05
    scp_iterations: int = 2
    delta_max_total: float = 0.05
    delta_max_pass1: float = 0.025
    delta_max_pass2: float = 0.025
    d_trigger_pass2_offset: float = 0.005
    margin_buffer: float = 0.005
    enable_local_waypoint_qp_after_certificate: bool = True
    local_waypoint_qp_window_radius: int = 2
    local_waypoint_qp_max_collision_segments: int = 2
    local_waypoint_qp_min_clearance_trigger: float = -0.01
    local_waypoint_qp_target_buffer: float = 0.005
    local_waypoint_qp_lambda_s: float = 0.25
    local_waypoint_qp_delta_max: float = 0.02
    local_waypoint_qp_max_velocity_step: float = 0.2
    local_waypoint_qp_max_acceleration_step: float = 0.4
    local_waypoint_qp_maxiter: int = 100
    lambda_s: float = 0.25
    rho: float = 1.0e5
    ddim_eta: float = 0.0
    joint_limit_steps: int = 32
    guidance_targets: tuple[float, ...] = field(default_factory=lambda: tuple(DEFAULT_GUIDANCE_TARGETS))
    fallback_to_terminal_cbf: bool = True


@dataclass
class GuidanceLog:
    dp_time: float = 0.0
    guidance_time: float = 0.0
    qp_time: float = 0.0
    certificate_time: float = 0.0
    total_time: float = 0.0
    h_min_before_guidance: float = math.nan
    h_min_after_guidance: float = math.nan
    h_min_final: float = math.nan
    num_qp_called: int = 0
    num_qp_success: int = 0
    num_active_constraints: int = 0
    certificate_success: bool = False
    fallback_used: bool = False
    goal_error: float = math.nan
    smoothness: float = math.nan
    path_length: float = math.nan
    best_candidate_index: int | None = None
    selected_candidate_index: int | None = None
    used_existing_terminal_cbf: bool = False
    selected_by_certificate: bool = False
    num_candidates_total: int = 0
    num_candidates_screened: int = 0
    candidate_ranking: list[int] = field(default_factory=list)
    repair_attempt_count: int = 0
    repair_attempted_candidate_indices: list[int] = field(default_factory=list)
    risk_segment_count_total: int = 0
    risk_segment_count_selected: int = 0
    selected_risk_segments: list[dict[str, Any]] = field(default_factory=list)
    selected_window_timesteps: list[int] = field(default_factory=list)
    selected_constraint_count: int = 0
    scp_iterations_configured: int = 2
    scp_passes_attempted_total: int = 0
    scp_passes_succeeded_total: int = 0
    selected_candidate_pass_count: int = 0
    selected_candidate_passes_succeeded: int = 0
    selected_candidate_pre_qp_min_clearance: float = math.nan
    selected_candidate_post_qp1_min_clearance: float = math.nan
    selected_candidate_post_qp2_min_clearance: float = math.nan
    selected_candidate_certificate_min_clearance: float = math.nan
    local_waypoint_qp_attempted_total: int = 0
    local_waypoint_qp_success_total: int = 0
    selected_candidate_local_waypoint_qp_attempted: bool = False
    selected_candidate_local_waypoint_qp_success: bool = False
    selected_candidate_post_local_waypoint_qp_min_clearance: float = math.nan
    final_success_source: str = "failure"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingContext:
    condition_data: Any
    condition_mask: Any
    local_cond: Any
    global_cond: Any


@dataclass
class GuidanceResult:
    best_index: int
    best_normalized_free_residual: np.ndarray
    best_control_points_normalized: np.ndarray
    best_joint_trajectory: np.ndarray
    candidate_infos: list[dict[str, Any]]
    log: GuidanceLog


class PyBulletSurfaceEnvironmentAdapter:
    """Thin adapter around PyBulletCollisionValidator for guidance-time geometry queries.

    This wrapper is intentionally lazy: the project can import this module in test-only
    environments that do not have pybullet/trimesh/diffusers installed.
    """

    def __init__(
        self,
        *,
        validator,
        workpiece_id: int,
        joint_lower_limits: np.ndarray,
        joint_upper_limits: np.ndarray,
        surface_points_per_link_override: dict[str, int] | None = None,
    ):
        self.validator = validator
        self.workpiece_id = int(workpiece_id)
        self.joint_lower_limits = np.asarray(joint_lower_limits, dtype=np.float32).reshape(-1)
        self.joint_upper_limits = np.asarray(joint_upper_limits, dtype=np.float32).reshape(-1)
        self.pb = validator.pb
        self.robot_id = validator.robot_id
        self.client_id = validator.client_id
        self.revolute_joint_indices = tuple(int(v) for v in validator.revolute_joint_indices)
        self.link_index_to_name = dict(validator.link_index_to_name)
        self.robot_surface_points_by_link = self._select_surface_points_by_link(
            validator_surface_points_by_link=validator.robot_surface_points_by_link,
            points_per_link_override=surface_points_per_link_override,
        )
        self._surface_samples = self._build_surface_sample_index()

    @property
    def surface_samples(self) -> list[dict[str, Any]]:
        return self._surface_samples

    def _select_surface_points_by_link(
        self,
        *,
        validator_surface_points_by_link: dict[int, np.ndarray],
        points_per_link_override: dict[str, int] | None,
    ) -> dict[int, np.ndarray]:
        if points_per_link_override is None:
            return {
                int(link_index): np.asarray(points, dtype=np.float32)
                for link_index, points in validator_surface_points_by_link.items()
            }
        selected: dict[int, np.ndarray] = {}
        link_name_to_index = {str(name): int(index) for index, name in self.link_index_to_name.items()}
        missing_links: list[str] = []
        for link_name, requested_count in points_per_link_override.items():
            link_index = link_name_to_index.get(str(link_name))
            if link_index is None or link_index not in validator_surface_points_by_link:
                missing_links.append(str(link_name))
                continue
            count = int(requested_count)
            if count <= 0:
                continue
            points = np.asarray(validator_surface_points_by_link[link_index], dtype=np.float32)
            selected[link_index] = points[: min(count, points.shape[0])].astype(np.float32)
        if missing_links:
            available_link_names = [
                self.link_index_to_name.get(int(link_index), str(link_index))
                for link_index in validator_surface_points_by_link.keys()
            ]
            raise ValueError(
                "Surface guidance requested robot surface links that are not available "
                f"in the validator samples: missing={missing_links}, available={available_link_names}"
            )
        if not selected:
            raise ValueError(
                "surface_points_per_link_override did not select any robot surface samples. "
                f"override={points_per_link_override}"
            )
        return selected

    @property
    def dof(self) -> int:
        return len(self.revolute_joint_indices)

    def joint_scale(self) -> np.ndarray:
        return ((self.joint_upper_limits - self.joint_lower_limits) * 0.5).astype(np.float32)

    def normalized_to_actual(self, q_normalized: np.ndarray) -> np.ndarray:
        q_normalized = np.asarray(q_normalized, dtype=np.float32)
        lower = self.joint_lower_limits.reshape((1,) * (q_normalized.ndim - 1) + (-1,))
        upper = self.joint_upper_limits.reshape((1,) * (q_normalized.ndim - 1) + (-1,))
        normalized_01 = (q_normalized + 1.0) * 0.5
        return (lower + normalized_01 * (upper - lower)).astype(np.float32)

    def load_sdf_grid(self):
        sdf_grid = self.validator._load_workpiece_sdf(self.workpiece_id)
        if sdf_grid is None:
            raise ValueError(
                "Surface CBF-QP guidance requires an SDF grid. The current validator returned None."
            )
        return sdf_grid

    def collect_joint_trajectory_sdf_with_link_details(self, joint_trajectory_actual: np.ndarray) -> dict[str, Any]:
        return self.validator.collect_joint_trajectory_sdf_with_link_details(
            workpiece_id=self.workpiece_id,
            joint_trajectory=np.asarray(joint_trajectory_actual, dtype=np.float32),
        )

    def collect_joint_trajectory_sdf_with_link_details_any_length(self, joint_trajectory_actual: np.ndarray) -> dict[str, Any]:
        sdf_grid = self.load_sdf_grid()
        joint_trajectory_actual = np.asarray(joint_trajectory_actual, dtype=np.float32)
        if joint_trajectory_actual.ndim != 2:
            raise ValueError(
                "joint_trajectory_actual must be rank-2 [T, J], "
                f"got shape {joint_trajectory_actual.shape}"
            )
        expected_dof = len(self.revolute_joint_indices)
        if joint_trajectory_actual.shape[1] != expected_dof:
            raise ValueError(
                "joint_trajectory_actual joint dimension mismatch, "
                f"expected {expected_dof}, got {joint_trajectory_actual.shape[1]}"
            )

        trajectory_sdf_values = []
        trajectory_sdf_values_by_link: dict[str, list[np.ndarray]] = {
            self.link_index_to_name.get(int(link_index), str(link_index)): []
            for link_index in self.robot_surface_points_by_link.keys()
        }
        for joint_state in joint_trajectory_actual:
            self.set_robot_joints(joint_state)
            timestep_values = []
            for link_index, local_points in self.robot_surface_points_by_link.items():
                position, orientation = self.validator._get_link_pose(int(link_index))
                rotation = np.asarray(
                    self.pb.getMatrixFromQuaternion(orientation),
                    dtype=np.float32,
                ).reshape(3, 3)
                world_points = local_points @ rotation.T + position.reshape(1, 3)
                link_sdf_values = sdf_grid.query(world_points).astype(np.float32)
                link_name = self.link_index_to_name.get(int(link_index), str(link_index))
                trajectory_sdf_values_by_link[link_name].append(link_sdf_values)
                timestep_values.append(link_sdf_values)
            if not timestep_values:
                return {
                    "all_sdf_values": np.empty((0, 0), dtype=np.float32),
                    "sdf_values_by_link": {},
                }
            trajectory_sdf_values.append(np.concatenate(timestep_values, axis=0).astype(np.float32))
        return {
            "all_sdf_values": np.stack(trajectory_sdf_values, axis=0).astype(np.float32),
            "sdf_values_by_link": {
                link_name: np.stack(link_values, axis=0).astype(np.float32)
                for link_name, link_values in trajectory_sdf_values_by_link.items()
            },
        }

    def set_robot_joints(self, q_actual: np.ndarray) -> None:
        self.validator._set_robot_joints(q_actual)

    def surface_point_world(self, *, link_index: int, local_point: np.ndarray) -> np.ndarray:
        position, orientation = self.validator._get_link_pose(int(link_index))
        rotation = np.asarray(
            self.pb.getMatrixFromQuaternion(orientation),
            dtype=np.float32,
        ).reshape(3, 3)
        return (np.asarray(local_point, dtype=np.float32).reshape(1, 3) @ rotation.T + position.reshape(1, 3))[0]

    def surface_point_jacobian(self, *, link_index: int, local_point: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
        self.set_robot_joints(q_actual)
        zero_vec = [0.0] * self.dof
        local_position = np.asarray(local_point, dtype=np.float32).reshape(3).tolist()
        jacobian_result = self.pb.calculateJacobian(
            self.robot_id,
            int(link_index),
            local_position,
            [float(v) for v in np.asarray(q_actual, dtype=np.float32).reshape(-1)],
            zero_vec,
            zero_vec,
            physicsClientId=self.client_id,
        )
        if isinstance(jacobian_result, (tuple, list)):
            if len(jacobian_result) >= 1:
                jacobian_t = jacobian_result[0]
            else:
                raise ValueError("pybullet.calculateJacobian returned an empty result")
        else:
            raise TypeError(
                f"Unexpected pybullet.calculateJacobian return type: {type(jacobian_result)!r}"
            )
        return np.asarray(jacobian_t, dtype=np.float32)

    def try_existing_terminal_cbf(self, joint_trajectory_actual: np.ndarray) -> dict[str, Any] | None:
        validator = self.validator
        for method_name in (
            "terminal_cbf_projection",
            "apply_terminal_cbf_projection",
            "fallback_planner",
            "plan_fallback_trajectory",
        ):
            method = getattr(validator, method_name, None)
            if callable(method):
                result = method(
                    workpiece_id=self.workpiece_id,
                    joint_trajectory=np.asarray(joint_trajectory_actual, dtype=np.float32),
                )
                return {
                    "method_name": method_name,
                    "result": result,
                }
        return None

    def _build_surface_sample_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        offset = 0
        for link_index, local_points in self.robot_surface_points_by_link.items():
            for point_index, local_point in enumerate(np.asarray(local_points, dtype=np.float32)):
                samples.append(
                    {
                        "flat_index": int(offset),
                        "link_index": int(link_index),
                        "link_name": self.link_index_to_name.get(int(link_index), str(link_index)),
                        "point_index": int(point_index),
                        "local_point": np.asarray(local_point, dtype=np.float32),
                    }
                )
                offset += 1
        return samples


def build_guidance_target_schedule(
    num_guidance_steps: int,
    target_values: Sequence[float] | None = None,
) -> np.ndarray:
    if num_guidance_steps <= 0:
        raise ValueError(f"num_guidance_steps must be positive, got {num_guidance_steps}")
    target_values = tuple(DEFAULT_GUIDANCE_TARGETS if target_values is None else target_values)
    if len(target_values) == num_guidance_steps:
        return np.asarray(target_values, dtype=np.float32)
    x_src = np.linspace(0.0, 1.0, num=len(target_values), dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, num=num_guidance_steps, dtype=np.float32)
    return np.interp(x_dst, x_src, np.asarray(target_values, dtype=np.float32)).astype(np.float32)


def select_guidance_candidate_indices(
    *,
    h_min_values: np.ndarray,
    qp_per_step: int,
    d_trigger: float,
    eps_deep: float,
) -> np.ndarray:
    h_min_values = np.asarray(h_min_values, dtype=np.float32).reshape(-1)
    if qp_per_step <= 0:
        return np.empty((0,), dtype=np.int64)
    candidate_indices = [
        int(index)
        for index, h_value in enumerate(h_min_values.tolist())
        if (h_value < float(d_trigger)) and (h_value >= -float(eps_deep))
    ]
    candidate_indices.sort(key=lambda index: float(h_min_values[index]))
    return np.asarray(candidate_indices[: int(qp_per_step)], dtype=np.int64)


def rank_screened_candidates(candidate_infos: Sequence[dict[str, Any]]) -> list[int]:
    def _sort_key(candidate_info: dict[str, Any]) -> tuple[float, int, float, float, float, int]:
        min_margin = float(candidate_info.get("coarse_min_margin", math.nan))
        dangerous_steps = int(candidate_info.get("coarse_dangerous_timestep_count", 0))
        total_risk = float(candidate_info.get("coarse_total_risk", 0.0))
        path_length = float(candidate_info.get("coarse_path_length", 0.0))
        smoothness = float(candidate_info.get("coarse_smoothness", 0.0))
        if not math.isfinite(min_margin):
            min_margin = -math.inf
        return (
            -min_margin,
            dangerous_steps,
            total_risk,
            path_length,
            smoothness,
            int(candidate_info.get("candidate_index", 0)),
        )

    return [
        int(candidate_info["candidate_index"])
        for candidate_info in sorted(candidate_infos, key=_sort_key)
    ]


def interpolate_swept_segments(joint_trajectory: np.ndarray, num_intermediate: int) -> np.ndarray:
    joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
    if joint_trajectory.ndim != 2:
        raise ValueError(f"joint_trajectory must be rank-2 [T, J], got shape {joint_trajectory.shape}")
    if joint_trajectory.shape[0] <= 1 or num_intermediate <= 0:
        return joint_trajectory.astype(np.float32)
    dense_points = [joint_trajectory[0].astype(np.float32)]
    for start_index in range(joint_trajectory.shape[0] - 1):
        q0 = joint_trajectory[start_index]
        q1 = joint_trajectory[start_index + 1]
        for inner_index in range(1, num_intermediate + 1):
            beta = float(inner_index) / float(num_intermediate + 1)
            dense_points.append(((1.0 - beta) * q0 + beta * q1).astype(np.float32))
        dense_points.append(q1.astype(np.float32))
    return np.asarray(dense_points, dtype=np.float32)


def compute_path_length(joint_trajectory: np.ndarray) -> float:
    joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
    if joint_trajectory.ndim != 2 or joint_trajectory.shape[0] <= 1:
        return 0.0
    diffs = np.diff(joint_trajectory, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def compute_smoothness(joint_trajectory: np.ndarray) -> float:
    joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
    if joint_trajectory.ndim != 2 or joint_trajectory.shape[0] <= 2:
        return 0.0
    accel = joint_trajectory[2:] - 2.0 * joint_trajectory[1:-1] + joint_trajectory[:-2]
    return float(np.mean(np.sum(accel ** 2, axis=1)))


def summarize_sdf_risk(
    *,
    sdf_result: dict[str, Any],
    d_safe: float,
    d_trigger: float,
) -> dict[str, Any]:
    all_sdf = np.asarray(
        sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)),
        dtype=np.float32,
    )
    if all_sdf.size == 0 or all_sdf.ndim != 2:
        return {
            "min_margin": math.nan,
            "min_clearance": math.nan,
            "dangerous_timestep_count": 0,
            "total_risk": 0.0,
            "finite_sdf_count": 0,
            "total_sdf_count": int(all_sdf.size),
            "finite_timestep_count": 0,
            "min_margin_per_step": np.empty((0,), dtype=np.float32),
        }

    finite_mask = np.isfinite(all_sdf)
    has_finite_value = np.any(finite_mask, axis=1)
    min_per_step = np.full(all_sdf.shape[0], np.nan, dtype=np.float32)
    if np.any(has_finite_value):
        min_per_step[has_finite_value] = np.min(
            np.where(finite_mask[has_finite_value], all_sdf[has_finite_value], np.inf),
            axis=1,
        )
    h_per_step = min_per_step - float(d_safe)
    valid_mask = np.isfinite(h_per_step)
    total_risk = float(np.sum(np.maximum(float(d_trigger) - h_per_step[valid_mask], 0.0))) if np.any(valid_mask) else 0.0
    dangerous_timestep_count = int(np.count_nonzero(valid_mask & (h_per_step < float(d_trigger))))
    min_margin = float(np.min(h_per_step[valid_mask])) if np.any(valid_mask) else math.nan
    min_clearance = float(np.min(min_per_step[has_finite_value])) if np.any(has_finite_value) else math.nan
    return {
        "min_margin": min_margin,
        "min_clearance": min_clearance,
        "dangerous_timestep_count": dangerous_timestep_count,
        "total_risk": total_risk,
        "finite_sdf_count": int(np.count_nonzero(finite_mask)),
        "total_sdf_count": int(all_sdf.size),
        "finite_timestep_count": int(np.count_nonzero(has_finite_value)),
        "min_margin_per_step": h_per_step.astype(np.float32),
    }


def build_risk_segments(
    *,
    sdf_result: dict[str, Any],
    d_trigger: float,
) -> list[dict[str, Any]]:
    all_sdf = np.asarray(
        sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)),
        dtype=np.float32,
    )
    if all_sdf.size == 0 or all_sdf.ndim != 2:
        return []

    finite_mask = np.isfinite(all_sdf)
    has_finite_value = np.any(finite_mask, axis=1)
    min_clearance_per_step = np.full(all_sdf.shape[0], np.nan, dtype=np.float32)
    if np.any(has_finite_value):
        min_clearance_per_step[has_finite_value] = np.min(
            np.where(finite_mask[has_finite_value], all_sdf[has_finite_value], np.inf),
            axis=1,
        )

    segments: list[dict[str, Any]] = []
    current_steps: list[int] = []
    trigger_value = float(d_trigger)

    def _finalize_segment(step_indices: list[int]) -> None:
        if not step_indices:
            return
        clearances = min_clearance_per_step[np.asarray(step_indices, dtype=np.int64)]
        peak_local_index = int(np.argmin(clearances))
        peak_timestep = int(step_indices[peak_local_index])
        min_clearance = float(clearances[peak_local_index])
        accumulated_risk = float(np.sum(np.maximum(trigger_value - clearances, 0.0)))
        deep_penalty = max(-min_clearance, 0.0)
        segments.append(
            {
                "segment_index": int(len(segments)),
                "start_timestep": int(step_indices[0]),
                "end_timestep": int(step_indices[-1]),
                "timesteps": [int(v) for v in step_indices],
                "peak_timestep": peak_timestep,
                "min_clearance": min_clearance,
                "risk_score": float(accumulated_risk + deep_penalty),
                "accumulated_risk": accumulated_risk,
                "deep_penalty": deep_penalty,
            }
        )

    for timestep, clearance in enumerate(min_clearance_per_step.tolist()):
        if math.isfinite(clearance) and clearance < trigger_value:
            current_steps.append(int(timestep))
        else:
            _finalize_segment(current_steps)
            current_steps = []
    _finalize_segment(current_steps)

    segments.sort(
        key=lambda segment: (
            -float(segment["risk_score"]),
            float(segment["min_clearance"]),
            int(segment["start_timestep"]),
        )
    )
    for segment_index, segment in enumerate(segments):
        segment["segment_index"] = int(segment_index)
    return segments


def _build_segment_window(segment: dict[str, Any], *, horizon: int, window_radius: int) -> list[int]:
    peak_timestep = int(segment["peak_timestep"])
    start = max(0, peak_timestep - int(window_radius))
    end = min(int(horizon) - 1, peak_timestep + int(window_radius))
    return list(range(start, end + 1))


def select_segment_window_timesteps(
    *,
    sdf_result: dict[str, Any],
    segment: dict[str, Any],
    points_per_segment: int,
    window_radius: int,
) -> list[int]:
    all_sdf = np.asarray(sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    if all_sdf.size == 0 or all_sdf.ndim != 2:
        return []
    window = _build_segment_window(segment, horizon=all_sdf.shape[0], window_radius=window_radius)
    scored_timesteps: list[tuple[float, int]] = []
    for timestep in window:
        step_sdf = np.asarray(all_sdf[timestep], dtype=np.float32).reshape(-1)
        finite_step = step_sdf[np.isfinite(step_sdf)]
        if finite_step.size == 0:
            continue
        scored_timesteps.append((float(np.min(finite_step)), int(timestep)))
    if not scored_timesteps:
        return []
    scored_timesteps.sort(key=lambda item: (item[0], item[1]))
    chosen = [int(timestep) for _, timestep in scored_timesteps[: int(points_per_segment)]]
    if int(segment["peak_timestep"]) not in chosen:
        chosen.append(int(segment["peak_timestep"]))
    return sorted(set(chosen))


def build_segment_window_cbf_constraints(
    *,
    segments: Sequence[dict[str, Any]],
    sdf_result: dict[str, Any],
    check_basis: np.ndarray,
    q_check_norm: np.ndarray,
    environment: PyBulletSurfaceEnvironmentAdapter,
    d_trigger: float,
    points_per_segment: int,
    min_constraints_per_segment: int,
    window_radius: int,
    max_active: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    all_sdf = np.asarray(sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    if all_sdf.size == 0 or all_sdf.ndim != 2 or not segments:
        return [], [], []

    selected_segments: list[dict[str, Any]] = []
    selected_timesteps: list[int] = []
    all_constraints: list[dict[str, Any]] = []

    for segment in segments:
        window_timesteps = _build_segment_window(
            segment,
            horizon=all_sdf.shape[0],
            window_radius=window_radius,
        )
        anchor_timesteps = select_segment_window_timesteps(
            sdf_result=sdf_result,
            segment=segment,
            points_per_segment=points_per_segment,
            window_radius=window_radius,
        )
        if not anchor_timesteps:
            continue
        segment_constraints: list[dict[str, Any]] = []
        for timestep in anchor_timesteps:
            step_sdf = np.asarray(all_sdf[timestep], dtype=np.float32).reshape(-1)
            finite_indices = np.flatnonzero(np.isfinite(step_sdf))
            if finite_indices.size == 0:
                continue
            ordered_indices = finite_indices[np.argsort(step_sdf[finite_indices])]
            for flat_index in ordered_indices[: max(1, int(points_per_segment))]:
                sample_info = _flat_index_to_surface_sample(environment.surface_samples, int(flat_index))
                if sample_info is None:
                    continue
                q_norm = np.asarray(q_check_norm[timestep], dtype=np.float32).reshape(-1)
                q_actual = environment.normalized_to_actual(q_norm)
                point_world = environment.surface_point_world(
                    link_index=int(sample_info["link_index"]),
                    local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
                )
                grad_world = approximate_sdf_gradient(environment.load_sdf_grid(), point_world)
                jacobian = environment.surface_point_jacobian(
                    link_index=int(sample_info["link_index"]),
                    local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
                    q_actual=q_actual,
                )
                j_pos = np.asarray(jacobian[:3, :], dtype=np.float32)
                g_actual = (j_pos.T @ grad_world.reshape(3)).astype(np.float32)
                segment_constraints.append(
                    {
                        "segment_index": int(segment["segment_index"]),
                        "t_index": int(timestep),
                        "h_value": float(step_sdf[int(flat_index)]),
                        "basis_row": np.asarray(check_basis[timestep], dtype=np.float32),
                        "g_actual": g_actual,
                        "link_index": int(sample_info["link_index"]),
                        "point_index": int(sample_info["point_index"]),
                    }
                )

        if len(segment_constraints) < int(min_constraints_per_segment):
            fallback_constraints: list[dict[str, Any]] = []
            for timestep in window_timesteps:
                step_sdf = np.asarray(all_sdf[timestep], dtype=np.float32).reshape(-1)
                finite_indices = np.flatnonzero(np.isfinite(step_sdf))
                if finite_indices.size == 0:
                    continue
                ordered_indices = finite_indices[np.argsort(step_sdf[finite_indices])]
                for flat_index in ordered_indices:
                    sample_info = _flat_index_to_surface_sample(environment.surface_samples, int(flat_index))
                    if sample_info is None:
                        continue
                    q_norm = np.asarray(q_check_norm[timestep], dtype=np.float32).reshape(-1)
                    q_actual = environment.normalized_to_actual(q_norm)
                    point_world = environment.surface_point_world(
                        link_index=int(sample_info["link_index"]),
                        local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
                    )
                    grad_world = approximate_sdf_gradient(environment.load_sdf_grid(), point_world)
                    jacobian = environment.surface_point_jacobian(
                        link_index=int(sample_info["link_index"]),
                        local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
                        q_actual=q_actual,
                    )
                    j_pos = np.asarray(jacobian[:3, :], dtype=np.float32)
                    g_actual = (j_pos.T @ grad_world.reshape(3)).astype(np.float32)
                    fallback_constraints.append(
                        {
                            "segment_index": int(segment["segment_index"]),
                            "t_index": int(timestep),
                            "h_value": float(step_sdf[int(flat_index)]),
                            "basis_row": np.asarray(check_basis[timestep], dtype=np.float32),
                            "g_actual": g_actual,
                            "link_index": int(sample_info["link_index"]),
                            "point_index": int(sample_info["point_index"]),
                        }
                    )
            segment_constraints.extend(fallback_constraints)

        deduped_constraints: list[dict[str, Any]] = []
        seen_keys: set[tuple[int, int, int]] = set()
        for constraint in sorted(segment_constraints, key=lambda item: (float(item["h_value"]), int(item["t_index"]))):
            key = (int(constraint["t_index"]), int(constraint["link_index"]), int(constraint["point_index"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_constraints.append(constraint)
            if len(deduped_constraints) >= max(int(min_constraints_per_segment), int(points_per_segment)):
                break

        if not deduped_constraints:
            continue
        selected_segments.append(
            {
                "segment_index": int(segment["segment_index"]),
                "start_timestep": int(segment["start_timestep"]),
                "end_timestep": int(segment["end_timestep"]),
                "peak_timestep": int(segment["peak_timestep"]),
                "risk_score": float(segment["risk_score"]),
                "window_timesteps": [int(v) for v in window_timesteps],
                "anchor_timesteps": [int(v) for v in anchor_timesteps],
                "constraint_count": int(len(deduped_constraints)),
            }
        )
        selected_timesteps.extend(window_timesteps)
        all_constraints.extend(deduped_constraints)

    all_constraints.sort(key=lambda item: (int(item["segment_index"]), float(item["h_value"]), int(item["t_index"])))
    return all_constraints[: int(max_active)], selected_segments, sorted(set(int(v) for v in selected_timesteps))


def build_collision_windows_from_clearance(
    *,
    min_clearance_per_step: np.ndarray,
    collision_threshold: float,
    window_radius: int,
    max_segments: int,
) -> list[dict[str, Any]]:
    min_clearance_per_step = np.asarray(min_clearance_per_step, dtype=np.float32).reshape(-1)
    if min_clearance_per_step.size == 0:
        return []
    segments: list[list[int]] = []
    current: list[int] = []
    for timestep, clearance in enumerate(min_clearance_per_step.tolist()):
        if math.isfinite(clearance) and clearance < float(collision_threshold):
            current.append(int(timestep))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    ranked = []
    for seg in segments:
        clearances = min_clearance_per_step[np.asarray(seg, dtype=np.int64)]
        peak_local_index = int(np.argmin(clearances))
        peak_timestep = int(seg[peak_local_index])
        start = max(0, int(seg[0]) - int(window_radius))
        end = min(int(min_clearance_per_step.shape[0]) - 1, int(seg[-1]) + int(window_radius))
        ranked.append(
            {
                "start_timestep": int(seg[0]),
                "end_timestep": int(seg[-1]),
                "peak_timestep": peak_timestep,
                "min_clearance": float(clearances[peak_local_index]),
                "window_start": int(start),
                "window_end": int(end),
                "window_timesteps": list(range(start, end + 1)),
                "timesteps": [int(v) for v in seg],
            }
        )
    ranked.sort(key=lambda item: (float(item["min_clearance"]), int(item["start_timestep"])))
    return ranked[: int(max_segments)]


def should_attempt_local_waypoint_qp(
    *,
    enable_local_waypoint_qp_after_certificate: bool,
    min_clearance: float,
    min_clearance_trigger: float,
    collision_threshold: float,
) -> bool:
    if not bool(enable_local_waypoint_qp_after_certificate):
        return False
    if not np.isfinite(min_clearance):
        return False
    return float(min_clearance_trigger) <= float(min_clearance) < float(collision_threshold)


class SurfaceCBFQPGuidanceRunner:
    def __init__(self, *, config: SurfaceCBFQPGuidanceConfig, environment: PyBulletSurfaceEnvironmentAdapter):
        self.config = config
        self.environment = environment
        self.sdf_grid = environment.load_sdf_grid()

    def run(
        self,
        *,
        candidate_residuals: np.ndarray,
        q_start_normalized: np.ndarray,
        q_goal_normalized: np.ndarray,
        delta_w_mean: np.ndarray,
        delta_w_std: np.ndarray,
        num_control_points: int,
        spline_degree: int,
    ) -> GuidanceResult:
        start_time = time.perf_counter()
        candidate_residuals = np.asarray(candidate_residuals, dtype=np.float32)
        q_start_normalized = np.asarray(q_start_normalized, dtype=np.float32).reshape(-1)
        q_goal_normalized = np.asarray(q_goal_normalized, dtype=np.float32).reshape(-1)
        delta_w_mean = np.asarray(delta_w_mean, dtype=np.float32).reshape(1, -1)
        delta_w_std = np.asarray(delta_w_std, dtype=np.float32).reshape(1, -1)
        log = GuidanceLog(scp_iterations_configured=int(self.config.scp_iterations))
        guidance_targets = build_guidance_target_schedule(
            int(self.config.guidance_steps),
            self.config.guidance_targets,
        )
        check_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.config.check_steps), degree=int(spline_degree))
        cert_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.config.cert_steps), degree=int(spline_degree))
        limit_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.config.joint_limit_steps), degree=int(spline_degree))
        free_slice = slice(FIXED_CONTROL_POINTS_PER_SIDE, num_control_points - FIXED_CONTROL_POINTS_PER_SIDE)

        candidate_infos: list[dict[str, Any]] = []
        selected_candidate_info: dict[str, Any] | None = None
        fallback = None

        for candidate_index, free_residual in enumerate(candidate_residuals):
            screening_start = time.perf_counter()
            control_points = reconstruct_control_points_from_free_residual(
                normalized_free_residual=free_residual,
                q_start_normalized=q_start_normalized,
                q_goal_normalized=q_goal_normalized,
                delta_w_mean=delta_w_mean,
                delta_w_std=delta_w_std,
                num_control_points=num_control_points,
            )
            q_check_norm = check_basis @ control_points
            q_check_actual = self.environment.normalized_to_actual(q_check_norm)
            sdf_result = self.environment.collect_joint_trajectory_sdf_with_link_details_any_length(q_check_actual)
            risk_summary = summarize_sdf_risk(
                sdf_result=sdf_result,
                d_safe=float(self.config.d_safe),
                d_trigger=float(self.config.d_trigger),
            )
            candidate_info = {
                "candidate_index": int(candidate_index),
                "coarse_min_margin": float(risk_summary["min_margin"]),
                "coarse_min_clearance_m": float(risk_summary["min_clearance"]),
                "coarse_dangerous_timestep_count": int(risk_summary["dangerous_timestep_count"]),
                "coarse_total_risk": float(risk_summary["total_risk"]),
                "coarse_path_length": compute_path_length(q_check_actual),
                "coarse_smoothness": compute_smoothness(q_check_actual),
                "coarse_screening_time": time.perf_counter() - screening_start,
                "h_min_before_guidance": float(risk_summary["min_margin"]),
                "h_min_after_guidance": math.nan,
                "h_min_final": math.nan,
                "certificate_success": False,
                "repair_attempted": False,
                "repair_attempt_order": None,
                "qp_attempted": False,
                "qp_success": False,
                "qp_skip_reason": None,
                "qp_solver_message": None,
                "scp_iterations_configured": int(self.config.scp_iterations),
                "scp_passes_attempted": 0,
                "scp_passes_succeeded": 0,
                "scp_stopped_after_pass": 0,
                "unrepairable_by_local_qp": False,
                "deep_penetration_flag": False,
                "pre_qp_min_clearance": float(risk_summary["min_clearance"]),
                "post_qp1_min_clearance": math.nan,
                "post_qp2_min_clearance": math.nan,
                "certificate_min_clearance": math.nan,
                "pass1_total_slack": math.nan,
                "pass2_total_slack": math.nan,
                "pass1_risk_segment_count": 0,
                "pass2_risk_segment_count": 0,
                "new_risk_segments_after_qp1": False,
                "scp_pass_details": [],
                "final_success_source": "failure",
                "local_waypoint_qp_attempted": False,
                "local_waypoint_qp_success": False,
                "local_waypoint_qp_skip_reason": None,
                "local_waypoint_qp_window_count": 0,
                "local_waypoint_qp_windows": [],
                "local_waypoint_qp_constraint_count": 0,
                "local_waypoint_qp_total_slack": math.nan,
                "local_waypoint_qp_solver_message": None,
                "post_local_waypoint_qp_min_clearance": math.nan,
                "post_local_waypoint_qp_certificate_success": False,
                "num_worst_timesteps": 0,
                "num_topk_constraints": 0,
                "risk_segment_count_total": 0,
                "risk_segment_count_selected": 0,
                "selected_risk_segments": [],
                "selected_window_timesteps": [],
                "selected_constraint_count": 0,
                "sdf_value_count": int(risk_summary["total_sdf_count"]),
                "finite_sdf_value_count": int(risk_summary["finite_sdf_count"]),
                "finite_sdf_timestep_count": int(risk_summary["finite_timestep_count"]),
                "control_points_normalized": control_points.astype(np.float32),
                "joint_trajectory": q_check_actual.astype(np.float32),
                "normalized_free_residual": np.asarray(free_residual, dtype=np.float32),
                "path_length": compute_path_length(q_check_actual),
                "smoothness": compute_smoothness(q_check_actual),
                "goal_error": 0.0,
                "candidate_time": 0.0,
            }
            candidate_infos.append(candidate_info)

        ranking = rank_screened_candidates(candidate_infos)
        log.num_candidates_total = int(candidate_residuals.shape[0])
        log.num_candidates_screened = int(len(candidate_infos))
        log.candidate_ranking = [int(idx) for idx in ranking]
        log.dp_time = float(sum(float(info["coarse_screening_time"]) for info in candidate_infos))

        max_repair_attempts = min(len(ranking), MAX_REPAIR_ATTEMPTS)
        repair_candidate_indices = ranking[:max_repair_attempts]
        actual_attempt_count = 0
        for attempt_order, candidate_index in enumerate(repair_candidate_indices, start=1):
            actual_attempt_count = attempt_order
            candidate_info = candidate_infos[candidate_index]
            self._repair_single_candidate(
                candidate_info=candidate_info,
                q_start_normalized=q_start_normalized,
                q_goal_normalized=q_goal_normalized,
                delta_w_mean=delta_w_mean,
                delta_w_std=delta_w_std,
                check_basis=check_basis,
                cert_basis=cert_basis,
                limit_basis=limit_basis,
                free_slice=free_slice,
                guidance_targets=guidance_targets,
                attempt_order=attempt_order,
                log=log,
            )
            if bool(candidate_info.get("certificate_success", False)):
                selected_candidate_info = candidate_info
                log.certificate_success = True
                log.selected_by_certificate = str(candidate_info.get("final_success_source", "certificate_success")) == "certificate_success"
                log.final_success_source = str(candidate_info.get("final_success_source", "certificate_success"))
                break

        attempted_candidate_indices = repair_candidate_indices[:actual_attempt_count]
        log.repair_attempt_count = int(actual_attempt_count)
        log.repair_attempted_candidate_indices = [int(idx) for idx in attempted_candidate_indices]
        attempted_candidates = [candidate_infos[idx] for idx in attempted_candidate_indices]
        if selected_candidate_info is None:
            attempted_candidates.sort(
                key=lambda info: (
                    -float(info["h_min_final"]) if math.isfinite(float(info["h_min_final"])) else math.inf,
                    float(info["path_length"]),
                    int(info["candidate_index"]),
                )
            )
            selected_candidate_info = attempted_candidates[0] if attempted_candidates else candidate_infos[ranking[0]]
            if self.config.fallback_to_terminal_cbf:
                fallback = self.environment.try_existing_terminal_cbf(selected_candidate_info["joint_trajectory"])
            log.fallback_used = fallback is not None
            log.used_existing_terminal_cbf = fallback is not None
            log.final_success_source = "fallback_success" if fallback is not None else "failure"

        selected_index = int(selected_candidate_info["candidate_index"])
        log.best_candidate_index = selected_index
        log.selected_candidate_index = selected_index
        log.goal_error = float(selected_candidate_info["goal_error"])
        log.smoothness = float(selected_candidate_info["smoothness"])
        log.path_length = float(selected_candidate_info["path_length"])
        log.h_min_before_guidance = float(selected_candidate_info["h_min_before_guidance"])
        log.h_min_after_guidance = float(selected_candidate_info["h_min_after_guidance"])
        log.h_min_final = float(selected_candidate_info["h_min_final"])
        log.selected_candidate_pass_count = int(selected_candidate_info.get("scp_passes_attempted", 0))
        log.selected_candidate_passes_succeeded = int(selected_candidate_info.get("scp_passes_succeeded", 0))
        log.selected_candidate_pre_qp_min_clearance = float(selected_candidate_info.get("pre_qp_min_clearance", math.nan))
        log.selected_candidate_post_qp1_min_clearance = float(selected_candidate_info.get("post_qp1_min_clearance", math.nan))
        log.selected_candidate_post_qp2_min_clearance = float(selected_candidate_info.get("post_qp2_min_clearance", math.nan))
        log.selected_candidate_certificate_min_clearance = float(selected_candidate_info.get("certificate_min_clearance", math.nan))
        log.total_time = time.perf_counter() - start_time
        return GuidanceResult(
            best_index=selected_index,
            best_normalized_free_residual=np.asarray(selected_candidate_info["normalized_free_residual"], dtype=np.float32),
            best_control_points_normalized=np.asarray(selected_candidate_info["control_points_normalized"], dtype=np.float32),
            best_joint_trajectory=np.asarray(selected_candidate_info["joint_trajectory"], dtype=np.float32),
            candidate_infos=candidate_infos,
            log=log,
        )

    def _repair_single_candidate(
        self,
        *,
        candidate_info: dict[str, Any],
        q_start_normalized: np.ndarray,
        q_goal_normalized: np.ndarray,
        delta_w_mean: np.ndarray,
        delta_w_std: np.ndarray,
        check_basis: np.ndarray,
        cert_basis: np.ndarray,
        limit_basis: np.ndarray,
        free_slice: slice,
        guidance_targets: np.ndarray,
        attempt_order: int,
        log: GuidanceLog,
    ) -> None:
        candidate_start_time = time.perf_counter()
        _ = guidance_targets
        original_control_points = np.asarray(candidate_info["control_points_normalized"], dtype=np.float32)
        current_control_points = original_control_points.copy()
        initial_state = self._evaluate_candidate_state(
            control_points=original_control_points,
            check_basis=check_basis,
            d_trigger=float(self.config.d_trigger),
        )
        initial_summary = initial_state["risk_summary"]
        initial_min_clearance = float(initial_summary["min_clearance"])
        guided_control_points = original_control_points.copy()
        qp_attempted = False
        qp_success = False
        qp_skip_reason: str | None = None
        pass_results: list[dict[str, Any]] = []

        candidate_info["repair_attempted"] = True
        candidate_info["repair_attempt_order"] = int(attempt_order)
        repair_mode = classify_candidate_repair(
            min_clearance=initial_min_clearance,
            d_trigger=float(self.config.d_trigger),
            eps_deep=float(self.config.eps_deep),
        )
        deep_penetration_flag = repair_mode == "deep"
        safe_enough = repair_mode == "safe"

        if deep_penetration_flag:
            qp_skip_reason = "deep_penetration_unrepairable"
        elif safe_enough:
            qp_skip_reason = "safe_candidate_no_repair"
        else:
            for pass_index in range(int(self.config.scp_iterations)):
                pass_trigger = compute_scp_pass_trigger(
                    d_trigger=float(self.config.d_trigger),
                    pass_index=pass_index,
                    pass2_offset=float(self.config.d_trigger_pass2_offset),
                )
                state = self._evaluate_candidate_state(
                    control_points=current_control_points,
                    check_basis=check_basis,
                    d_trigger=pass_trigger,
                )
                pass_detail = self._build_scp_pass_detail(
                    pass_index=pass_index,
                    state=state,
                    check_basis=check_basis,
                    d_trigger=pass_trigger,
                )
                pass_results.append(pass_detail)
                if pass_index == 1 and bool(pass_detail["risk_segment_count_total"] > 0):
                    prev_ids = {
                        tuple(segment["timesteps"])
                        for segment in pass_results[0].get("risk_segments_raw", [])
                    }
                    curr_ids = {
                        tuple(segment["timesteps"])
                        for segment in pass_detail.get("risk_segments_raw", [])
                    }
                    candidate_info["new_risk_segments_after_qp1"] = bool(curr_ids - prev_ids)
                if not np.isfinite(float(state["risk_summary"]["min_margin"])):
                    qp_skip_reason = "non_finite_h_min_before"
                    pass_detail["skip_reason"] = qp_skip_reason
                    break
                if not pass_detail["risk_segments_raw"]:
                    qp_skip_reason = None if pass_index > 0 else "no_risk_segments"
                    pass_detail["skip_reason"] = qp_skip_reason
                    break
                if not pass_detail["active_constraints"]:
                    qp_skip_reason = "no_topk_constraints"
                    pass_detail["skip_reason"] = qp_skip_reason
                    break

                qp_attempted = True
                candidate_info["scp_passes_attempted"] += 1
                log.num_qp_called += 1
                log.scp_passes_attempted_total += 1
                log.num_active_constraints += int(pass_detail["selected_constraint_count"])
                guidance_start = time.perf_counter()
                qp_result = self._solve_guidance_qp(
                    base_control_points=current_control_points,
                    reference_control_points=original_control_points,
                    active_constraints=pass_detail["active_constraints"],
                    target_margin=float(self.config.d_cert) + float(self.config.margin_buffer),
                    limit_basis=limit_basis,
                    free_slice=free_slice,
                    delta_max_local=float(
                        self.config.delta_max_pass1 if pass_index == 0 else self.config.delta_max_pass2
                    ),
                    delta_max_total=float(self.config.delta_max_total),
                )
                log.guidance_time += time.perf_counter() - guidance_start
                if qp_result is not None:
                    log.qp_time += float(qp_result.get("solve_time", 0.0))
                pass_detail["solver_success"] = bool(qp_result is not None and qp_result.get("success", False))
                pass_detail["solver_message"] = None if qp_result is None else qp_result.get("message")
                pass_detail["total_slack"] = float(
                    np.sum(np.asarray(qp_result.get("slack", []), dtype=np.float32))
                ) if qp_result is not None else math.nan
                if qp_result is None or not bool(qp_result.get("success", False)):
                    qp_skip_reason = "solver_failure"
                    candidate_info["qp_solver_message"] = None if qp_result is None else qp_result.get("message")
                    break
                current_control_points = np.asarray(qp_result["control_points"], dtype=np.float32)
                guided_control_points = current_control_points.copy()
                candidate_info["scp_passes_succeeded"] += 1
                log.num_qp_success += 1
                log.scp_passes_succeeded_total += 1
                pass_detail["control_points_updated"] = True
                updated_state = self._evaluate_candidate_state(
                    control_points=current_control_points,
                    check_basis=check_basis,
                    d_trigger=pass_trigger,
                )
                pass_detail["post_min_clearance"] = float(updated_state["risk_summary"]["min_clearance"])
            qp_success = bool(qp_attempted and candidate_info["scp_passes_attempted"] == candidate_info["scp_passes_succeeded"])
            if qp_attempted and qp_skip_reason is None and candidate_info["scp_passes_attempted"] < int(self.config.scp_iterations):
                qp_skip_reason = "no_risk_segments"

        after_state = self._evaluate_candidate_state(
            control_points=guided_control_points,
            check_basis=check_basis,
            d_trigger=float(self.config.d_trigger),
        )
        after_summary = after_state["risk_summary"]

        certificate_start = time.perf_counter()
        cert_result = self._certificate_check(
            control_points=guided_control_points,
            cert_basis=cert_basis,
        )
        log.certificate_time += time.perf_counter() - certificate_start
        local_waypoint_qp_result = {"attempted": False, "success": False, "skip_reason": None}
        final_joint_trajectory = np.asarray(cert_result["joint_trajectory"], dtype=np.float32)
        final_cert_success = bool(cert_result["success"])
        final_success_source = "certificate_success" if final_cert_success else "failure"
        if not final_cert_success:
            local_waypoint_qp_result = self._try_local_waypoint_qp(
                cert_result=cert_result,
                log=log,
            )
            if bool(local_waypoint_qp_result.get("success", False)):
                cert_result = dict(local_waypoint_qp_result["recert_result"])
                final_joint_trajectory = np.asarray(local_waypoint_qp_result["joint_trajectory"], dtype=np.float32)
                final_cert_success = True
                final_success_source = "local_qp_success"
            else:
                final_joint_trajectory = np.asarray(cert_result["joint_trajectory"], dtype=np.float32)
        serialized_pass_details = [self._serialize_pass_detail(detail) for detail in pass_results]
        while len(serialized_pass_details) < int(self.config.scp_iterations):
            serialized_pass_details.append(
                self._serialize_pass_detail(
                    {
                        "pass_index": len(serialized_pass_details) + 1,
                        "d_trigger": float(self.config.d_trigger)
                        + (float(self.config.d_trigger_pass2_offset) if len(serialized_pass_details) == 1 else 0.0),
                        "risk_segment_count_total": 0,
                        "risk_segment_count_selected": 0,
                        "selected_segments": [],
                        "selected_window_timesteps": [],
                        "selected_constraint_count": 0,
                        "solver_success": False,
                        "solver_message": None,
                        "total_slack": math.nan,
                        "skip_reason": None,
                        "post_min_clearance": math.nan,
                    }
                )
            )

        candidate_info.update(
            {
                "h_min_before_guidance": float(initial_summary["min_margin"]),
                "h_min_after_guidance": float(after_summary["min_margin"]),
                "h_min_final": float(cert_result["h_min_final"]),
                "certificate_success": bool(final_cert_success),
                "qp_attempted": bool(qp_attempted),
                "qp_success": bool(qp_success),
                "qp_skip_reason": qp_skip_reason,
                "scp_stopped_after_pass": int(candidate_info["scp_passes_attempted"]),
                "unrepairable_by_local_qp": bool(deep_penetration_flag),
                "deep_penetration_flag": bool(deep_penetration_flag),
                "pre_qp_min_clearance": float(initial_summary["min_clearance"]),
                "post_qp1_min_clearance": float(pass_results[0].get("post_min_clearance", math.nan)) if pass_results else math.nan,
                "post_qp2_min_clearance": float(pass_results[1].get("post_min_clearance", math.nan)) if len(pass_results) > 1 else math.nan,
                "certificate_min_clearance": float(cert_result["min_clearance"]),
                "pass1_total_slack": float(pass_results[0].get("total_slack", math.nan)) if pass_results else math.nan,
                "pass2_total_slack": float(pass_results[1].get("total_slack", math.nan)) if len(pass_results) > 1 else math.nan,
                "pass1_risk_segment_count": int(pass_results[0].get("risk_segment_count_total", 0)) if pass_results else 0,
                "pass2_risk_segment_count": int(pass_results[1].get("risk_segment_count_total", 0)) if len(pass_results) > 1 else 0,
                "scp_pass_details": serialized_pass_details,
                "final_success_source": final_success_source,
                "local_waypoint_qp_attempted": bool(local_waypoint_qp_result.get("attempted", False)),
                "local_waypoint_qp_success": bool(local_waypoint_qp_result.get("success", False)),
                "local_waypoint_qp_skip_reason": local_waypoint_qp_result.get("skip_reason"),
                "local_waypoint_qp_window_count": int(len(local_waypoint_qp_result.get("windows", []) or [])),
                "local_waypoint_qp_windows": list(local_waypoint_qp_result.get("windows", []) or []),
                "local_waypoint_qp_constraint_count": int(local_waypoint_qp_result.get("constraint_count", 0) or 0),
                "local_waypoint_qp_total_slack": float(local_waypoint_qp_result.get("total_slack", math.nan)),
                "local_waypoint_qp_solver_message": local_waypoint_qp_result.get("solver_message"),
                "post_local_waypoint_qp_min_clearance": float(
                    math.nan
                    if not local_waypoint_qp_result.get("success", False)
                    else local_waypoint_qp_result["recert_result"]["min_clearance"]
                ),
                "post_local_waypoint_qp_certificate_success": bool(local_waypoint_qp_result.get("success", False)),
                "num_worst_timesteps": int(sum(len(segment.get("anchor_timesteps", [])) for detail in pass_results for segment in detail.get("selected_segments", []))),
                "num_topk_constraints": int(max((detail.get("selected_constraint_count", 0) for detail in pass_results), default=0)),
                "risk_segment_count_total": int(pass_results[-1].get("risk_segment_count_total", 0)) if pass_results else int(len(initial_state["risk_segments"])),
                "risk_segment_count_selected": int(pass_results[-1].get("risk_segment_count_selected", 0)) if pass_results else 0,
                "selected_risk_segments": list(pass_results[-1].get("selected_segments", [])) if pass_results else [],
                "selected_window_timesteps": [int(v) for v in pass_results[-1].get("selected_window_timesteps", [])] if pass_results else [],
                "selected_constraint_count": int(pass_results[-1].get("selected_constraint_count", 0)) if pass_results else 0,
                "control_points_normalized": guided_control_points.astype(np.float32),
                "joint_trajectory": final_joint_trajectory,
                "normalized_free_residual": control_points_to_normalized_free_residual(
                    guided_control_points,
                    q_start_normalized=q_start_normalized,
                    q_goal_normalized=q_goal_normalized,
                    delta_w_mean=delta_w_mean,
                    delta_w_std=delta_w_std,
                ),
                "path_length": compute_path_length(final_joint_trajectory),
                "smoothness": compute_smoothness(final_joint_trajectory),
                "goal_error": 0.0,
                "candidate_time": time.perf_counter() - candidate_start_time,
            }
        )
        log.risk_segment_count_total = int(candidate_info["risk_segment_count_total"])
        log.risk_segment_count_selected = int(candidate_info["risk_segment_count_selected"])
        log.selected_risk_segments = list(candidate_info["selected_risk_segments"])
        log.selected_window_timesteps = list(candidate_info["selected_window_timesteps"])
        log.selected_constraint_count = int(candidate_info["selected_constraint_count"])
        log.selected_candidate_pass_count = int(candidate_info["scp_passes_attempted"])
        log.selected_candidate_passes_succeeded = int(candidate_info["scp_passes_succeeded"])
        log.selected_candidate_pre_qp_min_clearance = float(candidate_info["pre_qp_min_clearance"])
        log.selected_candidate_post_qp1_min_clearance = float(candidate_info["post_qp1_min_clearance"])
        log.selected_candidate_post_qp2_min_clearance = float(candidate_info["post_qp2_min_clearance"])
        log.selected_candidate_certificate_min_clearance = float(candidate_info["certificate_min_clearance"])
        log.selected_candidate_local_waypoint_qp_attempted = bool(candidate_info["local_waypoint_qp_attempted"])
        log.selected_candidate_local_waypoint_qp_success = bool(candidate_info["local_waypoint_qp_success"])
        log.selected_candidate_post_local_waypoint_qp_min_clearance = float(candidate_info["post_local_waypoint_qp_min_clearance"])
        log.final_success_source = str(candidate_info["final_success_source"])

    def _evaluate_candidate_state(
        self,
        *,
        control_points: np.ndarray,
        check_basis: np.ndarray,
        d_trigger: float,
    ) -> dict[str, Any]:
        q_check_norm = check_basis @ np.asarray(control_points, dtype=np.float32)
        q_check_actual = self.environment.normalized_to_actual(q_check_norm)
        sdf_result = self.environment.collect_joint_trajectory_sdf_with_link_details_any_length(q_check_actual)
        risk_summary = summarize_sdf_risk(
            sdf_result=sdf_result,
            d_safe=float(self.config.d_safe),
            d_trigger=float(d_trigger),
        )
        risk_segments = build_risk_segments(
            sdf_result=sdf_result,
            d_trigger=float(d_trigger),
        )
        return {
            "q_check_norm": q_check_norm,
            "q_check_actual": q_check_actual,
            "sdf_result": sdf_result,
            "risk_summary": risk_summary,
            "risk_segments": risk_segments,
        }

    def _build_scp_pass_detail(
        self,
        *,
        pass_index: int,
        state: dict[str, Any],
        check_basis: np.ndarray,
        d_trigger: float,
    ) -> dict[str, Any]:
        risk_segments = list(state["risk_segments"])
        selected_segments = risk_segments[: int(self.config.max_risk_segments)]
        constraints, selected_segment_summaries, selected_window_timesteps = build_segment_window_cbf_constraints(
            segments=selected_segments,
            sdf_result=state["sdf_result"],
            check_basis=check_basis,
            q_check_norm=state["q_check_norm"],
            environment=self.environment,
            d_trigger=float(d_trigger),
            points_per_segment=int(self.config.points_per_segment),
            min_constraints_per_segment=int(self.config.min_constraints_per_segment),
            window_radius=int(self.config.window_radius),
            max_active=int(self.config.active_constraints),
        ) if selected_segments else ([], [], [])
        return {
            "pass_index": int(pass_index + 1),
            "d_trigger": float(d_trigger),
            "risk_segments_raw": risk_segments,
            "risk_segment_count_total": int(len(risk_segments)),
            "risk_segment_count_selected": int(len(selected_segment_summaries)),
            "selected_segments": selected_segment_summaries,
            "selected_window_timesteps": [int(v) for v in selected_window_timesteps],
            "selected_constraint_count": int(len(constraints)),
            "active_constraints": constraints,
            "solver_success": False,
            "solver_message": None,
            "total_slack": math.nan,
            "skip_reason": None,
            "control_points_updated": False,
            "post_min_clearance": math.nan,
        }

    def _serialize_pass_detail(self, detail: dict[str, Any]) -> dict[str, Any]:
        return {
            "pass_index": int(detail["pass_index"]),
            "d_trigger": float(detail["d_trigger"]),
            "risk_segment_count_total": int(detail["risk_segment_count_total"]),
            "risk_segment_count_selected": int(detail["risk_segment_count_selected"]),
            "selected_segments": list(detail["selected_segments"]),
            "selected_window_timesteps": [int(v) for v in detail["selected_window_timesteps"]],
            "selected_constraint_count": int(detail["selected_constraint_count"]),
            "solver_success": bool(detail["solver_success"]),
            "solver_message": detail["solver_message"],
            "total_slack": float(detail["total_slack"]) if detail["total_slack"] is not None else math.nan,
            "skip_reason": detail["skip_reason"],
            "post_min_clearance": float(detail["post_min_clearance"]),
        }

    def _certificate_check_joint_trajectory(self, *, joint_trajectory_actual: np.ndarray) -> dict[str, Any]:
        joint_trajectory_actual = np.asarray(joint_trajectory_actual, dtype=np.float32)
        q_cert_swept = interpolate_swept_segments(joint_trajectory_actual, int(self.config.cert_swept_intermediate))
        sdf_cert = self.environment.collect_joint_trajectory_sdf_with_link_details_any_length(q_cert_swept)
        h_cert = flatten_margin_values(sdf_cert, d_safe=float(self.config.d_safe))
        all_sdf = np.asarray(sdf_cert.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
        finite_sdf = all_sdf[np.isfinite(all_sdf)]
        min_clearance_per_step = np.full(all_sdf.shape[0], np.nan, dtype=np.float32) if all_sdf.ndim == 2 else np.empty((0,), dtype=np.float32)
        if all_sdf.ndim == 2 and all_sdf.size > 0:
            finite_mask = np.isfinite(all_sdf)
            has_finite = np.any(finite_mask, axis=1)
            if np.any(has_finite):
                min_clearance_per_step[has_finite] = np.min(
                    np.where(finite_mask[has_finite], all_sdf[has_finite], np.inf),
                    axis=1,
                )
        h_min_final = float(np.min(h_cert)) if h_cert.size > 0 else math.nan
        min_clearance = float(np.min(finite_sdf)) if finite_sdf.size > 0 else math.nan
        return {
            "success": bool(np.isfinite(h_min_final) and h_min_final >= float(self.config.d_cert)),
            "h_min_final": h_min_final,
            "min_clearance": min_clearance,
            "joint_trajectory": joint_trajectory_actual.astype(np.float32),
            "certificate_joint_trajectory": q_cert_swept.astype(np.float32),
            "sdf_result": sdf_cert,
            "min_clearance_per_step": min_clearance_per_step.astype(np.float32),
        }

    def _build_local_waypoint_constraints(
        self,
        *,
        joint_trajectory_actual: np.ndarray,
        sdf_result: dict[str, Any],
        window_timesteps: list[int],
        target_clearance: float,
    ) -> list[dict[str, Any]]:
        all_sdf = np.asarray(sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
        if all_sdf.size == 0 or all_sdf.ndim != 2:
            return []
        constraints: list[dict[str, Any]] = []
        for timestep in window_timesteps:
            if timestep < 0 or timestep >= all_sdf.shape[0]:
                continue
            step_sdf = np.asarray(all_sdf[timestep], dtype=np.float32).reshape(-1)
            finite_indices = np.flatnonzero(np.isfinite(step_sdf))
            if finite_indices.size == 0:
                continue
            worst_flat_index = int(finite_indices[np.argmin(step_sdf[finite_indices])])
            clearance = float(step_sdf[worst_flat_index])
            if clearance >= float(target_clearance):
                continue
            sample_info = _flat_index_to_surface_sample(self.environment.surface_samples, worst_flat_index)
            if sample_info is None:
                continue
            q_actual = np.asarray(joint_trajectory_actual[timestep], dtype=np.float32).reshape(-1)
            point_world = self.environment.surface_point_world(
                link_index=int(sample_info["link_index"]),
                local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
            )
            grad_world = approximate_sdf_gradient(self.environment.load_sdf_grid(), point_world)
            jacobian = self.environment.surface_point_jacobian(
                link_index=int(sample_info["link_index"]),
                local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
                q_actual=q_actual,
            )
            j_pos = np.asarray(jacobian[:3, :], dtype=np.float32)
            g_actual = (j_pos.T @ grad_world.reshape(3)).astype(np.float32)
            constraints.append(
                {
                    "t_index": int(timestep),
                    "clearance": clearance,
                    "target_clearance": float(target_clearance),
                    "g_actual": g_actual,
                    "link_index": int(sample_info["link_index"]),
                    "point_index": int(sample_info["point_index"]),
                }
            )
        return constraints

    def _solve_local_waypoint_qp(
        self,
        *,
        joint_trajectory_actual: np.ndarray,
        windows: list[dict[str, Any]],
        active_constraints: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not active_constraints or not windows:
            return None
        base = np.asarray(joint_trajectory_actual, dtype=np.float32)
        dof = base.shape[1]
        variable_timesteps = sorted(
            {
                int(t)
                for window in windows
                for t in range(int(window["window_start"]) + 1, int(window["window_end"]))
            }
        )
        if not variable_timesteps:
            return None
        timestep_to_local = {t: i for i, t in enumerate(variable_timesteps)}
        num_selected = len(variable_timesteps)
        num_slack = len(active_constraints)
        joint_lower = self.environment.joint_lower_limits.astype(np.float64)
        joint_upper = self.environment.joint_upper_limits.astype(np.float64)
        lower_bounds = np.full(num_selected * dof + num_slack, -np.inf, dtype=np.float64)
        upper_bounds = np.full(num_selected * dof + num_slack, np.inf, dtype=np.float64)
        for timestep, local_index in timestep_to_local.items():
            q_base = base[timestep].astype(np.float64)
            local_lower = np.maximum(-float(self.config.local_waypoint_qp_delta_max), joint_lower - q_base)
            local_upper = np.minimum(float(self.config.local_waypoint_qp_delta_max), joint_upper - q_base)
            lower_bounds[local_index * dof : (local_index + 1) * dof] = local_lower
            upper_bounds[local_index * dof : (local_index + 1) * dof] = local_upper
        lower_bounds[num_selected * dof :] = 0.0

        linear_rows = []
        linear_lb = []
        linear_ub = []
        for slack_index, constraint in enumerate(active_constraints):
            timestep = int(constraint["t_index"])
            if timestep not in timestep_to_local:
                continue
            row = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
            local_index = timestep_to_local[timestep]
            row[local_index * dof : (local_index + 1) * dof] = np.asarray(constraint["g_actual"], dtype=np.float64)
            row[num_selected * dof + slack_index] = 1.0
            linear_rows.append(row)
            linear_lb.append(float(constraint["target_clearance"]) - float(constraint["clearance"]))
            linear_ub.append(np.inf)

        max_vel = float(self.config.local_waypoint_qp_max_velocity_step)
        for timestep in range(base.shape[0] - 1):
            base_diff = base[timestep + 1].astype(np.float64) - base[timestep].astype(np.float64)
            for joint_index in range(dof):
                row = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
                if timestep in timestep_to_local:
                    row[timestep_to_local[timestep] * dof + joint_index] -= 1.0
                if (timestep + 1) in timestep_to_local:
                    row[timestep_to_local[timestep + 1] * dof + joint_index] += 1.0
                if not np.any(row):
                    continue
                linear_rows.append(row.copy())
                linear_lb.append(-np.inf)
                linear_ub.append(max_vel - float(base_diff[joint_index]))
                linear_rows.append(row)
                linear_lb.append(-max_vel - float(base_diff[joint_index]))
                linear_ub.append(np.inf)

        max_acc = float(self.config.local_waypoint_qp_max_acceleration_step)
        for timestep in range(1, base.shape[0] - 1):
            base_acc = base[timestep + 1].astype(np.float64) - 2.0 * base[timestep].astype(np.float64) + base[timestep - 1].astype(np.float64)
            for joint_index in range(dof):
                row = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
                if (timestep - 1) in timestep_to_local:
                    row[timestep_to_local[timestep - 1] * dof + joint_index] += 1.0
                if timestep in timestep_to_local:
                    row[timestep_to_local[timestep] * dof + joint_index] -= 2.0
                if (timestep + 1) in timestep_to_local:
                    row[timestep_to_local[timestep + 1] * dof + joint_index] += 1.0
                if not np.any(row):
                    continue
                linear_rows.append(row.copy())
                linear_lb.append(-np.inf)
                linear_ub.append(max_acc - float(base_acc[joint_index]))
                linear_rows.append(row)
                linear_lb.append(-max_acc - float(base_acc[joint_index]))
                linear_ub.append(np.inf)

        constraints = []
        if linear_rows:
            constraints.append(
                LinearConstraint(
                    np.stack(linear_rows, axis=0),
                    np.asarray(linear_lb, dtype=np.float64),
                    np.asarray(linear_ub, dtype=np.float64),
                )
            )

        def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            delta_selected = np.asarray(vector[: num_selected * dof], dtype=np.float64).reshape(num_selected, dof)
            slack = np.asarray(vector[num_selected * dof :], dtype=np.float64)
            return delta_selected, slack

        def objective(vector: np.ndarray) -> float:
            delta_selected, slack = unpack(vector)
            delta_full = np.zeros_like(base, dtype=np.float64)
            for timestep, local_index in timestep_to_local.items():
                delta_full[timestep] = delta_selected[local_index]
            smooth_term = delta_full[2:] - 2.0 * delta_full[1:-1] + delta_full[:-2]
            return float(
                np.sum(delta_selected ** 2)
                + float(self.config.local_waypoint_qp_lambda_s) * np.sum(smooth_term ** 2)
                + float(self.config.rho) * np.sum(slack ** 2)
            )

        x0 = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
        solve_start = time.perf_counter()
        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=Bounds(lower_bounds, upper_bounds),
                constraints=constraints,
                options={"maxiter": int(self.config.local_waypoint_qp_maxiter), "ftol": 1e-6, "disp": False},
            )
        except Exception as exc:
            return {
                "success": False,
                "solve_time": time.perf_counter() - solve_start,
                "message": str(exc),
                "slack": np.empty((0,), dtype=np.float32),
            }
        delta_selected, slack = unpack(np.asarray(result.x, dtype=np.float64))
        repaired = base.astype(np.float64).copy()
        for timestep, local_index in timestep_to_local.items():
            repaired[timestep] += delta_selected[local_index]
        return {
            "success": bool(result.success),
            "joint_trajectory": repaired.astype(np.float32),
            "slack": np.asarray(slack, dtype=np.float32),
            "message": str(result.message),
            "solve_time": time.perf_counter() - solve_start,
        }

    def _try_local_waypoint_qp(
        self,
        *,
        cert_result: dict[str, Any],
        log: GuidanceLog,
    ) -> dict[str, Any]:
        collision_threshold = float(self.config.d_safe) + float(self.config.d_cert)
        if not should_attempt_local_waypoint_qp(
            enable_local_waypoint_qp_after_certificate=bool(self.config.enable_local_waypoint_qp_after_certificate),
            min_clearance=float(cert_result["min_clearance"]),
            min_clearance_trigger=float(self.config.local_waypoint_qp_min_clearance_trigger),
            collision_threshold=collision_threshold,
        ):
            return {"attempted": False, "success": False, "skip_reason": "local_waypoint_qp_not_applicable"}
        windows = build_collision_windows_from_clearance(
            min_clearance_per_step=np.asarray(cert_result["min_clearance_per_step"], dtype=np.float32),
            collision_threshold=collision_threshold,
            window_radius=int(self.config.local_waypoint_qp_window_radius),
            max_segments=int(self.config.local_waypoint_qp_max_collision_segments),
        )
        if not windows:
            return {"attempted": False, "success": False, "skip_reason": "no_local_collision_windows"}
        active_constraints: list[dict[str, Any]] = []
        for window in windows:
            active_constraints.extend(
                self._build_local_waypoint_constraints(
                    joint_trajectory_actual=np.asarray(cert_result["certificate_joint_trajectory"], dtype=np.float32),
                    sdf_result=cert_result["sdf_result"],
                    window_timesteps=[int(v) for v in window["window_timesteps"]],
                    target_clearance=float(self.config.d_safe) + float(self.config.local_waypoint_qp_target_buffer),
                )
            )
        log.local_waypoint_qp_attempted_total += 1
        if not active_constraints:
            return {
                "attempted": True,
                "success": False,
                "skip_reason": "no_local_waypoint_constraints",
                "windows": windows,
                "constraint_count": 0,
            }
        solve_result = self._solve_local_waypoint_qp(
            joint_trajectory_actual=np.asarray(cert_result["certificate_joint_trajectory"], dtype=np.float32),
            windows=windows,
            active_constraints=active_constraints,
        )
        if solve_result is None or not bool(solve_result.get("success", False)):
            return {
                "attempted": True,
                "success": False,
                "skip_reason": "local_waypoint_solver_failure",
                "windows": windows,
                "constraint_count": int(len(active_constraints)),
                "solver_message": None if solve_result is None else solve_result.get("message"),
                "total_slack": math.nan if solve_result is None else float(np.sum(np.asarray(solve_result.get("slack", []), dtype=np.float32))),
            }
        recert_result = self._certificate_check_joint_trajectory(
            joint_trajectory_actual=np.asarray(solve_result["joint_trajectory"], dtype=np.float32)
        )
        if bool(recert_result["success"]):
            log.local_waypoint_qp_success_total += 1
        return {
            "attempted": True,
            "success": bool(recert_result["success"]),
            "skip_reason": None if bool(recert_result["success"]) else "local_waypoint_certificate_failure",
            "windows": windows,
            "constraint_count": int(len(active_constraints)),
            "solver_message": solve_result.get("message"),
            "total_slack": float(np.sum(np.asarray(solve_result.get("slack", []), dtype=np.float32))),
            "joint_trajectory": np.asarray(solve_result["joint_trajectory"], dtype=np.float32),
            "recert_result": recert_result,
        }

    def _certificate_check(self, *, control_points: np.ndarray, cert_basis: np.ndarray) -> dict[str, Any]:
        q_cert_norm = cert_basis @ control_points
        q_cert_actual = self.environment.normalized_to_actual(q_cert_norm)
        return self._certificate_check_joint_trajectory(joint_trajectory_actual=q_cert_actual)

    def _solve_guidance_qp(
        self,
        *,
        base_control_points: np.ndarray,
        reference_control_points: np.ndarray,
        active_constraints: list[dict[str, Any]],
        target_margin: float,
        limit_basis: np.ndarray,
        free_slice: slice,
        delta_max_local: float,
        delta_max_total: float,
    ) -> dict[str, Any] | None:
        solve_start = time.perf_counter()
        dof = base_control_points.shape[1]
        num_slack = len(active_constraints)
        if num_slack <= 0:
            return None
        base_control_points = np.asarray(base_control_points, dtype=np.float32)
        reference_control_points = np.asarray(reference_control_points, dtype=np.float32)

        # ---- identify which free control points influence the active constraints ----
        all_free_indices = list(range(free_slice.start, free_slice.stop))
        influencing = set()
        for constraint in active_constraints:
            basis_row = np.asarray(constraint["basis_row"], dtype=np.float32)
            for cp_idx in all_free_indices:
                if abs(float(basis_row[cp_idx])) > 1e-8:
                    influencing.add(int(cp_idx))
        selected_free_indices = sorted(influencing)
        if not selected_free_indices:
            return None
        num_selected = len(selected_free_indices)
        # Map selected (global) control-point indices → local QP indices 0..num_selected-1
        selected_to_local = {cp: i for i, cp in enumerate(selected_free_indices)}

        d2_matrix = build_second_difference_matrix(num_control_points=base_control_points.shape[0])
        lower_bounds = np.full(num_selected * dof + num_slack, -np.inf, dtype=np.float64)
        upper_bounds = np.full(num_selected * dof + num_slack, np.inf, dtype=np.float64)
        for cp_idx, local_idx in selected_to_local.items():
            base_delta_total = (
                base_control_points[cp_idx].astype(np.float64) - reference_control_points[cp_idx].astype(np.float64)
            )
            local_lower, local_upper = compute_delta_box_bounds(
                base_delta_total=base_delta_total,
                delta_max_local=float(delta_max_local),
                delta_max_total=float(delta_max_total),
            )
            if np.any(local_lower > local_upper):
                return {
                    "success": False,
                    "solve_time": time.perf_counter() - solve_start,
                    "message": "infeasible cumulative delta bounds",
                    "slack": np.empty((0,), dtype=np.float32),
                }
            lower_bounds[local_idx * dof : (local_idx + 1) * dof] = local_lower
            upper_bounds[local_idx * dof : (local_idx + 1) * dof] = local_upper
        lower_bounds[num_selected * dof :] = 0.0
        joint_lower_norm = np.full(dof, -1.0, dtype=np.float32)
        joint_upper_norm = np.full(dof, 1.0, dtype=np.float32)
        joint_scale = self.environment.joint_scale().reshape(-1)

        linear_rows = []
        linear_lb = []
        linear_ub = []

        # CBF constraints collected from the selected risk-segment windows.
        for slack_index, constraint in enumerate(active_constraints):
            row = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
            g_actual = np.asarray(constraint["g_actual"], dtype=np.float32).reshape(-1)
            g_norm = g_actual * joint_scale
            basis_row = np.asarray(constraint["basis_row"], dtype=np.float32)
            for cp_idx, local_idx in selected_to_local.items():
                coeff = float(basis_row[cp_idx])
                row[local_idx * dof : (local_idx + 1) * dof] = coeff * g_norm.astype(np.float64)
            row[num_selected * dof + slack_index] = 1.0
            linear_rows.append(row)
            linear_lb.append(float(target_margin) - float(constraint["h_value"]))
            linear_ub.append(np.inf)

        # Joint-limit constraints (all joint-limit timesteps, but only selected control points)
        for basis_row in np.asarray(limit_basis, dtype=np.float32):
            for joint_index in range(dof):
                upper_row = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
                lower_row = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
                for cp_idx, local_idx in selected_to_local.items():
                    coeff = float(basis_row[cp_idx])
                    upper_row[local_idx * dof + joint_index] = coeff
                    lower_row[local_idx * dof + joint_index] = coeff
                current_value = float((basis_row @ base_control_points[:, joint_index]).item())
                linear_rows.append(upper_row)
                linear_lb.append(-np.inf)
                linear_ub.append(float(joint_upper_norm[joint_index] - current_value))
                linear_rows.append(lower_row)
                linear_lb.append(float(joint_lower_norm[joint_index] - current_value))
                linear_ub.append(np.inf)

        constraints = []
        if linear_rows:
            constraints.append(
                LinearConstraint(
                    np.stack(linear_rows, axis=0),
                    np.asarray(linear_lb, dtype=np.float64),
                    np.asarray(linear_ub, dtype=np.float64),
                )
            )

        def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            delta_selected = np.asarray(vector[: num_selected * dof], dtype=np.float64).reshape(num_selected, dof)
            slack = np.asarray(vector[num_selected * dof :], dtype=np.float64)
            return delta_selected, slack

        def objective(vector: np.ndarray) -> float:
            delta_selected, slack = unpack(vector)
            delta_full = np.zeros_like(base_control_points, dtype=np.float64)
            for cp_idx, local_idx in selected_to_local.items():
                delta_full[cp_idx] = delta_selected[local_idx]
            control_new = base_control_points.astype(np.float64) + delta_full
            smooth_term = d2_matrix @ control_new
            return float(
                np.sum(delta_selected ** 2)
                + float(self.config.lambda_s) * np.sum(smooth_term ** 2)
                + float(self.config.rho) * np.sum(slack ** 2)
            )

        x0 = np.zeros(num_selected * dof + num_slack, dtype=np.float64)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds during a minimize step, clipping to bounds",
                    category=RuntimeWarning,
                )
                result = minimize(
                    objective,
                    x0,
                    method="SLSQP",
                    bounds=Bounds(lower_bounds, upper_bounds),
                    constraints=constraints,
                    options={"maxiter": 200, "ftol": 1e-6, "disp": False},
                )
        except Exception as exc:
            return {
                "success": False,
                "solve_time": time.perf_counter() - solve_start,
                "error": str(exc),
            }
        delta_selected, slack = unpack(np.asarray(result.x, dtype=np.float64))
        delta_full = np.zeros_like(base_control_points, dtype=np.float64)
        for cp_idx, local_idx in selected_to_local.items():
            delta_full[cp_idx] = delta_selected[local_idx]
        control_new = (base_control_points.astype(np.float64) + delta_full).astype(np.float32)
        control_new[:FIXED_CONTROL_POINTS_PER_SIDE] = base_control_points[:FIXED_CONTROL_POINTS_PER_SIDE]
        control_new[-FIXED_CONTROL_POINTS_PER_SIDE:] = base_control_points[-FIXED_CONTROL_POINTS_PER_SIDE:]
        return {
            "success": bool(result.success),
            "control_points": control_new,
            "slack": np.asarray(slack, dtype=np.float32),
            "message": str(result.message),
            "solve_time": time.perf_counter() - solve_start,
        }


def build_second_difference_matrix(num_control_points: int) -> np.ndarray:
    if num_control_points < 3:
        return np.zeros((0, num_control_points), dtype=np.float32)
    rows = []
    for index in range(num_control_points - 2):
        row = np.zeros((num_control_points,), dtype=np.float32)
        row[index] = 1.0
        row[index + 1] = -2.0
        row[index + 2] = 1.0
        rows.append(row)
    return np.stack(rows, axis=0).astype(np.float32)


def classify_candidate_repair(
    *,
    min_clearance: float,
    d_trigger: float,
    eps_deep: float,
) -> str:
    if np.isfinite(min_clearance) and min_clearance < -float(eps_deep):
        return "deep"
    if np.isfinite(min_clearance) and min_clearance >= float(d_trigger):
        return "safe"
    return "repair"


def compute_scp_pass_trigger(*, d_trigger: float, pass_index: int, pass2_offset: float) -> float:
    if int(pass_index) <= 0:
        return float(d_trigger)
    return float(d_trigger) + float(pass2_offset)


def compute_delta_box_bounds(
    *,
    base_delta_total: np.ndarray,
    delta_max_local: float,
    delta_max_total: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_delta_total = np.asarray(base_delta_total, dtype=np.float64)
    local_lower = np.maximum(-float(delta_max_local), -float(delta_max_total) - base_delta_total)
    local_upper = np.minimum(float(delta_max_local), float(delta_max_total) - base_delta_total)
    return local_lower.astype(np.float64), local_upper.astype(np.float64)


def reconstruct_control_points_from_free_residual(
    *,
    normalized_free_residual: np.ndarray,
    q_start_normalized: np.ndarray,
    q_goal_normalized: np.ndarray,
    delta_w_mean: np.ndarray,
    delta_w_std: np.ndarray,
    num_control_points: int,
) -> np.ndarray:
    normalized_free_residual = np.asarray(normalized_free_residual, dtype=np.float32)
    q_start_normalized = np.asarray(q_start_normalized, dtype=np.float32).reshape(-1)
    q_goal_normalized = np.asarray(q_goal_normalized, dtype=np.float32).reshape(-1)
    base_control_points = build_linear_control_points(
        start_state=q_start_normalized,
        end_state=q_goal_normalized,
        num_control_points=num_control_points,
    ).astype(np.float32)
    control_points = base_control_points.copy()
    free_slice = slice(FIXED_CONTROL_POINTS_PER_SIDE, num_control_points - FIXED_CONTROL_POINTS_PER_SIDE)
    delta_free = normalized_free_residual.astype(np.float32) * delta_w_std.astype(np.float32) + delta_w_mean.astype(np.float32)
    control_points[free_slice] = base_control_points[free_slice] + delta_free.astype(np.float32)
    control_points[:FIXED_CONTROL_POINTS_PER_SIDE] = q_start_normalized.reshape(1, -1)
    control_points[-FIXED_CONTROL_POINTS_PER_SIDE:] = q_goal_normalized.reshape(1, -1)
    return control_points.astype(np.float32)


def control_points_to_normalized_free_residual(
    control_points_normalized: np.ndarray,
    *,
    q_start_normalized: np.ndarray,
    q_goal_normalized: np.ndarray,
    delta_w_mean: np.ndarray,
    delta_w_std: np.ndarray,
) -> np.ndarray:
    control_points_normalized = np.asarray(control_points_normalized, dtype=np.float32)
    base_control_points = build_linear_control_points(
        start_state=np.asarray(q_start_normalized, dtype=np.float32).reshape(-1),
        end_state=np.asarray(q_goal_normalized, dtype=np.float32).reshape(-1),
        num_control_points=control_points_normalized.shape[0],
    ).astype(np.float32)
    free_slice = slice(FIXED_CONTROL_POINTS_PER_SIDE, control_points_normalized.shape[0] - FIXED_CONTROL_POINTS_PER_SIDE)
    delta_free = control_points_normalized[free_slice] - base_control_points[free_slice]
    return ((delta_free - delta_w_mean.astype(np.float32)) / delta_w_std.astype(np.float32)).astype(np.float32)


def flatten_margin_values(sdf_result: dict[str, Any], d_safe: float) -> np.ndarray:
    all_sdf = np.asarray(sdf_result["all_sdf_values"], dtype=np.float32)
    if all_sdf.size == 0:
        return np.empty((0,), dtype=np.float32)
    # Out-of-grid queries are represented by NaN. They are unknown samples, not
    # evidence that every other surface sample at the timestep is invalid.
    finite_sdf = all_sdf.reshape(-1)[np.isfinite(all_sdf.reshape(-1))]
    return (finite_sdf - float(d_safe)).astype(np.float32)


def collect_active_constraints(
    *,
    sdf_result: dict[str, Any],
    d_safe: float,
    d_trigger: float,
    max_active: int,
    environment: PyBulletSurfaceEnvironmentAdapter,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    per_link = {
        str(link_name): np.asarray(link_values, dtype=np.float32)
        for link_name, link_values in dict(sdf_result.get("sdf_values_by_link", {})).items()
    }
    all_h = []
    active = []
    for sample in environment.surface_samples:
        link_name = str(sample["link_name"])
        link_values = per_link.get(link_name)
        if link_values is None or link_values.size == 0:
            continue
        point_values = link_values[:, int(sample["point_index"])] - float(d_safe)
        for t_index, h_value in enumerate(np.asarray(point_values, dtype=np.float32).reshape(-1)):
            all_h.append(float(h_value))
            if float(h_value) < float(d_trigger):
                active.append(
                    {
                        "t_index": int(t_index),
                        "h_value": float(h_value),
                        "link_index": int(sample["link_index"]),
                        "link_name": link_name,
                        "local_point": np.asarray(sample["local_point"], dtype=np.float32),
                        "point_index": int(sample["point_index"]),
                    }
                )
    active.sort(key=lambda item: float(item["h_value"]))
    return np.asarray(all_h, dtype=np.float32), active[: int(max_active)]


def find_worst_trajectory_timesteps(
    *,
    sdf_result: dict[str, Any],
    d_safe: float,
    topk: int = 3,
    d_trigger: float | None = None,
    eps_deep: float | None = None,
) -> list[int]:
    """Find the `topk` timesteps with smallest SDF along a trajectory."""
    all_sdf = np.asarray(
        sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)),
        dtype=np.float32,
    )
    if all_sdf.size == 0:
        return []
    if all_sdf.ndim != 2:
        return []
    finite_mask = np.isfinite(all_sdf)
    has_finite_value = np.any(finite_mask, axis=1)
    min_per_step = np.full(all_sdf.shape[0], np.nan, dtype=np.float32)
    if np.any(has_finite_value):
        min_per_step[has_finite_value] = np.min(
            np.where(finite_mask[has_finite_value], all_sdf[has_finite_value], np.inf),
            axis=1,
        )
    h_per_step = min_per_step - float(d_safe)
    valid_mask = np.isfinite(h_per_step)
    if not np.any(valid_mask):
        return []
    valid_indices = np.flatnonzero(valid_mask)
    valid_h = h_per_step[valid_indices]
    if d_trigger is not None:
        trigger_mask = valid_h < float(d_trigger)
        if eps_deep is not None:
            trigger_mask &= valid_h >= -float(eps_deep)
        valid_indices = valid_indices[trigger_mask]
        valid_h = valid_h[trigger_mask]
    if valid_indices.size == 0:
        return []
    order = np.argsort(valid_h)
    topk = min(int(topk), valid_indices.size)
    return valid_indices[order[:topk]].tolist()


def _flat_index_to_surface_sample(
    surface_samples: list[dict[str, Any]],
    flat_index: int,
) -> dict[str, Any] | None:
    for sample in surface_samples:
        if int(sample.get("flat_index", -1)) == int(flat_index):
            return sample
    return None


def build_topk_cbf_constraints(
    *,
    worst_timesteps: list[int],
    sdf_result: dict[str, Any],
    check_basis: np.ndarray,
    q_check_norm: np.ndarray,
    environment: PyBulletSurfaceEnvironmentAdapter,
    max_active: int | None = None,
) -> list[dict[str, Any]]:
    """Build CBF constraints for the top-k worst trajectory timesteps."""
    all_sdf = np.asarray(sdf_result["all_sdf_values"], dtype=np.float32)
    constraints: list[dict[str, Any]] = []
    for t_idx in worst_timesteps:
        if t_idx < 0 or t_idx >= all_sdf.shape[0]:
            continue
        step_sdf = np.asarray(all_sdf[t_idx], dtype=np.float32).reshape(-1)
        finite_indices = np.flatnonzero(np.isfinite(step_sdf))
        if finite_indices.size == 0:
            continue
        worst_flat_idx = int(
            finite_indices[np.argmin(step_sdf[finite_indices])]
        )
        sample_info = _flat_index_to_surface_sample(
            environment.surface_samples, worst_flat_idx
        )
        if sample_info is None:
            continue
        q_norm = np.asarray(q_check_norm[t_idx], dtype=np.float32).reshape(-1)
        q_actual = environment.normalized_to_actual(q_norm)
        point_world = environment.surface_point_world(
            link_index=int(sample_info["link_index"]),
            local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
        )
        grad_world = approximate_sdf_gradient(environment.load_sdf_grid(), point_world)
        jacobian = environment.surface_point_jacobian(
            link_index=int(sample_info["link_index"]),
            local_point=np.asarray(sample_info["local_point"], dtype=np.float32),
            q_actual=q_actual,
        )
        j_pos = np.asarray(jacobian[:3, :], dtype=np.float32)
        g_actual = (j_pos.T @ grad_world.reshape(3)).astype(np.float32)
        constraints.append({
            "t_index": int(t_idx),
            "h_value": float(step_sdf[worst_flat_idx]),
            "basis_row": np.asarray(check_basis[t_idx], dtype=np.float32),
            "g_actual": g_actual,
            "link_index": int(sample_info["link_index"]),
            "point_index": int(sample_info["point_index"]),
        })
    constraints.sort(key=lambda item: float(item["h_value"]))
    if max_active is not None:
        return constraints[: int(max_active)]
    return constraints


def approximate_sdf_gradient(sdf_grid, point_world: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    point_world = np.asarray(point_world, dtype=np.float32).reshape(3)
    basis = np.eye(3, dtype=np.float32) * float(eps)
    grads = []
    for axis in range(3):
        p_plus = point_world + basis[axis]
        p_minus = point_world - basis[axis]
        sdf_plus = float(sdf_grid.query(p_plus.reshape(1, 3))[0])
        sdf_minus = float(sdf_grid.query(p_minus.reshape(1, 3))[0])
        grads.append((sdf_plus - sdf_minus) / (2.0 * float(eps)))
    grad = np.asarray(grads, dtype=np.float32)
    norm = float(np.linalg.norm(grad))
    if norm <= 1e-12 or (not np.all(np.isfinite(grad))):
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return (grad / norm).astype(np.float32)


def _build_basis(*, num_control_points: int, num_steps: int, degree: int) -> np.ndarray:
    sample_parameters = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float64)
    return build_bspline_basis_matrix(
        sample_parameters=sample_parameters,
        num_control_points=int(num_control_points),
        degree=int(degree),
    ).astype(np.float32)


def guided_policy_sample(
    *,
    policy,
    context: SamplingContext,
    q_start_normalized: np.ndarray,
    q_goal_normalized: np.ndarray,
    delta_w_mean: np.ndarray,
    delta_w_std: np.ndarray,
    num_control_points: int,
    spline_degree: int,
    guidance_runner: SurfaceCBFQPGuidanceRunner,
    generator=None,
    num_inference_steps: int | None = None,
    scheduler_step_kwargs: dict[str, Any] | None = None,
) -> GuidanceResult:
    # Independent path: keep original predict_action untouched.
    if num_inference_steps is None:
        num_inference_steps = int(policy.num_inference_steps)
    scheduler = policy.noise_scheduler
    step_kwargs = dict(scheduler_step_kwargs or {})
    if generator is not None:
        step_kwargs.setdefault("generator", generator)
    step_kwargs = _filter_scheduler_step_kwargs(scheduler, step_kwargs)
    scheduler.set_timesteps(int(num_inference_steps))
    timesteps = list(scheduler.timesteps)
    batch_size = int(guidance_runner.config.num_candidates)
    sample_shape = (batch_size, int(policy.horizon), int(policy.action_dim))
    trajectory = np.asarray(
        policy.device.new_zeros(1).detach().cpu().numpy() if False else None
    )
    import torch

    noisy = torch.randn(
        size=sample_shape,
        dtype=policy.dtype,
        device=policy.device,
        generator=generator,
    )
    cond_data = context.condition_data.expand(batch_size, -1, -1).clone()
    cond_mask = context.condition_mask.expand(batch_size, -1, -1).clone()
    local_cond = None if context.local_cond is None else context.local_cond.expand(batch_size, *context.local_cond.shape[1:])
    global_cond = None if context.global_cond is None else context.global_cond.expand(batch_size, *context.global_cond.shape[1:])
    pred_type = str(policy.noise_scheduler.config.prediction_type)
    if pred_type not in {"epsilon", "sample"}:
        raise ValueError(
            "Surface CBF-QP guidance currently supports prediction_type 'epsilon' or 'sample', "
            f"got {pred_type!r}."
        )
    for step_index, timestep in enumerate(timesteps):
        noisy[cond_mask] = cond_data[cond_mask]
        model_output = policy.model(
            sample=noisy,
            timestep=timestep,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        if (len(timesteps) - step_index) <= int(guidance_runner.config.guidance_steps):
            alpha_bar_t = _alpha_bar_for_timestep(scheduler, timestep, device=policy.device, dtype=policy.dtype)
            alpha_bar_prev = _alpha_bar_for_previous_step(
                scheduler=scheduler,
                timesteps=timesteps,
                index=step_index,
                device=policy.device,
                dtype=policy.dtype,
            )
            if pred_type == "epsilon":
                x0_hat = (noisy - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
            else:
                x0_hat = model_output
            guidance_result = guidance_runner.run(
                candidate_residuals=x0_hat.detach().cpu().numpy(),
                q_start_normalized=np.asarray(q_start_normalized, dtype=np.float32),
                q_goal_normalized=np.asarray(q_goal_normalized, dtype=np.float32),
                delta_w_mean=np.asarray(delta_w_mean, dtype=np.float32),
                delta_w_std=np.asarray(delta_w_std, dtype=np.float32),
                num_control_points=int(num_control_points),
                spline_degree=int(spline_degree),
            )
            x0_proj = torch.from_numpy(
                np.asarray(guidance_result.best_normalized_free_residual, dtype=np.float32)
            ).to(device=policy.device, dtype=policy.dtype)
            x0_proj = x0_proj.unsqueeze(0).expand(batch_size, -1, -1).clone()
            eps_proj = (noisy - torch.sqrt(alpha_bar_t) * x0_proj) / torch.sqrt(1.0 - alpha_bar_t)
            noisy = torch.sqrt(alpha_bar_prev) * x0_proj + torch.sqrt(1.0 - alpha_bar_prev) * eps_proj
            noisy[cond_mask] = cond_data[cond_mask]
            return guidance_result
        noisy = scheduler.step(model_output, timestep, noisy, **step_kwargs).prev_sample
    raise RuntimeError("Guided sampling exited without entering the configured guidance window.")


def _alpha_bar_for_timestep(scheduler, timestep, *, device, dtype):
    alpha_bar = scheduler.alphas_cumprod[int(timestep)]
    if not hasattr(alpha_bar, "to"):
        import torch

        alpha_bar = torch.tensor(float(alpha_bar), device=device, dtype=dtype)
    else:
        alpha_bar = alpha_bar.to(device=device, dtype=dtype)
    return alpha_bar.reshape(1, 1, 1)


def _alpha_bar_for_previous_step(*, scheduler, timesteps: list[Any], index: int, device, dtype):
    if index >= len(timesteps) - 1:
        prev = 1.0
    else:
        prev_timestep = timesteps[index + 1]
        prev = scheduler.alphas_cumprod[int(prev_timestep)]
    if not hasattr(prev, "to"):
        import torch

        prev = torch.tensor(float(prev), device=device, dtype=dtype)
    else:
        prev = prev.to(device=device, dtype=dtype)
    return prev.reshape(1, 1, 1)
