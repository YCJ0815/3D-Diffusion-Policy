from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence
import inspect
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from diffusion_policy_3d.common.bspline import (
    FIXED_CONTROL_POINTS_PER_SIDE,
    build_bspline_basis_matrix,
    build_linear_control_points,
    evaluate_quintic_bspline,
)


DEFAULT_GUIDANCE_TARGETS = (-0.02, 0.0)


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
    qp_candidates: int = 4
    active_constraints: int = 16
    check_steps: int = 64
    cert_steps: int = 256
    cert_swept_intermediate: int = 3
    d_safe: float = 0.03
    d_trigger: float = 0.06
    d_cert: float = 0.01
    eps_deep: float = 0.03
    delta_max: float = 0.05
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
    used_existing_terminal_cbf: bool = False
    selected_by_certificate: bool = False

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
        self.robot_surface_points_by_link = {
            int(link_index): np.asarray(points, dtype=np.float32)
            for link_index, points in validator.robot_surface_points_by_link.items()
        }
        self._surface_samples = self._build_surface_sample_index()

    @property
    def surface_samples(self) -> list[dict[str, Any]]:
        return self._surface_samples

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
        jacobian_t, _, _ = self.pb.calculateJacobian(
            self.robot_id,
            int(link_index),
            local_position,
            [float(v) for v in np.asarray(q_actual, dtype=np.float32).reshape(-1)],
            zero_vec,
            zero_vec,
            physicsClientId=self.client_id,
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
        log = GuidanceLog()
        guidance_targets = build_guidance_target_schedule(
            int(self.config.guidance_steps),
            self.config.guidance_targets,
        )
        check_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.config.check_steps), degree=int(spline_degree))
        cert_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.config.cert_steps), degree=int(spline_degree))
        limit_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.config.joint_limit_steps), degree=int(spline_degree))
        free_slice = slice(FIXED_CONTROL_POINTS_PER_SIDE, num_control_points - FIXED_CONTROL_POINTS_PER_SIDE)

        candidate_infos: list[dict[str, Any]] = []
        before_h_values = []
        after_h_values = []
        final_h_values = []
        selected_index = 0

        for candidate_index, free_residual in enumerate(candidate_residuals):
            candidate_start_time = time.perf_counter()
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
            sdf_result = self.environment.collect_joint_trajectory_sdf_with_link_details(q_check_actual)
            h_before = flatten_margin_values(sdf_result, d_safe=float(self.config.d_safe))
            h_min_before = float(np.min(h_before)) if h_before.size > 0 else math.nan
            before_h_values.append(h_min_before)
            guided_control_points = control_points.copy()
            qp_success = False
            qp_result: dict[str, Any] | None = None

            worst_timesteps: list[int] = find_worst_trajectory_timesteps(
                sdf_result=sdf_result,
                d_safe=float(self.config.d_safe),
                topk=3,
            )
            topk_constraints = build_topk_cbf_constraints(
                worst_timesteps=worst_timesteps,
                sdf_result=sdf_result,
                check_basis=check_basis,
                q_check_norm=q_check_norm,
                environment=self.environment,
            ) if worst_timesteps else []

            if topk_constraints and np.isfinite(h_min_before):
                log.num_qp_called += 1
                log.num_active_constraints += len(topk_constraints)
                guidance_start = time.perf_counter()
                qp_result = self._solve_guidance_qp(
                    control_points=control_points,
                    q_check_norm=q_check_norm,
                    q_check_actual=q_check_actual,
                    active_constraints=topk_constraints,
                    target_margin=float(guidance_targets[-1]),
                    limit_basis=limit_basis,
                    free_slice=free_slice,
                )
                log.guidance_time += time.perf_counter() - guidance_start
                log.qp_time += float(qp_result.get("solve_time", 0.0)) if qp_result is not None else 0.0
                if qp_result is not None and bool(qp_result.get("success", False)):
                    guided_control_points = np.asarray(qp_result["control_points"], dtype=np.float32)
                    qp_success = True
                    log.num_qp_success += 1
            q_after_norm = check_basis @ guided_control_points
            q_after_actual = self.environment.normalized_to_actual(q_after_norm)
            sdf_after = self.environment.collect_joint_trajectory_sdf_with_link_details(q_after_actual)
            h_after = flatten_margin_values(sdf_after, d_safe=float(self.config.d_safe))
            h_min_after = float(np.min(h_after)) if h_after.size > 0 else math.nan
            after_h_values.append(h_min_after)

            certificate_start = time.perf_counter()
            cert_result = self._certificate_check(
                control_points=guided_control_points,
                cert_basis=cert_basis,
            )
            log.certificate_time += time.perf_counter() - certificate_start
            final_h_values.append(float(cert_result["h_min_final"]))
            candidate_info = {
                "candidate_index": int(candidate_index),
                "h_min_before_guidance": h_min_before,
                "h_min_after_guidance": h_min_after,
                "h_min_final": float(cert_result["h_min_final"]),
                "certificate_success": bool(cert_result["success"]),
                "qp_success": bool(qp_success),
                "control_points_normalized": guided_control_points.astype(np.float32),
                "joint_trajectory": np.asarray(cert_result["joint_trajectory"], dtype=np.float32),
                "normalized_free_residual": control_points_to_normalized_free_residual(
                    guided_control_points,
                    q_start_normalized=q_start_normalized,
                    q_goal_normalized=q_goal_normalized,
                    delta_w_mean=delta_w_mean,
                    delta_w_std=delta_w_std,
                ),
                "path_length": compute_path_length(cert_result["joint_trajectory"]),
                "smoothness": compute_smoothness(cert_result["joint_trajectory"]),
                "goal_error": float(np.linalg.norm(cert_result["joint_trajectory"][-1] - cert_result["joint_trajectory"][-1])),
                "candidate_time": time.perf_counter() - candidate_start_time,
            }
            candidate_infos.append(candidate_info)

        log.dp_time = sum(float(info["candidate_time"]) for info in candidate_infos)
        log.h_min_before_guidance = float(np.nanmin(np.asarray(before_h_values, dtype=np.float32))) if before_h_values else math.nan
        log.h_min_after_guidance = float(np.nanmin(np.asarray(after_h_values, dtype=np.float32))) if after_h_values else math.nan
        log.h_min_final = float(np.nanmin(np.asarray(final_h_values, dtype=np.float32))) if final_h_values else math.nan

        successful = [info for info in candidate_infos if bool(info["certificate_success"])]
        if successful:
            successful.sort(key=lambda info: (-float(info["h_min_final"]), float(info["path_length"])))
            best = successful[0]
            log.certificate_success = True
            log.selected_by_certificate = True
        else:
            candidate_infos.sort(key=lambda info: float(info["h_min_final"]))
            best = candidate_infos[-1]
            fallback = None
            if self.config.fallback_to_terminal_cbf:
                fallback = self.environment.try_existing_terminal_cbf(best["joint_trajectory"])
            log.fallback_used = fallback is not None
            log.used_existing_terminal_cbf = fallback is not None
        selected_index = int(best["candidate_index"])
        log.best_candidate_index = selected_index
        log.goal_error = float(best["goal_error"])
        log.smoothness = float(best["smoothness"])
        log.path_length = float(best["path_length"])
        log.total_time = time.perf_counter() - start_time
        return GuidanceResult(
            best_index=selected_index,
            best_normalized_free_residual=np.asarray(best["normalized_free_residual"], dtype=np.float32),
            best_control_points_normalized=np.asarray(best["control_points_normalized"], dtype=np.float32),
            best_joint_trajectory=np.asarray(best["joint_trajectory"], dtype=np.float32),
            candidate_infos=candidate_infos,
            log=log,
        )

    def _certificate_check(self, *, control_points: np.ndarray, cert_basis: np.ndarray) -> dict[str, Any]:
        q_cert_norm = cert_basis @ control_points
        q_cert_actual = self.environment.normalized_to_actual(q_cert_norm)
        q_cert_swept = interpolate_swept_segments(q_cert_actual, int(self.config.cert_swept_intermediate))
        sdf_cert = self.environment.collect_joint_trajectory_sdf_with_link_details_any_length(q_cert_swept)
        h_cert = flatten_margin_values(sdf_cert, d_safe=float(self.config.d_safe))
        h_min_final = float(np.min(h_cert)) if h_cert.size > 0 else math.nan
        return {
            "success": bool(np.isfinite(h_min_final) and h_min_final >= float(self.config.d_cert)),
            "h_min_final": h_min_final,
            "joint_trajectory": q_cert_swept.astype(np.float32),
        }

    def _solve_guidance_qp(
        self,
        *,
        control_points: np.ndarray,
        q_check_norm: np.ndarray,
        q_check_actual: np.ndarray,
        active_constraints: list[dict[str, Any]],
        target_margin: float,
        limit_basis: np.ndarray,
        free_slice: slice,
    ) -> dict[str, Any] | None:
        solve_start = time.perf_counter()
        dof = control_points.shape[1]
        num_slack = len(active_constraints)
        if num_slack <= 0:
            return None
        base_control_points = np.asarray(control_points, dtype=np.float32)

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
        lower_bounds[: num_selected * dof] = -float(self.config.delta_max)
        upper_bounds[: num_selected * dof] = float(self.config.delta_max)
        lower_bounds[num_selected * dof :] = 0.0
        joint_lower_norm = np.full(dof, -1.0, dtype=np.float32)
        joint_upper_norm = np.full(dof, 1.0, dtype=np.float32)
        joint_scale = self.environment.joint_scale().reshape(-1)

        linear_rows = []
        linear_lb = []
        linear_ub = []

        # CBF constraints (only for the selected worst timesteps)
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
    return (all_sdf.reshape(-1) - float(d_safe)).astype(np.float32)


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
) -> list[int]:
    """Find the `topk` timesteps with smallest SDF along a trajectory."""
    all_sdf = np.asarray(
        sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)),
        dtype=np.float32,
    )
    if all_sdf.size == 0:
        return []
    min_per_step = np.min(all_sdf, axis=1)
    h_per_step = min_per_step - float(d_safe)
    valid_mask = np.isfinite(h_per_step)
    if not np.any(valid_mask):
        return []
    valid_indices = np.flatnonzero(valid_mask)
    valid_h = h_per_step[valid_indices]
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
) -> list[dict[str, Any]]:
    """Build CBF constraints for the top-k worst trajectory timesteps."""
    all_sdf = np.asarray(sdf_result["all_sdf_values"], dtype=np.float32)
    constraints: list[dict[str, Any]] = []
    for t_idx in worst_timesteps:
        if t_idx < 0 or t_idx >= all_sdf.shape[0]:
            continue
        step_sdf = np.asarray(all_sdf[t_idx], dtype=np.float32).reshape(-1)
        worst_flat_idx = int(np.argmin(step_sdf))
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
