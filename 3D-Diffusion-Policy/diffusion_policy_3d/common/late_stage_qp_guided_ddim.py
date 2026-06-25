from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math
import time

import numpy as np
import torch

from diffusion_policy_3d.common.bspline import (
    FIXED_CONTROL_POINTS_PER_SIDE,
    build_bspline_basis_matrix,
)
from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
    GuidanceLog,
    PyBulletSurfaceEnvironmentAdapter,
    SurfaceCBFQPGuidanceConfig,
    SurfaceCBFQPGuidanceRunner,
    _build_basis,
    _filter_scheduler_step_kwargs,
    build_risk_segments,
    compute_path_length,
    compute_scp_pass_trigger,
    compute_smoothness,
    control_points_to_normalized_free_residual,
    reconstruct_control_points_from_free_residual,
    summarize_sdf_risk,
)


@dataclass
class LateStageQPGuidedDDIMConfig:
    enabled: bool = True
    num_candidates: int = 32
    guidance_steps: int = 3
    guidance_timesteps: tuple[int, ...] = (10, 5, 1)
    qp_candidates: int = 2
    qp_inner_scp_rounds: int = 1
    coarse_check_steps: int = 16
    guidance_trigger_distance: float = 0.06
    guidance_safe_distance: float = 0.05
    trust_region_start: float = 0.015
    trust_region_end: float = 0.05
    blend_weights: tuple[float, ...] = (0.25, 0.5, 0.75)
    repair_score_weights: tuple[float, float, float] = (1.0, 10.0, 1.0)
    ddim_eta: float = 0.0
    scp_config: SurfaceCBFQPGuidanceConfig = field(default_factory=SurfaceCBFQPGuidanceConfig)


@dataclass
class LateStageQPGuidedDDIMResult:
    best_index: int
    best_normalized_free_residual: np.ndarray
    best_control_points_normalized: np.ndarray
    best_joint_trajectory: np.ndarray
    planning_success: bool
    candidate_infos: list[dict[str, Any]]
    log: dict[str, Any]


class DDIMX0OverrideStepper:
    """Local DDIM step helper that supports a guided x0 without patching diffusers."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def _previous_timestep(self, timestep: int) -> int:
        scheduler = self.scheduler
        step = int(getattr(scheduler, "config").num_train_timesteps // scheduler.num_inference_steps)
        return int(timestep) - step

    def step_with_x0_override(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        *,
        pred_original_sample_override: torch.Tensor | None = None,
        eta: float = 0.0,
        generator=None,
    ) -> torch.Tensor:
        scheduler = self.scheduler
        t_value = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
        prev_timestep = self._previous_timestep(t_value)
        alpha_prod_t = scheduler.alphas_cumprod[t_value].to(device=sample.device, dtype=sample.dtype)
        if prev_timestep >= 0:
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep].to(device=sample.device, dtype=sample.dtype)
        else:
            alpha_prod_t_prev = scheduler.final_alpha_cumprod.to(device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1.0 - alpha_prod_t
        prediction_type = str(scheduler.config.prediction_type)
        if prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif prediction_type == "sample":
            pred_original_sample = model_output
        elif prediction_type == "v_prediction":
            pred_original_sample = torch.sqrt(alpha_prod_t) * sample - torch.sqrt(beta_prod_t) * model_output
        else:
            raise ValueError(f"Unsupported DDIM prediction_type: {prediction_type!r}")

        if pred_original_sample_override is not None:
            pred_original_sample = pred_original_sample_override.to(device=sample.device, dtype=sample.dtype)
        if bool(getattr(scheduler.config, "clip_sample", False)):
            clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
            pred_original_sample = torch.clamp(pred_original_sample, -clip_range, clip_range)

        pred_epsilon = (sample - torch.sqrt(alpha_prod_t) * pred_original_sample) / torch.sqrt(beta_prod_t)
        variance = scheduler._get_variance(t_value, prev_timestep)
        variance = variance.to(device=sample.device, dtype=sample.dtype) if hasattr(variance, "to") else torch.as_tensor(variance, device=sample.device, dtype=sample.dtype)
        std_dev_t = float(eta) * torch.sqrt(variance)
        direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_prod_t_prev - std_dev_t ** 2, min=0.0))
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + direction_scale * pred_epsilon
        if float(eta) > 0.0:
            noise = torch.randn(model_output.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + std_dev_t * noise
        return prev_sample


def _alpha_bar_for_timestep(scheduler, timestep, *, device, dtype):
    return scheduler.alphas_cumprod[int(timestep.item()) if hasattr(timestep, "item") else int(timestep)].to(device=device, dtype=dtype)


def predict_x0_from_model_output(scheduler, sample: torch.Tensor, model_output: torch.Tensor, timestep) -> torch.Tensor:
    alpha_bar_t = _alpha_bar_for_timestep(scheduler, timestep, device=sample.device, dtype=sample.dtype)
    pred_type = str(scheduler.config.prediction_type)
    if pred_type == "epsilon":
        return (sample - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
    if pred_type == "sample":
        return model_output
    if pred_type == "v_prediction":
        return torch.sqrt(alpha_bar_t) * sample - torch.sqrt(1.0 - alpha_bar_t) * model_output
    raise ValueError(f"Unsupported prediction_type: {pred_type!r}")


def _metric_from_sdf(sdf_result: dict[str, Any], *, d_safe: float, d_trigger: float) -> dict[str, float]:
    all_sdf = np.asarray(sdf_result.get("all_sdf_values", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    finite = all_sdf[np.isfinite(all_sdf)]
    if finite.size == 0:
        return {
            "min_sdf": math.nan,
            "num_penetration": 0,
            "max_penetration_depth": math.nan,
            "collision_risk": math.inf,
        }
    min_sdf = float(np.min(finite))
    penetration = finite[finite < 0.0]
    max_penetration_depth = float(np.max(-penetration)) if penetration.size else 0.0
    collision_risk = float(np.sum(np.maximum(float(d_trigger) - finite, 0.0)))
    return {
        "min_sdf": min_sdf,
        "num_penetration": int(penetration.size),
        "max_penetration_depth": max_penetration_depth,
        "collision_risk": collision_risk,
    }


class LateStageQPGuidedDDIMRunner:
    def __init__(self, *, config: LateStageQPGuidedDDIMConfig, environment: PyBulletSurfaceEnvironmentAdapter):
        self.config = config
        self.environment = environment
        self.scp_config = config.scp_config
        self.scp_config.enabled = True
        self.scp_config.num_candidates = int(config.num_candidates)
        self.scp_config.guidance_steps = int(config.guidance_steps)
        self.scp_config.scp_iterations = int(config.qp_inner_scp_rounds)
        self.scp_config.check_steps = int(config.coarse_check_steps)
        self.scp_config.d_trigger = float(config.guidance_trigger_distance)
        self.scp_config.d_safe = float(config.guidance_safe_distance)
        self.scp_config.enable_local_waypoint_qp_after_certificate = False
        self.runner = SurfaceCBFQPGuidanceRunner(config=self.scp_config, environment=environment)

    @staticmethod
    def _passes_positive_clearance_certificate(cert_result: dict[str, Any]) -> bool:
        min_clearance = float(cert_result.get("min_clearance", math.nan))
        return bool(np.isfinite(min_clearance) and min_clearance > 0.0)

    def _evaluate_residual(
        self,
        *,
        residual: np.ndarray,
        q_start_normalized: np.ndarray,
        q_goal_normalized: np.ndarray,
        delta_w_mean: np.ndarray,
        delta_w_std: np.ndarray,
        num_control_points: int,
        check_basis: np.ndarray,
    ) -> dict[str, Any]:
        control_points = reconstruct_control_points_from_free_residual(
            normalized_free_residual=residual,
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
            d_safe=float(self.scp_config.d_safe),
            d_trigger=float(self.scp_config.d_trigger),
        )
        metrics = _metric_from_sdf(
            sdf_result,
            d_safe=float(self.scp_config.d_safe),
            d_trigger=float(self.scp_config.d_trigger),
        )
        return {
            "control_points": control_points.astype(np.float32),
            "joint_trajectory": q_check_actual.astype(np.float32),
            "sdf_result": sdf_result,
            "risk_summary": risk_summary,
            **metrics,
        }

    def _probe_candidate(
        self,
        *,
        control_points: np.ndarray,
        check_basis: np.ndarray,
        limit_basis: np.ndarray,
        free_slice: slice,
        trust_region: float,
    ) -> dict[str, Any]:
        state = self.runner._evaluate_candidate_state(
            control_points=control_points,
            check_basis=check_basis,
            d_trigger=float(self.scp_config.d_trigger),
        )
        min_clearance = float(state["risk_summary"]["min_clearance"])
        if np.isfinite(min_clearance) and min_clearance >= float(self.config.guidance_trigger_distance):
            return {"status": "safe_noop", "repair_cost": 0.0, "slack_sum": 0.0, "delta_norm": 0.0}
        if np.isfinite(min_clearance) and min_clearance < -float(self.scp_config.eps_deep):
            return {"status": "deep_unrepairable", "repair_cost": math.inf, "slack_sum": math.inf, "delta_norm": 0.0}
        pass_detail = self.runner._build_scp_pass_detail(
            pass_index=0,
            state=state,
            check_basis=check_basis,
            d_trigger=float(self.scp_config.d_trigger),
        )
        if not pass_detail["active_constraints"]:
            return {"status": "no_constraints", "repair_cost": math.inf, "slack_sum": math.inf, "delta_norm": 0.0}
        qp_result = self.runner._solve_guidance_qp(
            base_control_points=control_points,
            reference_control_points=control_points,
            active_constraints=pass_detail["active_constraints"],
            target_margin=float(self.scp_config.d_cert) + float(self.scp_config.margin_buffer),
            limit_basis=limit_basis,
            free_slice=free_slice,
            delta_max_local=float(trust_region),
            delta_max_total=float(trust_region),
        )
        if qp_result is None or not bool(qp_result.get("success", False)):
            return {"status": "probe_solver_failure", "repair_cost": math.inf, "slack_sum": math.inf, "delta_norm": 0.0}
        delta = np.asarray(qp_result["control_points"], dtype=np.float32) - np.asarray(control_points, dtype=np.float32)
        delta_norm = float(np.linalg.norm(delta))
        slack_sum = float(np.sum(np.asarray(qp_result.get("slack", []), dtype=np.float32)))
        return {
            "status": "probe_success",
            "repair_cost": delta_norm,
            "slack_sum": slack_sum,
            "delta_norm": delta_norm,
        }

    def _repair_control_points(
        self,
        *,
        control_points: np.ndarray,
        check_basis: np.ndarray,
        limit_basis: np.ndarray,
        free_slice: slice,
        trust_region: float,
    ) -> dict[str, Any]:
        original = np.asarray(control_points, dtype=np.float32)
        current = original.copy()
        total_slack = 0.0
        status = "safe_noop"
        for pass_index in range(int(self.config.qp_inner_scp_rounds)):
            pass_trigger = compute_scp_pass_trigger(
                d_trigger=float(self.scp_config.d_trigger),
                pass_index=pass_index,
                pass2_offset=float(self.scp_config.d_trigger_pass2_offset),
            )
            state = self.runner._evaluate_candidate_state(
                control_points=current,
                check_basis=check_basis,
                d_trigger=pass_trigger,
            )
            if not np.isfinite(float(state["risk_summary"]["min_margin"])):
                return {"success": False, "status": "non_finite_margin", "control_points": original, "slack_sum": total_slack, "delta_norm": 0.0}
            if not build_risk_segments(sdf_result=state["sdf_result"], d_trigger=pass_trigger):
                return {"success": True, "status": status, "control_points": current, "slack_sum": total_slack, "delta_norm": float(np.linalg.norm(current - original))}
            pass_detail = self.runner._build_scp_pass_detail(
                pass_index=pass_index,
                state=state,
                check_basis=check_basis,
                d_trigger=pass_trigger,
            )
            if not pass_detail["active_constraints"]:
                return {"success": False, "status": "no_constraints", "control_points": original, "slack_sum": total_slack, "delta_norm": 0.0}
            qp_result = self.runner._solve_guidance_qp(
                base_control_points=current,
                reference_control_points=original,
                active_constraints=pass_detail["active_constraints"],
                target_margin=float(self.scp_config.d_cert) + float(self.scp_config.margin_buffer),
                limit_basis=limit_basis,
                free_slice=free_slice,
                delta_max_local=float(trust_region),
                delta_max_total=float(trust_region),
            )
            if qp_result is None or not bool(qp_result.get("success", False)):
                return {"success": False, "status": "solver_failure", "control_points": original, "slack_sum": total_slack, "delta_norm": 0.0}
            current = np.asarray(qp_result["control_points"], dtype=np.float32)
            current[:FIXED_CONTROL_POINTS_PER_SIDE] = original[:FIXED_CONTROL_POINTS_PER_SIDE]
            current[-FIXED_CONTROL_POINTS_PER_SIDE:] = original[-FIXED_CONTROL_POINTS_PER_SIDE:]
            slack = np.asarray(qp_result.get("slack", []), dtype=np.float32)
            total_slack += float(np.sum(slack))
            status = "qp_success"
            if total_slack > 1.0:
                return {"success": False, "status": "slack_too_large", "control_points": original, "slack_sum": total_slack, "delta_norm": 0.0}
            if float(np.max(np.abs(current - original))) > float(trust_region) + 1e-5:
                return {"success": False, "status": "delta_too_large", "control_points": original, "slack_sum": total_slack, "delta_norm": 0.0}
        return {
            "success": True,
            "status": status,
            "control_points": current,
            "slack_sum": total_slack,
            "delta_norm": float(np.linalg.norm(current - original)),
        }

    def guide_x0_candidates(
        self,
        *,
        x0_candidates: np.ndarray,
        guidance_step_index: int,
        q_start_normalized: np.ndarray,
        q_goal_normalized: np.ndarray,
        delta_w_mean: np.ndarray,
        delta_w_std: np.ndarray,
        num_control_points: int,
        spline_degree: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
        start_time = time.perf_counter()
        x0_candidates = np.asarray(x0_candidates, dtype=np.float32)
        check_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.config.coarse_check_steps), degree=int(spline_degree))
        limit_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.scp_config.joint_limit_steps), degree=int(spline_degree))
        free_slice = slice(FIXED_CONTROL_POINTS_PER_SIDE, num_control_points - FIXED_CONTROL_POINTS_PER_SIDE)
        if int(self.config.guidance_steps) <= 1:
            trust_region = float(self.config.trust_region_end)
        else:
            alpha = float(guidance_step_index) / float(max(1, int(self.config.guidance_steps) - 1))
            trust_region = (1.0 - alpha) * float(self.config.trust_region_start) + alpha * float(self.config.trust_region_end)
        weights = tuple(float(v) for v in self.config.repair_score_weights)

        infos: list[dict[str, Any]] = []
        for candidate_index, residual in enumerate(x0_candidates):
            eval_result = self._evaluate_residual(
                residual=residual,
                q_start_normalized=q_start_normalized,
                q_goal_normalized=q_goal_normalized,
                delta_w_mean=delta_w_mean,
                delta_w_std=delta_w_std,
                num_control_points=num_control_points,
                check_basis=check_basis,
            )
            probe = self._probe_candidate(
                control_points=eval_result["control_points"],
                check_basis=check_basis,
                limit_basis=limit_basis,
                free_slice=free_slice,
                trust_region=trust_region,
            )
            collision_risk = float(eval_result["collision_risk"])
            repair_cost = float(probe["repair_cost"])
            slack_sum = float(probe["slack_sum"])
            repairability_score = weights[0] * repair_cost + weights[1] * slack_sum + weights[2] * collision_risk
            infos.append({
                "candidate_index": int(candidate_index),
                "control_points": eval_result["control_points"],
                "normalized_free_residual": np.asarray(residual, dtype=np.float32),
                "probe_status": str(probe["status"]),
                "repair_cost": repair_cost,
                "slack_sum": slack_sum,
                "min_sdf": float(eval_result["min_sdf"]),
                "num_penetration": int(eval_result["num_penetration"]),
                "max_penetration_depth": float(eval_result["max_penetration_depth"]),
                "collision_risk": collision_risk,
                "repairability_score": float(repairability_score),
                "trust_region": float(trust_region),
                "qp_status": str(probe["status"]),
                "qp_delta_norm": float(probe.get("delta_norm", 0.0)),
            })

        order = sorted(
            range(len(infos)),
            key=lambda idx: (
                math.inf if not math.isfinite(float(infos[idx]["repairability_score"])) else float(infos[idx]["repairability_score"]),
                int(infos[idx]["candidate_index"]),
            ),
        )
        selected = order[: max(0, int(self.config.qp_candidates))]
        guided = x0_candidates.copy()
        blend_values = tuple(float(v) for v in self.config.blend_weights)
        blend = blend_values[min(int(guidance_step_index), len(blend_values) - 1)] if blend_values else 1.0
        for candidate_index in selected:
            info = infos[candidate_index]
            if str(info["probe_status"]) in {"safe_noop", "deep_unrepairable"}:
                continue
            repair = self._repair_control_points(
                control_points=np.asarray(info["control_points"], dtype=np.float32),
                check_basis=check_basis,
                limit_basis=limit_basis,
                free_slice=free_slice,
                trust_region=trust_region,
            )
            info["qp_status"] = str(repair["status"])
            info["qp_slack_sum"] = float(repair["slack_sum"])
            info["qp_delta_norm"] = float(repair["delta_norm"])
            if not bool(repair["success"]):
                continue
            repaired_residual = control_points_to_normalized_free_residual(
                np.asarray(repair["control_points"], dtype=np.float32),
                q_start_normalized=q_start_normalized,
                q_goal_normalized=q_goal_normalized,
                delta_w_mean=delta_w_mean,
                delta_w_std=delta_w_std,
            )
            guided[candidate_index] = ((1.0 - blend) * x0_candidates[candidate_index] + blend * repaired_residual).astype(np.float32)
            info["blend_weight"] = float(blend)
        return guided.astype(np.float32), infos, {"guided_qp_time": time.perf_counter() - start_time}

    def finalize_candidates(
        self,
        *,
        x0_candidates: np.ndarray,
        q_start_normalized: np.ndarray,
        q_goal_normalized: np.ndarray,
        delta_w_mean: np.ndarray,
        delta_w_std: np.ndarray,
        num_control_points: int,
        spline_degree: int,
    ) -> LateStageQPGuidedDDIMResult:
        start_time = time.perf_counter()
        cert_basis = _build_basis(num_control_points=num_control_points, num_steps=int(self.scp_config.cert_steps), degree=int(spline_degree))
        candidate_infos: list[dict[str, Any]] = []
        best_info: dict[str, Any] | None = None
        waypoint_time = 0.0
        for candidate_index, residual in enumerate(np.asarray(x0_candidates, dtype=np.float32)):
            control_points = reconstruct_control_points_from_free_residual(
                normalized_free_residual=residual,
                q_start_normalized=q_start_normalized,
                q_goal_normalized=q_goal_normalized,
                delta_w_mean=delta_w_mean,
                delta_w_std=delta_w_std,
                num_control_points=num_control_points,
            )
            cert = self.runner._certificate_check(control_points=control_points, cert_basis=cert_basis)
            recovered = False
            final_cert = cert
            joint_trajectory = np.asarray(cert["joint_trajectory"], dtype=np.float32)
            cert_positive = self._passes_positive_clearance_certificate(cert)
            if not cert_positive:
                waypoint_start = time.perf_counter()
                wp = self.runner._try_local_waypoint_qp(cert_result=cert, log=GuidanceLog())
                waypoint_time += time.perf_counter() - waypoint_start
                if bool(wp.get("success", False)):
                    recovered = True
                    final_cert = wp["recert_result"]
                    joint_trajectory = np.asarray(wp["joint_trajectory"], dtype=np.float32)
            final_positive = self._passes_positive_clearance_certificate(final_cert)
            info = {
                "candidate_index": int(candidate_index),
                "normalized_free_residual": np.asarray(residual, dtype=np.float32),
                "control_points_normalized": control_points.astype(np.float32),
                "joint_trajectory": joint_trajectory.astype(np.float32),
                "certified_before_waypoint_qp": bool(cert_positive),
                "recovered_by_waypoint_qp": bool(recovered),
                "planning_success": bool(final_positive),
                "min_clearance": float(final_cert["min_clearance"]),
                "h_min_final": float(final_cert["h_min_final"]),
                "path_length": compute_path_length(joint_trajectory),
                "smoothness": compute_smoothness(joint_trajectory),
            }
            candidate_infos.append(info)
            if bool(info["planning_success"]) and (
                best_info is None
                or float(info["min_clearance"]) > float(best_info["min_clearance"])
                or (
                    float(info["min_clearance"]) == float(best_info["min_clearance"])
                    and float(info["path_length"]) < float(best_info["path_length"])
                )
            ):
                best_info = info
        certification_time = time.perf_counter() - start_time
        if best_info is None:
            fallback = candidate_infos[0] if candidate_infos else {}
            return LateStageQPGuidedDDIMResult(
                best_index=-1,
                best_normalized_free_residual=np.asarray(fallback.get("normalized_free_residual", np.empty((0, 0))), dtype=np.float32),
                best_control_points_normalized=np.asarray(fallback.get("control_points_normalized", np.empty((0, 0))), dtype=np.float32),
                best_joint_trajectory=np.asarray(fallback.get("joint_trajectory", np.empty((0, 0))), dtype=np.float32),
                planning_success=False,
                candidate_infos=candidate_infos,
                log={
                    "planner_mode": "qp_guided_diffusion",
                    "planning_success": False,
                    "certification_time": float(certification_time - waypoint_time),
                    "waypoint_fallback_time": float(waypoint_time),
                    "selected_candidate_index": -1,
                    "certificate_rule": "min_sdf_gt_0",
                },
            )
        return LateStageQPGuidedDDIMResult(
            best_index=int(best_info["candidate_index"]),
            best_normalized_free_residual=np.asarray(best_info["normalized_free_residual"], dtype=np.float32),
            best_control_points_normalized=np.asarray(best_info["control_points_normalized"], dtype=np.float32),
            best_joint_trajectory=np.asarray(best_info["joint_trajectory"], dtype=np.float32),
            planning_success=True,
            candidate_infos=candidate_infos,
            log={
                "planner_mode": "qp_guided_diffusion",
                "planning_success": True,
                "certification_time": float(certification_time - waypoint_time),
                "waypoint_fallback_time": float(waypoint_time),
                "certified_before_waypoint_qp": bool(best_info["certified_before_waypoint_qp"]),
                "recovered_by_waypoint_qp": bool(best_info["recovered_by_waypoint_qp"]),
                "selected_candidate_index": int(best_info["candidate_index"]),
                "min_clearance": float(best_info["min_clearance"]),
                "certificate_rule": "min_sdf_gt_0",
            },
        )


def sample_late_stage_qp_guided_ddim(
    *,
    policy,
    context,
    q_start_normalized: np.ndarray,
    q_goal_normalized: np.ndarray,
    delta_w_mean: np.ndarray,
    delta_w_std: np.ndarray,
    num_control_points: int,
    spline_degree: int,
    guidance_runner: LateStageQPGuidedDDIMRunner,
    generator=None,
    num_inference_steps: int | None = None,
    scheduler_step_kwargs: dict[str, Any] | None = None,
) -> LateStageQPGuidedDDIMResult:
    total_start = time.perf_counter()
    if num_inference_steps is None:
        num_inference_steps = int(policy.num_inference_steps)
    scheduler = policy.noise_scheduler
    step_kwargs = dict(scheduler_step_kwargs or {})
    eta = float(step_kwargs.pop("eta", guidance_runner.config.ddim_eta))
    if generator is not None:
        step_kwargs.setdefault("generator", generator)
    step_kwargs = _filter_scheduler_step_kwargs(scheduler, step_kwargs)
    scheduler.set_timesteps(int(num_inference_steps))
    timesteps = list(scheduler.timesteps)
    batch_size = int(guidance_runner.config.num_candidates)
    sample_shape = (batch_size, int(policy.horizon), int(policy.action_dim))
    noisy = torch.randn(size=sample_shape, dtype=policy.dtype, device=policy.device, generator=generator)
    cond_data = context.condition_data.expand(batch_size, -1, -1).clone()
    cond_mask = context.condition_mask.expand(batch_size, -1, -1).clone()
    local_cond = None if context.local_cond is None else context.local_cond.expand(batch_size, *context.local_cond.shape[1:])
    global_cond = None if context.global_cond is None else context.global_cond.expand(batch_size, *context.global_cond.shape[1:])
    stepper = DDIMX0OverrideStepper(scheduler)
    guidance_step_infos: list[dict[str, Any]] = []
    diffusion_time = 0.0
    guided_qp_time = 0.0
    configured_guidance_timesteps = tuple(
        int(v)
        for v in getattr(guidance_runner.config, "guidance_timesteps", ())
        if int(v) > 0
    )
    guidance_remaining_steps = tuple(
        sorted(set(configured_guidance_timesteps), reverse=True)
    ) or tuple(range(min(int(guidance_runner.config.guidance_steps), len(timesteps)), 0, -1))
    guidance_step_lookup = {
        int(remaining_step): int(index)
        for index, remaining_step in enumerate(guidance_remaining_steps)
    }
    for step_index, timestep in enumerate(timesteps):
        noisy[cond_mask] = cond_data[cond_mask]
        model_start = time.perf_counter()
        model_output = policy.model(sample=noisy, timestep=timestep, local_cond=local_cond, global_cond=global_cond)
        diffusion_time += time.perf_counter() - model_start
        tail_index = len(timesteps) - step_index
        if int(tail_index) in guidance_step_lookup:
            x0_hat = predict_x0_from_model_output(scheduler, noisy, model_output, timestep)
            guidance_index = guidance_step_lookup[int(tail_index)]
            guided_np, candidate_infos, timing = guidance_runner.guide_x0_candidates(
                x0_candidates=x0_hat.detach().cpu().numpy(),
                guidance_step_index=int(guidance_index),
                q_start_normalized=np.asarray(q_start_normalized, dtype=np.float32),
                q_goal_normalized=np.asarray(q_goal_normalized, dtype=np.float32),
                delta_w_mean=np.asarray(delta_w_mean, dtype=np.float32),
                delta_w_std=np.asarray(delta_w_std, dtype=np.float32),
                num_control_points=int(num_control_points),
                spline_degree=int(spline_degree),
            )
            guided_qp_time += float(timing["guided_qp_time"])
            guidance_step_infos.append({
                "step_index": int(step_index),
                "timestep": int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
                "guidance_step_index": int(guidance_index),
                "candidate_infos": candidate_infos,
            })
            guided_x0 = torch.from_numpy(guided_np).to(device=policy.device, dtype=policy.dtype)
            noisy = stepper.step_with_x0_override(
                model_output,
                timestep,
                noisy,
                pred_original_sample_override=guided_x0,
                eta=eta,
                generator=generator,
            )
        else:
            noisy = scheduler.step(model_output, timestep, noisy, eta=eta, **step_kwargs).prev_sample
        noisy[cond_mask] = cond_data[cond_mask]
    final_x0 = noisy.detach().cpu().numpy().astype(np.float32)
    result = guidance_runner.finalize_candidates(
        x0_candidates=final_x0,
        q_start_normalized=np.asarray(q_start_normalized, dtype=np.float32),
        q_goal_normalized=np.asarray(q_goal_normalized, dtype=np.float32),
        delta_w_mean=np.asarray(delta_w_mean, dtype=np.float32),
        delta_w_std=np.asarray(delta_w_std, dtype=np.float32),
        num_control_points=int(num_control_points),
        spline_degree=int(spline_degree),
    )
    log = dict(result.log)
    log.update({
        "planner_mode": "qp_guided_diffusion",
        "num_candidates_guided": int(batch_size),
        "guidance_steps_applied": int(len(guidance_step_infos)),
        "guidance_step_infos": guidance_step_infos,
        "diffusion_time": float(diffusion_time),
        "guided_qp_time": float(guided_qp_time),
        "total_planning_time": float(time.perf_counter() - total_start),
    })
    if guidance_step_infos:
        last_infos = guidance_step_infos[-1]["candidate_infos"]
        if last_infos:
            selected_idx = int(log.get("selected_candidate_index", -1))
            selected_info = next((info for info in last_infos if int(info["candidate_index"]) == selected_idx), last_infos[0])
            log.update({
                "repairability_score": float(selected_info.get("repairability_score", math.nan)),
                "qp_status": str(selected_info.get("qp_status", "unknown")),
                "qp_slack_sum": float(selected_info.get("qp_slack_sum", selected_info.get("slack_sum", math.nan))),
                "qp_delta_norm": float(selected_info.get("qp_delta_norm", math.nan)),
                "min_sdf": float(selected_info.get("min_sdf", math.nan)),
                "num_penetration": int(selected_info.get("num_penetration", 0)),
                "max_penetration_depth": float(selected_info.get("max_penetration_depth", math.nan)),
            })
    result.log.clear()
    result.log.update(log)
    return result


def config_to_dict(config: LateStageQPGuidedDDIMConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["scp_config"] = asdict(config.scp_config)
    return payload
