from __future__ import annotations

import argparse
import copy
import pathlib
from typing import Any

from omegaconf import OmegaConf


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SURFACE_CBF_QP_GUIDANCE_CONFIG_PATH = (
    PROJECT_ROOT
    / "3D-Diffusion-Policy"
    / "diffusion_policy_3d"
    / "config"
    / "guidance"
    / "surface_cbf_qp.yaml"
)


GUIDANCE_CONFIG_FALLBACKS: dict[str, Any] = {
    "planner_mode": "baseline",
    "enable_surface_cbf_qp_guidance": False,
    "num_candidates": 32,
    "candidate_inference_steps": None,
    "qp_candidates": 4,
    "guidance_timesteps": [],
    "qp_inner_scp_rounds": 2,
    "coarse_check_steps": 32,
    "guidance_pen_link_points": 80,
    "guidance_wrist3_points": 16,
    "final_post_qp_candidates": 1,
    "final_backup_candidates": 1,
    "final_post_qp_rounds": 2,
    "guidance_trigger_distance": 0.06,
    "guidance_safe_distance": 0.05,
    "trust_region_start": 0.015,
    "trust_region_end": 0.05,
    "blend_weights": [0.25, 0.5, 0.75],
    "repair_score_weights": [1.0, 10.0, 1.0],
    "guidance_steps": 3,
    "guidance_max_risk_segments": 3,
    "guidance_window_radius": 2,
    "guidance_points_per_segment": 2,
    "guidance_min_constraints_per_segment": 4,
    "guidance_active_constraints": 24,
    "guidance_check_steps": 32,
    "guidance_cert_steps": 64,
    "guidance_cert_swept_intermediate": 3,
    "guidance_d_safe": 0.03,
    "guidance_d_trigger": 0.06,
    "guidance_d_cert": 0.01,
    "guidance_eps_deep": 0.03,
    "guidance_delta_max": 0.05,
    "guidance_scp_iterations": 2,
    "guidance_delta_max_total": 0.05,
    "guidance_delta_max_pass1": 0.025,
    "guidance_delta_max_pass2": 0.025,
    "guidance_d_trigger_pass2_offset": 0.005,
    "guidance_margin_buffer": 0.005,
    "enable_local_waypoint_qp_after_certificate": True,
    "local_waypoint_qp_window_radius": 2,
    "local_waypoint_qp_max_collision_segments": 2,
    "local_waypoint_qp_min_clearance_trigger": -0.01,
    "local_waypoint_qp_target_buffer": 0.005,
    "local_waypoint_qp_lambda_s": 0.25,
    "local_waypoint_qp_delta_max": 0.02,
    "local_waypoint_qp_max_velocity_step": 0.2,
    "local_waypoint_qp_max_acceleration_step": 0.4,
    "local_waypoint_qp_maxiter": 100,
    "guidance_lambda_s": 0.25,
    "guidance_rho": 1.0e5,
    "guidance_ddim_eta": 0.0,
    "guidance_joint_limit_steps": 32,
    "guidance_targets": [-0.02, 0.0],
    "guidance_fallback_to_terminal_cbf": True,
    "robot_surface_points_per_link": {"pen_link": 80, "wrist_3_link": 16},
}


GUIDANCE_CONFIG_KEY_PATHS: dict[str, tuple[str, ...]] = {
    "planner_mode": ("surface_cbf_qp_guidance", "planner_mode"),
    "enable_surface_cbf_qp_guidance": ("surface_cbf_qp_guidance", "enabled"),
    "num_candidates": ("surface_cbf_qp_guidance", "num_candidates"),
    "candidate_inference_steps": ("surface_cbf_qp_guidance", "candidate_inference_steps"),
    "qp_candidates": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "qp_candidates"),
    "guidance_timesteps": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "guidance_timesteps"),
    "qp_inner_scp_rounds": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "qp_inner_scp_rounds"),
    "coarse_check_steps": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "coarse_check_steps"),
    "guidance_pen_link_points": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "guidance_pen_link_points"),
    "guidance_wrist3_points": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "guidance_wrist3_points"),
    "final_post_qp_candidates": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "final_post_qp_candidates"),
    "final_backup_candidates": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "final_backup_candidates"),
    "final_post_qp_rounds": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "final_post_qp_rounds"),
    "guidance_trigger_distance": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "guidance_trigger_distance"),
    "guidance_safe_distance": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "guidance_safe_distance"),
    "trust_region_start": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "trust_region_start"),
    "trust_region_end": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "trust_region_end"),
    "blend_weights": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "blend_weights"),
    "repair_score_weights": ("surface_cbf_qp_guidance", "qp_guided_diffusion", "repair_score_weights"),
    "guidance_steps": ("surface_cbf_qp_guidance", "runner", "guidance_steps"),
    "guidance_max_risk_segments": ("surface_cbf_qp_guidance", "runner", "max_risk_segments"),
    "guidance_window_radius": ("surface_cbf_qp_guidance", "runner", "window_radius"),
    "guidance_points_per_segment": ("surface_cbf_qp_guidance", "runner", "points_per_segment"),
    "guidance_min_constraints_per_segment": ("surface_cbf_qp_guidance", "runner", "min_constraints_per_segment"),
    "guidance_active_constraints": ("surface_cbf_qp_guidance", "runner", "active_constraints"),
    "guidance_check_steps": ("surface_cbf_qp_guidance", "runner", "check_steps"),
    "guidance_cert_steps": ("surface_cbf_qp_guidance", "runner", "cert_steps"),
    "guidance_cert_swept_intermediate": ("surface_cbf_qp_guidance", "runner", "cert_swept_intermediate"),
    "guidance_d_safe": ("surface_cbf_qp_guidance", "runner", "d_safe"),
    "guidance_d_trigger": ("surface_cbf_qp_guidance", "runner", "d_trigger"),
    "guidance_d_cert": ("surface_cbf_qp_guidance", "runner", "d_cert"),
    "guidance_eps_deep": ("surface_cbf_qp_guidance", "runner", "eps_deep"),
    "guidance_delta_max": ("surface_cbf_qp_guidance", "runner", "delta_max"),
    "guidance_scp_iterations": ("surface_cbf_qp_guidance", "runner", "scp_iterations"),
    "guidance_delta_max_total": ("surface_cbf_qp_guidance", "runner", "delta_max_total"),
    "guidance_delta_max_pass1": ("surface_cbf_qp_guidance", "runner", "delta_max_pass1"),
    "guidance_delta_max_pass2": ("surface_cbf_qp_guidance", "runner", "delta_max_pass2"),
    "guidance_d_trigger_pass2_offset": ("surface_cbf_qp_guidance", "runner", "d_trigger_pass2_offset"),
    "guidance_margin_buffer": ("surface_cbf_qp_guidance", "runner", "margin_buffer"),
    "enable_local_waypoint_qp_after_certificate": ("surface_cbf_qp_guidance", "runner", "enable_local_waypoint_qp_after_certificate"),
    "local_waypoint_qp_window_radius": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_window_radius"),
    "local_waypoint_qp_max_collision_segments": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_max_collision_segments"),
    "local_waypoint_qp_min_clearance_trigger": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_min_clearance_trigger"),
    "local_waypoint_qp_target_buffer": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_target_buffer"),
    "local_waypoint_qp_lambda_s": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_lambda_s"),
    "local_waypoint_qp_delta_max": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_delta_max"),
    "local_waypoint_qp_max_velocity_step": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_max_velocity_step"),
    "local_waypoint_qp_max_acceleration_step": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_max_acceleration_step"),
    "local_waypoint_qp_maxiter": ("surface_cbf_qp_guidance", "runner", "local_waypoint_qp_maxiter"),
    "guidance_lambda_s": ("surface_cbf_qp_guidance", "runner", "lambda_s"),
    "guidance_rho": ("surface_cbf_qp_guidance", "runner", "rho"),
    "guidance_ddim_eta": ("surface_cbf_qp_guidance", "runner", "ddim_eta"),
    "guidance_joint_limit_steps": ("surface_cbf_qp_guidance", "runner", "joint_limit_steps"),
    "guidance_targets": ("surface_cbf_qp_guidance", "runner", "guidance_targets"),
    "guidance_fallback_to_terminal_cbf": ("surface_cbf_qp_guidance", "runner", "fallback_to_terminal_cbf"),
    "robot_surface_points_per_link": ("surface_cbf_qp_guidance", "pybullet", "robot_surface_points_per_link"),
}


GUIDANCE_PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "switches": (
        "planner_mode",
        "enable_surface_cbf_qp_guidance",
        "guidance_fallback_to_terminal_cbf",
    ),
    "sampling": (
        "num_candidates",
        "candidate_inference_steps",
        "guidance_steps",
        "guidance_ddim_eta",
        "qp_candidates",
        "guidance_timesteps",
        "blend_weights",
        "guidance_pen_link_points",
        "guidance_wrist3_points",
        "final_post_qp_candidates",
        "final_backup_candidates",
        "final_post_qp_rounds",
    ),
    "cbf_thresholds": (
        "guidance_d_safe",
        "guidance_d_trigger",
        "guidance_d_cert",
        "guidance_eps_deep",
        "guidance_targets",
        "guidance_trigger_distance",
        "guidance_safe_distance",
    ),
    "qp_optimization": (
        "qp_inner_scp_rounds",
        "coarse_check_steps",
        "trust_region_start",
        "trust_region_end",
        "repair_score_weights",
        "guidance_max_risk_segments",
        "guidance_window_radius",
        "guidance_points_per_segment",
        "guidance_min_constraints_per_segment",
        "guidance_active_constraints",
        "guidance_delta_max",
        "guidance_scp_iterations",
        "guidance_delta_max_total",
        "guidance_delta_max_pass1",
        "guidance_delta_max_pass2",
        "guidance_d_trigger_pass2_offset",
        "guidance_margin_buffer",
        "enable_local_waypoint_qp_after_certificate",
        "local_waypoint_qp_window_radius",
        "local_waypoint_qp_max_collision_segments",
        "local_waypoint_qp_min_clearance_trigger",
        "local_waypoint_qp_target_buffer",
        "local_waypoint_qp_lambda_s",
        "local_waypoint_qp_delta_max",
        "local_waypoint_qp_max_velocity_step",
        "local_waypoint_qp_max_acceleration_step",
        "local_waypoint_qp_maxiter",
        "guidance_lambda_s",
        "guidance_rho",
        "guidance_joint_limit_steps",
    ),
    "trajectory_checks": (
        "guidance_check_steps",
        "guidance_cert_steps",
        "guidance_cert_swept_intermediate",
    ),
    "geometry": (
        "robot_surface_points_per_link",
    ),
}


def add_surface_cbf_qp_guidance_parser_args(
    parser: argparse.ArgumentParser,
    *,
    include_num_candidates: bool,
    include_candidate_inference_steps: bool,
) -> None:
    parser.add_argument(
        "--surface-cbf-qp-guidance-config",
        type=str,
        default=str(DEFAULT_SURFACE_CBF_QP_GUIDANCE_CONFIG_PATH),
        help=(
            "Path to the shared surface CBF-QP guidance YAML config. "
            "CLI flags override values from this file."
        ),
    )
    parser.add_argument(
        "--planner-mode",
        choices=("baseline", "post_qp", "qp_guided_diffusion", "qp_guided_diffusion_post_qp"),
        default=argparse.SUPPRESS,
        help=(
            "Planner mode: baseline keeps raw DDIM, post_qp keeps the legacy "
            "surface CBF-QP post-processing path, qp_guided_diffusion runs the "
            "new late-stage QP-guided DDIM sampler, and "
            "qp_guided_diffusion_post_qp runs late-stage guided diffusion "
            "followed by the legacy post-QP repair/certification path."
        ),
    )
    enabled_group = parser.add_mutually_exclusive_group()
    enabled_group.add_argument(
        "--enable-surface-cbf-qp-guidance",
        dest="enable_surface_cbf_qp_guidance",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable the legacy post-generation surface CBF-QP path; maps planner_mode=baseline to post_qp.",
    )
    enabled_group.add_argument(
        "--disable-surface-cbf-qp-guidance",
        dest="enable_surface_cbf_qp_guidance",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable the legacy post-generation surface CBF-QP path even if the config file enables it.",
    )
    if include_num_candidates:
        parser.add_argument(
            "--num-candidates",
            type=int,
            default=argparse.SUPPRESS,
            help="Candidate count for candidate-pool selection or guided sampling.",
        )
    if include_candidate_inference_steps:
        parser.add_argument(
            "--candidate-inference-steps",
            type=int,
            default=argparse.SUPPRESS,
            help="Optional override for diffusion inference steps during candidate or guided sampling.",
        )
    parser.add_argument("--qp-candidates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-timesteps", type=int, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--qp-inner-scp-rounds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--coarse-check-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-pen-link-points", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-wrist3-points", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--final-post-qp-candidates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--final-backup-candidates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--final-post-qp-rounds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-trigger-distance", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-safe-distance", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--trust-region-start", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--trust-region-end", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--blend-weights", type=float, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--repair-score-weights", type=float, nargs=3, default=argparse.SUPPRESS)
    parser.add_argument(
        "--guidance-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of final denoising steps that run surface-sample CBF-QP guidance.",
    )
    parser.add_argument(
        "--guidance-max-risk-segments",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum number of independent risk segments repaired within one candidate trajectory.",
    )
    parser.add_argument(
        "--guidance-window-radius",
        type=int,
        default=argparse.SUPPRESS,
        help="Half-width of the timestep window centered at each selected risk peak.",
    )
    parser.add_argument(
        "--guidance-points-per-segment",
        type=int,
        default=argparse.SUPPRESS,
        help="How many high-risk timesteps to sample inside each selected risk segment window.",
    )
    parser.add_argument(
        "--guidance-min-constraints-per-segment",
        type=int,
        default=argparse.SUPPRESS,
        help="Minimum number of CBF constraints to keep for each selected risk segment.",
    )
    parser.add_argument(
        "--guidance-active-constraints",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum number of active surface constraints kept in each guidance QP.",
    )
    parser.add_argument(
        "--guidance-check-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Dense B-spline evaluation steps used during per-step guidance collision checks.",
    )
    parser.add_argument(
        "--guidance-cert-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Dense B-spline evaluation steps used for the final certificate check.",
    )
    parser.add_argument(
        "--guidance-cert-swept-intermediate",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of swept interpolation points inserted between adjacent certificate waypoints.",
    )
    parser.add_argument("--guidance-d-safe", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-d-trigger", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-d-cert", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-eps-deep", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-delta-max", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-scp-iterations", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-delta-max-total", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-delta-max-pass1", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-delta-max-pass2", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-d-trigger-pass2-offset", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-margin-buffer", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--enable-local-waypoint-qp-after-certificate", dest="enable_local_waypoint_qp_after_certificate", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--disable-local-waypoint-qp-after-certificate", dest="enable_local_waypoint_qp_after_certificate", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-window-radius", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-max-collision-segments", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-min-clearance-trigger", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-target-buffer", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-lambda-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-delta-max", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-max-velocity-step", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-max-acceleration-step", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--local-waypoint-qp-maxiter", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-lambda-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--guidance-rho", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--guidance-ddim-eta",
        type=float,
        default=argparse.SUPPRESS,
        help="Deterministic DDIM eta used during the guided denoising tail.",
    )


def apply_surface_cbf_qp_guidance_config(
    args: argparse.Namespace,
    *,
    include_num_candidates: bool,
    include_candidate_inference_steps: bool,
) -> argparse.Namespace:
    config_path = pathlib.Path(args.surface_cbf_qp_guidance_config).expanduser().resolve()
    planner_mode_was_explicit = hasattr(args, "planner_mode")
    if not config_path.is_file():
        raise FileNotFoundError(f"Surface CBF-QP guidance config not found: {config_path}")
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Surface CBF-QP guidance config must deserialize to a mapping, got {type(loaded)!r}"
        )

    keys = [
        "planner_mode",
        "enable_surface_cbf_qp_guidance",
        "qp_candidates",
        "guidance_timesteps",
        "qp_inner_scp_rounds",
        "coarse_check_steps",
        "guidance_pen_link_points",
        "guidance_wrist3_points",
        "final_post_qp_candidates",
        "final_backup_candidates",
        "final_post_qp_rounds",
        "guidance_trigger_distance",
        "guidance_safe_distance",
        "trust_region_start",
        "trust_region_end",
        "blend_weights",
        "repair_score_weights",
        "guidance_steps",
        "guidance_max_risk_segments",
        "guidance_window_radius",
        "guidance_points_per_segment",
        "guidance_min_constraints_per_segment",
        "guidance_active_constraints",
        "guidance_check_steps",
        "guidance_cert_steps",
        "guidance_cert_swept_intermediate",
        "guidance_d_safe",
        "guidance_d_trigger",
        "guidance_d_cert",
        "guidance_eps_deep",
        "guidance_delta_max",
        "guidance_scp_iterations",
        "guidance_delta_max_total",
        "guidance_delta_max_pass1",
        "guidance_delta_max_pass2",
        "guidance_d_trigger_pass2_offset",
        "guidance_margin_buffer",
        "enable_local_waypoint_qp_after_certificate",
        "local_waypoint_qp_window_radius",
        "local_waypoint_qp_max_collision_segments",
        "local_waypoint_qp_min_clearance_trigger",
        "local_waypoint_qp_target_buffer",
        "local_waypoint_qp_lambda_s",
        "local_waypoint_qp_delta_max",
        "local_waypoint_qp_max_velocity_step",
        "local_waypoint_qp_max_acceleration_step",
        "local_waypoint_qp_maxiter",
        "guidance_lambda_s",
        "guidance_rho",
        "guidance_ddim_eta",
        "guidance_joint_limit_steps",
        "guidance_targets",
        "guidance_fallback_to_terminal_cbf",
        "robot_surface_points_per_link",
    ]
    if include_num_candidates:
        keys.append("num_candidates")
    if include_candidate_inference_steps:
        keys.append("candidate_inference_steps")

    for key in keys:
        if hasattr(args, key):
            continue
        setattr(args, key, copy.deepcopy(_resolve_config_value(loaded, key)))

    args.surface_cbf_qp_guidance_config = str(config_path)
    args.surface_cbf_qp_guidance_config_values = {
        key: copy.deepcopy(getattr(args, key))
        for key in keys
    }
    if (
        bool(getattr(args, "enable_surface_cbf_qp_guidance", False))
        and str(getattr(args, "planner_mode", "baseline")) == "baseline"
        and not planner_mode_was_explicit
    ):
        args.planner_mode = "post_qp"
        args.surface_cbf_qp_guidance_config_values["planner_mode"] = "post_qp"
    return args


def build_surface_cbf_qp_parameter_summary(
    args: argparse.Namespace,
    *,
    include_num_candidates: bool,
    include_candidate_inference_steps: bool,
) -> dict[str, Any]:
    included_keys = {
        "planner_mode",
        "enable_surface_cbf_qp_guidance",
        "qp_candidates",
        "guidance_timesteps",
        "qp_inner_scp_rounds",
        "coarse_check_steps",
        "guidance_pen_link_points",
        "guidance_wrist3_points",
        "final_post_qp_candidates",
        "final_backup_candidates",
        "final_post_qp_rounds",
        "guidance_trigger_distance",
        "guidance_safe_distance",
        "trust_region_start",
        "trust_region_end",
        "blend_weights",
        "repair_score_weights",
        "guidance_steps",
        "guidance_max_risk_segments",
        "guidance_window_radius",
        "guidance_points_per_segment",
        "guidance_min_constraints_per_segment",
        "guidance_active_constraints",
        "guidance_check_steps",
        "guidance_cert_steps",
        "guidance_cert_swept_intermediate",
        "guidance_d_safe",
        "guidance_d_trigger",
        "guidance_d_cert",
        "guidance_eps_deep",
        "guidance_delta_max",
        "guidance_scp_iterations",
        "guidance_delta_max_total",
        "guidance_delta_max_pass1",
        "guidance_delta_max_pass2",
        "guidance_d_trigger_pass2_offset",
        "guidance_margin_buffer",
        "enable_local_waypoint_qp_after_certificate",
        "local_waypoint_qp_window_radius",
        "local_waypoint_qp_max_collision_segments",
        "local_waypoint_qp_min_clearance_trigger",
        "local_waypoint_qp_target_buffer",
        "local_waypoint_qp_lambda_s",
        "local_waypoint_qp_delta_max",
        "local_waypoint_qp_max_velocity_step",
        "local_waypoint_qp_max_acceleration_step",
        "local_waypoint_qp_maxiter",
        "guidance_lambda_s",
        "guidance_rho",
        "guidance_ddim_eta",
        "guidance_joint_limit_steps",
        "guidance_targets",
        "guidance_fallback_to_terminal_cbf",
        "robot_surface_points_per_link",
    }
    if include_num_candidates:
        included_keys.add("num_candidates")
    if include_candidate_inference_steps:
        included_keys.add("candidate_inference_steps")

    summary: dict[str, Any] = {
        "config_path": str(args.surface_cbf_qp_guidance_config),
        "groups": {},
    }
    for group_name, keys in GUIDANCE_PARAMETER_GROUPS.items():
        group_values = {}
        for key in keys:
            if key not in included_keys or not hasattr(args, key):
                continue
            group_values[key] = copy.deepcopy(getattr(args, key))
        if group_values:
            summary["groups"][group_name] = group_values
    return summary


def _resolve_config_value(config: dict[str, Any], key: str) -> Any:
    path = GUIDANCE_CONFIG_KEY_PATHS[key]
    current: Any = config
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return copy.deepcopy(GUIDANCE_CONFIG_FALLBACKS[key])
        current = current[part]
    return copy.deepcopy(current)
