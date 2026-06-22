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
    "enable_surface_cbf_qp_guidance": False,
    "num_candidates": 32,
    "candidate_inference_steps": None,
    "guidance_steps": 2,
    "guidance_qp_candidates": 4,
    "guidance_active_constraints": 16,
    "guidance_check_steps": 64,
    "guidance_cert_steps": 256,
    "guidance_cert_swept_intermediate": 3,
    "guidance_d_safe": 0.03,
    "guidance_d_trigger": 0.06,
    "guidance_d_cert": 0.01,
    "guidance_eps_deep": 0.03,
    "guidance_delta_max": 0.05,
    "guidance_lambda_s": 0.25,
    "guidance_rho": 1.0e5,
    "guidance_ddim_eta": 0.0,
    "guidance_joint_limit_steps": 32,
    "guidance_targets": [-0.02, 0.0],
    "guidance_fallback_to_terminal_cbf": True,
    "robot_surface_points_per_link": {"pen_link": 80, "wrist3": 16},
}


GUIDANCE_CONFIG_KEY_PATHS: dict[str, tuple[str, ...]] = {
    "enable_surface_cbf_qp_guidance": ("surface_cbf_qp_guidance", "enabled"),
    "num_candidates": ("surface_cbf_qp_guidance", "num_candidates"),
    "candidate_inference_steps": ("surface_cbf_qp_guidance", "candidate_inference_steps"),
    "guidance_steps": ("surface_cbf_qp_guidance", "runner", "guidance_steps"),
    "guidance_qp_candidates": ("surface_cbf_qp_guidance", "runner", "qp_candidates"),
    "guidance_active_constraints": ("surface_cbf_qp_guidance", "runner", "active_constraints"),
    "guidance_check_steps": ("surface_cbf_qp_guidance", "runner", "check_steps"),
    "guidance_cert_steps": ("surface_cbf_qp_guidance", "runner", "cert_steps"),
    "guidance_cert_swept_intermediate": ("surface_cbf_qp_guidance", "runner", "cert_swept_intermediate"),
    "guidance_d_safe": ("surface_cbf_qp_guidance", "runner", "d_safe"),
    "guidance_d_trigger": ("surface_cbf_qp_guidance", "runner", "d_trigger"),
    "guidance_d_cert": ("surface_cbf_qp_guidance", "runner", "d_cert"),
    "guidance_eps_deep": ("surface_cbf_qp_guidance", "runner", "eps_deep"),
    "guidance_delta_max": ("surface_cbf_qp_guidance", "runner", "delta_max"),
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
        "enable_surface_cbf_qp_guidance",
        "guidance_fallback_to_terminal_cbf",
    ),
    "sampling": (
        "num_candidates",
        "candidate_inference_steps",
        "guidance_steps",
        "guidance_ddim_eta",
    ),
    "cbf_thresholds": (
        "guidance_d_safe",
        "guidance_d_trigger",
        "guidance_d_cert",
        "guidance_eps_deep",
        "guidance_targets",
    ),
    "qp_optimization": (
        "guidance_qp_candidates",
        "guidance_active_constraints",
        "guidance_delta_max",
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
    enabled_group = parser.add_mutually_exclusive_group()
    enabled_group.add_argument(
        "--enable-surface-cbf-qp-guidance",
        dest="enable_surface_cbf_qp_guidance",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable the independent late-stage surface-sample CBF-QP guided denoising path.",
    )
    enabled_group.add_argument(
        "--disable-surface-cbf-qp-guidance",
        dest="enable_surface_cbf_qp_guidance",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable surface-sample CBF-QP guided denoising even if the config file enables it.",
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
    parser.add_argument(
        "--guidance-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of final denoising steps that run surface-sample CBF-QP guidance.",
    )
    parser.add_argument(
        "--guidance-qp-candidates",
        type=int,
        default=argparse.SUPPRESS,
        help="How many near-collision candidates to project with QP per guidance step.",
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
    if not config_path.is_file():
        raise FileNotFoundError(f"Surface CBF-QP guidance config not found: {config_path}")
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Surface CBF-QP guidance config must deserialize to a mapping, got {type(loaded)!r}"
        )

    keys = [
        "enable_surface_cbf_qp_guidance",
        "guidance_steps",
        "guidance_qp_candidates",
        "guidance_active_constraints",
        "guidance_check_steps",
        "guidance_cert_steps",
        "guidance_cert_swept_intermediate",
        "guidance_d_safe",
        "guidance_d_trigger",
        "guidance_d_cert",
        "guidance_eps_deep",
        "guidance_delta_max",
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
    return args


def build_surface_cbf_qp_parameter_summary(
    args: argparse.Namespace,
    *,
    include_num_candidates: bool,
    include_candidate_inference_steps: bool,
) -> dict[str, Any]:
    included_keys = {
        "enable_surface_cbf_qp_guidance",
        "guidance_steps",
        "guidance_qp_candidates",
        "guidance_active_constraints",
        "guidance_check_steps",
        "guidance_cert_steps",
        "guidance_cert_swept_intermediate",
        "guidance_d_safe",
        "guidance_d_trigger",
        "guidance_d_cert",
        "guidance_eps_deep",
        "guidance_delta_max",
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
