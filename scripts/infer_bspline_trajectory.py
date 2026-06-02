import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from train import TrainDP3Workspace
from diffusion_policy_3d.common.bspline import (
    fit_quintic_bspline_to_npz_trajectory,
    load_delta_w_stats,
    reconstruct_trajectory_from_normalized_free_residual,
    unnormalize_joint_trajectory_with_urdf_limits,
)
from diffusion_policy_3d.common.input_data import load_bspline_planning_input_data
from diffusion_policy_3d.common.pointcloud_roi import (
    extract_normalized_xy_radius_height_roi_from_stl_and_npz,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained B-spline diffusion policy on a transition NPZ and recover "
            "the full 64x6 joint trajectory from normalized predicted control-point residuals."
        )
    )
    parser.add_argument("--stl-path", type=str, required=True, help="Path to the workpiece STL.")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to the transition NPZ.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to a trained checkpoint (.ckpt).")
    parser.add_argument(
        "--stats-path",
        type=str,
        required=True,
        help="Path to the B-spline delta_w statistics (.npz) used during training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write inference artifacts. Default: alongside NPZ.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--norm-m", type=float, default=0.1)
    parser.add_argument("--radius-m", type=float, default=0.1)
    parser.add_argument("--height-m", type=float, default=0.1)
    parser.add_argument("--num-output-points", type=int, default=512)
    parser.add_argument("--num-mesh-sample-points", type=int, default=100000)
    parser.add_argument("--stl-x-offset-mm", type=float, default=500.0)
    parser.add_argument("--urdf-path", type=str, default=None)
    parser.add_argument("--trajectory-key", type=str, default="q_plan")
    parser.add_argument("--target-steps", type=int, default=64)
    parser.add_argument("--num-control-points", type=int, default=16)
    parser.add_argument("--spline-degree", type=int, default=5)
    parser.add_argument("--use-poisson-disk", action="store_true")
    return parser


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_obs_dict(
    stl_path: str,
    npz_path: str,
    norm_m: float,
    radius_m: float,
    height_m: float,
    num_output_points: int,
    num_mesh_sample_points: int,
    stl_x_offset_mm: float,
    urdf_path: str | None,
    use_poisson_disk: bool,
    n_obs_steps: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    pointcloud_result = extract_normalized_xy_radius_height_roi_from_stl_and_npz(
        stl_path=stl_path,
        npz_path=npz_path,
        radius_m=radius_m,
        height_m=height_m,
        norm_m=norm_m,
        num_output_points=num_output_points,
        num_mesh_sample_points=num_mesh_sample_points,
        use_poisson_disk=use_poisson_disk,
        stl_x_offset_mm=stl_x_offset_mm,
    )
    planning_result = load_bspline_planning_input_data(
        npz_path=npz_path,
        norm=norm_m,
        urdf_path=urdf_path,
    )

    raw_obs = {
        "point_cloud": pointcloud_result.point_cloud.astype(np.float32),
        "goal_position": planning_result.goal_position.astype(np.float32),
        "goal_direction": planning_result.goal_direction.astype(np.float32),
        "first_joint_angles_normalized": planning_result.first_joint_angles_normalized.astype(np.float32),
        "last_joint_angles_normalized": planning_result.last_joint_angles_normalized.astype(np.float32),
    }

    obs_dict = {}
    for key, value in raw_obs.items():
        value = np.asarray(value, dtype=np.float32)
        value = np.expand_dims(value, axis=0)
        value = np.expand_dims(value, axis=0)
        value = np.repeat(value, n_obs_steps, axis=1)
        obs_dict[key] = torch.from_numpy(value).to(device)
    return obs_dict, raw_obs


def save_joint_plot(
    pred_joint_traj: np.ndarray,
    output_path: pathlib.Path,
    gt_joint_traj: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes = axes.reshape(-1)
    x_pred = np.arange(pred_joint_traj.shape[0], dtype=np.int32)

    for joint_idx, ax in enumerate(axes):
        ax.plot(x_pred, pred_joint_traj[:, joint_idx], label="pred", linewidth=1.8)
        if gt_joint_traj is not None:
            x_gt = np.arange(gt_joint_traj.shape[0], dtype=np.int32)
            ax.plot(x_gt, gt_joint_traj[:, joint_idx], label="gt", linewidth=1.2, alpha=0.75)
        ax.set_title(f"joint_{joint_idx}")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    npz_path = pathlib.Path(args.npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Transition NPZ not found: {npz_path}")

    stats_path = pathlib.Path(args.stats_path)
    if not stats_path.is_file():
        raise FileNotFoundError(f"delta_w stats file not found: {stats_path}")

    output_dir = (
        pathlib.Path(args.output_dir)
        if args.output_dir is not None
        else npz_path.parent / f"{npz_path.stem}_bspline_inference"
    )
    ensure_dir(output_dir)

    device = torch.device(args.device)
    workspace = TrainDP3Workspace.create_from_checkpoint(str(checkpoint_path))
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    policy = policy.to(device)
    policy.eval()

    obs_dict, raw_obs = build_obs_dict(
        stl_path=args.stl_path,
        npz_path=str(npz_path),
        norm_m=args.norm_m,
        radius_m=args.radius_m,
        height_m=args.height_m,
        num_output_points=args.num_output_points,
        num_mesh_sample_points=args.num_mesh_sample_points,
        stl_x_offset_mm=args.stl_x_offset_mm,
        urdf_path=args.urdf_path,
        use_poisson_disk=args.use_poisson_disk,
        n_obs_steps=workspace.cfg.n_obs_steps,
        device=device,
    )

    with torch.no_grad():
        result = policy.predict_action(obs_dict)

    pred_action_window = result["action"][0].detach().cpu().numpy().astype(np.float32)
    pred_action_horizon = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
    expected_action_shape = (args.num_control_points - 6, 6)
    if pred_action_horizon.shape != expected_action_shape:
        raise ValueError(
            "Predicted normalized free control-point residual has incompatible shape. "
            f"Expected {expected_action_shape} from num_control_points={args.num_control_points}, "
            f"got {pred_action_horizon.shape}. "
            "Check that the checkpoint was trained with the same B-spline action horizon."
        )

    planning_result = load_bspline_planning_input_data(
        npz_path=str(npz_path),
        norm=args.norm_m,
        urdf_path=args.urdf_path,
    )
    mean, std = load_delta_w_stats(str(stats_path))
    recon_result = reconstruct_trajectory_from_normalized_free_residual(
        normalized_free_delta_w=pred_action_horizon,
        start_state=planning_result.first_joint_angles_normalized,
        end_state=planning_result.last_joint_angles_normalized,
        mean=mean,
        std=std,
        num_control_points=args.num_control_points,
        num_steps=args.target_steps,
        degree=args.spline_degree,
    )
    pred_joint_horizon_normalized = recon_result["fitted_trajectory"].astype(np.float32)
    pred_joint_horizon = unnormalize_joint_trajectory_with_urdf_limits(
        normalized_trajectory=pred_joint_horizon_normalized,
        lower_limits=planning_result.joint_lower_limits,
        upper_limits=planning_result.joint_upper_limits,
    )

    gt_fit_result = None
    gt_joint_traj = None
    npz_data = np.load(npz_path)
    if planning_result.trajectory_key in npz_data.files:
        gt_joint_traj = np.asarray(npz_data[planning_result.trajectory_key], dtype=np.float32)
        gt_fit_result = fit_quintic_bspline_to_npz_trajectory(
            npz_path=str(npz_path),
            trajectory_key=args.trajectory_key,
            target_steps=args.target_steps,
            urdf_path=args.urdf_path,
            num_control_points=args.num_control_points,
            degree=args.spline_degree,
        )

    np.save(output_dir / "pred_action_window_normalized.npy", pred_action_window)
    np.save(output_dir / "pred_action_horizon_normalized.npy", pred_action_horizon)
    np.save(output_dir / "pred_delta_w.npy", recon_result["delta_w"])
    np.save(output_dir / "pred_w_line.npy", recon_result["w_line"])
    np.save(output_dir / "pred_w_star.npy", recon_result["w_star"])
    np.save(output_dir / "pred_joint_horizon_normalized.npy", pred_joint_horizon_normalized)
    np.save(output_dir / "pred_joint_horizon.npy", pred_joint_horizon)
    np.save(output_dir / "point_cloud.npy", raw_obs["point_cloud"])

    if gt_fit_result is not None:
        np.save(output_dir / "gt_w_star.npy", gt_fit_result["w_star"].astype(np.float32))
        np.save(output_dir / "gt_delta_w.npy", gt_fit_result["delta_w"].astype(np.float32))
        np.save(output_dir / "gt_joint_horizon_normalized.npy", gt_fit_result["normalized_trajectory"].astype(np.float32))
        np.save(
            output_dir / "gt_joint_horizon.npy",
            unnormalize_joint_trajectory_with_urdf_limits(
                normalized_trajectory=gt_fit_result["normalized_trajectory"],
                lower_limits=planning_result.joint_lower_limits,
                upper_limits=planning_result.joint_upper_limits,
            ),
        )

    save_joint_plot(
        pred_joint_traj=pred_joint_horizon,
        gt_joint_traj=gt_joint_traj,
        output_path=output_dir / "pred_joint_horizon.png",
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "npz_path": str(npz_path),
        "stl_path": args.stl_path,
        "stats_path": str(stats_path),
        "output_dir": str(output_dir),
        "uses_axis_symmetric_tcp_canonicalization": True,
        "n_obs_steps": int(workspace.cfg.n_obs_steps),
        "n_action_steps": int(workspace.cfg.n_action_steps),
        "policy_horizon": int(workspace.cfg.horizon),
        "target_steps": int(args.target_steps),
        "num_control_points": int(args.num_control_points),
        "spline_degree": int(args.spline_degree),
        "pred_action_window_shape": list(pred_action_window.shape),
        "pred_action_horizon_shape": list(pred_action_horizon.shape),
        "pred_joint_horizon_shape": list(pred_joint_horizon.shape),
        "trajectory_key": planning_result.trajectory_key,
        "has_ground_truth_trajectory": bool(gt_joint_traj is not None),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"output_dir: {output_dir}")
    print(f"pred_action_window_shape: {pred_action_window.shape}")
    print(f"pred_action_horizon_shape: {pred_action_horizon.shape}")
    print(f"pred_joint_horizon_shape: {pred_joint_horizon.shape}")
    if gt_joint_traj is not None:
        print(f"gt_trajectory_shape: {gt_joint_traj.shape}")
    print("uses_axis_symmetric_tcp_canonicalization: True")
    print(f"saved_plot: {output_dir / 'pred_joint_horizon.png'}")


if __name__ == "__main__":
    main()
