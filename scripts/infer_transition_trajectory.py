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
from diffusion_policy_3d.common.increment import load_increment_stats
from diffusion_policy_3d.common.input_data import load_planning_input_data
from diffusion_policy_3d.common.pointcloud_roi import (
    extract_normalized_xy_radius_height_roi_from_stl_and_npz,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained diffusion policy on a new STL + transition NPZ pair "
            "and export predicted delta and joint trajectories."
        )
    )
    parser.add_argument("--stl-path", type=str, required=True, help="Path to the workpiece STL.")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to the transition NPZ.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to a trained checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="data/raw_data/results/job_000_increment_stats.npz",
        help="Path to increment mean/std statistics used during training.",
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
    parser.add_argument("--num-output-points", type=int, default=1024)
    parser.add_argument("--num-mesh-sample-points", type=int, default=100000)
    parser.add_argument("--stl-x-offset-mm", type=float, default=500.0)
    parser.add_argument("--urdf-path", type=str, default=None)
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
    planning_result = load_planning_input_data(
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


def denormalize_deltas(normalized_deltas: np.ndarray, stats_path: str) -> np.ndarray:
    mean, std = load_increment_stats(stats_path)
    mean = mean.reshape(1, -1)
    std = std.reshape(1, -1)
    return (normalized_deltas * std + mean).astype(np.float32)


def reconstruct_joint_trajectory(q_start: np.ndarray, delta_trajectory: np.ndarray) -> np.ndarray:
    q_start = np.asarray(q_start, dtype=np.float32).reshape(1, -1)
    delta_trajectory = np.asarray(delta_trajectory, dtype=np.float32)
    joints = [q_start[0]]
    current = q_start[0].copy()
    for delta in delta_trajectory:
        current = current + delta
        joints.append(current.copy())
    return np.stack(joints, axis=0).astype(np.float32)


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

    output_dir = (
        pathlib.Path(args.output_dir)
        if args.output_dir is not None
        else npz_path.parent / f"{npz_path.stem}_inference"
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
    pred_delta_horizon = denormalize_deltas(pred_action_horizon, args.stats_path)

    npz_data = np.load(npz_path)
    if "q_start" not in npz_data.files:
        raise KeyError(f"`q_start` is required in {npz_path} to reconstruct joint trajectories.")
    q_start = np.asarray(npz_data["q_start"], dtype=np.float32)

    pred_joint_horizon = reconstruct_joint_trajectory(q_start, pred_delta_horizon)

    gt_joint_traj = None
    if "q_plan" in npz_data.files:
        gt_joint_traj = np.asarray(npz_data["q_plan"], dtype=np.float32)

    np.save(output_dir / "pred_action_window_normalized.npy", pred_action_window)
    np.save(output_dir / "pred_action_horizon_normalized.npy", pred_action_horizon)
    np.save(output_dir / "pred_delta_horizon.npy", pred_delta_horizon)
    np.save(output_dir / "pred_joint_horizon.npy", pred_joint_horizon)
    np.save(output_dir / "point_cloud.npy", raw_obs["point_cloud"])

    save_joint_plot(
        pred_joint_traj=pred_joint_horizon,
        gt_joint_traj=gt_joint_traj,
        output_path=output_dir / "pred_joint_horizon.png",
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "npz_path": str(npz_path),
        "stl_path": args.stl_path,
        "stats_path": args.stats_path,
        "output_dir": str(output_dir),
        "n_obs_steps": int(workspace.cfg.n_obs_steps),
        "n_action_steps": int(workspace.cfg.n_action_steps),
        "horizon": int(workspace.cfg.horizon),
        "pred_action_window_shape": list(pred_action_window.shape),
        "pred_action_horizon_shape": list(pred_action_horizon.shape),
        "pred_joint_horizon_shape": list(pred_joint_horizon.shape),
        "has_ground_truth_q_plan": bool(gt_joint_traj is not None),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"output_dir: {output_dir}")
    print(f"pred_action_window_shape: {pred_action_window.shape}")
    print(f"pred_action_horizon_shape: {pred_action_horizon.shape}")
    print(f"pred_joint_horizon_shape: {pred_joint_horizon.shape}")
    if gt_joint_traj is not None:
        print(f"gt_q_plan_shape: {gt_joint_traj.shape}")
    print(f"saved_plot: {output_dir / 'pred_joint_horizon.png'}")


if __name__ == "__main__":
    main()
