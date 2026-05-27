import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.increment import save_increment_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute mean/std statistics for resampled joint delta trajectories."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw_data/results/job_000",
        help="Directory containing transition NPZ files.",
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        default="data/raw_data/results/job_000_increment_stats.npz",
        help="Path to save the increment statistics .npz file.",
    )
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="q_plan",
        help="Trajectory key in each NPZ. Use the same key as build_transition_zarr.py.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=65,
        help="Absolute trajectory length before delta conversion.",
    )
    parser.add_argument(
        "--std-eps",
        type=float,
        default=1e-6,
        help="Minimum std used to avoid division by zero during normalization.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    npz_paths = sorted(str(path) for path in pathlib.Path(args.input_dir).rglob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {args.input_dir}")

    stats = save_increment_stats(
        npz_paths=npz_paths,
        output_path=args.output_stats,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
        std_eps=args.std_eps,
    )

    print(f"npz_count: {len(npz_paths)}")
    print(f"delta_count: {int(stats['count'])}")
    print(f"mean: {stats['mean']}")
    print(f"std: {stats['std']}")
    print(f"var: {stats['var']}")
    print(f"saved_stats: {args.output_stats}")


if __name__ == "__main__":
    main()
