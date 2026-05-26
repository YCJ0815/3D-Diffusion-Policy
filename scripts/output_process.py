import argparse
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.increment import build_normalized_increment_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a normalized delta joint trajectory for a single transition NPZ."
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        required=True,
        help="Path to the source transition NPZ file.",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        required=True,
        help="Path to save the normalized delta joint trajectory as .npy.",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="data/raw_data/results/job_000_increment_stats.npz",
        help="Path to the dataset joint mean/std statistics .npz file.",
    )
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="q_plan",
        help="Trajectory key in the NPZ to process. Default: q_plan.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=65,
        help="Absolute trajectory length before delta conversion. Default: 65.",
    )
    return parser


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = build_parser().parse_args()

    normalized_increment = build_normalized_increment_trajectory(
        npz_path=args.npz_path,
        stats_path=args.stats_path,
        trajectory_key=args.trajectory_key,
        target_steps=args.target_steps,
    )

    output_npy = pathlib.Path(args.output_npy)
    ensure_parent(output_npy)
    np.save(output_npy, normalized_increment.astype(np.float32))

    print(f"trajectory_key: {args.trajectory_key}")
    print(f"normalized_delta_trajectory: {normalized_increment.shape}")
    print(f"stats_path: {args.stats_path}")
    print(f"saved_npy: {output_npy}")


if __name__ == "__main__":
    main()
