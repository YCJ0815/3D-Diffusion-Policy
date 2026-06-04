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
        "--input-dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of directories containing transition NPZ files. "
            "When provided, all directories are scanned and merged."
        ),
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


def collect_npz_paths(input_dir: str, input_dirs: list[str] | None) -> list[str]:
    search_dirs = list(input_dirs) if input_dirs else [input_dir]
    npz_paths: list[str] = []
    for directory in search_dirs:
        npz_paths.extend(
            sorted(str(path) for path in pathlib.Path(directory).rglob("transition_*.npz"))
        )
    return sorted(set(npz_paths))


def main() -> None:
    args = build_parser().parse_args()
    npz_paths = collect_npz_paths(
        input_dir=args.input_dir,
        input_dirs=args.input_dirs,
    )
    if not npz_paths:
        searched = args.input_dirs if args.input_dirs else [args.input_dir]
        raise FileNotFoundError(f"No transition_*.npz files found under: {searched}")

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
    print(f"input_dirs: {args.input_dirs if args.input_dirs else [args.input_dir]}")
    print(f"saved_stats: {args.output_stats}")


if __name__ == "__main__":
    main()
