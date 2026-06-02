import argparse
import pathlib
import sys

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


DEFAULT_KEYS = (
    "action",
    "goal_position",
    "goal_direction",
    "first_joint_angles_normalized",
    "last_joint_angles_normalized",
    "point_cloud",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print value ranges for normalized transition zarr datasets."
    )
    parser.add_argument("zarr_path", type=str, help="Path to the transition zarr dataset.")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=list(DEFAULT_KEYS),
        help="Dataset keys under the zarr data group to inspect.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=4096,
        help="Maximum rows to load per key. Use -1 to scan the full array.",
    )
    return parser


def summarize_array(array, max_rows: int) -> dict[str, object]:
    if max_rows is not None and max_rows > 0 and array.shape[0] > max_rows:
        indices = np.linspace(0, array.shape[0] - 1, max_rows, dtype=np.int64)
        values = array.get_orthogonal_selection((indices, ...))
        rows = int(max_rows)
        sampled = True
    else:
        values = array[:]
        rows = int(array.shape[0])
        sampled = False

    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1)
    quantiles = np.quantile(flat, [0.01, 0.5, 0.99])
    return {
        "shape": tuple(array.shape),
        "rows": rows,
        "sampled": sampled,
        "min": float(np.min(flat)),
        "p01": float(quantiles[0]),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "max": float(np.max(flat)),
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "zarr is required to inspect compressed zarr datasets. "
            "Run this script in the same environment used for training."
        ) from exc

    root = zarr.open(args.zarr_path, mode="r")
    data = root["data"]
    for key in args.keys:
        if key not in data:
            print(f"{key}: missing")
            continue
        stats = summarize_array(data[key], args.max_rows)
        sampled_suffix = " sampled" if stats["sampled"] else ""
        print(f"{key}: shape={stats['shape']} rows={stats['rows']}{sampled_suffix}")
        print(
            "  "
            f"min={stats['min']:.6g} p01={stats['p01']:.6g} "
            f"mean={stats['mean']:.6g} std={stats['std']:.6g} "
            f"median={stats['median']:.6g} p99={stats['p99']:.6g} "
            f"max={stats['max']:.6g}"
        )


if __name__ == "__main__":
    main()
