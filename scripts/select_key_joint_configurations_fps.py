import argparse
import json
import pathlib
from datetime import datetime, timezone

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select key joint configurations from a normalized joint pool using "
            "farthest point sampling in joint space."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="analysis_outputs/joint_configuration_pool",
        help="Directory containing first-stage joint pool artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs/key_joint_configurations_fps",
        help="Directory to save FPS-selected key joint configurations.",
    )
    parser.add_argument(
        "--num-key-configs",
        type=int,
        default=128,
        help="Number of key joint configurations to select.",
    )
    return parser


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_required_arrays(input_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict]:
    raw_path = input_dir / "joint_configuration_pool_raw.npy"
    normalized_path = input_dir / "joint_configuration_pool_normalized.npy"
    manifest_path = input_dir / "manifest.json"

    missing = [
        str(path.name)
        for path in (raw_path, normalized_path, manifest_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required first-stage artifacts in {input_dir}: {missing}"
        )

    raw_pool = np.load(raw_path)
    normalized_pool = np.load(normalized_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        input_manifest = json.load(f)

    raw_pool = np.asarray(raw_pool, dtype=np.float32)
    normalized_pool = np.asarray(normalized_pool, dtype=np.float32)

    if raw_pool.ndim != 2 or raw_pool.shape[1] != 6:
        raise ValueError(f"raw joint pool must have shape [N, 6], got {raw_pool.shape}")
    if normalized_pool.ndim != 2 or normalized_pool.shape[1] != 6:
        raise ValueError(
            f"normalized joint pool must have shape [N, 6], got {normalized_pool.shape}"
        )
    if raw_pool.shape != normalized_pool.shape:
        raise ValueError(
            f"raw and normalized joint pools must have the same shape, got {raw_pool.shape} and {normalized_pool.shape}"
        )
    if raw_pool.shape[0] == 0:
        raise ValueError("Joint configuration pool is empty.")

    return raw_pool, normalized_pool, input_manifest


def select_first_index_from_median(normalized_pool: np.ndarray) -> int:
    median_vector = np.median(normalized_pool, axis=0).astype(np.float32)
    squared_distances = np.sum((normalized_pool - median_vector[None, :]) ** 2, axis=1)
    return int(np.argmin(squared_distances))


def farthest_point_sampling_joint_space(
    normalized_pool: np.ndarray,
    num_key_configs: int,
) -> np.ndarray:
    pool_size = normalized_pool.shape[0]
    if num_key_configs <= 0:
        raise ValueError(f"num_key_configs must be positive, got {num_key_configs}")
    if num_key_configs > pool_size:
        raise ValueError(
            f"num_key_configs ({num_key_configs}) cannot exceed pool size ({pool_size})."
        )

    selected_indices = np.empty(num_key_configs, dtype=np.int64)
    min_distances = np.full(pool_size, np.inf, dtype=np.float32)

    first_index = select_first_index_from_median(normalized_pool)
    selected_indices[0] = first_index
    first_distances = np.sum(
        (normalized_pool - normalized_pool[first_index][None, :]) ** 2,
        axis=1,
    )
    min_distances = np.minimum(min_distances, first_distances)
    min_distances[first_index] = -1.0

    for sample_idx in range(1, num_key_configs):
        next_index = int(np.argmax(min_distances))
        selected_indices[sample_idx] = next_index
        current_distances = np.sum(
            (normalized_pool - normalized_pool[next_index][None, :]) ** 2,
            axis=1,
        )
        min_distances = np.minimum(min_distances, current_distances)
        min_distances[selected_indices[: sample_idx + 1]] = -1.0

    return selected_indices.astype(np.int64)


def build_output_manifest(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    raw_pool: np.ndarray,
    selected_indices: np.ndarray,
    input_manifest: dict,
) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "pool_size": int(raw_pool.shape[0]),
        "joint_dim": int(raw_pool.shape[1]),
        "num_key_configs": int(selected_indices.shape[0]),
        "first_index_strategy": "nearest_to_per_joint_median_vector",
        "distance_metric": "squared_l2_on_normalized_joint_pool",
        "selected_indices_shape": list(selected_indices.shape),
        "key_joint_configurations_shape": [int(selected_indices.shape[0]), int(raw_pool.shape[1])],
        "input_pool_manifest_created_at_utc": input_manifest.get("created_at_utc"),
        "input_pool_num_trajectories": input_manifest.get("num_trajectories"),
    }


def main() -> None:
    args = build_parser().parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    ensure_dir(output_dir)

    raw_pool, normalized_pool, input_manifest = load_required_arrays(input_dir=input_dir)
    selected_indices = farthest_point_sampling_joint_space(
        normalized_pool=normalized_pool,
        num_key_configs=int(args.num_key_configs),
    )

    key_raw = raw_pool[selected_indices].astype(np.float32)
    key_normalized = normalized_pool[selected_indices].astype(np.float32)

    np.save(output_dir / "key_joint_configurations_raw.npy", key_raw)
    np.save(output_dir / "key_joint_configurations_normalized.npy", key_normalized)
    np.save(output_dir / "key_joint_configuration_indices.npy", selected_indices.astype(np.int64))

    manifest = build_output_manifest(
        input_dir=input_dir,
        output_dir=output_dir,
        raw_pool=raw_pool,
        selected_indices=selected_indices,
        input_manifest=input_manifest,
    )
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"pool_size: {raw_pool.shape[0]}")
    print(f"num_key_configs: {selected_indices.shape[0]}")
    print(f"key_joint_configurations_raw: {key_raw.shape}")
    print(f"key_joint_configurations_normalized: {key_normalized.shape}")
    print(f"key_joint_configuration_indices: {selected_indices.shape}")
    print(f"saved_output_dir: {output_dir}")


if __name__ == "__main__":
    main()
