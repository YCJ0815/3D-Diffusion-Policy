#!/usr/bin/env python3
"""Export complete joint trajectories for selected validation episodes.

``validate_all_trajectories.py`` serializes NumPy arrays with ``default=str``.
Consequently, the 32x6 ``joint_trajectory`` strings in its metrics JSON contain
an ellipsis and cannot be replayed directly.  The selected candidate's 16x6
normalized B-spline control points are small enough to be stored in full.  This
script reconstructs the trajectory with the same spline evaluation and URDF
denormalization functions used by validation, then writes replayable files.

The reconstruction is numerically very close to the original validation
trajectory, but is not bit-exact because the JSON control-point strings contain
only NumPy's printed decimal precision.  For future bit-exact exports, save the
``selected_joint_trajectory`` array as NPY/NPZ inside the validation loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.bspline import (  # noqa: E402
    evaluate_quintic_bspline,
    unnormalize_joint_trajectory_with_urdf_limits,
)
from diffusion_policy_3d.common.input_data import (  # noqa: E402
    _default_urdf_path,
    _load_joint_limits_from_urdf,
)


DEFAULT_EPISODE_IDS = (26301, 26297, 26120, 26303)
DEFAULT_METRICS_PATH = (
    REPOSITORY_ROOT / "experiments" / "dp-dual" / "per_trajectory_metrics.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exported_trajectories"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct and export complete [T, 6] joint trajectories from "
            "validate_all_trajectories.py metrics JSON."
        )
    )
    parser.add_argument(
        "--metrics-json",
        type=pathlib.Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics JSON path (default: {DEFAULT_METRICS_PATH}).",
    )
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPISODE_IDS),
        help="Episode IDs to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--urdf-path",
        type=pathlib.Path,
        default=_default_urdf_path(),
        help="URDF used for joint names, limits, and control-point denormalization.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override output steps. By default each entry's total_steps is used.",
    )
    parser.add_argument(
        "--spline-degree",
        type=int,
        default=5,
        help="B-spline degree used by validation (default: 5).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "Optional seconds between waypoints. If omitted, normalized spline "
            "parameters are exported but physical timestamps are not invented."
        ),
    )
    return parser


def parse_numpy_array_string(value: Any, *, columns: int = 6) -> np.ndarray:
    if not isinstance(value, str):
        array = np.asarray(value, dtype=np.float32)
    else:
        if "..." in value:
            raise ValueError("array text is truncated with '...'")
        flattened = np.fromstring(
            value.replace("[", " ").replace("]", " "),
            sep=" ",
            dtype=np.float32,
        )
        if flattened.size == 0 or flattened.size % columns != 0:
            raise ValueError(
                f"could not parse array into rows of {columns}; values={flattened.size}"
            )
        array = flattened.reshape(-1, columns)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"expected [N, {columns}] array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("array contains non-finite values")
    return array.astype(np.float32)


def selected_candidate(entry: dict[str, Any]) -> dict[str, Any]:
    selected_index = int(
        entry.get("selected_candidate_index", entry.get("selected_candidate_idx", -1))
    )
    candidates = entry.get("guidance_candidates") or []
    for candidate in candidates:
        if int(candidate.get("candidate_index", -1)) == selected_index:
            return candidate
    raise KeyError(
        f"selected candidate {selected_index} is absent from guidance_candidates"
    )


def trajectory_quality(trajectory: np.ndarray) -> tuple[float, float]:
    differences = np.diff(trajectory, axis=0)
    path_length = float(np.linalg.norm(differences, axis=1).sum())
    if trajectory.shape[0] <= 2:
        return path_length, 0.0
    second_differences = np.diff(trajectory, n=2, axis=0)
    smoothness = float(np.mean(np.sum(second_differences**2, axis=1)))
    return path_length, smoothness


def write_csv(
    path: pathlib.Path,
    trajectory: np.ndarray,
    joint_names: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(joint_names)
        writer.writerows(trajectory.tolist())


def export_episode(
    entry: dict[str, Any],
    *,
    output_dir: pathlib.Path,
    joint_names: tuple[str, ...],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    urdf_path: pathlib.Path,
    num_steps_override: int | None,
    spline_degree: int,
    dt: float | None,
) -> dict[str, Any]:
    episode_idx = int(entry["episode_idx"])
    candidate = selected_candidate(entry)
    control_points = parse_numpy_array_string(
        candidate["control_points_normalized"], columns=len(joint_names)
    )
    num_steps = int(
        num_steps_override
        if num_steps_override is not None
        else round(float(entry.get("total_steps", 32)))
    )
    if num_steps <= 1:
        raise ValueError(f"episode {episode_idx}: num_steps must be > 1, got {num_steps}")

    normalized_trajectory = evaluate_quintic_bspline(
        control_points=control_points,
        num_steps=num_steps,
        degree=spline_degree,
    )
    joint_positions = unnormalize_joint_trajectory_with_urdf_limits(
        normalized_trajectory=normalized_trajectory,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    ).astype(np.float32)
    sample_parameters = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    timestamps_sec = (
        np.arange(num_steps, dtype=np.float64) * float(dt)
        if dt is not None
        else np.empty((0,), dtype=np.float64)
    )

    stem = f"episode_{episode_idx}_trajectory"
    npz_path = output_dir / f"{stem}.npz"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        joint_positions=joint_positions,
        trajectory=joint_positions,
        joint_names=np.asarray(joint_names),
        sample_parameters=sample_parameters,
        timestamps_sec=timestamps_sec,
        dt_sec=np.asarray(np.nan if dt is None else dt, dtype=np.float64),
        episode_idx=np.asarray(episode_idx, dtype=np.int64),
        validation_subset_index=np.asarray(
            int(entry.get("validation_subset_index", -1)), dtype=np.int64
        ),
        workpiece_id=np.asarray(int(entry.get("workpiece_id", -1)), dtype=np.int64),
        selected_candidate_index=np.asarray(
            int(candidate["candidate_index"]), dtype=np.int64
        ),
        normalized_control_points=control_points,
        spline_degree=np.asarray(spline_degree, dtype=np.int64),
        urdf_path=np.asarray(str(urdf_path)),
    )
    write_csv(csv_path, joint_positions, joint_names)

    json_payload: dict[str, Any] = {
        "episode_idx": episode_idx,
        "validation_subset_index": int(entry.get("validation_subset_index", -1)),
        "workpiece_id": int(entry.get("workpiece_id", -1)),
        "selected_candidate_index": int(candidate["candidate_index"]),
        "joint_names": list(joint_names),
        "joint_positions": joint_positions.tolist(),
        "sample_parameters": sample_parameters.tolist(),
        "dt_sec": dt,
        "timestamps_sec": None if dt is None else timestamps_sec.tolist(),
        "units": {"joint_positions": "rad", "timestamps": "s"},
        "reconstruction": {
            "source": "guidance_candidates.control_points_normalized",
            "num_control_points": int(control_points.shape[0]),
            "spline_degree": int(spline_degree),
            "knot_vector": "open_uniform",
            "num_steps": num_steps,
            "urdf_path": str(urdf_path),
            "precision_note": (
                "Reconstructed from decimal control points in metrics JSON; "
                "numerically close, not guaranteed bit-exact."
            ),
        },
    }
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reconstructed_path_length, reconstructed_smoothness = trajectory_quality(
        joint_positions
    )
    return {
        "episode_idx": episode_idx,
        "validation_subset_index": int(entry.get("validation_subset_index", -1)),
        "workpiece_id": int(entry.get("workpiece_id", -1)),
        "selected_candidate_index": int(candidate["candidate_index"]),
        "shape": list(joint_positions.shape),
        "dt_sec": dt,
        "source_joint_path_length_rad": float(entry["joint_path_length_rad"]),
        "reconstructed_joint_path_length_rad": reconstructed_path_length,
        "source_joint_smoothness": float(entry["joint_smoothness"]),
        "reconstructed_joint_smoothness": reconstructed_smoothness,
        "files": {
            "npz": npz_path.name,
            "csv": csv_path.name,
            "json": json_path.name,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    metrics_path = args.metrics_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    urdf_path = args.urdf_path.expanduser().resolve()
    if args.dt is not None and args.dt <= 0.0:
        raise ValueError(f"--dt must be positive, got {args.dt}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics JSON not found: {metrics_path}")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    entries = payload.get("per_trajectory")
    if not isinstance(entries, list):
        raise ValueError(f"{metrics_path} does not contain a per_trajectory list")
    by_episode = {int(entry["episode_idx"]): entry for entry in entries}
    missing = [episode for episode in args.episode_ids if episode not in by_episode]
    if missing:
        raise KeyError(f"episode IDs absent from metrics JSON: {missing}")

    joint_names, lower_limits, upper_limits = _load_joint_limits_from_urdf(
        str(urdf_path)
    )
    if len(joint_names) != 6:
        raise ValueError(
            f"expected six revolute joints in URDF, found {len(joint_names)}: {joint_names}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for episode_idx in args.episode_ids:
        result = export_episode(
            by_episode[episode_idx],
            output_dir=output_dir,
            joint_names=joint_names,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            urdf_path=urdf_path,
            num_steps_override=args.num_steps,
            spline_degree=args.spline_degree,
            dt=args.dt,
        )
        exported.append(result)
        print(
            f"episode {episode_idx}: exported {result['shape']} -> "
            f"{result['files']['npz']}"
        )

    manifest = {
        "source_metrics_json": str(metrics_path),
        "urdf_path": str(urdf_path),
        "joint_names": list(joint_names),
        "dt_sec": args.dt,
        "precision_note": (
            "Trajectories were reconstructed from normalized control points "
            "serialized as decimal text by validate_all_trajectories.py."
        ),
        "episodes": exported,
    }
    manifest_path = output_dir / "trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
