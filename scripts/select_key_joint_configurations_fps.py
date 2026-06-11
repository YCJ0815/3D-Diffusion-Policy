import argparse
import json
import math
import pathlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
JOB_DIR_PATTERN = re.compile(r"^job_\d{3}$")


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
    parser.add_argument(
        "--filter-by-source-sdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before FPS, keep only configurations whose TCP lies inside the SDF grid "
            "of the workpiece that generated the source trajectory."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional override for the regular trajectory results directory.",
    )
    parser.add_argument(
        "--simple-results-dir",
        type=str,
        default=None,
        help="Optional override for the simple trajectory results directory.",
    )
    parser.add_argument(
        "--jobs-sdf-root",
        type=str,
        default="data/raw_data/jobs",
        help="Directory containing regular job SDF folders.",
    )
    parser.add_argument(
        "--simple-sdf-root",
        type=str,
        default="data/raw_data/simple_jobs",
        help="Directory containing simple job SDF folders.",
    )
    parser.add_argument(
        "--sdf-filename",
        type=str,
        default="workpiece_sdf.npz",
        help="SDF filename inside each job folder.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="config/robot-model/ur5e_with_pen.urdf",
        help="Robot URDF used to compute TCP forward kinematics.",
    )
    parser.add_argument(
        "--tcp-link-name",
        type=str,
        default="tool0",
        help="TCP link used for source-SDF coverage filtering.",
    )
    parser.add_argument(
        "--tcp-sdf-margin-m",
        type=float,
        default=0.0,
        help="Optional inward margin applied to each SDF axis bound.",
    )
    parser.add_argument(
        "--missing-sdf-policy",
        choices=("error", "skip"),
        default="error",
        help="Whether a source trajectory with a missing SDF raises or is excluded.",
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


def _parse_vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected three values, got {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    rotation_y = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rotation_z = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError(f"Joint axis must be nonzero, got {axis}")
    x, y, z = axis / norm
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


class UrdfForwardKinematics:
    def __init__(self, urdf_path: pathlib.Path, tcp_link_name: str):
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
        root = ET.parse(urdf_path).getroot()
        parent_joint_by_child = {}
        actuated_joint_names = []
        for joint_element in root.findall("joint"):
            joint_name = joint_element.get("name")
            joint_type = joint_element.get("type")
            parent_element = joint_element.find("parent")
            child_element = joint_element.find("child")
            if (
                joint_name is None
                or joint_type is None
                or parent_element is None
                or child_element is None
            ):
                continue
            parent_link = parent_element.get("link")
            child_link = child_element.get("link")
            if parent_link is None or child_link is None:
                continue
            origin_element = joint_element.find("origin")
            origin_xyz = _parse_vector(
                None if origin_element is None else origin_element.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            origin_rpy = _parse_vector(
                None if origin_element is None else origin_element.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            axis_element = joint_element.find("axis")
            axis = _parse_vector(
                None if axis_element is None else axis_element.get("xyz"),
                (1.0, 0.0, 0.0),
            )
            joint = {
                "name": joint_name,
                "type": joint_type,
                "parent": parent_link,
                "child": child_link,
                "origin": _transform(_rpy_matrix(origin_rpy), origin_xyz),
                "axis": axis,
            }
            parent_joint_by_child[child_link] = joint
            if joint_type in ("revolute", "continuous"):
                actuated_joint_names.append(joint_name)

        chain = []
        current_link = tcp_link_name
        while current_link in parent_joint_by_child:
            joint = parent_joint_by_child[current_link]
            chain.append(joint)
            current_link = str(joint["parent"])
        chain.reverse()
        if not chain or str(chain[-1]["child"]) != tcp_link_name:
            raise KeyError(f"TCP link {tcp_link_name!r} is not reachable in {urdf_path}")
        if len(actuated_joint_names) != 6:
            raise ValueError(
                f"Expected 6 actuated robot joints in {urdf_path}, got {actuated_joint_names}"
            )
        self.chain = chain
        self.actuated_joint_names = actuated_joint_names

    def tcp_positions(self, joint_configurations: np.ndarray) -> np.ndarray:
        joint_configurations = np.asarray(joint_configurations, dtype=np.float64)
        if joint_configurations.ndim != 2 or joint_configurations.shape[1] != 6:
            raise ValueError(
                f"joint_configurations must have shape [N, 6], got {joint_configurations.shape}"
            )
        joint_name_to_column = {
            name: column for column, name in enumerate(self.actuated_joint_names)
        }
        positions = np.empty((joint_configurations.shape[0], 3), dtype=np.float32)
        for row_index, joint_values in enumerate(joint_configurations):
            pose = np.eye(4, dtype=np.float64)
            for joint in self.chain:
                pose = pose @ joint["origin"]
                joint_type = str(joint["type"])
                if joint_type in ("revolute", "continuous"):
                    angle = float(joint_values[joint_name_to_column[str(joint["name"])]])
                    pose = pose @ _transform(
                        _axis_angle_matrix(np.asarray(joint["axis"]), angle),
                        np.zeros(3, dtype=np.float64),
                    )
                elif joint_type == "prismatic":
                    raise ValueError("Prismatic joints are not supported by this 6-DoF FK helper.")
            positions[row_index] = pose[:3, 3].astype(np.float32)
        return positions


def load_trajectory_paths(
    input_dir: pathlib.Path,
    input_manifest: dict,
    num_trajectories: int,
    results_dir_override: str | None,
    simple_results_dir_override: str | None,
) -> tuple[list[pathlib.Path], pathlib.Path, pathlib.Path]:
    archive_path = input_dir / "trajectories_before_extraction.npz"
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Source-SDF filtering requires {archive_path.name} in {input_dir}"
        )
    with np.load(archive_path) as archive:
        if "paths" not in archive.files:
            raise KeyError(f"{archive_path} does not contain `paths`.")
        serialized_paths = [str(value) for value in archive["paths"].tolist()]
    if len(serialized_paths) != num_trajectories:
        raise ValueError(
            f"Expected {num_trajectories} source paths, got {len(serialized_paths)}"
        )

    results_dir = pathlib.Path(
        results_dir_override or input_manifest.get("results_dir", "data/raw_data/results")
    ).expanduser().resolve()
    simple_results_dir = pathlib.Path(
        simple_results_dir_override
        or input_manifest.get("simple_results_dir", "data/raw_data/simple_results")
    ).expanduser().resolve()
    scan_roots = [results_dir, simple_results_dir]
    scanned_paths = sorted(
        {
            path.resolve()
            for root in scan_roots
            if root.exists()
            for path in root.rglob("transition_*.npz")
        }
    )
    if len(scanned_paths) == num_trajectories:
        reconstructed_serialized_paths = []
        for path in scanned_paths:
            for root in scan_roots:
                try:
                    reconstructed_serialized_paths.append(str(path.relative_to(root)))
                    break
                except ValueError:
                    continue
        if reconstructed_serialized_paths == serialized_paths:
            return scanned_paths, results_dir, simple_results_dir

    resolved_paths = []
    for serialized_path in serialized_paths:
        candidate = pathlib.Path(serialized_path).expanduser()
        if candidate.is_absolute() and candidate.exists():
            resolved_paths.append(candidate.resolve())
            continue
        matches = [
            (root / candidate).resolve()
            for root in scan_roots
            if (root / candidate).exists()
        ]
        if len(matches) == 0:
            raise ValueError(
                f"Could not resolve source trajectory {serialized_path!r} in any scan root."
            )
        if len(matches) > 1:
            # When path exists in both results_dir and simple_results_dir,
            # prefer results_dir (first scan root).
            matches = [matches[0]]
        resolved_paths.append(matches[0])
    return resolved_paths, results_dir, simple_results_dir


def _is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def build_source_sdf_candidate_mask(
    raw_pool: np.ndarray,
    input_dir: pathlib.Path,
    input_manifest: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    num_inner_samples = int(input_manifest.get("num_inner_samples", 8))
    if raw_pool.shape[0] % num_inner_samples != 0:
        raise ValueError(
            f"Pool size {raw_pool.shape[0]} is not divisible by num_inner_samples={num_inner_samples}"
        )
    num_trajectories = raw_pool.shape[0] // num_inner_samples
    source_paths, results_dir, simple_results_dir = load_trajectory_paths(
        input_dir=input_dir,
        input_manifest=input_manifest,
        num_trajectories=num_trajectories,
        results_dir_override=args.results_dir,
        simple_results_dir_override=args.simple_results_dir,
    )
    fk = UrdfForwardKinematics(
        pathlib.Path(args.urdf_path).expanduser().resolve(),
        tcp_link_name=str(args.tcp_link_name),
    )
    tcp_positions = fk.tcp_positions(raw_pool)
    jobs_sdf_root = pathlib.Path(args.jobs_sdf_root).expanduser().resolve()
    simple_sdf_root = pathlib.Path(args.simple_sdf_root).expanduser().resolve()
    margin = float(args.tcp_sdf_margin_m)
    if margin < 0:
        raise ValueError(f"tcp_sdf_margin_m must be non-negative, got {margin}")

    candidate_mask = np.zeros(raw_pool.shape[0], dtype=bool)
    missing_sdf_trajectories = 0
    sdf_bounds_cache: dict[pathlib.Path, tuple[np.ndarray, np.ndarray] | None] = {}
    source_type_counts = {"regular": 0, "simple": 0}
    accepted_type_counts = {"regular": 0, "simple": 0}

    for trajectory_index, source_path in enumerate(source_paths):
        if not JOB_DIR_PATTERN.fullmatch(source_path.parent.name):
            raise ValueError(f"Unexpected source job directory: {source_path.parent.name}")
        if _is_relative_to(source_path, results_dir):
            source_type = "regular"
            sdf_root = jobs_sdf_root
        elif _is_relative_to(source_path, simple_results_dir):
            source_type = "simple"
            sdf_root = simple_sdf_root
        else:
            raise ValueError(
                f"Source trajectory {source_path} is outside results roots "
                f"{results_dir} and {simple_results_dir}"
            )
        source_type_counts[source_type] += num_inner_samples
        sdf_path = sdf_root / source_path.parent.name / str(args.sdf_filename)
        if sdf_path not in sdf_bounds_cache:
            if not sdf_path.exists():
                if args.missing_sdf_policy == "error":
                    raise FileNotFoundError(f"Missing source workpiece SDF: {sdf_path}")
                sdf_bounds_cache[sdf_path] = None
            else:
                with np.load(sdf_path) as sdf_data:
                    missing_keys = [
                        key for key in ("x", "y", "z") if key not in sdf_data.files
                    ]
                    if missing_keys:
                        raise KeyError(f"{sdf_path} is missing SDF axes: {missing_keys}")
                    lower = np.asarray(
                        [sdf_data["x"][0], sdf_data["y"][0], sdf_data["z"][0]],
                        dtype=np.float32,
                    )
                    upper = np.asarray(
                        [sdf_data["x"][-1], sdf_data["y"][-1], sdf_data["z"][-1]],
                        dtype=np.float32,
                    )
                    sdf_bounds_cache[sdf_path] = (lower, upper)
        bounds = sdf_bounds_cache[sdf_path]
        start = trajectory_index * num_inner_samples
        stop = start + num_inner_samples
        if bounds is None:
            missing_sdf_trajectories += 1
            continue
        lower, upper = bounds
        lower = lower + margin
        upper = upper - margin
        if np.any(lower > upper):
            raise ValueError(
                f"tcp_sdf_margin_m={margin} collapses SDF bounds for {sdf_path}"
            )
        trajectory_mask = np.all(
            (tcp_positions[start:stop] >= lower[None, :])
            & (tcp_positions[start:stop] <= upper[None, :]),
            axis=1,
        )
        candidate_mask[start:stop] = trajectory_mask
        accepted_type_counts[source_type] += int(np.count_nonzero(trajectory_mask))

    diagnostics = {
        "filter_strategy": "tcp_inside_source_workpiece_sdf_bounds",
        "tcp_link_name": str(args.tcp_link_name),
        "tcp_sdf_margin_m": margin,
        "urdf_path": str(pathlib.Path(args.urdf_path).expanduser().resolve()),
        "candidate_count": int(np.count_nonzero(candidate_mask)),
        "rejected_count": int(candidate_mask.size - np.count_nonzero(candidate_mask)),
        "candidate_ratio": float(np.mean(candidate_mask)),
        "missing_sdf_trajectories": int(missing_sdf_trajectories),
        "source_type_counts": source_type_counts,
        "accepted_type_counts": accepted_type_counts,
    }
    return candidate_mask, tcp_positions, diagnostics


def select_first_index_from_median(normalized_pool: np.ndarray) -> int:
    median_vector = np.median(normalized_pool, axis=0).astype(np.float32)
    squared_distances = np.sum((normalized_pool - median_vector[None, :]) ** 2, axis=1)
    return int(np.argmin(squared_distances))


def farthest_point_sampling_joint_space(
    normalized_pool: np.ndarray,
    num_key_configs: int,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    full_pool_size = normalized_pool.shape[0]
    if candidate_indices is None:
        candidate_indices = np.arange(full_pool_size, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if candidate_indices.size == 0:
        raise ValueError("FPS candidate pool is empty.")
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= full_pool_size):
        raise ValueError("candidate_indices contains indices outside the joint pool.")
    if np.unique(candidate_indices).shape[0] != candidate_indices.shape[0]:
        raise ValueError("candidate_indices must be unique.")
    candidate_pool = normalized_pool[candidate_indices]
    pool_size = candidate_pool.shape[0]
    if num_key_configs <= 0:
        raise ValueError(f"num_key_configs must be positive, got {num_key_configs}")
    if num_key_configs > pool_size:
        raise ValueError(
            f"num_key_configs ({num_key_configs}) cannot exceed FPS candidate size ({pool_size})."
        )

    selected_candidate_indices = np.empty(num_key_configs, dtype=np.int64)
    min_distances = np.full(pool_size, np.inf, dtype=np.float32)

    first_index = select_first_index_from_median(candidate_pool)
    selected_candidate_indices[0] = first_index
    first_distances = np.sum(
        (candidate_pool - candidate_pool[first_index][None, :]) ** 2,
        axis=1,
    )
    min_distances = np.minimum(min_distances, first_distances)
    min_distances[first_index] = -1.0

    for sample_idx in range(1, num_key_configs):
        next_index = int(np.argmax(min_distances))
        selected_candidate_indices[sample_idx] = next_index
        current_distances = np.sum(
            (candidate_pool - candidate_pool[next_index][None, :]) ** 2,
            axis=1,
        )
        min_distances = np.minimum(min_distances, current_distances)
        min_distances[selected_candidate_indices[: sample_idx + 1]] = -1.0

    return candidate_indices[selected_candidate_indices].astype(np.int64)


def build_output_manifest(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    raw_pool: np.ndarray,
    selected_indices: np.ndarray,
    input_manifest: dict,
    filter_diagnostics: dict,
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
        "candidate_filter": filter_diagnostics,
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
    candidate_mask = np.ones(raw_pool.shape[0], dtype=bool)
    tcp_positions = None
    filter_diagnostics = {
        "filter_strategy": "none",
        "candidate_count": int(raw_pool.shape[0]),
        "rejected_count": 0,
        "candidate_ratio": 1.0,
    }
    if args.filter_by_source_sdf:
        candidate_mask, tcp_positions, filter_diagnostics = build_source_sdf_candidate_mask(
            raw_pool=raw_pool,
            input_dir=input_dir,
            input_manifest=input_manifest,
            args=args,
        )
    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
    selected_indices = farthest_point_sampling_joint_space(
        normalized_pool=normalized_pool,
        num_key_configs=int(args.num_key_configs),
        candidate_indices=candidate_indices,
    )

    key_raw = raw_pool[selected_indices].astype(np.float32)
    key_normalized = normalized_pool[selected_indices].astype(np.float32)

    np.save(output_dir / "key_joint_configurations_raw.npy", key_raw)
    np.save(output_dir / "key_joint_configurations_normalized.npy", key_normalized)
    np.save(output_dir / "key_joint_configuration_indices.npy", selected_indices.astype(np.int64))
    np.save(output_dir / "fps_candidate_indices.npy", candidate_indices)
    if tcp_positions is not None:
        np.save(
            output_dir / "key_tcp_positions.npy",
            tcp_positions[selected_indices].astype(np.float32),
        )

    manifest = build_output_manifest(
        input_dir=input_dir,
        output_dir=output_dir,
        raw_pool=raw_pool,
        selected_indices=selected_indices,
        input_manifest=input_manifest,
        filter_diagnostics=filter_diagnostics,
    )
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"pool_size: {raw_pool.shape[0]}")
    print(f"fps_candidate_count: {candidate_indices.shape[0]}")
    print(f"fps_rejected_count: {raw_pool.shape[0] - candidate_indices.shape[0]}")
    print(f"num_key_configs: {selected_indices.shape[0]}")
    print(f"key_joint_configurations_raw: {key_raw.shape}")
    print(f"key_joint_configurations_normalized: {key_normalized.shape}")
    print(f"key_joint_configuration_indices: {selected_indices.shape}")
    print(f"saved_output_dir: {output_dir}")


if __name__ == "__main__":
    main()
