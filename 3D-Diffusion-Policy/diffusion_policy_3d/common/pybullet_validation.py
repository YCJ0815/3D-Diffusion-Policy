from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from diffusion_policy_3d.common.bspline import (
    FIXED_CONTROL_POINTS_PER_SIDE,
    load_delta_w_stats,
    reconstruct_trajectory_from_normalized_free_residual,
    unnormalize_joint_trajectory_with_urdf_limits,
)
from diffusion_policy_3d.common.increment import load_increment_stats
from diffusion_policy_3d.common.input_data import (
    _default_urdf_path,
    _load_joint_limits_from_urdf,
)


def _episode_bounds(episode_ends: np.ndarray, episode_idx: int) -> tuple[int, int]:
    end_idx = int(episode_ends[episode_idx])
    start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    return start_idx, end_idx


def _resolve_stats_mode(stats_path: str) -> str:
    stats = np.load(stats_path)
    if "basis_matrix" in stats.files or "knot_vector" in stats.files:
        return "bspline"
    return "increment"


def _sanitize_metric_suffix(name: str) -> str:
    safe = []
    for char in str(name):
        if char.isalnum():
            safe.append(char.lower())
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _resolve_workpiece_stl_path(
    workpiece_id: int,
    jobs_root: str,
    simple_jobs_root: str | None,
    simple_workpiece_id_offset: int,
    job_name_template: str,
    workpiece_filename: str,
) -> Path:
    return _resolve_workpiece_file_path(
        workpiece_id=workpiece_id,
        jobs_root=jobs_root,
        simple_jobs_root=simple_jobs_root,
        simple_workpiece_id_offset=simple_workpiece_id_offset,
        job_name_template=job_name_template,
        filename=workpiece_filename,
        file_label="STL",
    )


def _resolve_workpiece_file_path(
    workpiece_id: int,
    jobs_root: str,
    simple_jobs_root: str | None,
    simple_workpiece_id_offset: int,
    job_name_template: str,
    filename: str,
    file_label: str,
) -> Path:
    workpiece_id = int(workpiece_id)
    resolved_jobs_root = Path(jobs_root).expanduser().resolve()
    local_workpiece_id = workpiece_id
    if simple_jobs_root is not None and workpiece_id >= int(simple_workpiece_id_offset):
        resolved_jobs_root = Path(simple_jobs_root).expanduser().resolve()
        local_workpiece_id = workpiece_id - int(simple_workpiece_id_offset)
    job_name = job_name_template.format(workpiece_id=int(local_workpiece_id))
    file_path = resolved_jobs_root / job_name / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Workpiece {file_label} not found for workpiece_id={workpiece_id}: {file_path}")
    return file_path


def _resolve_mesh_filename(filename: str, package_roots: list[Path]) -> str:
    if not filename.startswith("package://"):
        return filename

    package_rel = filename[len("package://"):]
    package_name, _, rel_path = package_rel.partition("/")
    for root in package_roots:
        candidate = root / package_name / rel_path
        if candidate.is_file():
            return str(candidate)
        candidate = root / rel_path
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Unable to resolve URDF mesh URI `{filename}` with package roots: "
        f"{[str(root) for root in package_roots]}"
    )


def _expand_urdf_package_roots(urdf_path: str, package_roots: list[str]) -> list[Path]:
    urdf_parent = Path(urdf_path).expanduser().resolve().parent
    candidates: list[Path] = []
    for root in package_roots:
        root_path = Path(root).expanduser()
        if root_path.is_absolute():
            candidates.append(root_path.resolve())
        else:
            candidates.extend([
                root_path.resolve(),
                (urdf_parent / root_path).resolve(),
                (urdf_parent.parent / root_path).resolve(),
            ])
    candidates.extend([
        urdf_parent.resolve(),
        (urdf_parent / "robot-model").resolve(),
        urdf_parent.parent.resolve(),
    ])

    unique_roots = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_roots.append(candidate)
    return unique_roots


def _rewrite_urdf_package_uris(urdf_path: str, package_roots: list[str]) -> str:
    roots = _expand_urdf_package_roots(urdf_path, package_roots)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None or not filename.startswith("package://"):
            continue
        mesh.set("filename", _resolve_mesh_filename(filename, roots))
        changed = True

    if not changed:
        return urdf_path

    tmp_dir = Path(tempfile.mkdtemp(prefix="dp3_pybullet_urdf_"))
    resolved_path = tmp_dir / Path(urdf_path).name
    tree.write(resolved_path, encoding="utf-8", xml_declaration=True)
    return str(resolved_path)


def _parse_float_vector(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    parsed = [float(item) for item in value.split()]
    return np.asarray(parsed, dtype=np.float32)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(item) for item in rpy.reshape(3)]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _apply_origin_transform(points: np.ndarray, origin_elem) -> np.ndarray:
    if origin_elem is None:
        return points.astype(np.float32)
    xyz = _parse_float_vector(origin_elem.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_float_vector(origin_elem.get("rpy"), (0.0, 0.0, 0.0))
    rotation = _rpy_to_matrix(rpy)
    return (points @ rotation.T + xyz.reshape(1, 3)).astype(np.float32)


def _select_deterministic_surface_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices].astype(np.float32)


def _select_lowest_candidate_score_index(score_keys: np.ndarray) -> int:
    score_keys = np.asarray(score_keys, dtype=np.float32)
    if score_keys.ndim == 1:
        score_keys = score_keys.reshape(-1, 1)
    valid_mask = np.all(np.isfinite(score_keys), axis=1)
    if not np.any(valid_mask):
        raise ValueError("All PyBullet validation candidates have invalid SDF scores.")
    ranked_score_keys = np.where(valid_mask[:, None], score_keys, np.inf)
    sort_keys = tuple(ranked_score_keys[:, col] for col in range(ranked_score_keys.shape[1] - 1, -1, -1))
    return int(np.lexsort(sort_keys)[0])


def _add_count(counts: dict[str, float], key: str, value: float = 1.0) -> None:
    counts[str(key)] = counts.get(str(key), 0.0) + float(value)


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float32)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size > 0 else float("nan")


def _finite_percentile(values: list[float], percentile: float) -> float:
    array = np.asarray(values, dtype=np.float32)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size > 0 else float("nan")


def _score_safety_sdf_candidate(
    sdf_values: np.ndarray,
    topk: int,
    d_select: float,
    sdf_values_by_link: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    sdf_values = np.asarray(sdf_values, dtype=np.float32)
    flat_sdf_values = sdf_values.reshape(-1)
    valid_sdf_values = flat_sdf_values[np.isfinite(flat_sdf_values)]
    valid_ratio = (
        float(valid_sdf_values.size / flat_sdf_values.size)
        if flat_sdf_values.size > 0
        else 0.0
    )
    sdf_finite_ratio_by_link: dict[str, float] = {}
    pen_point_count_by_link: dict[str, float] = {}
    penetrating_link_names: list[str] = []
    if sdf_values_by_link is not None:
        for link_name, link_sdf_values in sdf_values_by_link.items():
            link_sdf_values = np.asarray(link_sdf_values, dtype=np.float32).reshape(-1)
            if link_sdf_values.size == 0:
                sdf_finite_ratio_by_link[str(link_name)] = 0.0
            else:
                sdf_finite_ratio_by_link[str(link_name)] = float(
                    np.isfinite(link_sdf_values).sum() / link_sdf_values.size
                )
            link_pen_mask = (link_sdf_values < 0.0) & np.isfinite(link_sdf_values)
            link_pen_count = float(link_pen_mask.sum())
            pen_point_count_by_link[str(link_name)] = link_pen_count
            if link_pen_count > 0.0:
                penetrating_link_names.append(str(link_name))
    if valid_sdf_values.size == 0:
        return {
            "has_pen": float("inf"),
            "pen_step_count": float("inf"),
            "pen_point_count": float("inf"),
            "num_pen": float("inf"),
            "neg_min_sdf": float("inf"),
            "neg_worstk_mean": float("inf"),
            "margin_violation": float("inf"),
            "min_sdf_distance_m": float("nan"),
            "sdf_finite_ratio": valid_ratio,
            "sdf_finite_ratio_by_link": sdf_finite_ratio_by_link,
            "pen_point_count_by_link": pen_point_count_by_link,
            "penetrating_link_names": penetrating_link_names,
        }
    if topk <= 0:
        raise ValueError(f"selection_topk must be positive, got {topk}")
    if d_select <= 0:
        raise ValueError(f"selection_d_safe must be positive, got {d_select}")

    worst_k = min(int(topk), int(valid_sdf_values.size))
    worst_distances = np.partition(valid_sdf_values, worst_k - 1)[:worst_k]
    valid_mask = np.isfinite(sdf_values)
    pen_mask = (sdf_values < 0.0) & valid_mask
    has_pen = float(np.any(pen_mask))
    pen_step_count = float(np.any(pen_mask, axis=1).sum()) if pen_mask.ndim >= 2 else float(np.any(pen_mask))
    pen_point_count = float(pen_mask.sum())
    min_sdf = float(np.min(valid_sdf_values))
    worstk_mean = float(np.mean(worst_distances))
    margin_violation = float(np.mean(np.maximum(0.0, float(d_select) - worst_distances)))
    if pen_point_count > 0.0 and min_sdf >= 0.0:
        print(
            "[PyBullet validation] WARNING: inconsistent sdf stats: "
            f"pen_point_count={pen_point_count}, min_sdf={min_sdf:.6f}"
        )
    return {
        "has_pen": has_pen,
        "pen_step_count": pen_step_count,
        "pen_point_count": pen_point_count,
        "num_pen": pen_point_count,
        "neg_min_sdf": -min_sdf,
        "neg_worstk_mean": -worstk_mean,
        "margin_violation": margin_violation,
        "min_sdf_distance_m": min_sdf,
        "sdf_finite_ratio": valid_ratio,
        "sdf_finite_ratio_by_link": sdf_finite_ratio_by_link,
        "pen_point_count_by_link": pen_point_count_by_link,
        "penetrating_link_names": penetrating_link_names,
    }


@dataclass
class SDFGrid:
    sdf: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    out_of_bounds_value_m: float | None = None

    @classmethod
    def load(cls, path: Path, out_of_bounds_value_m: float | None = None) -> "SDFGrid":
        data = np.load(path)
        missing = [key for key in ("sdf", "x", "y", "z") if key not in data.files]
        if missing:
            raise KeyError(f"SDF file {path} is missing required keys: {missing}")
        return cls(
            sdf=np.asarray(data["sdf"], dtype=np.float32),
            x=np.asarray(data["x"], dtype=np.float32),
            y=np.asarray(data["y"], dtype=np.float32),
            z=np.asarray(data["z"], dtype=np.float32),
            out_of_bounds_value_m=out_of_bounds_value_m,
        )

    def query(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if self.sdf.shape != (self.x.shape[0], self.y.shape[0], self.z.shape[0]):
            raise ValueError(
                "SDF grid shape must match x/y/z axes, got "
                f"sdf={self.sdf.shape}, axes={(self.x.shape[0], self.y.shape[0], self.z.shape[0])}"
            )
        valid = (
            (points[:, 0] >= self.x[0]) & (points[:, 0] <= self.x[-1]) &
            (points[:, 1] >= self.y[0]) & (points[:, 1] <= self.y[-1]) &
            (points[:, 2] >= self.z[0]) & (points[:, 2] <= self.z[-1])
        )
        values = np.full(points.shape[0], np.nan, dtype=np.float32)
        if self.out_of_bounds_value_m is not None:
            values[~valid] = float(self.out_of_bounds_value_m)
        if not np.any(valid):
            return values

        valid_points = points[valid]
        ix0, ix1, tx = self._axis_indices(self.x, valid_points[:, 0])
        iy0, iy1, ty = self._axis_indices(self.y, valid_points[:, 1])
        iz0, iz1, tz = self._axis_indices(self.z, valid_points[:, 2])

        c000 = self.sdf[ix0, iy0, iz0]
        c100 = self.sdf[ix1, iy0, iz0]
        c010 = self.sdf[ix0, iy1, iz0]
        c110 = self.sdf[ix1, iy1, iz0]
        c001 = self.sdf[ix0, iy0, iz1]
        c101 = self.sdf[ix1, iy0, iz1]
        c011 = self.sdf[ix0, iy1, iz1]
        c111 = self.sdf[ix1, iy1, iz1]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        values[valid] = (c0 * (1.0 - tz) + c1 * tz).astype(np.float32)
        return values

    @staticmethod
    def _axis_indices(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        upper = np.searchsorted(axis, values, side="right")
        upper = np.clip(upper, 1, axis.shape[0] - 1)
        lower = upper - 1
        span = axis[upper] - axis[lower]
        weights = np.divide(
            values - axis[lower],
            span,
            out=np.zeros_like(values, dtype=np.float32),
            where=span > 0,
        )
        return lower.astype(np.int64), upper.astype(np.int64), weights.astype(np.float32)


@dataclass
class PyBulletValidationConfig:
    enabled: bool = False
    stats_path: str | None = None
    stats_mode: str = "auto"
    include_regular_jobs: bool = True
    include_simple_jobs: bool = True
    jobs_root: str = "data/raw_data/jobs"
    simple_jobs_root: str | None = "data/raw_data/simple_jobs"
    simple_workpiece_id_offset: int = 1000
    job_name_template: str = "job_{workpiece_id:03d}"
    workpiece_filename: str = "workpiece.stl"
    urdf_path: str | None = None
    urdf_package_roots: tuple[str, ...] = ("config/robot-model",)
    tcp_link_name: str = "tool0"
    stl_x_offset_m: float = 0.5
    collision_distance_threshold: float = 0.0
    interpolate_for_collision: bool = False
    max_joint_step_rad: float = 0.01
    min_interpolated_steps_per_segment: int = 1
    goal_position_norm_m: float = 0.1
    goal_tolerance_m: float = 0.01
    num_control_points: int = 12
    spline_degree: int = 5
    target_steps: int = 32
    max_episodes: int | None = None
    random_sample_episodes: bool = False
    random_seed: int = 42
    diffusion_sampling_seed: int | None = None
    inference_num_steps: int | None = None
    num_candidates: int = 16
    candidate_scheduler_eta: float | None = 1.0
    candidate_action_noise_std: float = 0.0
    candidate_action_noise_clip: float | None = None
    candidate_selection: str = "weighted_sdf"
    selection_topk: int = 128
    selection_d_safe: float = 0.005
    selection_d_pen: float = 0.005
    selection_margin_weight: float = 1.0
    selection_penetration_weight: float = 2.0
    selection_smooth_weight: float = 0.01
    selection_length_weight: float = 0.005
    sdf_filename: str = "workpiece_sdf.npz"
    sdf_required: bool = True
    robot_surface_points_per_link: int = 256
    sdf_out_of_bounds_value_m: float | None = None
    log_legacy_pybullet_metrics: bool = True
    collision_log_path: str | None = None
    progress_mininterval_sec: float = 1.0
    num_workers: int = 1
    inference_batch_size: int = 32
    worker_start_method: str = "spawn"
    worker_chunksize: int = 1

    @classmethod
    def from_omegaconf(cls, cfg) -> "PyBulletValidationConfig":
        sdf_out_of_bounds_value_m = cfg.get("sdf_out_of_bounds_value_m", None)
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            stats_path=cfg.get("stats_path"),
            stats_mode=str(cfg.get("stats_mode", "auto")),
            include_regular_jobs=bool(cfg.get("include_regular_jobs", True)),
            include_simple_jobs=bool(cfg.get("include_simple_jobs", True)),
            jobs_root=str(cfg.get("jobs_root", "data/raw_data/jobs")),
            simple_jobs_root=cfg.get("simple_jobs_root", "data/raw_data/simple_jobs"),
            simple_workpiece_id_offset=int(cfg.get("simple_workpiece_id_offset", 1000)),
            job_name_template=str(cfg.get("job_name_template", "job_{workpiece_id:03d}")),
            workpiece_filename=str(cfg.get("workpiece_filename", "workpiece.stl")),
            urdf_path=cfg.get("urdf_path"),
            urdf_package_roots=tuple(cfg.get("urdf_package_roots", ["config/robot-model"])),
            tcp_link_name=str(cfg.get("tcp_link_name", "tool0")),
            stl_x_offset_m=float(cfg.get("stl_x_offset_m", 0.5)),
            collision_distance_threshold=float(cfg.get("collision_distance_threshold", 0.0)),
            interpolate_for_collision=bool(cfg.get("interpolate_for_collision", False)),
            max_joint_step_rad=float(cfg.get("max_joint_step_rad", 0.01)),
            min_interpolated_steps_per_segment=int(cfg.get("min_interpolated_steps_per_segment", 1)),
            goal_position_norm_m=float(cfg.get("goal_position_norm_m", 0.1)),
            goal_tolerance_m=float(cfg.get("goal_tolerance_m", 0.01)),
            num_control_points=int(cfg.get("num_control_points", 12)),
            spline_degree=int(cfg.get("spline_degree", 5)),
            target_steps=int(cfg.get("target_steps", 32)),
            max_episodes=cfg.get("max_episodes"),
            random_sample_episodes=bool(cfg.get("random_sample_episodes", False)),
            random_seed=int(cfg.get("random_seed", 42)),
            diffusion_sampling_seed=(
                None
                if cfg.get("diffusion_sampling_seed", None) is None
                else int(cfg.get("diffusion_sampling_seed"))
            ),
            inference_num_steps=(
                None
                if cfg.get("inference_num_steps", None) is None
                else int(cfg.get("inference_num_steps"))
            ),
            num_candidates=int(cfg.get("num_candidates", 16)),
            candidate_scheduler_eta=(
                None
                if cfg.get("candidate_scheduler_eta", 1.0) is None
                else float(cfg.get("candidate_scheduler_eta", 1.0))
            ),
            candidate_action_noise_std=float(cfg.get("candidate_action_noise_std", 0.0)),
            candidate_action_noise_clip=(
                None
                if cfg.get("candidate_action_noise_clip", None) is None
                else float(cfg.get("candidate_action_noise_clip"))
            ),
            candidate_selection=str(cfg.get("candidate_selection", "weighted_sdf")),
            selection_topk=int(cfg.get("selection_topk", 128)),
            selection_d_safe=float(cfg.get("selection_d_safe", 0.005)),
            selection_d_pen=float(cfg.get("selection_d_pen", 0.005)),
            selection_margin_weight=float(
                cfg.get("selection_margin_weight", 1.0)
            ),
            selection_penetration_weight=float(
                cfg.get("selection_penetration_weight", 2.0)
            ),
            selection_smooth_weight=float(
                cfg.get("selection_smooth_weight", 0.01)
            ),
            selection_length_weight=float(
                cfg.get("selection_length_weight", 0.005)
            ),
            sdf_filename=str(cfg.get("sdf_filename", "workpiece_sdf.npz")),
            sdf_required=bool(cfg.get("sdf_required", True)),
            robot_surface_points_per_link=int(cfg.get("robot_surface_points_per_link", 256)),
            sdf_out_of_bounds_value_m=(
                None if sdf_out_of_bounds_value_m is None else float(sdf_out_of_bounds_value_m)
            ),
            log_legacy_pybullet_metrics=bool(cfg.get("log_legacy_pybullet_metrics", True)),
            collision_log_path=cfg.get("collision_log_path", None),
            progress_mininterval_sec=float(cfg.get("progress_mininterval_sec", 1.0)),
            num_workers=int(cfg.get("num_workers", 1)),
            inference_batch_size=int(cfg.get("inference_batch_size", 32)),
            worker_start_method=str(cfg.get("worker_start_method", "spawn")),
            worker_chunksize=int(cfg.get("worker_chunksize", 1)),
        )


_PYBULLET_VALIDATION_WORKER: PyBulletCollisionValidator | None = None


def _init_pybullet_validation_worker(cfg: PyBulletValidationConfig) -> None:
    global _PYBULLET_VALIDATION_WORKER
    _PYBULLET_VALIDATION_WORKER = PyBulletCollisionValidator(cfg)


def _run_pybullet_validation_task(task: dict[str, object]) -> dict[str, object]:
    global _PYBULLET_VALIDATION_WORKER
    if _PYBULLET_VALIDATION_WORKER is None:
        raise RuntimeError("PyBullet validation worker was not initialized.")
    validator = _PYBULLET_VALIDATION_WORKER
    start_joint_normalized = np.asarray(task["start_joint_normalized"], dtype=np.float32)
    start_joint_state = validator._unnormalize_joint_state(start_joint_normalized)
    return validator.evaluate_trajectory(
        workpiece_id=int(task["workpiece_id"]),
        joint_trajectory=np.asarray(task["joint_trajectory"], dtype=np.float32),
        start_joint_state=start_joint_state,
        goal_position_normalized=np.asarray(task["goal_position_normalized"], dtype=np.float32),
        episode_idx=int(task["episode_idx"]) if "episode_idx" in task else None,
    )


class PyBulletCollisionValidator:
    def __init__(self, cfg: PyBulletValidationConfig):
        self.cfg = cfg
        if not self.cfg.enabled:
            raise ValueError("PyBulletCollisionValidator should only be created when enabled=True.")
        if not self.cfg.stats_path:
            raise ValueError("training.pybullet_eval.stats_path is required when pybullet validation is enabled.")

        try:
            import pybullet as pb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pybullet validation was enabled, but `pybullet` is not installed in the current environment."
            ) from exc

        self.pb = pb
        self.client_id = pb.connect(pb.DIRECT)
        resolved_urdf_path = self.cfg.urdf_path if self.cfg.urdf_path is not None else str(_default_urdf_path())
        resolved_urdf_path = _rewrite_urdf_package_uris(
            urdf_path=resolved_urdf_path,
            package_roots=list(self.cfg.urdf_package_roots),
        )
        self.resolved_urdf_path = resolved_urdf_path
        self.robot_id = pb.loadURDF(
            resolved_urdf_path,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.joint_names, self.joint_lower_limits, self.joint_upper_limits = _load_joint_limits_from_urdf(
            self.cfg.urdf_path if self.cfg.urdf_path is not None else str(_default_urdf_path())
        )
        self.revolute_joint_indices = []
        self.link_name_to_index = {}
        self.link_index_to_name = {-1: "base_link"}
        for joint_idx in range(pb.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_info = pb.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client_id)
            joint_type = joint_info[2]
            child_link_name = joint_info[12].decode("utf-8")
            self.link_name_to_index[child_link_name] = joint_idx
            self.link_index_to_name[joint_idx] = child_link_name
            if joint_type == pb.JOINT_REVOLUTE:
                self.revolute_joint_indices.append(joint_idx)
        if len(self.revolute_joint_indices) != len(self.joint_names):
            raise ValueError(
                f"URDF revolute joint count mismatch: parsed {len(self.joint_names)} from XML, "
                f"loaded {len(self.revolute_joint_indices)} in pybullet."
            )
        if self.cfg.tcp_link_name not in self.link_name_to_index:
            raise KeyError(
                f"TCP link `{self.cfg.tcp_link_name}` not found in URDF. "
                f"Available links: {sorted(self.link_name_to_index.keys())}"
            )
        self.tcp_link_index = self.link_name_to_index[self.cfg.tcp_link_name]
        self.stats_mode = (
            _resolve_stats_mode(self.cfg.stats_path)
            if self.cfg.stats_mode == "auto"
            else self.cfg.stats_mode
        )
        if self.stats_mode not in ("increment", "bspline"):
            raise ValueError(f"Unsupported pybullet stats mode: {self.stats_mode}")
        if self.stats_mode == "bspline":
            self.stats_mean, self.stats_std = load_delta_w_stats(self.cfg.stats_path)
        else:
            self.stats_mean, self.stats_std = load_increment_stats(self.cfg.stats_path)
        if self.cfg.robot_surface_points_per_link <= 0:
            raise ValueError(
                "training.pybullet_eval.robot_surface_points_per_link must be positive, "
                f"got {self.cfg.robot_surface_points_per_link}"
            )
        self.workpiece_cache: dict[int, int] = {}
        self.workpiece_stl_path_cache: dict[int, Path] = {}
        self.sdf_cache: dict[int, SDFGrid] = {}
        self.robot_surface_points_by_link = self._build_robot_collision_surface_points(
            resolved_urdf_path=self.resolved_urdf_path,
            points_per_link=self.cfg.robot_surface_points_per_link,
        )
        if self.cfg.max_joint_step_rad <= 0:
            raise ValueError(
                f"training.pybullet_eval.max_joint_step_rad must be positive, got {self.cfg.max_joint_step_rad}"
            )
        if self.cfg.min_interpolated_steps_per_segment <= 0:
            raise ValueError(
                "training.pybullet_eval.min_interpolated_steps_per_segment must be positive, "
                f"got {self.cfg.min_interpolated_steps_per_segment}"
            )

    def close(self) -> None:
        if getattr(self, "client_id", None) is not None:
            self.pb.disconnect(self.client_id)
            self.client_id = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _unnormalize_joint_state(self, normalized_joint_state: np.ndarray) -> np.ndarray:
        return unnormalize_joint_trajectory_with_urdf_limits(
            normalized_trajectory=np.asarray(normalized_joint_state, dtype=np.float32).reshape(1, -1),
            lower_limits=self.joint_lower_limits,
            upper_limits=self.joint_upper_limits,
        )[0]

    def _load_workpiece_body(self, workpiece_id: int) -> int:
        workpiece_id = int(workpiece_id)
        if workpiece_id in self.workpiece_cache:
            return self.workpiece_cache[workpiece_id]
        stl_path = _resolve_workpiece_stl_path(
            workpiece_id=workpiece_id,
            jobs_root=self.cfg.jobs_root,
            simple_jobs_root=self.cfg.simple_jobs_root,
            simple_workpiece_id_offset=self.cfg.simple_workpiece_id_offset,
            job_name_template=self.cfg.job_name_template,
            workpiece_filename=self.cfg.workpiece_filename,
        )
        self.workpiece_stl_path_cache[workpiece_id] = stl_path
        collision_shape = self.pb.createCollisionShape(
            shapeType=self.pb.GEOM_MESH,
            fileName=str(stl_path),
            meshScale=[0.001, 0.001, 0.001],
            flags=self.pb.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=self.client_id,
        )
        body_id = self.pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=[self.cfg.stl_x_offset_m, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        self.workpiece_cache[workpiece_id] = body_id
        return body_id

    def _workpiece_collision_object_name(self, workpiece_id: int) -> str:
        workpiece_id = int(workpiece_id)
        stl_path = self.workpiece_stl_path_cache.get(workpiece_id)
        if stl_path is None:
            stl_path = _resolve_workpiece_stl_path(
                workpiece_id=workpiece_id,
                jobs_root=self.cfg.jobs_root,
                simple_jobs_root=self.cfg.simple_jobs_root,
                simple_workpiece_id_offset=self.cfg.simple_workpiece_id_offset,
                job_name_template=self.cfg.job_name_template,
                workpiece_filename=self.cfg.workpiece_filename,
            )
            self.workpiece_stl_path_cache[workpiece_id] = stl_path
        return f"workpiece_id={workpiece_id} path={stl_path}"

    def _load_workpiece_sdf(self, workpiece_id: int) -> SDFGrid | None:
        workpiece_id = int(workpiece_id)
        if workpiece_id in self.sdf_cache:
            return self.sdf_cache[workpiece_id]
        try:
            sdf_path = _resolve_workpiece_file_path(
                workpiece_id=workpiece_id,
                jobs_root=self.cfg.jobs_root,
                simple_jobs_root=self.cfg.simple_jobs_root,
                simple_workpiece_id_offset=self.cfg.simple_workpiece_id_offset,
                job_name_template=self.cfg.job_name_template,
                filename=self.cfg.sdf_filename,
                file_label="SDF",
            )
        except FileNotFoundError:
            if self.cfg.sdf_required:
                raise
            return None
        sdf_grid = SDFGrid.load(
            sdf_path,
            out_of_bounds_value_m=self.cfg.sdf_out_of_bounds_value_m,
        )
        self.sdf_cache[workpiece_id] = sdf_grid
        return sdf_grid

    def _build_robot_collision_surface_points(
        self,
        resolved_urdf_path: str,
        points_per_link: int,
    ) -> dict[int, np.ndarray]:
        try:
            import trimesh
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SDF validation requires `trimesh` to sample robot collision geometry."
            ) from exc

        package_roots = [Path(root).expanduser().resolve() for root in self.cfg.urdf_package_roots]
        package_roots.append(Path(resolved_urdf_path).expanduser().resolve().parent)
        tree = ET.parse(resolved_urdf_path)
        root = tree.getroot()
        points_by_link: dict[int, list[np.ndarray]] = {}

        for link_elem in root.findall("link"):
            link_name = link_elem.get("name")
            if not link_name:
                continue
            if link_name in self.link_name_to_index:
                link_index = self.link_name_to_index[link_name]
            elif link_name == "base_link":
                link_index = -1
            else:
                continue

            link_candidates = []
            for collision_elem in link_elem.findall("collision"):
                geometry_elem = collision_elem.find("geometry")
                if geometry_elem is None:
                    continue
                mesh_elem = geometry_elem.find("mesh")
                box_elem = geometry_elem.find("box")
                local_points = None
                if mesh_elem is not None:
                    filename = mesh_elem.get("filename")
                    if filename is None:
                        continue
                    mesh_path = _resolve_mesh_filename(filename, package_roots)
                    mesh = trimesh.load_mesh(mesh_path, force="mesh")
                    scale = _parse_float_vector(mesh_elem.get("scale"), (1.0, 1.0, 1.0)).reshape(1, 3)
                    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
                    face_centers = np.asarray(mesh.triangles_center, dtype=np.float32) * scale
                    local_points = np.concatenate([vertices, face_centers], axis=0)
                elif box_elem is not None:
                    size = _parse_float_vector(box_elem.get("size"), (0.0, 0.0, 0.0)).reshape(3)
                    half = size * 0.5
                    corners = np.asarray(
                        [
                            [sx * half[0], sy * half[1], sz * half[2]]
                            for sx in (-1.0, 1.0)
                            for sy in (-1.0, 1.0)
                            for sz in (-1.0, 1.0)
                        ],
                        dtype=np.float32,
                    )
                    face_centers = np.asarray(
                        [
                            [half[0], 0.0, 0.0], [-half[0], 0.0, 0.0],
                            [0.0, half[1], 0.0], [0.0, -half[1], 0.0],
                            [0.0, 0.0, half[2]], [0.0, 0.0, -half[2]],
                        ],
                        dtype=np.float32,
                    )
                    local_points = np.concatenate([corners, face_centers], axis=0)
                if local_points is None or local_points.size == 0:
                    continue
                local_points = _apply_origin_transform(local_points, collision_elem.find("origin"))
                link_candidates.append(local_points.astype(np.float32))

            if link_candidates:
                merged = np.concatenate(link_candidates, axis=0)
                selected = _select_deterministic_surface_points(merged, points_per_link)
                points_by_link.setdefault(link_index, []).append(selected)

        return {
            link_index: np.concatenate(point_chunks, axis=0).astype(np.float32)
            for link_index, point_chunks in points_by_link.items()
        }

    def _get_link_pose(self, link_index: int) -> tuple[np.ndarray, np.ndarray]:
        if link_index == -1:
            position, orientation = self.pb.getBasePositionAndOrientation(
                self.robot_id,
                physicsClientId=self.client_id,
            )
            return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)
        link_state = self.pb.getLinkState(
            self.robot_id,
            link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return np.asarray(link_state[4], dtype=np.float32), np.asarray(link_state[5], dtype=np.float32)

    def _robot_surface_points_world(self) -> np.ndarray:
        world_points = []
        for link_index, local_points in self.robot_surface_points_by_link.items():
            position, orientation = self._get_link_pose(link_index)
            rotation = np.asarray(
                self.pb.getMatrixFromQuaternion(orientation),
                dtype=np.float32,
            ).reshape(3, 3)
            world_points.append(local_points @ rotation.T + position.reshape(1, 3))
        if not world_points:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(world_points, axis=0).astype(np.float32)

    def _min_sdf_detail_for_current_robot_state(
        self,
        sdf_grid: SDFGrid | None,
    ) -> dict[str, float | int | str | None]:
        if sdf_grid is None:
            return {
                "min_sdf_distance_m": float("nan"),
                "nearest_sdf_link_index": None,
                "nearest_sdf_link_name": None,
            }
        best_distance = float("nan")
        best_link_index = None
        best_link_name = None
        for link_index, local_points in self.robot_surface_points_by_link.items():
            position, orientation = self._get_link_pose(link_index)
            rotation = np.asarray(
                self.pb.getMatrixFromQuaternion(orientation),
                dtype=np.float32,
            ).reshape(3, 3)
            world_points = local_points @ rotation.T + position.reshape(1, 3)
            sdf_values = sdf_grid.query(world_points)
            if np.all(np.isnan(sdf_values)):
                continue
            link_min = float(np.nanmin(sdf_values))
            if np.isnan(best_distance) or link_min < best_distance:
                best_distance = link_min
                best_link_index = int(link_index)
                best_link_name = self.link_index_to_name.get(int(link_index), str(link_index))
        return {
            "min_sdf_distance_m": float(best_distance),
            "nearest_sdf_link_index": best_link_index,
            "nearest_sdf_link_name": best_link_name,
        }

    def _min_sdf_distance_for_current_robot_state(self, sdf_grid: SDFGrid | None) -> float:
        return float(self._min_sdf_detail_for_current_robot_state(sdf_grid)["min_sdf_distance_m"])

    def collect_joint_trajectory_sdf(
        self,
        workpiece_id: int,
        joint_trajectory: np.ndarray,
    ) -> np.ndarray:
        return self.collect_joint_trajectory_sdf_with_link_details(
            workpiece_id=workpiece_id,
            joint_trajectory=joint_trajectory,
        )["all_sdf_values"]

    def collect_joint_trajectory_sdf_with_link_details(
        self,
        workpiece_id: int,
        joint_trajectory: np.ndarray,
    ) -> dict[str, object]:
        sdf_grid = self._load_workpiece_sdf(workpiece_id)
        if sdf_grid is None:
            return {
                "all_sdf_values": np.empty((0, 0), dtype=np.float32),
                "sdf_values_by_link": {},
            }
        joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
        expected_shape = (int(self.cfg.target_steps), len(self.revolute_joint_indices))
        if joint_trajectory.shape != expected_shape:
            raise ValueError(
                "SDF candidate scoring expects joint trajectory shape "
                f"{expected_shape}, got {joint_trajectory.shape}."
            )

        trajectory_sdf_values = []
        trajectory_sdf_values_by_link: dict[str, list[np.ndarray]] = {
            self.link_index_to_name.get(int(link_index), str(link_index)): []
            for link_index in self.robot_surface_points_by_link.keys()
        }
        for joint_state in joint_trajectory:
            self._set_robot_joints(joint_state)
            timestep_values = []
            for link_index, local_points in self.robot_surface_points_by_link.items():
                position, orientation = self._get_link_pose(link_index)
                rotation = np.asarray(
                    self.pb.getMatrixFromQuaternion(orientation),
                    dtype=np.float32,
                ).reshape(3, 3)
                world_points = local_points @ rotation.T + position.reshape(1, 3)
                link_sdf_values = sdf_grid.query(world_points).astype(np.float32)
                link_name = self.link_index_to_name.get(int(link_index), str(link_index))
                trajectory_sdf_values_by_link[link_name].append(link_sdf_values)
                timestep_values.append(link_sdf_values)
            if not timestep_values:
                return {
                    "all_sdf_values": np.empty((0, 0), dtype=np.float32),
                    "sdf_values_by_link": {},
                }
            trajectory_sdf_values.append(np.concatenate(timestep_values, axis=0).astype(np.float32))
        return {
            "all_sdf_values": np.stack(
                trajectory_sdf_values,
                axis=0,
            ).astype(np.float32),
            "sdf_values_by_link": {
                link_name: np.stack(link_values, axis=0).astype(np.float32)
                for link_name, link_values in trajectory_sdf_values_by_link.items()
            },
        }

    def score_candidate(
        self,
        workpiece_id: int,
        normalized_control_points: np.ndarray,
        joint_trajectory: np.ndarray,
    ) -> dict[str, object]:
        sdf_result = self.collect_joint_trajectory_sdf_with_link_details(
            workpiece_id=workpiece_id,
            joint_trajectory=joint_trajectory,
        )
        _ = normalized_control_points
        return _score_safety_sdf_candidate(
            sdf_values=np.asarray(sdf_result["all_sdf_values"], dtype=np.float32),
            topk=self.cfg.selection_topk,
            d_select=self.cfg.selection_d_safe,
            sdf_values_by_link={
                str(link_name): np.asarray(link_values, dtype=np.float32)
                for link_name, link_values in dict(sdf_result["sdf_values_by_link"]).items()
            },
        )

    def _set_robot_joints(self, joint_state: np.ndarray) -> None:
        joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
        if joint_state.shape[0] != len(self.revolute_joint_indices):
            raise ValueError(
                f"Joint state length mismatch, expected {len(self.revolute_joint_indices)}, got {joint_state.shape[0]}"
            )
        for joint_idx, joint_value in zip(self.revolute_joint_indices, joint_state):
            self.pb.resetJointState(
                self.robot_id,
                joint_idx,
                float(joint_value),
                physicsClientId=self.client_id,
            )

    def _get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        link_state = self.pb.getLinkState(
            self.robot_id,
            self.tcp_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        position = np.asarray(link_state[4], dtype=np.float32)
        orientation = np.asarray(link_state[5], dtype=np.float32)
        return position, orientation

    def _target_world_position(
        self,
        start_joint_state: np.ndarray,
        goal_position_normalized: np.ndarray,
    ) -> np.ndarray:
        self._set_robot_joints(start_joint_state)
        start_tcp_position, start_tcp_quat = self._get_tcp_pose()
        goal_position_local_m = np.asarray(goal_position_normalized, dtype=np.float32).reshape(3) * self.cfg.goal_position_norm_m
        rotation_matrix = np.asarray(
            self.pb.getMatrixFromQuaternion(start_tcp_quat),
            dtype=np.float32,
        ).reshape(3, 3)
        return start_tcp_position + rotation_matrix @ goal_position_local_m

    def reconstruct_candidate(
        self,
        pred_action_horizon: np.ndarray,
        start_joint_normalized: np.ndarray,
        end_joint_normalized: np.ndarray,
    ) -> dict[str, np.ndarray]:
        start_joint_state = self._unnormalize_joint_state(start_joint_normalized)
        end_joint_state = self._unnormalize_joint_state(end_joint_normalized)
        pred_action_horizon = np.asarray(pred_action_horizon, dtype=np.float32)
        if pred_action_horizon.ndim != 2 or pred_action_horizon.shape[1] != 6:
            raise ValueError(
                f"Predicted action horizon must have shape [T, 6] for pybullet validation, got {pred_action_horizon.shape}"
            )
        if self.stats_mode == "bspline":
            predicted_free_control_points = int(pred_action_horizon.shape[0])
            inferred_num_control_points = (
                predicted_free_control_points + 2 * FIXED_CONTROL_POINTS_PER_SIDE
            )
            configured_free_control_points = (
                int(self.cfg.num_control_points) - 2 * FIXED_CONTROL_POINTS_PER_SIDE
            )
            num_control_points = int(self.cfg.num_control_points)
            if configured_free_control_points != predicted_free_control_points:
                print(
                    "[PyBulletValidation] overriding num_control_points from "
                    f"{self.cfg.num_control_points} to {inferred_num_control_points} "
                    f"to match predicted free control points={predicted_free_control_points}."
                )
                num_control_points = inferred_num_control_points
            recon_result = reconstruct_trajectory_from_normalized_free_residual(
                normalized_free_delta_w=pred_action_horizon,
                start_state=np.asarray(start_joint_normalized, dtype=np.float32),
                end_state=np.asarray(end_joint_normalized, dtype=np.float32),
                mean=self.stats_mean,
                std=self.stats_std,
                num_control_points=num_control_points,
                num_steps=self.cfg.target_steps,
                degree=self.cfg.spline_degree,
            )
            joint_trajectory = unnormalize_joint_trajectory_with_urdf_limits(
                normalized_trajectory=recon_result["fitted_trajectory"],
                lower_limits=self.joint_lower_limits,
                upper_limits=self.joint_upper_limits,
            )
            return {
                "normalized_control_points": np.asarray(
                    recon_result["w_star"],
                    dtype=np.float32,
                ),
                "joint_trajectory": np.asarray(
                    joint_trajectory,
                    dtype=np.float32,
                ),
            }

        denormalized_deltas = pred_action_horizon * self.stats_std.reshape(1, 6) + self.stats_mean.reshape(1, 6)
        cumulative = start_joint_state.reshape(1, 6) + np.cumsum(denormalized_deltas, axis=0)
        joint_trajectory = np.concatenate([start_joint_state.reshape(1, 6), cumulative], axis=0).astype(np.float32)
        if joint_trajectory.shape[0] >= 2:
            joint_trajectory[-1] = joint_trajectory[-1].astype(np.float32)
        _ = end_joint_state
        return {
            "normalized_control_points": np.asarray(
                pred_action_horizon,
                dtype=np.float32,
            ),
            "joint_trajectory": joint_trajectory,
        }

    def reconstruct_joint_trajectory(
        self,
        pred_action_horizon: np.ndarray,
        start_joint_normalized: np.ndarray,
        end_joint_normalized: np.ndarray,
    ) -> np.ndarray:
        return self.reconstruct_candidate(
            pred_action_horizon=pred_action_horizon,
            start_joint_normalized=start_joint_normalized,
            end_joint_normalized=end_joint_normalized,
        )["joint_trajectory"]

    def densify_joint_trajectory(self, joint_trajectory: np.ndarray) -> np.ndarray:
        joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
        if joint_trajectory.ndim != 2 or joint_trajectory.shape[1] != len(self.revolute_joint_indices):
            raise ValueError(
                "joint_trajectory must have shape [T, J] matching robot joints, "
                f"got {joint_trajectory.shape}"
            )
        if (not self.cfg.interpolate_for_collision) or joint_trajectory.shape[0] <= 1:
            return joint_trajectory.astype(np.float32)

        dense_segments = [joint_trajectory[0]]
        for segment_idx in range(joint_trajectory.shape[0] - 1):
            q0 = joint_trajectory[segment_idx]
            q1 = joint_trajectory[segment_idx + 1]
            max_delta = float(np.max(np.abs(q1 - q0)))
            interpolation_steps = max(
                self.cfg.min_interpolated_steps_per_segment,
                int(np.ceil(max_delta / self.cfg.max_joint_step_rad)),
            )
            for step_idx in range(1, interpolation_steps + 1):
                alpha = float(step_idx) / float(interpolation_steps)
                dense_segments.append(((1.0 - alpha) * q0 + alpha * q1).astype(np.float32))
        return np.asarray(dense_segments, dtype=np.float32)

    def evaluate_trajectory(
        self,
        workpiece_id: int,
        joint_trajectory: np.ndarray,
        start_joint_state: np.ndarray,
        goal_position_normalized: np.ndarray,
        episode_idx: int | None = None,
    ) -> dict[str, object]:
        workpiece_body_id = self._load_workpiece_body(workpiece_id)
        sdf_grid = self._load_workpiece_sdf(workpiece_id)
        collision_object_name = self._workpiece_collision_object_name(workpiece_id)
        target_world_position = self._target_world_position(start_joint_state, goal_position_normalized)
        joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
        if joint_trajectory.ndim != 2 or joint_trajectory.shape[1] != len(self.revolute_joint_indices):
            raise ValueError(
                "joint_trajectory must have shape [T, J] matching robot joints, "
                f"got {joint_trajectory.shape}"
            )
        if joint_trajectory.shape[0] != self.cfg.target_steps:
            raise ValueError(
                f"PyBullet validation expects exactly {self.cfg.target_steps} reconstructed joint states, "
                f"got {joint_trajectory.shape[0]}. Adjust training.pybullet_eval.target_steps or action reconstruction."
            )
        has_collision = False
        segment_collision_steps = 0
        min_sdf_distance_m = float("nan")
        final_tcp_position = None
        collision_events = []
        collision_link_names = set()
        for timestep, joint_state in enumerate(joint_trajectory):
            self._set_robot_joints(joint_state)
            self.pb.performCollisionDetection(physicsClientId=self.client_id)
            contacts = self.pb.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=workpiece_body_id,
                distance=self.cfg.collision_distance_threshold,
                physicsClientId=self.client_id,
            )
            sdf_detail = self._min_sdf_detail_for_current_robot_state(sdf_grid)
            if contacts:
                has_collision = True
                segment_collision_steps += 1
                collision_link_indices = sorted({int(contact[3]) for contact in contacts})
                for collision_link_index in collision_link_indices:
                    collision_link_name = self.link_index_to_name.get(
                        collision_link_index,
                        str(collision_link_index),
                    )
                    collision_link_names.add(collision_link_name)
                    collision_events.append({
                        "episode_idx": None if episode_idx is None else int(episode_idx),
                        "workpiece_id": int(workpiece_id),
                        "collision_link_name": collision_link_name,
                        "collision_object_name": collision_object_name,
                        "collision_timestep": int(timestep),
                        "timestep_min_sdf_distance_m": float(sdf_detail["min_sdf_distance_m"]),
                        "nearest_sdf_link_name": sdf_detail["nearest_sdf_link_name"],
                    })
            step_min_sdf_distance_m = float(sdf_detail["min_sdf_distance_m"])
            if not np.isnan(step_min_sdf_distance_m):
                if np.isnan(min_sdf_distance_m):
                    min_sdf_distance_m = step_min_sdf_distance_m
                else:
                    min_sdf_distance_m = min(min_sdf_distance_m, step_min_sdf_distance_m)
            final_tcp_position, _ = self._get_tcp_pose()
        if final_tcp_position is None:
            raise ValueError("Empty joint trajectory is not valid for pybullet validation.")

        legacy_collision_steps = float(segment_collision_steps)
        legacy_trajectory_steps = float(joint_trajectory.shape[0])
        if self.cfg.interpolate_for_collision and self.cfg.log_legacy_pybullet_metrics:
            dense_joint_trajectory = self.densify_joint_trajectory(joint_trajectory)
            legacy_collision_steps = 0.0
            for joint_state in dense_joint_trajectory:
                self._set_robot_joints(joint_state)
                self.pb.performCollisionDetection(physicsClientId=self.client_id)
                contacts = self.pb.getClosestPoints(
                    bodyA=self.robot_id,
                    bodyB=workpiece_body_id,
                    distance=self.cfg.collision_distance_threshold,
                    physicsClientId=self.client_id,
                )
                if contacts:
                    legacy_collision_steps += 1.0
            legacy_trajectory_steps = float(dense_joint_trajectory.shape[0])

        goal_error = float(np.linalg.norm(final_tcp_position - target_world_position))
        goal_reached = goal_error <= self.cfg.goal_tolerance_m
        success = bool(not has_collision)
        return {
            "has_collision": bool(has_collision),
            "segment_collision_steps": float(segment_collision_steps),
            "segment_steps": float(joint_trajectory.shape[0]),
            "min_sdf_distance_m": float(min_sdf_distance_m),
            "legacy_collision_steps": float(legacy_collision_steps),
            "legacy_trajectory_steps": float(legacy_trajectory_steps),
            "goal_error_m": goal_error,
            "goal_reached": bool(goal_reached),
            "success": success,
            "collision_link_names": sorted(collision_link_names),
            "collision_events": collision_events,
        }


class PyBulletValidationRunner:
    def __init__(self, cfg: PyBulletValidationConfig):
        self.cfg = cfg
        if self.cfg.target_steps not in (32, 64):
            raise ValueError(
                "PyBullet multi-candidate validation requires target_steps=32 or 64, "
                f"got {self.cfg.target_steps}"
            )
        if self.cfg.interpolate_for_collision:
            raise ValueError(
                "PyBullet multi-candidate validation requires "
                "interpolate_for_collision=false."
            )
        if self.cfg.num_candidates <= 0:
            raise ValueError(
                "training.pybullet_eval.num_candidates must be positive, "
                f"got {self.cfg.num_candidates}"
            )
        if self.cfg.candidate_selection != "weighted_sdf":
            raise ValueError(
                "training.pybullet_eval.candidate_selection must be "
                f"`weighted_sdf`, got {self.cfg.candidate_selection!r}"
            )
        if self.cfg.selection_topk <= 0:
            raise ValueError(
                "training.pybullet_eval.selection_topk must be positive, "
                f"got {self.cfg.selection_topk}"
            )
        if self.cfg.selection_d_safe <= 0:
            raise ValueError(
                "training.pybullet_eval.selection_d_safe must be positive, "
                f"got {self.cfg.selection_d_safe}"
            )
        if self.cfg.num_candidates > 1 and self.cfg.diffusion_sampling_seed is None:
            raise ValueError(
                "training.pybullet_eval.diffusion_sampling_seed is required "
                "for deterministic multi-candidate validation."
            )
        self.validator = PyBulletCollisionValidator(cfg)
        if self.validator.stats_mode != "bspline":
            raise ValueError(
                "Safety-key PyBullet candidate selection requires B-spline stats."
            )
        if self.cfg.collision_log_path is not None:
            log_path = Path(self.cfg.collision_log_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
        self._fixed_episode_indices: np.ndarray | None = None

    def close(self) -> None:
        self.validator.close()

    def _append_collision_events(self, collision_events: list[dict[str, object]]) -> None:
        if not collision_events or self.cfg.collision_log_path is None:
            return
        log_path = Path(self.cfg.collision_log_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            for event in collision_events:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def prepare_episode_subset(
        self,
        replay_buffer,
        episode_mask: np.ndarray,
    ) -> np.ndarray:
        if "workpiece_ids" not in replay_buffer.meta:
            raise KeyError(
                "PyBullet validation requires `meta/workpiece_ids` in the zarr dataset. "
                "Rebuild the dataset with workpiece metadata."
            )
        workpiece_ids = np.asarray(
            replay_buffer.meta["workpiece_ids"][:],
            dtype=np.int64,
        )
        episode_indices = np.flatnonzero(np.asarray(episode_mask, dtype=bool))
        if not self.cfg.include_regular_jobs or not self.cfg.include_simple_jobs:
            selected_episode_indices = []
            for episode_idx in episode_indices.tolist():
                workpiece_id = int(workpiece_ids[episode_idx])
                is_simple_job = (
                    workpiece_id >= int(self.cfg.simple_workpiece_id_offset)
                )
                if is_simple_job and not self.cfg.include_simple_jobs:
                    continue
                if (not is_simple_job) and not self.cfg.include_regular_jobs:
                    continue
                selected_episode_indices.append(episode_idx)
            episode_indices = np.asarray(
                selected_episode_indices,
                dtype=np.int64,
            )
        if self.cfg.max_episodes is not None:
            max_episodes = int(self.cfg.max_episodes)
            if max_episodes <= 0:
                raise ValueError(
                    "training.pybullet_eval.max_episodes must be positive, "
                    f"got {max_episodes}"
                )
            if episode_indices.size > max_episodes:
                if self.cfg.random_sample_episodes:
                    rng = np.random.default_rng(self.cfg.random_seed)
                    episode_indices = np.sort(
                        rng.choice(
                            episode_indices,
                            size=max_episodes,
                            replace=False,
                        )
                    )
                else:
                    episode_indices = episode_indices[:max_episodes]
        self._fixed_episode_indices = episode_indices.copy()
        print(
            "[PyBullet validation] fixed episode subset prepared: "
            f"{self._fixed_episode_indices.size} episodes "
            f"(seed={self.cfg.random_seed})"
        )
        return self._fixed_episode_indices.copy()

    def _build_obs_entry(
        self,
        replay_buffer,
        episode_idx: int,
        workpiece_id: int,
        obs_keys: tuple[str, ...],
        n_obs_steps: int,
        dataset=None,
        policy=None,
    ) -> dict[str, np.ndarray]:
        episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
        start_idx, end_idx = _episode_bounds(episode_ends, episode_idx)
        episode_length = end_idx - start_idx
        if episode_length <= 0:
            raise ValueError(f"Episode {episode_idx} is empty.")
        raw_obs: dict[str, np.ndarray] = {}
        for key in obs_keys:
            value = np.asarray(replay_buffer[key][start_idx:end_idx], dtype=np.float32)
            value = value[:n_obs_steps]
            if value.shape[0] < n_obs_steps:
                pad_count = n_obs_steps - value.shape[0]
                pad = np.repeat(value[-1:], pad_count, axis=0)
                value = np.concatenate([value, pad], axis=0)
            raw_obs[key] = value.copy()

        cspace_feature_key = getattr(policy, "cspace_feature_key", None)
        if cspace_feature_key is not None:
            if dataset is None or not hasattr(dataset, "get_cspace_feature_by_workpiece_id"):
                raise KeyError(
                    "PyBullet validation requires a dataset that can provide "
                    f"{cspace_feature_key!r} for the current policy."
                )
            cspace_feature = dataset.get_cspace_feature_by_workpiece_id(workpiece_id)
            raw_obs[cspace_feature_key] = np.asarray(cspace_feature, dtype=np.float32).copy()
        return raw_obs

    def _build_obs_batch(
        self,
        replay_buffer,
        episode_idx: int,
        workpiece_id: int,
        obs_keys: tuple[str, ...],
        n_obs_steps: int,
        device: torch.device,
        dataset=None,
        policy=None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        import torch

        raw_obs = self._build_obs_entry(
            replay_buffer=replay_buffer,
            episode_idx=episode_idx,
            workpiece_id=workpiece_id,
            obs_keys=obs_keys,
            n_obs_steps=n_obs_steps,
            dataset=dataset,
            policy=policy,
        )
        obs_batch: dict[str, torch.Tensor] = {
            key: torch.from_numpy(value[None]).to(device)
            for key, value in raw_obs.items()
        }
        return obs_batch, raw_obs

    def _build_obs_batch_for_episodes(
        self,
        replay_buffer,
        episode_indices: list[int],
        workpiece_ids: np.ndarray,
        obs_keys: tuple[str, ...],
        n_obs_steps: int,
        device: torch.device,
        dataset=None,
        policy=None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, np.ndarray]]]:
        import torch

        raw_obs_list = [
            self._build_obs_entry(
                replay_buffer=replay_buffer,
                episode_idx=int(episode_idx),
                workpiece_id=int(workpiece_ids[int(episode_idx)]),
                obs_keys=obs_keys,
                n_obs_steps=n_obs_steps,
                dataset=dataset,
                policy=policy,
            )
            for episode_idx in episode_indices
        ]
        obs_batch = {
            key: torch.from_numpy(
                np.stack([raw_obs[key] for raw_obs in raw_obs_list], axis=0)
            ).to(device)
            for key in raw_obs_list[0].keys()
        }
        return obs_batch, raw_obs_list

    @staticmethod
    def _update_progress_stats(
        metric: dict[str, object],
        processed_count: int,
        running_collision_count: int,
        running_valid_sdf_count: int,
        running_min_sdf_distance_m: float,
    ) -> tuple[int, int, float]:
        if metric["has_collision"]:
            running_collision_count += 1
        current_min_sdf_distance_m = float(metric["min_sdf_distance_m"])
        if not np.isnan(current_min_sdf_distance_m):
            running_valid_sdf_count += 1
            running_min_sdf_distance_m = min(running_min_sdf_distance_m, current_min_sdf_distance_m)
        _ = processed_count
        return (
            running_collision_count,
            running_valid_sdf_count,
            running_min_sdf_distance_m,
        )

    def run(
        self,
        policy,
        replay_buffer,
        episode_mask: np.ndarray,
        obs_keys: tuple[str, ...],
        n_obs_steps: int,
        device: torch.device,
        dataset=None,
    ) -> dict[str, object]:
        import torch
        try:
            import tqdm
        except ModuleNotFoundError:
            tqdm = None

        if self._fixed_episode_indices is None:
            self.prepare_episode_subset(
                replay_buffer=replay_buffer,
                episode_mask=episode_mask,
            )
        workpiece_ids = np.asarray(replay_buffer.meta["workpiece_ids"][:], dtype=np.int64)
        episode_indices = self._fixed_episode_indices
        if episode_indices.size == 0:
            return {}

        if self.cfg.num_workers <= 0:
            raise ValueError(
                f"training.pybullet_eval.num_workers must be >= 1, got {self.cfg.num_workers}"
            )
        if self.cfg.inference_batch_size <= 0:
            raise ValueError(
                "training.pybullet_eval.inference_batch_size must be >= 1, "
                f"got {self.cfg.inference_batch_size}"
            )
        if self.cfg.worker_chunksize <= 0:
            raise ValueError(
                "training.pybullet_eval.worker_chunksize must be >= 1, "
                f"got {self.cfg.worker_chunksize}"
            )

        sample_metrics = []
        running_collision_count = 0
        running_valid_sdf_count = 0
        running_min_sdf_distance_m = float("inf")
        invalid_candidate_count = 0
        selected_candidate_indices = []
        selected_candidate_min_sdf = []
        selected_candidate_min_sdf_gains = []
        selected_candidate_has_pen_scores = []
        selected_candidate_pen_step_counts = []
        selected_candidate_pen_point_counts = []
        selected_candidate_num_pen_scores = []
        selected_candidate_neg_min_sdf_scores = []
        selected_candidate_neg_worstk_mean_scores = []
        selected_candidate_margin_violation_scores = []
        selected_candidate_sdf_finite_ratios = []
        selected_candidate_sdf_finite_ratios_by_link: dict[str, list[float]] = {}
        candidate_any_zero_pen_count = 0
        candidate_all_has_pen_count = 0
        selected_has_pen_when_zero_pen_exists_count = 0
        zero_pen_exists_count = 0
        candidate_zero_pen_counts = []
        candidate_best_min_sdf_values = []
        candidate_sdf_pen_link_episode_counts: dict[str, float] = {}
        selected_sdf_pen_link_counts: dict[str, float] = {}
        candidate_debug_printed = False
        candidate_diversity_debug_printed = False
        raw_identical_candidate_episode_count = 0
        reconstructed_identical_candidate_episode_count = 0
        policy.eval()
        episode_list = episode_indices.tolist()
        tasks: list[dict[str, object]] = []
        print(
            "[PyBullet validation] policy class="
            f"{policy.__class__.__module__}.{policy.__class__.__name__}"
        )
        scheduler = getattr(policy, "noise_scheduler", None)
        print(
            "[PyBullet validation] scheduler class="
            f"{scheduler.__class__.__module__}.{scheduler.__class__.__name__}"
            if scheduler is not None
            else "[PyBullet validation] scheduler class=None"
        )
        print(
            "[PyBullet validation] candidate sampling "
            f"eta={self.cfg.candidate_scheduler_eta} "
            f"action_noise_std={self.cfg.candidate_action_noise_std} "
            f"action_noise_clip={self.cfg.candidate_action_noise_clip}"
        )
        if tqdm is not None:
            inference_progress = tqdm.tqdm(
                total=len(episode_list),
                desc="PyBullet inference",
                leave=True,
                mininterval=max(float(self.cfg.progress_mininterval_sec), 0.1),
            )
        else:
            print(f"[PyBullet inference] start {len(episode_list)} episodes")
            inference_progress = None
        with torch.no_grad():
            for batch_start in range(0, len(episode_list), int(self.cfg.inference_batch_size)):
                batch_episode_indices = episode_list[
                    batch_start: batch_start + int(self.cfg.inference_batch_size)
                ]
                obs_dict, raw_obs_list = self._build_obs_batch_for_episodes(
                    replay_buffer=replay_buffer,
                    episode_indices=batch_episode_indices,
                    workpiece_ids=workpiece_ids,
                    obs_keys=obs_keys,
                    n_obs_steps=n_obs_steps,
                    device=device,
                    dataset=dataset,
                    policy=policy,
                )
                candidate_action_batches = []
                candidate_seeds = []
                scheduler_step_kwargs = {}
                if self.cfg.candidate_scheduler_eta is not None:
                    scheduler_step_kwargs["eta"] = float(self.cfg.candidate_scheduler_eta)
                for candidate_idx in range(int(self.cfg.num_candidates)):
                    candidate_seed = (
                        int(self.cfg.diffusion_sampling_seed)
                        + candidate_idx * 1_000_003
                        + int(batch_start)
                    )
                    generator = torch.Generator(device=device)
                    generator.manual_seed(candidate_seed)
                    candidate_seeds.append(int(candidate_seed))
                    result = policy.predict_action(
                        obs_dict,
                        generator=generator,
                        num_inference_steps=self.cfg.inference_num_steps,
                        scheduler_step_kwargs=scheduler_step_kwargs,
                    )
                    candidate_action = result["action_pred"].detach().cpu().numpy().astype(np.float32)
                    if candidate_idx > 0 and self.cfg.candidate_action_noise_std > 0.0:
                        rng = np.random.default_rng(candidate_seed)
                        candidate_noise = rng.normal(
                            loc=0.0,
                            scale=float(self.cfg.candidate_action_noise_std),
                            size=candidate_action.shape,
                        ).astype(np.float32)
                        if self.cfg.candidate_action_noise_clip is not None:
                            candidate_noise = np.clip(
                                candidate_noise,
                                -float(self.cfg.candidate_action_noise_clip),
                                float(self.cfg.candidate_action_noise_clip),
                            ).astype(np.float32)
                        candidate_action = (candidate_action + candidate_noise).astype(np.float32)
                    candidate_action_batches.append(candidate_action)
                candidate_action_batch = np.stack(candidate_action_batches, axis=1)
                for local_idx, episode_idx in enumerate(batch_episode_indices):
                    raw_obs = raw_obs_list[local_idx]
                    workpiece_id = int(workpiece_ids[int(episode_idx)])
                    start_joint_normalized = raw_obs[
                        "first_joint_angles_normalized"
                    ][0].astype(np.float32)
                    end_joint_normalized = raw_obs[
                        "last_joint_angles_normalized"
                    ][0].astype(np.float32)
                    candidate_results = []
                    candidate_score_details = []
                    for candidate_idx in range(int(self.cfg.num_candidates)):
                        candidate_result = self.validator.reconstruct_candidate(
                            pred_action_horizon=candidate_action_batch[
                                local_idx, candidate_idx
                            ],
                            start_joint_normalized=start_joint_normalized,
                            end_joint_normalized=end_joint_normalized,
                        )
                        candidate_results.append(candidate_result)
                        candidate_score_details.append(
                            self.validator.score_candidate(
                                workpiece_id=workpiece_id,
                                normalized_control_points=candidate_result[
                                    "normalized_control_points"
                                ],
                                joint_trajectory=candidate_result[
                                    "joint_trajectory"
                                ],
                            )
                        )
                    raw_candidate_actions = np.asarray(
                        candidate_action_batch[local_idx],
                        dtype=np.float32,
                    )
                    raw_candidate_diffs = np.max(
                        np.abs(raw_candidate_actions - raw_candidate_actions[:1]),
                        axis=(1, 2),
                    )
                    raw_candidates_identical = bool(
                        raw_candidate_diffs.shape[0] <= 1
                        or np.all(raw_candidate_diffs[1:] <= 1e-7)
                    )
                    if raw_candidates_identical:
                        raw_identical_candidate_episode_count += 1
                        print(
                            "[PyBullet validation] WARNING: all raw candidate actions are identical "
                            f"for episode_idx={episode_idx}, workpiece_id={workpiece_id}, "
                            f"seeds={candidate_seeds}"
                        )
                    reconstructed_candidate_trajectories = np.stack(
                        [
                            np.asarray(candidate_result["joint_trajectory"], dtype=np.float32)
                            for candidate_result in candidate_results
                        ],
                        axis=0,
                    )
                    reconstructed_candidate_diffs = np.max(
                        np.abs(
                            reconstructed_candidate_trajectories
                            - reconstructed_candidate_trajectories[:1]
                        ),
                        axis=(1, 2),
                    )
                    reconstructed_candidates_identical = bool(
                        reconstructed_candidate_diffs.shape[0] <= 1
                        or np.all(reconstructed_candidate_diffs[1:] <= 1e-7)
                    )
                    if not candidate_diversity_debug_printed:
                        print(
                            "[PyBullet validation] candidate diversity "
                            f"episode_idx={episode_idx}, workpiece_id={workpiece_id}, "
                            f"raw_max_diff={float(np.max(raw_candidate_diffs[1:]) if raw_candidate_diffs.shape[0] > 1 else 0.0):.8f}, "
                            f"reconstructed_max_diff={float(np.max(reconstructed_candidate_diffs[1:]) if reconstructed_candidate_diffs.shape[0] > 1 else 0.0):.8f}, "
                            f"seeds={candidate_seeds}"
                        )
                        candidate_diversity_debug_printed = True
                    if reconstructed_candidates_identical:
                        reconstructed_identical_candidate_episode_count += 1
                        print(
                            "[PyBullet validation] WARNING: all reconstructed candidate trajectories are identical "
                            f"for episode_idx={episode_idx}, workpiece_id={workpiece_id}, "
                            f"seeds={candidate_seeds}, "
                            f"raw_identical={raw_candidates_identical}"
                        )
                    candidate_score_keys_array = np.asarray(
                        [
                            [
                                score_details["has_pen"],
                                score_details["pen_step_count"],
                                score_details["pen_point_count"],
                                score_details["neg_min_sdf"],
                                score_details["neg_worstk_mean"],
                                score_details["margin_violation"],
                            ]
                            for score_details in candidate_score_details
                        ],
                        dtype=np.float32,
                    )
                    invalid_candidate_count += int(
                        np.count_nonzero(~np.all(np.isfinite(candidate_score_keys_array), axis=1))
                    )
                    finite_candidate_mask = np.all(
                        np.isfinite(candidate_score_keys_array),
                        axis=1,
                    )
                    candidate_has_pen_array = np.asarray(
                        [
                            float(score_details["has_pen"])
                            for score_details in candidate_score_details
                        ],
                        dtype=np.float32,
                    )
                    candidate_min_sdf_array = np.asarray(
                        [
                            float(score_details["min_sdf_distance_m"])
                            for score_details in candidate_score_details
                        ],
                        dtype=np.float32,
                    )
                    finite_has_pen = candidate_has_pen_array[finite_candidate_mask]
                    zero_pen_candidate_count = int(np.count_nonzero(finite_has_pen == 0.0))
                    zero_pen_exists = zero_pen_candidate_count > 0
                    all_valid_candidates_have_pen = bool(
                        finite_has_pen.size > 0 and np.all(finite_has_pen > 0.0)
                    )
                    if zero_pen_exists:
                        candidate_any_zero_pen_count += 1
                        zero_pen_exists_count += 1
                    if all_valid_candidates_have_pen:
                        candidate_all_has_pen_count += 1
                    candidate_zero_pen_counts.append(float(zero_pen_candidate_count))
                    finite_candidate_min_sdf = candidate_min_sdf_array[
                        finite_candidate_mask & np.isfinite(candidate_min_sdf_array)
                    ]
                    if finite_candidate_min_sdf.size > 0:
                        candidate_best_min_sdf_values.append(
                            float(np.max(finite_candidate_min_sdf))
                        )
                    episode_pen_link_names = set()
                    for score_details in candidate_score_details:
                        for link_name in score_details.get("penetrating_link_names", []):
                            episode_pen_link_names.add(str(link_name))
                    for link_name in episode_pen_link_names:
                        _add_count(candidate_sdf_pen_link_episode_counts, link_name)
                    try:
                        selected_candidate_idx = _select_lowest_candidate_score_index(
                            candidate_score_keys_array
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "All SDF candidates are invalid for PyBullet validation "
                            f"episode_idx={episode_idx}, workpiece_id={workpiece_id}."
                        ) from exc
                    if not candidate_debug_printed:
                        print(
                            "[PyBullet validation] candidate score table "
                            f"episode_idx={episode_idx}, workpiece_id={workpiece_id}"
                        )
                        for candidate_idx, score_details in enumerate(candidate_score_details):
                            print(
                                "  "
                                f"cand={candidate_idx:02d} "
                                f"has_pen={float(score_details['has_pen']):.0f} "
                                f"pen_steps={float(score_details['pen_step_count']):.0f} "
                                f"pen_points={float(score_details['pen_point_count']):.0f} "
                                f"min_sdf={float(score_details['min_sdf_distance_m']):.6f} "
                                f"worstk_mean={-float(score_details['neg_worstk_mean']):.6f} "
                                f"margin_violation={float(score_details['margin_violation']):.6f} "
                                f"sdf_finite_ratio={float(score_details['sdf_finite_ratio']):.3f}"
                            )
                        candidate_debug_printed = True
                    selected_details = candidate_score_details[
                        selected_candidate_idx
                    ]
                    baseline_details = candidate_score_details[0]
                    selected_min_sdf = float(
                        selected_details["min_sdf_distance_m"]
                    )
                    baseline_min_sdf = float(
                        baseline_details["min_sdf_distance_m"]
                    )
                    selected_candidate_indices.append(selected_candidate_idx)
                    selected_candidate_has_pen_scores.append(
                        float(selected_details["has_pen"])
                    )
                    selected_candidate_pen_step_counts.append(
                        float(selected_details["pen_step_count"])
                    )
                    selected_candidate_pen_point_counts.append(
                        float(selected_details["pen_point_count"])
                    )
                    selected_candidate_min_sdf.append(selected_min_sdf)
                    selected_candidate_num_pen_scores.append(
                        float(selected_details["num_pen"])
                    )
                    selected_candidate_neg_min_sdf_scores.append(
                        float(selected_details["neg_min_sdf"])
                    )
                    selected_candidate_neg_worstk_mean_scores.append(
                        float(selected_details["neg_worstk_mean"])
                    )
                    selected_candidate_margin_violation_scores.append(
                        float(selected_details["margin_violation"])
                    )
                    selected_candidate_sdf_finite_ratios.append(
                        float(selected_details["sdf_finite_ratio"])
                    )
                    for link_name, link_ratio in dict(selected_details["sdf_finite_ratio_by_link"]).items():
                        selected_candidate_sdf_finite_ratios_by_link.setdefault(str(link_name), []).append(
                            float(link_ratio)
                        )
                    if zero_pen_exists and float(selected_details["has_pen"]) > 0.0:
                        selected_has_pen_when_zero_pen_exists_count += 1
                    for link_name in selected_details.get("penetrating_link_names", []):
                        _add_count(selected_sdf_pen_link_counts, str(link_name))
                    if np.isfinite(baseline_min_sdf):
                        selected_candidate_min_sdf_gains.append(
                            selected_min_sdf - baseline_min_sdf
                        )
                    tasks.append({
                        "episode_idx": int(episode_idx),
                        "workpiece_id": workpiece_id,
                        "joint_trajectory": candidate_results[
                            selected_candidate_idx
                        ]["joint_trajectory"],
                        "start_joint_normalized": start_joint_normalized,
                        "goal_position_normalized": raw_obs["goal_position"][0].astype(np.float32),
                        "selected_candidate_index": selected_candidate_idx,
                        "selected_candidate_score_key": [
                            float(selected_details["has_pen"]),
                            float(selected_details["pen_step_count"]),
                            float(selected_details["pen_point_count"]),
                            float(selected_details["neg_min_sdf"]),
                            float(selected_details["neg_worstk_mean"]),
                            float(selected_details["margin_violation"]),
                        ],
                    })
                if inference_progress is not None:
                    inference_progress.update(len(batch_episode_indices))
                else:
                    processed = len(tasks)
                    if processed == len(episode_list) or processed == len(batch_episode_indices) or processed % 10 == 0:
                        print(f"[PyBullet inference] {processed}/{len(episode_list)} episodes")
        if inference_progress is not None:
            inference_progress.close()
        else:
            print(f"[PyBullet inference] done {len(episode_list)} episodes")

        if tqdm is not None:
            validation_progress = tqdm.tqdm(
                total=len(tasks),
                desc="PyBullet validation",
                leave=True,
                mininterval=max(float(self.cfg.progress_mininterval_sec), 0.1),
            )
        else:
            print(f"[PyBullet validation] start {len(tasks)} episodes")
            validation_progress = None

        def _report_validation_progress(processed_count: int, workpiece_id: int) -> None:
            min_d_str = (
                f"{running_min_sdf_distance_m:.4f}"
                if running_valid_sdf_count > 0
                else "nan"
            )
            if validation_progress is not None:
                validation_progress.update(1)
                validation_progress.set_postfix(
                    {
                        "wp": workpiece_id,
                        "coll_rate": f"{running_collision_count / processed_count:.3f}",
                        "min_d": min_d_str,
                    },
                    refresh=False,
                )
            elif processed_count == 1 or processed_count == len(tasks) or processed_count % 10 == 0:
                print(
                    "[PyBullet validation] "
                    f"{processed_count}/{len(tasks)} "
                    f"workpiece_id={workpiece_id} "
                    f"collision_rate={running_collision_count / processed_count:.3f} "
                    f"min_d_min={min_d_str}"
                )

        if int(self.cfg.num_workers) == 1:
            for processed_count, task in enumerate(tasks, start=1):
                start_joint_state = self.validator._unnormalize_joint_state(
                    np.asarray(task["start_joint_normalized"], dtype=np.float32)
                )
                metric = self.validator.evaluate_trajectory(
                    workpiece_id=int(task["workpiece_id"]),
                    joint_trajectory=np.asarray(
                        task["joint_trajectory"],
                        dtype=np.float32,
                    ),
                    start_joint_state=start_joint_state,
                    goal_position_normalized=np.asarray(task["goal_position_normalized"], dtype=np.float32),
                    episode_idx=int(task["episode_idx"]),
                )
                sample_metrics.append(metric)
                self._append_collision_events(metric.get("collision_events", []))
                (
                    running_collision_count,
                    running_valid_sdf_count,
                    running_min_sdf_distance_m,
                ) = self._update_progress_stats(
                    metric=metric,
                    processed_count=processed_count,
                    running_collision_count=running_collision_count,
                    running_valid_sdf_count=running_valid_sdf_count,
                    running_min_sdf_distance_m=running_min_sdf_distance_m,
                )
                _report_validation_progress(
                    processed_count=processed_count,
                    workpiece_id=int(task["workpiece_id"]),
                )
        else:
            mp_context = mp.get_context(self.cfg.worker_start_method)
            with mp_context.Pool(
                processes=int(self.cfg.num_workers),
                initializer=_init_pybullet_validation_worker,
                initargs=(self.cfg,),
            ) as pool:
                for processed_count, (task, metric) in enumerate(
                    zip(
                        tasks,
                        pool.imap(
                            _run_pybullet_validation_task,
                            tasks,
                            chunksize=int(self.cfg.worker_chunksize),
                        ),
                    ),
                    start=1,
                ):
                    sample_metrics.append(metric)
                    self._append_collision_events(metric.get("collision_events", []))
                    (
                        running_collision_count,
                        running_valid_sdf_count,
                        running_min_sdf_distance_m,
                    ) = self._update_progress_stats(
                        metric=metric,
                        processed_count=processed_count,
                        running_collision_count=running_collision_count,
                        running_valid_sdf_count=running_valid_sdf_count,
                        running_min_sdf_distance_m=running_min_sdf_distance_m,
                    )
                    _report_validation_progress(
                        processed_count=processed_count,
                        workpiece_id=int(task["workpiece_id"]),
                    )

        if validation_progress is not None:
            validation_progress.close()
        else:
            print(f"[PyBullet validation] done {len(tasks)} episodes")

        total = float(len(sample_metrics))
        collision_count = sum(1.0 for item in sample_metrics if item["has_collision"])
        success_count = sum(1.0 for item in sample_metrics if item["success"])
        total_segment_collision_steps = sum(float(item["segment_collision_steps"]) for item in sample_metrics)
        total_segment_steps = sum(float(item["segment_steps"]) for item in sample_metrics)
        mean_goal_error = sum(float(item["goal_error_m"]) for item in sample_metrics) / total
        min_sdf_distances = np.asarray(
            [float(item["min_sdf_distance_m"]) for item in sample_metrics],
            dtype=np.float32,
        )
        pybullet_collision_link_episode_counts: dict[str, float] = {}
        for item in sample_metrics:
            for link_name in item.get("collision_link_names", []):
                _add_count(pybullet_collision_link_episode_counts, str(link_name))
        valid_sdf_mask = ~np.isnan(min_sdf_distances)
        mean_min_sdf_distance_m = (
            float(np.mean(min_sdf_distances[valid_sdf_mask]))
            if np.any(valid_sdf_mask)
            else float("nan")
        )
        log_data = {
            "val_traj_collision_rate": collision_count / total,
            "val_pybullet_collision_rate": collision_count / total,
            "val_segment_collision_rate": (
                total_segment_collision_steps / total_segment_steps
                if total_segment_steps > 0
                else 0.0
            ),
            "val_mean_min_sdf_distance_m": mean_min_sdf_distance_m,
            "val_pybullet_eval_episodes": total,
            "val_traj_has_valid_sdf_rate": float(np.mean(valid_sdf_mask.astype(np.float32))),
            "val_pybullet_num_candidates": float(self.cfg.num_candidates),
            "val_candidate_any_zero_pen_rate": candidate_any_zero_pen_count / total,
            "val_candidate_all_has_pen_rate": candidate_all_has_pen_count / total,
            "val_candidate_zero_pen_count_mean": _finite_mean(candidate_zero_pen_counts),
            "val_candidate_best_min_sdf_mean": _finite_mean(candidate_best_min_sdf_values),
            "val_candidate_best_min_sdf_p10": _finite_percentile(
                candidate_best_min_sdf_values,
                10.0,
            ),
            "val_selected_has_pen_when_zero_pen_exists_rate": (
                selected_has_pen_when_zero_pen_exists_count / float(zero_pen_exists_count)
                if zero_pen_exists_count > 0
                else 0.0
            ),
            "val_raw_identical_candidate_episode_rate": (
                raw_identical_candidate_episode_count / total
            ),
            "val_reconstructed_identical_candidate_episode_rate": (
                reconstructed_identical_candidate_episode_count / total
            ),
            "val_selected_candidate_mean_min_sdf_m": float(
                np.mean(np.asarray(selected_candidate_min_sdf, dtype=np.float32))
            ),
            "val_selected_candidate_mean_sdf_gain_m": (
                float(
                    np.mean(
                        np.asarray(
                            selected_candidate_min_sdf_gains,
                            dtype=np.float32,
                        )
                    )
                )
                if selected_candidate_min_sdf_gains
                else float("nan")
            ),
            "val_selected_candidate_mean_has_pen": float(
                np.mean(np.asarray(selected_candidate_has_pen_scores, dtype=np.float32))
            ),
            "val_selected_candidate_mean_pen_step_count": float(
                np.mean(np.asarray(selected_candidate_pen_step_counts, dtype=np.float32))
            ),
            "val_selected_candidate_mean_pen_point_count": float(
                np.mean(np.asarray(selected_candidate_pen_point_counts, dtype=np.float32))
            ),
            "val_selected_candidate_mean_num_pen": float(
                np.mean(
                    np.asarray(
                        selected_candidate_num_pen_scores,
                        dtype=np.float32,
                    )
                )
            ),
            "val_selected_candidate_mean_neg_min_sdf": float(
                np.mean(
                    np.asarray(
                        selected_candidate_neg_min_sdf_scores,
                        dtype=np.float32,
                    )
                )
            ),
            "val_selected_candidate_mean_neg_worstk_mean": float(
                np.mean(
                    np.asarray(
                        selected_candidate_neg_worstk_mean_scores,
                        dtype=np.float32,
                    )
                )
            ),
            "val_selected_candidate_mean_margin_violation": float(
                np.mean(
                    np.asarray(
                        selected_candidate_margin_violation_scores,
                        dtype=np.float32,
                    )
                )
            ),
            "val_selected_candidate_mean_sdf_finite_ratio": float(
                np.mean(
                    np.asarray(
                        selected_candidate_sdf_finite_ratios,
                        dtype=np.float32,
                    )
                )
            ),
            "val_invalid_candidate_rate": (
                invalid_candidate_count
                / (total * float(self.cfg.num_candidates))
            ),
            "val_selected_candidate_index_mean": float(
                np.mean(np.asarray(selected_candidate_indices, dtype=np.float32))
            ),
        }
        for link_name, link_ratios in selected_candidate_sdf_finite_ratios_by_link.items():
            metric_key = "val_selected_candidate_mean_sdf_finite_ratio_" + _sanitize_metric_suffix(link_name)
            log_data[metric_key] = float(np.mean(np.asarray(link_ratios, dtype=np.float32)))
        for link_name, count in candidate_sdf_pen_link_episode_counts.items():
            metric_key = "val_candidate_sdf_pen_link_episode_rate_" + _sanitize_metric_suffix(link_name)
            log_data[metric_key] = float(count) / total
        for link_name, count in selected_sdf_pen_link_counts.items():
            metric_key = "val_selected_sdf_pen_link_rate_" + _sanitize_metric_suffix(link_name)
            log_data[metric_key] = float(count) / total
        for link_name, count in pybullet_collision_link_episode_counts.items():
            metric_key = "val_pybullet_collision_link_episode_rate_" + _sanitize_metric_suffix(link_name)
            log_data[metric_key] = float(count) / total
        if self.cfg.diffusion_sampling_seed is not None:
            log_data["val_pybullet_diffusion_sampling_seed"] = float(
                self.cfg.diffusion_sampling_seed
            )
        if self.cfg.inference_num_steps is not None:
            log_data["val_pybullet_inference_num_steps"] = float(
                self.cfg.inference_num_steps
            )
        if self.cfg.candidate_scheduler_eta is not None:
            log_data["val_pybullet_candidate_scheduler_eta"] = float(
                self.cfg.candidate_scheduler_eta
            )
        log_data["val_pybullet_candidate_action_noise_std"] = float(
            self.cfg.candidate_action_noise_std
        )
        if self.cfg.candidate_action_noise_clip is not None:
            log_data["val_pybullet_candidate_action_noise_clip"] = float(
                self.cfg.candidate_action_noise_clip
            )
        if self.cfg.log_legacy_pybullet_metrics:
            log_data.update({
                "val_pybullet_success_rate": success_count / total,
                "val_pybullet_mean_goal_error_m": mean_goal_error,
            })
        return log_data
