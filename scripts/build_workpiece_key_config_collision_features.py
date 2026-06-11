import argparse
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.pybullet_validation import (  # noqa: E402
    PyBulletCollisionValidator,
    PyBulletValidationConfig,
    SDFGrid,
    _default_urdf_path,
    _load_joint_limits_from_urdf,
    _resolve_workpiece_file_path,
    _rewrite_urdf_package_uris,
)


JOB_DIR_PATTERN = re.compile(r"^job_(\d{3})$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build workpiece-level collision and minimum-distance features for "
            "all key joint configurations."
        )
    )
    parser.add_argument(
        "--key-config-dir",
        type=str,
        default="analysis_outputs/key_joint_configurations_fps",
        help="Directory containing key joint configuration artifacts from stage two.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs/workpiece_key_config_collision_features",
        help="Directory to save workpiece/key-configuration features.",
    )
    parser.add_argument(
        "--jobs-root",
        type=str,
        default="data/raw_data/jobs",
        help="Directory containing regular workpiece STL folders.",
    )
    parser.add_argument(
        "--simple-jobs-root",
        type=str,
        default="data/raw_data/simple_jobs",
        help="Directory containing simple workpiece STL folders.",
    )
    parser.add_argument(
        "--jobs-sdf-root",
        type=str,
        default=None,
        help="Optional override for regular workpiece SDF folders. Defaults to --jobs-root.",
    )
    parser.add_argument(
        "--simple-sdf-root",
        type=str,
        required=True,
        help="External directory containing simple workpiece SDF folders.",
    )
    parser.add_argument(
        "--simple-workpiece-id-offset",
        type=int,
        default=1000,
        help="Offset used to encode simple workpiece ids.",
    )
    parser.add_argument(
        "--d-safe",
        type=float,
        default=0.001,
        help="Safety distance threshold in meters for collision flag computation.",
    )
    parser.add_argument(
        "--sdf-out-of-bounds-value-m",
        type=float,
        default=1.0,
        help=(
            "Fallback SDF distance in meters for robot sample points outside the SDF grid. "
            "Use a positive value larger than d_safe to avoid NaN failures for far-away configurations."
        ),
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Optional robot URDF path.",
    )
    parser.add_argument(
        "--urdf-package-roots",
        type=str,
        nargs="+",
        default=["config/robot-model"],
        help="Package roots used to resolve URDF mesh URIs.",
    )
    parser.add_argument(
        "--tcp-link-name",
        type=str,
        default="tool0",
        help="TCP link name in the URDF.",
    )
    parser.add_argument(
        "--stl-x-offset-m",
        type=float,
        default=0.5,
        help="Translation applied to workpiece STL bodies in PyBullet.",
    )
    parser.add_argument(
        "--robot-surface-points-per-link",
        type=int,
        default=256,
        help="Number of deterministic robot collision surface points retained per link.",
    )
    parser.add_argument(
        "--sdf-query-link-names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional robot link subset for SDF d_min queries. By default all robot "
            "collision links are used, matching the full-arm minimum-distance definition."
        ),
    )
    parser.add_argument(
        "--workpiece-filename",
        type=str,
        default="workpiece.stl",
        help="Workpiece mesh filename inside each job directory.",
    )
    parser.add_argument(
        "--sdf-filename",
        type=str,
        default="workpiece_sdf.npz",
        help="Workpiece SDF filename inside each SDF directory.",
    )
    parser.add_argument(
        "--job-name-template",
        type=str,
        default="job_{workpiece_id:03d}",
        help="Template used to resolve workpiece ids to job folder names.",
    )
    return parser


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def load_key_config_artifacts(key_config_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict]:
    raw_path = key_config_dir / "key_joint_configurations_raw.npy"
    idx_path = key_config_dir / "key_joint_configuration_indices.npy"
    manifest_path = key_config_dir / "manifest.json"
    missing = [
        path.name
        for path in (raw_path, idx_path, manifest_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required key-configuration artifacts in {key_config_dir}: {missing}"
        )

    key_configs_raw = np.asarray(np.load(raw_path), dtype=np.float32)
    key_config_indices = np.asarray(np.load(idx_path), dtype=np.int64).reshape(-1)
    with open(manifest_path, "r", encoding="utf-8") as f:
        key_manifest = json.load(f)

    if key_configs_raw.ndim != 2 or key_configs_raw.shape[1] != 6:
        raise ValueError(
            f"key_joint_configurations_raw.npy must have shape [K, 6], got {key_configs_raw.shape}"
        )
    if key_config_indices.shape[0] != key_configs_raw.shape[0]:
        raise ValueError(
            "key configuration index count must match key configuration count, "
            f"got {key_config_indices.shape[0]} and {key_configs_raw.shape[0]}"
        )
    return key_configs_raw, key_config_indices, key_manifest


def list_canonical_job_dirs(root: pathlib.Path) -> list[tuple[int, str, pathlib.Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Workpiece root does not exist: {root}")
    entries: list[tuple[int, str, pathlib.Path]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = JOB_DIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        local_id = int(match.group(1))
        entries.append((local_id, path.name, path.resolve()))
    return entries


def build_workpiece_table(
    jobs_root: pathlib.Path,
    simple_jobs_root: pathlib.Path,
    simple_workpiece_id_offset: int,
) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for local_id, name, path in list_canonical_job_dirs(jobs_root):
        table.append(
            {
                "workpiece_id": int(local_id),
                "local_id": int(local_id),
                "name": name,
                "type": "regular",
                "job_dir": path,
            }
        )
    for local_id, name, path in list_canonical_job_dirs(simple_jobs_root):
        table.append(
            {
                "workpiece_id": int(simple_workpiece_id_offset + local_id),
                "local_id": int(local_id),
                "name": name,
                "type": "simple",
                "job_dir": path,
            }
        )
    return table


class WorkpieceKeyConfigEvaluator(PyBulletCollisionValidator):
    def __init__(
        self,
        cfg: PyBulletValidationConfig,
        jobs_sdf_root: str,
        simple_sdf_root: str,
    ):
        self.jobs_sdf_root = str(pathlib.Path(jobs_sdf_root).expanduser().resolve())
        self.simple_sdf_root = str(pathlib.Path(simple_sdf_root).expanduser().resolve())
        self.cfg = cfg
        if not self.cfg.enabled:
            raise ValueError("WorkpieceKeyConfigEvaluator should only be created when enabled=True.")

        try:
            import pybullet as pb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pybullet is required to build workpiece/key-configuration collision features."
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
        for joint_idx in range(pb.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_info = pb.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client_id)
            joint_type = joint_info[2]
            child_link_name = joint_info[12].decode("utf-8")
            self.link_name_to_index[child_link_name] = joint_idx
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
        if self.cfg.robot_surface_points_per_link <= 0:
            raise ValueError(
                "robot_surface_points_per_link must be positive, "
                f"got {self.cfg.robot_surface_points_per_link}"
            )
        self.workpiece_cache = {}
        self.sdf_cache = {}
        self.robot_surface_points_by_link = self._build_robot_collision_surface_points(
            resolved_urdf_path=self.resolved_urdf_path,
            points_per_link=self.cfg.robot_surface_points_per_link,
        )
        requested_sdf_query_link_names = list(
            getattr(self.cfg, "sdf_query_link_names", None) or []
        )
        self.sdf_query_link_indices = self._resolve_sdf_query_link_indices(
            requested_link_names=requested_sdf_query_link_names
        )
        if not self.sdf_query_link_indices:
            raise ValueError(
                "No valid SDF query links were resolved from "
                f"{requested_sdf_query_link_names}. Available collision links: "
                f"{sorted(self._collision_link_names_with_points())}"
            )

    def _collision_link_names_with_points(self) -> list[str]:
        names = []
        reverse = {index: name for name, index in self.link_name_to_index.items()}
        for link_index in self.robot_surface_points_by_link.keys():
            if link_index == -1:
                names.append("base_link")
            elif link_index in reverse:
                names.append(reverse[link_index])
        return sorted(set(names))

    def _resolve_sdf_query_link_indices(self, requested_link_names: list[str]) -> list[int]:
        if not requested_link_names:
            return list(self.robot_surface_points_by_link.keys())
        resolved = []
        for name in requested_link_names:
            if name == "base_link":
                link_index = -1
            elif name not in self.link_name_to_index:
                continue
            else:
                link_index = self.link_name_to_index[name]
            if link_index in self.robot_surface_points_by_link:
                resolved.append(link_index)
        ordered_unique = []
        seen = set()
        for idx in resolved:
            if idx in seen:
                continue
            seen.add(idx)
            ordered_unique.append(idx)
        return ordered_unique

    def _robot_surface_points_world_for_sdf(self) -> np.ndarray:
        world_points = []
        for link_index in self.sdf_query_link_indices:
            local_points = self.robot_surface_points_by_link[link_index]
            position, orientation = self._get_link_pose(link_index)
            rotation = np.asarray(
                self.pb.getMatrixFromQuaternion(orientation),
                dtype=np.float32,
            ).reshape(3, 3)
            world_points.append(local_points @ rotation.T + position.reshape(1, 3))
        if not world_points:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(world_points, axis=0).astype(np.float32)

    def _min_sdf_distance_for_current_robot_state(self, sdf_grid):
        if sdf_grid is None:
            return float("nan")
        robot_points = self._robot_surface_points_world_for_sdf()
        if robot_points.size == 0:
            return float("nan")
        sdf_values = sdf_grid.query(robot_points)
        if np.all(np.isnan(sdf_values)):
            return float("nan")
        return float(np.nanmin(sdf_values))

    def _load_workpiece_sdf(self, workpiece_id: int):
        workpiece_id = int(workpiece_id)
        if workpiece_id in self.sdf_cache:
            return self.sdf_cache[workpiece_id]
        sdf_path = _resolve_workpiece_file_path(
            workpiece_id=workpiece_id,
            jobs_root=self.jobs_sdf_root,
            simple_jobs_root=self.simple_sdf_root,
            simple_workpiece_id_offset=self.cfg.simple_workpiece_id_offset,
            job_name_template=self.cfg.job_name_template,
            filename=self.cfg.sdf_filename,
            file_label="SDF",
        )
        sdf_grid = self.sdf_cache[workpiece_id] = SDFGrid.load(
            sdf_path,
            out_of_bounds_value_m=self.cfg.sdf_out_of_bounds_value_m,
        )
        return sdf_grid

    def evaluate_single_configuration(
        self,
        workpiece_id: int,
        joint_state: np.ndarray,
        d_safe: float,
    ) -> tuple[float, float]:
        workpiece_body_id = self._load_workpiece_body(workpiece_id)
        sdf_grid = self._load_workpiece_sdf(workpiece_id)
        joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
        self._set_robot_joints(joint_state)
        self.pb.performCollisionDetection(physicsClientId=self.client_id)
        contacts = self.pb.getClosestPoints(
            bodyA=self.robot_id,
            bodyB=workpiece_body_id,
            distance=0.0,
            physicsClientId=self.client_id,
        )
        mesh_collision = bool(len(contacts) > 0)
        d_min = float(self._min_sdf_distance_for_current_robot_state(sdf_grid))
        if np.isnan(d_min):
            raise ValueError(
                f"SDF query returned NaN for workpiece_id={workpiece_id}. "
                "This likely means all robot collision sample points fell outside the SDF bounds. "
                "Consider setting --sdf-out-of-bounds-value-m to a positive fallback distance."
            )
        collision_flag = float(mesh_collision or (d_min < float(d_safe)))
        return collision_flag, d_min


def build_validator(args: argparse.Namespace) -> WorkpieceKeyConfigEvaluator:
    jobs_root = pathlib.Path(args.jobs_root).expanduser().resolve()
    simple_jobs_root = pathlib.Path(args.simple_jobs_root).expanduser().resolve()
    # jobs_sdf_root and simple_sdf_root already validated during SDF filtering above
    simple_sdf_root = pathlib.Path(args.simple_sdf_root).expanduser().resolve()
    jobs_sdf_root = pathlib.Path(args.jobs_sdf_root).expanduser().resolve() if args.jobs_sdf_root else jobs_root

    cfg = PyBulletValidationConfig(
        enabled=True,
        stats_path="unused_for_single_configuration_features",
        stats_mode="auto",
        jobs_root=str(jobs_root),
        simple_jobs_root=str(simple_jobs_root),
        simple_workpiece_id_offset=int(args.simple_workpiece_id_offset),
        job_name_template=str(args.job_name_template),
        workpiece_filename=str(args.workpiece_filename),
        urdf_path=args.urdf_path,
        urdf_package_roots=tuple(args.urdf_package_roots),
        tcp_link_name=str(args.tcp_link_name),
        stl_x_offset_m=float(args.stl_x_offset_m),
        collision_distance_threshold=0.0,
        interpolate_for_collision=False,
        max_joint_step_rad=0.01,
        min_interpolated_steps_per_segment=1,
        goal_position_norm_m=0.1,
        goal_tolerance_m=0.01,
        num_control_points=12,
        spline_degree=5,
        target_steps=64,
        max_episodes=None,
        sdf_filename=str(args.sdf_filename),
        sdf_required=True,
        robot_surface_points_per_link=int(args.robot_surface_points_per_link),
        sdf_out_of_bounds_value_m=float(args.sdf_out_of_bounds_value_m),
        log_legacy_pybullet_metrics=False,
    )
    setattr(
        cfg,
        "sdf_query_link_names",
        None if args.sdf_query_link_names is None else tuple(args.sdf_query_link_names),
    )
    return WorkpieceKeyConfigEvaluator(
        cfg=cfg,
        jobs_sdf_root=str(jobs_sdf_root),
        simple_sdf_root=str(simple_sdf_root),
    )


def build_manifest(
    args: argparse.Namespace,
    output_dir: pathlib.Path,
    workpiece_table: list[dict[str, object]],
    key_configs_raw: np.ndarray,
    key_manifest: dict,
) -> dict[str, object]:
    regular_count = sum(1 for item in workpiece_table if item["type"] == "regular")
    simple_count = sum(1 for item in workpiece_table if item["type"] == "simple")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "key_config_dir": str(pathlib.Path(args.key_config_dir).expanduser().resolve()),
        "output_dir": str(output_dir.resolve()),
        "jobs_root": str(pathlib.Path(args.jobs_root).expanduser().resolve()),
        "simple_jobs_root": str(pathlib.Path(args.simple_jobs_root).expanduser().resolve()),
        "jobs_sdf_root": str(
            pathlib.Path(args.jobs_sdf_root).expanduser().resolve()
            if args.jobs_sdf_root
            else pathlib.Path(args.jobs_root).expanduser().resolve()
        ),
        "simple_sdf_root": str(pathlib.Path(args.simple_sdf_root).expanduser().resolve()),
        "d_safe_m": float(args.d_safe),
        "sdf_out_of_bounds_value_m": float(args.sdf_out_of_bounds_value_m),
        "sdf_query_link_names": (
            "all_collision_links"
            if args.sdf_query_link_names is None
            else list(args.sdf_query_link_names)
        ),
        "simple_workpiece_id_offset": int(args.simple_workpiece_id_offset),
        "workpiece_count": len(workpiece_table),
        "regular_workpiece_count": regular_count,
        "simple_workpiece_count": simple_count,
        "num_key_configs": int(key_configs_raw.shape[0]),
        "feature_shape": [len(workpiece_table), int(key_configs_raw.shape[0]), 3],
        "feature_layout": {
            "0": "collision_flag_float32",
            "1": "d_min_m_float32",
            "2": "safety_flag_float32",
        },
        "collision_rule": "unsafe = mesh_collision OR d_min <= d_safe",
        "key_config_manifest_created_at_utc": key_manifest.get("created_at_utc"),
    }



def filter_workpieces_with_sdf(
    workpiece_table: list[dict[str, object]],
    jobs_sdf_root: pathlib.Path,
    simple_sdf_root: pathlib.Path,
    simple_workpiece_id_offset: int,
    job_name_template: str,
    sdf_filename: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Filter out workpieces that do not have an SDF file."""
    kept: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    for item in workpiece_table:
        workpiece_id = int(item["workpiece_id"])
        if workpiece_id >= simple_workpiece_id_offset:
            root = simple_sdf_root
            local_id = workpiece_id - simple_workpiece_id_offset
        else:
            root = jobs_sdf_root
            local_id = workpiece_id
        job_name = job_name_template.format(workpiece_id=int(local_id))
        sdf_path = root / job_name / sdf_filename
        if sdf_path.is_file():
            kept.append(item)
        else:
            dropped.append(item)
    return kept, dropped

def main() -> None:
    args = build_parser().parse_args()
    if args.d_safe < 0:
        raise ValueError(f"d_safe must be non-negative, got {args.d_safe}")

    key_config_dir = pathlib.Path(args.key_config_dir).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    key_configs_raw, key_config_indices, key_manifest = load_key_config_artifacts(key_config_dir)
    workpiece_table = build_workpiece_table(
        jobs_root=pathlib.Path(args.jobs_root).expanduser().resolve(),
        simple_jobs_root=pathlib.Path(args.simple_jobs_root).expanduser().resolve(),
        simple_workpiece_id_offset=int(args.simple_workpiece_id_offset),
    )
    if not workpiece_table:
        raise ValueError("No canonical workpiece directories were found.")

    # Filter workpieces: only keep those with SDF files
    jobs_sdf_root = pathlib.Path(args.jobs_sdf_root).expanduser().resolve() if args.jobs_sdf_root else pathlib.Path(args.jobs_root).expanduser().resolve()
    simple_sdf_root = pathlib.Path(args.simple_sdf_root).expanduser().resolve()
    workpiece_table, dropped_table = filter_workpieces_with_sdf(
        workpiece_table=workpiece_table,
        jobs_sdf_root=jobs_sdf_root,
        simple_sdf_root=simple_sdf_root,
        simple_workpiece_id_offset=int(args.simple_workpiece_id_offset),
        job_name_template=str(args.job_name_template),
        sdf_filename=str(args.sdf_filename),
    )
    if dropped_table:
        dropped_ids = sorted([int(item["workpiece_id"]) for item in dropped_table])
        print(f"[INFO] Dropped {len(dropped_table)} workpiece(s) without SDF: {dropped_ids}")
    if not workpiece_table:
        raise ValueError("No workpieces remain after SDF filtering.")

    validator = build_validator(args)
    try:
        features = np.empty((len(workpiece_table), key_configs_raw.shape[0], 3), dtype=np.float32)
        total_evaluations = len(workpiece_table) * key_configs_raw.shape[0]
        evaluated_count = 0
        collision_count = 0
        running_min_d_min = float("inf")
        workpiece_iter = progress(
            enumerate(workpiece_table),
            total=len(workpiece_table),
            desc="Evaluating workpieces",
            unit="workpiece",
        )
        for workpiece_idx, item in workpiece_iter:
            workpiece_id = int(item["workpiece_id"])
            workpiece_collision_count = 0
            workpiece_min_d_min = float("inf")
            key_iter = progress(
                enumerate(key_configs_raw),
                total=key_configs_raw.shape[0],
                desc=f"{item['type']}:{item['name']}",
                unit="keycfg",
                leave=False,
            )
            for key_idx, joint_state in key_iter:
                collision_flag, d_min = validator.evaluate_single_configuration(
                    workpiece_id=workpiece_id,
                    joint_state=joint_state,
                    d_safe=float(args.d_safe),
                )
                features[workpiece_idx, key_idx, 0] = np.float32(collision_flag)
                features[workpiece_idx, key_idx, 1] = np.float32(d_min)
                safety_flag = int(collision_flag > 0.5) or int(d_min <= float(args.d_safe))
                features[workpiece_idx, key_idx, 2] = np.float32(safety_flag)
                evaluated_count += 1
                collision_count += int(collision_flag > 0.5)
                workpiece_collision_count += int(collision_flag > 0.5)
                running_min_d_min = min(running_min_d_min, float(d_min))
                workpiece_min_d_min = min(workpiece_min_d_min, float(d_min))

                global_collision_rate = collision_count / evaluated_count if evaluated_count > 0 else 0.0
                workpiece_collision_rate = (
                    workpiece_collision_count / (key_idx + 1)
                    if key_idx + 1 > 0 else 0.0
                )
                if hasattr(key_iter, "set_postfix"):
                    key_iter.set_postfix(
                        collision_rate=f"{workpiece_collision_rate:.3f}",
                        min_d_min=f"{workpiece_min_d_min:.6f}",
                    )
                if hasattr(workpiece_iter, "set_postfix"):
                    workpiece_iter.set_postfix(
                        global_collision_rate=f"{global_collision_rate:.3f}",
                        min_d_min=f"{running_min_d_min:.6f}",
                        done=f"{evaluated_count}/{total_evaluations}",
                    )
    finally:
        validator.close()

    workpiece_ids = np.asarray([int(item["workpiece_id"]) for item in workpiece_table], dtype=np.int64)
    workpiece_types = np.asarray([str(item["type"]) for item in workpiece_table], dtype="<U16")
    workpiece_names = np.asarray([str(item["name"]) for item in workpiece_table], dtype="<U64")

    np.save(output_dir / "workpiece_key_config_features.npy", features)
    np.save(output_dir / "workpiece_ids.npy", workpiece_ids)
    np.save(output_dir / "workpiece_types.npy", workpiece_types)
    np.save(output_dir / "workpiece_names.npy", workpiece_names)
    np.save(output_dir / "key_config_indices.npy", key_config_indices.astype(np.int64))

    manifest = build_manifest(
        args=args,
        output_dir=output_dir,
        workpiece_table=workpiece_table,
        key_configs_raw=key_configs_raw,
        key_manifest=key_manifest,
    )
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"workpiece_count: {len(workpiece_table)}")
    print(f"num_key_configs: {key_configs_raw.shape[0]}")
    print(f"feature_shape: {features.shape}")
    print(f"saved_output_dir: {output_dir}")


if __name__ == "__main__":
    main()
