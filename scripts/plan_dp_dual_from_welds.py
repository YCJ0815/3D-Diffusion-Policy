#!/usr/bin/env python3
"""End-to-end workpiece-to-trajectory entry point for DP-Dual.

The command intentionally reuses the repository's existing implementations:

1. ``weld-robot/data_generation/seam_extract`` (through its maintained worker)
   extracts weld segments and endpoint surface normals from STEP/STP CAD.
2. ``weld-robot/scripts/workpiece_sdf.py`` voxelizes the placed STL into SDF.
3. ``weld-robot/scripts/rrt_welding_planning_demo.py`` supplies TCP-frame and
   UR5e inverse-kinematics conventions.
4. ``infer_bspline_trajectories_batch.py`` runs DP-Dual mode
   (guided diffusion plus the final QP repair).

STL is used for SDF/collision geometry.  STEP/STP is required separately for
automatic seam extraction because STL does not preserve CAD face topology.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DP_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = DP_ROOT.parents[1]
WELD_ROOT = Path("/root/autodl-tmp/3D-Diffusion-Policy")  # WORKSPACE_ROOT / "weld-robot"
WELD_SCRIPTS = WELD_ROOT / "scripts"
SEAM_WORKER = Path("/root/weld-robot/data_generation/src/main.py")
DP_BATCH_SCRIPT = Path("/root/autodl-tmp/3D-Diffusion-Policy/scripts/infer_bspline_trajectories_batch.py")
CSPACE_BUILD_SCRIPT = Path("/root/autodl-tmp/3D-Diffusion-Policy/scripts/build_workpiece_key_config_collision_features.py")
DEFAULT_STATS = Path("/root/autodl-tmp/3D-Diffusion-Policy/data/raw_data/realdex_bspline_stats_free10.npz")
DEFAULT_KEY_CONFIG_DIR = DP_ROOT / "analysis_outputs" / "key_joint_configurations_fps"
DEFAULT_URDF = Path("/root/autodl-tmp/3D-Diffusion-Policy/config/robot-model/ur5e_with_pen.urdf")


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _env_path(name: str, fallback: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    return _path(value) if value else fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract welds, build a placed-workpiece SDF, infer endpoint poses/IK, "
            "and run DP-Dual trajectory planning."
        )
    )
    parser.add_argument("--workpiece-stl", type=_path, required=True, help="STL used for SDF and collision checks.")
    parser.add_argument(
        "--cad-step",
        type=_path,
        required=True,
        help="STEP/STP model passed directly to data_generation/seam_extract for weld extraction.",
    )
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--point-frame",
        choices=("workpiece", "world"),
        default="workpiece",
        help="Frame of --start/--goal. Workpiece coordinates use the CAD/STL source unit.",
    )
    parser.add_argument(
        "--workpiece-position",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Arbitrary XYZ translation of the workpiece in robot-world meters.",
    )
    parser.add_argument(
        "--geometry-unit-scale",
        type=float,
        default=0.001,
        help="Source CAD/STL unit to meters (default 0.001 for millimetres).",
    )
    parser.add_argument("--output-dir", type=_path, default=_path("outputs/dp_dual_plan"))
    parser.add_argument("--checkpoint-path", type=_path, default=_env_path("DP_DUAL_CHECKPOINT"))
    parser.add_argument("--stats-path", type=_path, default=_env_path("DP_DUAL_STATS", DEFAULT_STATS))
    parser.add_argument("--device", default=os.environ.get("DP_DUAL_DEVICE", "cuda:0"))
    parser.add_argument("--dp-python", type=_path, default=_path(sys.executable))
    parser.add_argument(
        "--cspace-python",
        type=_path,
        default=_env_path("DP_DUAL_CSPACE_PYTHON"),
        help="Python environment containing pybullet. Defaults to --dp-python.",
    )
    parser.add_argument(
        "--extract-python",
        type=_path,
        default=_env_path("WELD_EXTRACT_PYTHON", _path(sys.executable)),
        help="Python environment containing pythonocc-core for seam_extract.",
    )
    parser.add_argument("--urdf-path", type=_path, default=DEFAULT_URDF.resolve())
    parser.add_argument("--pose-normal-tol", type=float, default=1e-2)
    parser.add_argument("--max-seam-distance", type=float, default=5.0, help="Maximum start/goal-to-seam distance in CAD units.")
    parser.add_argument(
        "--keep-input-points",
        action="store_true",
        help="Keep start/goal coordinates instead of snapping them to the nearest extracted seam.",
    )
    parser.add_argument("--tcp-normal-offset", type=float, default=0.0, help="TCP offset along the seam normal in meters.")
    parser.add_argument("--endpoint-retreat-step", type=float, default=0.005)
    parser.add_argument("--endpoint-retreat-max-steps", type=int, default=8)
    parser.add_argument("--endpoint-random-seeds", type=int, default=24)
    parser.add_argument("--ik-max-iters", type=int, default=300)
    parser.add_argument("--ik-rotation-weight", type=float, default=0.35)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sdf-voxel-pitch", type=float, default=0.004)
    parser.add_argument("--sdf-lateral-margin", type=float, default=0.05)
    parser.add_argument("--sdf-bottom-margin", type=float, default=0.03)
    parser.add_argument("--sdf-top-margin", type=float, default=0.10)
    parser.add_argument("--sdf-penetration-tol", type=float, default=-0.001)
    parser.add_argument("--rebuild-sdf", action="store_true")
    parser.add_argument("--target-steps", type=int, default=64)
    parser.add_argument("--num-control-points", type=int, default=16)
    parser.add_argument("--num-output-points", type=int, default=1024)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument(
        "--key-config-dir",
        type=_path,
        default=_env_path("DP_DUAL_KEY_CONFIG_DIR", DEFAULT_KEY_CONFIG_DIR.resolve()),
        help="Training-aligned 128 key configurations used to generate this workpiece's C-space feature.",
    )
    parser.add_argument(
        "--cspace-feature-dir",
        type=_path,
        default=None,
        help="Explicit prebuilt feature directory. When set, automatic generation is skipped.",
    )
    parser.add_argument(
        "--cspace-generation",
        choices=("on", "off"),
        default="on",
        help="Generate a matching C-space feature after SDF construction (default: on).",
    )
    parser.add_argument("--cspace-d-safe", type=float, default=0.01)
    parser.add_argument("--cspace-robot-points-per-link", type=int, default=256)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after seam extraction, SDF, pose/IK, and DP-Dual input preparation.",
    )
    return parser


def _require_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"{label} is not configured")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _run(command: list[str], cwd: Path, label: str) -> None:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_seam_extract(args: argparse.Namespace, seam_root: Path) -> Path:
    """Invoke the existing seam_extract pipeline through its subprocess worker."""
    extract_dir = seam_root / "intermediate"
    final_dir = seam_root / "final"
    vector_dir = seam_root / "vectors"
    command = [
        str(_require_file(args.extract_python, "seam_extract Python")),
        str(_require_file(SEAM_WORKER, "seam_extract worker")),
        "--extract-worker",
        "--step-file",
        str(args.cad_step),
        "--extract-dir",
        str(extract_dir),
        "--final-dir",
        str(final_dir),
        "--vector-dir",
        str(vector_dir),
        "--pose-normal-tol",
        str(args.pose_normal_tol),
    ]
    _run(command, WELD_ROOT / "data_generation", "seam_extract")
    return _require_file(vector_dir / f"{args.cad_step.stem}_weld_vectors.json", "weld vector output")


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return np.asarray(vector, dtype=float) / norm


def fallback_normal(tangent: np.ndarray) -> np.ndarray:
    tangent = normalize(tangent)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, reference))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    return normalize(np.cross(tangent, reference))


def _endpoint_normal(endpoint: dict[str, Any], tangent: np.ndarray) -> np.ndarray:
    pose = endpoint.get("pose")
    if pose is None:
        return fallback_normal(tangent)
    normal = np.asarray(pose, dtype=float).reshape(-1)
    return normalize(normal) if normal.size == 3 and np.linalg.norm(normal) > 1e-8 else fallback_normal(tangent)


def match_point_to_weld(point_local: np.ndarray, welds: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for index, weld in enumerate(welds):
        p0 = np.asarray(weld["start"]["xyz"], dtype=float)
        p1 = np.asarray(weld["end"]["xyz"], dtype=float)
        segment = p1 - p0
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            continue
        alpha = float(np.clip(np.dot(point_local - p0, segment) / length_sq, 0.0, 1.0))
        projection = p0 + alpha * segment
        distance = float(np.linalg.norm(point_local - projection))
        tangent = normalize(segment)
        n0 = _endpoint_normal(weld["start"], tangent)
        n1 = _endpoint_normal(weld["end"], tangent)
        if np.dot(n0, n1) < 0.0:
            n1 = -n1
        blended = (1.0 - alpha) * n0 + alpha * n1
        normal = normalize(blended) if np.linalg.norm(blended) > 1e-8 else n0
        candidate = {
            "weld_index": index,
            "alpha": alpha,
            "distance_source_units": distance,
            "requested_local": point_local,
            "projection_local": projection,
            "tangent_local": tangent,
            "normal_local": normal,
        }
        if best is None or distance < best["distance_source_units"]:
            best = candidate
    if best is None:
        raise RuntimeError("seam_extract returned no non-degenerate weld segments")
    return best


def _world_point(local: np.ndarray, scale: float, offset: np.ndarray) -> np.ndarray:
    return np.asarray(local, dtype=float) * float(scale) + np.asarray(offset, dtype=float)


def _local_point(point: np.ndarray, frame: str, scale: float, offset: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    if frame == "world":
        return (point - offset) / scale
    return point


def load_and_match_endpoints(args: argparse.Namespace, vector_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    with vector_path.open("r", encoding="utf-8") as stream:
        welds = json.load(stream).get("welds", [])
    if not welds:
        raise RuntimeError(f"No welds found in {vector_path}")
    offset = np.asarray(args.workpiece_position, dtype=float)
    start_local = _local_point(np.asarray(args.start), args.point_frame, args.geometry_unit_scale, offset)
    goal_local = _local_point(np.asarray(args.goal), args.point_frame, args.geometry_unit_scale, offset)
    start_match = match_point_to_weld(start_local, welds)
    goal_match = match_point_to_weld(goal_local, welds)
    for label, match in (("start", start_match), ("goal", goal_match)):
        if match["distance_source_units"] > args.max_seam_distance:
            raise ValueError(
                f"{label} is {match['distance_source_units']:.6g} source units from its nearest seam, "
                f"exceeding --max-seam-distance={args.max_seam_distance}"
            )
        selected_local = match["requested_local"] if args.keep_input_points else match["projection_local"]
        match["selected_local"] = selected_local
        match["world"] = _world_point(selected_local, args.geometry_unit_scale, offset)
        match["tangent_world"] = normalize(match["tangent_local"])
        match["normal_world"] = normalize(match["normal_local"])
    return welds, start_match, goal_match


def import_weld_planning_modules() -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(WELD_SCRIPTS))
    import rrt_welding_planning_demo as welding_demo
    import sdf_trajopt
    import sim_welding_arm
    import workpiece_sdf

    return welding_demo, sdf_trajopt, sim_welding_arm, workpiece_sdf


def solve_endpoint(
    *,
    label: str,
    match: dict[str, Any],
    kinematics: Any,
    evaluator: Any,
    frame_builder: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
    extra_seeds: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=float)
    seeds = [home, np.zeros(6, dtype=float)]
    seeds.extend(extra_seeds or [])
    for _ in range(args.endpoint_random_seeds):
        seeds.append(rng.uniform(kinematics.lower, kinematics.upper))

    failures: list[str] = []
    for retreat_step in range(args.endpoint_retreat_max_steps + 1):
        tcp_offset = args.tcp_normal_offset + retreat_step * args.endpoint_retreat_step
        target_tf = frame_builder(
            match["world"],
            match["normal_world"],
            match["tangent_world"],
            tcp_offset,
        )
        try:
            q = kinematics.solve_ik(
                target_tf,
                seeds=seeds,
                max_iters=args.ik_max_iters,
                rot_weight=args.ik_rotation_weight,
            )
        except RuntimeError as exc:
            failures.append(f"retreat={tcp_offset:.4f}: {exc}")
            continue
        collision = evaluator.evaluate_state(q)
        if bool(collision["nonpenetrating"]):
            info = {
                "label": label,
                "tcp_normal_offset_m": tcp_offset,
                "collision": collision,
                "joint_radians": q,
                "target_tf": target_tf,
            }
            return q, target_tf, info
        failures.append(
            f"retreat={tcp_offset:.4f}: penetrating arm_min={collision['arm_min']:.5f}, "
            f"tool_min={collision['tool_min']:.5f}"
        )
    raise RuntimeError(f"Could not find a collision-free IK solution for {label}: " + " | ".join(failures[-4:]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2)


def validate_cspace_feature_dir(feature_dir: Path) -> Path:
    feature_path = _require_file(feature_dir / "workpiece_key_config_features.npy", "C-space features")
    ids_path = _require_file(feature_dir / "workpiece_ids.npy", "C-space workpiece IDs")
    features = np.asarray(np.load(feature_path), dtype=np.float32)
    workpiece_ids = np.asarray(np.load(ids_path), dtype=np.int64).reshape(-1)
    if features.shape != (1, 128, 2):
        raise ValueError(
            f"Current-workpiece C-space features must have shape (1, 128, 2), got {features.shape}"
        )
    if workpiece_ids.shape != (1,) or int(workpiece_ids[0]) != 0:
        raise ValueError(f"Current-workpiece C-space IDs must be [0], got {workpiece_ids.tolist()}")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"C-space features contain non-finite values: {feature_path}")
    return feature_dir


def build_cspace_features(
    args: argparse.Namespace,
    jobs_root: Path,
    output_dir: Path,
) -> Path | None:
    if args.cspace_generation == "off":
        return validate_cspace_feature_dir(args.cspace_feature_dir) if args.cspace_feature_dir else None
    if args.cspace_feature_dir is not None:
        print(f"[C-space] Using explicitly supplied features: {args.cspace_feature_dir}")
        return validate_cspace_feature_dir(args.cspace_feature_dir)

    key_config_dir = args.key_config_dir
    required_key_files = (
        "key_joint_configurations_raw.npy",
        "key_joint_configuration_indices.npy",
        "manifest.json",
    )
    missing = [name for name in required_key_files if not (key_config_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot generate C-space features: {key_config_dir} is missing {missing}. "
            "Provide the training-aligned key-configuration directory with --key-config-dir "
            "or set DP_DUAL_KEY_CONFIG_DIR. Use --cspace-generation off only with a non-C-space checkpoint."
        )
    key_configs = np.asarray(np.load(key_config_dir / "key_joint_configurations_raw.npy"))
    if key_configs.shape != (128, 6):
        raise ValueError(
            f"C-space policy requires exactly 128 key configurations with 6 joints, got {key_configs.shape}"
        )

    cspace_output = output_dir / "cspace_features"
    empty_simple_root = output_dir / "_empty_simple_jobs"
    empty_simple_root.mkdir(parents=True, exist_ok=True)
    cspace_python = args.cspace_python or args.dp_python
    command = [
        str(_require_file(cspace_python, "C-space Python")),
        str(_require_file(CSPACE_BUILD_SCRIPT, "C-space feature generator")),
        "--key-config-dir",
        str(key_config_dir),
        "--output-dir",
        str(cspace_output),
        "--jobs-root",
        str(jobs_root),
        "--jobs-sdf-root",
        str(jobs_root),
        "--simple-jobs-root",
        str(empty_simple_root),
        "--simple-sdf-root",
        str(empty_simple_root),
        "--sdf-filename",
        "sdf.npz",
        "--urdf-path",
        str(args.urdf_path),
        "--stl-offset-m",
        *(str(value) for value in args.workpiece_position),
        "--workpiece-mesh-scale-m",
        str(args.geometry_unit_scale),
        "--d-safe",
        str(args.cspace_d_safe),
        "--robot-surface-points-per-link",
        str(args.cspace_robot_points_per_link),
    ]
    _run(command, DP_ROOT, "C-space")
    return validate_cspace_feature_dir(cspace_output)


def prepare_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    _require_file(args.workpiece_stl, "workpiece STL")
    _require_file(args.cad_step, "CAD STEP/STP")
    _require_file(args.urdf_path, "robot URDF")
    if args.geometry_unit_scale <= 0.0:
        raise ValueError("--geometry-unit-scale must be positive")
    if args.target_steps < 2:
        raise ValueError("--target-steps must be at least 2")
    if args.num_control_points < 4:
        raise ValueError("--num-control-points must be at least 4")
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be positive")
    if args.cspace_d_safe < 0.0:
        raise ValueError("--cspace-d-safe must be non-negative")
    if args.cspace_robot_points_per_link < 1:
        raise ValueError("--cspace-robot-points-per-link must be positive")

    output_dir = args.output_dir
    seam_root = output_dir / "seam_extract"
    jobs_root = output_dir / "jobs"
    job_dir = jobs_root / "job_000"
    input_dir = output_dir / "transition_inputs"
    job_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    vector_path = run_seam_extract(args, seam_root)
    shutil.copy2(args.workpiece_stl, job_dir / "workpiece.stl")
    shutil.copy2(vector_path, job_dir / "weld_vectors.json")
    welds, start_match, goal_match = load_and_match_endpoints(args, vector_path)

    welding_demo, sdf_trajopt, sim_welding_arm, workpiece_sdf = import_weld_planning_modules()
    sdf_config = workpiece_sdf.SDFBuildConfig(
        voxel_pitch=args.sdf_voxel_pitch,
        lateral_margin=args.sdf_lateral_margin,
        bottom_margin=args.sdf_bottom_margin,
        top_margin=args.sdf_top_margin,
    )
    sdf_path = job_dir / "sdf.npz"
    sdf_layer, _ = workpiece_sdf.load_or_build_workpiece_sdf(
        stl_path=job_dir / "workpiece.stl",
        scale=args.geometry_unit_scale,
        z_offset=0.0,
        local_offset=tuple(float(v) for v in args.workpiece_position),
        npz_path=sdf_path,
        config=sdf_config,
        logger=print,
        rebuild=args.rebuild_sdf,
    )
    cspace_feature_dir = build_cspace_features(args, jobs_root, output_dir)
    resolved_urdf = sim_welding_arm.make_resolved_urdf(args.urdf_path)
    kinematics = welding_demo.URDFKinematics(resolved_urdf)
    collision_config = sdf_trajopt.SDFTrajOptConfig(penetration_tol=args.sdf_penetration_tol)
    evaluator = sdf_trajopt.KinematicSDFCollisionEvaluator(kinematics, sdf_layer, collision_config)
    rng = np.random.default_rng(args.random_seed)
    q_start, start_tf, start_info = solve_endpoint(
        label="start",
        match=start_match,
        kinematics=kinematics,
        evaluator=evaluator,
        frame_builder=welding_demo.target_frame_from_weld,
        args=args,
        rng=rng,
    )
    q_goal, goal_tf, goal_info = solve_endpoint(
        label="goal",
        match=goal_match,
        kinematics=kinematics,
        evaluator=evaluator,
        frame_builder=welding_demo.target_frame_from_weld,
        args=args,
        rng=rng,
        extra_seeds=[q_start],
    )

    transition_delta = goal_tf[:3, 3] - start_tf[:3, 3]
    if np.linalg.norm(transition_delta) <= 1e-8:
        raise ValueError("Start and goal resolve to the same TCP position; no transition can be planned")
    seed_path = np.linspace(q_start, q_goal, args.target_steps, dtype=np.float32)
    transition_path = input_dir / "transition_0000_0001.npz"
    np.savez_compressed(
        transition_path,
        q_start=q_start.astype(np.float32),
        q_goal=q_goal.astype(np.float32),
        q_plan=seed_path,
        start_tf=start_tf.astype(np.float32),
        goal_tf=goal_tf.astype(np.float32),
        start_xyz=start_tf[:3, 3].astype(np.float32),
        end_xyz=goal_tf[:3, 3].astype(np.float32),
        start_normal=start_match["normal_world"].astype(np.float32),
        end_normal=goal_match["normal_world"].astype(np.float32),
        tangent=normalize(transition_delta).astype(np.float32),
        workpiece_id=np.array(0, dtype=np.int64),
    )
    manifest = {
        "inputs": {
            "workpiece_stl": args.workpiece_stl,
            "cad_step": args.cad_step,
            "start": args.start,
            "goal": args.goal,
            "point_frame": args.point_frame,
            "workpiece_position_m": args.workpiece_position,
            "geometry_unit_scale": args.geometry_unit_scale,
        },
        "outputs": {
            "weld_vectors": vector_path,
            "sdf": sdf_path,
            "transition_npz": transition_path,
            "jobs_root": jobs_root,
            "cspace_feature_dir": cspace_feature_dir,
        },
        "seam_count": len(welds),
        "start_match": start_match,
        "goal_match": goal_match,
        "start_solution": start_info,
        "goal_solution": goal_info,
    }
    manifest_path = output_dir / "planning_request.json"
    write_json(manifest_path, manifest)
    manifest["outputs"]["request_manifest"] = manifest_path
    manifest["kinematics"] = kinematics
    return manifest


def run_dp_dual(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _require_file(args.checkpoint_path, "DP-Dual checkpoint (--checkpoint-path or DP_DUAL_CHECKPOINT)")
    stats = _require_file(args.stats_path, "DP-Dual statistics")
    transition_path = Path(prepared["outputs"]["transition_npz"])
    jobs_root = Path(prepared["outputs"]["jobs_root"])
    dp_output = args.output_dir / "dp_dual"
    command = [
        str(_require_file(args.dp_python, "DP-Dual Python")),
        str(_require_file(DP_BATCH_SCRIPT, "DP-Dual batch planner")),
        "--input-dirs",
        str(transition_path.parent),
        "--checkpoint-path",
        str(checkpoint),
        "--stats-path",
        str(stats),
        "--output-root",
        str(dp_output),
        "--jobs-root",
        str(jobs_root),
        "--device",
        args.device,
        "--sample-source",
        "regular",
        "--sample-count",
        "1",
        "--sampling-mode",
        "baseline",
        "--planner-mode",
        "qp_guided_diffusion_post_qp",
        "--stl-scale-to-m",
        str(args.geometry_unit_scale),
        "--stl-offset-m",
        *(str(value) for value in args.workpiece_position),
        "--urdf-path",
        str(args.urdf_path),
        "--target-steps",
        str(args.target_steps),
        "--num-control-points",
        str(args.num_control_points),
        "--num-output-points",
        str(args.num_output_points),
        "--num-candidates",
        str(args.num_candidates),
    ]
    cspace_feature_dir = prepared["outputs"].get("cspace_feature_dir")
    if cspace_feature_dir is not None:
        command.extend(["--cspace-feature-dir", str(cspace_feature_dir)])
    _run(command, DP_ROOT, "DP-Dual")

    predictions = sorted(dp_output.rglob("pred_joint_horizon.npy"))
    if len(predictions) != 1:
        raise RuntimeError(f"Expected one DP-Dual prediction under {dp_output}, found {len(predictions)}")
    joint_path = predictions[0]
    joints = np.asarray(np.load(joint_path), dtype=float)
    kinematics = prepared["kinematics"]
    tcp_transforms = np.stack([kinematics.forward(q) for q in joints], axis=0)
    tcp_path = joint_path.with_name("pred_tcp_transforms.npy")
    np.save(tcp_path, tcp_transforms)
    result = {
        "joint_trajectory": joint_path,
        "tcp_transforms": tcp_path,
        "summary": joint_path.with_name("summary.json"),
    }
    write_json(args.output_dir / "result.json", result)
    return result


def main() -> None:
    args = build_parser().parse_args()
    try:
        prepared = prepare_pipeline(args)
        print(f"\n[prepared] transition input: {prepared['outputs']['transition_npz']}")
        print(f"[prepared] SDF: {prepared['outputs']['sdf']}")
        if prepared["outputs"].get("cspace_feature_dir") is not None:
            print(f"[prepared] C-space features: {prepared['outputs']['cspace_feature_dir']}")
        if args.prepare_only:
            print("[done] Preparation completed; DP-Dual was not launched (--prepare-only).")
            return
        result = run_dp_dual(args, prepared)
        print(f"\n[done] joint trajectory: {result['joint_trajectory']}")
        print(f"[done] TCP transforms: {result['tcp_transforms']}")
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
