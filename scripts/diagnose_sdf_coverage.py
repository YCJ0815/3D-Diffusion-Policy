import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_feature_builder_module():
    module_path = PROJECT_ROOT / "scripts" / "build_workpiece_key_config_collision_features.py"
    spec = importlib.util.spec_from_file_location("workpiece_feature_builder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether robot collision sample points fall outside workpiece SDF coverage "
            "for the selected key joint configurations."
        )
    )
    parser.add_argument(
        "--key-config-dir",
        type=str,
        default="analysis_outputs/key_joint_configurations_fps",
        help="Directory containing key joint configurations.",
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
    parser.add_argument(
        "--max-workpieces",
        type=int,
        default=None,
        help="Optional cap on the number of workpieces to diagnose.",
    )
    parser.add_argument(
        "--max-key-configs",
        type=int,
        default=None,
        help="Optional cap on the number of key configurations to diagnose.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="analysis_outputs/sdf_coverage_diagnosis.json",
        help="Path to save the diagnosis summary JSON.",
    )
    return parser


def points_inside_sdf_bounds(points: np.ndarray, sdf_grid) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return (
        (points[:, 0] >= float(sdf_grid.x[0])) & (points[:, 0] <= float(sdf_grid.x[-1])) &
        (points[:, 1] >= float(sdf_grid.y[0])) & (points[:, 1] <= float(sdf_grid.y[-1])) &
        (points[:, 2] >= float(sdf_grid.z[0])) & (points[:, 2] <= float(sdf_grid.z[-1]))
    )


def main() -> None:
    args = build_parser().parse_args()
    feature_builder = load_feature_builder_module()

    key_configs_raw, key_config_indices, _ = feature_builder.load_key_config_artifacts(
        pathlib.Path(args.key_config_dir).expanduser().resolve()
    )
    if args.max_key_configs is not None:
        key_configs_raw = key_configs_raw[: int(args.max_key_configs)]
        key_config_indices = key_config_indices[: int(args.max_key_configs)]

    workpiece_table = feature_builder.build_workpiece_table(
        jobs_root=pathlib.Path(args.jobs_root).expanduser().resolve(),
        simple_jobs_root=pathlib.Path(args.simple_jobs_root).expanduser().resolve(),
        simple_workpiece_id_offset=int(args.simple_workpiece_id_offset),
    )
    if args.max_workpieces is not None:
        workpiece_table = workpiece_table[: int(args.max_workpieces)]

    validator = feature_builder.build_validator(args)
    try:
        fallback_matrix = np.zeros((len(workpiece_table), key_configs_raw.shape[0]), dtype=bool)
        inside_ratio_matrix = np.zeros((len(workpiece_table), key_configs_raw.shape[0]), dtype=np.float32)
        pair_records = []

        workpiece_iter = feature_builder.progress(
            enumerate(workpiece_table),
            total=len(workpiece_table),
            desc="Diagnosing workpieces",
            unit="workpiece",
        )
        for workpiece_idx, item in workpiece_iter:
            workpiece_id = int(item["workpiece_id"])
            sdf_grid = validator._load_workpiece_sdf(workpiece_id)
            sdf_min = np.array([sdf_grid.x[0], sdf_grid.y[0], sdf_grid.z[0]], dtype=np.float32)
            sdf_max = np.array([sdf_grid.x[-1], sdf_grid.y[-1], sdf_grid.z[-1]], dtype=np.float32)

            key_iter = feature_builder.progress(
                enumerate(key_configs_raw),
                total=key_configs_raw.shape[0],
                desc=f"{item['type']}:{item['name']}",
                unit="keycfg",
                leave=False,
            )
            workpiece_fallbacks = 0
            for key_idx, joint_state in key_iter:
                validator._set_robot_joints(joint_state)
                robot_points = validator._robot_surface_points_world()
                inside_mask = points_inside_sdf_bounds(robot_points, sdf_grid)
                inside_ratio = float(np.mean(inside_mask.astype(np.float32)))
                all_outside = bool(not np.any(inside_mask))
                fallback_matrix[workpiece_idx, key_idx] = all_outside
                inside_ratio_matrix[workpiece_idx, key_idx] = inside_ratio
                workpiece_fallbacks += int(all_outside)

                robot_min = robot_points.min(axis=0).astype(np.float32)
                robot_max = robot_points.max(axis=0).astype(np.float32)
                if all_outside or inside_ratio < 0.1:
                    pair_records.append(
                        {
                            "workpiece_id": workpiece_id,
                            "workpiece_type": str(item["type"]),
                            "workpiece_name": str(item["name"]),
                            "key_config_order": int(key_idx),
                            "key_config_pool_index": int(key_config_indices[key_idx]),
                            "inside_ratio": inside_ratio,
                            "all_points_outside": all_outside,
                            "robot_bbox_min": robot_min.tolist(),
                            "robot_bbox_max": robot_max.tolist(),
                            "sdf_bbox_min": sdf_min.tolist(),
                            "sdf_bbox_max": sdf_max.tolist(),
                        }
                    )
                if hasattr(key_iter, "set_postfix"):
                    key_iter.set_postfix(
                        fallback_rate=f"{workpiece_fallbacks / (key_idx + 1):.3f}",
                        inside_ratio=f"{inside_ratio:.3f}",
                    )

        per_workpiece_fallback = fallback_matrix.mean(axis=1)
        per_key_fallback = fallback_matrix.mean(axis=0)
        workpiece_types = np.asarray([str(item["type"]) for item in workpiece_table], dtype="<U16")
        regular_mask = workpiece_types == "regular"
        simple_mask = workpiece_types == "simple"
        summary = {
            "workpiece_count": len(workpiece_table),
            "key_config_count": int(key_configs_raw.shape[0]),
            "overall_fallback_ratio": float(fallback_matrix.mean()),
            "overall_mean_inside_ratio": float(inside_ratio_matrix.mean()),
            "regular_fallback_ratio": (
                float(fallback_matrix[regular_mask].mean()) if np.any(regular_mask) else None
            ),
            "simple_fallback_ratio": (
                float(fallback_matrix[simple_mask].mean()) if np.any(simple_mask) else None
            ),
            "top_fallback_workpieces": [
                {
                    "workpiece_id": int(workpiece_table[idx]["workpiece_id"]),
                    "workpiece_type": str(workpiece_table[idx]["type"]),
                    "workpiece_name": str(workpiece_table[idx]["name"]),
                    "fallback_ratio": float(per_workpiece_fallback[idx]),
                }
                for idx in np.argsort(-per_workpiece_fallback)[:20].tolist()
            ],
            "top_fallback_key_configs": [
                {
                    "key_config_order": int(idx),
                    "key_config_pool_index": int(key_config_indices[idx]),
                    "fallback_ratio": float(per_key_fallback[idx]),
                }
                for idx in np.argsort(-per_key_fallback)[:20].tolist()
            ],
            "suspicious_pairs": pair_records[:200],
        }
    finally:
        validator.close()

    output_path = pathlib.Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"workpiece_count: {summary['workpiece_count']}")
    print(f"key_config_count: {summary['key_config_count']}")
    print(f"overall_fallback_ratio: {summary['overall_fallback_ratio']:.6f}")
    print(f"overall_mean_inside_ratio: {summary['overall_mean_inside_ratio']:.6f}")
    print(f"regular_fallback_ratio: {summary['regular_fallback_ratio']}")
    print(f"simple_fallback_ratio: {summary['simple_fallback_ratio']}")
    print(f"saved_output_json: {output_path}")


if __name__ == "__main__":
    main()
