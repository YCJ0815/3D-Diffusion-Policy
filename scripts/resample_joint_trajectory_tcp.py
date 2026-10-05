#!/usr/bin/env python3
"""Resample a joint path by TCP translation arc length; no robot connection."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from replay_predicted_trajectory_pybullet import (
    DEFAULT_URDF, DEFAULT_WORKPIECE, JOINT_NAMES, create_workpiece,
    joint_indices_by_name, load_trajectory, resolve_urdf_meshes,
    set_joint_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_RUN_DIR = WORKSPACE_ROOT / "experiments" / "real-experiment" / "transition_0000_0001_bspline_inference"
DEFAULT_SOURCE_TRAJECTORY = DEFAULT_RUN_DIR / "pred_joint_horizon.npy"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "resampled_tcp_2mm"


def spacing_stats(transforms):
    distances = np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1)
    return {
        "min_mm": float(distances.min() * 1000),
        "max_mm": float(distances.max() * 1000),
        "mean_mm": float(distances.mean() * 1000),
        "std_mm": float(distances.std() * 1000),
        "coefficient_of_variation": float(distances.std() / distances.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_SOURCE_TRAJECTORY,
        help=f"Source [N,6] joint trajectory (default: {DEFAULT_SOURCE_TRAJECTORY})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite the five generated files in an existing output directory; preserve other contents.",
    )
    parser.add_argument("--spacing-mm", type=float, default=2.0)
    parser.add_argument("--dense-substeps", type=int, default=256)
    parser.add_argument("--check-substeps", type=int, default=16)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--workpiece-stl", type=Path, default=DEFAULT_WORKPIECE)
    parser.add_argument("--workpiece-position", type=float, nargs=3, default=(-0.23, -0.68, -0.14))
    parser.add_argument("--workpiece-scale", type=float, default=0.001)
    args = parser.parse_args()
    if not np.isfinite(args.spacing_mm) or args.spacing_mm <= 0:
        parser.error("spacing must be finite and positive")
    if args.dense_substeps < 2 or args.check_substeps < 1:
        parser.error("dense-substeps >= 2 and check-substeps >= 1 required")
    source = load_trajectory(args.trajectory)
    # Preserve prior runs unless overwriting the known generated files was explicit.
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            parser.error(f"output path is not a directory: {args.output_dir}")
        if not args.overwrite_output:
            parser.error(
                f"output directory already exists: {args.output_dir}; choose another "
                "--output-dir or pass --overwrite-output"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    import pybullet as pb
    client = pb.connect(pb.DIRECT)
    try:
        with tempfile.TemporaryDirectory(prefix="tcp_resampling_") as temp:
            urdf = resolve_urdf_meshes(args.urdf, Path(temp))
            robot = pb.loadURDF(str(urdf), useFixedBase=True, physicsClientId=client)
            joints, tcp, links = joint_indices_by_name(pb, robot, client)
            workpiece = create_workpiece(pb, args.workpiece_stl.resolve(),
                np.asarray(args.workpiece_position), args.workpiece_scale, client)

            def joint_at(parameters):
                return np.column_stack([
                    np.interp(parameters, np.arange(len(source)), source[:, j])
                    for j in range(6)
                ])

            def fk(q):
                set_joint_configuration(pb, robot, joints, q, client)
                state = pb.getLinkState(robot, tcp, computeForwardKinematics=True,
                                        physicsClientId=client)
                transform = np.eye(4)
                transform[:3, :3] = np.asarray(pb.getMatrixFromQuaternion(state[5])).reshape(3, 3)
                transform[:3, 3] = state[4]
                return transform

            original_tcp = np.asarray([fk(q) for q in source])
            parameter = np.linspace(0, len(source) - 1,
                                    (len(source) - 1) * args.dense_substeps + 1)
            dense_tcp = np.asarray([fk(q) for q in joint_at(parameter)])
            lengths = np.linalg.norm(np.diff(dense_tcp[:, :3, 3], axis=0), axis=1)
            arc = np.r_[0.0, np.cumsum(lengths)]
            if arc[-1] <= 1e-9:
                raise ValueError("TCP translation is zero; use an orientation/joint metric instead")
            # A positional metric cannot preserve a stationary-TCP orientation-only segment.
            for i in range(len(source) - 1):
                segment_length = arc[(i + 1) * args.dense_substeps] - arc[i * args.dense_substeps]
                if segment_length < 1e-8 and np.max(np.abs(source[i + 1] - source[i])) > 1e-6:
                    raise ValueError(f"Orientation-only segment {i}; positional resampling would omit it")
            keep = np.r_[True, np.diff(arc) > 0]
            count = max(2, int(np.ceil(arc[-1] / (args.spacing_mm / 1000))) + 1)
            target_arc = np.linspace(0, arc[-1], count)
            source_parameter = np.interp(target_arc, arc[keep], parameter[keep])
            source_parameter[[0, -1]] = [0, len(source) - 1]
            result = joint_at(source_parameter)
            result_tcp = np.asarray([fk(q) for q in result])

            # Check the actual linear joint interpolation of the NEW waypoint sequence.
            contacts = []
            worst_distance = 0.0
            contact_links = set()
            total_checks = (count - 1) * args.check_substeps + 1
            for k in range(total_checks):
                index = min(k // args.check_substeps, count - 2)
                alpha = (k - index * args.check_substeps) / args.check_substeps
                q = (1 - alpha) * result[index] + alpha * result[index + 1]
                set_joint_configuration(pb, robot, joints, q, client)
                pb.performCollisionDetection(physicsClientId=client)
                points = pb.getContactPoints(robot, workpiece, physicsClientId=client)
                if points:
                    contacts.append(k)
                    worst_distance = min(worst_distance, min(float(p[8]) for p in points))
                    contact_links.update(links[p[3]] for p in points)

            report = {
                "source": str(args.trajectory.resolve()),
                "method": "uniform TCP translation arc length on piecewise-linear joint path",
                "tcp_link": "tool0", "source_count": len(source), "output_count": count,
                "requested_spacing_mm": args.spacing_mm,
                "estimated_tcp_arc_length_m": float(arc[-1]),
                "arc_step_mm": float(target_arc[1] * 1000),
                "dense_substeps": args.dense_substeps,
                "original_spacing": spacing_stats(original_tcp),
                "resampled_spacing": spacing_stats(result_tcp),
                "endpoint_joint_error_rad": float(np.max(np.abs(result[[0, -1]] - source[[0, -1]]))),
                "max_joint_step_rad": float(np.abs(np.diff(result, axis=0)).max()),
                "collision_check": {
                    "type": "discrete robot-workpiece contacts, concave static STL",
                    "frames": total_checks, "contact_frames": len(contacts),
                    "contact_frame_indices": contacts, "contact_links": sorted(contact_links),
                    "max_penetration_mm": -worst_distance * 1000,
                    "workpiece_position_m": args.workpiece_position,
                    "workpiece_scale": args.workpiece_scale,
                    "workpiece_stl": str(args.workpiece_stl.resolve()),
                    "urdf": str(args.urdf.resolve()),
                    "excludes": ["self-collision", "table and fixtures", "continuous collision guarantee"],
                },
                "timing": "No execution timestamps assigned; not a time-parameterized robot program",
            }
            np.save(args.output_dir / "pred_joint_horizon.npy", result)
            np.save(args.output_dir / "pred_tcp_transforms.npy", result_tcp)
            np.savez(args.output_dir / "trajectory.npz", joint_positions=result,
                     joint_names=np.asarray(JOINT_NAMES), tcp_transforms=result_tcp,
                     source_parameter=source_parameter, tcp_arc_length_m=target_arc)
            np.savetxt(args.output_dir / "joint_trajectory.csv", result, delimiter=",",
                       header=",".join(JOINT_NAMES), comments="")
            (args.output_dir / "resampling_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
    finally:
        pb.disconnect(client)


if __name__ == "__main__":
    main()
