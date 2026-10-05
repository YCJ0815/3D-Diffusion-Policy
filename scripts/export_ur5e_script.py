#!/usr/bin/env python3
"""Offline NPY -> URScript export. No network access or robot execution.

Input: absolute joint radians, columns base/shoulder/elbow/wrist1/wrist2/wrist3.
Each movej ends at rest (r=0); this is deliberately a stop-at-waypoint replay.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
DEFAULT_URDF = Path(__file__).resolve().parents[1] / "config/robot-model/ur5e_with_pen.urdf"


def positive(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def vector(q):
    return "[" + ", ".join(f"{v:.10f}" for v in q) + "]"


def render(q, speed, acceleration, start_tolerance):
    lines = [
        "# Absolute joint angles in radians; base, shoulder, elbow, wrist1, wrist2, wrist3.",
        "# Offline export: verify in URSim and the calibrated workcell before execution.",
        "# No welding IO, TCP or payload changes. Every waypoint is a full stop.",
        "# Position the robot at the first waypoint using a separately verified approach.",
        "def replay_predicted_path():",
        f"  q_start = {vector(q[0])}",
        "  q_actual = get_actual_joint_positions()",
        "  qd_actual = get_actual_joint_speeds()",
        "  i = 0",
        "  while i < 6:",
        f"    if abs(q_actual[i] - q_start[i]) > {start_tolerance:.10f}:",
        '      textmsg("Start mismatch; position robot at q_start before replay")',
        "      halt",
        "    end",
        "    if abs(qd_actual[i]) > 0.01:",
        '      textmsg("Robot must be stationary before replay")',
        "      halt",
        "    end",
        "    i = i + 1",
        "  end",
    ]
    for i, waypoint in enumerate(q):
        lines += [f"  # waypoint {i}",
                  f"  movej({vector(waypoint)}, a={acceleration:.10f}, v={speed:.10f}, t=0, r=0)"]
    lines += ['  textmsg("Predicted path completed")', "end", "replay_predicted_path()", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--speed", type=positive, default=0.10, help="movej leading-axis speed, rad/s")
    parser.add_argument("--acceleration", type=positive, default=0.20, help="movej leading-axis acceleration, rad/s^2")
    parser.add_argument("--start-tolerance", type=positive, default=0.01, help="Per-joint start tolerance, rad; no angle wrapping")
    parser.add_argument("--max-joint-step", type=positive, default=0.25, help="Reject larger adjacent changes, rad")
    args = parser.parse_args()
    if "normalized" in args.trajectory.name.lower():
        parser.error("Use actual joint radians, not the normalized trajectory")
    q = np.asarray(np.load(args.trajectory, allow_pickle=False), dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or not np.isfinite(q).all():
        parser.error("Expected finite [N, 6] joint radians with N >= 2")
    joints = {j.attrib["name"]: j for j in ET.parse(args.urdf).getroot().findall("joint")}
    for column, name in enumerate(JOINT_NAMES):
        limit = joints[name].find("limit")
        if limit is None:
            parser.error(f"Missing joint limits: {name}")
        if np.any(q[:, column] < float(limit.attrib["lower"])) or np.any(q[:, column] > float(limit.attrib["upper"])):
            parser.error(f"Joint values exceed URDF limits: {name}")
        if args.speed > float(limit.attrib["velocity"]):
            parser.error(f"Requested speed exceeds URDF limit: {name}")
    max_step = float(np.max(np.abs(np.diff(q, axis=0))))
    if max_step > args.max_joint_step:
        parser.error(f"Adjacent joint step {max_step:.6f} exceeds {args.max_joint_step}; inspect angle continuity")
    report_path = args.output.with_suffix(".json")
    if args.output.suffix != ".script":
        parser.error("Output must have .script extension")
    if args.output.exists() or report_path.exists():
        parser.error("Output already exists; choose a new name")
    report = {
        "source": str(args.trajectory.resolve()), "joint_names": JOINT_NAMES,
        "point_count": len(q), "mode": "movej_stop_at_each_waypoint", "blend_radius_m": 0,
        "speed_rad_s": args.speed, "acceleration_rad_s2": args.acceleration,
        "start_tolerance_rad": args.start_tolerance, "max_joint_step_rad": max_step,
        "start_joints_rad": q[0].tolist(), "start_joints_deg": np.rad2deg(q[0]).tolist(),
        "end_joints_rad": q[-1].tolist(), "urdf": str(args.urdf.resolve()),
        "validation": "Offline format, finite values, URDF position/velocity limits, adjacent joint steps only",
        "robot_or_ursim_tested": False,
        "notes": ["No collision certification or workcell calibration performed by this exporter",
                  "Controller assigns movej timing; not the simulation playback timing",
                  "Start tolerance permits a small initial correction to q_start",
                  "No upload, welding IO, TCP configuration or payload configuration"],
    }
    script = render(q, args.speed, args.acceleration, args.start_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as stream:
        stream.write(script)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Exported {len(q)} waypoints: {args.output}")
    print(f"Report: {report_path}")
    print(f"Start joints (degrees): {np.rad2deg(q[0]).round(3).tolist()}")


if __name__ == "__main__":
    main()
