#!/usr/bin/env python3
"""Export the complete PyBullet demo sequence as controller-local URScript.

Sequence: default joints -> workpiece target with tool0 Z down -> lift ->
above prediction start -> descend -> continuous resampled prediction.
Load the output once in a PolyScope Script/File node; it calls its own entry
function. This exporter does not connect to or command a robot.
"""
import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from rrt_joint_approach import add_rrt_arguments, edge_durations, plan_pybullet_approach

from export_ur5e_script import JOINT_NAMES, DEFAULT_URDF, positive, vector
from replay_predicted_trajectory_pybullet import (
    DEFAULT_TRAJECTORY, DEFAULT_WORKPIECE, create_workpiece,
    build_parser as build_replay_parser, tcp_position,
    joint_indices_by_name, load_trajectory, resolve_urdf_meshes,
    set_joint_configuration, solve_vertical_down_ik,
)


def distance_at(t, length, velocity, ramp):
    duration = length / velocity + ramp
    t = np.clip(np.asarray(t), 0, duration)

    def ramp_distance(x):
        u = np.clip(x / ramp, 0, 1)
        return velocity * ramp * (u**3 - 0.5 * u**4)

    return np.where(
        t < ramp, ramp_distance(t),
        np.where(t > duration-ramp, length-ramp_distance(duration-t), velocity*(t-ramp/2)),
    )


def smooth_segment(start, goal, dt, minimum_duration, vmax, amax):
    """Quintic joint segment, lengthened until discrete limits are satisfied."""
    duration = float(minimum_duration)
    for _ in range(30):
        count = max(2, int(np.ceil(duration / dt)) + 1)
        time = np.arange(count, dtype=np.float64) * dt
        duration = float(time[-1])
        u = time / duration
        alpha = 10*u**3 - 15*u**4 + 6*u**5
        values = start[None, :] + alpha[:, None] * (goal-start)[None, :]
        velocity = np.diff(values, axis=0) / dt
        acceleration = np.diff(np.vstack([np.zeros(6), velocity, np.zeros(6)]), axis=0) / dt
        peak_v = float(np.abs(velocity).max())
        peak_a = float(np.abs(acceleration).max())
        if peak_v <= vmax and peak_a <= amax:
            return values, duration, peak_v, peak_a
        duration *= max(1.1, peak_v/vmax, np.sqrt(peak_a/amax)) * 1.02
    raise RuntimeError("Could not time-scale a default-pose segment within joint limits")


def build_prediction_targets(q, joint_at, inverse_arc, arc_length, args, vmax, dt):
    speed = args.tcp_speed_mm_s / 1000
    ramp = min(args.ramp_seconds, arc_length/speed)
    nominal_duration = arc_length/speed + ramp
    scale = 1.0
    for attempt in range(30):
        steps = int(np.ceil(nominal_duration*scale/dt))
        if steps > 1_000_000:
            raise ValueError("More than one million servo targets required; inspect path or speed")
        time = np.arange(steps+1, dtype=np.float64) * dt
        scale = float(time[-1]/nominal_duration)
        distance = distance_at(time/scale, arc_length, speed, ramp)
        targets = joint_at(inverse_arc(distance))
        targets[0], targets[-1] = q[0], q[-1]
        velocity = np.diff(targets, axis=0)/dt
        acceleration = np.diff(np.vstack([np.zeros(6), velocity, np.zeros(6)]), axis=0)/dt
        peak_v = float(np.abs(velocity).max())
        peak_a = float(np.abs(acceleration).max())
        if peak_v <= vmax and peak_a <= args.max_joint_acceleration:
            return targets, distance, scale, ramp, peak_v, peak_a
        print(f"Timing check {attempt}: speed={peak_v:.5f} rad/s, acceleration={peak_a:.5f} rad/s^2, scale={scale:.5f}")
        scale *= max(1.1, peak_v/vmax, np.sqrt(peak_a/args.max_joint_acceleration))*1.05
    raise RuntimeError("Could not time-scale prediction within joint limits")


def render_script(phase_arrays, default_joints, final_joints, args):
    lines = [
        "# Script/File node: load once; replay_complete_path() is called below.",
        "# Complete trajectory: default -> workpiece -> lift -> above start -> descend -> prediction.",
        "# Absolute joint angles are radians in UR joint order.",
        "# Verify in URSim and in the calibrated workcell before robot execution.",
        "def track(q_target):",
        "  actual = get_actual_joint_positions()", "  j = 0", "  while j < 6:",
        f"    if abs(actual[j] - q_target[j]) > {args.tracking_tolerance:.10f}:",
        "      stopj(1.0)", '      textmsg("Tracking error: trajectory stopped")',
        "      halt", "    end", "    j = j + 1", "  end",
        "  servoj(q_target, a=0, v=0, t=0.002, lookahead_time=0.1, gain=300)",
        "end", "def replay_complete_path():", f"  q_start = {vector(default_joints)}",
        "  q_actual = get_actual_joint_positions()", "  qd_actual = get_actual_joint_speeds()",
        "  i = 0", "  while i < 6:",
        f"    if abs(q_actual[i] - q_start[i]) > {args.start_tolerance:.10f}:",
        '      textmsg("Start mismatch: place robot at default joints")', "      halt", "    end",
        "    if abs(qd_actual[i]) > 0.01:",
        '      textmsg("Robot must be stationary before replay")', "      halt", "    end",
        "    i = i + 1", "  end",
    ]
    first = True
    for phase_name, values in phase_arrays:
        lines.append(f"  # phase: {phase_name}, targets: {len(values)}")
        for point in values:
            if first:
                first = False
                continue
            lines.append(f"  track({vector(point)})")
    lines += [
        "  k = 0", "  while k < 250:", f"    track({vector(final_joints)})",
        "    k = k + 1", "  end", "  stopj(1.0)", f"  q_end = {vector(final_joints)}",
        "  final_q = get_actual_joint_positions()", "  j = 0", "  while j < 6:",
        f"    if abs(final_q[j] - q_end[j]) > {args.start_tolerance:.10f}:",
        '      textmsg("Final position error: inspect execution")', "      halt", "    end",
        "    j = j + 1", "  end", '  textmsg("Complete trajectory finished")',
        "end", "replay_complete_path()", "",
    ]
    return "\n".join(lines)


def build_parser():
    replay_defaults = build_replay_parser().parse_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--workpiece-stl", type=Path, default=DEFAULT_WORKPIECE)
    parser.add_argument("--workpiece-position", type=float, nargs=3, default=replay_defaults.workpiece_position)
    parser.add_argument("--workpiece-scale", type=positive, default=replay_defaults.workpiece_scale)
    parser.add_argument("--workpiece-origin-target", type=float, nargs=3, default=None,
                        help="Override the visit target, independently of the workpiece position.")
    parser.add_argument("--default-joints", type=float, nargs=6, default=replay_defaults.default_joints)
    parser.add_argument("--vertical-yaw-deg", type=float, default=replay_defaults.vertical_yaw_deg)
    parser.add_argument("--approach-seconds", type=positive, default=replay_defaults.approach_seconds)
    parser.add_argument("--target-hold-seconds", type=positive, default=replay_defaults.target_hold_seconds)
    parser.add_argument("--trajectory-entry-seconds", type=positive, default=replay_defaults.trajectory_entry_seconds)
    parser.add_argument("--lift-height", type=positive, default=replay_defaults.lift_height)
    parser.add_argument("--lift-seconds", type=positive, default=replay_defaults.lift_seconds)
    parser.add_argument("--horizontal-seconds", type=positive, default=replay_defaults.horizontal_seconds)
    parser.add_argument("--tcp-speed-mm-s", type=positive, default=20.0)
    parser.add_argument("--ramp-seconds", type=positive, default=1.0)
    parser.add_argument("--max-joint-speed", type=positive, default=0.5)
    parser.add_argument("--max-joint-acceleration", type=positive, default=1.0)
    parser.add_argument("--start-tolerance", type=positive, default=0.005)
    parser.add_argument("--tracking-tolerance", type=positive, default=0.10)
    add_rrt_arguments(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    for name in ("default_joints", "workpiece_position", "vertical_yaw_deg", "workpiece_origin_target"):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value).all():
            parser.error(f"--{name.replace('_', '-')} must contain only finite values")

    if "normalized" in args.trajectory.name.lower():
        parser.error("Use absolute joint radians, not normalized values")
    q = load_trajectory(args.trajectory)
    if np.max(np.abs(np.diff(q, axis=0))) > 0.25:
        parser.error("Joint jump exceeds 0.25 rad; inspect the input path")
    outputs = [args.output, args.output.with_suffix(".json"), args.output.with_suffix(".npz")]
    if args.output.suffix != ".script" or any(path.exists() for path in outputs):
        parser.error("Choose a new .script output; existing files are not overwritten")

    limit_elements = {j.attrib["name"]: j.find("limit") for j in ET.parse(args.urdf).getroot().findall("joint")}
    vmax = min(args.max_joint_speed, *(float(limit_elements[n].attrib["velocity"]) for n in JOINT_NAMES))
    default_joints = np.asarray(args.default_joints, dtype=np.float64)
    workpiece_position = np.asarray(args.workpiece_position, dtype=np.float64)
    origin_target = np.asarray(args.workpiece_origin_target
                               if args.workpiece_origin_target is not None
                               else args.workpiece_position, dtype=np.float64)
    dt = 0.002

    import pybullet as pb
    client = pb.connect(pb.DIRECT)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            resolved_urdf = resolve_urdf_meshes(args.urdf, Path(temp_dir))
            robot = pb.loadURDF(
                str(resolved_urdf), useFixedBase=True,
                flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                physicsClientId=client,
            )
            joints, tcp, _ = joint_indices_by_name(pb, robot, client)
            workpiece = create_workpiece(pb, args.workpiece_stl.resolve(), workpiece_position, args.workpiece_scale, client)
            q_origin, ik_position_error, ik_down_error = solve_vertical_down_ik(
                pb, robot, joints, tcp, origin_target, args.vertical_yaw_deg,
                default_joints, client,
            )
            set_joint_configuration(pb, robot, joints, q[0], client)
            start_tcp = tcp_position(pb, robot, tcp, client)
            lift_target = origin_target + np.array([0.0, 0.0, args.lift_height])
            above_start_target = np.array([start_tcp[0], start_tcp[1], lift_target[2]])
            q_lift, lift_position_error, lift_down_error = solve_vertical_down_ik(
                pb, robot, joints, tcp, lift_target, args.vertical_yaw_deg, q_origin, client,
            )
            q_above_start, above_position_error, above_down_error = solve_vertical_down_ik(
                pb, robot, joints, tcp, above_start_target, args.vertical_yaw_deg, q_lift, client,
            )
            parameter = np.linspace(0, len(q)-1, (len(q)-1)*32+1)
            joint_at = CubicSpline(np.arange(len(q)), q, axis=0, bc_type="natural")
            xyz = []
            for point in joint_at(parameter):
                set_joint_configuration(pb, robot, joints, point, client)
                xyz.append(pb.getLinkState(robot, tcp, computeForwardKinematics=True, physicsClientId=client)[4])
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            if np.any(np.diff(arc[::32]) < 1e-8):
                parser.error("Stationary TCP segment cannot use translational-speed timing")
            keep = np.r_[True, np.diff(arc)>0]
            inverse_arc = PchipInterpolator(arc[keep], parameter[keep])
            prediction, prediction_distance, scale, ramp, _, _ = build_prediction_targets(
                q, joint_at, inverse_arc, float(arc[-1]), args, vmax, dt
            )
            outbound, _, _, _ = smooth_segment(
                default_joints, q_origin, dt, args.approach_seconds, vmax, args.max_joint_acceleration
            )
            lift, _, _, _ = smooth_segment(
                q_origin, q_lift, dt, args.lift_seconds, vmax, args.max_joint_acceleration
            )
            horizontal, _, _, _ = smooth_segment(
                q_lift, q_above_start, dt, args.horizontal_seconds, vmax, args.max_joint_acceleration
            )
            approach_path, approach_report = plan_pybullet_approach(
                pb, robot, joints, workpiece, q_above_start, q[0], client, args,
            )
            descend_parts = []
            for index, (start, goal, duration) in enumerate(zip(
                    approach_path[:-1], approach_path[1:],
                    edge_durations(approach_path, args.trajectory_entry_seconds))):
                segment, _, _, _ = smooth_segment(
                    start, goal, dt, max(dt, float(duration)), vmax, args.max_joint_acceleration
                )
                descend_parts.append(segment if index == 0 else segment[1:])
            descend = np.concatenate(descend_parts)
            hold = np.repeat(q_origin[None, :], max(1, int(round(args.target_hold_seconds/dt))), axis=0)
            phase_arrays = (
                ("default_to_workpiece_origin", outbound[:-1]),
                ("hold_at_workpiece_origin", hold),
                ("workpiece_origin_lift", lift[1:]),
                ("lift_to_above_start", horizontal[1:]),
                ("above_start_to_start", descend[1:-1]),
                ("predicted_trajectory", prediction),
            )
            complete = np.concatenate([values for _, values in phase_arrays], axis=0)
            # Check the emitted sequence, including every phase boundary.
            complete_velocity = np.diff(complete, axis=0) / dt
            complete_acceleration = np.diff(
                np.vstack([np.zeros(6), complete_velocity, np.zeros(6)]), axis=0
            ) / dt
            peak_v = float(np.abs(complete_velocity).max())
            peak_a = float(np.abs(complete_acceleration).max())
            if peak_v > vmax or peak_a > args.max_joint_acceleration:
                parser.error(f"Complete trajectory exceeds limits: speed={peak_v}, acceleration={peak_a}")
            phase_labels = np.concatenate([np.full(len(values), name) for name, values in phase_arrays])
            workpiece_contacts = {name: 0 for name, _ in phase_arrays}
            self_contacts = {name: 0 for name, _ in phase_arrays}
            for point, phase_name in zip(complete, phase_labels):
                set_joint_configuration(pb, robot, joints, point, client)
                pb.performCollisionDetection(physicsClientId=client)
                workpiece_contacts[str(phase_name)] += int(bool(pb.getContactPoints(robot, workpiece, physicsClientId=client)))
                self_contacts[str(phase_name)] += int(bool(pb.getContactPoints(robot, robot, physicsClientId=client)))
    finally:
        pb.disconnect(client)

    for column, name in enumerate(JOINT_NAMES):
        limit = limit_elements[name].attrib
        if (complete[:, column] < float(limit["lower"])).any() or (complete[:, column] > float(limit["upper"])).any():
            parser.error(f"Complete spline exceeds joint limits: {name}")
    if sum(workpiece_contacts.values()) or sum(self_contacts.values()):
        parser.error(f"Collision detected; export refused. workpiece={workpiece_contacts}, self={self_contacts}")

    script = render_script(phase_arrays, default_joints, q[-1], args)
    time = np.arange(len(complete), dtype=np.float64)*dt
    phase_points = {name: int(len(values)) for name, values in phase_arrays}
    report = {
        "source": str(args.trajectory.resolve()), "mode": "complete_controller_local_servoj",
        "sequence": list(phase_points), "source_points": int(len(q)),
        "servo_targets": int(len(complete)-1), "servo_period_seconds": dt,
        "total_motion_seconds": float(time[-1]), "final_settle_seconds": 0.5,
        "total_script_duration_seconds": float(time[-1]+0.5), "phase_points": phase_points,
        "phase_durations_seconds": {
            name: float((len(values) - (index == 0)) * dt)
            for index, (name, values) in enumerate(phase_arrays)
        },
        "execution": "polyscope_script_file_node",
        "approach_rrt": approach_report,
        "joint_names": list(JOINT_NAMES), "urdf": str(args.urdf.resolve()),
        "workpiece_origin_target_m": origin_target.tolist(),
        "lift_height_m": args.lift_height, "lift_target_m": lift_target.tolist(),
        "above_start_target_m": above_start_target.tolist(), "start_tcp_position_m": start_tcp.tolist(),
        "lift_joints_rad": q_lift.tolist(), "above_start_joints_rad": q_above_start.tolist(),
        "lift_ik_position_error_m": lift_position_error, "lift_ik_angle_error_deg": lift_down_error,
        "above_start_ik_position_error_m": above_position_error,
        "above_start_ik_angle_error_deg": above_down_error,
        "default_joints_rad": default_joints.tolist(), "workpiece_position_m": workpiece_position.tolist(),
        "vertical_yaw_deg": args.vertical_yaw_deg, "vertical_down_joints_rad": q_origin.tolist(),
        "vertical_ik_position_error_m": ik_position_error,
        "vertical_ik_angle_error_deg": ik_down_error,
        "requested_tcp_speed_mm_s": args.tcp_speed_mm_s,
        "nominal_prediction_cruise_speed_mm_s": args.tcp_speed_mm_s/scale,
        "prediction_tcp_arc_length_m": float(arc[-1]), "prediction_global_time_scale": scale,
        "prediction_ramp_seconds": ramp*scale,
        "peak_discrete_joint_speed_rad_s": peak_v,
        "peak_discrete_joint_acceleration_rad_s2": peak_a,
        "max_joint_speed_rad_s": vmax,
        "max_joint_acceleration_rad_s2": args.max_joint_acceleration,
        "robot_workpiece_contact_frames": workpiece_contacts,
        "robot_self_contact_frames": self_contacts,
        "start_tolerance_rad": args.start_tolerance,
        "tracking_tolerance_rad": args.tracking_tolerance, "robot_or_ursim_tested": False,
        "notes": [
            "Program starts only when actual joints match the configured default joints",
            "Load once in a Script/File node; the file calls replay_complete_path() itself",
            "TCP speed applies to prediction; approach phases use time-scaled quintic joint motion",
            "Phase sequence and approach joint curves match PyBullet; timing is rescaled for limits",
            "Prediction uses a cubic joint spline rather than the replay's linear interpolation",
            "Lift, horizontal and descent are joint-space curves, not guaranteed straight Cartesian lines",
            "Descent uses RRT-Connect and collision-checked shortcuts with quintic timing per edge",
            "RRT checks sampled edges; only the configured workpiece and robot are modeled",
            "Actual speed depends on robot speed scaling and servo tracking",
            "Collision checks use the configured URDF and workpiece and exclude table and fixtures",
            "No upload, welding IO, TCP configuration or payload configuration is performed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as stream:
        stream.write(script)
    with outputs[1].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    np.savez(
        outputs[2], joint_positions=complete, time_seconds=time, phase=phase_labels,
        joint_names=np.asarray(JOINT_NAMES), prediction_tcp_arc_length_m=prediction_distance,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
