#!/usr/bin/env python3
"""Replay a predicted UR5e joint trajectory with its workpiece in PyBullet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_EXPERIMENT_DIR = WORKSPACE_ROOT / "experiments" / "real-experiment"
DEFAULT_PREDICTION_DIR = DEFAULT_EXPERIMENT_DIR / "sdf_0001" / "resampled_tcp_2mm"
DEFAULT_TRAJECTORY = DEFAULT_PREDICTION_DIR / "pred_joint_horizon.npy"
DEFAULT_TCP_TRANSFORMS = DEFAULT_PREDICTION_DIR / "pred_tcp_transforms.npy"
# The resampled trajectory has its own shape and therefore must not be checked
# against the original 64-point prediction summary.
DEFAULT_SUMMARY = DEFAULT_PREDICTION_DIR / "summary.json"
DEFAULT_WORKPIECE = DEFAULT_EXPERIMENT_DIR / "工件" / "test.stl"
DEFAULT_URDF = PROJECT_ROOT / "config" / "robot-model" / "ur5e_with_pen.urdf"
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=path_arg, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--tcp-transforms", type=path_arg, default=DEFAULT_TCP_TRANSFORMS)
    parser.add_argument("--summary", type=path_arg, default=DEFAULT_SUMMARY)
    parser.add_argument("--urdf", type=path_arg, default=DEFAULT_URDF)
    parser.add_argument("--workpiece-stl", type=path_arg, default=DEFAULT_WORKPIECE)
    parser.add_argument(
        "--workpiece-position",
        type=float,
        nargs=3,
        default=(-0.23, -0.68, -0.14),
        metavar=("X", "Y", "Z"),
        help="Workpiece translation in world metres.",
    )
    parser.add_argument(
        "--workpiece-scale",
        type=float,
        default=0.001,
        help="STL source-unit to metre scale (default: 0.001 for millimetres).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--seconds-per-waypoint",
        type=float,
        default=1.0 / 30.0,
        help="Playback duration between adjacent predicted waypoints.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay continuously until the GUI closes (default: enabled).",
    )
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--workpiece-origin-round-trip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before trajectory playback: move from the default joints to the workpiece "
            "world origin with tool0 Z pointing down, return to default, then enter and play the trajectory."
        ),
    )
    parser.add_argument(
        "--default-joints",
        type=float,
        nargs=6,
        default=(-1.57, -1.57, 1.57, -1.57, -1.57, 0.0),
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
    )
    parser.add_argument(
        "--workpiece-origin-target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override the visit target. By default it equals --workpiece-position.",
    )
    parser.add_argument("--approach-seconds", type=float, default=5.0)
    parser.add_argument("--target-hold-seconds", type=float, default=1.0)
    parser.add_argument("--trajectory-entry-seconds", type=float, default=5.0)
    parser.add_argument(
        "--vertical-yaw-deg",
        type=float,
        default=0.0,
        help="Yaw of tool0 around world Z while its local Z axis points vertically down.",
    )
    parser.add_argument("--headless", action="store_true", help="Validate one replay without opening a window.")
    parser.add_argument("--no-tcp-trail", action="store_true")
    return parser


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_trajectory(path: Path) -> np.ndarray:
    trajectory = np.asarray(np.load(require_file(path, "trajectory")), dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"trajectory must have shape [T, 6], got {trajectory.shape}")
    if trajectory.shape[0] < 2 or not np.isfinite(trajectory).all():
        raise ValueError("trajectory must contain at least two finite waypoints")
    return trajectory


def resolve_urdf_meshes(urdf_path: Path, output_dir: Path) -> Path:
    """Replace package:// mesh references with absolute paths for PyBullet."""
    tree = ET.parse(urdf_path)
    package_root = urdf_path.parent
    for mesh in tree.getroot().findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None or not filename.startswith("package://"):
            continue
        package_relative = filename[len("package://") :]
        _, separator, relative_path = package_relative.partition("/")
        if not separator:
            raise ValueError(f"invalid package URI in URDF: {filename}")
        candidates = (package_root / relative_path, package_root.parent / relative_path)
        resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f"cannot resolve URDF mesh: {filename}")
        mesh.set("filename", str(resolved))
    resolved_urdf = output_dir / urdf_path.name
    tree.write(resolved_urdf, encoding="utf-8", xml_declaration=True)
    return resolved_urdf


def joint_indices_by_name(
    pb, robot_id: int, client_id: int
) -> tuple[list[int], int, dict[int, str]]:
    joint_by_name: dict[str, int] = {}
    link_by_name: dict[str, int] = {}
    link_name_by_index: dict[int, str] = {-1: "base_link"}
    for joint_index in range(pb.getNumJoints(robot_id, physicsClientId=client_id)):
        info = pb.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        joint_by_name[info[1].decode("utf-8")] = joint_index
        link_by_name[info[12].decode("utf-8")] = joint_index
        link_name_by_index[joint_index] = info[12].decode("utf-8")
    missing = [name for name in JOINT_NAMES if name not in joint_by_name]
    if missing:
        raise KeyError(f"URDF is missing trajectory joints: {missing}")
    if "tool0" not in link_by_name:
        raise KeyError("URDF is missing the tool0 TCP link")
    return (
        [joint_by_name[name] for name in JOINT_NAMES],
        link_by_name["tool0"],
        link_name_by_index,
    )


def set_joint_configuration(pb, robot_id: int, joint_indices: list[int], q: np.ndarray, client_id: int) -> None:
    for joint_index, value in zip(joint_indices, q):
        pb.resetJointState(
            robot_id,
            joint_index,
            targetValue=float(value),
            targetVelocity=0.0,
            physicsClientId=client_id,
        )


def tcp_position(pb, robot_id: int, tcp_link_index: int, client_id: int) -> np.ndarray:
    state = pb.getLinkState(
        robot_id,
        tcp_link_index,
        computeForwardKinematics=True,
        physicsClientId=client_id,
    )
    return np.asarray(state[4], dtype=np.float64)


def create_workpiece(pb, stl_path: Path, position: np.ndarray, scale: float, client_id: int) -> int:
    mesh_scale = [float(scale)] * 3
    collision = pb.createCollisionShape(
        pb.GEOM_MESH,
        fileName=str(stl_path),
        meshScale=mesh_scale,
        flags=pb.GEOM_FORCE_CONCAVE_TRIMESH,
        physicsClientId=client_id,
    )
    visual = pb.createVisualShape(
        pb.GEOM_MESH,
        fileName=str(stl_path),
        meshScale=mesh_scale,
        rgbaColor=[0.32, 0.48, 0.72, 1.0],
        specularColor=[0.15, 0.15, 0.15],
        physicsClientId=client_id,
    )
    if collision < 0 or visual < 0:
        raise RuntimeError(f"PyBullet failed to load workpiece mesh: {stl_path}")
    return pb.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position.tolist(),
        physicsClientId=client_id,
    )


def add_tcp_trail(pb, points: np.ndarray, client_id: int) -> None:
    for index in range(1, len(points)):
        fraction = index / max(1, len(points) - 1)
        color = [1.0 - 0.75 * fraction, 0.15 + 0.75 * fraction, 0.1]
        pb.addUserDebugLine(
            points[index - 1].tolist(),
            points[index].tolist(),
            lineColorRGB=color,
            lineWidth=3.0,
            lifeTime=0.0,
            physicsClientId=client_id,
        )
    for point, color in ((points[0], [0.1, 1.0, 0.1]), (points[-1], [1.0, 0.1, 0.1])):
        pb.addUserDebugLine(
            (point + [-0.012, 0.0, 0.0]).tolist(),
            (point + [0.012, 0.0, 0.0]).tolist(),
            color,
            4.0,
            physicsClientId=client_id,
        )


def interpolate_trajectory(trajectory: np.ndarray, fps: float, seconds_per_waypoint: float) -> np.ndarray:
    frames_per_segment = max(1, int(round(fps * seconds_per_waypoint)))
    frames = []
    for index in range(len(trajectory) - 1):
        for alpha in np.linspace(0.0, 1.0, frames_per_segment, endpoint=False):
            frames.append((1.0 - alpha) * trajectory[index] + alpha * trajectory[index + 1])
    frames.append(trajectory[-1])
    return np.asarray(frames, dtype=np.float64)


def smooth_joint_segment(start: np.ndarray, goal: np.ndarray, fps: float, duration: float) -> np.ndarray:
    """Quintic time scaling with zero endpoint velocity and acceleration."""
    frame_count = max(2, int(round(fps * duration)) + 1)
    u = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    alpha = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start[None, :] + alpha[:, None] * (goal - start)[None, :]


def solve_vertical_down_ik(
    pb,
    robot_id: int,
    joint_indices: list[int],
    tcp_link_index: int,
    target_position: np.ndarray,
    yaw_degrees: float,
    default_joints: np.ndarray,
    client_id: int,
) -> tuple[np.ndarray, float, float]:
    """Solve tool0-at-target with its local Z axis aligned to world -Z."""
    set_joint_configuration(pb, robot_id, joint_indices, default_joints, client_id)
    quaternion = pb.getQuaternionFromEuler(
        [np.pi, 0.0, np.deg2rad(float(yaw_degrees))]
    )
    solution = np.asarray(
        pb.calculateInverseKinematics(
            robot_id,
            tcp_link_index,
            targetPosition=target_position.tolist(),
            targetOrientation=quaternion,
            maxNumIterations=2000,
            residualThreshold=1e-10,
            physicsClientId=client_id,
        ),
        dtype=np.float64,
    )
    if solution.size < len(joint_indices):
        raise RuntimeError(f"PyBullet IK returned only {solution.size} joints")
    q_target = solution[: len(joint_indices)]
    for joint_index, value in zip(joint_indices, q_target):
        info = pb.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        lower, upper = float(info[8]), float(info[9])
        if lower < upper and not lower <= value <= upper:
            raise RuntimeError(
                f"Vertical-down IK violates {info[1].decode('utf-8')} limits: {value}"
            )
    set_joint_configuration(pb, robot_id, joint_indices, q_target, client_id)
    state = pb.getLinkState(
        robot_id, tcp_link_index, computeForwardKinematics=True, physicsClientId=client_id
    )
    actual_position = np.asarray(state[4], dtype=np.float64)
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[5]), dtype=np.float64).reshape(3, 3)
    position_error = float(np.linalg.norm(actual_position - target_position))
    down_angle = float(
        np.rad2deg(np.arccos(np.clip(np.dot(rotation[:, 2], [0.0, 0.0, -1.0]), -1.0, 1.0)))
    )
    if position_error > 1e-4 or down_angle > 0.1:
        raise RuntimeError(
            f"Vertical-down IK did not converge: position_error={position_error:.6g} m, "
            f"tool_z_down_error={down_angle:.6g} deg"
        )
    return q_target, position_error, down_angle


def main() -> None:
    args = build_parser().parse_args()
    positive_values = {
        "fps": args.fps,
        "seconds-per-waypoint": args.seconds_per_waypoint,
        "workpiece-scale": args.workpiece_scale,
        "approach-seconds": args.approach_seconds,
        "target-hold-seconds": args.target_hold_seconds,
        "trajectory-entry-seconds": args.trajectory_entry_seconds,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0.0]
    if invalid:
        raise ValueError(f"These values must be positive: {invalid}")

    trajectory = load_trajectory(args.trajectory)
    require_file(args.urdf, "URDF")
    require_file(args.workpiece_stl, "workpiece STL")
    if args.summary.is_file():
        with args.summary.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("planning_success") is not True:
            raise ValueError(f"prediction summary does not report planning success: {args.summary}")
        expected_shape = summary.get("pred_joint_horizon_shape")
        if expected_shape is not None and list(trajectory.shape) != list(expected_shape):
            raise ValueError(
                f"trajectory shape {list(trajectory.shape)} does not match summary {expected_shape}"
            )
        print(
            f"Prediction: planner_mode={summary.get('planner_mode')} "
            f"planning_success={summary.get('planning_success')} "
            f"selected_candidate={summary.get('selected_candidate_index')}"
        )

    try:
        import pybullet as pb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install PyBullet or run with the repository's pybullet Conda environment") from exc

    client_id = pb.connect(pb.DIRECT if args.headless else pb.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to connect to PyBullet")

    try:
        pb.resetSimulation(physicsClientId=client_id)
        pb.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        pb.setTimeStep(1.0 / float(args.fps), physicsClientId=client_id)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=client_id)

        workpiece_position = np.asarray(args.workpiece_position, dtype=np.float64)
        workpiece_id = create_workpiece(
            pb, args.workpiece_stl, workpiece_position, args.workpiece_scale, client_id
        )

        with tempfile.TemporaryDirectory(prefix="dp_dual_pybullet_") as tmp:
            resolved_urdf = resolve_urdf_meshes(args.urdf, Path(tmp))
            robot_id = pb.loadURDF(
                str(resolved_urdf),
                useFixedBase=True,
                flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                physicsClientId=client_id,
            )
            joint_indices, tcp_link_index, link_name_by_index = joint_indices_by_name(
                pb, robot_id, client_id
            )
            trajectory_playback = interpolate_trajectory(
                trajectory, args.fps, args.seconds_per_waypoint
            )
            playback = trajectory_playback
            phase_names = np.full(len(playback), "predicted_trajectory", dtype=object)

            if args.workpiece_origin_round_trip:
                default_joints = np.asarray(args.default_joints, dtype=np.float64)
                origin_target = np.asarray(
                    args.workpiece_origin_target
                    if args.workpiece_origin_target is not None
                    else args.workpiece_position,
                    dtype=np.float64,
                )
                q_origin, ik_position_error, ik_down_angle = solve_vertical_down_ik(
                    pb,
                    robot_id,
                    joint_indices,
                    tcp_link_index,
                    origin_target,
                    args.vertical_yaw_deg,
                    default_joints,
                    client_id,
                )
                outbound = smooth_joint_segment(
                    default_joints, q_origin, args.fps, args.approach_seconds
                )
                target_hold = np.repeat(
                    q_origin[None, :],
                    max(1, int(round(args.fps * args.target_hold_seconds))),
                    axis=0,
                )
                inbound = smooth_joint_segment(
                    q_origin, default_joints, args.fps, args.approach_seconds
                )
                trajectory_entry = smooth_joint_segment(
                    default_joints,
                    trajectory_playback[0],
                    args.fps,
                    args.trajectory_entry_seconds,
                )
                # Drop duplicated boundary frames while preserving each named phase.
                phase_arrays = (
                    ("default_to_workpiece_origin", outbound[:-1]),
                    ("hold_at_workpiece_origin", target_hold),
                    ("workpiece_origin_to_default", inbound[1:]),
                    ("default_to_trajectory_start", trajectory_entry[1:-1]),
                    ("predicted_trajectory", trajectory_playback),
                )
                playback = np.concatenate([values for _, values in phase_arrays], axis=0)
                phase_names = np.concatenate(
                    [np.full(len(values), name, dtype=object) for name, values in phase_arrays]
                )
                print(
                    "Workpiece-origin visit: "
                    f"target={origin_target.tolist()} m, vertical_yaw={args.vertical_yaw_deg:.3f} deg, "
                    f"IK_position_error={ik_position_error:.3e} m, "
                    f"tool_Z_down_error={ik_down_angle:.3e} deg"
                )
                print(f"Vertical-down joint target: {q_origin.tolist()}")

            tcp_points = []
            for waypoint in trajectory:
                set_joint_configuration(pb, robot_id, joint_indices, waypoint, client_id)
                tcp_points.append(tcp_position(pb, robot_id, tcp_link_index, client_id))
            tcp_points_array = np.asarray(tcp_points)

            if args.tcp_transforms.is_file():
                reference_tcp = np.asarray(np.load(args.tcp_transforms), dtype=np.float64)
                if reference_tcp.shape == (len(trajectory), 4, 4):
                    max_error = float(np.max(np.linalg.norm(tcp_points_array - reference_tcp[:, :3, 3], axis=1)))
                    print(f"TCP FK agreement max position error: {max_error:.6e} m")
                else:
                    print(f"Skipping TCP comparison: unexpected shape {reference_tcp.shape}")

            if not args.no_tcp_trail:
                if args.workpiece_origin_round_trip:
                    playback_tcp_points = []
                    for waypoint in playback:
                        set_joint_configuration(pb, robot_id, joint_indices, waypoint, client_id)
                        playback_tcp_points.append(
                            tcp_position(pb, robot_id, tcp_link_index, client_id)
                        )
                    add_tcp_trail(pb, np.asarray(playback_tcp_points), client_id)
                else:
                    add_tcp_trail(pb, tcp_points_array, client_id)

            camera_target = 0.5 * (np.zeros(3) + workpiece_position)
            camera_target[2] = max(0.15, float(workpiece_position[2]) + 0.15)
            if not args.headless:
                pb.resetDebugVisualizerCamera(
                    cameraDistance=1.35,
                    cameraYaw=48.0,
                    cameraPitch=-28.0,
                    cameraTargetPosition=camera_target.tolist(),
                    physicsClientId=client_id,
                )

            workpiece_contact_frames = 0
            self_contact_frames = 0
            contact_link_names: set[str] = set()
            minimum_contact_distance = 0.0
            phase_contact_frames = {str(name): 0 for name in dict.fromkeys(phase_names)}
            phase_self_contact_frames = {str(name): 0 for name in dict.fromkeys(phase_names)}
            while True:
                for q, phase_name in zip(playback, phase_names):
                    set_joint_configuration(pb, robot_id, joint_indices, q, client_id)
                    pb.performCollisionDetection(physicsClientId=client_id)
                    contacts = pb.getContactPoints(
                        bodyA=robot_id,
                        bodyB=workpiece_id,
                        physicsClientId=client_id,
                    )
                    self_contacts = pb.getContactPoints(
                        bodyA=robot_id,
                        bodyB=robot_id,
                        physicsClientId=client_id,
                    )
                    workpiece_contact_frames += int(bool(contacts))
                    self_contact_frames += int(bool(self_contacts))
                    phase_contact_frames[str(phase_name)] += int(bool(contacts))
                    phase_self_contact_frames[str(phase_name)] += int(bool(self_contacts))
                    for contact in contacts:
                        contact_link_names.add(link_name_by_index.get(int(contact[3]), str(contact[3])))
                        minimum_contact_distance = min(minimum_contact_distance, float(contact[8]))
                    if not args.headless:
                        time.sleep(1.0 / float(args.fps))
                if args.headless or not args.loop:
                    break

            final_tcp = tcp_position(pb, robot_id, tcp_link_index, client_id)
            print(f"Replayed {len(trajectory)} waypoints as {len(playback)} display frames")
            print(f"Final TCP position: {final_tcp.tolist()}")
            print(
                "Frames with robot-workpiece contacts: "
                f"{workpiece_contact_frames}/{len(playback)}"
            )
            print(f"Contact frames by phase: {phase_contact_frames}")
            print(f"Frames with robot self-contacts: {self_contact_frames}/{len(playback)}")
            print(f"Self-contact frames by phase: {phase_self_contact_frames}")
            if contact_link_names:
                print(
                    f"Contacting robot links: {sorted(contact_link_names)}; "
                    f"maximum penetration: {-minimum_contact_distance:.6e} m"
                )
            if not args.headless and not args.loop and args.hold_seconds > 0.0:
                time.sleep(float(args.hold_seconds))
    finally:
        pb.disconnect(client_id)


if __name__ == "__main__":
    main()
