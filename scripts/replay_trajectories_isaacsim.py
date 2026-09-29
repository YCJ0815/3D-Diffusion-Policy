#!/usr/bin/env python3
"""Replay exported UR5e trajectories in Isaac Sim and save exactly 32 frames.

Run this script with Isaac Sim's Python interpreter or its ``python.sh``.  It
loads the NPZ files produced by ``export_selected_episode_trajectories.py``,
imports the UR5e-with-pen URDF, applies one joint waypoint per output frame, and
captures RGB images with Replicator.

Isaac Sim is a raster renderer, so it cannot produce genuinely vector-valued
SVG robot renders.  The optional SVG output generated here is a standards-
compliant, standalone SVG container with the corresponding PNG embedded as a
base64 data URI.  Keep the PNG files for image processing and use the SVG
wrappers when an SVG file extension/container is required.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = (26301, 26297, 26120, 26303)
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "exported_trajectories"
DEFAULT_URDF = PROJECT_ROOT / "config" / "robot-model" / "ur5e_with_pen.urdf"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "isaacsim_replay_frames"


# Keep the replay exporter self-contained.  The project is often copied to a
# remote Isaac Sim machine without the sibling ``weld-robot`` repository that
# originally provided these helpers.  Isaac/Omniverse modules remain lazily
# imported so SimulationApp is still constructed before any Kit APIs are used.
def make_resolved_urdf(source_urdf: pathlib.Path) -> pathlib.Path:
    """Resolve package://urdf-pen mesh references in a temporary URDF."""
    if not source_urdf.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {source_urdf}")

    robot_model_dir = source_urdf.parent.resolve()
    tree = ET.parse(source_urdf)
    for mesh in tree.getroot().findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith("package://urdf-pen/"):
            local_path = robot_model_dir / filename.removeprefix(
                "package://urdf-pen/"
            )
            mesh.set("filename", str(local_path))

    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="ur5e_pen_urdf_"))
    resolved_urdf = temp_dir / source_urdf.name
    tree.write(resolved_urdf, encoding="utf-8", xml_declaration=True)
    return resolved_urdf


def enable_extension(extension_name: str) -> bool:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = None
    if hasattr(manager, "get_enabled_extension_id"):
        extension_id = manager.get_enabled_extension_id(extension_name)
    if extension_id:
        return True
    if hasattr(manager, "get_extension_id_by_module"):
        extension_id = manager.get_extension_id_by_module(extension_name)
    if not extension_id and hasattr(manager, "get_extension_id_by_name"):
        extension_id = manager.get_extension_id_by_name(extension_name)
    if not extension_id:
        return False
    manager.set_extension_enabled_immediate(extension_id, True)
    return True


def acquire_urdf_module() -> Any:
    for extension_name in (
        "isaacsim.asset.importer.urdf",
        "omni.importer.urdf",
        "omni.isaac.urdf",
    ):
        enable_extension(extension_name)
    try:
        from isaacsim.asset.importer.urdf import _urdf
    except ImportError:
        from omni.importer.urdf import _urdf
    return _urdf


def import_robot_from_urdf(
    urdf_path: pathlib.Path, prim_path: str, fix_base: bool
) -> str:
    _urdf = acquire_urdf_module()
    import_config = _urdf.ImportConfig()
    options = {
        "merge_fixed_joints": False,
        "fix_base": fix_base,
        "import_inertia_tensor": True,
        "convex_decomp": False,
        "self_collision": False,
        "distance_scale": 1.0,
        "make_default_prim": False,
        "default_drive_strength": 400.0,
        "default_position_drive_damping": 40.0,
    }
    for name, value in options.items():
        setter = getattr(import_config, f"set_{name}", None)
        if setter is not None:
            setter(value)
        elif hasattr(import_config, name):
            setattr(import_config, name, value)

    interface = _urdf.acquire_urdf_interface()
    root_path, file_name = str(urdf_path.parent), urdf_path.name
    parsed_robot = interface.parse_urdf(root_path, file_name, import_config)
    if isinstance(parsed_robot, tuple):
        success, parsed_robot = parsed_robot
        if not success:
            raise RuntimeError(f"Isaac Sim failed to parse URDF: {urdf_path}")
    imported_path = interface.import_robot(
        root_path, file_name, parsed_robot, import_config, prim_path
    )
    if isinstance(imported_path, tuple):
        imported_path = imported_path[-1]
    return str(imported_path or prim_path)


def ensure_xform(stage: Any, prim_path: str) -> Any:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    return prim


def set_xform_translation(
    prim: Any, translation: tuple[float, float, float]
) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))


def move_prim_to_path(stage: Any, source_path: str, target_path: str) -> str:
    if source_path == target_path:
        return target_path
    if not stage.GetPrimAtPath(source_path).IsValid():
        raise RuntimeError(f"Cannot move missing imported prim: {source_path}")

    target_parent = str(pathlib.PurePosixPath(target_path).parent)
    if target_parent != ".":
        ensure_xform(stage, target_parent)
    if stage.GetPrimAtPath(target_path).IsValid():
        stage.RemovePrim(target_path)

    try:
        import omni.kit.commands

        moved = omni.kit.commands.execute(
            "MovePrim", path_from=source_path, path_to=target_path
        )
        if isinstance(moved, tuple):
            moved = moved[0]
        if moved is False:
            raise RuntimeError("MovePrim command returned False")
    except Exception:
        from pxr import Sdf

        root_layer = stage.GetRootLayer()
        if not Sdf.CopySpec(root_layer, source_path, root_layer, target_path):
            raise RuntimeError(
                f"Failed to move imported prim from {source_path} to {target_path}"
            )
        stage.RemovePrim(source_path)

    prim = stage.GetPrimAtPath(target_path)
    if not prim.IsValid():
        raise RuntimeError(f"Imported prim was not moved to: {target_path}")
    set_xform_translation(prim, (0.0, 0.0, 0.0))
    return target_path


def parse_binary_stl(
    data: bytes,
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    if len(data) < 84:
        raise RuntimeError("Binary STL is too small")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if 84 + triangle_count * 50 > len(data):
        raise RuntimeError("Binary STL size does not match its triangle count")
    points: list[tuple[float, float, float]] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    offset = 84
    for _ in range(triangle_count):
        offset += 12  # normal
        for _ in range(3):
            points.append(struct.unpack_from("<fff", data, offset))
            face_indices.append(len(points) - 1)
            offset += 12
        face_counts.append(3)
        offset += 2
    return points, face_counts, face_indices


def parse_ascii_stl(
    text: str,
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    points = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            points.append(tuple(float(value) for value in parts[1:4]))
    if not points or len(points) % 3:
        raise RuntimeError("ASCII STL contains no complete triangle vertices")
    return points, [3] * (len(points) // 3), list(range(len(points)))


def load_stl_mesh(
    stl_path: pathlib.Path,
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    data = stl_path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + triangle_count * 50:
        return parse_binary_stl(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return parse_binary_stl(data)
    if text.lstrip().lower().startswith("solid"):
        return parse_ascii_stl(text)
    return parse_binary_stl(data)


def import_stl_as_mesh(
    stage: Any,
    stl_path: pathlib.Path,
    prim_path: str,
    scale: float,
    z_offset: float,
    local_offset: tuple[float, float, float],
    debug_box: bool,
) -> str:
    del debug_box  # Debug geometry is intentionally omitted by this exporter.
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    points, face_counts, face_indices = load_stl_mesh(stl_path)
    scaled_points = [
        (x * scale, y * scale, z * scale + z_offset) for x, y, z in points
    ]
    min_point = tuple(min(point[axis] for point in scaled_points) for axis in range(3))
    max_point = tuple(max(point[axis] for point in scaled_points) for axis in range(3))

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in scaled_points])
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([Gf.Vec3f(*min_point), Gf.Vec3f(*max_point)])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.62, 0.38)])
    set_xform_translation(mesh.GetPrim(), local_offset)

    material = UsdShade.Material.Define(stage, f"{prim_path}_Material")
    shader = UsdShade.Shader.Define(
        stage, f"{prim_path}_Material/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.78, 0.62, 0.38)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return prim_path


def default_trajectory_paths() -> list[pathlib.Path]:
    return [
        DEFAULT_TRAJECTORY_DIR / f"episode_{episode}_trajectory.npz"
        for episode in DEFAULT_EPISODES
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one or more UR5e trajectories and capture exactly 32 frames each."
    )
    parser.add_argument(
        "--trajectories",
        type=pathlib.Path,
        nargs="+",
        default=default_trajectory_paths(),
        help="NPZ trajectory files. Defaults to the four selected episodes.",
    )
    parser.add_argument("--urdf", type=pathlib.Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--rt-subframes", type=int, default=8)
    parser.add_argument(
        "--renderer",
        choices=("RayTracedLighting", "PathTracing"),
        default="RayTracedLighting",
        help="PathTracing is slower but can give higher-quality still images.",
    )
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        default=(1.8, -1.8, 1.35),
        metavar=("X", "Y", "Z"),
        help="Camera eye position in world coordinates (metres).",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=(0.35, 0.0, 0.45),
        metavar=("X", "Y", "Z"),
        help="World-space point at which the camera looks (metres).",
    )
    parser.add_argument(
        "--camera-orbit-deg",
        type=float,
        default=0.0,
        help="Rotate camera position around target about world Z without changing distance.",
    )
    parser.add_argument("--focal-length-mm", type=float, default=35.0)
    parser.add_argument("--focus-distance-m", type=float, default=3.0)
    parser.add_argument("--near-clip-m", type=float, default=0.01)
    parser.add_argument("--far-clip-m", type=float, default=1000.0)
    parser.add_argument("--dome-light-intensity", type=float, default=900.0)
    parser.add_argument("--distant-light-intensity", type=float, default=1500.0)
    parser.add_argument(
        "--robot-position",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Robot base translation in world coordinates (metres).",
    )
    parser.add_argument(
        "--workpiece-stl",
        type=pathlib.Path,
        default=None,
        help="Optional workpiece STL to show during replay.",
    )
    parser.add_argument(
        "--workpiece-position",
        type=float,
        nargs=3,
        default=(0.5, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Workpiece translation matching validation's default x offset.",
    )
    parser.add_argument(
        "--workpiece-scale",
        type=float,
        default=0.001,
        help="STL scale; validation loads millimetre STL data with 0.001.",
    )
    parser.add_argument(
        "--table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a static table below the robot and workpiece (default: enabled).",
    )
    parser.add_argument(
        "--table-size",
        type=float,
        nargs=2,
        default=(2.0, 1.5),
        metavar=("WIDTH_X", "DEPTH_Y"),
        help="Tabletop width and depth in metres.",
    )
    parser.add_argument(
        "--table-center-xy",
        type=float,
        nargs=2,
        default=(0.25, 0.0),
        metavar=("X", "Y"),
        help="Table centre position in the horizontal plane, in metres.",
    )
    parser.add_argument(
        "--table-thickness",
        type=float,
        default=0.08,
        help="Table slab thickness in metres; it extends downward from the tabletop.",
    )
    parser.add_argument(
        "--table-top-z",
        type=float,
        default=None,
        help=(
            "Absolute tabletop height in world Z. By default it is computed from "
            "the workpiece STL bottom so the table touches it exactly; without an "
            "STL, the default is Z=0."
        ),
    )
    parser.add_argument(
        "--table-color",
        type=float,
        nargs=3,
        default=(0.32, 0.18, 0.08),
        metavar=("R", "G", "B"),
        help="Table RGB colour, with every component in [0, 1].",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "png+svg"),
        default="png+svg",
        help="SVG files are standalone wrappers containing the raster PNG.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ground-plane",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional ground plane. Disabled by default to avoid intersecting the table.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_frames != 32:
        raise ValueError("This replay exporter requires --num-frames 32.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.fps <= 0.0 or args.physics_dt <= 0.0:
        raise ValueError("--fps and --physics-dt must be positive")
    if args.rt_subframes < 1:
        raise ValueError("--rt-subframes must be at least 1")
    if args.near_clip_m <= 0.0 or args.far_clip_m <= args.near_clip_m:
        raise ValueError("camera clipping range is invalid")
    if any(value <= 0.0 for value in args.table_size):
        raise ValueError("both --table-size values must be positive")
    if args.table_thickness <= 0.0:
        raise ValueError("--table-thickness must be positive")
    if any(value < 0.0 or value > 1.0 for value in args.table_color):
        raise ValueError("every --table-color component must be in [0, 1]")

    args.urdf = args.urdf.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.trajectories = [path.expanduser().resolve() for path in args.trajectories]
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")
    missing = [path for path in args.trajectories if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"trajectory files not found: {missing}")
    if args.workpiece_stl is not None:
        args.workpiece_stl = args.workpiece_stl.expanduser().resolve()
        if not args.workpiece_stl.is_file():
            raise FileNotFoundError(f"workpiece STL not found: {args.workpiece_stl}")


def load_trajectory(path: pathlib.Path) -> tuple[np.ndarray, list[str], int]:
    with np.load(path, allow_pickle=False) as data:
        key = next(
            (name for name in ("joint_positions", "trajectory", "q") if name in data),
            None,
        )
        if key is None:
            raise KeyError(f"{path} has no joint_positions/trajectory/q field")
        trajectory = np.asarray(data[key], dtype=np.float64)
        if "joint_names" not in data:
            raise KeyError(f"{path} has no joint_names field")
        joint_names = [str(value) for value in np.asarray(data["joint_names"]).tolist()]
        episode_idx = (
            int(np.asarray(data["episode_idx"]).reshape(()))
            if "episode_idx" in data
            else -1
        )

    if trajectory.ndim != 2 or trajectory.shape[1] != len(joint_names):
        raise ValueError(
            f"invalid trajectory {path}: shape={trajectory.shape}, joints={joint_names}"
        )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError(f"trajectory contains NaN/Inf: {path}")
    return trajectory, joint_names, episode_idx


def resample_trajectory(trajectory: np.ndarray, num_frames: int) -> np.ndarray:
    """Linearly resample in joint space; a 32-point input is returned unchanged."""
    if trajectory.shape[0] == num_frames:
        return trajectory.copy()
    source = np.linspace(0.0, 1.0, trajectory.shape[0], dtype=np.float64)
    target = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    return np.stack(
        [np.interp(target, source, trajectory[:, index]) for index in range(trajectory.shape[1])],
        axis=1,
    )


def orbit_camera_position(
    position: list[float] | tuple[float, ...],
    target: list[float] | tuple[float, ...],
    angle_deg: float,
) -> tuple[float, float, float]:
    position_array = np.asarray(position, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    relative = position_array - target_array
    angle = math.radians(float(angle_deg))
    rotated = np.asarray(
        [
            math.cos(angle) * relative[0] - math.sin(angle) * relative[1],
            math.sin(angle) * relative[0] + math.cos(angle) * relative[1],
            relative[2],
        ]
    )
    result = target_array + rotated
    return tuple(float(value) for value in result)


def create_lighting(stage: Any, args: argparse.Namespace) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/ReplayDomeLight")
    dome.CreateIntensityAttr(float(args.dome_light_intensity))
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    distant = UsdLux.DistantLight.Define(stage, "/World/ReplayKeyLight")
    distant.CreateIntensityAttr(float(args.distant_light_intensity))
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, -30.0))


def workpiece_bottom_z(args: argparse.Namespace) -> float | None:
    """Return the workpiece's lowest world-space Z coordinate."""
    if args.workpiece_stl is None:
        return None
    points, _, _ = load_stl_mesh(args.workpiece_stl)
    if not points:
        raise ValueError(f"workpiece STL contains no vertices: {args.workpiece_stl}")
    local_bottom = min(float(point[2]) for point in points) * float(
        args.workpiece_scale
    )
    return local_bottom + float(args.workpiece_position[2])


def resolve_table_top_z(args: argparse.Namespace) -> tuple[float, str]:
    if args.table_top_z is not None:
        return float(args.table_top_z), "command_line_override"
    bottom_z = workpiece_bottom_z(args)
    if bottom_z is not None:
        return float(bottom_z), "workpiece_bottom"
    return 0.0, "default_without_workpiece"


def create_table(
    stage: Any,
    args: argparse.Namespace,
    *,
    table_top_z: float,
) -> None:
    """Create a static collision table whose upper face is at table_top_z."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    width, depth = (float(value) for value in args.table_size)
    thickness = float(args.table_thickness)
    center_x, center_y = (float(value) for value in args.table_center_xy)
    center_z = float(table_top_z) - 0.5 * thickness

    cube = UsdGeom.Cube.Define(stage, "/World/ReplayTable")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in args.table_color])])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, center_z))
    xform.AddScaleOp().Set(Gf.Vec3f(width, depth, thickness))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    material = UsdShade.Material.Define(stage, "/World/ReplayTableMaterial")
    shader = UsdShade.Shader.Define(
        stage, "/World/ReplayTableMaterial/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(value) for value in args.table_color])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.62)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)


def create_articulation(robot_prim_path: str) -> Any:
    try:
        from isaacsim.core.prims import SingleArticulation
    except ImportError:
        from omni.isaac.core.articulations import Articulation as SingleArticulation

    robot = SingleArticulation(prim_path=robot_prim_path, name="replay_ur5e")
    robot.initialize()
    return robot


def resolve_dof_indices(robot: Any, joint_names: list[str]) -> list[int]:
    available = list(robot.dof_names)
    missing = [name for name in joint_names if name not in available]
    if missing:
        raise RuntimeError(f"robot is missing joints {missing}; available={available}")
    return [available.index(name) for name in joint_names]


def apply_waypoint(robot: Any, dof_indices: list[int], waypoint: np.ndarray) -> None:
    positions = np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)
    for source_index, dof_index in enumerate(dof_indices):
        positions[dof_index] = float(waypoint[source_index])
    robot.set_joint_positions(positions)
    try:
        velocities = np.asarray(robot.get_joint_velocities(), dtype=np.float64).reshape(-1)
        velocities[:] = 0.0
        robot.set_joint_velocities(velocities)
    except Exception:
        pass


def step_replicator(rep: Any, *, rt_subframes: int, fps: float) -> None:
    try:
        rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=1.0 / fps)
    except TypeError:
        try:
            rep.orchestrator.step(rt_subframes=rt_subframes)
        except TypeError:
            rep.orchestrator.step()


def prepare_episode_dir(path: pathlib.Path, overwrite: bool) -> pathlib.Path:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {path}; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def png_files_under(path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        candidate
        for candidate in path.rglob("*.png")
        if candidate.is_file()
    )


def write_embedded_svg(
    png_path: pathlib.Path,
    svg_path: pathlib.Path,
    *,
    width: int,
    height: int,
) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image width="{width}" height="{height}" '
        f'href="data:image/png;base64,{encoded}"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


def finalize_episode_outputs(
    episode_dir: pathlib.Path,
    *,
    num_frames: int,
    output_format: str,
    width: int,
    height: int,
) -> list[pathlib.Path]:
    png_files = png_files_under(episode_dir / "png")
    if len(png_files) != num_frames:
        raise RuntimeError(
            f"expected exactly {num_frames} PNG frames in {episode_dir}, found {len(png_files)}"
        )
    if output_format == "png+svg":
        svg_dir = episode_dir / "svg"
        svg_dir.mkdir(parents=True, exist_ok=True)
        for index, png_path in enumerate(png_files):
            write_embedded_svg(
                png_path,
                svg_dir / f"frame_{index:03d}.svg",
                width=width,
                height=height,
            )
    return png_files


def import_robot(stage: Any, resolved_urdf: pathlib.Path, robot_position: list[float]) -> str:
    requested_path = "/World/UR5ePen"
    imported_path = import_robot_from_urdf(
        resolved_urdf, requested_path, fix_base=True
    )
    if not stage.GetPrimAtPath(imported_path).IsValid() and stage.GetPrimAtPath(
        "/ur5e_pen"
    ).IsValid():
        imported_path = "/ur5e_pen"
    robot_path = move_prim_to_path(stage, imported_path, requested_path)

    set_xform_translation(
        stage.GetPrimAtPath(robot_path),
        tuple(float(value) for value in robot_position),
    )
    return robot_path


def write_episode_metadata(
    episode_dir: pathlib.Path,
    *,
    trajectory_path: pathlib.Path,
    episode_idx: int,
    args: argparse.Namespace,
    camera_position: tuple[float, float, float],
    table_top_z: float,
    table_top_z_source: str,
) -> None:
    metadata = {
        "episode_idx": episode_idx,
        "trajectory": str(trajectory_path),
        "num_frames": args.num_frames,
        "resolution": [args.width, args.height],
        "fps": args.fps,
        "renderer": args.renderer,
        "camera": {
            "position": list(camera_position),
            "target": list(args.camera_target),
            "orbit_deg": args.camera_orbit_deg,
            "focal_length_mm": args.focal_length_mm,
            "focus_distance_m": args.focus_distance_m,
            "clipping_range_m": [args.near_clip_m, args.far_clip_m],
        },
        "robot_position": list(args.robot_position),
        "workpiece_stl": None if args.workpiece_stl is None else str(args.workpiece_stl),
        "workpiece_position": list(args.workpiece_position),
        "workpiece_scale": args.workpiece_scale,
        "table": {
            "enabled": bool(args.table),
            "size_xy_m": list(args.table_size),
            "center_xy_m": list(args.table_center_xy),
            "thickness_m": args.table_thickness,
            "top_z_m": table_top_z,
            "top_z_source": table_top_z_source,
            "color_rgb": list(args.table_color),
        },
        "svg_note": (
            "SVG files embed raster PNG renders; they are not native vector geometry."
            if args.output_format == "png+svg"
            else None
        ),
    }
    (episode_dir / "replay_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    print(
        f"[replay:init] arguments validated; trajectories={len(args.trajectories)}, "
        f"urdf={args.urdf}",
        flush=True,
    )

    # SimulationApp must be created before importing any Isaac/Omniverse APIs.
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    print("[replay:init] creating SimulationApp", flush=True)
    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "enable_cameras": True,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
            "multi_gpu": False,
            "max_gpu_count": 1,
        }
    )
    print("[replay:init] SimulationApp ready", flush=True)

    rep = None
    render_product = None
    attached_writer = None
    try:
        print("[replay:init] importing Isaac Sim runtime APIs", flush=True)
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World
        import omni.replicator.core as rep
        from omni.usd import get_context
        print("[replay:init] runtime APIs imported", flush=True)

        try:
            rep.orchestrator.set_capture_on_play(False)
        except Exception:
            pass

        print("[replay:init] creating World", flush=True)
        world = World(
            physics_dt=args.physics_dt,
            rendering_dt=1.0 / args.fps,
            stage_units_in_meters=1.0,
        )
        if args.ground_plane:
            world.scene.add_default_ground_plane()
        stage = get_context().get_stage()
        ensure_xform(stage, "/World")
        create_lighting(stage, args)
        print("[replay:init] World and lighting ready", flush=True)

        resolved_urdf = make_resolved_urdf(args.urdf)
        print(f"[replay:init] importing URDF: {resolved_urdf}", flush=True)
        robot_path = import_robot(stage, resolved_urdf, args.robot_position)
        print(f"[replay:init] robot imported: {robot_path}", flush=True)
        if args.workpiece_stl is not None:
            import_stl_as_mesh(
                stage=stage,
                stl_path=args.workpiece_stl,
                prim_path="/World/Workpiece",
                scale=float(args.workpiece_scale),
                z_offset=0.0,
                local_offset=tuple(float(value) for value in args.workpiece_position),
                debug_box=False,
            )
        table_top_z, table_top_z_source = resolve_table_top_z(args)
        if args.table:
            create_table(stage, args, table_top_z=table_top_z)
            print(
                f"[table] top_z={table_top_z:.6f} m "
                f"(source={table_top_z_source}), thickness={args.table_thickness:.3f} m",
                flush=True,
            )

        print("[replay:init] resetting World", flush=True)
        world.reset()
        world.step(render=False)
        print("[replay:init] initializing robot articulation", flush=True)
        robot = create_articulation(robot_path)
        print(f"[replay:init] articulation ready; dofs={list(robot.dof_names)}", flush=True)

        camera_position = orbit_camera_position(
            args.camera_position, args.camera_target, args.camera_orbit_deg
        )
        camera = rep.create.camera(
            position=camera_position,
            look_at=tuple(float(value) for value in args.camera_target),
            focal_length=float(args.focal_length_mm),
            focus_distance=float(args.focus_distance_m),
            clipping_range=(float(args.near_clip_m), float(args.far_clip_m)),
        )
        render_product = rep.create.render_product(
            camera, resolution=(args.width, args.height)
        )
        print(
            f"[replay:init] camera/render product ready; "
            f"resolution={args.width}x{args.height}",
            flush=True,
        )

        for trajectory_path in args.trajectories:
            print(f"[replay] loading trajectory: {trajectory_path}", flush=True)
            trajectory, joint_names, episode_idx = load_trajectory(trajectory_path)
            trajectory = resample_trajectory(trajectory, args.num_frames)
            dof_indices = resolve_dof_indices(robot, joint_names)
            episode_name = (
                f"episode_{episode_idx}"
                if episode_idx >= 0
                else trajectory_path.stem
            )
            episode_dir = prepare_episode_dir(
                args.output_root / episode_name, args.overwrite
            )
            png_dir = episode_dir / "png"
            png_dir.mkdir(parents=True, exist_ok=True)

            # Settle the first pose before the writer is attached, so warm-up
            # renders never appear among the requested 32 output frames.
            apply_waypoint(robot, dof_indices, trajectory[0])
            print(f"[{episode_name}] warming up renderer", flush=True)
            for _ in range(3):
                world.step(render=True)

            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(png_dir), rgb=True)
            writer.attach([render_product])
            attached_writer = writer
            print(f"[{episode_name}] writer attached: {png_dir}", flush=True)
            for frame_index, waypoint in enumerate(trajectory):
                apply_waypoint(robot, dof_indices, waypoint)
                world.step(render=True)
                step_replicator(
                    rep, rt_subframes=args.rt_subframes, fps=args.fps
                )
                print(
                    f"[{episode_name}] captured frame {frame_index + 1:02d}/{args.num_frames}",
                    flush=True,
                )
            rep.orchestrator.wait_until_complete()
            writer.detach()
            attached_writer = None

            frames = finalize_episode_outputs(
                episode_dir,
                num_frames=args.num_frames,
                output_format=args.output_format,
                width=args.width,
                height=args.height,
            )
            write_episode_metadata(
                episode_dir,
                trajectory_path=trajectory_path,
                episode_idx=episode_idx,
                args=args,
                camera_position=camera_position,
                table_top_z=table_top_z,
                table_top_z_source=table_top_z_source,
            )
            print(
                f"[{episode_name}] completed: {len(frames)} PNG frames in {episode_dir}",
                flush=True,
            )
    finally:
        # Replicator writes asynchronously.  Complete and detach it before
        # destroying the Hydra render product; otherwise RTX resources can
        # still be in use while Kit is shutting down.
        if rep is not None:
            try:
                rep.orchestrator.wait_until_complete()
            except Exception as exc:
                print(f"[cleanup] Replicator wait failed: {exc}", flush=True)
        if attached_writer is not None:
            try:
                attached_writer.detach()
            except Exception as exc:
                print(f"[cleanup] Writer detach failed: {exc}", flush=True)
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception as exc:
                print(f"[cleanup] Render product destroy failed: {exc}", flush=True)
        try:
            # Isaac Sim 5.1 can crash in librtx.scenedb during full plugin
            # teardown on some Linux/Blackwell configurations.  Replicator has
            # already been flushed and its Hydra product destroyed above, so
            # fast shutdown is safe here and bypasses that native teardown bug.
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        except TypeError:
            # Compatibility with Isaac Sim releases predating skip_cleanup.
            simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
