#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
UR5E_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
DEFAULT_JOINT_ANGLES_RAD = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def transform_from_origin(origin):
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if origin is not None:
        if origin.get("xyz"):
            xyz = np.array([float(v) for v in origin.get("xyz").split()], dtype=float)
        if origin.get("rpy"):
            rpy = np.array([float(v) for v in origin.get("rpy").split()], dtype=float)
    mat = np.eye(4)
    mat[:3, :3] = rpy_matrix(rpy)
    mat[:3, 3] = xyz
    return mat


def axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(4)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    rot = np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )
    mat = np.eye(4)
    mat[:3, :3] = rot
    return mat


def parse_float3(value, default):
    if not value:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in value.split()], dtype=float)


def angle_to_rad(value, unit):
    value = float(value)
    if unit == "deg":
        return math.radians(value)
    return value


def parse_named_joint_angles(items, unit):
    values = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected joint override as name=value, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in UR5E_JOINT_ORDER:
            raise ValueError(f"Unknown joint {name!r}. Valid joints: {', '.join(UR5E_JOINT_ORDER)}")
        values[name] = angle_to_rad(value.strip(), unit)
    return values


def resolve_joint_angles(args):
    joint_values = dict(DEFAULT_JOINT_ANGLES_RAD)

    if args.joint_angles is not None:
        if len(args.joint_angles) != len(UR5E_JOINT_ORDER):
            raise ValueError(
                f"--joint-angles expects {len(UR5E_JOINT_ORDER)} values in this order: "
                f"{', '.join(UR5E_JOINT_ORDER)}"
            )
        joint_values.update(
            {
                name: angle_to_rad(value, args.joint_unit)
                for name, value in zip(UR5E_JOINT_ORDER, args.joint_angles)
            }
        )

    named_overrides = parse_named_joint_angles(args.joint_angle, args.joint_unit)
    joint_values.update(named_overrides)

    cli_overrides = {
        "shoulder_pan_joint": args.shoulder_pan,
        "shoulder_lift_joint": args.shoulder_lift,
        "elbow_joint": args.elbow,
        "wrist_1_joint": args.wrist_1,
        "wrist_2_joint": args.wrist_2,
        "wrist_3_joint": args.wrist_3,
    }
    for name, value in cli_overrides.items():
        if value is not None:
            joint_values[name] = angle_to_rad(value, args.joint_unit)

    return joint_values


def package_path(filename, urdf_path):
    if filename.startswith("package://urdf-pen/"):
        rel = filename.removeprefix("package://urdf-pen/")
        return urdf_path.parent / "robot-model" / rel
    return urdf_path.parent / filename


def parse_float_array(text):
    return np.fromstring(text or "", sep=" ", dtype=float)


def collada_id_ref(value):
    return value[1:] if value and value.startswith("#") else value


def collada_node_transform(node):
    mat = np.eye(4)
    matrix_el = node.find("c:matrix", COLLADA_NS)
    if matrix_el is not None and matrix_el.text:
        mat = np.array([float(v) for v in matrix_el.text.split()], dtype=float).reshape(4, 4)
    return mat


def read_collada_materials(root):
    effects = {}
    for effect in root.findall(".//c:library_effects/c:effect", COLLADA_NS):
        diffuse = effect.find(".//c:diffuse/c:color", COLLADA_NS)
        if diffuse is not None and diffuse.text:
            values = [float(v) for v in diffuse.text.split()]
            effects[effect.get("id")] = np.array(values[:3], dtype=float)

    materials = {}
    for material in root.findall(".//c:library_materials/c:material", COLLADA_NS):
        instance = material.find("c:instance_effect", COLLADA_NS)
        effect_id = collada_id_ref(instance.get("url")) if instance is not None else None
        if effect_id in effects:
            materials[material.get("id")] = effects[effect_id]
    return materials


def read_collada_geometries(root):
    geometries = {}
    for geom in root.findall(".//c:library_geometries/c:geometry", COLLADA_NS):
        mesh_el = geom.find("c:mesh", COLLADA_NS)
        if mesh_el is None:
            continue

        sources = {}
        for source in mesh_el.findall("c:source", COLLADA_NS):
            float_array = source.find("c:float_array", COLLADA_NS)
            accessor = source.find(".//c:accessor", COLLADA_NS)
            if float_array is None or accessor is None:
                continue
            stride = int(accessor.get("stride", "1"))
            values = parse_float_array(float_array.text)
            if len(values) % stride:
                values = values[: len(values) - (len(values) % stride)]
            sources[source.get("id")] = values.reshape((-1, stride))

        vertices_map = {}
        for vertices in mesh_el.findall("c:vertices", COLLADA_NS):
            position = vertices.find("c:input[@semantic='POSITION']", COLLADA_NS)
            if position is not None:
                vertices_map[vertices.get("id")] = collada_id_ref(position.get("source"))

        parts = []
        for primitive in list(mesh_el.findall("c:polylist", COLLADA_NS)) + list(mesh_el.findall("c:triangles", COLLADA_NS)):
            inputs = primitive.findall("c:input", COLLADA_NS)
            if not inputs:
                continue
            stride = max(int(inp.get("offset", "0")) for inp in inputs) + 1
            vertex_offset = None
            position_source = None
            for inp in inputs:
                semantic = inp.get("semantic")
                source_id = collada_id_ref(inp.get("source"))
                if semantic == "VERTEX":
                    vertex_offset = int(inp.get("offset", "0"))
                    position_source = vertices_map.get(source_id)
                elif semantic == "POSITION":
                    vertex_offset = int(inp.get("offset", "0"))
                    position_source = source_id
            if vertex_offset is None or position_source not in sources:
                continue

            index_text = primitive.findtext("c:p", namespaces=COLLADA_NS)
            indices = np.fromstring(index_text or "", sep=" ", dtype=int)
            if len(indices) == 0:
                continue
            index_rows = indices.reshape((-1, stride))
            vertex_indices = index_rows[:, vertex_offset]
            positions = sources[position_source][:, :3]

            if primitive.tag.endswith("polylist"):
                vcount = np.fromstring(primitive.findtext("c:vcount", namespaces=COLLADA_NS) or "", sep=" ", dtype=int)
                faces = []
                cursor = 0
                for count in vcount:
                    polygon = vertex_indices[cursor : cursor + count]
                    cursor += count
                    if count < 3:
                        continue
                    for i in range(1, count - 1):
                        faces.append([polygon[0], polygon[i], polygon[i + 1]])
                faces = np.array(faces, dtype=int)
            else:
                faces = vertex_indices.reshape((-1, 3))

            if len(faces):
                material = primitive.get("material")
                parts.append((positions.copy(), faces, material))

        geometries[geom.get("id")] = parts
    return geometries


def iter_collada_instances(node, parent_tf):
    node_tf = parent_tf @ collada_node_transform(node)
    for instance in node.findall("c:instance_geometry", COLLADA_NS):
        material_symbols = {}
        for bind in instance.findall(".//c:instance_material", COLLADA_NS):
            material_symbols[bind.get("symbol")] = collada_id_ref(bind.get("target"))
        yield collada_id_ref(instance.get("url")), node_tf, material_symbols
    for child in node.findall("c:node", COLLADA_NS):
        yield from iter_collada_instances(child, node_tf)


def load_collada_meshes(path, scale):
    root = ET.parse(path).getroot()
    materials = read_collada_materials(root)
    geometries = read_collada_geometries(root)
    meshes = []

    scene_nodes = root.findall(".//c:library_visual_scenes/c:visual_scene/c:node", COLLADA_NS)
    for node in scene_nodes:
        for geom_id, node_tf, material_symbols in iter_collada_instances(node, np.eye(4)):
            for positions, faces, symbol in geometries.get(geom_id, []):
                mesh = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
                mesh.apply_transform(node_tf)
                mesh.apply_scale(scale)
                material_id = material_symbols.get(symbol, symbol)
                color = materials.get(material_id)
                meshes.append((mesh, color))

    # Some simple DAE files omit a visual scene. Render their raw geometry if needed.
    if not meshes:
        for parts in geometries.values():
            for positions, faces, symbol in parts:
                mesh = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
                mesh.apply_scale(scale)
                meshes.append((mesh, materials.get(symbol)))
    return meshes


def load_meshes(path, scale):
    if path.suffix.lower() == ".dae":
        return load_collada_meshes(path, scale)

    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return [(mesh, None)]


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius *= 1.08
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default="config/ur5e_with_pen.urdf")
    parser.add_argument("--out", default="outputs/ur5e_visual_pose_transparent.png")
    parser.add_argument("--joint-json", default="outputs/ur5e_visual_pose_joints.json")
    parser.add_argument(
        "--joint-unit",
        choices=("rad", "deg"),
        default="rad",
        help="Unit used by all joint angle command-line values. Default: rad.",
    )
    parser.add_argument(
        "--joint-angles",
        nargs=6,
        type=float,
        metavar=("SHOULDER_PAN", "SHOULDER_LIFT", "ELBOW", "WRIST_1", "WRIST_2", "WRIST_3"),
        help="Six joint angles in UR5e joint order. Defaults are used when omitted.",
    )
    parser.add_argument(
        "--joint-angle",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override one joint by URDF name. Can be repeated, e.g. --joint-angle elbow_joint=1.2.",
    )
    parser.add_argument("--shoulder-pan", type=float, help="Override shoulder_pan_joint.")
    parser.add_argument("--shoulder-lift", type=float, help="Override shoulder_lift_joint.")
    parser.add_argument("--elbow", type=float, help="Override elbow_joint.")
    parser.add_argument("--wrist-1", type=float, help="Override wrist_1_joint.")
    parser.add_argument("--wrist-2", type=float, help="Override wrist_2_joint.")
    parser.add_argument("--wrist-3", type=float, help="Override wrist_3_joint.")
    args = parser.parse_args()

    root_dir = Path.cwd()
    urdf_path = (root_dir / args.urdf).resolve()
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    children_by_parent = {}
    joints = []
    joint_values = resolve_joint_angles(args)

    for joint in robot.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        jtype = joint.get("type", "fixed")
        origin_tf = transform_from_origin(joint.find("origin"))
        axis = parse_float3(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, [1, 0, 0])
        angle = 0.0
        if jtype in {"revolute", "continuous"}:
            angle = float(joint_values.get(joint.get("name"), 0.0))
        spec = {
            "name": joint.get("name"),
            "type": jtype,
            "parent": parent,
            "child": child,
            "origin_tf": origin_tf,
            "axis": axis,
            "angle": angle,
        }
        joints.append(spec)
        children_by_parent.setdefault(parent, []).append(spec)

    link_visuals = {}
    for link in robot.findall("link"):
        link_name = link.get("name")
        visuals = []
        for item in link.findall("visual"):
            geom = item.find("geometry")
            if geom is None or geom.find("mesh") is None:
                continue
            mesh_el = geom.find("mesh")
            filename = mesh_el.get("filename")
            if not filename:
                continue
            mesh_path = package_path(filename, urdf_path)
            if not mesh_path.exists():
                continue
            visuals.append(
                {
                    "path": mesh_path,
                    "scale": parse_float3(mesh_el.get("scale"), [1.0, 1.0, 1.0]),
                    "origin_tf": transform_from_origin(item.find("origin")),
                }
            )
        link_visuals[link_name] = visuals

    link_tf = {}

    def visit(link, tf):
        link_tf[link] = tf
        for joint in children_by_parent.get(link, []):
            motion = np.eye(4)
            if joint["type"] in {"revolute", "continuous"}:
                motion = axis_angle_matrix(joint["axis"], joint["angle"])
            visit(joint["child"], tf @ joint["origin_tf"] @ motion)

    roots = sorted(set(link_visuals) - {j["child"] for j in joints})
    if "world" in set(link_visuals):
        roots = ["world"]
    for root in roots:
        visit(root, np.eye(4))

    meshes = []
    all_vertices = []
    for link_name, visuals in link_visuals.items():
        if link_name not in link_tf:
            continue
        for visual in visuals:
            for mesh, color in load_meshes(visual["path"], visual["scale"]):
                mesh.apply_transform(link_tf[link_name] @ visual["origin_tf"])
                meshes.append((link_name, mesh, color))
                all_vertices.append(mesh.vertices)

    if not meshes:
        raise RuntimeError("No renderable meshes found from URDF.")

    all_points = np.vstack(all_vertices)
    fig = plt.figure(figsize=(7, 7), dpi=220)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((1, 1, 1, 0))
    ax.view_init(elev=21, azim=-42, roll=0)
    ax.set_axis_off()
    ax.grid(False)
    set_axes_equal(ax, all_points)

    light = np.array([0.25, -0.55, 0.8], dtype=float)
    light /= np.linalg.norm(light)
    base_colors = {
        "base_link": np.array([0.42, 0.47, 0.52]),
        "pen_link": np.array([0.08, 0.18, 0.22]),
    }
    default_color = np.array([0.70, 0.74, 0.78])

    for link_name, mesh, mesh_color in meshes:
        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.face_normals
        shade = np.clip(normals @ light, 0.0, 1.0)[:, None]
        base = mesh_color if mesh_color is not None else base_colors.get(link_name, default_color)
        colors = np.clip(base * (0.58 + 0.42 * shade), 0, 1)
        rgba = np.concatenate([colors, np.full((colors.shape[0], 1), 1.0)], axis=1)
        poly = Poly3DCollection(vertices[faces], linewidths=0.03)
        poly.set_facecolor(rgba)
        poly.set_edgecolor((0.18, 0.22, 0.25, 0.14))
        ax.add_collection3d(poly)

    out_path = (root_dir / args.out).resolve()
    json_path = (root_dir / args.joint_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "urdf": str(urdf_path.relative_to(root_dir)),
                "joint_order": UR5E_JOINT_ORDER,
                "default_joint_angles_rad": DEFAULT_JOINT_ANGLES_RAD,
                "default_joint_angles_deg": {k: math.degrees(v) for k, v in DEFAULT_JOINT_ANGLES_RAD.items()},
                "joint_angles_rad": joint_values,
                "joint_angles_deg": {k: math.degrees(v) for k, v in joint_values.items()},
                "image": str(out_path.relative_to(root_dir)),
                "visual_source": "URDF visual meshes (.dae from config/robot-model/meshes/ur5e/visual)",
            },
            f,
            indent=2,
        )
    print(out_path)
    print(json_path)


if __name__ == "__main__":
    main()
