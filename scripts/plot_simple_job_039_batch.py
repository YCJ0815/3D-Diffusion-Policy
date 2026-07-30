import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "config" / "ur5e_with_pen.urdf"
RESULTS_DIR = PROJECT_ROOT / "data" / "raw_data" / "simple_results" / "job_039"
JOBS_DIR = PROJECT_ROOT / "data" / "raw_data" / "simple_jobs" / "job_039"
OUT_DIR = RESULTS_DIR / "trajectory_plots"
NUM_POSES = 6
DECIMATION_TARGET = None  # full quality, no decimation


# ── Math ──
def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])

def transform_from_origin(origin):
    xyz = np.zeros(3); rpy_ = np.zeros(3)
    if origin is not None:
        if origin.get("xyz"): xyz = np.array([float(v) for v in origin.get("xyz").split()])
        if origin.get("rpy"): rpy_ = np.array([float(v) for v in origin.get("rpy").split()])
    mat = np.eye(4); mat[:3, :3] = rpy_matrix(rpy_); mat[:3, 3] = xyz
    return mat

def axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0: return np.eye(4)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    rot = np.array([
        [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
        [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
        [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
    ])
    mat = np.eye(4); mat[:3, :3] = rot
    return mat

def parse_float3(value, default):
    if not value: return np.array(default, dtype=float)
    return np.array([float(v) for v in value.split()])

def parse_float_array(text):
    return np.fromstring(text or "", sep=" ", dtype=float)

def collada_id_ref(value):
    return value[1:] if value and value.startswith("#") else value

def collada_node_transform(node):
    mat = np.eye(4)
    el = node.find("c:matrix", COLLADA_NS)
    if el is not None and el.text:
        mat = np.array([float(v) for v in el.text.split()]).reshape(4, 4)
    return mat

# ── Collada ──
def read_collada_materials(root):
    effects = {}
    for effect in root.findall(".//c:library_effects/c:effect", COLLADA_NS):
        diffuse = effect.find(".//c:diffuse/c:color", COLLADA_NS)
        if diffuse is not None and diffuse.text:
            effects[effect.get("id")] = np.array([float(v) for v in diffuse.text.split()][:3], dtype=np.float32)
    materials = {}
    for mat in root.findall(".//c:library_materials/c:material", COLLADA_NS):
        inst = mat.find("c:instance_effect", COLLADA_NS)
        eid = collada_id_ref(inst.get("url")) if inst is not None else None
        if eid in effects: materials[mat.get("id")] = effects[eid]
    return materials

def read_collada_geometries(root):
    geometries = {}
    for geom in root.findall(".//c:library_geometries/c:geometry", COLLADA_NS):
        mesh_el = geom.find("c:mesh", COLLADA_NS)
        if mesh_el is None: continue
        sources = {}
        for source in mesh_el.findall("c:source", COLLADA_NS):
            fa = source.find("c:float_array", COLLADA_NS)
            acc = source.find(".//c:accessor", COLLADA_NS)
            if fa is None or acc is None: continue
            stride = int(acc.get("stride", "1"))
            vals = parse_float_array(fa.text)
            if len(vals) % stride: vals = vals[:len(vals) - (len(vals) % stride)]
            sources[source.get("id")] = vals.reshape((-1, stride))
        vertices_map = {}
        for verts in mesh_el.findall("c:vertices", COLLADA_NS):
            pos = verts.find("c:input[@semantic='POSITION']", COLLADA_NS)
            if pos is not None: vertices_map[verts.get("id")] = collada_id_ref(pos.get("source"))
        parts = []
        for prim in list(mesh_el.findall("c:polylist", COLLADA_NS)) + list(mesh_el.findall("c:triangles", COLLADA_NS)):
            inputs = prim.findall("c:input", COLLADA_NS)
            if not inputs: continue
            stride = max(int(inp.get("offset", "0")) for inp in inputs) + 1
            vo, psrc = None, None
            for inp in inputs:
                sem = inp.get("semantic"); sid = collada_id_ref(inp.get("source"))
                if sem == "VERTEX": vo = int(inp.get("offset", "0")); psrc = vertices_map.get(sid)
                elif sem == "POSITION": vo = int(inp.get("offset", "0")); psrc = sid
            if vo is None or psrc not in sources: continue
            idx_text = prim.findtext("c:p", namespaces=COLLADA_NS)
            indices = np.fromstring(idx_text or "", sep=" ", dtype=int)
            if len(indices) == 0: continue
            rows = indices.reshape((-1, stride)); vi = rows[:, vo]; pos = sources[psrc][:, :3]
            if prim.tag.endswith("polylist"):
                vcount = np.fromstring(prim.findtext("c:vcount", namespaces=COLLADA_NS) or "", sep=" ", dtype=int)
                faces = []; c = 0
                for cnt in vcount:
                    poly = vi[c:c + cnt]; c += cnt
                    if cnt < 3: continue
                    for i in range(1, cnt - 1): faces.append([poly[0], poly[i], poly[i + 1]])
                faces = np.array(faces, dtype=int)
            else:
                faces = vi.reshape((-1, 3))
            if len(faces): parts.append((pos.astype(np.float32), faces.astype(np.int32), prim.get("material")))
        geometries[geom.get("id")] = parts
    return geometries

def iter_collada_instances(node, parent_tf):
    node_tf = parent_tf @ collada_node_transform(node)
    for inst in node.findall("c:instance_geometry", COLLADA_NS):
        mat_sym = {}
        for b in inst.findall(".//c:instance_material", COLLADA_NS):
            mat_sym[b.get("symbol")] = collada_id_ref(b.get("target"))
        yield collada_id_ref(inst.get("url")), node_tf, mat_sym
    for child in node.findall("c:node", COLLADA_NS):
        yield from iter_collada_instances(child, node_tf)

def load_dae_meshes(path, scale, decimate_ratio=None):
    root = ET.parse(path).getroot()
    materials = read_collada_materials(root)
    geometries = read_collada_geometries(root)
    meshes = []
    scene_nodes = root.findall(".//c:library_visual_scenes/c:visual_scene/c:node", COLLADA_NS)
    for node in scene_nodes:
        for geom_id, node_tf, mat_sym in iter_collada_instances(node, np.eye(4)):
            for positions, faces, symbol in geometries.get(geom_id, []):
                verts = positions @ node_tf[:3, :3].T + node_tf[:3, 3]
                verts *= scale
                color = materials.get(mat_sym.get(symbol, symbol))

                if decimate_ratio is not None and len(faces) > 24:
                    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    target = max(12, int(len(faces) * decimate_ratio))
                    try:
                        m2 = m.simplify_quadric_decimation(face_count=target, aggression=8)
                        verts = np.asarray(m2.vertices, dtype=np.float32)
                        faces = np.asarray(m2.faces, dtype=np.int32)
                    except Exception:
                        pass

                meshes.append((verts, faces, color))
    if not meshes:
        for parts in geometries.values():
            for positions, faces, symbol in parts:
                meshes.append((positions * scale, faces, materials.get(symbol)))
    return meshes

def package_path(filename, urdf_path):
    if filename.startswith("package://urdf-pen/"):
        rel = filename.removeprefix("package://urdf-pen/")
        return urdf_path.parent / "robot-model" / rel
    return urdf_path.parent / filename

def compute_face_normals(verts, faces):
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norms, 1e-10)


def main():
    print("Loading URDF and visual DAE meshes (with decimation)...")
    tree = ET.parse(URDF_PATH)
    robot = tree.getroot()

    children_by_parent = {}
    for joint in robot.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        jtype = joint.get("type", "fixed")
        origin_tf = transform_from_origin(joint.find("origin"))
        axis = parse_float3(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, [1, 0, 0])
        spec = {"name": joint.get("name"), "type": jtype, "parent": parent, "child": child, "origin_tf": origin_tf, "axis": axis}
        children_by_parent.setdefault(parent, []).append(spec)

    joint_names_ordered = [j["name"] for j in sum(children_by_parent.values(), []) if j["type"] in {"revolute", "continuous"}]

    def compute_fk(joint_angles):
        angles = list(joint_angles)
        link_tf = {"world": np.eye(4)}
        def visit(link, tf):
            link_tf[link] = tf
            for joint in children_by_parent.get(link, []):
                motion = np.eye(4)
                if joint["type"] in {"revolute", "continuous"}:
                    if joint["name"] in joint_names_ordered:
                        idx = joint_names_ordered.index(joint["name"])
                        if idx < len(angles):
                            motion = axis_angle_matrix(joint["axis"], float(angles[idx]))
                visit(joint["child"], tf @ joint["origin_tf"] @ motion)
        visit("world", np.eye(4))
        return link_tf

    # Load + decimate visual meshes
    link_meshes = {}
    fallback_colors = {
        "base_link": np.array([0.28, 0.28, 0.28], dtype=np.float32),
        "pen_link": np.array([0.08, 0.18, 0.22], dtype=np.float32),
    }
    default_color = np.array([0.70, 0.74, 0.78], dtype=np.float32)

    for link in robot.findall("link"):
        link_name = link.get("name")
        parts = []
        for item in link.findall("visual"):
            geom = item.find("geometry")
            if geom is None or geom.find("mesh") is None: continue
            mesh_el = geom.find("mesh")
            filename = mesh_el.get("filename")
            if not filename: continue
            mesh_path = package_path(filename, URDF_PATH)
            if not mesh_path.exists(): continue
            scale = parse_float3(mesh_el.get("scale"), [1.0, 1.0, 1.0])
            origin_tf = transform_from_origin(item.find("origin"))
            for verts, faces, dae_color in load_dae_meshes(mesh_path, scale, DECIMATION_TARGET):
                link_color = dae_color if dae_color is not None else fallback_colors.get(link_name, default_color)
                parts.append((verts, faces, link_color, origin_tf))
        if parts:
            link_meshes[link_name] = parts

    total_faces = sum(len(f) for ms in link_meshes.values() for _, f, _, _ in ms)
    print(f"Links: {list(link_meshes.keys())}, total faces: {total_faces}")

    # Workpiece
    stl_path = JOBS_DIR / "workpiece_sim.stl"
    wp_verts_m = None; wp_faces = None
    if stl_path.exists():
        wp = trimesh.load(stl_path)
        wv = np.asarray(wp.vertices, dtype=np.float64)
        wf = np.asarray(wp.faces, dtype=np.int32)
        wp_verts_m = wv * 0.001 + np.array([0.5, 0.0, 0.0])
        wp_faces = wf

    transitions = sorted([f.stem for f in RESULTS_DIR.glob("transition_*.npz")],
                         key=lambda n: tuple(int(x) for x in n.split("_")[1:3]))
    print(f"Processing {len(transitions)} transitions...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    light_dir = np.array([0.3, -0.5, 0.8]); light_dir /= np.linalg.norm(light_dir)

    for ti, tname in enumerate(transitions):
        npz = np.load(RESULTS_DIR / f"{tname}.npz")
        q_playback = npz["q_playback"]
        start_xyz = npz["start_xyz"]
        end_xyz = npz["end_xyz"]

        n_poses = min(NUM_POSES, len(q_playback))
        indices = np.linspace(0, len(q_playback) - 1, n_poses, dtype=int)

        fig = plt.figure(figsize=(12, 9))
        fig.patch.set_alpha(0)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor((1, 1, 1, 0))
        ax.grid(False)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("none")
        ax.set_axis_off()

        all_verts_list = []

        if wp_verts_m is not None:
            wn = compute_face_normals(wp_verts_m, wp_faces)
            ws = np.clip(wn @ light_dir, 0.0, 1.0)[:, None]
            wc = np.clip(np.array([[0.85, 0.82, 0.78]]) * (0.4 + 0.6 * ws), 0, 1)
            wr = np.concatenate([wc, np.full((wc.shape[0], 1), 0.45)], axis=1)
            wpoly = Poly3DCollection(wp_verts_m[wp_faces], linewidths=0.0)
            wpoly.set_facecolor(wr); wpoly.set_edgecolor("none")
            ax.add_collection3d(wpoly)
            all_verts_list.append(wp_verts_m)

        for wi, idx in enumerate(indices):
            q = q_playback[idx]
            progress = wi / max(n_poses - 1, 1)
            alpha = 0.06 + 0.94 * progress

            link_tf = compute_fk(q)

            for link_name, meshes in link_meshes.items():
                if link_name not in link_tf: continue
                world_tf = link_tf[link_name]
                for verts, faces, link_color, origin_tf in meshes:
                    full_tf = world_tf @ origin_tf
                    R = full_tf[:3, :3]; t = full_tf[:3, 3]
                    transformed = verts @ R.T + t
                    all_verts_list.append(transformed)

                    rgba = np.append(link_color, alpha)
                    poly = Poly3DCollection(transformed[faces], linewidths=0.02)
                    poly.set_facecolor(rgba)
                    poly.set_edgecolor((0.18, 0.22, 0.25, min(alpha * 0.2, 1.0)))
                    ax.add_collection3d(poly)

        ax.scatter(*start_xyz, c="limegreen", s=100, marker="o",
                   edgecolors="darkgreen", linewidths=1.0, alpha=0.15, zorder=300)
        ax.scatter(*end_xyz, c="crimson", s=100, marker="o",
                   edgecolors="darkred", linewidths=1.0, alpha=0.90, zorder=300)

        all_pts = np.vstack(all_verts_list) if all_verts_list else np.zeros((1, 3))
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        centers = (mins + maxs) / 2.0; r = (maxs - mins).max() / 2.0 * 1.08
        ax.set_xlim(centers[0] - r, centers[0] + r)
        ax.set_ylim(centers[1] - r, centers[1] + r)
        ax.set_zlim(centers[2] - r, centers[2] + r)
        ax.view_init(elev=22, azim=-48)

        out_path = OUT_DIR / f"{tname}_arm_trajectory.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="none", transparent=True, pad_inches=0.02)
        plt.close(fig)

        print(f"[{ti+1}/{len(transitions)}] {tname}")

    print(f"\nDone — {len(transitions)} images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
