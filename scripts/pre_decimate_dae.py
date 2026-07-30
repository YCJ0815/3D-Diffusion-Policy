import json
import math
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path("/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy")
URDF_PATH = PROJECT_ROOT / "config" / "ur5e_with_pen.urdf"
CACHE_PATH = PROJECT_ROOT / "data" / "ur5e_decimated_meshes.npz"
DECIMATION_RATIO = 0.04

COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}

def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                     [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                     [-sp, sr*cp, cr*cp]])

def transform_from_origin(origin):
    xyz = np.zeros(3); rpy_ = np.zeros(3)
    if origin is not None:
        if origin.get("xyz"): xyz = np.array([float(v) for v in origin.get("xyz").split()])
        if origin.get("rpy"): rpy_ = np.array([float(v) for v in origin.get("rpy").split()])
    mat = np.eye(4); mat[:3,:3] = rpy_matrix(rpy_); mat[:3,3] = xyz
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

def read_collada_materials(root):
    effects = {}
    for e in root.findall(".//c:library_effects/c:effect", COLLADA_NS):
        d = e.find(".//c:diffuse/c:color", COLLADA_NS)
        if d is not None and d.text: effects[e.get("id")] = np.array([float(v) for v in d.text.split()][:3], dtype=np.float32)
    mats = {}
    for m in root.findall(".//c:library_materials/c:material", COLLADA_NS):
        inst = m.find("c:instance_effect", COLLADA_NS)
        eid = collada_id_ref(inst.get("url")) if inst is not None else None
        if eid in effects: mats[m.get("id")] = effects[eid]
    return mats

def read_collada_geometries(root):
    geoms = {}
    for geom in root.findall(".//c:library_geometries/c:geometry", COLLADA_NS):
        mesh_el = geom.find("c:mesh", COLLADA_NS)
        if mesh_el is None: continue
        sources = {}
        for s in mesh_el.findall("c:source", COLLADA_NS):
            fa = s.find("c:float_array", COLLADA_NS); acc = s.find(".//c:accessor", COLLADA_NS)
            if fa is None or acc is None: continue
            stride = int(acc.get("stride", "1"))
            vals = parse_float_array(fa.text)
            if len(vals) % stride: vals = vals[:len(vals)-(len(vals)%stride)]
            sources[s.get("id")] = vals.reshape((-1, stride))
        vm = {}
        for v in mesh_el.findall("c:vertices", COLLADA_NS):
            pos = v.find("c:input[@semantic='POSITION']", COLLADA_NS)
            if pos is not None: vm[v.get("id")] = collada_id_ref(pos.get("source"))
        parts = []
        for prim in list(mesh_el.findall("c:polylist", COLLADA_NS)) + list(mesh_el.findall("c:triangles", COLLADA_NS)):
            inputs = prim.findall("c:input", COLLADA_NS)
            if not inputs: continue
            stride = max(int(inp.get("offset","0")) for inp in inputs) + 1
            vo, psrc = None, None
            for inp in inputs:
                sem = inp.get("semantic"); sid = collada_id_ref(inp.get("source"))
                if sem == "VERTEX": vo = int(inp.get("offset","0")); psrc = vm.get(sid)
                elif sem == "POSITION": vo = int(inp.get("offset","0")); psrc = sid
            if vo is None or psrc not in sources: continue
            idx_text = prim.findtext("c:p", namespaces=COLLADA_NS)
            indices = np.fromstring(idx_text or "", sep=" ", dtype=int)
            if len(indices) == 0: continue
            rows = indices.reshape((-1, stride)); vi = rows[:, vo]; pos = sources[psrc][:,:3]
            if prim.tag.endswith("polylist"):
                vcount = np.fromstring(prim.findtext("c:vcount", namespaces=COLLADA_NS) or "", sep=" ", dtype=int)
                faces = []; c = 0
                for cnt in vcount:
                    poly = vi[c:c+cnt]; c += cnt
                    if cnt < 3: continue
                    for i in range(1, cnt-1): faces.append([poly[0], poly[i], poly[i+1]])
                faces = np.array(faces, dtype=int)
            else:
                faces = vi.reshape((-1, 3))
            if len(faces): parts.append((pos.astype(np.float32), faces.astype(np.int32), prim.get("material")))
        geoms[geom.get("id")] = parts
    return geoms

def iter_collada_instances(node, parent_tf):
    node_tf = parent_tf @ collada_node_transform(node)
    for inst in node.findall("c:instance_geometry", COLLADA_NS):
        mat_sym = {}
        for b in inst.findall(".//c:instance_material", COLLADA_NS):
            mat_sym[b.get("symbol")] = collada_id_ref(b.get("target"))
        yield collada_id_ref(inst.get("url")), node_tf, mat_sym
    for child in node.findall("c:node", COLLADA_NS):
        yield from iter_collada_instances(child, node_tf)

def load_dae_meshes(path, scale, decimate_ratio):
    root = ET.parse(path).getroot()
    materials = read_collada_materials(root)
    geoms = read_collada_geometries(root)
    meshes = []
    sn = root.findall(".//c:library_visual_scenes/c:visual_scene/c:node", COLLADA_NS)
    for node in sn:
        for gid, node_tf, mat_sym in iter_collada_instances(node, np.eye(4)):
            for pos, faces, sym in geoms.get(gid, []):
                verts = pos @ node_tf[:3,:3].T + node_tf[:3,3]
                verts *= scale
                color = materials.get(mat_sym.get(sym, sym))
                if decimate_ratio is not None and len(faces) > 24:
                    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    target = max(12, int(len(faces) * decimate_ratio))
                    try:
                        m2 = m.simplify_quadric_decimation(target)
                        verts = np.asarray(m2.vertices, dtype=np.float32)
                        faces = np.asarray(m2.faces, dtype=np.int32)
                    except: pass
                meshes.append((verts, faces, color))
    if not meshes:
        for parts in geoms.values():
            for pos, faces, sym in parts:
                meshes.append((pos * scale, faces, materials.get(sym)))
    return meshes

def package_path(filename, urdf_path):
    if filename.startswith("package://urdf-pen/"):
        return urdf_path.parent / "robot-model" / filename.removeprefix("package://urdf-pen/")
    return urdf_path.parent / filename

def main():
    print("Pre-decimating DAE meshes...")
    tree = ET.parse(URDF_PATH)
    robot = tree.getroot()

    fallback_colors = {
        "base_link": np.array([0.28, 0.28, 0.28], dtype=np.float32),
        "pen_link": np.array([0.08, 0.18, 0.22], dtype=np.float32),
    }
    default_color = np.array([0.70, 0.74, 0.78], dtype=np.float32)

    all_data = {}
    total_faces = 0

    for link in robot.findall("link"):
        link_name = link.get("name")
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

            key = f"{link_name}/{mesh_path.name}"
            if key in all_data: continue

            print(f"  Processing {key}...")
            parts = load_dae_meshes(mesh_path, scale, DECIMATION_RATIO)
            processed = []
            for verts, faces, dae_color in parts:
                color = dae_color if dae_color is not None else fallback_colors.get(link_name, default_color)
                processed.append((verts, faces, color))
                total_faces += len(faces)
            all_data[key] = processed

    # Save
    save_dict = {}
    for key, parts in all_data.items():
        for pi, (verts, faces, color) in enumerate(parts):
            save_dict[f"{key}__part{pi}__v"] = verts
            save_dict[f"{key}__part{pi}__f"] = faces
            save_dict[f"{key}__part{pi}__c"] = color

    np.savez_compressed(CACHE_PATH, **save_dict)
    print(f"Saved {len(all_data)} meshes, {total_faces} total faces to {CACHE_PATH}")


if __name__ == "__main__":
    main()
