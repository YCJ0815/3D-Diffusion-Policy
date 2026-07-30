import json, math, os, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH = PROJECT_ROOT / "config" / "ur5e_with_pen.urdf"
RESULTS_DIR = PROJECT_ROOT / "data" / "raw_data" / "simple_results" / "job_039"
JOBS_DIR = PROJECT_ROOT / "data" / "raw_data" / "simple_jobs" / "job_039"
OUT_PATH = RESULTS_DIR / "trajectory_plots" / "transition_0001_0012_arm_trajectory.png"
NUM_POSES = 6

def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],[cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],[-sp, sr*cp, cr*cp]])

def transform_from_origin(origin):
    xyz = np.zeros(3); rpy_ = np.zeros(3)
    if origin is not None:
        if origin.get("xyz"): xyz = np.array([float(v) for v in origin.get("xyz").split()])
        if origin.get("rpy"): rpy_ = np.array([float(v) for v in origin.get("rpy").split()])
    mat = np.eye(4); mat[:3,:3]=rpy_matrix(rpy_); mat[:3,3]=xyz
    return mat

def axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float); norm = np.linalg.norm(axis)
    if norm == 0: return np.eye(4)
    x, y, z = axis / norm; c, s = math.cos(angle), math.sin(angle); c1 = 1.0 - c
    rot = np.array([[c+x*x*c1, x*y*c1-z*s, x*z*c1+y*s],[y*x*c1+z*s, c+y*y*c1, y*z*c1-x*s],[z*x*c1-y*s, z*y*c1+x*s, c+z*z*c1]])
    mat = np.eye(4); mat[:3,:3]=rot
    return mat

def parse_float3(value, default):
    if not value: return np.array(default, dtype=float)
    return np.array([float(v) for v in value.split()])

def parse_float_array(text):
    return np.fromstring(text or "", sep=" ", dtype=float)

def collada_id_ref(value):
    return value[1:] if value and value.startswith("#") else value

def collada_node_transform(node):
    mat = np.eye(4); el = node.find("c:matrix", COLLADA_NS)
    if el is not None and el.text: mat = np.array([float(v) for v in el.text.split()]).reshape(4, 4)
    return mat

def read_collada_materials(root):
    effects = {}
    for e in root.findall(".//c:library_effects/c:effect", COLLADA_NS):
        d = e.find(".//c:diffuse/c:color", COLLADA_NS)
        if d is not None and d.text: effects[e.get("id")] = np.array([float(v) for v in d.text.split()][:3], dtype=np.float32)
    mats = {}
    for m in root.findall(".//c:library_materials/c:material", COLLADA_NS):
        inst = m.find("c:instance_effect", COLLADA_NS); eid = collada_id_ref(inst.get("url")) if inst is not None else None
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
            stride = int(acc.get("stride", "1")); vals = parse_float_array(fa.text)
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
        for b in inst.findall(".//c:instance_material", COLLADA_NS): mat_sym[b.get("symbol")] = collada_id_ref(b.get("target"))
        yield collada_id_ref(inst.get("url")), node_tf, mat_sym
    for child in node.findall("c:node", COLLADA_NS): yield from iter_collada_instances(child, node_tf)

def load_dae_meshes(path, scale):
    root = ET.parse(path).getroot()
    materials = read_collada_materials(root); geoms = read_collada_geometries(root)
    meshes = []
    sn = root.findall(".//c:library_visual_scenes/c:visual_scene/c:node", COLLADA_NS)
    for node in sn:
        for gid, node_tf, mat_sym in iter_collada_instances(node, np.eye(4)):
            for pos, faces, sym in geoms.get(gid, []):
                verts = pos @ node_tf[:3,:3].T + node_tf[:3,3]; verts *= scale
                color = materials.get(mat_sym.get(sym, sym))
                meshes.append((verts, faces, color))
    if not meshes:
        for parts in geoms.values():
            for pos, faces, sym in parts: meshes.append((pos * scale, faces, materials.get(sym)))
    return meshes

def package_path(filename, urdf_path):
    if filename.startswith("package://urdf-pen/"): return urdf_path.parent / "robot-model" / filename.removeprefix("package://urdf-pen/")
    return urdf_path.parent / filename

def compute_face_normals(verts, faces):
    v0=verts[faces[:,0]]; v1=verts[faces[:,1]]; v2=verts[faces[:,2]]
    n=np.cross(v1-v0,v2-v0); norms=np.linalg.norm(n,axis=1,keepdims=True)
    return n/np.maximum(norms,1e-10)

print("Loading URDF and meshes...")
tree = ET.parse(URDF_PATH); robot = tree.getroot()

children_by_parent = {}
for joint in robot.findall("joint"):
    p=joint.find("parent").get("link"); c=joint.find("child").get("link")
    jt=joint.get("type", "fixed")
    ot=transform_from_origin(joint.find("origin"))
    ax=parse_float3(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, [1,0,0])
    children_by_parent.setdefault(p,[]).append({"name":joint.get("name"),"type":jt,"parent":p,"child":c,"origin_tf":ot,"axis":ax})

jnames = [j["name"] for j in sum(children_by_parent.values(),[]) if j["type"] in {"revolute","continuous"}]

def compute_fk(angles):
    a=list(angles); ltf={"world":np.eye(4)}
    def visit(link,tf):
        ltf[link]=tf
        for jt in children_by_parent.get(link,[]):
            mot=np.eye(4)
            if jt["type"] in {"revolute","continuous"}:
                if jt["name"] in jnames:
                    idx=jnames.index(jt["name"])
                    if idx<len(a): mot=axis_angle_matrix(jt["axis"],float(a[idx]))
            visit(jt["child"],tf@jt["origin_tf"]@mot)
    visit("world",np.eye(4)); return ltf

link_meshes = {}
fc = {"base_link":np.array([0.28,0.28,0.28],dtype=np.float32),"pen_link":np.array([0.08,0.18,0.22],dtype=np.float32)}
dc = np.array([0.70,0.74,0.78],dtype=np.float32)

for link in robot.findall("link"):
    ln = link.get("name"); parts = []
    for item in link.findall("visual"):
        geom=item.find("geometry")
        if geom is None or geom.find("mesh") is None: continue
        me=geom.find("mesh"); fn=me.get("filename")
        if not fn: continue
        mp=package_path(fn,URDF_PATH)
        if not mp.exists(): continue
        sc=parse_float3(me.get("scale"),[1.0,1.0,1.0])
        ot=transform_from_origin(item.find("origin"))
        for v,f,dae_c in load_dae_meshes(mp,sc):
            lc=dae_c if dae_c is not None else fc.get(ln,dc)
            parts.append((v,f,lc,ot))
    if parts: link_meshes[ln]=parts

print(f"Links: {list(link_meshes.keys())}, faces: {sum(len(f) for ms in link_meshes.values() for _,f,_,_ in ms)}")

# Load workpiece
stl = JOBS_DIR / "workpiece_sim.stl"
wp = trimesh.load(stl)
wv = np.asarray(wp.vertices,dtype=np.float64)*0.001+np.array([0.5,0.0,0.0])
wf = np.asarray(wp.faces,dtype=np.int32)

# Load transition
npz = np.load(RESULTS_DIR / "transition_0001_0012.npz")
qp = npz["q_playback"]; sx= npz["start_xyz"]; ex=npz["end_xyz"]

n_poses = min(NUM_POSES, len(qp))
indices = np.linspace(0,len(qp)-1,n_poses,dtype=int)

fig = plt.figure(figsize=(12,9)); fig.patch.set_alpha(0)
ax = fig.add_subplot(111,projection="3d"); ax.set_facecolor((1,1,1,0)); ax.grid(False)
for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: pane.fill=False; pane.set_edgecolor("none")
ax.set_axis_off()

ld = np.array([0.3,-0.5,0.8]); ld/=np.linalg.norm(ld)
av = []

wn = compute_face_normals(wv,wf); ws = np.clip(wn@ld,0.0,1.0)[:,None]
wc = np.clip(np.array([[0.592,0.627,0.682]])*(0.4+0.6*ws),0,1)
wr = np.concatenate([wc,np.full((wc.shape[0],1),0.45)],axis=1)
wpoly = Poly3DCollection(wv[wf],linewidths=0.0); wpoly.set_facecolor(wr); wpoly.set_edgecolor("none")
ax.add_collection3d(wpoly); av.append(wv)

for wi,idx in enumerate(indices):
    q=qp[idx]; progress=wi/max(n_poses-1,1)
    alpha=0.10+0.90*abs(2*progress-1)
    ltf=compute_fk(q)
    for ln,ms in link_meshes.items():
        if ln not in ltf: continue
        wt=ltf[ln]
        for v,f,lc,ot in ms:
            ft=wt@ot; R=ft[:3,:3]; t=ft[:3,3]
            tv=v@R.T+t; av.append(tv)
            rgba=np.append(lc,alpha)
            poly=Poly3DCollection(tv[f],linewidths=0.02)
            poly.set_facecolor(rgba); poly.set_edgecolor((0.18,0.22,0.25,min(alpha*0.2,1.0)))
            ax.add_collection3d(poly)

ax.scatter(*sx,c="limegreen",s=100,marker="o",edgecolors="darkgreen",linewidths=1.0,alpha=0.15,zorder=300)
ax.scatter(*ex,c="crimson",s=100,marker="o",edgecolors="darkred",linewidths=1.0,alpha=0.90,zorder=300)

apt=np.vstack(av); mn,mx=apt.min(axis=0),apt.max(axis=0); ct=(mn+mx)/2.0; rd=(mx-mn).max()/2.0*1.08
ax.set_xlim(ct[0]-rd,ct[0]+rd); ax.set_ylim(ct[1]-rd,ct[1]+rd); ax.set_zlim(ct[2]-rd,ct[2]+rd)
ax.view_init(elev=22,azim=-48)

OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT_PATH,dpi=180,bbox_inches="tight",facecolor="none",transparent=True,pad_inches=0.02)
plt.close(fig)
print(f"Saved to {OUT_PATH}")
