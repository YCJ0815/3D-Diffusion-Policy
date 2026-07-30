import json
import numpy as np
import sys
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

GLB_PATH = "/Users/ycj/Desktop/Project/app界面设计/code_generate_2/public/data/projects/A-102/cad_files/sub_assembly/SA-BOT-LON-01.glb"
JSON_PATH = "/Users/ycj/Desktop/Project/app界面设计/code_generate_2/public/data/projects/A-102/weld_seam/sub_assembly/SA-BOT-LON-01.json"
OUT_PATH = os.path.expanduser("~/Desktop/SA-BOT-LON-01_welds.png")


def compute_face_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norms, 1e-10)


def load_glb_mesh(path):
    scene = trimesh.load(path)
    verts_list = []
    faces_list = []
    voffset = 0
    for name, geom in scene.geometry.items():
        if hasattr(geom, "vertices") and hasattr(geom, "faces"):
            verts_list.append(np.asarray(geom.vertices, dtype=np.float64))
            faces_list.append(np.asarray(geom.faces, dtype=np.int32) + voffset)
            voffset += len(geom.vertices)
    if not verts_list:
        raise RuntimeError("No mesh geometry found")
    return np.vstack(verts_list), np.vstack(faces_list)


def main():
    print("Loading assembly GLB...")
    verts, faces = load_glb_mesh(GLB_PATH)

    print("Loading weld data...")
    with open(JSON_PATH) as f:
        data = json.load(f)

    weld_seams = data["weld_seams"]
    contact_edges = data["contact_edges"]

    # Rx(90°): align GLB Y-up convention to weld Z-up convention
    # Matrix: [[1,0,0],[0,0,-1],[0,1,0]]  →  X'=X, Y'=-Z, Z'=Y
    R = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    v_rot = verts @ R.T

    # Convert to meters (no offset — rotation alone is optimal based on nearest-neighbor analysis)
    verts_m = v_rot / 1000.0

    print(f"Mesh: {len(verts_m)} vertices, {len(faces)} faces")
    bounds_str = f"x=[{verts_m[:,0].min():.2f},{verts_m[:,0].max():.2f}] " \
                 f"y=[{verts_m[:,1].min():.2f},{verts_m[:,1].max():.2f}] " \
                 f"z=[{verts_m[:,2].min():.2f},{verts_m[:,2].max():.2f}]"
    print(f"Bounds (m): {bounds_str}")

    # Collect weld segments
    weld_segments = []
    for seam_name, seam in weld_seams.items():
        edge_ids = seam["edge_ids"]
        for eid in edge_ids:
            if eid in contact_edges and "samples" in contact_edges[eid]:
                pts = np.array(contact_edges[eid]["samples"], dtype=np.float64)
                if len(pts) >= 2:
                    weld_segments.append({
                        "weld": seam_name,
                        "edge": eid,
                        "points_m": pts / 1000.0,
                        "is_noncont": len(edge_ids) > 1,
                    })

    num_welds = len(weld_seams)
    weld_order = list(weld_seams.keys())
    all_cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c,
                 plt.cm.Set3, plt.cm.Paired]
    colors = []
    for i in range(num_welds):
        cm = all_cmaps[(i // 20) % len(all_cmaps)]
        colors.append(np.array(cm((i % 20) / 20.0)))
    weld_color_map = {wn: colors[i] for i, wn in enumerate(weld_order)}

    print(f"Weld seams: {num_welds}, segments: {len(weld_segments)}")
    print(f"Transformation: Rx(90°)")

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((1, 1, 1, 0))
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")
    ax.set_axis_off()

    light_dir = np.array([0.35, -0.35, 0.87])
    light_dir /= np.linalg.norm(light_dir)

    # Render assembly mesh
    norms = compute_face_normals(verts_m, faces)
    shade = np.clip(norms @ light_dir, 0.0, 1.0)[:, None]
    base_color = np.array([0.72, 0.74, 0.77])
    colors_mesh = np.clip(base_color * (0.45 + 0.55 * shade), 0, 1)
    rgba_mesh = np.concatenate([colors_mesh, np.full((colors_mesh.shape[0], 1), 0.35)], axis=1)
    poly = Poly3DCollection(verts_m[faces], linewidths=0.0)
    poly.set_facecolor(rgba_mesh)
    poly.set_edgecolor("none")
    ax.add_collection3d(poly)

    # Render weld seams
    scene_pts = [verts_m]

    for seg in weld_segments:
        pts_m = seg["points_m"]
        color = weld_color_map[seg["weld"]]
        scene_pts.append(pts_m)

        for i in range(len(pts_m) - 1):
            ax.plot(
                pts_m[i:i+2, 0], pts_m[i:i+2, 1], pts_m[i:i+2, 2],
                color=color[:3], linewidth=5.0, alpha=1.0,
                solid_capstyle="round",
            )

    all_pts = np.vstack(scene_pts)
    centers = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
    radius = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0 * 1.08
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    ax.view_init(elev=28, azim=-40)

    fig.savefig(OUT_PATH, dpi=250, bbox_inches="tight", facecolor="none",
                transparent=True, pad_inches=0.02)
    plt.close(fig)
    print(f"Saved to {OUT_PATH}")

    if "--no-show" not in sys.argv:
        import matplotlib
        matplotlib.use("TkAgg")
        fig2 = plt.figure(figsize=(16, 12))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.set_facecolor((1, 1, 1, 0))
        ax2.grid(False)
        for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("none")
        ax2.set_axis_off()
        p2 = Poly3DCollection(verts_m[faces], linewidths=0.0)
        p2.set_facecolor(rgba_mesh); p2.set_edgecolor("none")
        ax2.add_collection3d(p2)
        for seg in weld_segments:
            pts_m = seg["points_m"]
            color = weld_color_map[seg["weld"]]
            for i in range(len(pts_m) - 1):
                ax2.plot(pts_m[i:i+2,0], pts_m[i:i+2,1], pts_m[i:i+2,2],
                         color=color[:3], linewidth=5.0, alpha=1.0)
        ax2.set_xlim(centers[0]-radius, centers[0]+radius)
        ax2.set_ylim(centers[1]-radius, centers[1]+radius)
        ax2.set_zlim(centers[2]-radius, centers[2]+radius)
        ax2.view_init(elev=28, azim=-40)
        plt.show()


if __name__ == "__main__":
    main()
