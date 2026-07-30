from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


# Editable text in SVG/PDF and a Chinese-capable sans-serif fallback stack.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Arial",
    "DejaVu Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8


OUT_DIR = Path(__file__).resolve().parent

INK = "#24313F"
MUTED = "#66788A"
LIGHT_LINE = "#CBD6E2"
BLUE = "#315E9E"
BLUE_2 = "#7399CA"
BLUE_BG = "#EAF1FA"
TEAL = "#2D8C89"
TEAL_BG = "#E8F5F3"
VIOLET = "#7667A8"
VIOLET_BG = "#F0EDF8"
ORANGE = "#CE7B35"
ORANGE_BG = "#FBF0E5"
GREEN = "#2D8A57"
GREEN_BG = "#E8F4EC"
RED = "#B95252"
RED_BG = "#F9ECEB"
WHITE = "#FFFFFF"


def rounded_box(ax, xy, width, height, *, fc, ec, lw=1.0, radius=0.02, z=1):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=INK, lw=1.25, rad=0.0, style="-|>", z=6, ls="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=ls,
        zorder=z,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, size=8, color=INK, weight="normal", ha="center", va="center", z=10):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z)


def draw_point_cloud(ax, x0, y0, w, h, rng):
    pts = rng.normal(size=(105, 2))
    pts[:, 0] = x0 + w * (0.50 + 0.25 * pts[:, 0])
    pts[:, 1] = y0 + h * (0.50 + 0.24 * pts[:, 1])
    mask = (pts[:, 0] > x0 + 0.08 * w) & (pts[:, 0] < x0 + 0.92 * w)
    mask &= (pts[:, 1] > y0 + 0.12 * h) & (pts[:, 1] < y0 + 0.88 * h)
    pts = pts[mask]
    colors = np.linspace(0.25, 0.95, len(pts))
    ax.scatter(pts[:, 0], pts[:, 1], s=3.5, c=colors, cmap="Blues", alpha=0.82, linewidths=0, zorder=4)
    # Workpiece edge suggested by two clean seams.
    ax.plot([x0 + 0.18 * w, x0 + 0.78 * w], [y0 + 0.26 * h, y0 + 0.38 * h], color=BLUE, lw=1.1, zorder=5)
    ax.plot([x0 + 0.78 * w, x0 + 0.70 * w], [y0 + 0.38 * h, y0 + 0.76 * h], color=BLUE, lw=1.1, zorder=5)


def draw_robot_state(ax, x0, y0, w, h):
    joints = np.array(
        [
            [x0 + 0.14 * w, y0 + 0.20 * h],
            [x0 + 0.34 * w, y0 + 0.34 * h],
            [x0 + 0.52 * w, y0 + 0.64 * h],
            [x0 + 0.75 * w, y0 + 0.58 * h],
            [x0 + 0.86 * w, y0 + 0.77 * h],
        ]
    )
    ax.add_patch(Rectangle((x0 + 0.04 * w, y0 + 0.10 * h), 0.18 * w, 0.10 * h, fc="#A9B7C5", ec=INK, lw=0.6, zorder=3))
    ax.plot(joints[:, 0], joints[:, 1], color=TEAL, lw=3.2, solid_capstyle="round", zorder=4)
    ax.scatter(joints[:, 0], joints[:, 1], s=16, facecolor=WHITE, edgecolor=INK, linewidth=0.7, zorder=5)
    label(ax, x0 + 0.58 * w, y0 + 0.12 * h, r"$q_s$", size=7, color=TEAL, weight="bold")


def draw_goal_pose(ax, x0, y0, w, h):
    cx, cy = x0 + 0.52 * w, y0 + 0.48 * h
    arrow(ax, (cx, cy), (cx + 0.29 * w, cy), color=RED, lw=1.4)
    arrow(ax, (cx, cy), (cx, cy + 0.32 * h), color=GREEN, lw=1.4)
    arrow(ax, (cx, cy), (cx - 0.20 * w, cy - 0.20 * h), color=BLUE, lw=1.4)
    ax.scatter([cx], [cy], s=18, c=ORANGE, edgecolor=INK, linewidth=0.6, zorder=7)
    label(ax, x0 + 0.52 * w, y0 + 0.12 * h, r"$T_g$", size=7, color=ORANGE, weight="bold")


def draw_cspace(ax, x0, y0, w, h):
    xx = np.linspace(0, 1, 60)
    yy = 0.51 + 0.15 * np.sin(2.3 * np.pi * xx) + 0.05 * np.cos(5 * np.pi * xx)
    ax.fill_between(x0 + xx * w, y0, y0 + yy * h, color="#D5DAE0", alpha=0.9, zorder=2)
    ax.plot(x0 + xx * w, y0 + yy * h, color=MUTED, lw=0.8, zorder=3)
    safe_x = x0 + np.linspace(0.1, 0.9, 22) * w
    safe_y = y0 + (0.72 + 0.06 * np.sin(np.linspace(0, 2 * np.pi, 22))) * h
    ax.scatter(safe_x, safe_y, s=5, c=VIOLET, alpha=0.78, linewidth=0, zorder=4)


def draw_condition_cards(ax, rng):
    x, y, w, h = 0.025, 0.21, 0.155, 0.66
    rounded_box(ax, (x, y), w, h, fc="#F8FAFC", ec=LIGHT_LINE, lw=1.0, radius=0.018)
    label(ax, x + w / 2, y + h - 0.043, "多源任务条件", size=10, weight="bold")
    card_h = 0.117
    card_gap = 0.022
    cards = [
        ("局部工件\n点云", BLUE_BG, BLUE, draw_point_cloud),
        ("机器人\n当前状态", TEAL_BG, TEAL, draw_robot_state),
        ("目标焊接\n位姿", ORANGE_BG, ORANGE, draw_goal_pose),
        ("关节空间\n先验", VIOLET_BG, VIOLET, draw_cspace),
    ]
    for i, (title, bg, edge, icon_fn) in enumerate(cards):
        cy = y + h - 0.095 - (i + 1) * card_h - i * card_gap
        rounded_box(ax, (x + 0.014, cy), w - 0.028, card_h, fc=bg, ec=edge, lw=0.75, radius=0.012, z=2)
        icon_fn(ax, x + 0.022, cy + 0.033, 0.050, card_h - 0.043, rng) if icon_fn is draw_point_cloud else icon_fn(ax, x + 0.022, cy + 0.033, 0.050, card_h - 0.043)
        label(ax, x + 0.082, cy + card_h / 2, title, size=6.8, ha="left", weight="bold")


def draw_fusion(ax):
    x, y, w, h = 0.205, 0.405, 0.112, 0.265
    rounded_box(ax, (x, y), w, h, fc=BLUE_BG, ec=BLUE, lw=1.1, radius=0.025)
    # Four inputs converge into one latent condition.
    inputs = [(x + 0.018, y + h * v) for v in (0.20, 0.40, 0.60, 0.80)]
    hub = (x + 0.064, y + h * 0.50)
    for p, c in zip(inputs, [BLUE, TEAL, ORANGE, VIOLET]):
        ax.add_patch(Circle(p, 0.0065, fc=c, ec=WHITE, lw=0.5, zorder=5))
        ax.plot([p[0], hub[0]], [p[1], hub[1]], color=c, lw=0.9, alpha=0.85, zorder=4)
    ax.add_patch(Circle(hub, 0.017, fc=BLUE, ec=INK, lw=0.7, zorder=6))
    arrow(ax, (hub[0] + 0.021, hub[1]), (x + w - 0.014, hub[1]), color=BLUE, lw=1.2)
    label(ax, x + w / 2, y + h - 0.035, "条件融合", size=9, weight="bold")
    label(ax, x + w / 2, y + 0.047, "统一条件嵌入", size=6.5, color=MUTED)


def smooth_curve(points, n=120):
    # Cubic Bézier interpolation for compact conceptual trajectory glyphs.
    p = np.asarray(points, dtype=float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t**2 * p[2] + t**3 * p[3]


def draw_diffusion(ax, rng):
    x, y, w, h = 0.342, 0.245, 0.365, 0.61
    rounded_box(ax, (x, y), w, h, fc="#FBFCFE", ec=BLUE, lw=1.2, radius=0.022)
    label(ax, x + 0.018, y + h - 0.053, "点云条件关节空间\nDiffusion Policy", size=9.2, weight="bold", ha="left")

    # Denoising strip.
    strip_y = y + 0.350
    strip_h = 0.118
    rounded_box(ax, (x + 0.020, strip_y), w - 0.040, strip_h, fc=BLUE_BG, ec=LIGHT_LINE, lw=0.8, radius=0.013)
    centers = np.linspace(x + 0.055, x + w - 0.055, 5)
    for idx, cx in enumerate(centers):
        sigma = 0.022 * (1.0 - 0.18 * idx)
        px = cx + rng.normal(0, sigma, 25)
        py = strip_y + strip_h / 2 + rng.normal(0, 0.020 * (1.0 - 0.15 * idx), 25)
        ax.scatter(px, py, s=3.0 + idx * 0.35, c=BLUE_2, alpha=0.30 + 0.12 * idx, linewidth=0, zorder=4)
        if idx < len(centers) - 1:
            arrow(ax, (cx + 0.024, strip_y + strip_h / 2), (centers[idx + 1] - 0.025, strip_y + strip_h / 2), color=MUTED, lw=0.8)
    label(ax, centers[0], strip_y + strip_h - 0.021, "噪声控制点", size=6.7, color=MUTED)
    label(ax, centers[-1], strip_y + strip_h - 0.021, "候选", size=6.7, color=BLUE, weight="bold")
    label(ax, x + w / 2, strip_y - 0.020, "迭代去噪", size=7.2, color=BLUE, weight="bold")

    # Candidate control-point trajectories.
    traj_y = y + 0.155
    traj_h = 0.170
    rounded_box(ax, (x + 0.020, traj_y), w - 0.040, traj_h, fc=VIOLET_BG, ec=LIGHT_LINE, lw=0.8, radius=0.013)
    x_start = x + 0.050
    x_end = x + w - 0.050
    colors = [VIOLET, BLUE_2, TEAL]
    offsets = [0.018, 0.000, -0.016]
    for i, (c, off) in enumerate(zip(colors, offsets)):
        pts = [
            [x_start, traj_y + 0.048],
            [x + 0.145, traj_y + 0.145 + off],
            [x + 0.245, traj_y + 0.025 - off],
            [x_end, traj_y + 0.115],
        ]
        curve = smooth_curve(pts)
        ax.plot(curve[:, 0], curve[:, 1], color=c, lw=1.3 if i else 1.8, alpha=0.78, zorder=4)
        cp_idx = np.linspace(0, len(curve) - 1, 7).astype(int)
        ax.scatter(curve[cp_idx, 0], curve[cp_idx, 1], s=9, fc=WHITE, ec=c, lw=0.8, zorder=5)
    label(ax, x + 0.036, traj_y + traj_h - 0.022, "低维 B-spline 控制点候选", size=7.6, weight="bold", ha="left", color=VIOLET)

    # Shared-variable bridge.
    bridge_y = y + 0.046
    rounded_box(ax, (x + 0.020, bridge_y), w - 0.040, 0.065, fc=WHITE, ec=VIOLET, lw=1.0, radius=0.012)
    for cx in np.linspace(x + 0.060, x + w - 0.060, 10):
        ax.add_patch(Circle((cx, bridge_y + 0.046), 0.0045, fc=VIOLET, ec=WHITE, lw=0.4, zorder=5))
    label(ax, x + w / 2, bridge_y + 0.022, "扩散生成与安全修正共享控制点变量", size=6.8, weight="bold", color=VIOLET)


def draw_stage1(ax):
    x, y, w, h = 0.492, 0.718, 0.190, 0.088
    rounded_box(ax, (x, y), w, h, fc=ORANGE_BG, ec=ORANGE, lw=1.2, radius=0.015, z=7)
    ax.add_patch(Circle((x + 0.022, y + h / 2), 0.013, fc=ORANGE, ec=WHITE, lw=0.7, zorder=9))
    label(ax, x + 0.022, y + h / 2, "1", size=8.3, color=WHITE, weight="bold")
    label(ax, x + 0.043, y + 0.057, "后期去噪：CBF-QP 安全投影", size=7.4, color=ORANGE, weight="bold", ha="left")
    label(ax, x + 0.043, y + 0.027, "投影结果反馈下一步去噪", size=6.5, color=MUTED, ha="left")
    # Feedback loop into the denoising strip.
    arrow(ax, (x + w * 0.78, y), (0.625, 0.655), color=ORANGE, lw=1.35, rad=-0.12)
    arrow(ax, (0.548, 0.655), (x + w * 0.18, y), color=ORANGE, lw=1.0, rad=-0.12, style="-|>", ls="--")


def draw_stage2(ax):
    x, y, w, h = 0.735, 0.290, 0.155, 0.455
    rounded_box(ax, (x, y), w, h, fc=GREEN_BG, ec=GREEN, lw=1.2, radius=0.020)
    ax.add_patch(Circle((x + 0.023, y + h - 0.045), 0.014, fc=GREEN, ec=WHITE, lw=0.7, zorder=6))
    label(ax, x + 0.023, y + h - 0.045, "2", size=8.3, color=WHITE, weight="bold")
    label(ax, x + 0.045, y + h - 0.044, "最终安全保证", size=9.2, color=GREEN, weight="bold", ha="left")

    # Certification card.
    rounded_box(ax, (x + 0.018, y + 0.260), w - 0.036, 0.102, fc=WHITE, ec=GREEN, lw=0.8, radius=0.012)
    ax.add_patch(Circle((x + 0.043, y + 0.311), 0.016, fc=GREEN_BG, ec=GREEN, lw=1.0, zorder=4))
    ax.plot([x + 0.035, x + 0.041, x + 0.052], [y + 0.310, y + 0.302, y + 0.321], color=GREEN, lw=1.6, zorder=5)
    label(ax, x + 0.068, y + 0.326, "安全认证", size=7.6, weight="bold", ha="left")
    label(ax, x + 0.068, y + 0.296, "全轨迹约束检查", size=6.6, color=MUTED, ha="left")

    arrow(ax, (x + w / 2, y + 0.252), (x + w / 2, y + 0.216), color=GREEN, lw=1.0)
    label(ax, x + w / 2, y + 0.231, "若局部风险", size=6.3, color=MUTED)

    # Local repair card.
    rounded_box(ax, (x + 0.018, y + 0.095), w - 0.036, 0.106, fc=WHITE, ec=GREEN, lw=0.8, radius=0.012)
    xx = np.linspace(x + 0.037, x + w - 0.037, 60)
    yy_bad = y + 0.132 + 0.028 * np.sin(np.linspace(0, np.pi, 60))
    yy_good = y + 0.132 + 0.055 * np.sin(np.linspace(0, np.pi, 60))
    ax.plot(xx, yy_bad, color=RED, lw=0.9, ls="--", alpha=0.8, zorder=4)
    ax.plot(xx, yy_good, color=GREEN, lw=1.5, zorder=5)
    ax.add_patch(Circle((x + w / 2, y + 0.133), 0.014, fc=RED_BG, ec=RED, lw=0.7, alpha=0.9, zorder=3))
    label(ax, x + w / 2, y + 0.184, "局部 CBF-QP 修正", size=7.1, weight="bold", color=GREEN)
    label(ax, x + w / 2, y + 0.108, "仅调整风险邻域", size=6.2, color=MUTED)
    label(ax, x + w / 2, y + 0.047, "硬安全约束参与\n生成与修正", size=6.6, color=GREEN, weight="bold")


def draw_output(ax):
    x, y, w, h = 0.915, 0.255, 0.068, 0.545
    rounded_box(ax, (x, y), w, h, fc="#F8FAFC", ec=INK, lw=1.0, radius=0.018)
    label(ax, x + w / 2, y + h - 0.045, "安全轨迹", size=8.8, weight="bold")
    # Top-down conceptual safe transfer path around a workpiece obstacle.
    obstacle = Polygon(
        [
            (x + 0.016, y + 0.236),
            (x + 0.052, y + 0.244),
            (x + 0.057, y + 0.350),
            (x + 0.024, y + 0.365),
        ],
        closed=True,
        fc="#D7DEE5",
        ec=MUTED,
        lw=0.7,
        zorder=2,
    )
    ax.add_patch(obstacle)
    pts = [
        [x + 0.034, y + 0.092],
        [x + 0.002, y + 0.195],
        [x + 0.072, y + 0.390],
        [x + 0.036, y + 0.463],
    ]
    curve = smooth_curve(pts)
    ax.plot(curve[:, 0], curve[:, 1], color=GREEN, lw=2.0, zorder=5)
    cp_idx = np.linspace(0, len(curve) - 1, 6).astype(int)
    ax.scatter(curve[cp_idx, 0], curve[cp_idx, 1], s=11, fc=WHITE, ec=GREEN, lw=0.8, zorder=6)
    ax.scatter([curve[0, 0]], [curve[0, 1]], s=24, fc=BLUE, ec=WHITE, lw=0.7, zorder=7)
    ax.scatter([curve[-1, 0]], [curve[-1, 1]], s=30, marker="*", fc=ORANGE, ec=WHITE, lw=0.6, zorder=7)
    label(ax, x + w / 2, y + 0.045, "平滑 · 连续\n无碰撞", size=7.0, color=GREEN, weight="bold")


def add_contribution_ribbon(ax):
    y = 0.075
    items = [
        (0.170, BLUE_BG, BLUE, "点云条件关节空间扩散规划"),
        (0.470, ORANGE_BG, ORANGE, "双阶段 CBF-QP 安全引导"),
        (0.770, VIOLET_BG, VIOLET, "B-spline 低维平滑参数化"),
    ]
    for idx, (cx, bg, edge, text) in enumerate(items, start=1):
        rounded_box(ax, (cx - 0.125, y - 0.026), 0.250, 0.052, fc=bg, ec=edge, lw=0.85, radius=0.012)
        ax.add_patch(Circle((cx - 0.103, y), 0.011, fc=edge, ec=WHITE, lw=0.5, zorder=5))
        label(ax, cx - 0.103, y, str(idx), size=7.3, color=WHITE, weight="bold")
        label(ax, cx - 0.083, y, text, size=7.5, color=edge, weight="bold", ha="left")


def build_figure():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(7.2, 4.25), facecolor=WHITE)
    ax = fig.add_axes([0.015, 0.025, 0.97, 0.95])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(
        ax,
        0.025,
        0.955,
        "面向机器人焊接空移段的点云条件关节空间安全 Diffusion Policy",
        size=12.2,
        weight="bold",
        ha="left",
    )
    label(ax, 0.025, 0.912, "从局部三维环境表征直接生成平滑、连续且满足硬安全约束的关节轨迹", size=7.8, color=MUTED, ha="left")

    draw_condition_cards(ax, rng)
    draw_fusion(ax)
    draw_diffusion(ax, rng)
    draw_stage1(ax)
    draw_stage2(ax)
    draw_output(ax)

    arrow(ax, (0.180, 0.540), (0.205, 0.540), color=BLUE, lw=1.5)
    arrow(ax, (0.317, 0.540), (0.342, 0.540), color=BLUE, lw=1.5)
    arrow(ax, (0.707, 0.515), (0.735, 0.515), color=GREEN, lw=1.5)
    arrow(ax, (0.890, 0.515), (0.915, 0.515), color=GREEN, lw=1.5)

    add_contribution_ribbon(ax)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUT_DIR / "robot_welding_safe_diffusion_framework_zh"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor=WHITE)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=WHITE)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


if __name__ == "__main__":
    main()
