from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-network-architecture")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# Mandatory publication settings: editable SVG text and TrueType PDF text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8


OUT_DIR = Path(__file__).resolve().parent

INK = "#272727"
MUTED = "#606060"
LIGHT = "#D8D8D8"
PALE = "#F7F7F8"
WHITE = "#FFFFFF"

BLUE_DARK = "#484878"
BLUE = "#7884B4"
BLUE_SOFT = "#E4E8F5"
VIOLET = "#7968A8"
VIOLET_SOFT = "#EEEAF7"
TEAL = "#3C8F8A"
TEAL_SOFT = "#E3F2F0"
ROSE = "#B66A7A"
ROSE_SOFT = "#F4E5E9"
ORANGE = "#C67835"
ORANGE_SOFT = "#F8EBDD"
GREEN = "#3B8763"
GREEN_SOFT = "#E5F1EA"


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    fc=WHITE,
    ec=LIGHT,
    lw=0.8,
    radius=0.012,
    z=2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=INK,
    lw=0.9,
    rad=0.0,
    style="-|>",
    ls="-",
    z=8,
    mutation=7,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1,
        shrinkB=1,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(
    ax,
    x,
    y,
    text,
    *,
    size=7,
    color=INK,
    weight="normal",
    ha="center",
    va="center",
    z=10,
    rotation=0,
):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=z,
    )


def panel_label(ax, x, y, text):
    label(ax, x, y, text, size=9, weight="bold", ha="left", va="top")


def draw_point_cloud(ax, x, y, w, h):
    rng = np.random.default_rng(12)
    u = rng.uniform(0.10, 0.90, 80)
    v = 0.47 + 0.22 * np.sin(2.2 * np.pi * u) + rng.normal(0, 0.08, u.size)
    mask = (v > 0.08) & (v < 0.92)
    ax.scatter(
        x + w * u[mask],
        y + h * v[mask],
        s=2.0,
        c=BLUE,
        alpha=0.62,
        linewidths=0,
        zorder=5,
    )
    ax.plot(
        [x + 0.15 * w, x + 0.78 * w, x + 0.67 * w],
        [y + 0.27 * h, y + 0.37 * h, y + 0.77 * h],
        color=BLUE_DARK,
        lw=0.8,
        zorder=6,
    )


def draw_feature_vector(ax, x, y, w, h, color, count=8):
    gap = w * 0.035
    bar_w = (w - gap * (count - 1)) / count
    values = np.array([0.42, 0.76, 0.55, 0.88, 0.36, 0.68, 0.51, 0.81])
    for i in range(count):
        bh = h * values[i % len(values)]
        ax.add_patch(
            Rectangle(
                (x + i * (bar_w + gap), y),
                bar_w,
                bh,
                facecolor=color,
                edgecolor="none",
                alpha=0.72,
                zorder=5,
            )
        )


def draw_cspace_samples(ax, x, y, w, h):
    t = np.linspace(0.08, 0.92, 25)
    upper = 0.68 + 0.10 * np.sin(2.5 * np.pi * t)
    lower = 0.33 + 0.07 * np.cos(2.0 * np.pi * t)
    ax.fill_between(
        x + w * t,
        y + h * lower,
        y + h * upper,
        color=VIOLET_SOFT,
        zorder=3,
    )
    ax.scatter(
        x + w * t,
        y + h * (0.50 + 0.10 * np.sin(3.2 * np.pi * t)),
        s=3,
        color=VIOLET,
        zorder=5,
    )


def draw_resblock(ax, x, y, w, h, title, shape, *, fc, ec):
    rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=0.9, radius=0.010)
    label(ax, x + w / 2, y + h * 0.67, title, size=6.4, weight="bold")
    label(ax, x + w / 2, y + h * 0.38, shape, size=5.8, color=MUTED)
    label(ax, x + w / 2, y + h * 0.16, "Conv–GN–Mish · FiLM", size=4.8, color=MUTED)


def draw_condition_encoder(ax):
    panel_label(ax, 0.014, 0.955, "a")
    label(
        ax,
        0.043,
        0.942,
        "Multi-modal condition encoder",
        size=8.4,
        weight="bold",
        ha="left",
    )

    # Workspace point-cloud branch.
    rounded_box(ax, 0.034, 0.725, 0.082, 0.132, fc=BLUE_SOFT, ec=BLUE, radius=0.010)
    draw_point_cloud(ax, 0.041, 0.754, 0.068, 0.072)
    label(ax, 0.075, 0.742, "Point cloud", size=5.8, weight="bold")
    label(ax, 0.075, 0.727, r"$1024\times3$", size=5.2, color=MUTED)

    rounded_box(ax, 0.139, 0.713, 0.110, 0.154, fc=BLUE_SOFT, ec=BLUE, radius=0.010)
    label(ax, 0.194, 0.840, "PointNet", size=6.6, weight="bold")
    label(ax, 0.194, 0.813, "point MLP", size=5.2, color=MUTED)
    label(ax, 0.194, 0.787, r"$3\!\to\!64\!\to\!128\!\to\!256$", size=5.6)
    label(ax, 0.194, 0.758, "max pool", size=5.2, color=MUTED)
    label(ax, 0.194, 0.734, r"projection $\to64$", size=5.5)
    arrow(ax, (0.116, 0.791), (0.139, 0.791), color=BLUE_DARK)

    # Low-dimensional branch.
    low_inputs = [
        (r"$p_g$", "3"),
        (r"$r_g$", "6"),
        (r"$q_s$", "6"),
        (r"$q_g$", "6"),
    ]
    y0 = 0.523
    for i, (symbol, dim) in enumerate(low_inputs):
        yy = y0 + i * 0.045
        rounded_box(ax, 0.034, yy, 0.050, 0.031, fc=PALE, ec=LIGHT, radius=0.006)
        label(ax, 0.052, yy + 0.016, symbol, size=5.7, weight="bold")
        label(ax, 0.074, yy + 0.016, dim, size=4.8, color=MUTED)
        arrow(ax, (0.084, yy + 0.016), (0.113, yy + 0.016), color=MUTED, lw=0.65)

    rounded_box(ax, 0.116, 0.516, 0.133, 0.192, fc=BLUE_SOFT, ec=BLUE, radius=0.010)
    label(ax, 0.183, 0.680, "State MLP bank", size=6.4, weight="bold")
    label(ax, 0.183, 0.651, r"$d\!\to\!32\!\to\!32$", size=5.7)
    label(ax, 0.183, 0.623, "one MLP per state", size=5.0, color=MUTED)
    for i in range(4):
        yy = 0.546 + i * 0.022
        ax.plot([0.142, 0.224], [yy, yy], color=BLUE, lw=2.0, alpha=0.36 + 0.12 * i)
    label(ax, 0.183, 0.530, r"$4\times32$", size=5.3, color=MUTED)

    # Observation embedding.
    rounded_box(ax, 0.272, 0.636, 0.090, 0.124, fc=BLUE_SOFT, ec=BLUE_DARK, lw=1.0)
    label(ax, 0.317, 0.730, r"$z_{\mathrm{obs}}$", size=7.0, weight="bold")
    draw_feature_vector(ax, 0.286, 0.668, 0.062, 0.035, BLUE_DARK)
    label(ax, 0.317, 0.651, "192-D", size=5.5, color=MUTED)
    arrow(ax, (0.249, 0.790), (0.272, 0.714), color=BLUE_DARK, rad=0.10)
    arrow(ax, (0.249, 0.606), (0.272, 0.672), color=BLUE_DARK, rad=-0.10)

    # C-space branch.
    rounded_box(ax, 0.034, 0.355, 0.082, 0.115, fc=VIOLET_SOFT, ec=VIOLET, radius=0.010)
    draw_cspace_samples(ax, 0.041, 0.382, 0.068, 0.058)
    label(ax, 0.075, 0.371, "C-space", size=5.8, weight="bold")
    label(ax, 0.075, 0.357, r"$128\times2$", size=5.1, color=MUTED)

    rounded_box(ax, 0.139, 0.344, 0.110, 0.137, fc=VIOLET_SOFT, ec=VIOLET, radius=0.010)
    label(ax, 0.194, 0.455, "C-space encoder", size=6.3, weight="bold")
    label(ax, 0.194, 0.428, r"$2\!\to\!32\!\to\!64$", size=5.7)
    label(ax, 0.194, 0.401, "mean + max pool", size=5.0, color=MUTED)
    label(ax, 0.194, 0.374, r"projection $\to64$", size=5.4)
    arrow(ax, (0.116, 0.413), (0.139, 0.413), color=VIOLET)

    rounded_box(ax, 0.272, 0.357, 0.090, 0.100, fc=VIOLET_SOFT, ec=VIOLET, lw=1.0)
    label(ax, 0.317, 0.430, r"$z_{\mathrm{C}}$", size=7.0, weight="bold")
    draw_feature_vector(ax, 0.286, 0.384, 0.062, 0.029, VIOLET)
    label(ax, 0.317, 0.369, "64-D", size=5.5, color=MUTED)
    arrow(ax, (0.249, 0.413), (0.272, 0.413), color=VIOLET)

    # Fusion.
    rounded_box(ax, 0.385, 0.493, 0.091, 0.170, fc=TEAL_SOFT, ec=TEAL, lw=1.0)
    label(ax, 0.431, 0.634, "Feature fusion", size=6.4, weight="bold")
    label(ax, 0.431, 0.604, "concat", size=5.1, color=MUTED)
    label(ax, 0.431, 0.577, r"$192+64$", size=5.8)
    label(ax, 0.431, 0.548, "Linear–LN–ReLU", size=5.0, color=MUTED)
    label(ax, 0.431, 0.517, r"$z_c\in\mathbb{R}^{256}$", size=6.2, weight="bold")
    arrow(ax, (0.362, 0.698), (0.385, 0.606), color=BLUE_DARK, rad=0.10)
    arrow(ax, (0.362, 0.407), (0.385, 0.542), color=VIOLET, rad=-0.10)


def draw_temporal_unet(ax):
    panel_label(ax, 0.493, 0.955, "b")
    label(
        ax,
        0.522,
        0.942,
        "FiLM-conditioned temporal denoiser",
        size=8.4,
        weight="bold",
        ha="left",
    )

    # Timestep and conditioning bus.
    rounded_box(ax, 0.505, 0.853, 0.080, 0.057, fc=PALE, ec=LIGHT, radius=0.008)
    label(ax, 0.545, 0.890, "Diffusion step", size=5.2, weight="bold")
    label(ax, 0.545, 0.869, r"$t\to\mathrm{SinEmb}_{64}$", size=5.3)
    rounded_box(ax, 0.605, 0.846, 0.114, 0.071, fc=TEAL_SOFT, ec=TEAL, radius=0.008)
    label(ax, 0.662, 0.892, "Global feature", size=5.5, weight="bold")
    label(ax, 0.662, 0.869, r"$[\,\mathrm{SinEmb}_{64};z_c\,]$", size=5.4)
    label(ax, 0.662, 0.852, "320-D", size=4.9, color=MUTED)
    arrow(ax, (0.585, 0.882), (0.605, 0.882), color=TEAL)
    # Route the fused observation condition around the noisy-sample input so
    # the conditioning path does not visually imply a merge with x_t.
    ax.plot(
        [0.476, 0.492, 0.492, 0.592],
        [0.578, 0.578, 0.824, 0.824],
        color=TEAL,
        lw=0.9,
        zorder=6,
    )
    arrow(ax, (0.592, 0.824), (0.628, 0.846), color=TEAL, rad=-0.05)

    # Input and U-Net path.
    rounded_box(ax, 0.505, 0.700, 0.066, 0.092, fc=ROSE_SOFT, ec=ROSE, radius=0.008)
    label(ax, 0.538, 0.765, r"$x_t$", size=7.3, weight="bold")
    label(ax, 0.538, 0.740, r"$10\times6$", size=5.6)
    label(ax, 0.538, 0.716, "noisy residuals", size=4.8, color=MUTED)

    draw_resblock(
        ax,
        0.594,
        0.690,
        0.084,
        0.111,
        "Down block 1",
        r"$6\times10\to64\times10$",
        fc=BLUE_SOFT,
        ec=BLUE,
    )
    draw_resblock(
        ax,
        0.700,
        0.583,
        0.084,
        0.111,
        "Down block 2",
        r"$64\times5\to128\times5$",
        fc=BLUE_SOFT,
        ec=BLUE_DARK,
    )
    draw_resblock(
        ax,
        0.802,
        0.475,
        0.082,
        0.111,
        "Mid block",
        r"$128\times5$",
        fc=VIOLET_SOFT,
        ec=VIOLET,
    )
    draw_resblock(
        ax,
        0.900,
        0.583,
        0.081,
        0.111,
        "Up block",
        r"$256\times5\to64\times5$",
        fc=BLUE_SOFT,
        ec=BLUE_DARK,
    )
    rounded_box(ax, 0.908, 0.714, 0.073, 0.082, fc=GREEN_SOFT, ec=GREEN, radius=0.008)
    label(ax, 0.945, 0.775, "Output head", size=5.9, weight="bold")
    label(ax, 0.945, 0.750, r"$64\to6$", size=5.6)
    label(ax, 0.945, 0.728, r"$\hat{x}_0:10\times6$", size=5.7, color=GREEN, weight="bold")

    arrow(ax, (0.571, 0.746), (0.594, 0.746), color=ROSE)
    arrow(ax, (0.678, 0.717), (0.700, 0.646), color=BLUE_DARK)
    label(ax, 0.688, 0.685, "↓2", size=4.8, color=MUTED)
    arrow(ax, (0.784, 0.626), (0.802, 0.531), color=BLUE_DARK)
    arrow(ax, (0.884, 0.531), (0.900, 0.626), color=BLUE_DARK)
    arrow(ax, (0.941, 0.694), (0.944, 0.714), color=BLUE_DARK)
    label(ax, 0.964, 0.704, "↑2", size=4.8, color=MUTED)

    # Skip connection used by the current compact implementation.
    arrow(
        ax,
        (0.742, 0.694),
        (0.925, 0.694),
        color=BLUE_DARK,
        lw=1.0,
        rad=-0.25,
        style="-|>",
    )
    label(ax, 0.833, 0.729, "skip concatenation", size=5.0, color=BLUE_DARK)

    # FiLM bus and modulation arrows.
    ax.plot([0.662, 0.662, 0.946], [0.846, 0.824, 0.824], color=TEAL, lw=0.9, ls="--", zorder=5)
    for xx, yy in [(0.636, 0.801), (0.742, 0.694), (0.843, 0.586), (0.941, 0.694)]:
        arrow(
            ax,
            (xx, 0.824),
            (xx, yy),
            color=TEAL,
            lw=0.7,
            ls="--",
            mutation=6,
        )
    label(ax, 0.798, 0.839, r"FiLM: $(\gamma,\beta)$", size=5.4, color=TEAL, weight="bold")
    label(
        ax,
        0.744,
        0.426,
        "Each residual block: Conv1D (k=5) – GroupNorm (8) – Mish – FiLM – Conv1D + residual",
        size=5.2,
        color=MUTED,
    )


def draw_output_and_objective(ax):
    panel_label(ax, 0.014, 0.315, "c")
    label(
        ax,
        0.043,
        0.302,
        "Output parameterization, differentiable supervision, and safe inference",
        size=8.2,
        weight="bold",
        ha="left",
    )

    y = 0.105
    h = 0.114

    rounded_box(ax, 0.035, y, 0.112, h, fc=GREEN_SOFT, ec=GREEN, radius=0.009)
    label(ax, 0.091, y + 0.081, "Clean residual", size=6.0, weight="bold")
    label(ax, 0.091, y + 0.052, r"$\hat{x}_0\in\mathbb{R}^{10\times6}$", size=6.2)
    label(ax, 0.091, y + 0.025, "prediction type: sample", size=4.8, color=MUTED)

    rounded_box(ax, 0.175, y, 0.132, h, fc=VIOLET_SOFT, ec=VIOLET, radius=0.009)
    label(ax, 0.241, y + 0.081, "Control points", size=6.0, weight="bold")
    label(ax, 0.241, y + 0.052, r"$10$ free $+\,6$ fixed", size=5.8)
    label(ax, 0.241, y + 0.025, r"$C\in\mathbb{R}^{16\times6}$", size=5.5, color=MUTED)

    rounded_box(ax, 0.335, y, 0.125, h, fc=BLUE_SOFT, ec=BLUE, radius=0.009)
    label(ax, 0.398, y + 0.081, "Quintic B-spline", size=6.0, weight="bold")
    label(ax, 0.398, y + 0.052, r"$q(\tau)=B(\tau)C$", size=6.1)
    label(ax, 0.398, y + 0.025, "joint trajectory", size=4.9, color=MUTED)

    rounded_box(ax, 0.490, y, 0.137, h, fc=ORANGE_SOFT, ec=ORANGE, radius=0.009)
    label(ax, 0.559, y + 0.081, "Differentiable geometry", size=5.9, weight="bold")
    label(ax, 0.559, y + 0.052, "FK → surface points → SDF", size=5.2)
    label(ax, 0.559, y + 0.025, "collision · OOB · smoothness", size=4.8, color=MUTED)

    rounded_box(ax, 0.655, y, 0.140, h, fc=ROSE_SOFT, ec=ROSE, radius=0.009)
    label(ax, 0.725, y + 0.081, "Training objective", size=6.0, weight="bold")
    label(ax, 0.725, y + 0.052, r"$\mathcal{L}=\mathcal{L}_{\mathrm{diff}}+\mathcal{L}_{\mathrm{traj}}$", size=6.0)
    label(ax, 0.725, y + 0.025, "direct clean-sample regression", size=4.8, color=MUTED)

    rounded_box(ax, 0.823, y, 0.158, h, fc=TEAL_SOFT, ec=TEAL, radius=0.009)
    label(ax, 0.902, y + 0.084, "Inference-only safety layer", size=5.8, weight="bold")
    label(ax, 0.902, y + 0.055, "late-stage CBF–QP", size=6.1, color=TEAL, weight="bold")
    label(ax, 0.902, y + 0.026, r"project $\hat{x}_0$, then continue DDIM", size=4.7, color=MUTED)

    for x0, x1, color in [
        (0.147, 0.175, GREEN),
        (0.307, 0.335, VIOLET),
        (0.460, 0.490, BLUE),
        (0.627, 0.655, ORANGE),
    ]:
        arrow(ax, (x0, y + h / 2), (x1, y + h / 2), color=color)
    arrow(ax, (0.460, y + 0.085), (0.823, y + 0.085), color=TEAL, lw=0.8, ls="--", rad=-0.07)

    # Training target enters the diffusion objective.
    label(ax, 0.725, 0.075, r"$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$", size=5.4, color=MUTED)


def build_figure():
    fig = plt.figure(figsize=(7.2, 4.55), facecolor=WHITE)
    ax = fig.add_axes([0.012, 0.02, 0.976, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Quiet separators establish hierarchy without decorative panel boxes.
    ax.plot([0.487, 0.487], [0.344, 0.925], color=LIGHT, lw=0.65)
    ax.plot([0.018, 0.982], [0.335, 0.335], color=LIGHT, lw=0.65)

    draw_condition_encoder(ax)
    draw_temporal_unet(ax)
    draw_output_and_objective(ax)

    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUT_DIR / "simple_dp3_cspace_network_architecture"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03, facecolor=WHITE)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03, facecolor=WHITE)
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03, facecolor=WHITE)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor=WHITE,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
