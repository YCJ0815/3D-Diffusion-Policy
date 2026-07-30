#!/usr/bin/env python3
"""Figure: Planning success rate and inference time comparison.

Compares traditional RRT+Trajopt planning against three diffusion-policy variants
with collision-free success rate and planning/inference time.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Publication quality defaults
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
})

# ---------------------------------------------------------------------------
# Colour palette — NMI pastel family
# ---------------------------------------------------------------------------
TRAD_COLOR = "#484878"          # baseline dark — traditional planning
DP1_COLOR = "#7884B4"           # baseline mid — Post-QP (SCP)
DP2_COLOR = "#E4CCD8"           # ours base — QP-Guided Diffusion
DP3_COLOR = "#F0C0CC"           # ours large — QP-Guided Diff + Post-QP (hero)

ACCENT_UP = "#2E9E44"           # delta up green
NEUTRAL_DARK = "#606060"


def save_pub(fig, filename, dpi=600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {filename}.svg, {filename}.pdf, {filename}.tiff")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
METHODS = [
    "Traditional\nRRT+Trajopt",
    "Post-QP\n(SCP)",
    "QP-Guided\nDiffusion",
    "QP-Guided Diff\n+ Post-QP",
]

# Traditional: 90/165 trajopt success = 54.5%
TRAD_RATE = 90 / 165          # 0.545
TRAD_TIME = 76.25             # mean plan_time in seconds

# From validation JSONs (all workpieces)
DP_RATES = [0.8750, 0.8014, 0.9097]  # Post-QP, QP-Guided, QP-Guided+PostQP
DP_TIMES = [8.62, 14.85, 6.67]       # avg inference time (job_009)

SUCCESS_RATES = [TRAD_RATE] + DP_RATES
PLAN_TIMES = [TRAD_TIME] + DP_TIMES

COLORS = [TRAD_COLOR, DP1_COLOR, DP2_COLOR, DP3_COLOR]

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, (ax_rate, ax_time) = plt.subplots(
    1, 2, figsize=(7.2, 3.0),  # Nature single-column ~183mm → ~7.2in
    gridspec_kw={"wspace": 0.45},
)

y_positions = np.arange(len(METHODS))
bar_height = 0.55

# --- Panel A: Success Rate ---
bars_rate = ax_rate.barh(
    y_positions, SUCCESS_RATES, height=bar_height, color=COLORS, edgecolor="white", linewidth=0.5,
)
ax_rate.set_yticks(y_positions)
ax_rate.set_yticklabels(METHODS)
ax_rate.set_xlabel("Success Rate")
ax_rate.set_xlim(0, 1.05)
ax_rate.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
ax_rate.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax_rate.invert_yaxis()
ax_rate.set_title("Success Rate", fontweight="bold", loc="left", pad=4)

for bar, val in zip(bars_rate, SUCCESS_RATES):
    ax_rate.text(
        bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
        f"{val * 100:.1f}%", va="center", fontsize=6.5, color=NEUTRAL_DARK,
    )

# --- Panel B: Planning / Inference Time ---
bars_time = ax_time.barh(
    y_positions, PLAN_TIMES, height=bar_height, color=COLORS, edgecolor="white", linewidth=0.5,
)
ax_time.set_yticks(y_positions)
ax_time.set_yticklabels(METHODS)
ax_time.set_xlabel("Time (s)")
ax_time.set_xlim(0, max(PLAN_TIMES) * 1.25)
ax_time.invert_yaxis()
ax_time.set_title("Planning / Inference Time", fontweight="bold", loc="left", pad=4)

for bar, val in zip(bars_time, PLAN_TIMES):
    ax_time.text(
        bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}s", va="center", fontsize=6.5, color=NEUTRAL_DARK,
    )

# Speedup annotation between traditional and best DP
speedup = TRAD_TIME / DP_TIMES[-1]
ax_time.annotate(
    f"{speedup:.1f}× speedup",
    xy=(DP_TIMES[-1], y_positions[-1]),
    xytext=(DP_TIMES[-1] + 8, y_positions[-1] - 0.35),
    fontsize=6, color=ACCENT_UP, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ACCENT_UP, lw=0.8),
)

# ---------------------------------------------------------------------------
# Figure label (a, b)
# ---------------------------------------------------------------------------
for label, ax in zip(["a", "b"], [ax_rate, ax_time]):
    ax.text(
        -0.15, 1.08, label, transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="bottom", ha="left",
    )

fig.subplots_adjust(left=0.22, right=0.96, top=0.88, bottom=0.12, wspace=0.50)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_pub(fig, str(OUTPUT_DIR / "planning_success_comparison"))
plt.close()
print("Done.")
