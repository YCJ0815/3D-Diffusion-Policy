#!/usr/bin/env python3
"""Compare collision rate and planning time across validation result folders."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

_LOCAL_CACHE_DIR = Path(".cache/matplotlib")
_LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_CACHE_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny": "#E4E4F0",
    "ours_base": "#E4CCD8",
    "ours_large": "#F0C0CC",
    "delta_up": "#2E9E44",
    "delta_down": "#E53935",
    "neutral_dark": "#606060",
}


CONFIGS = [
    {
        "label": "3 steps\n2 cand.",
        "short_name": "3step-2candidate",
        "guidance_steps": 3,
        "qp_candidates": 2,
        "color": PALETTE["baseline_soft"],
        "input_dir": Path("/Volumes/research/outputs/validation/guide-3step-2candidate"),
    },
    {
        "label": "3 steps\n4 cand.",
        "short_name": "3step-4candidate",
        "guidance_steps": 3,
        "qp_candidates": 4,
        "color": PALETTE["baseline_mid"],
        "input_dir": Path("/Volumes/research/outputs/validation/guide-3step-4candidate"),
    },
    {
        "label": "3 steps\n8 cand.",
        "short_name": "3step-8candidate",
        "guidance_steps": 3,
        "qp_candidates": 8,
        "color": PALETTE["baseline_dark"],
        "input_dir": Path("/Volumes/research/outputs/validation/guide-3step-8candidate"),
    },
    {
        "label": "5 steps\n4 cand.",
        "short_name": "5step-4candidate",
        "guidance_steps": 5,
        "qp_candidates": 4,
        "color": PALETTE["ours_base"],
        "input_dir": Path("/Volumes/research/outputs/validation/guide-5step-4candidate"),
    },
]


OUTPUT_DIR = Path("figures/guidance_validation_comparison")
SUMMARY_CSV = OUTPUT_DIR / "guidance_validation_summary.csv"
TIMING_CSV = OUTPUT_DIR / "guidance_validation_timing_source_data.csv"
FIGURE_BASE = OUTPUT_DIR / "guidance_validation_comparison"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_metrics(config: dict) -> dict:
    metrics_path = config["input_dir"] / "per_trajectory_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_trajectory = data["per_trajectory"]
    summary = data["summary"]
    times = [item["inference_elapsed_sec"] for item in per_trajectory if item.get("inference_elapsed_sec") is not None]

    n = len(per_trajectory)
    collision_count = int(summary["trajectories_with_collision"])
    collision_rate = float(summary["trajectory_collision_rate"])
    collision_se = math.sqrt(collision_rate * (1.0 - collision_rate) / n)
    collision_ci = 1.96 * collision_se

    time_mean = float(np.mean(times))
    time_median = float(np.median(times))
    time_std = float(np.std(times, ddof=1))
    time_sem = time_std / math.sqrt(len(times))
    time_ci = 1.96 * time_sem

    result = dict(config)
    result.update(
        {
            "n_episodes": n,
            "collision_count": collision_count,
            "collision_rate": collision_rate,
            "collision_ci_low": max(0.0, collision_rate - collision_ci),
            "collision_ci_high": min(1.0, collision_rate + collision_ci),
            "collision_ci_halfwidth": collision_ci,
            "times": times,
            "time_mean_sec": time_mean,
            "time_median_sec": time_median,
            "time_std_sec": time_std,
            "time_ci_low_sec": time_mean - time_ci,
            "time_ci_high_sec": time_mean + time_ci,
            "time_q1_sec": float(np.percentile(times, 25)),
            "time_q3_sec": float(np.percentile(times, 75)),
        }
    )
    return result


def export_source_data(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "short_name",
                "guidance_steps",
                "qp_candidates",
                "n_episodes",
                "collision_count",
                "collision_rate",
                "collision_ci_low",
                "collision_ci_high",
                "time_mean_sec",
                "time_median_sec",
                "time_std_sec",
                "time_ci_low_sec",
                "time_ci_high_sec",
                "time_q1_sec",
                "time_q3_sec",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})

    with TIMING_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["short_name", "guidance_steps", "qp_candidates", "trajectory_index", "inference_elapsed_sec"],
        )
        writer.writeheader()
        for result in results:
            for idx, elapsed in enumerate(result["times"]):
                writer.writerow(
                    {
                        "short_name": result["short_name"],
                        "guidance_steps": result["guidance_steps"],
                        "qp_candidates": result["qp_candidates"],
                        "trajectory_index": idx,
                        "inference_elapsed_sec": elapsed,
                    }
                )


def make_figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.5),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )
    ax_rate, ax_time = axes

    x = np.arange(len(results))
    colors = [result["color"] for result in results]
    collision_pct = np.array([100.0 * result["collision_rate"] for result in results])
    collision_err = np.array([100.0 * result["collision_ci_halfwidth"] for result in results])

    bars = ax_rate.bar(
        x,
        collision_pct,
        width=0.68,
        color=colors,
        edgecolor=PALETTE["neutral_dark"],
        linewidth=0.8,
        yerr=collision_err,
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax_rate.set_ylabel("Trajectory collision rate (%)")
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels([result["label"] for result in results])
    ax_rate.set_ylim(0, max(collision_pct + collision_err) * 1.22)
    ax_rate.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax_rate.set_axisbelow(True)
    for bar, result in zip(bars, results):
        ax_rate.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.65,
            f"{100.0 * result['collision_rate']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax_rate.text(
            bar.get_x() + bar.get_width() / 2.0,
            0.35,
            f"n={result['collision_count']}/{result['n_episodes']}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=PALETTE["neutral_dark"],
            rotation=90,
        )

    time_data = [result["times"] for result in results]
    box = ax_time.boxplot(
        time_data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": PALETTE["neutral_dark"], "linewidth": 1.0},
        whiskerprops={"color": PALETTE["neutral_dark"], "linewidth": 0.8},
        capprops={"color": PALETTE["neutral_dark"], "linewidth": 0.8},
        boxprops={"edgecolor": PALETTE["neutral_dark"], "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    mean_times = [result["time_mean_sec"] for result in results]
    ax_time.scatter(
        np.arange(1, len(results) + 1),
        mean_times,
        marker="D",
        s=26,
        color=PALETTE["delta_down"],
        zorder=3,
        label="Mean",
    )
    for idx, result in enumerate(results, start=1):
        ax_time.text(
            idx + 0.08,
            result["time_mean_sec"] + 0.08,
            f"{result['time_mean_sec']:.2f}s",
            fontsize=6.8,
            color=PALETTE["delta_down"],
        )

    ax_time.set_ylabel("Planning time per trajectory (s)")
    ax_time.set_xticks(np.arange(1, len(results) + 1))
    ax_time.set_xticklabels([result["label"] for result in results])
    ax_time.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax_time.set_axisbelow(True)
    ax_time.legend(loc="upper right", fontsize=7, handletextpad=0.4)

    add_panel_label(ax_rate, "a")
    add_panel_label(ax_time, "b")

    fig.suptitle(
        "Guidance configuration changes have modest timing impact, but 5-step guidance increases collisions",
        fontsize=9,
        y=1.02,
    )

    fig.savefig(FIGURE_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(FIGURE_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = [load_metrics(config) for config in CONFIGS]
    export_source_data(results)
    make_figure(results)

    print("Figure contract")
    print("Core conclusion: 3-step guidance keeps collision rate nearly unchanged across 2-8 candidates, while 5-step guidance at 4 candidates increases collision rate without a planning-time gain.")
    print("Figure archetype: quantitative grid")
    print("Target journal/output: Nature-style comparison figure, SVG/PDF/PNG with source-data CSV")
    print("Backend: Python")
    print("Final size: 7.2 x 3.5 in (~183 x 89 mm)")
    print("Panel map: a) collision rate with 95% CI; b) planning time distribution with mean markers")
    print("Evidence hierarchy: hero evidence = collision rate; validation evidence = per-trajectory planning-time distribution")
    print("Statistics needed: binomial 95% CI for collision rate; mean/median/IQR for planning time")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
