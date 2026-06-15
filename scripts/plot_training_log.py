import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def save_overview(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].plot(df["epoch"], df["train_loss"], label="train_loss", linewidth=2)
    if "val_loss" in df:
        val_df = df.dropna(subset=["val_loss"])
        axes[0, 0].plot(val_df["epoch"], val_df["val_loss"], label="val_loss", linewidth=2)
    axes[0, 0].set_title("Training / Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(df["epoch"], df["bc_loss"], label="bc_loss", linewidth=2)
    axes[0, 1].plot(df["epoch"], df["total_loss"], label="total_loss", linewidth=2)
    axes[0, 1].plot(df["epoch"], df["smooth_loss"], label="smooth_loss", linewidth=2)
    axes[0, 1].set_title("Optimization Terms")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(df["epoch"], df["sdf_collision_loss"], label="sdf_collision_loss", linewidth=2)
    axes[1, 0].plot(df["epoch"], df["trajectory_collision_loss"], label="trajectory_collision_loss", linewidth=2)
    axes[1, 0].set_title("Collision-Related Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(df["epoch"], df["test_mean_score"], label="test_mean_score", linewidth=2)
    if "lr" in df:
        ax_lr = axes[1, 1].twinx()
        ax_lr.plot(df["epoch"], df["lr"], label="lr", linestyle="--", color="tab:orange", alpha=0.8)
        ax_lr.set_ylabel("Learning Rate")
        lr_lines, lr_labels = ax_lr.get_legend_handles_labels()
    else:
        lr_lines, lr_labels = [], []
    axes[1, 1].set_title("Test Score and Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Test Mean Score")
    axes[1, 1].grid(alpha=0.3)
    lines, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(lines + lr_lines, labels + lr_labels, loc="best")

    fig.suptitle("Training Log Overview", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_collision_focus(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    axes[0].bar(df["epoch"], df["sdf_collision_loss"], color="tab:red", alpha=0.75, label="sdf_collision_loss")
    axes[0].plot(df["epoch"], df["total_loss"], color="tab:blue", linewidth=2, label="total_loss")
    axes[0].set_title("Collision Spikes vs Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if "generalization_gap" in df:
        gap_df = df.dropna(subset=["generalization_gap"])
        axes[1].plot(gap_df["epoch"], gap_df["generalization_gap"], marker="o", linewidth=2, label="generalization_gap")
        if "early_stop_best_value" in gap_df:
            best_df = gap_df.dropna(subset=["early_stop_best_value"])
            axes[1].plot(best_df["epoch"], best_df["early_stop_best_value"], marker="s", linewidth=2, label="early_stop_best_value")
        axes[1].set_title("Validation Checkpoints")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training logs stored as JSON lines.")
    parser.add_argument("log_path", type=Path, help="Path to JSONL log file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/log_plots"),
        help="Directory to store generated plots.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".codex_mpl_cache").resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.log_path)
    save_overview(df, args.output_dir / "training_overview.png")
    save_collision_focus(df, args.output_dir / "collision_and_validation.png")

    summary = {
        "num_rows": int(len(df)),
        "epoch_start": int(df["epoch"].min()),
        "epoch_end": int(df["epoch"].max()),
        "best_train_loss": float(df["train_loss"].min()),
        "best_train_epoch": int(df.loc[df["train_loss"].idxmin(), "epoch"]),
    }
    if "val_loss" in df:
        val_df = df.dropna(subset=["val_loss"])
        if not val_df.empty:
            summary["best_val_loss"] = float(val_df["val_loss"].min())
            summary["best_val_epoch"] = int(val_df.loc[val_df["val_loss"].idxmin(), "epoch"])
    if "test_mean_score" in df:
        summary["best_test_score"] = float(df["test_mean_score"].max())
        summary["best_test_epoch"] = int(df.loc[df["test_mean_score"].idxmax(), "epoch"])

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
