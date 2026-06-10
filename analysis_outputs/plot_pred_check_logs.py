import json
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATH = Path("/Volumes/research/outputs/pred_check/logs.json.txt")
OUT_DIR = Path(__file__).resolve().parent


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_records(LOG_PATH)
    epochs = [r["epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss = [r["val_loss"] for r in records]
    gap = [r["generalization_gap"] for r in records]
    lr = [r["lr"] for r in records]

    best = min(records, key=lambda r: r["val_loss"])
    last = records[-1]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(epochs, train_loss, label="Train loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Validation loss", color="#d62728", linewidth=2)
    axes[0].axvline(best["epoch"], color="#2ca02c", linestyle="--", linewidth=1.5)
    axes[0].scatter([best["epoch"]], [best["val_loss"]], color="#2ca02c", zorder=5)
    axes[0].annotate(
        f"Best val: epoch {best['epoch']}\nval_loss={best['val_loss']:.4f}",
        xy=(best["epoch"], best["val_loss"]),
        xytext=(best["epoch"] + 5, best["val_loss"] + 0.035),
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
        fontsize=10,
    )
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend(frameon=True)

    axes[1].plot(epochs, gap, label="Generalization gap", color="#9467bd", linewidth=2)
    axes[1].axhline(0, color="#555555", linestyle=":", linewidth=1)
    axes[1].axvline(best["epoch"], color="#2ca02c", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("Gap")
    axes[1].set_title("Generalization Gap")
    axes[1].legend(frameon=True)

    axes[2].plot(epochs, lr, label="Learning rate", color="#ff7f0e", linewidth=2)
    axes[2].axvline(best["epoch"], color="#2ca02c", linestyle="--", linewidth=1.5)
    if last.get("early_stop"):
        axes[2].scatter([last["epoch"]], [last["lr"]], color="#111111", zorder=5)
        axes[2].annotate(
            f"Early stop: epoch {last['epoch']}",
            xy=(last["epoch"], last["lr"]),
            xytext=(last["epoch"] - 35, last["lr"] + 0.000004),
            arrowprops={"arrowstyle": "->", "color": "#111111"},
            fontsize=10,
        )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend(frameon=True)

    fig.suptitle("pred_check Training Log Summary", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = OUT_DIR / "pred_check_training_curves.png"
    pdf_path = OUT_DIR / "pred_check_training_curves.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
