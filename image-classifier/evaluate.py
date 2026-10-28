"""
Evaluates a trained checkpoint on the CIFAR-10 test set and writes:

  results/metrics.json          overall and per-class test accuracy
  results/confusion_matrix.png  10x10 confusion matrix (row-normalized)
  results/per_class_accuracy.png
  results/training_curves.png   loss and accuracy per epoch (needs --log)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from model import SimpleCNN  # noqa: E402
from train import CLASSES, get_datasets  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def collect_predictions(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_pred.append(model(images.to(device)).argmax(1).cpu())
            all_true.append(labels)
    return torch.cat(all_true).numpy(), torch.cat(all_pred).numpy()


def plot_confusion(cm: np.ndarray, out: Path):
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.grid(False)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (share of each true class)", loc="left")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if norm[i, j] >= 0.03:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_per_class(per_class: dict, overall: float, out: Path):
    names = sorted(per_class, key=per_class.get)
    values = [per_class[n] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.grid(axis="y", visible=False)
    bars = ax.barh(names, values, color=BLUE, height=0.62, zorder=2)
    ax.axvline(overall * 100, color=INK_2, linewidth=1, linestyle="--", zorder=1)
    ax.text(overall * 100, len(names) - 0.4, f" overall {overall * 100:.1f}%",
            color=INK_2, va="bottom", fontsize=10)
    for bar, v in zip(bars, values):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9, color=INK_2, zorder=3,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Test accuracy by class", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_curves(log_path: Path, out: Path):
    rows = list(csv.DictReader(open(log_path)))
    epochs = [int(r["epoch"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.plot(epochs, [float(r["train_loss"]) for r in rows], color=BLUE, linewidth=2)
    ax1.set_title("Training loss", loc="left")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy")

    train_acc = [float(r["train_acc"]) * 100 for r in rows]
    test_acc = [float(r["test_acc"]) * 100 for r in rows]
    ax2.plot(epochs, train_acc, color=BLUE, linewidth=2)
    ax2.plot(epochs, test_acc, color=ORANGE, linewidth=2)
    ax2.text(epochs[-1] + 0.3, train_acc[-1], "train", color=INK_2, va="center", fontsize=10)
    ax2.text(epochs[-1] + 0.3, test_acc[-1], "test", color=INK_2, va="center", fontsize=10)
    ax2.set_xlim(epochs[0] - 0.5, epochs[-1] + 2)
    ax2.set_title("Accuracy", loc="left")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 classifier")
    parser.add_argument("--checkpoint", default="classifier.pt")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--log", default=None, help="training_log.csv from train.py")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    _, test_set = get_datasets(args.data_dir, args.image_folder)
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    y_true, y_pred = collect_predictions(model, loader, device)

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    overall = float((y_true == y_pred).mean())
    per_class = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)}

    metrics = {
        "test_images": int(len(y_true)),
        "test_accuracy": round(overall, 4),
        "per_class_accuracy": {c: round(v, 4) for c, v in per_class.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_per_class(per_class, overall, out_dir / "per_class_accuracy.png")
    if args.log:
        plot_curves(Path(args.log), out_dir / "training_curves.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
