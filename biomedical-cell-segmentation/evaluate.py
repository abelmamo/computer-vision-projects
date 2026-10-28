"""
Evaluates the full pipeline (U-Net + watershed) on a held-out test set, at
each image's original resolution, and writes:

  results/metrics.json        pixel Dice and instance-level precision/recall/F1
  results/examples.png        test images with ground truth and predictions
  results/training_curves.png validation Dice per epoch (needs --log)

Instance matching follows the usual nucleus-segmentation protocol: a
predicted nucleus is a true positive if it overlaps a ground-truth nucleus
with IoU > 0.5 (at that threshold a match is always one-to-one).

The same model is also scored with a "foreground only" baseline that labels
connected components of the foreground mask and ignores the boundary
channel, to measure how much the boundary channel + watershed step adds.
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
from skimage.color import label2rgb  # noqa: E402
from skimage.measure import label as cc_label  # noqa: E402

from dataset import find_samples, load_instance_label, load_microscopy_image  # noqa: E402
from segment import instances_from_maps, load_model, predict_maps  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def match_instances(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5):
    """Returns (true positives, false positives, false negatives)."""
    n_gt, n_pred = int(gt.max()), int(pred.max())
    if n_gt == 0 or n_pred == 0:
        return 0, n_pred, n_gt

    # Intersection counts for every (gt, pred) label pair in one pass.
    pair_index = gt.ravel().astype(np.int64) * (n_pred + 1) + pred.ravel()
    inter = np.bincount(pair_index, minlength=(n_gt + 1) * (n_pred + 1))
    inter = inter.reshape(n_gt + 1, n_pred + 1)[1:, 1:]
    area_gt = np.bincount(gt.ravel(), minlength=n_gt + 1)[1:]
    area_pred = np.bincount(pred.ravel(), minlength=n_pred + 1)[1:]

    union = area_gt[:, None] + area_pred[None, :] - inter
    iou = inter / np.maximum(union, 1)
    tp = int((iou > iou_threshold).sum())
    return tp, n_pred - tp, n_gt - tp


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def summarize(tp: int, fp: int, fn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def plot_examples(examples, out: Path):
    rows = len(examples)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.1 * rows))
    titles = ["Input", "Ground truth", "Foreground only", "With boundary + watershed"]
    for r, ex in enumerate(examples):
        panels = [
            ex["image"],
            label2rgb(ex["gt"], image=ex["image"], bg_label=0, alpha=0.4),
            label2rgb(ex["baseline"], image=ex["image"], bg_label=0, alpha=0.4),
            label2rgb(ex["pred"], image=ex["image"], bg_label=0, alpha=0.4),
        ]
        captions = [
            "",
            f"{ex['gt'].max()} nuclei",
            f"{ex['baseline'].max()} found, F1 {ex['baseline_f1']:.2f}",
            f"{ex['pred'].max()} found, F1 {ex['f1']:.2f}",
        ]
        for c, (panel, caption) in enumerate(zip(panels, captions)):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray" if c == 0 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], loc="left")
            ax.set_xlabel(caption, color=INK_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curves(log_path: Path, out: Path):
    rows = list(csv.DictReader(open(log_path)))
    epochs = [int(r["epoch"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.plot(epochs, [float(r["train_loss"]) for r in rows], color=BLUE, linewidth=2)
    ax1.set_title("Training loss", loc="left")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE (foreground + 2 x boundary)")

    fg = [float(r["val_fg_dice"]) for r in rows]
    bd = [float(r["val_boundary_dice"]) for r in rows]
    ax2.plot(epochs, fg, color=BLUE, linewidth=2)
    ax2.plot(epochs, bd, color=ORANGE, linewidth=2)
    ax2.text(epochs[-1] + 0.3, fg[-1], "foreground", color=INK_2, va="center", fontsize=10)
    ax2.text(epochs[-1] + 0.3, bd[-1], "boundary", color=INK_2, va="center", fontsize=10)
    ax2.set_xlim(epochs[0] - 0.5, epochs[-1] + 4)
    ax2.set_ylim(0, 1)
    ax2.set_title("Validation Dice", loc="left")
    ax2.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate nucleus instance segmentation")
    parser.add_argument("--data-dir", required=True, help="Held-out test set")
    parser.add_argument("--checkpoint", default="nuclei_unet.pt")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--log", default=None, help="training_log.csv from train.py")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--n-examples", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    totals = {"pipeline": [0, 0, 0], "foreground_only": [0, 0, 0]}
    dice_scores, count_errors, per_image = [], [], []

    for image_path, mask_source, layout in find_samples(args.data_dir):
        image = load_microscopy_image(image_path)
        gt = load_instance_label(mask_source, layout, image.shape)

        foreground, boundary = predict_maps(model, image, args.img_size, device)
        pred = instances_from_maps(foreground, boundary)
        baseline = cc_label(foreground > 0.5)

        tp, fp, fn = match_instances(gt, pred)
        btp, bfp, bfn = match_instances(gt, baseline)
        for key, counts in (("pipeline", (tp, fp, fn)), ("foreground_only", (btp, bfp, bfn))):
            totals[key] = [t + c for t, c in zip(totals[key], counts)]

        dice_scores.append(dice(foreground > 0.5, gt > 0))
        count_errors.append(abs(int(pred.max()) - int(gt.max())))
        per_image.append({
            "image": image, "gt": gt, "pred": pred, "baseline": baseline,
            "f1": summarize(tp, fp, fn)["f1"], "baseline_f1": summarize(btp, bfp, bfn)["f1"],
        })

    metrics = {
        "test_images": len(per_image),
        "test_nuclei": int(sum(int(ex["gt"].max()) for ex in per_image)),
        "foreground_dice_mean": round(float(np.mean(dice_scores)), 4),
        "count_abs_error_mean": round(float(np.mean(count_errors)), 2),
        "instance_iou_threshold": 0.5,
        "pipeline": summarize(*totals["pipeline"]),
        "foreground_only_baseline": summarize(*totals["foreground_only"]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Show the most crowded test images, where splitting touching nuclei matters.
    crowded = sorted(per_image, key=lambda ex: int(ex["gt"].max()), reverse=True)
    plot_examples(crowded[:args.n_examples], out_dir / "examples.png")
    if args.log:
        plot_curves(Path(args.log), out_dir / "training_curves.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
