"""
Chooses the post-processing settings (foreground threshold, boundary
threshold, minimum seed size) on the validation split that train.py held
out, so the test set is never used to pick them.

The U-Net is run once per validation image; every setting in the grid then
reuses those probability maps, which keeps the search fast on CPU.

Writes:
  results/thresholds.json   best setting and the score of every setting tried
  results/tuning.png        validation F1 against the boundary threshold
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset import find_samples, load_instance_label, load_microscopy_image  # noqa: E402
from evaluate import BLUE, INK_2, ORANGE, match_instances, summarize  # noqa: E402
from segment import instances_from_maps, load_model, predict_maps  # noqa: E402

FG_THRESHOLDS = [0.4, 0.5, 0.6]
BOUNDARY_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MIN_SEED_SIZES = [0, 5, 10, 20, 30, 50]


def validation_split(data_dir: str, val_split: float, seed: int):
    """Same seeded split as train.py."""
    samples = find_samples(data_dir)
    order = np.random.RandomState(seed).permutation(len(samples))
    val_size = max(1, int(len(samples) * val_split))
    return [samples[i] for i in order[:val_size]]


def main():
    parser = argparse.ArgumentParser(description="Tune watershed post-processing on validation data")
    parser.add_argument("--data-dir", required=True, help="Training data folder given to train.py")
    parser.add_argument("--checkpoint", default="nuclei_unet.pt")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    cached = []
    for image_path, mask_source, layout in validation_split(args.data_dir, args.val_split, args.seed):
        image = load_microscopy_image(image_path)
        gt = load_instance_label(mask_source, layout, image.shape)
        foreground, boundary = predict_maps(model, image, args.img_size, device)
        cached.append((gt, foreground, boundary))
    print(f"Validation images: {len(cached)}")

    grid = []
    for fg_t, bd_t, seed_size in itertools.product(FG_THRESHOLDS, BOUNDARY_THRESHOLDS, MIN_SEED_SIZES):
        totals = np.zeros(3, dtype=int)
        for gt, foreground, boundary in cached:
            pred = instances_from_maps(foreground, boundary, fg_thresh=fg_t,
                                       boundary_thresh=bd_t, min_seed_size=seed_size)
            totals += match_instances(gt, pred)
        score = summarize(*totals)
        grid.append({"fg_thresh": fg_t, "boundary_thresh": bd_t, "min_seed_size": seed_size,
                     "f1": score["f1"], "precision": score["precision"], "recall": score["recall"]})
        print(f"fg {fg_t:.2f} | boundary {bd_t:.2f} | min seed {seed_size:2d} | F1 {score['f1']:.4f}",
              flush=True)

    best = max(grid, key=lambda g: g["f1"])
    default = next(g for g in grid if g["fg_thresh"] == 0.5 and g["boundary_thresh"] == 0.5
                   and g["min_seed_size"] == 0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "validation_images": len(cached),
        "best": {k: best[k] for k in ("fg_thresh", "boundary_thresh", "min_seed_size")},
        "best_validation_f1": best["f1"],
        "default_validation_f1": default["f1"],
        "grid": grid,
    }
    (out_dir / "thresholds.json").write_text(json.dumps(result, indent=2))

    # Validation F1 against the boundary threshold, at the best foreground
    # threshold, with and without the best minimum seed size.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    curves = [(best["min_seed_size"], BLUE)]
    if best["min_seed_size"] != 0:
        curves.append((0, ORANGE))
    for seed_size, color in curves:
        rows = [g for g in grid if g["fg_thresh"] == best["fg_thresh"]
                and g["min_seed_size"] == seed_size]
        xs = [g["boundary_thresh"] for g in rows]
        ys = [g["f1"] for g in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4,
                label=f"min seed size {seed_size} px")
    ax.scatter([best["boundary_thresh"]], [best["f1"]], s=60, color=BLUE, zorder=3)
    ax.annotate(f"chosen: F1 {best['f1']:.3f}", (best["boundary_thresh"], best["f1"]),
                xytext=(12, 4), textcoords="offset points", color=INK_2, fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2)
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("Boundary threshold")
    ax.set_ylabel("Instance F1 (IoU 0.5)")
    ax.set_title(f"Validation F1, foreground threshold {best['fg_thresh']}", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "tuning.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: result[k] for k in ("best", "best_validation_f1", "default_validation_f1")},
                     indent=2))


if __name__ == "__main__":
    main()
