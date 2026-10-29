import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot_curves(log_path: Path, out: Path):
    """Training box loss and validation mAP per epoch, from Ultralytics' results.csv."""
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(log_path))]
    epochs = [int(float(r["epoch"])) for r in rows]
    series = lambda key: [float(r[key]) for r in rows]  # noqa: E731

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # The two losses end close together, so a legend reads better than end labels.
    ax1.plot(epochs, series("train/box_loss"), color=BLUE, linewidth=2, label="train")
    ax1.plot(epochs, series("val/box_loss"), color=ORANGE, linewidth=2, label="validation")
    ax1.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK_2)
    ax1.set_xlim(epochs[0] - 1, epochs[-1] + 1)
    ax1.set_title("Box regression loss", loc="left")
    ax1.set_xlabel("Epoch")

    map50, map5095 = series("metrics/mAP50(B)"), series("metrics/mAP50-95(B)")
    ax2.plot(epochs, map50, color=BLUE, linewidth=2)
    ax2.plot(epochs, map5095, color=ORANGE, linewidth=2)
    ax2.text(epochs[-1] + 0.6, map50[-1], "mAP50", color=INK_2, va="center", fontsize=10)
    ax2.text(epochs[-1] + 0.6, map5095[-1], "mAP50-95", color=INK_2, va="center", fontsize=10)
    ax2.set_xlim(epochs[0] - 1, epochs[-1] + 9)
    ax2.set_ylim(0, 1)
    ax2.set_title("Validation mAP", loc="left")
    ax2.set_xlabel("Epoch")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def export_results(run_dir: Path, metrics, args, out_dir: Path):
    """Copies the run's figures into out_dir and writes the validation metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "val_batch0_pred.jpg": "val_predictions.jpg",
        "val_batch0_labels.jpg": "val_labels.jpg",
    }
    for src, dst in figures.items():
        if (run_dir / src).exists():
            shutil.copy(run_dir / src, out_dir / dst)
    if (run_dir / "results.csv").exists():
        shutil.copy(run_dir / "results.csv", out_dir / "training_log.csv")
        plot_curves(run_dir / "results.csv", out_dir / "training_curves.png")

    summary = {
        "data": args.data,
        "base_weights": Path(args.base_weights).name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLOv8 model on a custom dataset "
                     "(dataset must follow Ultralytics YOLO format, described by a data.yaml)"
    )
    parser.add_argument("--data", default="data.yaml",
                         help="Path to dataset YAML (see data.example.yaml for the expected format)")
    parser.add_argument("--base-weights", default="yolov8n.pt",
                         help="Pretrained weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--results-dir", default="results",
                        help="Where to copy the final metrics and figures")
    args = parser.parse_args()

    model = YOLO(args.base_weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # Re-validate the best checkpoint so the reported numbers match best.pt.
    run_dir = Path(model.trainer.save_dir)
    best = YOLO(run_dir / "weights" / "best.pt")
    metrics = best.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                       project=args.project, name=f"{args.name}_val", exist_ok=True)

    summary = export_results(run_dir, metrics, args, Path(args.results_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
