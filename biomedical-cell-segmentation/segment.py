import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import label2rgb
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.transform import resize as sk_resize

from unet import NucleiUNet


def load_model(checkpoint_path: str, device: torch.device) -> NucleiUNet:
    model = NucleiUNet(in_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict_maps(model, image: np.ndarray, img_size: int, device: torch.device):
    """Returns (foreground_prob, boundary_prob) at the original image resolution."""
    orig_shape = image.shape

    resized = sk_resize(image, (img_size, img_size), preserve_range=True, anti_aliasing=True)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (2, H, W)

    foreground = sk_resize(probs[0], orig_shape, preserve_range=True, anti_aliasing=True)
    boundary = sk_resize(probs[1], orig_shape, preserve_range=True, anti_aliasing=True)
    return foreground, boundary


def instances_from_maps(foreground: np.ndarray, boundary: np.ndarray,
                         fg_thresh: float = 0.5, boundary_thresh: float = 0.5) -> np.ndarray:
    """
    Foreground + boundary probability maps -> per-cell instance label map,
    using boundary-eroded seeds + watershed to split touching cells.
    """
    fg_mask = foreground > fg_thresh
    boundary_mask = boundary > boundary_thresh

    # Seeds are foreground pixels that are NOT near a predicted boundary,
    # i.e. confidently "interior" pixels of each cell.
    seed_mask = fg_mask & ~boundary_mask
    markers = cc_label(seed_mask)

    # Flood outward from seeds, guided by the (inverted) foreground confidence,
    # constrained to stay within the foreground mask.
    elevation = -foreground
    instance_labels = watershed(elevation, markers=markers, mask=fg_mask)

    return instance_labels


def save_stats_csv(instance_labels: np.ndarray, intensity_image: np.ndarray, out_csv: str):
    props = regionprops(instance_labels, intensity_image=intensity_image)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "area_px", "centroid_row", "centroid_col",
                          "eccentricity", "mean_intensity"])
        for p in props:
            writer.writerow([
                p.label, p.area,
                round(p.centroid[0], 1), round(p.centroid[1], 1),
                round(p.eccentricity, 3), round(p.mean_intensity, 4),
            ])

    return len(props)


def main():
    parser = argparse.ArgumentParser(
        description="Segment individual cells/nuclei in a microscopy image and export "
                     "per-cell morphology stats"
    )
    parser.add_argument("image", help="Path to a grayscale microscopy image")
    parser.add_argument("--checkpoint", default="nuclei_unet.pt")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--out-dir", default="./outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    image = np.array(Image.open(args.image).convert("L"), dtype=np.float32) / 255.0

    foreground, boundary = predict_maps(model, image, args.img_size, device)
    instance_labels = instances_from_maps(foreground, boundary)

    overlay = label2rgb(instance_labels, image=image, bg_label=0, alpha=0.4)
    overlay_path = out_dir / f"segmented_{Path(args.image).stem}.png"
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    csv_path = out_dir / f"stats_{Path(args.image).stem}.csv"
    n_cells = save_stats_csv(instance_labels, image, str(csv_path))

    print(f"Detected {n_cells} cell(s)/nucleus/nuclei.")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved per-cell stats: {csv_path}")


if __name__ == "__main__":
    main()
