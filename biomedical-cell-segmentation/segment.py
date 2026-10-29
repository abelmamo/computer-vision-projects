import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import label2rgb
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.transform import resize as sk_resize

from dataset import load_microscopy_image
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


DEFAULT_THRESHOLDS = {"fg_thresh": 0.5, "boundary_thresh": 0.5, "min_seed_size": 0}


def load_thresholds(path):
    """Post-processing settings chosen by tune.py on the validation split."""
    if path and Path(path).exists():
        return {**DEFAULT_THRESHOLDS, **json.loads(Path(path).read_text())["best"]}
    return dict(DEFAULT_THRESHOLDS)


def instances_from_maps(foreground: np.ndarray, boundary: np.ndarray,
                        fg_thresh: float = 0.5, boundary_thresh: float = 0.5,
                        min_seed_size: int = 0) -> np.ndarray:
    """
    Foreground + boundary probability maps -> per-nucleus instance label map,
    using boundary-eroded seeds + watershed to split touching nuclei.

    The boundary target marks the ring just outside each nucleus, so inside a
    merged blob the boundary probability stays fairly low. The threshold that
    best separates touching nuclei is therefore tuned on validation data
    (tune.py) instead of assumed to be 0.5.
    """
    fg_mask = foreground > fg_thresh
    boundary_mask = boundary > boundary_thresh

    # Seeds are foreground pixels that are NOT near a predicted boundary,
    # i.e. confidently "interior" pixels of each nucleus.
    seed_mask = fg_mask & ~boundary_mask
    markers = cc_label(seed_mask)
    if min_seed_size > 0:
        # Drop tiny seed fragments so one nucleus is not split into several.
        sizes = np.bincount(markers.ravel())
        small = sizes < min_seed_size
        small[0] = False
        markers[small[markers]] = 0
        markers = cc_label(markers > 0)

    # Flood outward from seeds, guided by the (inverted) foreground confidence,
    # constrained to stay within the foreground mask.
    elevation = -foreground
    instance_labels = watershed(elevation, markers=markers, mask=fg_mask)

    # A small nucleus can be covered entirely by its predicted boundary and
    # get no seed. Keep any such leftover foreground blob as its own nucleus.
    leftovers = cc_label(fg_mask & (instance_labels == 0))
    if leftovers.max() > 0:
        instance_labels = np.where(leftovers > 0, leftovers + instance_labels.max(),
                                   instance_labels)

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
    parser.add_argument("--thresholds", default="results/thresholds.json",
                        help="Post-processing settings written by tune.py")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    image = load_microscopy_image(args.image)

    foreground, boundary = predict_maps(model, image, args.img_size, device)
    instance_labels = instances_from_maps(foreground, boundary, **load_thresholds(args.thresholds))

    overlay = label2rgb(instance_labels, image=image, bg_label=0, alpha=0.4)
    overlay_path = out_dir / f"segmented_{Path(args.image).stem}.png"
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    csv_path = out_dir / f"stats_{Path(args.image).stem}.csv"
    n_cells = save_stats_csv(instance_labels, image, str(csv_path))

    print(f"Detected {n_cells} nuclei.")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved per-cell stats: {csv_path}")


if __name__ == "__main__":
    main()
