# Biomedical Cell/Nucleus Segmentation

Instance segmentation for microscopy images — finding and separating
individual cells or nuclei, not just a single foreground/background mask.
This is the kind of preprocessing step that quantitative biology work
actually needs: cell counts, sizes, and shapes per image, not just "is there
a cell here."

## Why not just a binary mask?

A plain foreground/background U-Net (see [`../image-segmentation`](../image-segmentation))
merges touching cells into one blob, which is useless if you need to count
them or measure individual cell size. This project predicts two things
instead of one:

1. **foreground** — is this pixel part of any cell
2. **boundary** — is this pixel on the border between two touching cells

The boundary map is used as a watershed barrier to split touching cells
apart into separate instances.

## Pipeline

1. `unet.py` — U-Net with a 2-channel output (foreground, boundary)
2. `dataset.py` — loads a DSB2018-style dataset (one folder per sample, with
   an `images/` file and a `masks/` folder of one binary PNG per cell) and
   derives the foreground/boundary targets automatically
3. `train.py` — trains the U-Net (BCE loss on both channels, boundary
   up-weighted since it's a thin minority of pixels)
4. `segment.py` — runs inference, uses the boundary-eroded foreground as
   watershed seeds to split touching cells, then measures each resulting
   cell (area, centroid, eccentricity, mean intensity) and writes it to CSV

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Expects the standard nuclei-segmentation layout used by the Kaggle
"2018 Data Science Bowl" dataset (a good public dataset to train/test on):

```
dataset/
  <sample_id>/
    images/<sample_id>.png
    masks/
      instance_1.png
      instance_2.png
      ...
```

## Train

```bash
python train.py --data-dir dataset/ --img-size 256 --epochs 40
```

Saves the best checkpoint (by combined foreground + boundary Dice on a
held-out split) to `nuclei_unet.pt`.

## Segment an image

```bash
python segment.py path/to/microscopy_image.png --checkpoint nuclei_unet.pt
```

Outputs, per image:
- `segmented_<name>.png` — each detected cell colored separately
- `stats_<name>.csv` — one row per cell: area, centroid, eccentricity, mean intensity

## Files

- `unet.py` — U-Net predicting foreground + boundary maps
- `dataset.py` — dataset loader that builds targets from per-instance masks
- `train.py` — training loop
- `segment.py` — inference, watershed instance splitting, per-cell stats export
- `requirements.txt` — dependencies
