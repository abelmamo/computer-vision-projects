# Image Segmentation (U-Net)

A from-scratch PyTorch implementation of U-Net for binary semantic
segmentation (e.g. foreground/background, object masks), trainable on any
image/mask dataset.

## Architecture

Classic U-Net encoder-decoder with skip connections (`unet.py`): 4 downsampling
blocks, a bottleneck, 4 upsampling blocks with concatenated skip features from
the encoder, and a 1x1 conv output head.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset format

Two directories with matching filenames (by stem, extension can differ):

```
dataset/
  images/  0001.jpg  0002.jpg  ...
  masks/   0001.png  0002.png  ...   (white = foreground, black = background)
```

`dataset.py` (`SegmentationDataset`) pairs files automatically.

## Train

```bash
python train.py --images-dir dataset/images --masks-dir dataset/masks \
    --img-size 256 --epochs 30 --batch-size 8
```

Trains with BCE-with-logits loss, tracks Dice coefficient on a held-out
validation split, and saves the best checkpoint to `unet.pt`.

## Predict

```bash
python predict.py path/to/image.jpg --checkpoint unet.pt
```

Saves `mask.png` (binary mask) and `overlay.png` (mask overlaid in red on the
original image).

## Files

- `unet.py` — U-Net architecture
- `dataset.py` — generic image/mask dataset loader
- `train.py` — training loop with Dice-based validation
- `predict.py` — inference + mask/overlay visualization
- `requirements.txt` — dependencies
