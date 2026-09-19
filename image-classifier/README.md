# Image Classifier (CNN, CIFAR-10)

A convolutional neural network trained from scratch in PyTorch to classify
32x32 RGB images into 10 CIFAR-10 categories (plane, car, bird, cat, deer,
dog, frog, horse, ship, truck).

## Architecture

Three convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool), followed
by a fully connected head with dropout. See `model.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 15 --batch-size 128 --lr 1e-3
```

The CIFAR-10 dataset is downloaded automatically on first run. The best
checkpoint (by test accuracy) is saved to `classifier.pt`.

Expected result: ~80-85% test accuracy after 15 epochs on a single GPU
(a few minutes), longer on CPU.

## Predict

```bash
python predict.py path/to/image.jpg --checkpoint classifier.pt --top-k 3
```

Prints the top-k predicted classes with their probabilities.

## Files

- `model.py` — CNN architecture (`SimpleCNN`)
- `train.py` — training loop, data augmentation, checkpointing
- `predict.py` — inference on a single image
- `requirements.txt` — dependencies
