# Computer Vision Projects

A collection of full, runnable computer vision projects covering classic
tasks: classification, detection, segmentation, and face recognition. Each
subfolder is self-contained with its own README, requirements, training
script, and inference script.

## Projects

| Project | Task | Approach |
|---|---|---|
| [`image-classifier`](./image-classifier) | Image classification | CNN trained from scratch (PyTorch, CIFAR-10) |
| [`object-detection`](./object-detection) | Object detection | YOLOv8 (Ultralytics), pretrained + fine-tuning |
| [`image-segmentation`](./image-segmentation) | Semantic segmentation | U-Net trained from scratch (PyTorch) |
| [`face-detection-recognition`](./face-detection-recognition) | Face detection + recognition | OpenCV Haar cascade + `face_recognition` embeddings |

Each project's README has full setup, training, and inference instructions.

## Requirements

Each project has its own `requirements.txt`. Recommended: use a separate
virtual environment per project since dependencies (e.g. `torch` vs
`ultralytics` vs `dlib`) don't need to overlap.

```bash
cd <project-folder>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
