# Object Detection (YOLOv8)

Real-time and image/video object detection built on Ultralytics YOLOv8,
with a script to run pretrained COCO detection out of the box and a
training script to fine-tune on your own labeled dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Detect (pretrained, 80 COCO classes)

```bash
# Single image
python detect.py path/to/image.jpg

# Video
python detect.py path/to/video.mp4

# Live webcam (omit source)
python detect.py
```

Detected images are saved to `./outputs/`, with per-object class, confidence,
and bounding box printed to the console. Webcam mode opens a live annotated
window (press `q` to quit).

## Fine-tune on a custom dataset

1. Label your data in YOLO format (see `data.example.yaml` for the expected
   folder layout: `images/train`, `images/val`, `labels/train`, `labels/val`).
2. Copy `data.example.yaml` to `data.yaml` and edit the class names/paths.
3. Run:

```bash
python train.py --data data.yaml --epochs 50 --imgsz 640
```

Training runs and weights are saved under `runs/train/exp/`. The best
checkpoint (`runs/train/exp/weights/best.pt`) can be passed back into
`detect.py --weights`.

## Files

- `detect.py` — inference on image/video/webcam using a pretrained or
  fine-tuned model
- `train.py` — fine-tune YOLOv8 on a custom dataset
- `data.example.yaml` — example dataset config (copy to `data.yaml`)
- `requirements.txt` — dependencies
