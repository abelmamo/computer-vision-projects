import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def run_image(model: YOLO, source: str, out_dir: Path, conf: float):
    results = model.predict(source=source, conf=conf, save=False, verbose=False)
    for result in results:
        annotated = result.plot()  # BGR numpy array with boxes/labels drawn
        out_path = out_dir / f"detected_{Path(result.path).name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")

        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            print(f"  {label:<15} conf={confidence:.2f} box=({x1}, {y1}, {x2}, {y2})")


def run_webcam(model: YOLO, conf: float, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Object Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection")
    parser.add_argument("source", nargs="?", default=None,
                         help="Path to image/video/directory. Omit to use webcam.")
    parser.add_argument("--weights", default="yolov8n.pt",
                         help="Path to weights (pretrained COCO by default, or a fine-tuned .pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out-dir", default="./outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    if args.source is None:
        run_webcam(model, args.conf)
    else:
        run_image(model, args.source, out_dir, args.conf)


if __name__ == "__main__":
    main()
