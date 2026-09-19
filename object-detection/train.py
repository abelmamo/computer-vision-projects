import argparse

from ultralytics import YOLO


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
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    model = YOLO(args.base_weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    metrics = model.val()
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")
    print(f"Validation mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
