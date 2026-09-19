import argparse
from pathlib import Path

import cv2


def get_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")
    return detector


def detect_in_frame(detector: cv2.CascadeClassifier, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, faces


def run_image(detector, source: str, out_dir: Path):
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    annotated, faces = detect_in_frame(detector, frame)
    out_path = out_dir / f"detected_{Path(source).name}"
    cv2.imwrite(str(out_path), annotated)

    print(f"Found {len(faces)} face(s). Saved: {out_path}")
    for i, (x, y, w, h) in enumerate(faces, start=1):
        print(f"  face {i}: box=({x}, {y}, {w}, {h})")


def run_webcam(detector, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, _ = detect_in_frame(detector, frame)
        cv2.imshow("Face Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect faces in an image or webcam feed")
    parser.add_argument("source", nargs="?", default=None,
                         help="Path to an image. Omit to use webcam.")
    parser.add_argument("--out-dir", default="./outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = get_detector()

    if args.source is None:
        run_webcam(detector)
    else:
        run_image(detector, args.source, out_dir)


if __name__ == "__main__":
    main()
