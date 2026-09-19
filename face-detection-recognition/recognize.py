import argparse
import pickle
from pathlib import Path

import cv2
import face_recognition
import numpy as np


def load_database(db_path: str):
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    return db["encodings"], db["names"]


def match_face(encoding, known_encodings, known_names, tolerance: float = 0.6):
    if not known_encodings:
        return "Unknown", None

    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    best_distance = distances[best_idx]

    if best_distance <= tolerance:
        return known_names[best_idx], float(best_distance)
    return "Unknown", float(best_distance)


def annotate(frame, box, label):
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, label, (left + 4, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def run_image(source: str, known_encodings, known_names, out_dir: Path, tolerance: float):
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for box, encoding in zip(boxes, encodings):
        name, distance = match_face(encoding, known_encodings, known_names, tolerance)
        annotate(frame, box, name)
        dist_str = f"{distance:.3f}" if distance is not None else "n/a"
        print(f"  {name:<15} distance={dist_str}")

    out_path = out_dir / f"recognized_{Path(source).name}"
    cv2.imwrite(str(out_path), frame)
    print(f"Found {len(boxes)} face(s). Saved: {out_path}")


def run_webcam(known_encodings, known_names, tolerance: float, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):
            name, _ = match_face(encoding, known_encodings, known_names, tolerance)
            # Scale coordinates back up to the full-size frame.
            box = (top * 4, right * 4, bottom * 4, left * 4)
            annotate(frame, box, name)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Recognize faces against an enrolled database")
    parser.add_argument("source", nargs="?", default=None,
                         help="Path to an image. Omit to use webcam.")
    parser.add_argument("--db", default="known_faces.pkl",
                         help="Database built by enroll.py")
    parser.add_argument("--tolerance", type=float, default=0.6,
                         help="Lower = stricter match (0.6 is a common default)")
    parser.add_argument("--out-dir", default="./outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    known_encodings, known_names = load_database(args.db)

    if args.source is None:
        run_webcam(known_encodings, known_names, args.tolerance)
    else:
        run_image(args.source, known_encodings, known_names, out_dir, args.tolerance)


if __name__ == "__main__":
    main()
