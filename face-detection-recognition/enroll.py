import argparse
import pickle
from pathlib import Path

import face_recognition


def build_database(known_faces_dir: str, out_path: str):
    """
    Expects a directory structured as:
        known_faces_dir/
          alice/  1.jpg  2.jpg ...
          bob/    1.jpg ...

    Each subfolder name is treated as the person's identity label.
    """
    known_faces_dir = Path(known_faces_dir)
    encodings, names = [], []

    for person_dir in sorted(known_faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        image_paths = [
            p for p in person_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        for image_path in image_paths:
            image = face_recognition.load_image_file(str(image_path))
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                print(f"  [skip] no face found in {image_path}")
                continue

            face_encodings = face_recognition.face_encodings(image, face_locations)
            encodings.append(face_encodings[0])
            names.append(person_name)
            print(f"  [ok] {image_path} -> {person_name}")

    with open(out_path, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    print(f"\nEnrolled {len(encodings)} face(s) across {len(set(names))} identity(ies).")
    print(f"Saved database to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a face-recognition database from labeled example photos"
    )
    parser.add_argument("known_faces_dir",
                         help="Directory with one subfolder of photos per person")
    parser.add_argument("--out", default="known_faces.pkl")
    args = parser.parse_args()

    build_database(args.known_faces_dir, args.out)
