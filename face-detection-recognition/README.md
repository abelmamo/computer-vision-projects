# Face Detection & Recognition

Two-stage pipeline: fast Haar-cascade face **detection** (OpenCV), plus a
full face **recognition** system built on `face_recognition` (dlib
embeddings) that enrolls known identities from example photos and matches
new faces against them.

## Setup

```bash
pip install -r requirements.txt
```

> `face_recognition` depends on `dlib`, which needs CMake and a C++ compiler
> to build. On Ubuntu: `sudo apt install cmake build-essential`.

## 1. Detect faces (no enrollment needed)

```bash
# Image
python detect_faces.py path/to/photo.jpg

# Webcam
python detect_faces.py
```

Draws bounding boxes around every detected face and saves to `./outputs/`.

## 2. Enroll known people

Organize reference photos as one folder per person:

```
known_faces/
  alice/  1.jpg  2.jpg
  bob/    1.jpg  2.jpg
```

```bash
python enroll.py known_faces/ --out known_faces.pkl
```

This computes a 128-d face embedding per photo and stores it in
`known_faces.pkl`.

## 3. Recognize faces

```bash
# Image
python recognize.py path/to/photo.jpg --db known_faces.pkl

# Webcam (live)
python recognize.py --db known_faces.pkl
```

Each detected face is compared against the enrolled database by embedding
distance; matches below `--tolerance` (default 0.6) are labeled with the
person's name, otherwise "Unknown".

## Files

- `detect_faces.py` — Haar-cascade face detection (image/webcam)
- `enroll.py` — builds a face-embedding database from labeled photos
- `recognize.py` — detects + identifies faces against the enrolled database
- `requirements.txt` — dependencies
