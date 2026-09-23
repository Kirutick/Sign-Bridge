"""
Sign Bridge — Real ISL Media Extraction Pipeline
=================================================
Extracts MediaPipe hand landmarks from real ISL image and video files,
converts them to canonical 30×63 normalized feature sequences, and saves
as canonical JSON samples ready for ingest_real_isl_dataset.py.

Usage:
    python ml/scripts/extract_landmarks_from_media.py \\
        --input path/to/media_file_or_dir \\
        --label HELLO \\
        --output ml/data/raw \\
        [--stride 10] \\
        [--signer signer_01] \\
        [--mirror_left]

Supported inputs:
    Images : .jpg .jpeg .png .webp
    Videos : .mp4 .webm .mov .mkv .avi

Output:
    ml/data/raw/<LABEL>/<LABEL>_<timestamp>.json
    Each JSON file contains one or more canonical samples.

Canonical sample schema (same as ingest_real_isl_dataset.py expects):
    {
      "id": "<uuid>",
      "label": "<LABEL>",
      "isSynthetic": false,
      "handedness": ["Right"],
      "metadata": {
          "signerId": "<signer_id>",
          "language": "ISL",
          "sourceFile": "<filename>",
          "sourceType": "image|video",
          "frameCount": 30,
          "fps": 30
      },
      "sequence": [ [63 floats] × 30 ]
    }
"""

import os
import sys
import json
import uuid
import argparse
import math
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python not installed. Run: pip install opencv-python", file=sys.stderr)
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.components.containers import NormalizedLandmark
except ImportError:
    print("[ERROR] mediapipe not installed. Run: pip install mediapipe>=0.10.14", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SIGN_LANGUAGE = "ISL"
SIGN_LANGUAGE_NAME = "Indian Sign Language"
OUTPUT_LANGUAGE = "English"
OUTPUT_LANGUAGE_CODE = "en"

WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
DEGENERATE_SCALE_EPSILON = 1e-6

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

# Local MediaPipe model path (will be downloaded if not present)
MP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "hand_landmarker.task"
)

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe Model Download
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_model(model_path: str) -> str:
    """Download hand_landmarker.task if not already present."""
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    import urllib.request
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    print(f"[INFO] Downloading MediaPipe hand_landmarker.task model to {model_path}...")
    urllib.request.urlretrieve(url, model_path)
    print("[INFO] Model downloaded successfully.")
    return model_path


# ──────────────────────────────────────────────────────────────────────────────
# Normalization  (canonical v1 — must match TypeScript & Python ingest)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_frame(landmarks_21: List[NormalizedLandmark], mirror: bool = False) -> Optional[np.ndarray]:
    """
    Canonical v1 normalization:
      1. Translate so wrist (index 0) is the origin.
      2. Scale by Euclidean distance from wrist to middle-MCP (index 9).
      3. Flatten to 63 floats in [x0,y0,z0, x1,y1,z1, …, x20,y20,z20] order.
    Returns None on degenerate scale or NaN/Inf values.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_21], dtype=np.float32)

    if not np.all(np.isfinite(pts)):
        return None

    if mirror:
        # Mirror x-coordinate so left-hand data matches right-hand convention
        pts[:, 0] = -pts[:, 0]

    wrist = pts[WRIST_INDEX].copy()
    pts -= wrist  # Translate to wrist origin

    scale = float(np.linalg.norm(pts[MIDDLE_MCP_INDEX]))
    if scale < DEGENERATE_SCALE_EPSILON:
        return None

    pts /= scale

    if not np.all(np.isfinite(pts)):
        return None

    return pts.flatten()  # shape (63,)


# ──────────────────────────────────────────────────────────────────────────────
# Handedness helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_handedness(result, hand_index: int) -> Tuple[str, float]:
    """Returns (handedness_str, confidence) for a detected hand."""
    if result.handedness and len(result.handedness) > hand_index:
        cats = result.handedness[hand_index]
        if cats:
            return cats[0].category_name, cats[0].score
    return "Unknown", 1.0


def _select_primary_hand(result):
    """
    Select the best single hand from a detection result.
    Preference order: highest handedness confidence → first detected.
    Returns (landmarks_21, handedness_str) or (None, None).
    """
    if not result.hand_landmarks:
        return None, None

    best_idx = 0
    best_score = -1.0
    for i in range(len(result.hand_landmarks)):
        _, score = _get_handedness(result, i)
        if score > best_score:
            best_score = score
            best_idx = i

    lms = result.hand_landmarks[best_idx]
    handedness, _ = _get_handedness(result, best_idx)
    return lms, handedness


# ──────────────────────────────────────────────────────────────────────────────
# Sample building
# ──────────────────────────────────────────────────────────────────────────────
def _build_sample(
    frames_63: List[np.ndarray],
    label: str,
    handedness: str,
    source_file: str,
    source_type: str,
    signer_id: str,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Package 30 normalized frames into a canonical JSON sample."""
    assert len(frames_63) == 30, f"Expected 30 frames, got {len(frames_63)}"
    return {
        "id": str(uuid.uuid4()),
        "label": label.upper(),
        "isSynthetic": False,
        "handedness": [handedness],
        "metadata": {
            "signerId": signer_id,
            "language": "ISL",
            "signLanguage": SIGN_LANGUAGE_NAME,
            "signLanguageCode": SIGN_LANGUAGE,
            "outputLanguage": OUTPUT_LANGUAGE,
            "outputLanguageCode": OUTPUT_LANGUAGE_CODE,
            "sourceFile": os.path.basename(source_file),
            "sourceType": source_type,
            "frameCount": 30,
            "fps": fps,
            "extractedAt": datetime.utcnow().isoformat() + "Z",
            "normalizationVersion": "v1",
        },
        "sequence": [f.tolist() for f in frames_63],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Image processing
# ──────────────────────────────────────────────────────────────────────────────
def process_image(
    file_path: str,
    label: str,
    landmarker_image,
    signer_id: str,
    mirror_left: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Process a single image file.
    Returns (sample_dict_or_None, rejection_reason_or_"ok")
    """
    img = cv2.imread(file_path)
    if img is None:
        return None, "Could not decode image"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = landmarker_image.detect(mp_image)

    lms, handedness = _select_primary_hand(result)
    if lms is None:
        return None, "No hand detected"

    mirror = mirror_left and handedness == "Left"
    vec = normalize_frame(lms, mirror=mirror)
    if vec is None:
        return None, "Normalization failed (degenerate scale or NaN)"

    # For static images, replicate the single frame 30 times
    # This is valid for static signs (ISL alphabet, static gestures)
    frames_63 = [vec.copy() for _ in range(30)]

    sample = _build_sample(
        frames_63=frames_63,
        label=label,
        handedness="Right" if (mirror or handedness == "Right") else handedness,
        source_file=file_path,
        source_type="image",
        signer_id=signer_id,
        fps=0.0,
    )
    # Mark as static sign
    sample["metadata"]["isStaticSign"] = True
    return sample, "ok"


# ──────────────────────────────────────────────────────────────────────────────
# Video processing
# ──────────────────────────────────────────────────────────────────────────────
def process_video(
    file_path: str,
    label: str,
    landmarker_video,
    signer_id: str,
    mirror_left: bool,
    sequence_length: int = 30,
    stride: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process a video file.
    Returns (list_of_samples, stats_dict).

    Strategy:
      - Decode frames in order (temporal integrity preserved)
      - Run MediaPipe on each frame
      - Collect valid normalized 63-D vectors into a buffer
      - When buffer has ≥sequence_length frames, create sequence using stride
      - NEVER shuffle frames before creating sequences
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return [], {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stats = {
        "frames_total": total_frames,
        "frames_read": 0,
        "hands_detected": 0,
        "frames_valid": 0,
        "frames_rejected": 0,
        "sequences_generated": 0,
        "rejection_reasons": {},
    }

    valid_frames: List[Tuple[np.ndarray, str]] = []  # (vec63, handedness)
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        stats["frames_read"] += 1
        frame_idx += 1
        timestamp_ms = int((frame_idx / fps) * 1000)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            result = landmarker_video.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            reason = f"MediaPipe error: {type(e).__name__}"
            stats["frames_rejected"] += 1
            stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
            continue

        lms, handedness = _select_primary_hand(result)

        if lms is None:
            stats["frames_rejected"] += 1
            reason = "No hand detected"
            stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
            continue

        stats["hands_detected"] += 1
        mirror = mirror_left and handedness == "Left"
        vec = normalize_frame(lms, mirror=mirror)

        if vec is None:
            stats["frames_rejected"] += 1
            reason = "Normalization failed"
            stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
            continue

        stats["frames_valid"] += 1
        eff_handedness = "Right" if (mirror or handedness == "Right") else handedness
        valid_frames.append((vec, eff_handedness))

    cap.release()

    if len(valid_frames) < sequence_length:
        print(
            f"  [WARN] Only {len(valid_frames)} valid frames extracted (need ≥{sequence_length}). "
            f"No sequences generated for {os.path.basename(file_path)}."
        )
        return [], stats

    # Sliding-window sequence creation (temporal order preserved)
    samples = []
    pos = 0
    while pos + sequence_length <= len(valid_frames):
        window = valid_frames[pos : pos + sequence_length]
        frames_63 = [f[0] for f in window]
        # Use handedness of the first frame in the window
        window_handedness = window[0][1]

        sample = _build_sample(
            frames_63=frames_63,
            label=label,
            handedness=window_handedness,
            source_file=file_path,
            source_type="video",
            signer_id=signer_id,
            fps=fps,
        )
        sample["metadata"]["isStaticSign"] = False
        sample["metadata"]["windowStart"] = pos
        sample["metadata"]["windowEnd"] = pos + sequence_length
        sample["metadata"]["sourceVideoHash"] = hashlib.sha256(
            os.path.basename(file_path).encode()
        ).hexdigest()[:16]
        samples.append(sample)
        pos += stride

    stats["sequences_generated"] = len(samples)
    return samples, stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from real ISL images/videos → canonical JSON samples"
    )
    parser.add_argument("--input", required=True, help="Image/video file or directory")
    parser.add_argument("--label", required=True, help="ISL English concept label, e.g. HELLO")
    parser.add_argument("--output", default="ml/data/raw", help="Root output directory")
    parser.add_argument("--stride", type=int, default=10, help="Sliding-window stride for video (frames)")
    parser.add_argument("--signer", default="signer_unknown", help="Signer/performer ID")
    parser.add_argument(
        "--mirror_left",
        action="store_true",
        help="Mirror left-hand detections to match right-hand convention",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence threshold",
    )
    args = parser.parse_args()

    label = args.label.strip().upper().replace(" ", "_")
    model_path = _ensure_model(MP_MODEL_PATH)

    # Collect media files
    input_path = args.input
    media_files = []
    if os.path.isfile(input_path):
        media_files = [input_path]
    elif os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                media_files.append(os.path.join(input_path, fname))
    else:
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not media_files:
        print(f"[ERROR] No supported media files found in: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f" SIGN BRIDGE — ISL LANDMARK EXTRACTION")
    print(f"{'='*70}")
    print(f" Label          : {label}")
    print(f" Media files    : {len(media_files)}")
    print(f" Signer ID      : {args.signer}")
    print(f" Mirror left    : {args.mirror_left}")
    print(f" Video stride   : {args.stride} frames")
    print(f" Min confidence : {args.confidence}")
    print(f"{'='*70}\n")

    # Output directory
    out_dir = os.path.join(args.output, label)
    os.makedirs(out_dir, exist_ok=True)

    # Build image-mode landmarker
    img_opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )

    # Build video-mode landmarker
    vid_opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )

    total_samples_saved = 0
    grand_stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "samples_total": 0,
    }

    for file_path in media_files:
        ext = os.path.splitext(file_path)[1].lower()
        fname = os.path.basename(file_path)
        print(f"  Processing: {fname}")

        samples = []
        if ext in IMAGE_EXTENSIONS:
            with mp_vision.HandLandmarker.create_from_options(img_opts) as lmk:
                sample, reason = process_image(
                    file_path, label, lmk, args.signer, args.mirror_left
                )
            if sample:
                samples = [sample]
                print(f"    ✅ Image: 1 valid frame → 1 static-sign sample")
            else:
                print(f"    ❌ Image rejected: {reason}")
                grand_stats["files_skipped"] += 1
                continue

        elif ext in VIDEO_EXTENSIONS:
            with mp_vision.HandLandmarker.create_from_options(vid_opts) as lmk:
                samples, stats = process_video(
                    file_path, label, lmk, args.signer, args.mirror_left,
                    sequence_length=30, stride=args.stride,
                )
            print(
                f"    Frames read: {stats.get('frames_read', 0)} | "
                f"Hands detected: {stats.get('hands_detected', 0)} | "
                f"Valid: {stats.get('frames_valid', 0)} | "
                f"Rejected: {stats.get('frames_rejected', 0)} | "
                f"Sequences: {stats.get('sequences_generated', 0)}"
            )
            if stats.get("rejection_reasons"):
                for reason, count in stats["rejection_reasons"].items():
                    print(f"      Rejection reason: {reason} ({count} frames)")
            if not samples:
                grand_stats["files_skipped"] += 1
                continue
        else:
            continue

        # Save to JSON
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        out_fname = f"{label}_{timestamp_str}.json"
        out_path = os.path.join(out_dir, out_fname)

        output_payload = {
            "version": 1,
            "signLanguage": SIGN_LANGUAGE_NAME,
            "signLanguageCode": SIGN_LANGUAGE,
            "outputLanguage": OUTPUT_LANGUAGE,
            "outputLanguageCode": OUTPUT_LANGUAGE_CODE,
            "isSynthetic": False,
            "featureCount": 63,
            "sequenceLength": 30,
            "samples": samples,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        grand_stats["files_processed"] += 1
        grand_stats["samples_total"] += len(samples)
        total_samples_saved += len(samples)
        print(f"    💾 Saved {len(samples)} sample(s) → {out_path}")

    print(f"\n{'='*70}")
    print(f" EXTRACTION COMPLETE")
    print(f" Files processed : {grand_stats['files_processed']}")
    print(f" Files skipped   : {grand_stats['files_skipped']}")
    print(f" Total samples   : {grand_stats['samples_total']}")
    print(f" Output dir      : {os.path.abspath(out_dir)}")
    print(f"{'='*70}\n")

    if grand_stats["samples_total"] == 0:
        print("[!] WARNING: No samples were extracted. Check that hands are visible in the media files.")


if __name__ == "__main__":
    main()

