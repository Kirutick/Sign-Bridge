# Real Indian Sign Language (ISL) Dataset Ingestion Guide

## Overview

Sign Bridge enforces strict data integrity:
- **Sign Language**: Indian Sign Language (ISL) exclusively.
- **Output Language**: English concepts.
- **Input Contract**: 30 frames per sequence, 63 normalized coordinates ($21 \times 3$ floats).
- **Prohibited**: Synthetic sine wave data, ASL, BSL, International Sign, or unverified data.

---

## How to Collect and Ingest Real ISL Recordings

### Method 1: Using the In-App Dataset Studio (Webcam)
1. Open the Sign Bridge web application at `http://localhost:3000/`.
2. Navigate to the **Dataset Studio** tab.
3. Select an ISL concept or letter (e.g. `WATER`, `FOOD`, `HELP`, `A`–`Z`).
4. Perform the authentic ISL sign in front of the camera and click **Record Sample (30 Frames)**.
5. In the **Dataset Manager** subtab, click **Export Dataset (JSON)**.
6. Save the exported `.json` file into:
   ```
   ml/data/real_isl/
   ```
7. Run the ingestion validator:
   ```powershell
   & "F:\all\New folder\ml\venv\Scripts\python.exe" ml/scripts/ingest_real_isl_dataset.py --input ml/data/real_isl
   ```

### Method 2: Ingesting External ISL Benchmark Datasets
If importing an external academic ISL dataset (e.g., from ISLRTC or recognized research institutions):
1. Extract MediaPipe HandLandmarker 21 landmarks for each frame of real video clips.
2. Format the sequences into JSON conforming to the schema below.
3. Run the ingestion script:
   ```powershell
   & "F:\all\New folder\ml\venv\Scripts\python.exe" ml/scripts/ingest_real_isl_dataset.py --input <path_to_isl_json>
   ```

---

## Canonical JSON Schema Contract

Each sample must be formatted as:
```json
{
  "id": "uuid-or-id-string",
  "label": "WATER",
  "sequenceLength": 30,
  "featureCount": 63,
  "isSynthetic": false,
  "handedness": ["Right"],
  "metadata": {
    "signerId": "signer_01_native",
    "language": "ISL",
    "fps": 30
  },
  "sequence": [
    [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20],
    ... (30 frames total)
  ]
}
```

---

## Training on Real Ingested ISL Data

Once real ISL data is validated and packaged into `ml/data/processed/dataset_real_isl_v1.npz`:
```powershell
& "F:\all\New folder\ml\venv\Scripts\python.exe" ml/scripts/train_isl_model.py ml/data/processed/dataset_real_isl_v1.npz
```
The script will automatically detect that the data is real human recordings, train without synthetic disclaimers, and export the validated model weights to `public/models/sign-model-isl/`.
