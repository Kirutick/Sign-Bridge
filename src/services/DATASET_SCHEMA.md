# Sign Bridge Dataset Schema & Data Pipeline Specification

Version: **1.0.0**  
Export Schema Version: **1**

---

## 1. IndexedDB Schema Specification (§5a)

- **Database Name:** `SignBridgeDatasetDB`
- **Database Version:** `1`
- **Object Store Name:** `samples`
- **Key Path:** `id` (string, UUID v4)
- **Indexes:**
  - `label`: Non-unique index on `label` for fast per-class queries (`listSamples(label)` and `listLabels()`) without full table scans.
  - `createdAt`: Non-unique index on `createdAt` timestamp for temporal sorting and recency queries.

### Migration Strategy (§5b):
If the record shape evolves in future releases, the database version will be incremented (`version = 2`), and an `onupgradeneeded` handler will perform structural data migrations without corrupting or deleting existing user datasets.

---

## 2. DatasetSample Structure (§6)

Each sample represents a temporal sequence of preprocessed landmark vectors captured for a single sign gesture.

```typescript
interface DatasetSample {
  id: string;                         // Unique UUID v4
  label: string;                      // Normalized string (e.g. "THANK_YOU")
  sequence: number[][];               // Array of length sequenceLength, each element of length featureCount
  sequenceLength: number;             // e.g. 30 frames
  featureCount: number;               // e.g. 63 (1 hand) or 126 (2 hands)
  createdAt: string;                  // ISO 8601 UTC timestamp
  handedness?: ('Left' | 'Right')[]; // Primary detected handedness per sample
  metadata?: {
    performerId?: string;             // Unique ID tracking performer variation
    notes?: string;                   // Optional notes
    fps?: number;                     // Average capture frame rate
    sourceResolution?: { width: number; height: number };
    appVersion?: string;              // "0.3.0"
  };
}
```

---

## 3. JSON Export Format (§8)

Export files use schema `version: 1`.

```json
{
  "version": 1,
  "featureCount": 63,
  "sequenceLength": 30,
  "exportedAt": "2026-08-25T22:00:00.000Z",
  "samples": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "label": "HELLO",
      "sequence": [
        [0.0, 0.0, 0.0, -0.281, -0.194, -0.041, /* ... 63 floats */],
        /* ... exactly 30 frame rows */
      ],
      "sequenceLength": 30,
      "featureCount": 63,
      "handedness": ["Right"],
      "createdAt": "2026-08-25T22:00:00.000Z",
      "metadata": {
        "performerId": "p1",
        "fps": 30,
        "appVersion": "0.3.0"
      }
    }
  ]
}
```

---

## 4. Frame Validity & Restart Policy (§3a)

> **Policy Decision:** A sign gesture is a continuous temporal signal. If a hand is lost at any tick during recording (frames 1–29), inserting zero-vectors, interpolated placeholders, or continuing with gaps disrupts temporal learning for LSTM models.
> 
> **Rule:** Upon detecting a hand loss during mid-recording, the current sample **immediately restarts from frame 0**. A prominent visual warning ("⚠ Hand lost — sample restarted") is displayed to the user.

---

## 5. Multi-Hand & Feature Configuration Policy (§3b)

- **One-Hand Signs:** `featureCount = 63` (21 landmarks $\times$ 3 coordinates $x, y, z$).
- **Two-Hand Signs:** `featureCount = 126` (42 landmarks $\times$ 3 coordinates $x, y, z$).
- **Config Enforcement:** The dataset sequence length (default 30) and feature count (default 63) are locked per dataset session. Samples added to a dataset must strictly match the declared `sequenceLength` and `featureCount`.
