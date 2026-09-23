# Model Card: Sign Bridge LSTM Baseline (`v1_20260825_230050_8025d6ac`)

## 1. Model Overview
- **Model Identifier:** `v1_20260825_230050_8025d6ac`
- **Architecture:** Keras Sequential 2-Layer LSTM Baseline
- **Input Tensor Shape:** `(sequence_length=30, feature_count=63)`
- **Output Classes (4):** `HELLO, NO, THANK_YOU, YES`
- **Primary Framework:** TensorFlow 2.17+ / Keras

---

## 2. Intended Use & Deployment Compatibility
- **Intended Use:** Internal baseline benchmark for real-time temporal sign language sequence classification.
- **Export Compatibility:** Standardized feature vector shape (`30x63`, wrist-relative normalized). Ready for TF.js / TFLite deployment handoff.
- **Explicit Non-Use:** Not intended for production safety-critical medical/legal translation without multi-performer validation.

---

## 3. Training & Validation Performance
- **Train Accuracy:** `0.4500`
- **Validation Accuracy:** `0.5000`
- **Held-out Test Accuracy:** `0.7500`
- **Test Macro F1:** `0.6667`
- **Overfitting Audit:** ✅ Healthy (No Significant Overfitting) (Train-Val Gap: `-5.0%`)

---

## 4. Per-Class Performance Breakdown (§7)

| Class Name | Test Samples (n) | Precision | Recall | F1-Score |
|---|---|---|---|---|
| `HELLO` | 5 | 1.0000 | 1.0000 | 1.0000 |
| `NO` | 5 | 0.0000 | 0.0000 | 0.0000 |
| `THANK_YOU` | 5 | 1.0000 | 1.0000 | 1.0000 |
| `YES` | 5 | 0.5000 | 1.0000 | 0.6667 |


### Most Confused Label Pairs:
- 'NO' predicted as 'YES' in 5/5 cases (100.0%)


---

## 5. Honest Limitations & Data Leakage Assessment (§12)

- **Leakage Prevention Status:** ✅ Group-Aware Performer Split Applied
- **Performer Generalization Disclaimer:** Performer group split enforced. Model evaluated on unseen performers.
- **Sample Coverage Notice:** All classes evaluated with n sample counts noted above. Small sample classes (n < 10) should be considered low-confidence benchmarks.
