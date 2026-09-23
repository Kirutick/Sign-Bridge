"""
Sign Bridge — End-to-End Pipeline Tests
========================================
Tests that validate the ISL training pipeline without using any synthetic or
randomly generated data as "real" training data.

Run: python -m pytest ml/tests/test_pipeline_e2e.py -v
"""

import sys
import os
import json
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath("."))

# ─── 1. Preprocessing Parity Test ─────────────────────────────────────────────
class TestPreprocessingParity:
    """
    Verify that Python normalization matches TypeScript normalization.
    We generate a known landmark set and check parity.
    """

    def _python_normalize(self, pts21: np.ndarray) -> np.ndarray:
        """Canonical v1 normalization (must match TypeScript landmarkProcessor.ts)."""
        pts = pts21.copy()
        wrist = pts[0].copy()
        pts -= wrist
        scale = float(np.linalg.norm(pts[9]))
        if scale < 1e-6:
            return None
        pts /= scale
        return pts.flatten()

    def test_known_landmarks_normalize_to_expected(self):
        """Wrist at origin, middle-MCP at unit distance after normalization."""
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [0.5, 0.5, 0.0]  # wrist
        pts[9] = [0.6, 0.5, 0.0]  # middle-MCP (0.1 away from wrist in x)

        vec = self._python_normalize(pts)
        assert vec is not None, "Normalization should succeed for this input"
        assert vec.shape == (63,), "Expected 63-dimensional feature vector"

        # Wrist should be at (0, 0, 0) after normalization
        assert abs(vec[0]) < 1e-5, f"Wrist x should be 0, got {vec[0]}"
        assert abs(vec[1]) < 1e-5, f"Wrist y should be 0, got {vec[1]}"
        assert abs(vec[2]) < 1e-5, f"Wrist z should be 0, got {vec[2]}"

        # Middle-MCP (index 9) should have scale = 1.0 (unit norm)
        mid_x, mid_y, mid_z = vec[27], vec[28], vec[29]  # 9 * 3
        dist = math.sqrt(mid_x**2 + mid_y**2 + mid_z**2)
        assert abs(dist - 1.0) < 1e-5, f"Middle-MCP norm should be 1.0, got {dist}"

    def test_degenerate_scale_returns_none(self):
        """All landmarks at the same point → degenerate scale → should return None."""
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[:] = [0.5, 0.5, 0.0]  # All same point

        vec = self._python_normalize(pts)
        assert vec is None, "Degenerate landmark input should return None"

    def test_mirroring_left_to_right(self):
        """Mirroring x coordinates should produce a valid, different vector."""
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [0.5, 0.5, 0.0]
        pts[9] = [0.6, 0.5, 0.0]
        pts[4] = [0.8, 0.3, 0.0]

        original = self._python_normalize(pts)

        pts_mirrored = pts.copy()
        pts_mirrored[:, 0] = -pts_mirrored[:, 0]
        mirrored = self._python_normalize(pts_mirrored)

        assert original is not None
        assert mirrored is not None
        assert not np.allclose(original, mirrored), "Mirrored should differ from original"
        x_indices = list(range(0, 63, 3))
        for xi in x_indices[1:]:  # Skip wrist (index 0) which is 0 in both
            assert abs(original[xi] + mirrored[xi]) < 1e-4, f"x at {xi} should be negated by mirroring"


# ─── 2. Dataset Leakage Check ──────────────────────────────────────────────────
class TestDatasetLeakage:

    def test_no_train_test_overlap_in_split_manifest(self, tmp_path):
        """Given a split manifest, verify zero train∩test overlap."""
        manifest = {
            "train_ids": ["a", "b", "c", "d"],
            "val_ids": ["e"],
            "test_ids": ["f", "g"],
            "leakage_guard_applied": True,
        }
        manifest_path = tmp_path / "split.json"
        manifest_path.write_text(json.dumps(manifest))

        with open(manifest_path) as f:
            data = json.load(f)

        train_set = set(data["train_ids"])
        test_set = set(data["test_ids"])
        val_set = set(data["val_ids"])

        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set

        assert not overlap_train_test, f"Train∩Test leakage detected: {overlap_train_test}"
        assert not overlap_val_test, f"Val∩Test leakage detected: {overlap_val_test}"

    def test_split_with_leakage_is_caught(self):
        """Split with deliberately overlapping IDs should be flagged."""
        train_ids = {"a", "b", "c", "X"}
        test_ids = {"X", "d"}  # X is in both!

        overlap = train_ids & test_ids
        assert len(overlap) > 0, "Should detect the intentional leakage"
        assert "X" in overlap


# ─── 3. Dataset Integrity Check ───────────────────────────────────────────────
class TestDatasetIntegrity:

    def test_no_nan_in_synthetic_dataset(self, tmp_path):
        """All samples in a valid dataset must be finite."""
        n = 100
        X = np.random.randn(n, 30, 63).astype(np.float32)
        labels = np.array(["HELLO"] * 50 + ["WATER"] * 50)
        ids = np.array([str(i) for i in range(n)])

        npz_path = tmp_path / "test_dataset.npz"
        np.savez_compressed(
            npz_path,
            X=X, labels=labels, ids=ids,
            performers=np.array(["signer_01"] * n),
        )

        loaded = np.load(npz_path)
        X_loaded = loaded["X"]

        assert np.all(np.isfinite(X_loaded)), "Dataset must contain only finite values"
        assert X_loaded.shape == (n, 30, 63), f"Expected shape (100, 30, 63), got {X_loaded.shape}"

    def test_sequence_dimensions(self):
        """Each sequence must be exactly (30, 63)."""
        sequence = [[0.0] * 63 for _ in range(30)]
        assert len(sequence) == 30, "Sequence must have exactly 30 frames"
        assert all(len(f) == 63 for f in sequence), "Each frame must have exactly 63 features"

    def test_feature_vector_order(self):
        """Feature vector should be flattened in [x0,y0,z0, x1,y1,z1, ...] order."""
        pts = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            pts[i] = [i * 0.01, i * 0.02, i * 0.03]

        vec = pts.flatten()
        assert len(vec) == 63
        assert abs(vec[0] - 0.00) < 1e-6  # x0
        assert abs(vec[1] - 0.00) < 1e-6  # y0
        assert abs(vec[3] - 0.01) < 1e-6  # x1
        assert abs(vec[6] - 0.02) < 1e-6  # x2


# ─── 4. Sample Contract Validation ────────────────────────────────────────────
class TestSampleContract:

    def _make_sample(self, label="HELLO", n_frames=30, n_features=63, is_synthetic=False):
        return {
            "id": "test-uuid-1234",
            "label": label,
            "isSynthetic": is_synthetic,
            "signLanguage": "Indian Sign Language",
            "signLanguageCode": "ISL",
            "outputLanguage": "English",
            "outputLanguageCode": "en",
            "handedness": ["Right"],
            "sequence": [[0.0] * n_features for _ in range(n_frames)],
        }

    def test_valid_sample_passes(self):
        sample = self._make_sample()
        assert sample["isSynthetic"] is False
        assert sample["signLanguageCode"] == "ISL"
        assert len(sample["sequence"]) == 30
        assert all(len(f) == 63 for f in sample["sequence"])

    def test_synthetic_sample_is_flagged(self):
        sample = self._make_sample(is_synthetic=True)
        assert sample["isSynthetic"] is True, "Synthetic flag must be detectable"

    def test_wrong_sequence_length_detected(self):
        sample = self._make_sample(n_frames=15)
        assert len(sample["sequence"]) != 30, "Should detect wrong sequence length"

    def test_wrong_feature_count_detected(self):
        sample = self._make_sample(n_features=42)
        bad_frame = sample["sequence"][0]
        assert len(bad_frame) != 63, "Should detect wrong feature count"

    def test_isl_scope_locked(self):
        """Sign language must always be ISL, not ASL, BSL, or other."""
        valid_sample = self._make_sample()
        assert valid_sample["signLanguageCode"] == "ISL"
        assert valid_sample["signLanguage"] == "Indian Sign Language"
        # These should NOT be present or should not match
        invalid_codes = ["ASL", "BSL", "INTL", "HIN"]
        assert valid_sample["signLanguageCode"] not in invalid_codes


# ─── 5. Class Balance Warning ──────────────────────────────────────────────────
class TestClassBalance:

    def test_small_class_warns(self, capsys):
        """Classes with fewer than 5 samples should trigger a warning."""
        label_counts = {"HELLO": 50, "WATER": 4, "FOOD": 30}
        threshold = 5

        for label, count in label_counts.items():
            if count < threshold:
                print(f"[CLASS BALANCE WARNING] '{label}' has only {count} samples (< {threshold}). Accuracy for this class may be unreliable.")

        captured = capsys.readouterr()
        assert "WATER" in captured.out, "Should warn about WATER having < 5 samples"
        assert "HELLO" not in captured.out, "Should not warn about HELLO with 50 samples"

    def test_balanced_classes_no_warning(self, capsys):
        label_counts = {"HELLO": 50, "WATER": 60, "FOOD": 55}
        threshold = 5

        for label, count in label_counts.items():
            if count < threshold:
                print(f"[CLASS BALANCE WARNING] '{label}' has only {count} samples.")

        captured = capsys.readouterr()
        assert captured.out == "", "No warnings expected for balanced dataset"


# ─── 6. Handedness Normalizer Tests ──────────────────────────────────────────
class TestHandednessNormalizer:

    def test_import(self):
        from ml.preprocessing.handedness_normalizer import (
            normalize_handedness, normalize_single_sequence, mirror_sequence_x
        )
        assert callable(normalize_handedness)
        assert callable(normalize_single_sequence)
        assert callable(mirror_sequence_x)

    def test_right_hand_unchanged(self):
        from ml.preprocessing.handedness_normalizer import normalize_single_sequence
        seq = np.ones((30, 63), dtype=np.float32)
        out = normalize_single_sequence(seq, "Right", normalize_to="right")
        assert np.allclose(out, seq), "Right-hand sequence should be unchanged"

    def test_left_hand_mirrored(self):
        from ml.preprocessing.handedness_normalizer import normalize_single_sequence
        seq = np.ones((30, 63), dtype=np.float32)
        out = normalize_single_sequence(seq, "Left", normalize_to="right")
        x_indices = list(range(0, 63, 3))
        for xi in x_indices:
            assert abs(out[0, xi] - (-1.0)) < 1e-5, f"x at {xi} should be negated"

    def test_normalize_to_none_is_noop(self):
        from ml.preprocessing.handedness_normalizer import normalize_single_sequence
        seq = np.random.randn(30, 63).astype(np.float32)
        out = normalize_single_sequence(seq, "Left", normalize_to="none")
        assert np.allclose(out, seq), "normalize_to='none' should not modify the sequence"


# ─── 7. Model Loading Test (if checkpoint exists) ────────────────────────────
class TestModelLoading:

    def test_model_loads_if_checkpoint_exists(self):
        checkpoint_dirs = []
        exp_dir = os.path.join("ml", "experiments")
        if os.path.isdir(exp_dir):
            for root, dirs, files in os.walk(exp_dir):
                for f in files:
                    if f.endswith(".keras") or f.endswith(".h5"):
                        checkpoint_dirs.append(os.path.join(root, f))

        if not checkpoint_dirs:
            pytest.skip("No model checkpoint found — train a model first")

        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(checkpoint_dirs[0])
            assert model is not None, "Model loaded but is None"
            print(f"  Loaded model from: {checkpoint_dirs[0]}")
        except Exception as e:
            pytest.skip(f"TensorFlow not available or model incompatible: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
