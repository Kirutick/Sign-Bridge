"""
Cross-Language Parity Test Suite: Python vs TypeScript Enhanced Feature Extractors

Validates that Python (enhanced_features.py) and TypeScript (enhancedFeatureProcessor.ts)
produce mathematically identical feature representations with maximum absolute error < 1e-6.
"""

import math
import subprocess
import json
import os
import tempfile
import numpy as np
import pytest
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.abspath("."))

from ml.preprocessing.enhanced_features import (
    extract_sequence_enhanced_features,
    FEATURE_SCHEMAS,
)


def generate_sequence_fixtures() -> List[Dict[str, Any]]:
    """Generates varied test fixture sequences (30 frames x 63 floats) to stress-test feature math."""
    fixtures = []

    # 1. Base Canonical Hand Frame
    base_hand = []
    for i in range(21):
        base_hand.extend([
            0.5 + 0.05 * math.cos(i * 0.3),
            0.5 + 0.05 * math.sin(i * 0.3),
            0.01 * (i % 5)
        ])

    # 1. Static Hand (No movement over 30 frames)
    static_seq = [list(base_hand) for _ in range(30)]
    fixtures.append({"name": "static_hand_sequence", "sequence": static_seq})

    # 2. Linear Moving Hand (Constant velocity)
    linear_seq = []
    for t in range(30):
        frame = []
        for i in range(21):
            frame.extend([
                base_hand[i * 3] + 0.01 * t,
                base_hand[i * 3 + 1] - 0.005 * t,
                base_hand[i * 3 + 2] + 0.002 * t,
            ])
        linear_seq.append(frame)
    fixtures.append({"name": "linear_moving_hand", "sequence": linear_seq})

    # 3. Accelerating Hand (Non-linear movement)
    accel_seq = []
    for t in range(30):
        frame = []
        for i in range(21):
            frame.extend([
                base_hand[i * 3] + 0.001 * (t ** 2),
                base_hand[i * 3 + 1] + 0.02 * math.sin(t * 0.2),
                base_hand[i * 3 + 2] + 0.01 * math.cos(t * 0.15),
            ])
        accel_seq.append(frame)
    fixtures.append({"name": "accelerating_oscillating_hand", "sequence": accel_seq})

    # 4. Large Scaled Hand Sequence
    large_seq = []
    for t in range(30):
        frame = []
        for i in range(21):
            frame.extend([
                base_hand[i * 3] * 3.5,
                base_hand[i * 3 + 1] * 3.5,
                base_hand[i * 3 + 2] * 3.5,
            ])
        large_seq.append(frame)
    fixtures.append({"name": "large_scaled_hand", "sequence": large_seq})

    # 5. Small Scaled Hand Sequence
    small_seq = []
    for t in range(30):
        frame = []
        for i in range(21):
            frame.extend([
                base_hand[i * 3] * 0.05,
                base_hand[i * 3 + 1] * 0.05,
                base_hand[i * 3 + 2] * 0.05,
            ])
        small_seq.append(frame)
    fixtures.append({"name": "small_scaled_hand", "sequence": small_seq})

    # 6. Negative Coordinates Hand Sequence
    neg_seq = []
    for t in range(30):
        frame = []
        for i in range(21):
            frame.extend([
                -0.8 + 0.02 * i - 0.003 * t,
                -0.5 + 0.03 * (i % 4) + 0.002 * t,
                -0.1 + 0.01 * i,
            ])
        neg_seq.append(frame)
    fixtures.append({"name": "negative_coordinates_sequence", "sequence": neg_seq})

    # 7. Degenerate Flat / Straightened Hand (Colinear points)
    flat_seq = []
    for t in range(30):
        frame = []
        for i in range(21):
            frame.extend([
                float(i) * 0.1, # Perfectly colinear in X
                0.0,
                0.0,
            ])
        flat_seq.append(frame)
    fixtures.append({"name": "degenerate_colinear_hand", "sequence": flat_seq})

    return fixtures


@pytest.mark.parametrize("schema", [
    "v1_single_hand_63",
    "v2_single_hand_geometry",
    "v3_single_hand_geom_vel",
    "v4_single_hand_geom_vel_acc",
    "v5_single_hand_all",
])
def test_cross_language_enhanced_features_parity(schema: str):
    fixtures = generate_sequence_fixtures()
    expected_dim = FEATURE_SCHEMAS[schema]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as in_f:
        json.dump(fixtures, in_f)
        in_path = in_f.name

    out_path = in_path.replace(".json", "_ts_out.json")

    try:
        proc = subprocess.run(
            ["npx", "tsx", "ml/scripts/run_ts_enhanced_features.ts", in_path, out_path, schema],
            capture_output=True,
            text=True,
            shell=True,
            check=True
        )

        with open(out_path, "r", encoding="utf-8") as f:
            ts_results = json.load(f)

        assert len(ts_results) == len(fixtures)

        max_diff_observed = 0.0
        diff_count = 0
        total_comparisons = 0
        tolerance = 1e-6

        for i, fix in enumerate(fixtures):
            py_seq = np.array(fix["sequence"], dtype=np.float64)
            py_enhanced = extract_sequence_enhanced_features(py_seq, schema=schema)

            ts_res = ts_results[i]
            assert ts_res["ok"] is True, f"TS failed on {fix['name']}: {ts_res.get('error')}"

            ts_enhanced = np.array(ts_res["features"], dtype=np.float32)

            assert py_enhanced.shape == (30, expected_dim), f"Py shape mismatch: {py_enhanced.shape}"
            assert ts_enhanced.shape == (30, expected_dim), f"TS shape mismatch: {ts_enhanced.shape}"

            diff_matrix = np.abs(py_enhanced - ts_enhanced)
            max_diff = float(np.max(diff_matrix))
            if max_diff > max_diff_observed:
                max_diff_observed = max_diff

            bad_indices = np.where(diff_matrix > tolerance)
            if len(bad_indices[0]) > 0:
                diff_count += len(bad_indices[0])
                first_t = bad_indices[0][0]
                first_feat = bad_indices[1][0]
                print(
                    f"Mismatch on {fix['name']} [{schema}] frame {first_t}, feat {first_feat}: "
                    f"Py={py_enhanced[first_t, first_feat]:.8f}, TS={ts_enhanced[first_t, first_feat]:.8f}, "
                    f"diff={diff_matrix[first_t, first_feat]:.10e}"
                )

            total_comparisons += 30 * expected_dim

        print(f"\n=======================================================")
        print(f" PYTHON <-> TYPESCRIPT ENHANCED FEATURE PARITY ({schema})")
        print(f"=======================================================")
        print(f"Fixtures Evaluated: {len(fixtures)}")
        print(f"Feature Dimension: {expected_dim}")
        print(f"Total Value Comparisons: {total_comparisons}")
        print(f"Max Absolute Difference: {max_diff_observed:.10e}")
        print(f"Tolerance Violations (> {tolerance}): {diff_count}")
        print(f"Status: {'PASS' if diff_count == 0 else 'FAIL'}")
        print(f"=======================================================\n")

        assert diff_count == 0, f"Found {diff_count} feature mismatches exceeding {tolerance} for schema {schema}"
        assert max_diff_observed < tolerance

    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)
