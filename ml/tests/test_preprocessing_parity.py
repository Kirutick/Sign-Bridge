import math
import subprocess
import json
import os
import tempfile
import numpy as np
import pytest
from typing import List, Dict, Any, Tuple

DEGENERATE_SCALE_EPSILON = 1e-6

def normalize_landmarks_python(landmarks: List[Dict[str, float]]) -> Tuple[bool, List[float], str]:
    """
    Python reference implementation of the canonical normalization algorithm (NORMALIZATION_VERSION = 'v1').
    Contract:
      - wrist = landmarks[0]
      - middle_mcp = landmarks[9]
      - scale = sqrt((x9-x0)^2 + (y9-y0)^2 + (z9-z0)^2)
      - if scale < 1e-6: return (False, [], "DEGENERATE_SCALE")
      - for each landmark i in 0..20:
          normalized[i] = (lm[i] - wrist) / scale
      - flat 63 features: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    """
    if len(landmarks) != 21:
        return False, [], "WRONG_LANDMARK_COUNT"
        
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    
    tx9 = middle_mcp["x"] - wrist["x"]
    ty9 = middle_mcp["y"] - wrist["y"]
    tz9 = middle_mcp["z"] - wrist["z"]
    
    scale = math.sqrt(tx9 * tx9 + ty9 * ty9 + tz9 * tz9)
    if scale < DEGENERATE_SCALE_EPSILON:
        return False, [], "DEGENERATE_SCALE"
        
    features = []
    for lm in landmarks:
        nx = (lm["x"] - wrist["x"]) / scale
        ny = (lm["y"] - wrist["y"]) / scale
        nz = (lm["z"] - wrist["z"]) / scale
        features.extend([nx, ny, nz])
        
    return True, features, ""

def generate_parity_test_fixtures() -> List[Dict[str, Any]]:
    """Generates varied test fixture cases to stress-test normalization math."""
    fixtures = []
    
    # 1. Standard Centered Hand
    base_hand = []
    for i in range(21):
        base_hand.append({
            "x": 0.5 + 0.05 * math.cos(i * 0.3),
            "y": 0.5 + 0.05 * math.sin(i * 0.3),
            "z": 0.01 * (i % 5)
        })
    fixtures.append({"name": "standard_centered", "landmarks": base_hand})
    
    # 2. Translated Hand (offset by +0.3, -0.2, +0.1)
    trans_hand = []
    for lm in base_hand:
        trans_hand.append({
            "x": lm["x"] + 0.3,
            "y": lm["y"] - 0.2,
            "z": lm["z"] + 0.1
        })
    fixtures.append({"name": "translated_hand", "landmarks": trans_hand})
    
    # 3. Scaled Hand (2.5x larger distance to middle MCP)
    scaled_hand = []
    wrist = base_hand[0]
    for lm in base_hand:
        scaled_hand.append({
            "x": wrist["x"] + (lm["x"] - wrist["x"]) * 2.5,
            "y": wrist["y"] + (lm["y"] - wrist["y"]) * 2.5,
            "z": wrist["z"] + (lm["z"] - wrist["z"]) * 2.5
        })
    fixtures.append({"name": "scaled_hand_large", "landmarks": scaled_hand})
    
    # 4. Small Scale Hand (0.1x distance)
    small_hand = []
    for lm in base_hand:
        small_hand.append({
            "x": wrist["x"] + (lm["x"] - wrist["x"]) * 0.1,
            "y": wrist["y"] + (lm["y"] - wrist["y"]) * 0.1,
            "z": wrist["z"] + (lm["z"] - wrist["z"]) * 0.1
        })
    fixtures.append({"name": "scaled_hand_small", "landmarks": small_hand})
    
    # 5. Negative / Extreme Coordinate Bounds
    neg_hand = []
    for i, lm in enumerate(base_hand):
        neg_hand.append({
            "x": -0.8 + 0.02 * i,
            "y": -0.5 + 0.03 * (i % 4),
            "z": -0.1 + 0.01 * i
        })
    fixtures.append({"name": "negative_coordinates", "landmarks": neg_hand})
    
    # 6. Degenerate Scale (wrist == middle_mcp)
    degen_hand = [dict(lm) for lm in base_hand]
    degen_hand[9] = dict(degen_hand[0]) # middle_mcp set to wrist
    fixtures.append({"name": "degenerate_scale", "landmarks": degen_hand})
    
    return fixtures

def test_preprocessing_parity_fixtures():
    fixtures = generate_parity_test_fixtures()
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as in_f:
        json.dump(fixtures, in_f)
        in_path = in_f.name
        
    out_path = in_path.replace(".json", "_out.json")
    
    try:
        proc = subprocess.run(
            ["npx", "tsx", "ml/scripts/run_ts_normalization.ts", in_path, out_path],
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
            py_ok, py_features, py_reason = normalize_landmarks_python(fix["landmarks"])
            ts_res = ts_results[i]
            
            assert py_ok == ts_res["ok"], f"Parity status mismatch on {fix['name']}: py={py_ok}, ts={ts_res['ok']}"
            
            if not py_ok:
                assert py_reason == ts_res["reason"], f"Rejection reason mismatch on {fix['name']}: py={py_reason}, ts={ts_res['reason']}"
                continue
                
            assert len(py_features) == 63
            assert len(ts_res["features"]) == 63
            
            for feat_idx in range(63):
                py_val = py_features[feat_idx]
                ts_val = ts_res["features"][feat_idx]
                diff = abs(py_val - ts_val)
                
                if diff > max_diff_observed:
                    max_diff_observed = diff
                    
                if diff > tolerance:
                    diff_count += 1
                    
                total_comparisons += 1
                
        print(f"\n=======================================================")
        print(f" PYTHON <-> TYPESCRIPT PREPROCESSING PARITY REPORT (B.8)")
        print(f"=======================================================")
        print(f"Fixtures Evaluated: {len(fixtures)}")
        print(f"Total Feature Comparisons: {total_comparisons}")
        print(f"Max Absolute Difference Observed: {max_diff_observed:.10e}")
        print(f"Differences Exceeding Tolerance ({tolerance}): {diff_count}")
        print(f"Parity Status: {'PASS' if diff_count == 0 else 'FAIL'}")
        print(f"=======================================================\n")
        
        assert diff_count == 0, f"Found {diff_count} features with difference > {tolerance}"
        assert max_diff_observed < tolerance
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)
