"""
Sign Bridge - Enhanced Hand Geometry & Motion Feature Extractor (Python Reference Implementation)

Implements mathematically rigorous, zero-division safe feature engineering
from MediaPipe 21 hand landmarks across feature groups A through K.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Union

# MediaPipe Canonical Landmark Indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

EPSILON = 1e-7

FEATURE_SCHEMAS = {
    "v1_single_hand_63": 63,
    "v2_single_hand_geometry": 134,
    "v3_single_hand_geom_vel": 203,
    "v4_single_hand_geom_vel_acc": 266,
    "v5_single_hand_all": 271,
}

# Anatomical angle triplets (A, B, C) where B is the vertex
ANGLE_TRIPLETS = [
    # Thumb (3)
    (WRIST, THUMB_CMC, THUMB_MCP),
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    # Index (3)
    (WRIST, INDEX_MCP, INDEX_PIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    # Middle (3)
    (WRIST, MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    # Ring (3)
    (WRIST, RING_MCP, RING_PIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    # Pinky (3)
    (WRIST, PINKY_MCP, PINKY_PIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]

# Finger segment pairs
FINGER_SEGMENTS = [
    # Thumb (3)
    (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # Index (3)
    (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # Middle (3)
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # Ring (3)
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # Pinky (3)
    (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]

# Pairwise fingertip pairs
FINGERTIP_PAIRS = [
    (THUMB_TIP, INDEX_TIP),
    (THUMB_TIP, MIDDLE_TIP),
    (THUMB_TIP, RING_TIP),
    (THUMB_TIP, PINKY_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (INDEX_TIP, RING_TIP),
    (INDEX_TIP, PINKY_TIP),
    (MIDDLE_TIP, RING_TIP),
    (MIDDLE_TIP, PINKY_TIP),
    (RING_TIP, PINKY_TIP),
]

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def _euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculates 3D Euclidean distance between two points."""
    diff = p1 - p2
    return float(np.sqrt(np.dot(diff, diff)))


def _calculate_joint_angle(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray) -> float:
    """
    Calculates angle in radians between vector (A - B) and vector (C - B).
    Vertex is at point B.
    Safely clamps cosine to [-1.0, 1.0] and guards against degenerate zero vectors.
    """
    u = p_a - p_b
    v = p_c - p_b
    
    norm_u = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    norm_v = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    
    if norm_u < EPSILON or norm_v < EPSILON:
        return 0.0
        
    dot_val = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    cos_theta = dot_val / (norm_u * norm_v)
    
    # Numerical clamp
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
        
    return float(math.acos(cos_theta))


def extract_geometric_features_frame(landmarks_21x3: np.ndarray) -> List[float]:
    """
    Extracts all static and geometric features (Groups A through G) for a single frame.
    Input: landmarks_21x3 of shape (21, 3), already normalized.
    Output: 134 float features.
    """
    features: List[float] = []
    
    # -------------------------------------------------------------
    # Group A: Raw Normalized Coordinates (63 features, indices 0..62)
    # -------------------------------------------------------------
    for i in range(21):
        features.extend([float(landmarks_21x3[i, 0]), float(landmarks_21x3[i, 1]), float(landmarks_21x3[i, 2])])
        
    # -------------------------------------------------------------
    # Group B: Finger Joint Angles (15 features, indices 63..77)
    # -------------------------------------------------------------
    for a_idx, b_idx, c_idx in ANGLE_TRIPLETS:
        angle = _calculate_joint_angle(landmarks_21x3[a_idx], landmarks_21x3[b_idx], landmarks_21x3[c_idx])
        features.append(angle)
        
    # -------------------------------------------------------------
    # Group C: Finger Segment Lengths & Total Lengths (20 features, indices 78..97)
    # -------------------------------------------------------------
    segment_lengths = []
    for p1_idx, p2_idx in FINGER_SEGMENTS:
        seg_len = _euclidean_dist(landmarks_21x3[p1_idx], landmarks_21x3[p2_idx])
        segment_lengths.append(seg_len)
        features.append(seg_len)
        
    # Total finger lengths (5 features: sum of 3 segments per finger)
    for f_idx in range(5):
        total_len = segment_lengths[f_idx * 3] + segment_lengths[f_idx * 3 + 1] + segment_lengths[f_idx * 3 + 2]
        features.append(total_len)
        
    # -------------------------------------------------------------
    # Group D: Fingertip Distances (15 features, indices 98..112)
    # -------------------------------------------------------------
    # Pairwise tip distances (10 features)
    for tip_a, tip_b in FINGERTIP_PAIRS:
        features.append(_euclidean_dist(landmarks_21x3[tip_a], landmarks_21x3[tip_b]))
        
    # Tip to wrist distances (5 features)
    p_wrist = landmarks_21x3[WRIST]
    for tip_idx in FINGERTIPS:
        features.append(_euclidean_dist(p_wrist, landmarks_21x3[tip_idx]))
        
    # -------------------------------------------------------------
    # Group E: Finger Spread & MCP Angles (7 features, indices 113..119)
    # -------------------------------------------------------------
    # Adjacent MCP angles subtended at wrist (4 features)
    features.append(_calculate_joint_angle(landmarks_21x3[THUMB_MCP], p_wrist, landmarks_21x3[INDEX_MCP]))
    features.append(_calculate_joint_angle(landmarks_21x3[INDEX_MCP], p_wrist, landmarks_21x3[MIDDLE_MCP]))
    features.append(_calculate_joint_angle(landmarks_21x3[MIDDLE_MCP], p_wrist, landmarks_21x3[RING_MCP]))
    features.append(_calculate_joint_angle(landmarks_21x3[RING_MCP], p_wrist, landmarks_21x3[PINKY_MCP]))
    
    # Adjacent fingertip angles subtended at wrist (3 features)
    features.append(_calculate_joint_angle(landmarks_21x3[THUMB_TIP], p_wrist, landmarks_21x3[INDEX_TIP]))
    features.append(_calculate_joint_angle(landmarks_21x3[INDEX_TIP], p_wrist, landmarks_21x3[MIDDLE_TIP]))
    features.append(_calculate_joint_angle(landmarks_21x3[MIDDLE_TIP], p_wrist, landmarks_21x3[RING_TIP]))
    
    # -------------------------------------------------------------
    # Group F: Palm Geometry (8 features, indices 120..127)
    # -------------------------------------------------------------
    # Wrist to MCP distances (4 features)
    for mcp_idx in MCPS:
        features.append(_euclidean_dist(p_wrist, landmarks_21x3[mcp_idx]))
        
    # Palm width & inter-MCP distances (4 features)
    features.append(_euclidean_dist(landmarks_21x3[INDEX_MCP], landmarks_21x3[PINKY_MCP])) # Palm span
    features.append(_euclidean_dist(landmarks_21x3[INDEX_MCP], landmarks_21x3[MIDDLE_MCP]))
    features.append(_euclidean_dist(landmarks_21x3[MIDDLE_MCP], landmarks_21x3[RING_MCP]))
    features.append(_euclidean_dist(landmarks_21x3[RING_MCP], landmarks_21x3[PINKY_MCP]))
    
    # -------------------------------------------------------------
    # Group G: Hand Orientation & Palm Normal (6 features, indices 128..133)
    # -------------------------------------------------------------
    v1 = landmarks_21x3[INDEX_MCP] - p_wrist
    v2 = landmarks_21x3[PINKY_MCP] - p_wrist
    
    norm_x = v1[1] * v2[2] - v1[2] * v2[1]
    norm_y = v1[2] * v2[0] - v1[0] * v2[2]
    norm_z = v1[0] * v2[1] - v1[1] * v2[0]
    
    n_len = math.sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z)
    if n_len < EPSILON:
        features.extend([0.0, 0.0, 1.0])
    else:
        features.extend([float(norm_x / n_len), float(norm_y / n_len), float(norm_z / n_len)])
        
    v_dir = landmarks_21x3[MIDDLE_MCP] - p_wrist
    d_len = math.sqrt(v_dir[0] * v_dir[0] + v_dir[1] * v_dir[1] + v_dir[2] * v_dir[2])
    if d_len < EPSILON:
        features.extend([0.0, 1.0, 0.0])
    else:
        features.extend([float(v_dir[0] / d_len), float(v_dir[1] / d_len), float(v_dir[2] / d_len)])
        
    return features


def extract_sequence_enhanced_features(
    sequence_30x63_or_30x21x3: np.ndarray,
    schema: str = "v5_single_hand_all"
) -> np.ndarray:
    """
    Extracts multi-group geometric and motion features across a 30-frame sequence.
    """
    if schema not in FEATURE_SCHEMAS:
        raise ValueError(f"Unknown feature schema '{schema}'. Allowed: {list(FEATURE_SCHEMAS.keys())}")
        
    seq = np.asarray(sequence_30x63_or_30x21x3, dtype=np.float64)
    if seq.shape == (30, 63):
        seq_21x3 = seq.reshape(30, 21, 3)
    elif seq.shape == (30, 21, 3):
        seq_21x3 = seq
    else:
        raise ValueError(f"Invalid sequence shape {seq.shape}. Expected (30, 63) or (30, 21, 3).")
        
    T = 30
    
    # 1. Extract static geometric features (Groups A through G: 134 features per frame)
    geom_features = [extract_geometric_features_frame(seq_21x3[t]) for t in range(T)]
    
    if schema == "v1_single_hand_63":
        return np.array([gf[:63] for gf in geom_features], dtype=np.float32)
        
    if schema == "v2_single_hand_geometry":
        return np.array(geom_features, dtype=np.float32)
        
    # 2. Velocities (Group H: 63 features)
    velocities = np.zeros((T, 21, 3), dtype=np.float64)
    for t in range(1, T):
        velocities[t] = seq_21x3[t] - seq_21x3[t - 1]
        
    # 3. Accelerations (Group I: 63 features)
    accelerations = np.zeros((T, 21, 3), dtype=np.float64)
    for t in range(2, T):
        accelerations[t] = velocities[t] - velocities[t - 1]
        
    # 4. Global Motion & Temporal Summary (Groups J & K)
    global_motion = np.zeros((T, 6), dtype=np.float64)
    cumulative_displacement = 0.0
    temporal_summary = np.zeros((T, 5), dtype=np.float64)
    
    for t in range(T):
        v_wrist = velocities[t, WRIST]
        wrist_speed = float(np.sqrt(np.dot(v_wrist, v_wrist)))
        
        if t == 0:
            palm_speed = 0.0
        else:
            palm_curr = (seq_21x3[t, WRIST] + seq_21x3[t, INDEX_MCP] + seq_21x3[t, MIDDLE_MCP] + seq_21x3[t, PINKY_MCP]) * 0.25
            palm_prev = (seq_21x3[t - 1, WRIST] + seq_21x3[t - 1, INDEX_MCP] + seq_21x3[t - 1, MIDDLE_MCP] + seq_21x3[t - 1, PINKY_MCP]) * 0.25
            v_palm = palm_curr - palm_prev
            palm_speed = float(np.sqrt(np.dot(v_palm, v_palm)))
            
        lm_speeds = [math.sqrt(velocities[t, i, 0]**2 + velocities[t, i, 1]**2 + velocities[t, i, 2]**2) for i in range(21)]
        mean_speed = float(np.mean(lm_speeds))
        max_speed = float(np.max(lm_speeds))
        
        global_motion[t, 0] = v_wrist[0]
        global_motion[t, 1] = v_wrist[1]
        global_motion[t, 2] = v_wrist[2]
        global_motion[t, 3] = wrist_speed
        global_motion[t, 4] = palm_speed
        global_motion[t, 5] = mean_speed
        
        cumulative_displacement += palm_speed
        
        if t <= 1:
            max_accel = 0.0
            mean_accel = 0.0
        else:
            lm_accels = [math.sqrt(accelerations[t, i, 0]**2 + accelerations[t, i, 1]**2 + accelerations[t, i, 2]**2) for i in range(21)]
            max_accel = float(np.max(lm_accels))
            mean_accel = float(np.mean(lm_accels))
            
        temporal_summary[t, 0] = cumulative_displacement
        temporal_summary[t, 1] = max_speed
        temporal_summary[t, 2] = mean_speed
        temporal_summary[t, 3] = max_accel
        temporal_summary[t, 4] = mean_accel
        
    out_matrix = np.zeros((T, FEATURE_SCHEMAS[schema]), dtype=np.float32)
    
    for t in range(T):
        row = list(geom_features[t])
        
        if schema in ["v3_single_hand_geom_vel", "v4_single_hand_geom_vel_acc", "v5_single_hand_all"]:
            row.extend(velocities[t].flatten().tolist())
            row.extend(global_motion[t].tolist())
            
        if schema in ["v4_single_hand_geom_vel_acc", "v5_single_hand_all"]:
            row.extend(accelerations[t].flatten().tolist())
            
        if schema == "v5_single_hand_all":
            row.extend(temporal_summary[t].tolist())
            
        out_matrix[t] = np.array(row, dtype=np.float32)
        
    return out_matrix


def transform_dataset_features(
    X_raw_30x63: np.ndarray,
    schema: str = "v5_single_hand_all"
) -> np.ndarray:
    """
    Batch transforms an entire dataset matrix (N, 30, 63) to (N, 30, feature_dim).
    """
    N = X_raw_30x63.shape[0]
    dim = FEATURE_SCHEMAS[schema]
    X_enhanced = np.zeros((N, 30, dim), dtype=np.float32)
    
    for i in range(N):
        X_enhanced[i] = extract_sequence_enhanced_features(X_raw_30x63[i], schema=schema)
        
    return X_enhanced
