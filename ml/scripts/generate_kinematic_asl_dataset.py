"""
Sign Bridge - Kinematic ASL Dataset Generator (Linguistically & Anatomically Distinct Trajectories)
Generates high-fidelity ASL kinematic sequences for HELLO, NO, THANK_YOU, YES.
"""

import os
import json
import numpy as np

def generate_hand_skeleton(open_ratio=1.0, pinch_ratio=0.0, tilt_angle=0.0):
    """
    Generates 21-landmark normalized hand geometry (wrist at 0, middle MCP at 1.0).
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)
    # Wrist (0)
    landmarks[0] = [0.0, 0.0, 0.0]
    
    # Thumb (1-4)
    landmarks[1] = [0.2, 0.2, 0.0]
    landmarks[2] = [0.35, 0.35, 0.0]
    landmarks[3] = [0.45, 0.45, 0.0]
    landmarks[4] = [0.5 * (1.0 - 0.5 * pinch_ratio), 0.5 * (1.0 - 0.5 * pinch_ratio), 0.0]
    
    # Index (5-8)
    landmarks[5] = [0.15, 0.8, 0.0]
    landmarks[6] = [0.18, 1.1 * open_ratio + 0.3 * (1.0 - open_ratio), 0.0]
    landmarks[7] = [0.20, 1.3 * open_ratio + 0.2 * (1.0 - open_ratio), 0.0]
    landmarks[8] = [0.22 * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 1.5 * open_ratio + 0.1 * (1.0 - open_ratio) * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 0.0]
    
    # Middle (9-12) - Key MCP scale reference (index 9) at (0.0, 1.0, 0.0)
    landmarks[9] = [0.0, 1.0, 0.0]
    landmarks[10] = [0.0, 1.35 * open_ratio + 0.3 * (1.0 - open_ratio), 0.0]
    landmarks[11] = [0.0, 1.60 * open_ratio + 0.2 * (1.0 - open_ratio), 0.0]
    landmarks[12] = [0.0 * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 1.80 * open_ratio + 0.1 * (1.0 - open_ratio) * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 0.0]
    
    # Ring (13-16)
    landmarks[13] = [-0.15, 0.85, 0.0]
    landmarks[14] = [-0.18, 1.15 * open_ratio + 0.3 * (1.0 - open_ratio), 0.0]
    landmarks[15] = [-0.20, 1.40 * open_ratio + 0.2 * (1.0 - open_ratio), 0.0]
    landmarks[16] = [-0.22, 1.60 * open_ratio + 0.1 * (1.0 - open_ratio), 0.0]
    
    # Pinky (17-20)
    landmarks[17] = [-0.3, 0.7, 0.0]
    landmarks[18] = [-0.35, 0.95 * open_ratio + 0.3 * (1.0 - open_ratio), 0.0]
    landmarks[19] = [-0.38, 1.15 * open_ratio + 0.2 * (1.0 - open_ratio), 0.0]
    landmarks[20] = [-0.4, 1.35 * open_ratio + 0.1 * (1.0 - open_ratio), 0.0]
    
    # Apply 2D planar rotation for tilt
    if tilt_angle != 0.0:
        c, s = np.cos(tilt_angle), np.sin(tilt_angle)
        rot_x = landmarks[:, 0] * c - landmarks[:, 1] * s
        rot_y = landmarks[:, 0] * s + landmarks[:, 1] * c
        landmarks[:, 0] = rot_x
        landmarks[:, 1] = rot_y
        
    return landmarks

def normalize_sequence(raw_seq):
    out = np.zeros((30, 63), dtype=np.float32)
    for t in range(30):
        frame = raw_seq[t]
        wrist = frame[0]
        mcp = frame[9]
        tx9 = mcp[0] - wrist[0]
        ty9 = mcp[1] - wrist[1]
        tz9 = mcp[2] - wrist[2]
        scale = np.sqrt(tx9**2 + ty9**2 + tz9**2)
        if scale < 1e-6:
            scale = 1.0
        norm_frame = (frame - wrist) / scale
        out[t] = norm_frame.flatten()
    return out

def generate_kinematic_dataset(samples_per_class=25, seed=42):
    rng = np.random.RandomState(seed)
    X = []
    y = []
    sample_meta = []
    
    CLASSES = ["HELLO", "NO", "THANK_YOU", "YES"]
    PERFORMERS = ["perf_01", "perf_02", "perf_03", "perf_04", "perf_05"]
    
    for c_idx, label in enumerate(CLASSES):
        for s in range(samples_per_class):
            performer = PERFORMERS[s % len(PERFORMERS)]
            session = f"sess_{(s // len(PERFORMERS)) + 1:02d}"
            speed_mult = rng.uniform(0.9, 1.1)
            phase_offset = rng.uniform(0.0, 0.2)
            noise_scale = rng.uniform(0.003, 0.008)
            
            raw_seq = np.zeros((30, 21, 3), dtype=np.float32)
            
            for t in range(30):
                t_norm = (t / 29.0) * speed_mult + phase_offset
                
                if c_idx == 0:  # HELLO: Side-to-side waving tilt (oscillating lateral angle)
                    wave_tilt = np.sin(t_norm * 6 * np.pi) * 0.45  # Pronounced lateral wave
                    hand = generate_hand_skeleton(open_ratio=1.0, pinch_ratio=0.0, tilt_angle=wave_tilt)
                    hand[:, 0] += np.sin(t_norm * 6 * np.pi) * 0.3
                    
                elif c_idx == 1:  # NO: Index + Middle snap to thumb
                    snap_t = np.clip(t_norm * 2.0, 0.0, 2.0)
                    pinch = np.sin(snap_t * np.pi)**2
                    open_ratio = 1.0 - 0.9 * pinch
                    hand = generate_hand_skeleton(open_ratio=open_ratio, pinch_ratio=pinch, tilt_angle=0.0)
                    
                elif c_idx == 2:  # THANK_YOU: Flat hand moving straight outwards along depth (z-axis)
                    move_z = np.clip(t_norm, 0.0, 1.0) * 1.5  # Pure forward depth trajectory
                    hand = generate_hand_skeleton(open_ratio=1.0, pinch_ratio=0.0, tilt_angle=0.0)
                    hand[:, 2] += move_z
                    hand[:, 1] -= move_z * 0.2
                    
                elif c_idx == 3:  # YES: S-hand fist nodding up and down
                    nod_y = np.sin(t_norm * 4 * np.pi) * 0.4
                    hand = generate_hand_skeleton(open_ratio=0.05, pinch_ratio=0.0, tilt_angle=0.0)
                    hand[:, 1] += nod_y
                
                hand += rng.normal(0, noise_scale, size=hand.shape)
                raw_seq[t] = hand
            
            norm_seq = normalize_sequence(raw_seq)
            X.append(norm_seq)
            y.append(c_idx)
            sample_meta.append({
                "sample_id": f"sample_{c_idx:02d}_{s:03d}",
                "label": label,
                "class_id": c_idx,
                "performer_id": performer,
                "session_id": session,
                "normalizationVersion": "v1",
                "featureSchemaVersion": "v1_single_hand"
            })
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    return X, y, sample_meta

if __name__ == "__main__":
    X, y, sample_meta = generate_kinematic_dataset()
    print(f"Generated Kinematic ASL dataset: X={X.shape}, y={y.shape}, meta={len(sample_meta)}")
    out_dir = "ml/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, "dataset_kinematic_v1.npz")
    np.savez_compressed(out_npz, X=X, y=y, sample_meta=sample_meta)
    print(f"Saved to {out_npz}")
