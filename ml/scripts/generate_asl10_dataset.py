"""
Sign Bridge - 10-Class ASL Dataset Generator (Refined Kinematics for Distinct Chest Motions)
"""

import os
import json
import numpy as np

CLASSES_10 = [
    "HELLO", "NO", "THANK_YOU", "YES", "HELP",
    "PLEASE", "SORRY", "GOODBYE", "NAME", "WATER"
]

PERFORMERS_10 = [
    "perf_01_native", "perf_02_native", "perf_03_interpreter", "perf_04_interpreter",
    "perf_05_coda", "perf_06_coda", "perf_07_learner", "perf_08_learner",
    "perf_09_holdout", "perf_10_holdout"
]

def generate_base_hand(open_ratio=1.0, pinch_ratio=0.0, tilt_angle=0.0, w_shape=False, h_shape=False, thumbs_up=False, a_fist=False):
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.0, 0.0, 0.0]  # Wrist
    
    # Thumb (1-4)
    if thumbs_up:
        lm[1] = [-0.1, 0.3, 0.1]
        lm[2] = [-0.15, 0.6, 0.2]
        lm[3] = [-0.2, 0.9, 0.3]
        lm[4] = [-0.25, 1.2, 0.4]
    elif a_fist:
        lm[1] = [0.1, 0.3, 0.0]
        lm[2] = [0.15, 0.5, 0.0]
        lm[3] = [0.2, 0.7, 0.0]
        lm[4] = [0.25, 0.9, 0.0]  # Thumb upright alongside index
    else:
        lm[1] = [0.2, 0.2, 0.0]
        lm[2] = [0.35, 0.35, 0.0]
        lm[3] = [0.45, 0.45, 0.0]
        lm[4] = [0.5 * (1.0 - 0.5 * pinch_ratio), 0.5 * (1.0 - 0.5 * pinch_ratio), 0.0]
        
    # Index (5-8)
    idx_ext = 1.0 if (w_shape or h_shape) else (0.05 if a_fist else open_ratio)
    lm[5] = [0.15, 0.8, 0.0]
    lm[6] = [0.18, 1.1 * idx_ext + 0.3 * (1.0 - idx_ext), 0.0]
    lm[7] = [0.20, 1.3 * idx_ext + 0.2 * (1.0 - idx_ext), 0.0]
    lm[8] = [0.22 * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 1.5 * idx_ext + 0.1 * (1.0 - idx_ext) * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 0.0]
    
    # Middle (9-12) - Scale Reference
    mid_ext = 1.0 if (w_shape or h_shape) else (0.05 if a_fist else open_ratio)
    lm[9] = [0.0, 1.0, 0.0]
    lm[10] = [0.0, 1.35 * mid_ext + 0.3 * (1.0 - mid_ext), 0.0]
    lm[11] = [0.0, 1.60 * mid_ext + 0.2 * (1.0 - mid_ext), 0.0]
    lm[12] = [0.0 * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 1.80 * mid_ext + 0.1 * (1.0 - mid_ext) * (1.0 - pinch_ratio) + 0.45 * pinch_ratio, 0.0]
    
    # Ring (13-16)
    ring_ext = 1.0 if w_shape else (0.05 if (h_shape or a_fist) else open_ratio)
    lm[13] = [-0.15, 0.85, 0.0]
    lm[14] = [-0.18, 1.15 * ring_ext + 0.3 * (1.0 - ring_ext), 0.0]
    lm[15] = [-0.20, 1.40 * ring_ext + 0.2 * (1.0 - ring_ext), 0.0]
    lm[16] = [-0.22, 1.60 * ring_ext + 0.1 * (1.0 - ring_ext), 0.0]
    
    # Pinky (17-20)
    pinky_ext = 0.05 if (w_shape or h_shape or a_fist) else open_ratio
    lm[17] = [-0.3, 0.7, 0.0]
    lm[18] = [-0.35, 0.95 * pinky_ext + 0.3 * (1.0 - pinky_ext), 0.0]
    lm[19] = [-0.38, 1.15 * pinky_ext + 0.2 * (1.0 - pinky_ext), 0.0]
    lm[20] = [-0.4, 1.35 * pinky_ext + 0.1 * (1.0 - pinky_ext), 0.0]
    
    if tilt_angle != 0.0:
        c, s = np.cos(tilt_angle), np.sin(tilt_angle)
        rx = lm[:, 0] * c - lm[:, 1] * s
        ry = lm[:, 0] * s + lm[:, 1] * c
        lm[:, 0] = rx
        lm[:, 1] = ry
        
    return lm

def normalize_sequence(raw_seq):
    out = np.zeros((30, 63), dtype=np.float32)
    for t in range(30):
        frame = raw_seq[t]
        wrist = frame[0]
        mcp = frame[9]
        scale = np.sqrt(np.sum((mcp - wrist)**2))
        if scale < 1e-6:
            scale = 1.0
        norm_frame = (frame - wrist) / scale
        out[t] = norm_frame.flatten()
    return out

def generate_asl10_dataset(samples_per_class=40, seed=42):
    rng = np.random.RandomState(seed)
    X = []
    y = []
    sample_meta = []
    
    for c_idx, label in enumerate(CLASSES_10):
        for s in range(samples_per_class):
            performer = PERFORMERS_10[s % len(PERFORMERS_10)]
            session = f"sess_{(s // len(PERFORMERS_10)) + 1:02d}"
            speed = rng.uniform(0.9, 1.1)
            phase = rng.uniform(0.0, 0.15)
            noise_scale = rng.uniform(0.003, 0.006)
            
            raw_seq = np.zeros((30, 21, 3), dtype=np.float32)
            
            for t in range(30):
                t_norm = (t / 29.0) * speed + phase
                
                if c_idx == 0:  # HELLO: Lateral waving tilt + x-sway
                    tilt = np.sin(t_norm * 6 * np.pi) * 0.45
                    hand = generate_base_hand(open_ratio=1.0, tilt_angle=tilt)
                    hand[:, 0] += np.sin(t_norm * 6 * np.pi) * 0.35
                    
                elif c_idx == 1:  # NO: Index + middle snap to thumb
                    snap = np.sin(np.clip(t_norm * 2.0, 0.0, 2.0) * np.pi)**2
                    hand = generate_base_hand(open_ratio=1.0 - 0.9 * snap, pinch_ratio=snap)
                    
                elif c_idx == 2:  # THANK_YOU: Forward extension along depth (z-axis)
                    fwd = np.clip(t_norm, 0.0, 1.0) * 1.8
                    hand = generate_base_hand(open_ratio=1.0)
                    hand[:, 2] += fwd
                    hand[:, 1] -= fwd * 0.2
                    
                elif c_idx == 3:  # YES: S-fist vertical nodding
                    nod = np.sin(t_norm * 4 * np.pi) * 0.45
                    hand = generate_base_hand(open_ratio=0.05)
                    hand[:, 1] += nod
                    
                elif c_idx == 4:  # HELP: Thumbs-up upward lift
                    lift = np.clip(t_norm, 0.0, 1.0) * 1.4
                    hand = generate_base_hand(thumbs_up=True, open_ratio=0.05)
                    hand[:, 1] += lift
                    
                elif c_idx == 5:  # PLEASE: Flat open palm circular rubbing (planar rotation + distinct z-contact)
                    theta = t_norm * 4 * np.pi
                    cx = np.cos(theta) * 0.4
                    cy = np.sin(theta) * 0.4
                    hand = generate_base_hand(open_ratio=1.0, tilt_angle=0.15 * np.cos(theta))
                    hand[:, 0] += cx
                    hand[:, 1] += cy
                    hand[:, 2] -= 0.2  # Hand pressed close to chest
                    
                elif c_idx == 6:  # SORRY: A-fist circular chest rubbing
                    theta = t_norm * 4 * np.pi
                    cx = np.cos(theta) * 0.35
                    cy = np.sin(theta) * 0.35
                    hand = generate_base_hand(a_fist=True)
                    hand[:, 0] += cx
                    hand[:, 1] += cy
                    hand[:, 2] -= 0.25
                    
                elif c_idx == 7:  # GOODBYE: Open palm finger flexing/waving
                    flex = 0.5 + 0.5 * np.sin(t_norm * 5 * np.pi)
                    hand = generate_base_hand(open_ratio=flex)
                    hand[:, 1] += 0.15 * np.sin(t_norm * 5 * np.pi)
                    
                elif c_idx == 8:  # NAME: H-hand tapping
                    tap = np.abs(np.sin(t_norm * 6 * np.pi)) * 0.35
                    hand = generate_base_hand(h_shape=True)
                    hand[:, 1] -= tap
                    
                elif c_idx == 9:  # WATER: W-hand tapping at chin
                    tap = np.abs(np.sin(t_norm * 5 * np.pi)) * 0.3
                    hand = generate_base_hand(w_shape=True)
                    hand[:, 0] += tap * 0.5
                
                hand += rng.normal(0, noise_scale, size=hand.shape)
                raw_seq[t] = hand
                
            norm_seq = normalize_sequence(raw_seq)
            X.append(norm_seq)
            y.append(c_idx)
            sample_meta.append({
                "sample_id": f"asl10_{c_idx:02d}_{s:03d}",
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
    X, y, meta = generate_asl10_dataset(samples_per_class=40)
    print(f"Generated Refined ASL-10 Dataset: X={X.shape}, y={y.shape}, meta={len(meta)}")
    out_dir = "ml/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_asl10_v1.npz")
    np.savez_compressed(out_path, X=X, y=y, sample_meta=meta)
    print(f"Saved dataset to {out_path}")
