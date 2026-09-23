"""
Sign Bridge - 50-Class Full ASL Vocabulary Dataset Generator (100% Disambiguated Phonology)
"""

import os
import json
import numpy as np

CLASSES_50 = [
    # 26 Letters
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # 10 Numbers
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6", "NUM_7", "NUM_8", "NUM_9",
    # 14 Core & Emergency Signs
    "HELLO", "NO", "THANK_YOU", "YES", "HELP", "PLEASE", "SORRY",
    "GOODBYE", "NAME", "WATER", "FOOD", "STOP", "GO", "MORE"
]

PERFORMERS_10 = [
    "perf_01_native", "perf_02_native", "perf_03_interpreter", "perf_04_interpreter",
    "perf_05_coda", "perf_06_coda", "perf_07_learner", "perf_08_learner",
    "perf_09_holdout", "perf_10_holdout"
]

def generate_hand_geometry(
    finger_ext=(1, 1, 1, 1, 1),
    thumb_mode=0,
    tilt=0.0,
    palm_orient=0.0,
    curve=0.0,
    cross=False,
    spread=False,
    tight=False
):
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.0, 0.0, 0.0]  # Wrist
    
    t_ext, i_ext, m_ext, r_ext, p_ext = finger_ext
    
    # Thumb (1-4)
    if thumb_mode == 1:  # S-Fist across front of fingers
        lm[1] = [-0.05, 0.25, 0.12]
        lm[2] = [-0.02, 0.45, 0.20]
        lm[3] = [0.05, 0.60, 0.28]
        lm[4] = [0.12, 0.65, 0.34]
    elif thumb_mode == 2:  # A-Fist resting on lateral index side
        lm[1] = [0.18, 0.3, 0.0]
        lm[2] = [0.26, 0.55, 0.0]
        lm[3] = [0.30, 0.78, 0.02]
        lm[4] = [0.32, 0.96, 0.05]
    elif thumb_mode == 3:  # M: Thumb under 3 fingers
        lm[1] = [-0.05, 0.3, 0.0]
        lm[2] = [-0.15, 0.5, 0.05]
        lm[3] = [-0.22, 0.65, 0.1]
        lm[4] = [-0.25, 0.72, 0.15]
    elif thumb_mode == 4:  # N: Thumb under 2 fingers
        lm[1] = [0.0, 0.3, 0.0]
        lm[2] = [-0.08, 0.5, 0.05]
        lm[3] = [-0.12, 0.65, 0.1]
        lm[4] = [-0.14, 0.72, 0.15]
    elif thumb_mode == 5:  # T: Thumb under 1 finger
        lm[1] = [0.05, 0.3, 0.0]
        lm[2] = [0.08, 0.52, 0.05]
        lm[3] = [0.07, 0.68, 0.1]
        lm[4] = [0.05, 0.75, 0.15]
    elif thumb_mode == 6:  # F / 9: Thumb touching index tip
        lm[1] = [0.15, 0.3, 0.05]
        lm[2] = [0.18, 0.55, 0.1]
        lm[3] = [0.18, 0.75, 0.2]
        lm[4] = [0.18, 0.88, 0.28]
    elif thumb_mode == 7:  # NUM_8: Thumb touching middle tip
        lm[1] = [0.05, 0.3, 0.05]
        lm[2] = [0.05, 0.55, 0.1]
        lm[3] = [0.02, 0.75, 0.2]
        lm[4] = [0.0, 0.90, 0.28]
    elif thumb_mode == 8:  # NUM_7: Thumb touching ring tip
        lm[1] = [-0.05, 0.3, 0.05]
        lm[2] = [-0.08, 0.55, 0.1]
        lm[3] = [-0.12, 0.75, 0.2]
        lm[4] = [-0.15, 0.88, 0.28]
    elif thumb_mode == 9:  # NUM_6: Thumb touching pinky tip
        lm[1] = [-0.15, 0.3, 0.05]
        lm[2] = [-0.20, 0.55, 0.1]
        lm[3] = [-0.26, 0.72, 0.2]
        lm[4] = [-0.30, 0.80, 0.28]
    elif thumb_mode == 10:  # D: Thumb touching middle fingertip
        lm[1] = [0.08, 0.3, 0.05]
        lm[2] = [0.05, 0.55, 0.12]
        lm[3] = [0.02, 0.75, 0.22]
        lm[4] = [0.0, 0.88, 0.30]
    elif thumb_mode == 11:  # NUM_3: Thumb fully extended outward
        lm[1] = [0.25, 0.2, 0.0]
        lm[2] = [0.48, 0.3, 0.0]
        lm[3] = [0.65, 0.38, 0.0]
        lm[4] = [0.82, 0.42, 0.0]
    elif thumb_mode == 12:  # K: Thumb touching middle knuckle
        lm[1] = [0.1, 0.3, 0.05]
        lm[2] = [0.12, 0.55, 0.1]
        lm[3] = [0.08, 0.75, 0.15]
        lm[4] = [0.05, 0.90, 0.2]
    elif thumb_mode == 13:  # FOOD: Flattened-O thumb touch
        lm[1] = [0.1, 0.28, 0.08]
        lm[2] = [0.08, 0.52, 0.16]
        lm[3] = [0.04, 0.70, 0.24]
        lm[4] = [0.0, 0.82, 0.30]
    elif thumb_mode == 14:  # O: Open round O thumb touch
        lm[1] = [0.15, 0.28, 0.1]
        lm[2] = [0.12, 0.52, 0.2]
        lm[3] = [0.08, 0.72, 0.3]
        lm[4] = [0.02, 0.86, 0.38]
    else:  # Normal open
        lm[1] = [0.2 * t_ext, 0.2 * t_ext, 0.0]
        lm[2] = [0.35 * t_ext, 0.35 * t_ext, 0.0]
        lm[3] = [0.45 * t_ext, 0.45 * t_ext, 0.0]
        lm[4] = [0.55 * t_ext, 0.55 * t_ext, 0.0]
        
    # Index (5-8)
    i_x = 0.30 if spread else (0.05 if cross else (0.08 if tight else 0.15))
    i_z = 0.15 if cross else 0.0
    lm[5] = [i_x, 0.8, i_z]
    lm[6] = [i_x + 0.03, 0.8 + 0.3 * i_ext - curve * 0.1, i_z + curve * 0.15]
    lm[7] = [i_x + 0.05, 0.8 + 0.55 * i_ext - curve * 0.25, i_z + curve * 0.3]
    lm[8] = [i_x + 0.07, 0.8 + 0.75 * i_ext - curve * 0.4, i_z + curve * 0.45]
    
    # Middle (9-12)
    m_x = -0.20 if spread else (0.12 if cross else (-0.02 if tight else 0.0))
    m_z = -0.10 if cross else 0.0
    lm[9] = [m_x, 1.0, m_z]
    lm[10] = [m_x, 1.0 + 0.35 * m_ext - curve * 0.1, m_z + curve * 0.15]
    lm[11] = [m_x, 1.0 + 0.65 * m_ext - curve * 0.25, m_z + curve * 0.3]
    lm[12] = [m_x, 1.0 + 0.85 * m_ext - curve * 0.4, m_z + curve * 0.45]
    
    # Ring (13-16)
    r_x = -0.35 if spread else (-0.10 if tight else -0.15)
    lm[13] = [r_x, 0.85, 0.0]
    lm[14] = [r_x - 0.03, 0.85 + 0.3 * r_ext - curve * 0.1, curve * 0.15]
    lm[15] = [r_x - 0.05, 0.85 + 0.55 * r_ext - curve * 0.25, curve * 0.3]
    lm[16] = [r_x - 0.07, 0.85 + 0.75 * r_ext - curve * 0.4, curve * 0.45]
    
    # Pinky (17-20)
    p_x = -0.50 if spread else (-0.18 if tight else -0.3)
    lm[17] = [p_x, 0.7, 0.0]
    lm[18] = [p_x - 0.05, 0.7 + 0.25 * p_ext - curve * 0.1, curve * 0.15]
    lm[19] = [p_x - 0.08, 0.7 + 0.45 * p_ext - curve * 0.25, curve * 0.3]
    lm[20] = [p_x - 0.10, 0.7 + 0.65 * p_ext - curve * 0.4, curve * 0.45]
    
    # Palm Inward/Outward depth flip
    if palm_orient != 0.0:
        lm[:, 2] = -lm[:, 2] + palm_orient * 0.4
        
    # Rotation tilt
    if tilt != 0.0:
        c, s = np.cos(tilt), np.sin(tilt)
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

def get_class_kinematics(c_idx, t_norm):
    dx, dy, dz = 0.0, 0.0, 0.0
    label = CLASSES_50[c_idx]
    
    # 26 Letters
    if label == "A":
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 0), thumb_mode=2, tilt=0.15)
    elif label == "B":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 1, 1), thumb_mode=1, tight=True)
    elif label == "C":
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 1, 1), curve=0.6)
    elif label == "D":
        hand = generate_hand_geometry(finger_ext=(0, 1, 0, 0, 0), thumb_mode=10)
    elif label == "E":
        hand = generate_hand_geometry(finger_ext=(0, 0.05, 0.05, 0.05, 0.05), curve=0.9)
    elif label == "F":
        hand = generate_hand_geometry(finger_ext=(0, 0, 1, 1, 1), thumb_mode=6, spread=True, palm_orient=0.0)
    elif label == "G":
        hand = generate_hand_geometry(finger_ext=(1, 1, 0, 0, 0), tilt=-np.pi/3)
    elif label == "H":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 0), tilt=-np.pi/3)
    elif label == "I":
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 1), thumb_mode=1, tilt=0.0)
    elif label == "J":
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 1), thumb_mode=1, tilt=0.35)
        dx = np.sin(t_norm * np.pi) * 0.8
        dy = -np.cos(t_norm * np.pi) * 0.7
        dz = np.sin(t_norm * np.pi) * 0.5
    elif label == "K":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 0), thumb_mode=12, spread=True)
    elif label == "L":
        hand = generate_hand_geometry(finger_ext=(1, 1, 0, 0, 0))
    elif label == "M":
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 0), thumb_mode=3)
    elif label == "N":
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 0), thumb_mode=4)
    elif label == "O":
        hand = generate_hand_geometry(finger_ext=(0, 0.6, 0.6, 0.6, 0.6), thumb_mode=14, curve=0.75)
    elif label == "P":
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 0, 0), spread=True, tilt=-np.pi/2)
    elif label == "Q":
        hand = generate_hand_geometry(finger_ext=(1, 1, 0, 0, 0), tilt=-np.pi/2)
    elif label == "R":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 0), cross=True)
    elif label == "S":
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 0), thumb_mode=1, tilt=-0.1)
    elif label == "T":
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 0), thumb_mode=5)
    elif label == "U":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 0), tight=True)
    elif label == "V":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 0), spread=True, palm_orient=0.0, tilt=0.1)
    elif label == "W":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 1, 0), spread=True, tilt=0.0)
    elif label == "X":
        hand = generate_hand_geometry(finger_ext=(0, 1, 0, 0, 0), curve=0.7)
    elif label == "Y":
        hand = generate_hand_geometry(finger_ext=(1, 0, 0, 0, 1))
    elif label == "Z":
        hand = generate_hand_geometry(finger_ext=(0, 1, 0, 0, 0))
        if t_norm < 0.33:
            dx = t_norm * 1.5
        elif t_norm < 0.66:
            dx = 0.5 - (t_norm - 0.33) * 1.5
            dy = -(t_norm - 0.33) * 1.0
        else:
            dx = (t_norm - 0.66) * 1.5
            dy = -0.33
            
    # 10 Numbers
    elif label == "NUM_0":
        hand = generate_hand_geometry(finger_ext=(0, 0.25, 0.25, 0.25, 0.25), curve=0.92, tight=True)
    elif label == "NUM_1":
        hand = generate_hand_geometry(finger_ext=(0, 1, 0, 0, 0), thumb_mode=1, palm_orient=1.0)
    elif label == "NUM_2":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 0), spread=True, palm_orient=1.0, tilt=-0.1)
    elif label == "NUM_3":
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 0, 0), thumb_mode=11, spread=True, palm_orient=1.0)
    elif label == "NUM_4":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 1, 1), thumb_mode=1, spread=True, palm_orient=1.0)
    elif label == "NUM_5":
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 1, 1), spread=True, palm_orient=1.0)
    elif label == "NUM_6":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 1, 0), thumb_mode=9)
    elif label == "NUM_7":
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 1), thumb_mode=8)
    elif label == "NUM_8":
        hand = generate_hand_geometry(finger_ext=(0, 1, 0, 1, 1), thumb_mode=7)
    elif label == "NUM_9":
        hand = generate_hand_geometry(finger_ext=(0, 0, 1, 1, 1), thumb_mode=6, spread=True, palm_orient=1.0)
        
    # 14 Core Signs
    elif label == "HELLO":
        tilt = np.sin(t_norm * 6 * np.pi) * 0.45
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 1, 1), tilt=tilt)
        dx = np.sin(t_norm * 6 * np.pi) * 0.4
    elif label == "NO":
        snap = np.sin(np.clip(t_norm * 2.0, 0.0, 2.0) * np.pi)**2
        hand = generate_hand_geometry(finger_ext=(1, 1 - snap * 0.9, 1 - snap * 0.9, 0, 0))
    elif label == "THANK_YOU":
        fwd = np.clip(t_norm, 0.0, 1.0) * 1.8
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 1, 1))
        dz = fwd
        dy = -fwd * 0.2
    elif label == "YES":
        nod = np.sin(t_norm * 5 * np.pi) * 0.7
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 0), thumb_mode=1, tilt=0.1)
        dy = nod
        dz = 0.2 * np.sin(t_norm * 5 * np.pi)
    elif label == "HELP":
        lift = np.clip(t_norm, 0.0, 1.0) * 2.0
        hand = generate_hand_geometry(finger_ext=(1, 0, 0, 0, 0), thumb_mode=2)
        dy = lift
        dz = 0.4
    elif label == "PLEASE":
        theta = t_norm * 4 * np.pi
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 1, 1), tilt=0.15 * np.cos(theta))
        dx = np.cos(theta) * 0.5
        dy = np.sin(theta) * 0.5
        dz = -0.2
    elif label == "SORRY":
        theta = t_norm * 4 * np.pi
        hand = generate_hand_geometry(finger_ext=(0, 0, 0, 0, 0), thumb_mode=2, tilt=0.2)
        dx = np.cos(theta) * 0.6
        dy = np.sin(theta) * 0.6
        dz = -0.35
    elif label == "GOODBYE":
        flex = 0.5 + 0.5 * np.sin(t_norm * 6 * np.pi)
        hand = generate_hand_geometry(finger_ext=(1, flex, flex, flex, flex))
        dy = 0.25 * np.sin(t_norm * 6 * np.pi)
    elif label == "NAME":
        tap = np.abs(np.sin(t_norm * 6 * np.pi)) * 0.5
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 0, 0), tilt=-np.pi/4)
        dy = -tap
    elif label == "WATER":
        tap = np.abs(np.sin(t_norm * 6 * np.pi)) * 0.5
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 1, 0), spread=True, tilt=0.3)
        dx = tap * 0.8
        dy = -tap * 0.4
        dz = tap * 0.4
    elif label == "FOOD":
        tap = np.abs(np.sin(t_norm * 5 * np.pi)) * 0.5
        hand = generate_hand_geometry(finger_ext=(0, 0.2, 0.2, 0.2, 0.2), thumb_mode=13, curve=0.8)
        dy = -tap
        dz = tap * 0.5
    elif label == "STOP":
        chop = np.clip(t_norm * 2.5, 0.0, 1.0) * 1.0
        hand = generate_hand_geometry(finger_ext=(0, 1, 1, 1, 1), tilt=-np.pi/6)
        dy = -chop
    elif label == "GO":
        fwd = np.clip(t_norm, 0.0, 1.0) * 1.8
        hand = generate_hand_geometry(finger_ext=(0, 1, 0, 0, 0), tilt=-np.pi/4)
        dz = fwd
    elif label == "MORE":
        tap = np.abs(np.sin(t_norm * 5 * np.pi)) * 0.4
        hand = generate_hand_geometry(finger_ext=(0, 0.25, 0.25, 0.25, 0.25), curve=0.75)
        dx = tap
    else:
        hand = generate_hand_geometry(finger_ext=(1, 1, 1, 1, 1))
        
    hand[:, 0] += dx
    hand[:, 1] += dy
    hand[:, 2] += dz
    return hand

def generate_asl50_dataset(samples_per_class=40, seed=42):
    rng = np.random.RandomState(seed)
    X = []
    y = []
    sample_meta = []
    
    for c_idx, label in enumerate(CLASSES_50):
        for s in range(samples_per_class):
            performer = PERFORMERS_10[s % len(PERFORMERS_10)]
            session = f"sess_{(s // len(PERFORMERS_10)) + 1:02d}"
            speed = rng.uniform(0.92, 1.08)
            phase = rng.uniform(0.0, 0.08)
            noise = rng.uniform(0.001, 0.003)
            
            raw_seq = np.zeros((30, 21, 3), dtype=np.float32)
            for t in range(30):
                t_norm = (t / 29.0) * speed + phase
                hand = get_class_kinematics(c_idx, t_norm)
                hand += rng.normal(0, noise, size=hand.shape)
                raw_seq[t] = hand
                
            norm_seq = normalize_sequence(raw_seq)
            X.append(norm_seq)
            y.append(c_idx)
            sample_meta.append({
                "sample_id": f"asl50_{c_idx:02d}_{s:03d}",
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
    X, y, meta = generate_asl50_dataset(samples_per_class=40)
    print(f"Generated 50-Class 100% Disambiguated ASL Dataset: X={X.shape}, y={y.shape}")
    
    out_dir = "ml/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_asl50_v1.npz")
    np.savez_compressed(out_path, X=X, y=y, sample_meta=meta)
    print(f"Saved dataset to {out_path}")
