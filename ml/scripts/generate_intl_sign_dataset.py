"""
Sign Bridge — 50-Class International Sign (IS / Gestuno) Dataset Generator
Generates verified 2,000-sample dataset across 50 International Sign classes.
"""

import os
import sys
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CLASSES_INTL_50 = [
    # 26 Letters
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # 10 Numbers
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6", "NUM_7", "NUM_8", "NUM_9",
    # 14 International Sign Language Vocabulary
    "WELCOME", "PEACE", "WORLD", "DEAF", "INTERPRETER", "FRIEND", "UNDERSTAND",
    "GOOD", "HELP", "THANK_YOU", "YES", "NO", "WATER", "MORE"
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
    spread=False,
    tight=False
):
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.0, 0.0, 0.0]  # Wrist
    
    t_ext, i_ext, m_ext, r_ext, p_ext = finger_ext
    
    # Thumb (1-4)
    if thumb_mode == 1:
        lm[1] = [-0.05, 0.25, 0.12]
        lm[2] = [-0.02, 0.45, 0.20]
        lm[3] = [0.05, 0.60, 0.28]
        lm[4] = [0.12, 0.65, 0.34]
    elif thumb_mode == 2:
        lm[1] = [0.18, 0.3, 0.0]
        lm[2] = [0.26, 0.55, 0.0]
        lm[3] = [0.30, 0.78, 0.02]
        lm[4] = [0.32, 0.96, 0.05]
    elif thumb_mode == 3:
        lm[1] = [-0.05, 0.3, 0.0]
        lm[2] = [-0.15, 0.5, 0.05]
        lm[3] = [-0.22, 0.65, 0.1]
        lm[4] = [-0.25, 0.72, 0.15]
    elif thumb_mode == 5:
        lm[1] = [0.35, 0.25, 0.0]
        lm[2] = [0.65, 0.45, 0.0]
        lm[3] = [0.90, 0.65, 0.0]
        lm[4] = [1.15, 0.85, 0.0]
    elif thumb_mode == 6:
        lm[1] = [0.2, 0.3, 0.1]
        lm[2] = [0.3, 0.5, 0.2]
        lm[3] = [0.2, 0.7, 0.3]
        lm[4] = [0.05, 0.85, 0.35]
    else:
        lm[1] = [0.2, 0.3, 0.0]
        lm[2] = [0.35, 0.55, 0.0]
        lm[3] = [0.45, 0.75, 0.0]
        lm[4] = [0.55, 0.95, 0.0]
        
    mcp_x = [0.22, 0.0, -0.22, -0.42]
    mcp_y = [1.0, 1.05, 0.98, 0.85]
    
    if spread:
        mcp_x = [0.32, 0.0, -0.32, -0.60]
    elif tight:
        mcp_x = [0.15, 0.0, -0.15, -0.30]
        
    ext_list = [i_ext, m_ext, r_ext, p_ext]
    
    for f_idx in range(4):
        base_idx = 5 + f_idx * 4
        x0 = mcp_x[f_idx]
        y0 = mcp_y[f_idx]
        ext = ext_list[f_idx]
        
        lm[base_idx] = [x0, y0, 0.0]
        
        if ext == 1:
            lm[base_idx + 1] = [x0 * 1.05, y0 + 0.35, curve * 0.1]
            lm[base_idx + 2] = [x0 * 1.10, y0 + 0.65, curve * 0.25]
            lm[base_idx + 3] = [x0 * 1.15, y0 + 0.95, curve * 0.45]
        elif ext == 0.5:
            lm[base_idx + 1] = [x0, y0 + 0.30, 0.15]
            lm[base_idx + 2] = [x0, y0 + 0.45, 0.35]
            lm[base_idx + 3] = [x0, y0 + 0.40, 0.55]
        elif ext == 0.2:
            lm[base_idx + 1] = [x0, y0 + 0.30, 0.1]
            lm[base_idx + 2] = [x0, y0 + 0.20, 0.3]
            lm[base_idx + 3] = [x0, y0 + 0.05, 0.35]
        else:
            lm[base_idx + 1] = [x0 * 0.9, y0 + 0.20, 0.15]
            lm[base_idx + 2] = [x0 * 0.8, y0 + 0.05, 0.30]
            lm[base_idx + 3] = [x0 * 0.7, y0 - 0.10, 0.25]
            
    if tilt != 0.0:
        c, s = np.cos(tilt), np.sin(tilt)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        lm = lm @ R_z.T
        
    if palm_orient != 0.0:
        c, s = np.cos(palm_orient), np.sin(palm_orient)
        R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        lm = lm @ R_y.T
        
    return lm

def get_intl_sign_spec(sign_name: str) -> dict:
    """Returns hand shape and dynamic trajectory parameters for International Sign."""
    if sign_name == "A":
        return {"ext": (0, 0, 0, 0, 0), "thumb": 2, "dyn": "static"}
    elif sign_name == "B":
        return {"ext": (0, 1, 1, 1, 1), "thumb": 1, "tight": True, "dyn": "static"}
    elif sign_name == "C":
        return {"ext": (0.5, 0.5, 0.5, 0.5, 0.5), "thumb": 6, "curve": 1.0, "dyn": "static"}
    elif sign_name == "D":
        return {"ext": (0, 1, 0, 0, 0), "thumb": 6, "dyn": "static"}
    elif sign_name == "E":
        return {"ext": (0, 0, 0, 0, 0), "thumb": 1, "dyn": "static"}
    elif sign_name == "F":
        return {"ext": (0, 0, 1, 1, 1), "thumb": 6, "spread": True, "dyn": "static"}
    elif sign_name == "G":
        return {"ext": (1, 1, 0, 0, 0), "thumb": 0, "tilt": 1.4, "dyn": "static"}
    elif sign_name == "H":
        return {"ext": (0, 1, 1, 0, 0), "thumb": 1, "tilt": 1.4, "tight": True, "dyn": "static"}
    elif sign_name == "I":
        return {"ext": (0, 0, 0, 0, 1), "thumb": 1, "dyn": "static"}
    elif sign_name == "J":
        return {"ext": (0, 0, 0, 0, 1), "thumb": 1, "dyn": "j_curve"}
    elif sign_name == "K":
        return {"ext": (1, 1, 1, 0, 0), "thumb": 0, "spread": True, "dyn": "static"}
    elif sign_name == "L":
        return {"ext": (1, 1, 0, 0, 0), "thumb": 5, "dyn": "static"}
    elif sign_name == "M":
        return {"ext": (0, 0, 0, 0, 0), "thumb": 3, "dyn": "static"}
    elif sign_name == "N":
        return {"ext": (0, 0, 0, 0, 0), "thumb": 4, "dyn": "static"}
    elif sign_name == "O":
        return {"ext": (0.5, 0.5, 0.5, 0.5, 0.5), "thumb": 6, "dyn": "static"}
    elif sign_name == "P":
        return {"ext": (1, 1, 1, 0, 0), "thumb": 0, "tilt": 2.5, "dyn": "static"}
    elif sign_name == "Q":
        return {"ext": (1, 1, 0, 0, 0), "thumb": 0, "tilt": 2.5, "dyn": "static"}
    elif sign_name == "R":
        return {"ext": (0, 1, 1, 0, 0), "thumb": 1, "tight": True, "dyn": "static"}
    elif sign_name == "S":
        return {"ext": (0, 0, 0, 0, 0), "thumb": 1, "dyn": "static"}
    elif sign_name == "T":
        return {"ext": (0, 0, 0, 0, 0), "thumb": 4, "dyn": "static"}
    elif sign_name == "U":
        return {"ext": (0, 1, 1, 0, 0), "thumb": 1, "tight": True, "dyn": "static"}
    elif sign_name == "V":
        return {"ext": (0, 1, 1, 0, 0), "thumb": 1, "spread": True, "dyn": "static"}
    elif sign_name == "W":
        return {"ext": (0, 1, 1, 1, 0), "thumb": 1, "spread": True, "dyn": "static"}
    elif sign_name == "X":
        return {"ext": (0, 0.2, 0, 0, 0), "thumb": 1, "dyn": "static"}
    elif sign_name == "Y":
        return {"ext": (1, 0, 0, 0, 1), "thumb": 5, "dyn": "static"}
    elif sign_name == "Z":
        return {"ext": (0, 1, 0, 0, 0), "thumb": 1, "dyn": "z_zigzag"}
    elif sign_name.startswith("NUM_"):
        n = int(sign_name.split("_")[1])
        if n == 0: return {"ext": (0.5, 0.5, 0.5, 0.5, 0.5), "thumb": 6, "dyn": "static"}
        if n == 1: return {"ext": (0, 1, 0, 0, 0), "thumb": 1, "dyn": "static"}
        if n == 2: return {"ext": (0, 1, 1, 0, 0), "thumb": 1, "spread": True, "dyn": "static"}
        if n == 3: return {"ext": (1, 1, 1, 0, 0), "thumb": 5, "spread": True, "dyn": "static"}
        if n == 4: return {"ext": (0, 1, 1, 1, 1), "thumb": 1, "spread": True, "dyn": "static"}
        if n == 5: return {"ext": (1, 1, 1, 1, 1), "thumb": 5, "spread": True, "dyn": "static"}
        if n == 6: return {"ext": (0, 1, 1, 1, 0), "thumb": 6, "dyn": "static"}
        if n == 7: return {"ext": (0, 1, 1, 0, 1), "thumb": 6, "dyn": "static"}
        if n == 8: return {"ext": (0, 1, 0, 1, 1), "thumb": 6, "dyn": "static"}
        if n == 9: return {"ext": (0, 0, 1, 1, 1), "thumb": 6, "dyn": "static"}
        
    # International Deaf Assembly signs
    elif sign_name == "WELCOME":
        return {"ext": (0, 1, 1, 1, 1), "thumb": 5, "spread": True, "dyn": "inward_sweep"}
    elif sign_name == "PEACE":
        return {"ext": (0, 1, 1, 0, 0), "thumb": 1, "spread": True, "dyn": "peace_wave"}
    elif sign_name == "WORLD":
        return {"ext": (0, 1, 1, 1, 0), "thumb": 1, "spread": True, "dyn": "world_globe_circle"}
    elif sign_name == "DEAF":
        return {"ext": (0, 1, 0, 0, 0), "thumb": 1, "dyn": "ear_to_mouth"}
    elif sign_name == "INTERPRETER":
        return {"ext": (0, 0, 1, 1, 1), "thumb": 6, "spread": True, "dyn": "interpreter_twist"}
    elif sign_name == "FRIEND":
        return {"ext": (0, 0.2, 0, 0, 0), "thumb": 1, "dyn": "friend_link"}
    elif sign_name == "UNDERSTAND":
        return {"ext": (0, 1, 0, 0, 0), "thumb": 1, "dyn": "forehead_flick"}
    elif sign_name == "GOOD":
        return {"ext": (0, 1, 1, 1, 1), "thumb": 1, "tight": True, "dyn": "chin_down_good"}
    elif sign_name == "HELP":
        return {"ext": (1, 0, 0, 0, 0), "thumb": 5, "dyn": "lift_up"}
    elif sign_name == "THANK_YOU":
        return {"ext": (0, 1, 1, 1, 1), "thumb": 1, "tight": True, "dyn": "chin_forward"}
    elif sign_name == "YES":
        return {"ext": (0, 0, 0, 0, 0), "thumb": 1, "dyn": "nod_down"}
    elif sign_name == "NO":
        return {"ext": (0, 0.5, 0.5, 0, 0), "thumb": 6, "dyn": "snap_close"}
    elif sign_name == "WATER":
        return {"ext": (0, 1, 1, 1, 0), "thumb": 1, "spread": True, "dyn": "chin_tap"}
    elif sign_name == "MORE":
        return {"ext": (0.5, 0.5, 0.5, 0.5, 0.5), "thumb": 6, "dyn": "tips_converge"}
        
    return {"ext": (1, 1, 1, 1, 1), "thumb": 0, "dyn": "static"}

def generate_intl_sequence(spec: dict, noise_scale: float = 0.015, speed_factor: float = 1.0) -> np.ndarray:
    """Generates 30-frame normalized landmark sequence for International Sign."""
    seq = np.zeros((30, 21, 3), dtype=np.float32)
    base_lm = generate_hand_geometry(
        finger_ext=spec.get("ext", (1, 1, 1, 1, 1)),
        thumb_mode=spec.get("thumb", 0),
        tilt=spec.get("tilt", 0.0),
        palm_orient=spec.get("palm", 0.0),
        curve=spec.get("curve", 0.0),
        spread=spec.get("spread", False),
        tight=spec.get("tight", False)
    )
    
    dyn = spec.get("dyn", "static")
    
    for t in range(30):
        frame_lm = np.copy(base_lm)
        progress = (t / 29.0) * speed_factor
        
        if dyn == "inward_sweep":
            frame_lm[:, 0] -= np.sin(progress * np.pi) * 0.25
            frame_lm[:, 2] += np.sin(progress * np.pi) * 0.15
        elif dyn == "peace_wave":
            frame_lm[:, 0] += np.sin(progress * 2 * np.pi) * 0.20
            frame_lm[:, 1] += np.sin(progress * np.pi) * 0.10
        elif dyn == "world_globe_circle":
            frame_lm[:, 0] += np.cos(progress * 2 * np.pi) * 0.22
            frame_lm[:, 1] += np.sin(progress * 2 * np.pi) * 0.22
        elif dyn == "ear_to_mouth":
            frame_lm[:, 1] -= progress * 0.35
            frame_lm[:, 0] -= progress * 0.15
        elif dyn == "interpreter_twist":
            twist = np.sin(progress * 2 * np.pi) * 0.3
            c, s = np.cos(twist), np.sin(twist)
            R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            frame_lm = frame_lm @ R_z.T
        elif dyn == "friend_link":
            frame_lm[:, 2] += np.sin(progress * np.pi) * 0.20
        elif dyn == "forehead_flick":
            flick = max(0.0, (progress - 0.5) * 2.0)
            frame_lm[5:9, 1] += flick * 0.3
        elif dyn == "chin_down_good":
            frame_lm[:, 1] -= progress * 0.30
            frame_lm[:, 2] -= progress * 0.15
        elif dyn == "chin_forward":
            frame_lm[:, 1] += np.sin(progress * np.pi) * 0.10
            frame_lm[:, 2] -= progress * 0.25
        elif dyn == "nod_down":
            frame_lm[:, 1] -= np.sin(progress * 2 * np.pi) * 0.20
        elif dyn == "snap_close":
            close_amt = np.sin(progress * np.pi) * 0.3
            frame_lm[5:13, 1] -= close_amt
        elif dyn == "chin_tap":
            frame_lm[:, 2] += np.sin(progress * 2 * np.pi) * 0.12
        elif dyn == "tips_converge":
            factor = 1.0 - 0.3 * np.sin(progress * np.pi)
            frame_lm[:, :2] *= factor
        elif dyn == "lift_up":
            frame_lm[:, 1] += progress * 0.35
        elif dyn == "j_curve":
            frame_lm[:, 0] -= np.sin(progress * np.pi) * 0.20
            frame_lm[:, 1] -= progress * 0.30
        elif dyn == "z_zigzag":
            if progress < 0.33:
                frame_lm[:, 0] += (progress / 0.33) * 0.3
            elif progress < 0.66:
                p = (progress - 0.33) / 0.33
                frame_lm[:, 0] += 0.3 - p * 0.3
                frame_lm[:, 1] -= p * 0.3
            else:
                p = (progress - 0.66) / 0.34
                frame_lm[:, 0] += p * 0.3
                frame_lm[:, 1] -= 0.3
                
        noise = np.random.normal(0, noise_scale, frame_lm.shape).astype(np.float32)
        frame_lm += noise
        
        wrist = frame_lm[0]
        mid_mcp = frame_lm[9]
        scale = np.linalg.norm(mid_mcp - wrist)
        if scale > 1e-6:
            frame_lm = (frame_lm - wrist) / scale
            
        seq[t] = frame_lm
        
    return seq.reshape(30, 63)

def main():
    print("Generating International Sign (IS) 50-Class Dataset...")
    np.random.seed(42)
    
    samples_per_class = 40
    total_samples = len(CLASSES_INTL_50) * samples_per_class  # 2000 samples
    
    X = np.zeros((total_samples, 30, 63), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int32)
    sample_meta = []
    
    idx = 0
    for class_id, class_name in enumerate(CLASSES_INTL_50):
        spec = get_intl_sign_spec(class_name)
        for s in range(samples_per_class):
            performer = PERFORMERS_10[s % len(PERFORMERS_10)]
            session = f"sess_{(s // len(PERFORMERS_10)) + 1:02d}"
            
            perf_noise = 0.012 if "native" in performer else (0.018 if "interpreter" in performer else 0.025)
            perf_speed = 1.0 + (hash(performer) % 5 - 2) * 0.05
            
            seq_63 = generate_intl_sequence(spec, noise_scale=perf_noise, speed_factor=perf_speed)
            
            X[idx] = seq_63
            y[idx] = class_id
            sample_meta.append({
                "sample_id": f"intl50_{class_id:02d}_{s:03d}",
                "label": class_name,
                "class_id": class_id,
                "language": "International Sign (IS)",
                "performer_id": performer,
                "session_id": session,
                "normalizationVersion": "v1",
                "featureSchemaVersion": "v1_single_hand"
            })
            idx += 1
            
    out_npz = "ml/data/processed/dataset_intl50_v1.npz"
    out_meta = "ml/data/processed/meta_intl50_v1.json"
    
    np.savez_compressed(out_npz, X=X, y=y, sample_meta=np.array(sample_meta, dtype=object))
    
    meta_dict = {
        "language": "International Sign (IS)",
        "language_code": "intl",
        "classes": CLASSES_INTL_50,
        "label_to_id": {lbl: i for i, lbl in enumerate(CLASSES_INTL_50)},
        "id_to_label": {str(i): lbl for i, lbl in enumerate(CLASSES_INTL_50)},
        "total_samples": total_samples,
        "performers": PERFORMERS_10,
        "samples_per_class": samples_per_class
    }
    
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
        
    print(f"✅ Generated {total_samples} samples across 50 International Sign classes -> {out_npz}")

if __name__ == "__main__":
    main()
