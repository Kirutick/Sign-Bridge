"""
Sign Bridge — Handedness Normalization Module
=============================================
Ensures that left-hand and right-hand ISL recordings are normalized
consistently so the LSTM classifier does not conflate handedness with sign identity.

Strategy:
  - normalize_to = "right" (default):
    Mirror x-coordinates of left-hand sequences so all sequences appear right-handed.
  - normalize_to = "none":
    No mirroring. Useful when training on explicitly handedness-aware data.

Mirroring formula:
  x_mirrored = -x  (applied to all 21 landmarks, all 30 frames)
  y and z are unchanged.

This preserves the canonical v1 normalization invariant:
  wrist at origin, scale = wrist-to-middle-MCP distance.
Mirroring a normalized sequence produces another valid normalized sequence.
"""

import numpy as np
from typing import List


NORMALIZE_TO_RIGHT = "right"
NORMALIZE_TO_NONE = "none"


def mirror_sequence_x(sequence: np.ndarray) -> np.ndarray:
    """
    Mirror x-coordinates of all landmarks across all 30 frames.
    sequence shape: (30, 63) where features are [x0,y0,z0,x1,y1,z1,...,x20,y20,z20]
    x-indices: 0, 3, 6, 9, ..., 60  (every 3rd starting at 0)
    """
    out = sequence.copy()
    x_indices = list(range(0, 63, 3))
    out[:, x_indices] = -out[:, x_indices]
    return out


def normalize_handedness(
    X: np.ndarray,
    handedness_list: List[str],
    normalize_to: str = NORMALIZE_TO_RIGHT,
) -> np.ndarray:
    """
    Apply handedness normalization to a dataset.

    Args:
        X: numpy array of shape (N, 30, 63)
        handedness_list: list of N handedness strings ("Left", "Right", or "Unknown")
        normalize_to: "right" (mirror left-hand sequences) or "none" (no-op)

    Returns:
        X_normalized: numpy array of shape (N, 30, 63) with left-hand sequences mirrored
    """
    if normalize_to == NORMALIZE_TO_NONE:
        return X.copy()

    if normalize_to != NORMALIZE_TO_RIGHT:
        raise ValueError(f"Unknown normalize_to value: '{normalize_to}'. Expected 'right' or 'none'.")

    X_out = X.copy()
    mirrored_count = 0

    for i, handedness in enumerate(handedness_list):
        if isinstance(handedness, (list, tuple)):
            handedness = handedness[0] if handedness else "Unknown"
        if str(handedness).strip().lower() == "left":
            X_out[i] = mirror_sequence_x(X_out[i])
            mirrored_count += 1

    print(f"[Handedness Normalizer] Mirrored {mirrored_count}/{len(X)} left-hand sequences to right-hand convention.")
    return X_out


def normalize_single_sequence(
    sequence: np.ndarray,
    handedness: str,
    normalize_to: str = NORMALIZE_TO_RIGHT,
) -> np.ndarray:
    """
    Normalize handedness for a single sequence (30, 63).

    Args:
        sequence: numpy array of shape (30, 63)
        handedness: "Left", "Right", or "Unknown"
        normalize_to: "right" or "none"

    Returns:
        Normalized sequence of shape (30, 63)
    """
    if normalize_to == NORMALIZE_TO_NONE:
        return sequence.copy()

    if isinstance(handedness, (list, tuple)):
        handedness = handedness[0] if handedness else "Unknown"

    if str(handedness).strip().lower() == "left" and normalize_to == NORMALIZE_TO_RIGHT:
        return mirror_sequence_x(sequence)

    return sequence.copy()
