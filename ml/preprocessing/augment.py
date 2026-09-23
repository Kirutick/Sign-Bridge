import numpy as np
from typing import Dict, Any, Tuple

def augment_landmark_sequence(
    sequence: np.ndarray,
    translation_std: float = 0.02,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    rotation_angle_degrees: float = 15.0,
    jitter_std: float = 0.005,
    random_state: np.random.RandomState = None
) -> np.ndarray:
  """
  Applies geometry-aware data augmentation to a single (30, 63) landmark sequence.
  Input sequence shape: (sequence_length, 63) where 63 represents 21 (x, y, z) landmarks.
  """
  if random_state is None:
    random_state = np.random.RandomState()

  seq_len, feature_count = sequence.shape
  num_landmarks = feature_count // 3

  # Reshape to (seq_len, 21, 3) for 3D geometric operations
  coords = sequence.reshape(seq_len, num_landmarks, 3).copy()

  # 1. Sequence-consistent 2D Rotation (around Z axis)
  angle_rad = np.radians(random_state.uniform(-rotation_angle_degrees, rotation_angle_degrees))
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
  rot_matrix = np.array([
    [cos_a, -sin_a, 0.0],
    [sin_a,  cos_a, 0.0],
    [0.0,    0.0,   1.0]
  ], dtype=np.float32)

  # Apply rotation to all 30 frames consistently
  coords = np.dot(coords, rot_matrix)

  # 2. Sequence-consistent Uniform Scaling
  scale_factor = random_state.uniform(scale_range[0], scale_range[1])
  coords *= scale_factor

  # 3. Sequence-consistent Translation (shift origin)
  shift_x = random_state.normal(0, translation_std)
  shift_y = random_state.normal(0, translation_std)
  shift_z = random_state.normal(0, translation_std / 2.0)
  coords[:, :, 0] += shift_x
  coords[:, :, 1] += shift_y
  coords[:, :, 2] += shift_z

  # 4. Small per-frame sensor tracking jitter
  if jitter_std > 0:
    jitter = random_state.normal(0, jitter_std, size=coords.shape)
    coords += jitter

  # Reshape back to (seq_len, 63)
  return coords.reshape(seq_len, feature_count).astype(np.float32)

def augment_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    augmentation_factor: int = 1,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Augments training dataset by generating augmented copies.
  Applied ONLY to training split (§10).
  """
  rng = np.random.RandomState(random_seed)
  num_samples = X_train.shape[0]

  aug_X = [X_train]
  aug_y = [y_train]

  for factor in range(augmentation_factor):
    seqs = []
    for i in range(num_samples):
      aug_seq = augment_landmark_sequence(X_train[i], random_state=rng)
      seqs.append(aug_seq)
    
    aug_X.append(np.array(seqs, dtype=np.float32))
    aug_y.append(y_train.copy())

  X_out = np.vstack(aug_X)
  y_out = np.concatenate(aug_y)

  print(f"🪄 Data Augmentation Applied (Factor x{augmentation_factor + 1}):")
  print(f" - Original Training Samples: {num_samples}")
  print(f" - Augmented Training Samples: {X_out.shape[0]}")

  return X_out, y_out
