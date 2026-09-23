import json
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Dict, Any

@dataclass
class TrainingConfig:
  epochs: int = 100
  batch_size: int = 16
  learning_rate: float = 1e-3
  optimizer: str = "adam"
  sequence_length: int = 30
  feature_count: int = 63
  lstm_units_1: int = 64
  dropout_1: float = 0.3
  lstm_units_2: int = 32
  dropout_2: float = 0.3
  dense_units: int = 32
  dropout_3: float = 0.2
  early_stopping_patience: int = 15
  early_stopping_monitor: str = "val_loss"
  restore_best_weights: bool = True
  random_seed: int = 42
  augmentation_enabled: bool = False
  augmentation_factor: int = 1

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  def save_json(self, path: str):
    with open(path, "w", encoding="utf-8") as f:
      json.dump(self.to_dict(), f, indent=2)

  @classmethod
  def load_json(cls, path: str) -> "TrainingConfig":
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
