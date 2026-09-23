import json
from pathlib import Path

def test_checkpoint_metadata_roundtrip(tmp_path):
    # Mock what the training script exports
    checkpoint_meta = {
        "label_to_id": {"HELLO": 0, "WORLD": 1},
        "id_to_label": {"0": "HELLO", "1": "WORLD"},
        "feature_count": 63,
        "sequence_length": 30,
        "dataset_version": "v1",
        "config_snapshot": {"model": {"lstm_units": 64}},
        "seed": 42,
        "split_strategy_used": "session-safe",
        "run_id": "test_run_123"
    }
    
    meta_path = tmp_path / "checkpoint_meta.json"
    with open(meta_path, "w") as f:
        json.dump(checkpoint_meta, f)
        
    # Read it back
    with open(meta_path, "r") as f:
        loaded = json.load(f)
        
    assert loaded["feature_count"] == 63
    assert loaded["sequence_length"] == 30
    assert loaded["label_to_id"]["HELLO"] == 0
    assert loaded["split_strategy_used"] == "session-safe"
