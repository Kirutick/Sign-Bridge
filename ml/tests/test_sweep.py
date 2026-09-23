import os
import sys
import yaml
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml.scripts.generate_grid import generate_grid

def test_generate_grid_exactly_648_configs(tmp_path):
    # Use the real config file
    config_path = Path("ml/configs/sweep_config.yaml")
    if not config_path.exists():
        pytest.skip("sweep_config.yaml not found yet")
        
    out_dir = generate_grid(str(config_path), sweep_id="test_sweep")
    
    # Assert exactly 648 files generated
    yaml_files = list(Path(out_dir).glob("configs/*.yaml"))
    assert len(yaml_files) == 648

@patch('ml.scripts.sweep.train_model')
@patch('ml.scripts.sweep.load_and_validate_dataset')
def test_sweep_resumability_skips_completed(mock_load, mock_train, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # Setup mock dataset
    dummy_X = np.zeros((10, 30, 63))
    dummy_y = np.zeros(10)
    mock_load.return_value = (dummy_X, dummy_y, {"A": 0}, {"0": "A"}, [{} for _ in range(10)], "v1", "v1")
    mock_train.return_value = (None, {'loss': [0.1], 'val_accuracy': [0.9], 'accuracy': [0.9], 'val_loss': [0.1]})
    
    # Create fake sweep configs
    sweep_dir = tmp_path / "ml/experiments/grid_resume"
    configs_dir = sweep_dir / "configs"
    configs_dir.mkdir(parents=True)
    
    with open(configs_dir / "exp_001.yaml", "w") as f:
        yaml.dump({"experiment_id": "exp_001", "sweep_id": "grid_resume"}, f)
    with open(configs_dir / "exp_002.yaml", "w") as f:
        yaml.dump({"experiment_id": "exp_002", "sweep_id": "grid_resume"}, f)
        
    # Mark exp_001 as completed by creating its meta file
    meta_dir = tmp_path / "ml/exports/grid_resume/exp_001"
    meta_dir.mkdir(parents=True)
    with open(meta_dir / "checkpoint_meta.json", "w") as f:
        json.dump({}, f)
        
    # Run sweep main logic
    from ml.scripts.sweep import main
    # We must patch sys.argv
    test_args = ["sweep.py", "--sweep_dir", str(sweep_dir)]
    with patch.object(sys, 'argv', test_args):
        try:
            main()
        except SystemExit:
            pass
            
    # train_model should only have been called ONCE (for exp_002)
    assert mock_train.call_count == 1

@patch('ml.scripts.final_eval.load_and_validate_dataset')
@patch('ml.scripts.final_eval.tf.keras.models.load_model')
def test_firewall_blocks_second_run(mock_model, mock_load, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    
    rec_dir = tmp_path / "ml/exports/grid_test/exp_001"
    rec_dir.mkdir(parents=True)
    
    with open(rec_dir / "checkpoint_meta.json", "w") as f:
        json.dump({"seed": 42, "config_snapshot": {}}, f)
    with open(rec_dir / "best_model.keras", "w") as f:
        f.write("mock")
        
    # Create the lock file
    with open(rec_dir / ".eval_lock", "w") as f:
        f.write("locked")
        
    from ml.scripts.final_eval import main
    
    test_args = ["final_eval.py", "--recommendation", str(rec_dir)]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
        
    captured = capsys.readouterr()
    assert "FIREWALL ERROR" in captured.out

def test_recommendation_logic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    sweep_id = "test_logic"
    exp_dir = tmp_path / "ml/experiments" / sweep_id
    export_dir = tmp_path / "ml/exports" / sweep_id
    
    exp_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    
    def create_run(idx, val_acc, val_f1, train_acc, params, time):
        run_id = f"exp_{idx}"
        
        run_exp_dir = exp_dir / run_id
        run_exp_dir.mkdir(parents=True)
        with open(run_exp_dir / "history.json", "w") as f:
            json.dump({
                "val_accuracy": [val_acc],
                "accuracy": [train_acc],
                "val_f1_macro": [val_f1],
                "_meta": {"parameter_count": params}
            }, f)
            
        run_export_dir = export_dir / run_id
        run_export_dir.mkdir(parents=True)
        with open(run_export_dir / "checkpoint_meta.json", "w") as f:
            json.dump({
                "config_snapshot": {"model": {}, "training": {}},
                "train_time_seconds": time
            }, f)
            
    # Run 1: Overfits massively (1.0 vs 0.7) -> Should be dropped
    create_run("1", 0.70, 0.65, 1.0, 1000, 10.0)
    
    # Run 2: Max F1 (0.95), but huge params (5000)
    create_run("2", 0.95, 0.95, 0.96, 5000, 20.0)
    
    # Run 3: Within tolerance (0.942 > 0.95 - 0.01), very small params (500) -> SHOULD WIN
    create_run("3", 0.94, 0.942, 0.95, 500, 5.0)
    
    from ml.scripts.summarize_sweep import generate_summary
    generate_summary(sweep_id, overfit_threshold=0.15, f1_tolerance=0.01)
    
    # Read recommendation md
    rec_file = export_dir / "RECOMMENDATION.md"
    assert rec_file.exists()
    
    with open(rec_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "**Selected Model:** `exp_3`" in content
    assert "Test set was not used in this selection." in content
