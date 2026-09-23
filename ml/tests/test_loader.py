import os
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from ml.preprocessing.loader import load_and_validate_dataset, find_dataset

@pytest.fixture
def clean_workspace(tmp_path):
    """Sets up a temporary directory mocking the ml/ folder structure."""
    ml_dir = tmp_path / "ml"
    raw_dir = ml_dir / "data" / "raw"
    raw_dir.mkdir(parents=True)
    
    # We will override the working directory for tests where needed
    yield ml_dir

def create_dummy_dataset(shape=(2, 30, 63)):
    X = np.random.rand(*shape).tolist()
    y = [0, 1]
    return {
        "X": X,
        "y": y,
        "labels": {"0": "A", "1": "B"},
        "metadata": {"sampleCount": shape[0]}
    }

def test_loader_detects_portable_export(clean_workspace, monkeypatch):
    monkeypatch.chdir(clean_workspace.parent)
    
    # Create portable export
    portable_path = clean_workspace / "data" / "raw" / "export.json"
    with open(portable_path, "w") as f:
        json.dump(create_dummy_dataset(), f)
        
    path, source_type = find_dataset("dummy_repo.json")
    assert source_type == "portable"
    assert path.resolve() == portable_path.resolve()

def test_loader_detects_in_repo_export(clean_workspace, monkeypatch):
    monkeypatch.chdir(clean_workspace.parent)
    
    # Create in-repo export only
    in_repo = clean_workspace / "repo_export.json"
    with open(in_repo, "w") as f:
        json.dump(create_dummy_dataset(), f)
        
    path, source_type = find_dataset("repo_export.json")
    assert source_type == "in_repo"
    assert path.resolve() == in_repo.resolve()

def test_loader_raises_when_neither_exists(clean_workspace, monkeypatch):
    monkeypatch.chdir(clean_workspace.parent)
    
    with pytest.raises(FileNotFoundError, match="Could not find dataset export"):
        find_dataset("nonexistent.json")

def test_loader_raises_on_shape_mismatch(clean_workspace, monkeypatch):
    monkeypatch.chdir(clean_workspace.parent)
    
    portable_path = clean_workspace / "data" / "raw" / "export.json"
    with open(portable_path, "w") as f:
        # Create invalid shape: 62 features instead of 63
        json.dump(create_dummy_dataset(shape=(2, 30, 62)), f)
        
    with pytest.raises(ValueError, match="Feature count must be 63"):
        load_and_validate_dataset()

@patch('ml.preprocessing.loader._load_json_dataset')
def test_loader_caching_behavior(mock_load_json, clean_workspace, monkeypatch):
    monkeypatch.chdir(clean_workspace.parent)
    
    # Create portable export
    portable_path = clean_workspace / "data" / "raw" / "export.json"
    
    # Setup mock to return valid tuple: X, y_str, labels_dict, sample_meta, norm_version
    dummy = create_dummy_dataset()
    mock_load_json.return_value = (
        np.array(dummy["X"], dtype=np.float32), 
        np.array(["A", "B"]), 
        dummy["labels"],
        [{}, {}],
        "v1"
    )
    
    with open(portable_path, "w") as f:
        json.dump(dummy, f)
        
    # First call: should parse and cache
    load_and_validate_dataset()
    assert mock_load_json.call_count == 1
    
    # Second call: should hit cache
    load_and_validate_dataset()
    assert mock_load_json.call_count == 1  # Did not increase
