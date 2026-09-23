import os
import json
import hashlib
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any
import logging

from .encoder import create_label_mapping, encode_labels

logger = logging.getLogger(__name__)

def _hash_file(filepath: Path) -> str:
    """Computes MD5 hash of a file for caching purposes."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_dataset(configured_in_repo_path: str = "../dataset_export.json") -> Tuple[Path, str]:
    """
    Detects dataset location.
    Checks ml/data/raw/ for portable exports, then checks the configured in-repo path.
    Returns (path, source_type).
    Raises FileNotFoundError if neither exists.
    """
    raw_dir = Path("ml/data/raw")
    portable_files = []
    
    if raw_dir.exists() and raw_dir.is_dir():
        for ext in ["*.json", "*.jsonl", "*.npz", "*.zip"]:
            portable_files.extend(list(raw_dir.glob(ext)))
            
    in_repo_path = Path("ml") / configured_in_repo_path
    
    if portable_files and in_repo_path.exists():
        logger.warning(f"Both portable export ({portable_files[0]}) and in-repo export ({in_repo_path}) found. Preferring portable export.")
        return portable_files[0], "portable"
    
    if portable_files:
        return portable_files[0], "portable"
        
    if in_repo_path.exists():
        return in_repo_path, "in_repo"
        
    raise FileNotFoundError(
        f"Could not find dataset export! Checked ml/data/raw/ for portable exports "
        f"and {in_repo_path} for in-repo export."
    )

def _load_json_dataset(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], List[Dict[str, Any]]]:
    """Loads a JSON dataset, assuming the structure from Data Collection 3."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if 'X' not in data or 'y' not in data:
        raise ValueError("JSON payload missing 'X' or 'y' keys.")
        
    X = np.array(data['X'], dtype=np.float32)
    y_raw = data['y']
    
    # We might have string labels in y, or ints that map via data['labels']
    labels_dict = data.get('labels', {})
    if len(labels_dict) > 0 and isinstance(y_raw[0], int):
        # Phase 3 format: y contains ints, labels maps "0": "HELLO"
        str_labels = [labels_dict.get(str(label_id), "UNKNOWN") for label_id in y_raw]
    else:
        # Generic fallback if y contains strings directly
        str_labels = [str(val) for val in y_raw]
        
    sample_meta = data.get('sample_meta', [{} for _ in range(len(y_raw))])
    
    metadata = data.get('metadata', {})
    normalization_version = metadata.get('normalizationVersion', 'unknown')
    
    return X, np.array(str_labels), labels_dict, sample_meta, normalization_version

def _load_npz_dataset(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], List[Dict[str, Any]]]:
    """Loads an NPZ dataset."""
    data = np.load(filepath, allow_pickle=True)
    
    if 'X' not in data or 'y' not in data:
        raise ValueError("NPZ file missing 'X' or 'y' arrays.")
        
    X = data['X'].astype(np.float32)
    y_raw = data['y']
    
    str_labels = [str(val) for val in y_raw]
    sample_meta = data['sample_meta'].tolist() if 'sample_meta' in data else [{} for _ in range(len(y_raw))]
    
    metadata = data['metadata'].item() if 'metadata' in data else {}
    normalization_version = metadata.get('normalizationVersion', 'unknown')
    
    return X, np.array(str_labels), {}, sample_meta, normalization_version

def load_and_validate_dataset(
    configured_in_repo_path: str = "../dataset_export.json",
    force_reparse: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str], List[Dict[str, Any]], str, str]:
    """
    Finds, parses, strictly validates, and caches the dataset.
    Returns: X, y_encoded, label_to_id, id_to_label, sample_meta, dataset_version, normalization_version
    """
    dataset_path, _ = find_dataset(configured_in_repo_path)
    file_hash = _hash_file(dataset_path)
    
    cache_dir = Path("ml/data/processed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_npz = cache_dir / f"dataset_{file_hash}.npz"
    cache_json = cache_dir / f"meta_{file_hash}.json"
    
    if not force_reparse and cache_npz.exists() and cache_json.exists():
        logger.info(f"Cache hit for dataset hash {file_hash[:8]}. Loading from cache.")
        cached_data = np.load(cache_npz, allow_pickle=True)
        with open(cache_json, 'r') as f:
            meta = json.load(f)
            
        return (
            cached_data['X'], 
            cached_data['y'], 
            meta['label_to_id'], 
            {int(k): v for k, v in meta['id_to_label'].items()}, 
            cached_data['sample_meta'].tolist(),
            meta['dataset_version'],
            meta.get('normalization_version', 'unknown')
        )
        
    logger.info(f"Parsing dataset from {dataset_path}...")
    
    if dataset_path.suffix.lower() == '.json':
        X, y_str, _, sample_meta, norm_version = _load_json_dataset(dataset_path)
    elif dataset_path.suffix.lower() == '.npz':
        X, y_str, _, sample_meta, norm_version = _load_npz_dataset(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
        
    # --- Strict Validation ---
    if X.ndim != 3:
        raise ValueError(f"X must be 3-dimensional, got {X.ndim}D with shape {X.shape}")
    if X.shape[1] != 30:
        raise ValueError(f"Sequence length must be 30, got {X.shape[1]}")
    if X.shape[2] != 63:
        raise ValueError(f"Feature count must be 63, got {X.shape[2]}")
    if X.shape[0] != len(y_str):
        raise ValueError(f"X length ({X.shape[0]}) does not match y length ({len(y_str)})")
    if X.shape[0] != len(sample_meta):
        raise ValueError(f"X length ({X.shape[0]}) does not match sample_meta length ({len(sample_meta)})")
        
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X array contains NaN or Infinity values")
        
    # --- Encoding ---
    label_to_id, id_to_label = create_label_mapping(y_str.tolist())
    y_encoded = np.array(encode_labels(y_str.tolist(), label_to_id), dtype=np.int32)
    
    dataset_version = file_hash  # Fallback version is the hash itself
    
    # --- Caching ---
    np.savez_compressed(
        cache_npz, 
        X=X, 
        y=y_encoded, 
        sample_meta=np.array(sample_meta, dtype=object)
    )
    
    with open(cache_json, 'w') as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "dataset_version": dataset_version,
            "normalization_version": norm_version
        }, f, indent=2)
        
    return X, y_encoded, label_to_id, id_to_label, sample_meta, dataset_version, norm_version
