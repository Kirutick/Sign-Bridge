import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

logger = logging.getLogger(__name__)

def perform_dataset_split(
    X: np.ndarray, 
    y: np.ndarray, 
    sample_meta: List[Dict[str, Any]], 
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], str]:
    """
    Splits the dataset into train, val, and test sets.
    Prioritizes leakage-safe splitting if session_id or performer_id is fully present.
    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test), strategy_used
    """
    num_samples = len(y)
    
    # Try to extract grouping IDs
    group_ids = []
    missing_groups = 0
    
    for meta in sample_meta:
        gid = meta.get('session_id') or meta.get('performer_id')
        if gid is None or str(gid).strip() == "":
            missing_groups += 1
        group_ids.append(gid)
        
    strategy = "unknown"
    
    # Calculate sizes for the initial train / (val+test) split
    combined_test_val_size = test_size + val_size
    # Then split the (val+test) into val and test
    val_ratio_of_remainder = val_size / combined_test_val_size if combined_test_val_size > 0 else 0.5
    
    if missing_groups == 0 and len(set(group_ids)) >= 3:
        # 1. Fully present session/performer IDs -> Use GroupShuffleSplit
        strategy = "session-safe"
        logger.info("Using session-safe split (GroupShuffleSplit) based on session_id/performer_id.")
        
        gss = GroupShuffleSplit(n_splits=1, test_size=combined_test_val_size, random_state=random_state)
        train_idx, val_test_idx = next(gss.split(X, y, groups=group_ids))
        
        # Split val/test
        val_test_groups = [group_ids[i] for i in val_test_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_of_remainder, random_state=random_state)
        
        try:
            val_idx_relative, test_idx_relative = next(gss_val.split(
                X[val_test_idx], 
                y[val_test_idx], 
                groups=val_test_groups
            ))
            val_idx = val_test_idx[val_idx_relative]
            test_idx = val_test_idx[test_idx_relative]
        except ValueError:
            # Fallback if too few groups in the test set to split further
            logger.warning("Not enough groups to split val and test cleanly. Falling back to Stratified for val/test.")
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_of_remainder, random_state=random_state)
            val_idx_relative, test_idx_relative = next(sss_val.split(X[val_test_idx], y[val_test_idx]))
            val_idx = val_test_idx[val_idx_relative]
            test_idx = val_test_idx[test_idx_relative]
            
    else:
        # 2 & 3. Partially present or fully absent -> StratifiedShuffleSplit
        strategy = "stratified-random-fallback"
        
        if missing_groups < num_samples and missing_groups > 0:
            logger.warning(
                f"Dataset metadata is inconsistent: {num_samples - missing_groups} samples have session IDs, "
                f"but {missing_groups} are missing them. Falling back to stratified random split."
            )
            
        # The required unmissable warning (§4.3)
        warning_msg = (
            "WARNING: No session/performer metadata (or it was incomplete) — split may leak "
            "near-duplicate sequences between train and test; validation accuracy may be optimistic."
        )
        logger.warning(warning_msg)
        print(f"\n{warning_msg}\n")
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=combined_test_val_size, random_state=random_state)
        try:
            train_idx, val_test_idx = next(sss.split(X, y))
        except ValueError as e:
            # Fallback if class sizes are too small for stratified
            logger.warning(f"StratifiedShuffleSplit failed ({e}), falling back to purely random split.")
            indices = np.random.RandomState(random_state).permutation(num_samples)
            split_point = int(num_samples * (1 - combined_test_val_size))
            train_idx, val_test_idx = indices[:split_point], indices[split_point:]
            
        if len(val_test_idx) > 1:
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_of_remainder, random_state=random_state)
            try:
                val_idx_relative, test_idx_relative = next(sss_val.split(X[val_test_idx], y[val_test_idx]))
                val_idx = val_test_idx[val_idx_relative]
                test_idx = val_test_idx[test_idx_relative]
            except ValueError:
                # purely random if classes are too small
                indices = np.random.RandomState(random_state).permutation(len(val_test_idx))
                split_point = int(len(val_test_idx) * (1 - val_ratio_of_remainder))
                val_idx = val_test_idx[indices[:split_point]]
                test_idx = val_test_idx[indices[split_point:]]
        else:
            val_idx = val_test_idx
            test_idx = np.array([], dtype=int)

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),
        strategy
    )
