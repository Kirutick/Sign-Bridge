import json
from typing import Dict, List, Tuple

def create_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates a deterministic lexicographical mapping of string labels to integer IDs.
    
    Args:
        labels: List or set of unique string labels
        
    Returns:
        (label_to_id, id_to_label) dictionaries
    """
    unique_sorted = sorted(list(set(labels)))
    label_to_id = {label: idx for idx, label in enumerate(unique_sorted)}
    id_to_label = {idx: label for idx, label in enumerate(unique_sorted)}
    return label_to_id, id_to_label

def encode_labels(labels_list: List[str], label_to_id: Dict[str, int]) -> List[int]:
    """
    Encodes a list of string labels to integers using a specific mapping.
    
    Raises:
        ValueError if a label is not found in the mapping.
    """
    encoded = []
    for lbl in labels_list:
        if lbl not in label_to_id:
            raise ValueError(f"Label '{lbl}' not found in label mapping!")
        encoded.append(label_to_id[lbl])
    return encoded
