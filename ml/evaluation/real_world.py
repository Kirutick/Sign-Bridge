import os
import json
import numpy as np

def run_real_world_evaluation(model, real_world_dir, label_to_id):
    """
    Evaluates real-world samples staged in `ml/data/real_world/<category>/<label>/*.json`
    Returns a dictionary of metrics per category.
    """
    if not os.path.exists(real_world_dir):
        return None
        
    categories = [d for d in os.listdir(real_world_dir) if os.path.isdir(os.path.join(real_world_dir, d))]
    
    if not categories:
        return None
        
    results = {}
    found_samples = False
    
    for category in categories:
        category_dir = os.path.join(real_world_dir, category)
        labels = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        
        category_total = 0
        category_correct = 0
        
        for condition_label in labels:
            condition_dir = os.path.join(category_dir, condition_label)
            
            for file_name in os.listdir(condition_dir):
                if not file_name.endswith(".json"):
                    continue
                    
                file_path = os.path.join(condition_dir, file_name)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                    true_label = data.get("true_label")
                    sequence = data.get("sequence", [])
                    
                    if not true_label or not sequence:
                        continue
                        
                    if true_label not in label_to_id:
                        continue
                        
                    seq_arr = np.array(sequence)
                    if seq_arr.shape != (30, 63):
                        continue
                        
                    # Expand batch dimension
                    seq_arr = np.expand_dims(seq_arr, axis=0)
                    
                    pred_probs = model.predict(seq_arr, verbose=0)
                    pred_idx = np.argmax(pred_probs, axis=1)[0]
                    
                    true_idx = label_to_id[true_label]
                    
                    category_total += 1
                    if pred_idx == true_idx:
                        category_correct += 1
                        
                    found_samples = True
                except Exception:
                    pass
                    
        if category_total > 0:
            results[category] = {
                "accuracy": category_correct / category_total,
                "correct": category_correct,
                "total": category_total
            }
            
    if not found_samples:
        return None
        
    return results
