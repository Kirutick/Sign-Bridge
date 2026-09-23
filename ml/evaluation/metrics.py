import numpy as np
from sklearn.metrics import confusion_matrix # type: ignore

def compute_confidence_analysis(y_true, y_pred, y_pred_probs):
    """
    Bucket predictions into correct and incorrect, returning mean/median max confidence for both.
    """
    max_probs = np.max(y_pred_probs, axis=1)
    
    correct_mask = (y_true == y_pred)
    incorrect_mask = (y_true != y_pred)
    
    correct_probs = max_probs[correct_mask]
    incorrect_probs = max_probs[incorrect_mask]
    
    result = {
        "correct_mean": float(np.mean(correct_probs)) if len(correct_probs) > 0 else 0.0,
        "correct_median": float(np.median(correct_probs)) if len(correct_probs) > 0 else 0.0,
        "incorrect_mean": float(np.mean(incorrect_probs)) if len(incorrect_probs) > 0 else 0.0,
        "incorrect_median": float(np.median(incorrect_probs)) if len(incorrect_probs) > 0 else 0.0,
        "correct_count": len(correct_probs),
        "incorrect_count": len(incorrect_probs)
    }
    
    return result

def compute_threshold_analysis(y_true, y_pred, y_pred_probs, thresholds=[0.60, 0.70, 0.80, 0.90]):
    """
    Evaluate coverage and accepted accuracy for varying confidence thresholds.
    """
    max_probs = np.max(y_pred_probs, axis=1)
    total_samples = len(y_true)
    
    results = {}
    
    for t in thresholds:
        accepted_mask = max_probs >= t
        accepted_count = np.sum(accepted_mask)
        coverage = accepted_count / total_samples if total_samples > 0 else 0.0
        
        if accepted_count > 0:
            accepted_correct = np.sum(y_true[accepted_mask] == y_pred[accepted_mask])
            accuracy = accepted_correct / accepted_count
        else:
            accuracy = 0.0
            
        results[t] = {
            "coverage": float(coverage),
            "accuracy": float(accuracy),
            "accepted_count": int(accepted_count)
        }
        
    return results

def analyze_confusion_matrix(y_true, y_pred, id_to_label, support_fraction=0.10):
    """
    Generate confusion matrix and extract pairs exceeding support_fraction of true class support.
    """
    num_classes = len(id_to_label)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    frequent_pairs = []
    
    for i in range(num_classes):
        true_label = id_to_label.get(i, id_to_label.get(str(i), f"Class_{i}"))
        total_true = int(np.sum(cm[i, :]))
        if total_true == 0:
            continue
            
        for j in range(num_classes):
            if i != j:
                count = int(cm[i, j])
                rate = count / total_true
                if rate > support_fraction:
                    pred_label = id_to_label.get(j, id_to_label.get(str(j), f"Class_{j}"))
                    frequent_pairs.append({
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": count,
                        "rate": rate,
                        "true_support": total_true
                    })
                    
    # Sort by absolute count descending, then rate descending
    frequent_pairs.sort(key=lambda x: (x["count"], x["rate"]), reverse=True)
    
    return cm.tolist(), frequent_pairs
