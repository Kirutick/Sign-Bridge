import numpy as np

def perform_error_analysis(y_train, f1_scores, support_test, id_to_label, frequent_pairs, f1_threshold=0.70):
    """
    Perform evidence-driven error analysis on classes with F1 below f1_threshold.
    """
    num_classes = len(id_to_label)
    
    # Compute training support
    train_counts = {}
    for i in range(num_classes):
        train_counts[i] = int(np.sum(y_train == i))
        
    median_train_support = float(np.median(list(train_counts.values())))
    
    analysis_results = {}
    
    for i in range(num_classes):
        if f1_scores[i] < f1_threshold:
            class_name = id_to_label.get(i, id_to_label.get(str(i), f"Class_{i}"))
            
            explanations = []
            
            # Check 1: Insufficient examples
            if train_counts[i] < 50:
                explanations.append(f"insufficient examples: {train_counts[i]} training samples (threshold: 50)")
                
            # Check 2: Class imbalance
            if train_counts[i] < (median_train_support * 0.5):
                explanations.append(f"likely class imbalance: {train_counts[i]} training samples vs. a median of {median_train_support:.1f} across all classes")
                
            # Check 3: Similar hand configurations / Confused pairs
            pairs_for_class = [p for p in frequent_pairs if p["true_label"] == class_name]
            if pairs_for_class:
                top_pair = pairs_for_class[0]
                explanations.append(f"similar hand configurations: frequently confused with '{top_pair['pred_label']}' ({top_pair['count']}/{top_pair['true_support']} cases, {top_pair['rate']*100:.1f}%)")
                
            # Check 4: Noisy data / Label inconsistency
            # We don't have this metadata in `y_train` directly (loaders produce `sample_meta` which we could parse,
            # but without concrete variance/annotator data, we skip this to adhere to the hard rule).
            
            if not explanations:
                analysis_results[class_name] = "no clear cause identified from available data"
            else:
                analysis_results[class_name] = "; ".join(explanations)
                
    return analysis_results
