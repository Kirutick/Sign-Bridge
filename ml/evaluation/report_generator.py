import os
from datetime import datetime

def generate_evaluation_report(
    output_path,
    model_version,
    dataset_version,
    test_sample_count,
    loss,
    accuracy,
    macro_f1,
    weighted_f1,
    precision,
    recall,
    f1_scores,
    support,
    id_to_label,
    cm,
    frequent_pairs,
    error_analysis,
    split_strategy,
    seen_performer_acc,
    unseen_performer_acc,
    real_world_results,
    conf_analysis,
    threshold_results,
    recommended_threshold
):
    """
    Generates the comprehensive EVALUATION_REPORT.md per the specification.
    """
    num_classes = len(id_to_label)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# EVALUATION REPORT: Sign Bridge LSTM ({model_version})\n")
        f.write(f"Generated on: {datetime.now().isoformat()}\n")
        f.write(f"Dataset version: {dataset_version}\n\n")
        
        # Section 1: Overall Metrics
        f.write("## 1. Overall Metrics\n")
        f.write(f"- **Test Samples**: {test_sample_count}\n")
        f.write(f"- **Loss**: {loss:.4f}\n")
        f.write(f"- **Accuracy**: {accuracy:.4f}\n")
        f.write(f"- **Macro F1** (Primary): {macro_f1:.4f} *(Note: Macro is primary to prevent high-support classes from hiding poor performance on rare ones)*\n")
        f.write(f"- **Weighted F1**: {weighted_f1:.4f}\n\n")
        
        # Section 2: Confusion Matrix & Confused Pairs
        f.write("## 2. Confusion Matrix & Confused Pairs\n")
        f.write("*(Full confusion matrix array saved in evaluation artifacts)*\n\n")
        f.write("### Frequently Confused Pairs (>10% of true class support)\n")
        if not frequent_pairs:
            f.write("- No class pairs exceed the 10% confusion threshold.\n\n")
        else:
            for idx, pair in enumerate(frequent_pairs):
                f.write(f"{idx+1}. **{pair['true_label']}** predicted as **{pair['pred_label']}** "
                        f"({pair['count']}/{pair['true_support']} cases, {pair['rate']*100:.1f}%)\n")
        f.write("\n")
        
        # Section 3: Per-Class Report
        f.write("## 3. Per-Class Report\n\n")
        
        class_stats = []
        for i in range(num_classes):
            name = id_to_label.get(i, id_to_label.get(str(i), f"Class_{i}"))
            class_stats.append({
                "class": name,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1_scores[i],
                "support": int(support[i])
            })
            
        f.write("### Sorted by F1 (Ascending - Worst First)\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---|---|---|---|\n")
        sorted_by_f1 = sorted(class_stats, key=lambda x: x["f1"])
        for c in sorted_by_f1:
            f.write(f"| {c['class']} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['support']} |\n")
        f.write("\n")
        
        f.write("### Sorted Alphabetically\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---|---|---|---|\n")
        sorted_by_name = sorted(class_stats, key=lambda x: x["class"])
        for c in sorted_by_name:
            f.write(f"| {c['class']} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['support']} |\n")
        f.write("\n")
        
        # Section 4: Error Analysis
        f.write("## 4. Error Analysis (Evidence-Cited)\n")
        f.write("*(Investigating classes with F1 < 0.70)*\n\n")
        if not error_analysis:
            f.write("- All classes achieved F1 >= 0.70.\n\n")
        else:
            for cls, explanation in error_analysis.items():
                f.write(f"- **{cls}**: {explanation}\n")
        f.write("\n")
        
        # Section 5: Cross-Performer Testing
        f.write("## 5. Cross-Performer Testing\n")
        if split_strategy == "session_safe":
            f.write("- **Split Strategy**: `session-safe` (Performers were held out during splitting).\n")
            if seen_performer_acc is not None and unseen_performer_acc is not None:
                f.write(f"  - Accuracy on Seen Performers: {seen_performer_acc:.4f}\n")
                f.write(f"  - Accuracy on Unseen Performers: {unseen_performer_acc:.4f}\n")
            else:
                f.write("  - *Test set samples lack performer metadata, cannot break down by performer.*\n")
        else:
            f.write("- **Split Strategy**: `stratified-random` (Fallback).\n")
            f.write("- **WARNING**: A genuine unseen-performer evaluation is **not possible** from this test set, because performers were not held out during splitting. Test accuracy may be optimistic due to near-duplicate sequences across splits.\n")
        f.write("\n")
        
        # Section 6: Real-World Testing
        f.write("## 6. Real-World Testing\n")
        if not real_world_results:
            f.write("NO REAL-WORLD SAMPLES FOUND — SECTION SKIPPED\n\n")
        else:
            for cat, res in real_world_results.items():
                f.write(f"- **{cat}**: {res['accuracy']*100:.1f}% ({res['correct']}/{res['total']})\n")
            f.write("\n")
            
        # Section 7: Confidence Analysis
        f.write("## 7. Confidence Analysis\n")
        if conf_analysis:
            corr_mean = conf_analysis['correct_mean']
            incorr_mean = conf_analysis['incorrect_mean']
            f.write(f"- Mean confidence for CORRECT predictions: {corr_mean:.4f}\n")
            f.write(f"- Mean confidence for INCORRECT predictions: {incorr_mean:.4f}\n")
            
            if incorr_mean > 0.8:
                f.write("- **Finding**: The model is highly overconfident when wrong.\n")
            elif abs(corr_mean - incorr_mean) < 0.1:
                f.write("- **Finding**: The model's confidence does not effectively distinguish correct from incorrect predictions.\n")
            else:
                f.write("- **Finding**: Incorrect predictions are generally less confident than correct ones.\n")
        f.write("\n")
        
        # Section 8: Threshold Analysis
        f.write("## 8. Threshold Analysis\n")
        f.write("| Threshold | Coverage | Accepted Accuracy |\n")
        f.write("|---|---|---|\n")
        for t, res in sorted(threshold_results.items()):
            f.write(f"| {t:.2f} | {res['coverage']*100:.1f}% | {res['accuracy']*100:.1f}% |\n")
        f.write("\n")
        
        if recommended_threshold:
            t = recommended_threshold
            res = threshold_results[t]
            f.write(f"**Recommended Operating Point**: Threshold **{t:.2f}**. ")
            f.write(f"This retains {res['coverage']*100:.1f}% coverage at {res['accuracy']*100:.1f}% accuracy, balancing responsiveness with reliability.\n\n")
            
        # Section 9: Limitations
        f.write("## 9. Limitations\n")
        f.write("- This report is a snapshot of the held-out test set and is not subject to further model-selection tuning.\n")
        if not real_world_results:
            f.write("- **No real-world data yet**: Generalization to diverse environments and out-of-distribution recording conditions is entirely unknown.\n")
        if split_strategy != "session_safe":
            f.write("- **Unseen-performer testing unavailable**: The training split did not hold out performers due to missing metadata.\n")
        
        low_support_classes = [c['class'] for c in class_stats if c['support'] < 10]
        if low_support_classes:
            f.write(f"- **Low-support classes**: The following classes have < 10 test samples, making their metrics statistically unstable: {', '.join(low_support_classes)}.\n")
