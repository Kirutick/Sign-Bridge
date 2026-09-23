import os
import sys
import json
import argparse
from pathlib import Path

def generate_summary(sweep_id: str, overfit_threshold: float = 0.15, f1_tolerance: float = 0.01):
    sweep_exp_dir = Path("ml/experiments") / sweep_id
    sweep_export_dir = Path("ml/exports") / sweep_id
    
    if not sweep_exp_dir.exists() or not sweep_export_dir.exists():
        print(f"Error: Could not find sweep data for {sweep_id}")
        sys.exit(1)
        
    runs = []
    
    # Iterate through all experiment folders
    for exp_dir in sweep_export_dir.glob("exp_*"):
        exp_id = exp_dir.name
        meta_path = exp_dir / "checkpoint_meta.json"
        history_path = sweep_exp_dir / exp_id / "history.json"
        
        if not meta_path.exists() or not history_path.exists():
            continue
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        best_epoch = history.get('val_accuracy', []).index(max(history.get('val_accuracy', [0])))
        val_acc = history['val_accuracy'][best_epoch]
        train_acc = history['accuracy'][best_epoch]
        
        # We stored val_f1_macro from our callback
        val_f1 = history.get('val_f1_macro', [0.0])[best_epoch]
        
        config = meta['config_snapshot']
        
        run_data = {
            "exp_id": exp_id,
            "architecture": f"{config['model'].get('layers', 1)}xLSTM({config['model'].get('lstm_units', 64)})-Dense({config['model'].get('dense_units', 64)})",
            "dropout": config['model'].get('dropout_rate', 0.5),
            "lr": config['training'].get('learning_rate', 0.001),
            "batch_size": config['training'].get('batch_size', 32),
            "val_acc": val_acc,
            "val_f1": val_f1,
            "train_acc": train_acc,
            "params": history['_meta'].get('parameter_count', 0),
            "time": meta.get('train_time_seconds', 0.0),
            "split_strategy": meta.get('split_strategy_used', 'unknown')
        }
        
        # Overfit flagging
        run_data['is_overfit'] = (train_acc - val_acc) > overfit_threshold
        runs.append(run_data)
        
    if not runs:
        print("No completed runs found.")
        return
        
    # Sort by Val F1 descending
    runs.sort(key=lambda x: x['val_f1'], reverse=True)
    
    # Recommendation Logic
    valid_runs = [r for r in runs if not r['is_overfit']]
    recommendation = None
    if valid_runs:
        max_f1 = valid_runs[0]['val_f1']
        # Filter within tolerance
        candidates = [r for r in valid_runs if (max_f1 - r['val_f1']) <= f1_tolerance]
        # Min parameters, then min time
        recommendation = min(candidates, key=lambda x: (x['params'], x['time']))
        
    # Generate Markdown Table
    md = [f"# Hyperparameter Sweep Summary: {sweep_id}\n"]
    if runs:
        md.append(f"**Split Strategy Used:** {runs[0]['split_strategy']}")
        if runs[0]['split_strategy'] != "session-safe":
            md.append("> **WARNING:** No session/performer metadata (or it was incomplete) — split may leak near-duplicate sequences between train and test; validation accuracy may be optimistic.\n")
            
    md.append("## Recommendation\n")
    if recommendation:
        md.append(f"**Selected Model:** `{recommendation['exp_id']}`")
        md.append(f"- **Reasoning**: Chosen from models within {f1_tolerance} F1 of the max ({max_f1:.4f}), selecting the one with the fewest parameters ({recommendation['params']}), tie-broken by training time ({recommendation['time']:.1f}s). Overfitting threshold was set to {overfit_threshold} gap.\n")
    else:
        md.append("*No valid recommendation found (all runs overfit).* \n")
        
    md.append("> **IMPORTANT:** Test set was not used in this selection.\n")
    
    md.append("## Top 10 Models (Condensed View)\n")
    headers = "| Experiment | Architecture | Dropout | LR | Batch | Val Acc | Val F1 | Params | Time (s) | Overfit? |"
    divider = "|---|---|---|---|---|---|---|---|---|---|"
    
    def format_row(r):
        overfit_str = "⚠️ YES" if r['is_overfit'] else "No"
        return f"| {r['exp_id']} | {r['architecture']} | {r['dropout']} | {r['lr']} | {r['batch_size']} | {r['val_acc']:.4f} | {r['val_f1']:.4f} | {r['params']} | {r['time']:.1f} | {overfit_str} |"

    md.append(headers)
    md.append(divider)
    for r in runs[:10]:
        md.append(format_row(r))
        
    md.append("\n## Full Results (648 Grid)\n")
    md.append(headers)
    md.append(divider)
    for r in runs:
        md.append(format_row(r))
        
    out_path = sweep_export_dir / "RECOMMENDATION.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
        
    print(f"Successfully generated summary: {out_path}")
    print("\n--- TOP 10 VIEW ---")
    print("\n".join(md[6:19]))
    print("\nTest set was not used in this selection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", required=True)
    parser.add_argument("--overfit_threshold", type=float, default=0.15)
    parser.add_argument("--f1_tolerance", type=float, default=0.01)
    args = parser.parse_args()
    
    generate_summary(args.sweep_id, args.overfit_threshold, args.f1_tolerance)
