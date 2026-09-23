import os
import sys
import yaml
import copy
import itertools
import argparse
import datetime
from pathlib import Path

def generate_grid(sweep_config_path: str, sweep_id: str = None) -> str:
    with open(sweep_config_path, 'r') as f:
        sweep_def = yaml.safe_load(f)
        
    grid = sweep_def.get('grid', {})
    
    # Flatten the grid definitions for itertools.product
    keys = []
    val_lists = []
    
    for section, params in grid.items():
        for param, values in params.items():
            keys.append((section, param))
            val_lists.append(values)
            
    combinations = list(itertools.product(*val_lists))
    print(f"Generated {len(combinations)} combinations from the grid definition.")
    
    if not sweep_id:
        sweep_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    out_dir = Path(f"ml/experiments/grid_{sweep_id}/configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, combo in enumerate(combinations):
        # Start with the constants base
        run_config = copy.deepcopy(sweep_def.get('constants', {}))
        if 'model' not in run_config:
            run_config['model'] = {}
        if 'training' not in run_config:
            run_config['training'] = {}
            
        # Apply the specific combination overrides
        for i, (section, param) in enumerate(keys):
            run_config[section][param] = combo[i]
            
        # Write file
        exp_id = f"exp_{idx+1:04d}"
        run_config['experiment_id'] = exp_id
        run_config['sweep_id'] = sweep_id
        
        out_file = out_dir / f"{exp_id}.yaml"
        with open(out_file, 'w') as f:
            yaml.dump(run_config, f, default_flow_style=False)
            
    print(f"Successfully wrote {len(combinations)} configuration files to {out_dir}")
    return str(out_dir.parent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/sweep_config.yaml", help="Path to sweep definition YAML")
    parser.add_argument("--sweep_id", default=None, help="Optional manual sweep ID")
    args = parser.parse_args()
    
    generate_grid(args.config, args.sweep_id)
