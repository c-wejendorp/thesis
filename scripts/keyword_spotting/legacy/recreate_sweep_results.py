"""
Script to recreate sweep_results.json from existing model runs by reading their history files.
Usage: python recreate_sweep_results.py <sweep_folder_path>
"""

import sys
import json
import numpy as np
from pathlib import Path
import re


def parse_run_folder_name(folder_name):
    """
    Parse folder name to extract hyperparameters.
    Expected format: YYYY-MM-DD_HH-MM-SS_rank_XX_weight_YY_mode_ZZ
    """
    # Try to extract from folder name
    rank_match = re.search(r'rank_(\d+)', folder_name)
    weight_match = re.search(r'weight_([\d.]+)', folder_name)
    mode_match = re.search(r'mode_(\w+)', folder_name)
    
    target_rank = int(rank_match.group(1)) if rank_match else None
    rank_loss_weight = float(weight_match.group(1)) if weight_match else None
    rank_loss_mode = mode_match.group(1) if mode_match else None
    
    return target_rank, rank_loss_weight, rank_loss_mode


def extract_metrics_from_history(history_path):
    """
    Load history.npy and extract the metrics we need.
    """
    try:
        history = np.load(history_path, allow_pickle=True).item()
        epochs_log = history.get("epochs_log", [])
        
        if not epochs_log:
            return None
        
        last_epoch = epochs_log[-1]
        val_accs = [e.get("val_acc") for e in epochs_log if e.get("val_acc") is not None]
        val_losses = [e.get("val_loss") for e in epochs_log if e.get("val_loss") is not None]
        
        # Find epochs with best metrics
        best_val_acc_idx = val_accs.index(max(val_accs)) if val_accs else None
        best_val_loss_idx = val_losses.index(min(val_losses)) if val_losses else None
        
        metrics = {
            "final_train_loss": last_epoch.get("train_loss"),
            "final_train_acc": last_epoch.get("train_acc"),
            "best_val_acc": max(val_accs) if val_accs else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "final_epoch_rank": last_epoch.get("train_expected_rank"),
            "best_val_acc_rank": epochs_log[best_val_acc_idx].get("train_expected_rank") if best_val_acc_idx is not None else None,
            "best_val_loss_rank": epochs_log[best_val_loss_idx].get("train_expected_rank") if best_val_loss_idx is not None else None,
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error loading {history_path}: {e}")
        return None


def recreate_sweep_results(sweep_folder):
    """
    Scan sweep folder, load all history files, and create sweep_results_updated.json
    """
    sweep_path = Path(sweep_folder)
    
    if not sweep_path.exists():
        print(f"Error: Sweep folder {sweep_folder} does not exist")
        return
    
    sweep_results = []
    
    # Find all subdirectories that contain history.npy
    for run_dir in sorted(sweep_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        history_path = run_dir / "history.npy"
        
        if not history_path.exists():
            print(f"Warning: No history.npy found in {run_dir.name}, skipping...")
            continue
        
        print(f"Processing {run_dir.name}...")
        
        # Parse hyperparameters from folder name
        target_rank, rank_loss_weight, rank_loss_mode = parse_run_folder_name(run_dir.name)
        
        # Extract metrics from history
        metrics = extract_metrics_from_history(history_path)
        
        if metrics is None:
            print(f"  ✗ Failed to extract metrics")
            sweep_results.append({
                "target_rank": target_rank,
                "rank_loss_weight": rank_loss_weight,
                "rank_loss_mode": rank_loss_mode,
                "error": "Failed to extract metrics",
                "run_dir": str(run_dir),
            })
            continue
        
        # Combine hyperparameters and metrics
        result = {
            "target_rank": target_rank,
            "rank_loss_weight": rank_loss_weight,
            "rank_loss_mode": rank_loss_mode,
            **metrics,
            "run_dir": str(run_dir),
        }
        
        sweep_results.append(result)
        print(f"  ✓ Extracted metrics")
        if metrics["best_val_acc"]:
            print(f"    Best val acc: {metrics['best_val_acc']:.2f}%")
        if metrics["best_val_loss"]:
            print(f"    Best val loss: {metrics['best_val_loss']:.4f}")
    
    # Save results
    output_path = sweep_path / "sweep_results_updated.json"
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Processed {len(sweep_results)} runs")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Print summary of top configurations
    successful_runs = [r for r in sweep_results if "error" not in r and r.get("best_val_acc") is not None]
    if successful_runs:
        successful_runs.sort(key=lambda x: x["best_val_acc"], reverse=True)
        
        print("\nTop 5 configurations:")
        for i, result in enumerate(successful_runs[:5], 1):
            print(f"\n{i}. Rank={result['target_rank']}, Weight={result['rank_loss_weight']}, Mode={result['rank_loss_mode']}")
            print(f"   Val Acc: {result['best_val_acc']:.2f}%, Val Loss: {result['best_val_loss']:.4f}")
            if result.get('final_epoch_rank'):
                print(f"   Final Epoch Rank: {result['final_epoch_rank']:.1f}", end="")
            if result.get('best_val_acc_rank'):
                print(f", Best Val Acc Rank: {result['best_val_acc_rank']:.1f}", end="")
            if result.get('best_val_loss_rank'):
                print(f", Best Val Loss Rank: {result['best_val_loss_rank']:.1f}")
            else:
                print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recreate_sweep_results.py <sweep_folder_path>")
        sys.exit(1)
    
    sweep_folder = sys.argv[1]
    recreate_sweep_results(sweep_folder)
