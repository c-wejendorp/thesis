"""
Compare two trained dynamic models to verify training determinism.
This script loads two models from their run directories and compares:
- Model weights
- Training configurations
- Training logs (epoch and step level)
"""
import torch
from pathlib import Path
import json

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.keyword_spotting.base_schema import KeyWordSpottingBaseConfig
from thesis_project.models.keyword_spotting.dynamic_schema import DynamicTrainingConfig
from thesis_project.models.components.routers import GRURouter

# ============================================================================
# CONFIGURATION: Specify the two model run directories to compare
# ============================================================================
RUN_DIR_1 = "model_runs/dynamic/2026-02-05_19-06-21"
RUN_DIR_2 = "model_runs/dynamic/2026-02-05_19-00-05"



def load_model_from_run_dir(run_dir: str, device: torch.device):
    """Load a trained dynamic model from its run directory."""
    run_path = Path(run_dir)
    
    print(f"\nLoading model from: {run_path}")
    
    # Load configuration
    config_path = run_path / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    full_config = DynamicTrainingConfig(**config_dict)
    
    # Load base model config
    base_model_dir = Path(full_config.dynamic_model.base_model_dir)
    base_config_path = base_model_dir / "base_config.json"
    with open(base_config_path, 'r') as f:
        base_config_dict = json.load(f)
    base_model_cfg = KeyWordSpottingBaseConfig(**base_config_dict)
    
    # Create base model
    base_model = KWSBase(base_model_cfg).to(device)
    base_model_state = torch.load(base_model_dir / "best_model.pth", map_location=device)
    base_model.load_state_dict(base_model_state)
    
    # Calculate ranks
    base_model_full_rank = min(base_model.cfg.backbone.n_channels_ext, base_model.cfg.backbone.n_channels_int)
    maximum_useful_rank = base_model_full_rank // 2
    
    # Create router
    router = GRURouter(
        input_dim=base_model.frontend.out_channels,
        gru_hidden_dim=full_config.router.hidden_dim,
        num_gru_layers=full_config.router.num_gru_layers,
        max_rank=maximum_useful_rank,
        num_rank_outputs=full_config.router.num_rank_outputs,
        last_layer_bias_init=full_config.router.last_layer_bias_init,
        pool_type=full_config.router.pool_mode,
        pool_reduction_factor=full_config.router.pool_reduction_factor,
    )
    
    # Create dynamic model
    dynamic_model = KWSDynamic(
        base=base_model,
        router=router,
        use_global_rank=full_config.router.use_global_rank,
        low_rank_frontend=full_config.dynamic_model.low_rank_frontend,
        freeze_base=full_config.dynamic_model.freeze_base
    ).to(device)
    
    # Load trained weights
    model_path = run_path / "best_model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            dynamic_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            dynamic_model.load_state_dict(checkpoint['model_state'])
        else:
            # Assume the entire dict is the state dict
            dynamic_model.load_state_dict(checkpoint)
    else:
        # Assume it's directly the state dict
        dynamic_model.load_state_dict(checkpoint)
    
    print(f"  ✅ Model loaded successfully")
    return dynamic_model, full_config


def load_logs_from_run_dir(run_dir: str):
    """Load training logs from a run directory."""
    run_path = Path(run_dir)
    
    epoch_logs = None
    step_logs = None
    
    # Load epoch logs
    epoch_log_path = run_path / "history.json"
    if epoch_log_path.exists():
        with open(epoch_log_path, 'r') as f:
            epoch_logs = json.load(f)
        print(f"  ✅ Loaded {len(epoch_logs)} epoch logs")
    
    # Load step logs
    step_log_path = run_path / "step_history.json"
    if step_log_path.exists():
        with open(step_log_path, 'r') as f:
            step_logs = json.load(f)
        print(f"  ✅ Loaded {len(step_logs)} step logs")
    
    return epoch_logs, step_logs


def compare_configs(config1: DynamicTrainingConfig, config2: DynamicTrainingConfig):
    """Compare two configurations."""
    print(f"\n{'='*70}")
    print("Comparing Configurations")
    print(f"{'='*70}")
    
    dict1 = config1.model_dump()
    dict2 = config2.model_dump()
    
    if dict1 == dict2:
        print("✅ Configurations are identical")
        return True
    else:
        print("❌ Configurations differ")
        # Find differences
        def find_diffs(d1, d2, prefix=""):
            for key in set(d1.keys()) | set(d2.keys()):
                if key not in d1:
                    print(f"  {prefix}{key}: missing in config 1")
                elif key not in d2:
                    print(f"  {prefix}{key}: missing in config 2")
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    find_diffs(d1[key], d2[key], prefix=f"{prefix}{key}.")
                elif d1[key] != d2[key]:
                    print(f"  {prefix}{key}: {d1[key]} != {d2[key]}")
        find_diffs(dict1, dict2)
        return False


def compare_models(model1, model2):
    """Compare two model's parameters."""
    print(f"\n{'='*70}")
    print("Comparing Model Weights")
    print(f"{'='*70}")
    
    mismatches = 0
    max_diff = 0.0
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
        
        if not torch.allclose(param1, param2, atol=1e-7):
            diff = (param1 - param2).abs().max().item()
            max_diff = max(max_diff, diff)
            mismatches += 1
            print(f"  ❌ {name1}: max diff = {diff:.2e}")
    
    if mismatches == 0:
        print("✅ All model parameters match perfectly!")
    else:
        print(f"❌ Found {mismatches} parameter mismatches")
        print(f"   Maximum difference: {max_diff:.2e}")
    
    return mismatches == 0


def compare_logs(logs1, logs2, log_type="epoch", max_items=None):
    """Compare training logs."""
    print(f"\n{'='*70}")
    print(f"Comparing {log_type.capitalize()} Logs")
    print(f"{'='*70}")
    
    if logs1 is None or logs2 is None:
        print(f"⚠️  One or both {log_type} logs are missing")
        return None
    
    if len(logs1) != len(logs2):
        print(f"❌ Different number of {log_type}s: {len(logs1)} vs {len(logs2)}")
        return False
    
    # Limit comparison if requested
    if max_items:
        logs1 = logs1[:max_items]
        logs2 = logs2[:max_items]
        print(f"Comparing first {max_items} {log_type}s")
    
    mismatches = 0
    checked_keys = set()
    
    for i, (log1, log2) in enumerate(zip(logs1, logs2)):
        for key in log1.keys():
            if key not in log2:
                print(f"  ❌ {log_type.capitalize()} {i}: Key '{key}' missing in run 2")
                mismatches += 1
                continue
            
            val1, val2 = log1[key], log2[key]
            
            # Skip nested dicts for now
            if isinstance(val1, dict) and isinstance(val2, dict):
                continue
            
            # Handle None values
            if val1 is None and val2 is None:
                continue
            if val1 is None or val2 is None:
                print(f"  ❌ {log_type.capitalize()} {i}, {key}: One is None, other is not")
                mismatches += 1
                continue
            
            # Handle lists
            if isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    print(f"  ❌ {log_type.capitalize()} {i}, {key}: Different list lengths")
                    mismatches += 1
                continue
            
            # Skip non-numeric values
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                continue
            
            checked_keys.add(key)
            
            if abs(val1 - val2) > 1e-6:
                print(f"  ❌ {log_type.capitalize()} {i}, {key}: {val1:.6f} vs {val2:.6f} (diff: {abs(val1-val2):.2e})")
                mismatches += 1
    
    if mismatches == 0:
        print(f"✅ All {log_type} logs match!")
        if checked_keys:
            print(f"   Checked keys: {', '.join(sorted(checked_keys))}")
        return True
    else:
        print(f"❌ Found {mismatches} {log_type} log mismatches")
        return False



def main():
    print(f"{'='*70}")
    print("Testing Training Determinism")
    print(f"{'='*70}")
    print(f"Run Directory 1: {RUN_DIR_1}")
    print(f"Run Directory 2: {RUN_DIR_2}")
    print(f"{'='*70}")
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load models and configs
    print(f"\n{'='*70}")
    print("Loading Models and Configurations")
    print(f"{'='*70}")
    
    model1, config1 = load_model_from_run_dir(RUN_DIR_1, device)
    epoch_logs1, step_logs1 = load_logs_from_run_dir(RUN_DIR_1)
    
    model2, config2 = load_model_from_run_dir(RUN_DIR_2, device)
    epoch_logs2, step_logs2 = load_logs_from_run_dir(RUN_DIR_2)
    
    # Compare configurations
    configs_match = compare_configs(config1, config2)
    
    # Compare model weights
    models_match = compare_models(model1, model2)
    
    # Compare epoch logs
    if epoch_logs1 and epoch_logs2:
        epochs_match = compare_logs(epoch_logs1, epoch_logs2, "epoch")
    else:
        print("\n⚠️  Skipping epoch log comparison (logs not available)")
        epochs_match = None
    
    # Compare step logs (first 100 steps to keep output manageable)
    if step_logs1 and step_logs2:
        steps_match = compare_logs(step_logs1, step_logs2, "step", max_items=100)
    else:
        print("\n⚠️  Skipping step log comparison (logs not available)")
        steps_match = None
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    
    all_match = (configs_match and models_match and 
                 (epochs_match is not False) and (steps_match is not False))
    
    if all_match:
        print("✅ SUCCESS: Models are identical!")
        print("   - Configurations match")
        print("   - Model weights match")
        if epochs_match is not None:
            print("   - Epoch logs match")
        if steps_match is not None:
            print("   - Step logs match")
    else:
        print("❌ FAILURE: Models differ")
        print(f"   - Configurations: {'✅ Match' if configs_match else '❌ Differ'}")
        print(f"   - Model weights: {'✅ Match' if models_match else '❌ Differ'}")
        if epochs_match is not None:
            print(f"   - Epoch logs: {'✅ Match' if epochs_match else '❌ Differ'}")
        if steps_match is not None:
            print(f"   - Step logs: {'✅ Match' if steps_match else '❌ Differ'}")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
