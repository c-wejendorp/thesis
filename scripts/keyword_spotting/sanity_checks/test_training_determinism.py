"""
Test that training produces identical models across multiple runs.
"""
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import copy
import json

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.components.routers import GRURouter
from thesis_project.utils.paths import get_data_dir
from thesis_project.utils.compute_macs import compute_macs_base_model, compute_avg_rank_from_macs
from thesis_project.datasets import SpeechCommandsGoogle
import sys
sys.path.append(str(Path(__file__).parent))
from base_config_temp import cfg, noise_train_cfg
from thesis_project.training.key_word_spotting import fit_dynamic_model, create_dynamic_routing_loss

# Configuration
SEED = 42
NUM_EPOCHS = 2  # Keep small for quick testing
BATCH_SIZE = 64

# Optional: Load from saved models/logs instead of training
# Set to None to train from scratch
NUM_RANK_OUTPUTS = 1  # should match the number used in the saved models

MODEL1_RUN_DIR = "model_runs/dynamic/2026-02-04_21-40-28"
MODEL2_RUN_DIR = "model_runs/dynamic/2026-02-04_21-42-14"

MODEL1_PATH = MODEL1_RUN_DIR + "/best_model.pth"
EPOCH_LOGS1_PATH = MODEL1_RUN_DIR + "/history.json"  # Epoch logs contain validation metrics
STEP_LOGS1_PATH = MODEL1_RUN_DIR + "/step_history.json"

MODEL2_PATH = MODEL2_RUN_DIR + "/best_model.pth"
EPOCH_LOGS2_PATH = MODEL2_RUN_DIR + "/history.json"  # Epoch logs contain validation metrics
STEP_LOGS2_PATH = MODEL2_RUN_DIR + "/step_history.json"

def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model_and_setup(device):
    """Create model, optimizer, scheduler, etc."""
    # Load base model
    BASE_MODEL_PATH = "model_runs/base/2025-12-14_18-48-50/best_model.pth"
    base_model_state = torch.load(BASE_MODEL_PATH)
    base_model = KWSBase(cfg).to(device)
    base_model.load_state_dict(base_model_state)
    base_model_full_rank = min(base_model.cfg.backbone.n_channels_ext, base_model.cfg.backbone.n_channels_int)
    maximum_useful_rank = base_model_full_rank // 2
    
    # Create router
    router = GRURouter(
        input_dim=base_model.frontend.out_channels,
        gru_hidden_dim=48,
        num_gru_layers=1,
        max_rank=maximum_useful_rank,
        num_rank_outputs=NUM_RANK_OUTPUTS,
        last_layer_bias_init=3.0,
        pool_type="subsample",
        pool_reduction_factor=2,
    )
    
    # Create dynamic model
    dynamic_model = KWSDynamic(
        base=base_model,
        router=router,
        use_global_rank=False,
        low_rank_frontend=False,
        freeze_base=True
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=1e-4)
    
    # Scheduler
    return dynamic_model, optimizer, maximum_useful_rank

def create_dataloader(seed):
    """Create training dataloader."""
    train_set = SpeechCommandsGoogle(
        root=str(get_data_dir()),
        subset="training",
        download=True,
        seed=seed,
        **noise_train_cfg.model_dump()
    )
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    
    return train_loader

def create_val_dataloader(seed):
    """Create validation dataloader."""
    val_set = SpeechCommandsGoogle(
        root=str(get_data_dir()),
        subset="validation",
        download=True,
        seed=seed,
        **noise_train_cfg.model_dump()
    )
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        generator=generator,
    )
    
    return val_loader

def train_model(run_name):
    """Train a model and return results."""
    print(f"\n{'='*70}")
    print(f"Training Run: {run_name}")
    print(f"{'='*70}")
    
    # Set seeds
    set_all_seeds(SEED)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create model and dataloader
    model, optimizer, max_rank = create_model_and_setup(device)
    train_loader = create_dataloader(SEED)
    val_loader = create_val_dataloader(SEED)
    
    # Loss function
    COMPRESSED_BASE_MODEL_RANK_PR_STACK = [30, 30, 30]
    TARGET_MAC_FRACTION = 0.5
    
    macs_base = compute_macs_base_model(
        model.base,
        time_steps=63, 
        rank_pr_stack=COMPRESSED_BASE_MODEL_RANK_PR_STACK
    )
    macs_router = model.router.compute_macs(sequence_length=63)
    macs_target = TARGET_MAC_FRACTION * macs_base
    macs_target_budget = int(macs_target - macs_router)
    
    ranks_target = compute_avg_rank_from_macs(
        model.base,
        time_steps=63,
        macs=macs_target_budget,
        num_stacks=3
    )
    
    avg_target_rank_normalized = (ranks_target - 1) / (max_rank - 1)
    
    criterion = create_dynamic_routing_loss(
        rank_loss_mode="one_sided_mse",
        rank_loss_weight=20,
        target_rank_normalized=avg_target_rank_normalized
    )
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=0.5e-4,
    )
    
    # Train
    run_dir = Path(f"model_runs/dynamic/test_{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    trained_model, epoch_logs, step_logs = fit_dynamic_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=NUM_EPOCHS,
        scheduler=scheduler,
        run_dir=str(run_dir),
        max_rank=max_rank,
        val_loader=val_loader,
        val_snr_values=[0, 5, float('inf')],  # Use subset of SNRs for faster testing
        main_val_snr=None,  # Use average across all SNRs for best model selection
        val_epoch=1,  # Validate every epoch
        enable_rank_supervision=False,
        rank_supervision_stable=False,
        save_epoch_checkpoints=False,
        log_every_n_steps=1,
        scheduler_mode="step",
    )
    
    return trained_model, epoch_logs, step_logs

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
            print(f"❌ {name1}: max diff = {diff:.2e}")
    
    if mismatches == 0:
        print("✅ All model parameters match perfectly!")
    else:
        print(f"❌ Found {mismatches} parameter mismatches")
        print(f"   Maximum difference: {max_diff:.2e}")
    
    return mismatches == 0

def compare_logs(logs1, logs2, log_type="epoch"):
    """Compare training logs."""
    print(f"\n{'='*70}")
    print(f"Comparing {log_type.capitalize()} Logs")
    print(f"{'='*70}")
    
    if len(logs1) != len(logs2):
        print(f"❌ Different number of {log_type}s: {len(logs1)} vs {len(logs2)}")
        return False
    
    mismatches = 0
    checked_keys = set()
    val_keys = set()
    
    for i, (log1, log2) in enumerate(zip(logs1, logs2)):
        for key in log1.keys():
            if key not in log2:
                print(f"❌ {log_type.capitalize()} {i}: Key '{key}' missing in run 2")
                mismatches += 1
                continue
            
            val1, val2 = log1[key], log2[key]
            
            # Handle nested dicts (e.g., val_results_all_snr)
            if isinstance(val1, dict) and isinstance(val2, dict):
                continue  # Skip nested dicts for now, could recurse if needed
            
            # Handle lists
            if isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    print(f"❌ {log_type.capitalize()} {i}, {key}: Different list lengths")
                    mismatches += 1
                    continue
                # Compare numeric lists element-wise
                for j, (v1, v2) in enumerate(zip(val1, val2)):
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        if abs(v1 - v2) > 1e-6:
                            print(f"❌ {log_type.capitalize()} {i}, {key}[{j}]: {v1:.6f} vs {v2:.6f} (diff: {abs(v1-v2):.2e})")
                            mismatches += 1
                continue
            
            # Handle None values (e.g., when validation not performed)
            if val1 is None and val2 is None:
                continue
            if val1 is None or val2 is None:
                print(f"❌ {log_type.capitalize()} {i}, {key}: One is None, other is not")
                mismatches += 1
                continue
            
            # Skip non-numeric values
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                continue
            
            checked_keys.add(key)
            
            # Track validation keys separately
            if key.startswith('val_'):
                val_keys.add(key)
            
            if abs(val1 - val2) > 1e-6:
                print(f"❌ {log_type.capitalize()} {i}, {key}: {val1:.6f} vs {val2:.6f} (diff: {abs(val1-val2):.2e})")
                mismatches += 1
    
    if mismatches == 0:
        print(f"✅ All {log_type} logs match!")
        if checked_keys:
            print(f"   Checked keys: {', '.join(sorted(checked_keys))}")
        if val_keys:
            print(f"   Validation keys checked: {', '.join(sorted(val_keys))}")
        return True
    else:
        print(f"❌ Found {mismatches} {log_type} log mismatches")
        return False

def load_model_from_path(model_path, device):
    """Load a trained model from a checkpoint file."""
    print(f"\n{'='*70}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*70}")
    
    # Set seeds for consistency
    set_all_seeds(SEED)
    
    # Create model structure
    model, _, _ = create_model_and_setup(device)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the entire dict is the state dict
            model.load_state_dict(checkpoint)
    else:
        # Assume it's directly the state dict
        model.load_state_dict(checkpoint)
    
    print("✅ Model loaded successfully")
    return model

def load_logs_from_path(log_path):
    """Load training logs from a JSON file."""
    print(f"\n{'='*70}")
    print(f"Loading logs from: {log_path}")
    print(f"{'='*70}")
    
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    # Handle different log formats
    if isinstance(logs, dict):
        # Format: {"epoch_logs": [...], "step_logs": [...]}
        epoch_logs = logs.get('epoch_logs', [])
        step_logs = logs.get('step_logs', [])
    elif isinstance(logs, list):
        # Format: just a list of logs (assume step logs)
        epoch_logs = []
        step_logs = logs
    else:
        print("⚠️  Warning: Unknown log format")
        epoch_logs = []
        step_logs = []
    
    print(f"✅ Loaded {len(epoch_logs)} epoch logs and {len(step_logs)} step logs")
    return epoch_logs, step_logs

def main():
    print(f"{'='*70}")
    print("Testing Training Determinism")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Seed: {SEED}")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    print(f"{'='*70}")
    
    # Get first model and logs
    if MODEL1_PATH:
        model1 = load_model_from_path(MODEL1_PATH, device)
        if EPOCH_LOGS1_PATH:
            epoch_logs1, _ = load_logs_from_path(EPOCH_LOGS1_PATH)
        else:
            epoch_logs1 = None
        if STEP_LOGS1_PATH:
            _, step_logs1 = load_logs_from_path(STEP_LOGS1_PATH)
        else:
            step_logs1 = None
    else:
        model1, epoch_logs1, step_logs1 = train_model("run1")
    
    # Get second model and logs
    if MODEL2_PATH:
        model2 = load_model_from_path(MODEL2_PATH, device)
        if EPOCH_LOGS2_PATH:
            epoch_logs2, _ = load_logs_from_path(EPOCH_LOGS2_PATH)
        else:
            epoch_logs2 = None
        if STEP_LOGS2_PATH:
            _, step_logs2 = load_logs_from_path(STEP_LOGS2_PATH)
        else:
            step_logs2 = None
    else:
        model2, epoch_logs2, step_logs2 = train_model("run2")
    
    # Compare
    models_match = compare_models(model1, model2)
    
    # Compare logs only if both are available
    if epoch_logs1 and epoch_logs2:
        epochs_match = compare_logs(epoch_logs1, epoch_logs2, "epoch")
    else:
        print("\n⚠️  Skipping epoch log comparison (logs not available)")
        epochs_match = None
    
    # Only compare first 100 steps to keep output manageable
    if step_logs1 and step_logs2:
        steps_match = compare_logs(step_logs1[:100], step_logs2[:100], "step")
    else:
        print("\n⚠️  Skipping step log comparison (logs not available)")
        steps_match = None
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    
    all_match = models_match and (epochs_match is not False) and (steps_match is not False)
    
    if all_match:
        print("✅ SUCCESS: Training is fully deterministic!")
        print("   - Model weights match")
        if epochs_match is not None:
            print("   - Epoch logs match")
        if steps_match is not None:
            print("   - Step logs match")
    else:
        print("❌ FAILURE: Training is NOT deterministic")
        print(f"   - Model weights: {'✅ Match' if models_match else '❌ Differ'}")
        if epochs_match is not None:
            print(f"   - Epoch logs: {'✅ Match' if epochs_match else '❌ Differ'}")
        if steps_match is not None:
            print(f"   - Step logs: {'✅ Match' if steps_match else '❌ Differ'}")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
