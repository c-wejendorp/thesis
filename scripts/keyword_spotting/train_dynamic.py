import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from itertools import product
from datetime import datetime
from pathlib import Path
import json

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.components.routers import GRURouter
from thesis_project.utils.paths import get_data_dir
from thesis_project.datasets import SpeechCommandsGoogle
from base_config_temp import cfg, noise_train_cfg, noise_eval_cfg
from thesis_project.training.key_word_spotting import CrossEntropyPlusRankLoss, fit_dynamic_model, validate_dynamic_model
from thesis_project.models.components.layers import LowRankPointwiseConv1d

#TODO: Should not be hardcoded here
# Training configuration
batch_size = 64
num_epochs = 5  # Reduced for faster initial sweep
init_learning_rate = 1e-4
min_learning_rate = 0.5e-4
maximum_useful_rank = 64

val_snr_values = [-5, 0, 5, 10, 15, float('inf')]

# PHASE 1: Coarse grid search
# Hyperparameter grid - sparse sampling for initial exploration
target_ranks = [20,24,27]  # Representative sample
#rank_loss_weights = [9.0, 13.0, 17.0]  # Low, medium, high
rank_loss_weights = [1.0,5.0]  # Low, medium, high
#rank_loss_weights = [11.0]  # Low, medium, high
#rank_loss_modes = ["avg_rank", "batch_mean_mse", "per_sample_mse", "one_sided_mse"]  # All modes
rank_loss_modes = ["ce_gated"]

#pool_modes = [None, "avg"]  # No pooling and average pooling
pool_modes = ["avg"]  # No pooling and average pooling
pool_kernel_sizes = [2]  # Kernel size for pooling (only used when pool_mode is not None)

torch.manual_seed(42)
np.random.seed(42)

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    pin_memory = True
    num_workers = 16
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    pin_memory = False
    num_workers = 0
    print("Using MPS")
else:
    device = torch.device("cpu")
    pin_memory = False
    print("Using CPU")

# Datasets and dataloaders
data_dir = get_data_dir()
train_set = SpeechCommandsGoogle(root=str(data_dir), subset="training", download=True, **noise_train_cfg.model_dump())

# Use 10k subset for faster iteration
#indices = np.random.choice(len(train_set), size=100, replace=False)
#train_set = torch.utils.data.Subset(train_set, indices)  # type: ignore

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

#val_set = SpeechCommandsGoogle(root=str(data_dir), subset="validation", download=True, **noise_eval_cfg.model_dump())
#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

# Load base model once
base_model_state = torch.load("model_runs/base/2025-12-14_18-48-50/best_model.pth")

# Create timestamped sweep folder
sweep_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sweep_dir = Path(f"model_runs/dynamic/{sweep_timestamp}_sweep")
sweep_dir.mkdir(parents=True, exist_ok=True)
print(f"Sweep directory: {sweep_dir}")

# Calculate total runs
total_runs = len(target_ranks) * len(rank_loss_weights) * len(rank_loss_modes) * len(pool_modes) * len(pool_kernel_sizes)
print(f"Total runs: {total_runs}")
print(f"Estimated time: {total_runs * num_epochs * 3 / 60:.1f} hours\n")

# Track results for all runs
sweep_results = []

# Loop over hyperparameter combinations
for idx, (target_rank, rank_loss_weight, rank_loss_mode, pool_mode, pool_kernel_size) in enumerate(product(target_ranks, rank_loss_weights, rank_loss_modes, pool_modes, pool_kernel_sizes)):
    print(f"\n{'='*80}")
    print(f"Run {idx+1}/{total_runs}: target_rank={target_rank}, rank_loss_weight={rank_loss_weight}, rank_loss_mode={rank_loss_mode}, pool_mode={pool_mode}, pool_kernel_size={pool_kernel_size}")
    print(f"{'='*80}\n")
    
    # Normalize target rank
    target_rank_normalized = (target_rank - 1) / (maximum_useful_rank - 1)
    
    # Create fresh model for each run
    base_model = KWSBase(cfg).to(device)
    base_model.load_state_dict(base_model_state)
    
    router = GRURouter(
        input_dim=base_model.frontend.out_channels,
        gru_hidden_dim=48,
        num_gru_layers=1,
        max_rank=maximum_useful_rank,
        num_rank_outputs=3,
        last_layer_bias_init=None,
        pool_type=pool_mode,
        pool_kernel_size=pool_kernel_size
    ).to(device)
    
    model = KWSDynamic(base=base_model, routers=nn.ModuleList([router]), freeze_base=True).to(device)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable}")
    
    # Loss function and optimizer
    criterion = CrossEntropyPlusRankLoss(
        target_rank_normalized=target_rank_normalized,
        rank_loss_weight=rank_loss_weight,
        rank_loss_mode=rank_loss_mode,
        asymmetric_alpha=0.0,
        rank_var_weight=0,
        ce_gate_threshold=0.1,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_learning_rate,
    )
    
    # Train model
    pool_suffix = f"_pool_{pool_mode}_kernel_{pool_kernel_size}" if pool_mode is not None else ""
    run_name = f"rank_{target_rank}_weight_{rank_loss_weight}_mode_{rank_loss_mode}{pool_suffix}"
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = sweep_dir / f"{run_timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model, history = fit_dynamic_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_snr_values=val_snr_values,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=num_epochs,
            scheduler=scheduler,
            run_dir=str(run_dir),
            max_rank=maximum_useful_rank,
            enable_rank_supervision=False,
        )
        
        # Track results - extract from epochs_log structure
        epochs_log = history.get("epochs_log", [])
        if epochs_log:
            last_epoch = epochs_log[-1]
            val_accs = [e.get("val_acc") for e in epochs_log if e.get("val_acc") is not None]
            val_losses = [e.get("val_loss") for e in epochs_log if e.get("val_loss") is not None]
            
            # Find epochs with best metrics
            best_val_acc_idx = val_accs.index(max(val_accs)) if val_accs else None
            best_val_loss_idx = val_losses.index(min(val_losses)) if val_losses else None
            
            result = {
                "target_rank": target_rank,
                "rank_loss_weight": rank_loss_weight,
                "rank_loss_mode": rank_loss_mode,
                "pool_mode": pool_mode,
                "pool_kernel_size": pool_kernel_size,
                "final_train_loss": last_epoch.get("train_loss"),
                "final_train_acc": last_epoch.get("train_acc"),
                "best_val_acc": max(val_accs) if val_accs else None,
                "best_val_acc_loss": epochs_log[best_val_acc_idx].get("val_loss") if best_val_acc_idx is not None else None,
                "best_val_acc_rank": epochs_log[best_val_acc_idx].get("train_expected_rank") if best_val_acc_idx is not None else None,
                "best_val_loss": min(val_losses) if val_losses else None,
                "best_val_loss_acc": epochs_log[best_val_loss_idx].get("val_acc") if best_val_loss_idx is not None else None,
                "best_val_loss_rank": epochs_log[best_val_loss_idx].get("train_expected_rank") if best_val_loss_idx is not None else None,
                "final_epoch_rank": last_epoch.get("train_expected_rank"),
                "run_dir": str(run_dir),
            }
        else:
            result = {
                "target_rank": target_rank,
                "rank_loss_weight": rank_loss_weight,
                "rank_loss_mode": rank_loss_mode,
                "pool_mode": pool_mode,
                "pool_kernel_size": pool_kernel_size,
                "error": "No epochs logged",
                "run_dir": str(run_dir),
            }
        sweep_results.append(result)
        
        # Save intermediate results after each run
        with open(sweep_dir / "sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2)
        
        print(f"✓ Completed training for {run_name}")
        if result['final_train_acc']:
            print(f"  Final train acc: {result['final_train_acc']:.2f}%")
        if result['best_val_acc']:
            print(f"  Best val acc: {result['best_val_acc']:.2f}%")
        if result['best_val_loss']:
            print(f"  Best val loss: {result['best_val_loss']:.4f}")
        if result['final_epoch_rank']:
            print(f"  Final epoch rank: {result['final_epoch_rank']:.1f}")
        if result['best_val_acc_rank']:
            print(f"  Best val acc rank: {result['best_val_acc_rank']:.1f}")
        if result['best_val_loss_rank']:
            print(f"  Best val loss rank: {result['best_val_loss_rank']:.1f}")
        
    except Exception as e:
        print(f"✗ Failed training for {run_name}: {str(e)}")
        sweep_results.append({
            "target_rank": target_rank,
            "rank_loss_weight": rank_loss_weight,
            "rank_loss_mode": rank_loss_mode,
            "pool_mode": pool_mode,
            "pool_kernel_size": pool_kernel_size,
            "error": str(e),
            "run_dir": str(run_dir),
        })
        continue
    
    # Clean up GPU memory
    del model, base_model, router, optimizer, scheduler, criterion
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Final summary
print("\n" + "="*80)
print("SWEEP COMPLETED!")
print("="*80)

