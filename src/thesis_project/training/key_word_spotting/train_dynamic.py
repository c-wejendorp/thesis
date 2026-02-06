"""
Complete training pipeline for dynamic keyword spotting model.
Handles model setup, dataset creation, training execution, and result logging.
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Tuple

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.keyword_spotting.base_schema import KeyWordSpottingBaseConfig
from thesis_project.models.keyword_spotting.dynamic_schema import DynamicTrainingConfig
from thesis_project.models.components.routers import GRURouter
from thesis_project.utils.paths import get_data_dir
from thesis_project.utils.compute_macs import compute_macs_base_model, compute_avg_rank_from_macs
from thesis_project.datasets import SpeechCommandsGoogle
from .train_dynamic_loop import fit_dynamic_model
from .loss_functions import create_dynamic_routing_loss


def setup_and_train_dynamic_model(
    full_config: DynamicTrainingConfig,
    base_model_cfg: KeyWordSpottingBaseConfig,
    base_model_dir: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> Tuple[KWSDynamic, Optional[list], Optional[list]]:
    """
    Complete pipeline for training a dynamic keyword spotting model.
    
    Args:
        full_config: Complete training configuration
        base_model_cfg: Base model configuration
        base_model_dir: Path to base model directory (if None, uses config value)
        run_dir: Custom run directory (if None, creates timestamped directory)
    
    Returns:
        Tuple of (trained_model, epoch_logs, step_logs)
    """
    SEED = full_config.seed
    
    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Setup device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")    
    print(f"Using device: {DEVICE}")
    
    # Determine base model directory
    if base_model_dir is None:
        base_model_dir = Path(full_config.dynamic_model.base_model_dir)
    
    # Load base model
    base_model_path = base_model_dir / "best_model.pth"
    base_model_state = torch.load(base_model_path)
    base_model = KWSBase(base_model_cfg).to(DEVICE)
    base_model.load_state_dict(base_model_state)
    base_model_full_rank = min(base_model.cfg.backbone.n_channels_ext, base_model.cfg.backbone.n_channels_int)
    maximum_useful_rank = base_model_full_rank // 2
    
    # Validate router config against base model
    assert (full_config.router.use_global_rank and full_config.router.num_rank_outputs == 1) or \
           (not full_config.router.use_global_rank and full_config.router.num_rank_outputs >= 1)
    assert full_config.router.num_rank_outputs == len(base_model.backbone) or full_config.router.use_global_rank, \
           "NUM_RANK_OUTPUTS must match number of stacks in base model when USE_GLOBAL_RANK is False"
    
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
    ).to(DEVICE)
    
    n_trainable = sum(p.numel() for p in dynamic_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in dynamic_model.parameters())
    print(f"Total params: {total_params}")
    print(f"Trainable params: {n_trainable}")
    
    # Create dataset and loaders
    train_set = SpeechCommandsGoogle(
        root=str(get_data_dir()), 
        subset="training", 
        download=True, 
        seed=SEED, 
        **full_config.noise_train.model_dump()
    )
    if full_config.data_loader.use_subset:
        rng = np.random.default_rng(SEED)
        train_set = torch.utils.data.Subset(
            train_set, 
            rng.choice(len(train_set), full_config.data_loader.subset_size, replace=False) #type: ignore
        )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=full_config.data_loader.batch_size, 
        shuffle=True, 
        pin_memory=full_config.data_loader.pin_memory, 
        num_workers=full_config.data_loader.num_workers,
        generator=torch.Generator().manual_seed(SEED)
    )
    
    if full_config.data_loader.do_validation:
        val_set = SpeechCommandsGoogle(
            root=str(get_data_dir()), 
            subset="validation", 
            download=True, 
            seed=SEED, 
            **full_config.noise_eval.model_dump()
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=full_config.data_loader.batch_size, 
            shuffle=False, 
            pin_memory=full_config.data_loader.pin_memory, 
            num_workers=full_config.data_loader.num_workers
        )
    else: 
        val_loader = None
    
    # Validate loss config against base model
    assert len(full_config.loss.compressed_base_model_rank_pr_stack) == len(base_model.backbone), \
           "compressed_base_model_rank_pr_stack must match number of stacks in base model backbone"
    
    # Compute MACs for compressed base model (reference point for comparison)
    macs_compressed_base_model = compute_macs_base_model(
        base_model, 
        time_steps=63, 
        rank_pr_stack=full_config.loss.compressed_base_model_rank_pr_stack
    )
    macs_router = router.compute_macs(sequence_length=63)
    breakeven_macs = macs_compressed_base_model - macs_router
    breakeven_avg_rank = compute_avg_rank_from_macs(
        base_model, 
        time_steps=63, 
        macs=breakeven_macs,
        num_stacks=len(base_model.backbone)
    )

    # Compute target MACs based on base_model_rank_reference
    # This represents the total MAC budget (matching base model at reference rank)
    macs_base_at_reference = compute_macs_base_model(
        base_model,
        time_steps=63,
        rank_pr_stack=[int(full_config.loss.base_model_rank_reference)] * len(base_model.backbone)
    )
    
    # After accounting for router overhead, compute achievable average rank for dynamic model
    macs_low_rank_budget = int(macs_base_at_reference - macs_router)
    target_avg_rank = compute_avg_rank_from_macs(
        base_model, 
        time_steps=63, 
        macs=macs_low_rank_budget,
        num_stacks=len(base_model.backbone)
    )


    
    target_avg_rank_normalized = (target_avg_rank - 1) / (maximum_useful_rank - 1)
    print(f"\nMACs Breakdown:")
    print(f"  Compressed base model: {macs_compressed_base_model}")
    print(f"  Router overhead: {macs_router}")
    print(f"  Breakeven MACs (compressed base - router): {breakeven_macs}")
    print(f"  Breakeven average rank: {breakeven_avg_rank:.2f}")
    print(f"\nTarget Configuration (base_model_rank_reference={full_config.loss.base_model_rank_reference}):")
    print(f"  Target total budget (= base at reference rank): {macs_base_at_reference:.2f}")
    print(f"  Target budget for dynamic model (after router overhead): {macs_low_rank_budget}")
    print(f"  Target budget for dynamic model in ranks: {target_avg_rank:.2f}")
    print(f"  Target budget for dynamic model in ranks (normalized): {target_avg_rank_normalized:.3f}\n")
    
    # Create loss criterion
    criterion = create_dynamic_routing_loss(
        rank_loss_mode=full_config.loss.rank_loss_mode,
        rank_loss_weight=full_config.loss.rank_loss_weight,
        target_rank_normalized=target_avg_rank_normalized
    )
    
    # Setup optimizer
    optimizer_class = getattr(torch.optim, full_config.training.optimizer)
    optimizer = optimizer_class(
        dynamic_model.parameters(), 
        lr=full_config.training.init_learning_rate
    )
    
    # Setup scheduler (optional)
    if full_config.training.scheduler is None:
        scheduler = None
        print("No scheduler - using constant learning rate")
    else:
        # Schema validation ensures min_learning_rate is not None when scheduler is set
        if full_config.training.scheduler_mode == "step":
            steps_per_epoch = len(train_loader)
            total_steps = full_config.training.num_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=full_config.training.min_learning_rate,  # type: ignore
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=full_config.training.num_epochs,
                eta_min=full_config.training.min_learning_rate,  # type: ignore
            )
    
    # Create run directory
    if run_dir is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(f"model_runs/dynamic/{run_timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save configuration schema directly
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(full_config.model_dump(), f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Save runtime information separately
    runtime_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "device": str(DEVICE),
        "base_model_info": {
            "full_rank": base_model_full_rank,
            "maximum_useful_rank": maximum_useful_rank,
            "n_stacks": len(base_model.backbone),
            "n_channels_ext": base_model.cfg.backbone.n_channels_ext,
            "n_channels_int": base_model.cfg.backbone.n_channels_int,
            "kernel_size": base_model.cfg.backbone.kernel_size,
            "n_blocks_pr_stack": base_model.cfg.backbone.n_blocks_pr_stack,
        },
        "model_params": {
            "total_params": total_params,
            "trainable_params": n_trainable,
        },
        "macs": {
            "compressed_base_model": float(macs_compressed_base_model),
            "router": float(macs_router),
            "breakeven_macs": float(breakeven_macs),
            "breakeven_avg_rank": float(breakeven_avg_rank),
            "base_at_reference_rank": float(macs_base_at_reference),
            "base_model_rank_reference": float(full_config.loss.base_model_rank_reference),
            "low_rank_budget": float(macs_low_rank_budget),
            "achievable_avg_rank": float(target_avg_rank),
            "achievable_avg_rank_normalized": float(target_avg_rank_normalized),
        },
    }
    runtime_path = run_dir / "runtime_info.json"
    with open(runtime_path, 'w') as f:
        json.dump(runtime_info, f, indent=2)
    print(f"Runtime information saved to: {runtime_path} \n")
    
    # Train model
    model, epoch_logs, step_logs = fit_dynamic_model(
        model=dynamic_model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epochs=full_config.training.num_epochs,
        scheduler=scheduler,
        run_dir=str(run_dir),
        max_rank=maximum_useful_rank,
        val_loader=val_loader,
        val_snr_values=full_config.data_loader.val_snr_values if full_config.data_loader.do_validation else None,
        val_epoch=1,
        enable_rank_supervision=full_config.loss.enable_rank_supervision,
        rank_supervision_stable=full_config.loss.rank_supervision_stable,
        save_epoch_checkpoints=full_config.training.save_epoch_checkpoints,
        log_every_n_steps=full_config.training.log_every_n_steps,
        scheduler_mode=full_config.training.scheduler_mode,
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {run_dir}")
    
    # Print final metrics
    if epoch_logs is not None and len(epoch_logs) > 0:
        last_epoch = epoch_logs[-1]
        print(f"\nFinal metrics:")
        print(f"  Train loss: {last_epoch.get('train_loss'):.4f}   Train acc: {last_epoch.get('train_acc'):.2f}%")
        print(f"  Best train loss: {last_epoch.get('best_train_loss'):.4f}   Acc at best: {last_epoch.get('best_train_acc'):.2f}%")
        if last_epoch.get('val_loss') is not None:
            print(f"  Val loss: {last_epoch.get('val_loss'):.4f}   Val acc: {last_epoch.get('val_acc'):.2f}%")
            print(f"  Best val loss: {last_epoch.get('best_val_loss'):.4f}   Acc at best: {last_epoch.get('best_val_acc'):.2f}%")
        print(f"  Expected rank: {last_epoch.get('train_expected_rank'):.1f}")
    
    if step_logs is not None and len(step_logs) > 0:
        print(f"\nStep-level logs saved: {len(step_logs)} steps tracked")
    
    return model, epoch_logs, step_logs
