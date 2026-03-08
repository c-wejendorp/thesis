"""
Standard base configuration for dynamic model sweeps.
This configuration is used as a baseline and can be overridden during sweeps.
"""

from thesis_project.models.keyword_spotting.dynamic_schema import (
    RouterConfig, DynamicModelConfig, DataLoaderConfig,
    LossConfig, TrainingConfig
)
from thesis_project.models.keyword_spotting.base_schema import NoiseConfig


def get_sweep_base_config(use_subset: bool = False, subset_size: int = 1000):
    """
    Get the standard base configuration for sweeps.
    
    Args:
        use_subset: Whether to use a subset of data for quick testing
        subset_size: Size of subset when use_subset is True
        
    Returns:
        Dictionary with configuration components that can be used to construct
        DynamicTrainingConfig with sweep-specific overrides
    """
    
    router_cfg = RouterConfig(
        hidden_dim=48,
        num_gru_layers=1,
        use_global_rank=True,  # Will be overridden in sweep
        num_rank_outputs=1,     # Will be overridden in sweep
        pool_mode="avg",
        pool_reduction_factor=2,
        last_layer_bias_init=3.0,
    )
    
    dynamic_model_cfg = DynamicModelConfig(
        freeze_base=True,
        low_rank_frontend=False,
        base_model_dir="model_runs/base/2025-12-14_18-48-50",
    )
    
    data_loader_cfg = DataLoaderConfig(
        batch_size=64,
        use_subset=use_subset,
        num_workers=0,
        pin_memory=False,
        subset_size=subset_size,
        do_validation=True,
        val_snr_values=[-5, 0, 5, 10, 15, float('inf')],
    )
    
    loss_cfg = LossConfig(
        rank_loss_weight=20,  # Will be overridden in sweep
        rank_loss_mode="avg_rank",  # Will be overridden in sweep
        compressed_base_model_rank_pr_stack=[30, 30, 30],
        base_model_rank_reference=15.0,  # Will be overridden in sweep
        enable_rank_supervision=False,
        rank_supervision_stable=False,
    )
    
    training_cfg = TrainingConfig(
        num_epochs=5,
        init_learning_rate=1e-4,
        min_learning_rate=0.5e-4,
        optimizer="Adam",
        scheduler="CosineAnnealingLR",
        scheduler_mode="step",
        log_every_n_steps=1,
        save_epoch_checkpoints=False,
    )
    
    noise_train_cfg = NoiseConfig(
        add_noise=True,
        noise_prob=0.8,
        snr=(-5, 15)
    )
    
    noise_eval_cfg = NoiseConfig(
        add_noise=True,
        noise_prob=1.0,
        snr=None
    )
    
    return {
        'seed': 42,
        'router': router_cfg,
        'dynamic_model': dynamic_model_cfg,
        'data_loader': data_loader_cfg,
        'loss': loss_cfg,
        'training': training_cfg,
        'noise_train': noise_train_cfg,
        'noise_eval': noise_eval_cfg,
    }
