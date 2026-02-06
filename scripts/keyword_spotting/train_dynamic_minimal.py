import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.keyword_spotting.base_schema import (
    KeyWordSpottingBaseConfig, SpectrogramConfig, BackboneConfig, 
    NoiseConfig, DatasetConfig
)
from thesis_project.models.keyword_spotting.dynamic_schema import (
    RouterConfig, DynamicModelConfig, DataLoaderConfig, 
    LossConfig, TrainingConfig, DynamicTrainingConfig
)
from thesis_project.models.components.routers import GRURouter
from thesis_project.utils.paths import get_data_dir
from thesis_project.utils.compute_macs import compute_macs_base_model, compute_avg_rank_from_macs
from thesis_project.datasets import SpeechCommandsGoogle
from thesis_project.datasets.speech_commands.config import CANONICAL
from thesis_project.training.key_word_spotting import setup_and_train_dynamic_model

# ============================================================================
# DYNAMIC MODEL CONFIGURATION
# ============================================================================

# Create configuration using schemas
router_cfg = RouterConfig(
    hidden_dim=48,
    num_gru_layers=1,
    use_global_rank=True,
    num_rank_outputs=1,
    pool_mode="subsample",
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
    use_subset=False,
    num_workers=0,
    pin_memory=False,
    subset_size=1000,
    do_validation=True,
    val_snr_values=[-5, 0, 5, 10, 15, float('inf')],
)

loss_cfg = LossConfig(
    rank_loss_weight=20,
    rank_loss_mode="avg_rank",
    compressed_base_model_rank_pr_stack=[30, 30, 30],
    base_model_rank_reference=15.0,  # Reference rank for MAC budget computation
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

# Create complete config
full_config = DynamicTrainingConfig(
    seed=42,
    router=router_cfg,
    dynamic_model=dynamic_model_cfg,
    data_loader=data_loader_cfg,
    loss=loss_cfg,
    training=training_cfg,
    noise_train=noise_train_cfg,
    noise_eval=noise_eval_cfg,
)

# Load base model config from saved file
BASE_MODEL_DIR = Path(full_config.dynamic_model.base_model_dir)
base_model_cfg = KeyWordSpottingBaseConfig(**json.loads((BASE_MODEL_DIR / "base_config.json").read_text()))

# Run complete training pipeline
model, epoch_logs, step_logs = setup_and_train_dynamic_model(
    full_config=full_config,
    base_model_cfg=base_model_cfg,
)
