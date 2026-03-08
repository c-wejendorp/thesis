from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
from .base_schema import NoiseConfig


class RouterConfig(BaseModel):
    """Configuration for the router network that predicts ranks."""
    hidden_dim: int = Field(..., description="Hidden dimension for GRU layers")
    num_gru_layers: int = Field(..., ge=1, description="Number of GRU layers")
    use_global_rank: bool = Field(..., description="Whether to use a single global rank for all stacks")
    num_rank_outputs: int = Field(..., ge=1, description="Number of rank outputs (should match number of stacks if not using global rank)")
    pool_mode: Optional[Literal["avg", "max", "subsample"]] = Field(
        None, 
        description="Pooling mode for temporal dimension. None means no pooling"
    )
    pool_reduction_factor: int = Field(
        ..., 
        ge=1, 
        description="Reduction factor for pooling (e.g., 2 means pick every 2nd element for subsample)"
    )
    last_layer_bias_init: Optional[float] = Field(
        None, 
        description="Bias initialization for last layer (positive values bias towards higher ranks)"
    )
    
    @model_validator(mode='after')
    def validate_rank_outputs(self):
        """Validate that num_rank_outputs is 1 when using global rank."""
        if self.use_global_rank and self.num_rank_outputs != 1:
            raise ValueError("num_rank_outputs must be 1 when use_global_rank is True")
        return self


class DynamicModelConfig(BaseModel):
    """Configuration for the dynamic model wrapper."""
    freeze_base: bool = Field(..., description="Whether to freeze base model parameters during training")
    low_rank_frontend: bool = Field(..., description="Whether to apply low-rank decomposition to frontend")
    base_model_dir: str = Field(..., description="Path to the base model run directory (contains best_model.pth and config files)")


class DataLoaderConfig(BaseModel):
    """Configuration for data loading."""
    batch_size: int = Field(..., ge=1, description="Batch size for training and validation")
    pin_memory: bool = Field(False, description="Whether to pin memory for faster GPU transfer")
    num_workers: int = Field(0, ge=0, description="Number of worker processes for data loading")
    use_subset: bool = Field(..., description="Whether to use a subset of data for quick testing")
    subset_size: int = Field(..., ge=1, description="Number of samples in subset if use_subset is True")
    do_validation: bool = Field(..., description="Whether to perform validation during training")
    val_snr_values: list[float] = Field(
        ...,
        description="SNR values to evaluate at during validation"
    )


class LossConfig(BaseModel):
    """Configuration for the loss function."""
    rank_loss_weight: float = Field(..., ge=0.0, description="Weight for rank regularization loss")
    rank_loss_mode: Literal[
        "batch_mean_mse", 
        "per_sample_mse", 
        "avg_rank", 
        "ce_weighted_rank", 
        "asymmetric_mse", 
        "one_sided_mse", 
        "ce_gated"
    ] = Field(
        ..., 
        description="Mode for rank loss computation"
    )
    compressed_base_model_rank_pr_stack: list[int] = Field(
        ...,
        description="Rank per stack for a compressed base model used as a comparison reference point (not the target)"
    )
    base_model_rank_reference: float = Field(
        ..., 
        gt=0.0,
        description="Reference rank for computing MAC budget. Total MACs = base_model_at_this_rank. After router overhead, the achievable dynamic model rank will be lower."
    )
    enable_rank_supervision: bool = Field(
        ..., 
        description="Whether to enable rank supervision (find minimum correct rank)"
    )
    rank_supervision_stable: bool = Field(
        ..., 
        description="If True, require prediction to be correct at all higher ranks too"
    )
    
    @model_validator(mode='after')
    def validate_rank_supervision(self):
        """Validate that rank_supervision_stable is only used with enable_rank_supervision."""
        if self.rank_supervision_stable and not self.enable_rank_supervision:
            raise ValueError(
                "rank_supervision_stable can only be True when enable_rank_supervision is True"
            )
        return self


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters."""
    num_epochs: int = Field(..., ge=1, description="Number of training epochs")
    init_learning_rate: float = Field(..., gt=0.0, description="Initial learning rate")
    min_learning_rate: float | None = Field(None, description="Minimum learning rate for scheduler (not used if scheduler is None)")
    optimizer: str = Field(..., description="Optimizer name")
    scheduler: str | None = Field(None, description="Learning rate scheduler name (None for constant learning rate)")
    scheduler_mode: Literal["epoch", "step"] | None = Field(
        None, 
        description="When to step the scheduler: 'epoch' or 'step' (not used if scheduler is None)"
    )
    log_every_n_steps: Optional[int] = Field(
        ..., 
        ge=1, 
        description="Log metrics every N steps. None to disable step logging"
    )
    save_epoch_checkpoints: bool = Field(
        ..., 
        description="Whether to save model checkpoints after each epoch"
    )
    
    @field_validator('min_learning_rate')
    @classmethod
    def validate_min_lr(cls, v, info):
        """Ensure min_learning_rate is less than init_learning_rate when both are set."""
        if v is not None and 'init_learning_rate' in info.data and v >= info.data['init_learning_rate']:
            raise ValueError("min_learning_rate must be less than init_learning_rate")
        return v
    
    @model_validator(mode='after')
    def validate_scheduler_config(self):
        """Ensure scheduler-related fields are set consistently."""
        if self.scheduler is not None:
            if self.scheduler_mode is None:
                raise ValueError("scheduler_mode must be set when scheduler is not None")
            if self.min_learning_rate is None:
                raise ValueError("min_learning_rate must be set when scheduler is not None")
        return self


class DynamicTrainingConfig(BaseModel):
    """Complete configuration for training a dynamic keyword spotting model."""
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    router: RouterConfig
    dynamic_model: DynamicModelConfig
    data_loader: DataLoaderConfig
    loss: LossConfig
    training: TrainingConfig
    noise_train: NoiseConfig = Field(..., description="Noise configuration for training")
    noise_eval: NoiseConfig = Field(..., description="Noise configuration for evaluation")
    
    @model_validator(mode='after')
    def validate_compressed_ranks_match_stacks(self):
        """
        Note: This validation would require knowing the number of stacks from the base model.
        This check should be done at runtime when the base model is loaded.
        """
        # Could add runtime validation here if base model info is passed
        return self
