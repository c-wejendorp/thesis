import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Any, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from thesis_project.utils.paths import create_run_folder
import numpy as np
import os
from tqdm import tqdm


@dataclass
class DynamicRoutingLossOutput:
    """
    Structured output for dynamic routing losses.

    Attributes:
        loss:
            Total scalar loss used for backprop.

        ce_loss:
            Cross-entropy classification loss (detached).

        rank_loss:
            Rank / efficiency loss term (detached, unscaled).
        
        rank_loss_weighted:
            Rank loss scaled by rank_loss_weight (detached).
            This is the actual contribution to total loss.
            
        batch_mean_rank_normalized:
            Batch mean of per-sample average normalized ranks.
            The forward() method receives r_normalized_batch with shape
            (B, num_stacks) containing multiple ranks per sample, averages
            across stacks to get (B,) per-sample averages, then computes
            the batch mean: mean(mean(r_normalized_batch, dim=-1)).
            Multiply by max_rank to get expected rank.

        rank_variance:
            Variance of r_normalized_batch (detached).
            Useful to detect router collapse.
    """
    loss: torch.Tensor
    ce_loss: torch.Tensor
    rank_loss: torch.Tensor
    rank_loss_weighted: torch.Tensor
    batch_mean_rank_normalized: torch.Tensor
    rank_variance: torch.Tensor

class DynamicRoutingLoss(nn.Module, ABC):
    """
    Base class for dynamic routing losses with CE + rank components.
    
    Total loss:
        L_total = L_ce + λ_rank * L_rank
    
    where:
        - L_ce: cross-entropy classification loss
        - L_rank: rank/efficiency loss (subclass-specific)
        - λ_rank: rank_loss_weight
    """
    
    def __init__(
        self,
        *,
        rank_loss_weight: float = 1.0,
        class_weight: Optional[torch.Tensor] = None,
        target_rank_normalized: Optional[float] = None,
    ):
        """
        Args:
            rank_loss_weight:
                Weight applied to L_rank. Can be annealed during training.
            
            class_weight:
                Optional class weights for cross-entropy, shape (C,).
            
            target_rank_normalized:
                Default target normalized rank in [0,1].
                Required for losses with REQUIRES_TARGET = True (unless
                overridden per-batch via rank_supervision in forward).
        """
        super().__init__()
        self.rank_loss_weight = float(rank_loss_weight)
        self.target_rank_normalized = target_rank_normalized
        
        # Register CE weights so they follow device placement
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None
    
    @abstractmethod
    def compute_rank_loss(
        self,
        r_normalized_batch: torch.Tensor,
        target_rank_normalized: Optional[float],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the rank loss component. Must be implemented by subclasses.
        
        Args:
            r_normalized_batch: (B,) per-sample average normalized rank in [0,1].
                This has already been averaged from the original (B, num_stacks)
                input by the forward() method. Each element is the mean rank
                across all stacks for that sample.
            target_rank_normalized: Target normalized rank in [0,1].
                This is either self.target_rank_normalized from init or the
                rank_supervision override from forward().
            **kwargs: Additional arguments such as:
                - ce_loss_per_sample: (B,) per-sample CE loss (optional)
            
        Returns:
            Scalar rank loss tensor
        """
        pass
    
    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        r_normalized_batch: torch.Tensor,
        rank_supervision: Optional[float] = None,
        **kwargs
    ) -> DynamicRoutingLossOutput:
        """
        Args:
            logits: (B, C) classification logits
            y: (B,) int64 class labels
            r_normalized_batch: (B, num_stacks) normalized ranks in [0,1].
                Each sample has multiple ranks (one per stack). This function
                averages them across stacks (dim=-1) internally before computing
                loss. Also accepts (B,) if already averaged.
            rank_supervision: Optional per-batch target override in [0,1].
                If provided, overrides self.target_rank_normalized for this batch.
                If None, uses self.target_rank_normalized from init.
            **kwargs: Additional arguments passed to compute_rank_loss
            
        Returns:
            DynamicRoutingLossOutput with loss components
        """
        # Determine effective target (supervision overrides default)
        effective_target = rank_supervision if rank_supervision is not None else self.target_rank_normalized
        
        # Validate target requirement
        if getattr(self.__class__, 'REQUIRES_TARGET', False):
            if effective_target is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires target_rank_normalized to be provided "
                    f"(either at init or via rank_supervision parameter)"
                )
        
        # Average ranks across stacks if multiple ranks per sample
        if r_normalized_batch.dim() == 2:
            # Shape: (B, num_stacks) -> (B,)
            r_normalized_batch = r_normalized_batch.mean(dim=-1)
        elif r_normalized_batch.dim() != 1:
            raise ValueError(
                f"Expected r_normalized_batch to have shape (B,) or (B, num_stacks), "
                f"got shape {r_normalized_batch.shape}"
            )
        # Numerical safety
        r_normalized_batch = r_normalized_batch.clamp(0.0, 1.0)
        # Classification loss
        ce_loss_per_sample = F.cross_entropy(
            logits, y, weight=self.class_weight, reduction="none"
        )
        ce_loss = ce_loss_per_sample.mean()
        
        # Rank loss (subclass-specific)
        rank_loss = self.compute_rank_loss(
            r_normalized_batch,
            effective_target,
            ce_loss_per_sample=ce_loss_per_sample,
            **kwargs
        )
        
        # Statistics
        batch_mean_rank = r_normalized_batch.mean()
        rank_variance = r_normalized_batch.var(unbiased=False)
        
        rank_loss_weighted = self.rank_loss_weight * rank_loss
        total_loss = ce_loss + rank_loss_weighted
        
        return DynamicRoutingLossOutput(
            loss=total_loss,
            ce_loss=ce_loss.detach(),
            rank_loss=rank_loss.detach(),
            rank_loss_weighted=rank_loss_weighted.detach(),
            batch_mean_rank_normalized=batch_mean_rank.detach(),
            rank_variance=rank_variance.detach(),
        )


class BatchMeanMSELoss(DynamicRoutingLoss):
    """
    Controls average compute budget: (mean(r) - target)^2
    Enforces a global compute budget across the batch.
    
    Requires 'target_rank_normalized' in forward kwargs.
    """
    
    REQUIRED_PARAMS = []
    REQUIRES_TARGET = True
    
    def compute_rank_loss(self, r_normalized_batch, target_rank_normalized, **kwargs):
        return (r_normalized_batch.mean() - target_rank_normalized).pow(2) # type: ignore


class PerSampleMSELoss(DynamicRoutingLoss):
    """
    Forces each sample toward target: mean((r - target)^2)
    Enforces per-sample closeness to target (discourages dynamic spread).
    
    Requires 'target_rank_normalized' in forward kwargs.
    """
    
    REQUIRED_PARAMS = []
    REQUIRES_TARGET = True
    
    def compute_rank_loss(self, r_normalized_batch, target_rank_normalized, **kwargs):
        return (r_normalized_batch - target_rank_normalized).pow(2).mean() # type: ignore

class AvgRankLoss(DynamicRoutingLoss):
    """
    Directly minimize average rank (no target).
    Pushes toward using the lowest ranks possible.
    """
    
    REQUIRED_PARAMS = []
    REQUIRES_TARGET = False
    
    def compute_rank_loss(self, r_normalized_batch, target_rank_normalized, **kwargs):
        return r_normalized_batch.mean()
    
class OneSidedMSELoss(DynamicRoutingLoss):
    """
    Only penalizes ranks above target (no penalty below).
    
    Allows the model to use lower ranks freely while preventing
    it from exceeding the target.
    
    Requires 'target_rank_normalized' in forward kwargs.
    """
    
    REQUIRED_PARAMS = []
    REQUIRES_TARGET = True
    
    def compute_rank_loss(self, r_normalized_batch, target_rank_normalized, **kwargs):
        error = r_normalized_batch - target_rank_normalized # type: ignore
        return F.relu(error).pow(2).mean()


class RiccardoSpecialLoss(DynamicRoutingLoss):
    """
    CE-weighted rank loss: samples with higher CE get more rank penalty.
    Encourages the model to use more compute on harder samples.
    """
    
    REQUIRED_PARAMS = []
    REQUIRES_TARGET = False

    def compute_rank_loss(self, r_normalized_batch, target_rank_normalized, **kwargs):
        ce_loss_per_sample = kwargs["ce_loss_per_sample"]
        # Normalize CE losses to sum to batch size
        weights = ce_loss_per_sample.detach() / ce_loss_per_sample.detach().sum()
        weights *= len(weights)
        return (weights * r_normalized_batch).mean()


class AsymmetricMSELoss(DynamicRoutingLoss):
    """
    Penalizes overshooting target more than undershooting.
    
    Uses asymmetric_alpha to control the asymmetry:
        - alpha * error^2 if error > 0 (overshooting)
        - error^2 if error <= 0 (undershooting)
    
    This encourages the distribution tail towards lower ranks.
    
    Requires 'target_rank_normalized' in forward kwargs.
    """
    
    REQUIRED_PARAMS = ["asymmetric_alpha"]
    REQUIRES_TARGET = True
    
    def __init__(self, *, asymmetric_alpha: float, **kwargs):
        """
        Args:
            asymmetric_alpha:
                Multiplier for penalty when rank exceeds target.
                Example: 2.0 means overshooting is penalized 2x more.
        """
        super().__init__(**kwargs)
        self.asymmetric_alpha = float(asymmetric_alpha)
    
    def compute_rank_loss(self, r_normalized_batch, target_rank_normalized, **kwargs):
        error = r_normalized_batch - target_rank_normalized # type: ignore
        return torch.where(
            error > 0,
            self.asymmetric_alpha * error.pow(2),
            error.pow(2),
        ).mean()

class CEGatedLoss(DynamicRoutingLoss):
    """
    CE-gated compute penalty: apply rank pressure only when CE is acceptable.
    
    L = CE + λ * g(CE) * r
    
    where g(CE) = sigmoid((threshold - CE) * smoothness) is a smooth gate:
        - g ≈ 1 when CE < threshold (good) → apply rank penalty
        - g ≈ 0 when CE > threshold (bad) → no rank penalty
    
    This encourages compute efficiency only when accuracy is safe.
    """
    
    REQUIRED_PARAMS = ["ce_gate_threshold", "ce_gate_smoothness"]
    REQUIRES_TARGET = False
    
    def __init__(
        self,
        *,
        ce_gate_threshold: float,
        ce_gate_smoothness: float,
        **kwargs
    ):
        """
        Args:
            ce_gate_threshold:
                Threshold τ for CE gating. CE values below this are
                considered "good" (gate opens). Example: 0.5.
            
            ce_gate_smoothness:
                Controls steepness of sigmoid gate. Higher = sharper
                transition. Example: 10.0.
        """
        super().__init__(**kwargs)
        self.ce_gate_threshold = float(ce_gate_threshold)
        self.ce_gate_smoothness = float(ce_gate_smoothness)
    
    def compute_rank_loss(self, r_normalized_batch, target_rank_normalized, **kwargs):
        ce_loss_per_sample = kwargs["ce_loss_per_sample"]
        gate = torch.sigmoid(
            (self.ce_gate_threshold - ce_loss_per_sample.detach()) 
            * self.ce_gate_smoothness
        )
        return (gate * r_normalized_batch).mean()


def create_dynamic_routing_loss(
    rank_loss_mode: str,
    rank_loss_weight: float = 1.0,
    target_rank_normalized: Optional[float] = None,
    **mode_specific_kwargs
) -> DynamicRoutingLoss:
    """
    Factory function to create loss instances based on mode string.
    
    Args:
        rank_loss_mode: Loss type identifier
        rank_loss_weight: Weight for rank loss component
        target_rank_normalized: Default target normalized rank in [0,1].
            Required for losses with REQUIRES_TARGET = True.
        **mode_specific_kwargs: Additional arguments for specific loss types
        
    Returns:
        Instance of appropriate DynamicRoutingLoss subclass
        
    Raises:
        ValueError: If rank_loss_mode is unknown
        TypeError: If required parameters are missing for the selected loss
        
    Example:
        >>> loss = create_dynamic_routing_loss(
        ...     "asymmetric_mse",
        ...     rank_loss_weight=1.0,
        ...     target_rank_normalized=0.25,
        ...     asymmetric_alpha=2.0
        ... )
    """
    loss_classes = {
        "batch_mean_mse": BatchMeanMSELoss,
        "per_sample_mse": PerSampleMSELoss,
        "avg_rank": AvgRankLoss,
        "riccardo_special": RiccardoSpecialLoss,
        "asymmetric_mse": AsymmetricMSELoss,
        "one_sided_mse": OneSidedMSELoss,
        "ce_gated": CEGatedLoss,
    }
    
    if rank_loss_mode not in loss_classes:
        raise ValueError(
            f"Unknown rank_loss_mode={rank_loss_mode}. "
            f"Available modes: {list(loss_classes.keys())}"
        )
    
    loss_class = loss_classes[rank_loss_mode]
    
    # Validate required parameters
    if hasattr(loss_class, 'REQUIRED_PARAMS'):
        missing_params = [
            param for param in loss_class.REQUIRED_PARAMS
            if param not in mode_specific_kwargs
        ]
        if missing_params:
            raise TypeError(
                f"Loss '{rank_loss_mode}' requires the following parameters: "
                f"{loss_class.REQUIRED_PARAMS}. Missing: {missing_params}"
            )
    
    return loss_class(
        rank_loss_weight=rank_loss_weight,
        target_rank_normalized=target_rank_normalized,
        **mode_specific_kwargs
    )


