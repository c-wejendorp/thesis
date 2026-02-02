from typing import Sequence, Optional
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from .base_model import KWSBase
from thesis_project.models.components.routers import GRURouter
from typing import Sequence, Optional
from thesis_project.models.components.layers import LowRankPointwiseConv1d

class KWSDynamic(nn.Module):
    def __init__(
        self,
        base: KWSBase,
        router: nn.Module,          # Single router placed after frontend
        *,
        use_global_rank: bool = True,
        low_rank_frontend: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.router = router
        self.use_global_rank = use_global_rank
        self.freeze_base = freeze_base
        
        # Check that router's num_rank_outputs doesn't exceed num_stacks
        n_stacks = len(self.base.backbone)
        
        # Router must have num_rank_outputs attribute
        if not hasattr(router, 'num_rank_outputs'):
            raise ValueError(
                f"Router must have 'num_rank_outputs' attribute. "
                f"Got router of type {type(router).__name__}."
            )
        
        if router.num_rank_outputs > n_stacks:
            raise ValueError(
                f"Router num_rank_outputs ({router.num_rank_outputs}) cannot exceed "
                f"number of stacks ({n_stacks})."
            )
        
        # If using global rank, router must output exactly 1 rank
        if use_global_rank and router.num_rank_outputs != 1:
            raise ValueError(
                f"When use_global_rank=True, router must have num_rank_outputs=1, "
                f"got {router.num_rank_outputs}."
            )

        # build low-rank params + optionally freeze base
        self.toggle_low_rank(low_rank_frontend=low_rank_frontend)

    def toggle_low_rank(
        self,
        enabled: Sequence[bool] | None = None,
        *,
        low_rank_frontend: bool = False,
    ) -> dict:
        toggle_log = self.base.toggle_low_rank(enabled, low_rank_frontend=low_rank_frontend)

        if self.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.base.eval()

        return toggle_log

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    @staticmethod
    def _router_input_from_feat(x_feat: torch.Tensor) -> torch.Tensor:
        # x_feat: (B,C,T) -> (B,T,C) for GRU batch_first=True
        return x_feat.permute(0, 2, 1)

    def forward(self, x: torch.Tensor):
        # --- Spectrogram ---
        x_spec = self.base.compute_spectrogram(x)   # (B, spec_bins, T)

        # --- Frontend ---
        low_rank_frontend = getattr(self.base, "_frontend_low_rank_enabled")
        x = self.base.frontend(x_spec, ranks=None if not low_rank_frontend else None)  # (B,C,T)

        low_rank_stacks = getattr(self.base, "_stacks_low_rank_enabled")

        # ---------- ROUTER ----------
        # Single router placed after frontend, outputs 1 to n_stacks ranks
        router_out = self.router(self._router_input_from_feat(x))
        ranks_cont = router_out["ranks_cont"]  # (B, num_rank_outputs)
        
        # Ensure 2D shape: (B, num_rank_outputs)
        if ranks_cont.dim() == 1:
            ranks_cont = ranks_cont.unsqueeze(-1)  # (B, 1)
        
        num_rank_outputs = ranks_cont.shape[1]

        # ---------- BACKBONE ----------
        for s_idx, stack in enumerate(self.base.backbone):
            # Determine ranks for this stack
            if self.use_global_rank:
                # Global rank used across all stacks (enforced: num_rank_outputs == 1)
                ranks_to_stack = ranks_cont[:, 0]  # (B,)
            elif s_idx < num_rank_outputs:
                # Use the corresponding rank for this stack
                ranks_to_stack = ranks_cont[:, s_idx]  # (B,)
            else:
                # No rank available for this stack, use full rank
                ranks_to_stack = None
            
            # Override with None if stack doesn't have low-rank enabled
            if not low_rank_stacks[s_idx]:
                raise ValueError(f"Stack {s_idx} does not have low-rank enabled but was given ranks.")
                ranks_to_stack = None

            x_pre_stack = x
            for block in stack:
                if self.base.backbone_residual_in_blocks:
                    x = block(x, ranks=ranks_to_stack) + x
                else:
                    x = block(x, ranks=ranks_to_stack)

            if self.base.backbone_residual_in_stacks:
                x = x + x_pre_stack

        # --- Classifier ---
        logits = self.base.classifier(x.mean(dim=-1))

        # --- Full rank baseline (optional) ---
        with torch.no_grad():
            logits_full_rank = self.base.forward(x_spec, ranks=None, x_is_spec=True)

        return logits, router_out, logits_full_rank