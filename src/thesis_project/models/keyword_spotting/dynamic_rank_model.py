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
        routers: nn.ModuleList,          # len 1 for global; len n_stacks for per-stack
        *,
        use_global_router: bool = True,
        low_rank_frontend: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.routers = routers
        self.use_global_router = use_global_router
        self.freeze_base = freeze_base

        # build low-rank params + optionally freeze base
        self.toggle_low_rank(low_rank_frontend=low_rank_frontend)

        # sanity checks
        if use_global_router:
            if len(self.routers) < 1:
                raise ValueError("Global router mode requires routers to have length >= 1.")
        else:
            n_stacks = len(self.base.backbone)
            if len(self.routers) != n_stacks:
                raise ValueError(f"Per-stack router mode requires len(routers)==n_stacks ({n_stacks}), got {len(self.routers)}")

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
        x_feat = self.base.frontend(x_spec, ranks=None if not low_rank_frontend else None)  # (B,C,T)

        low_rank_stacks = getattr(self.base, "_stacks_low_rank_enabled")
        n_stacks = len(self.base.backbone)

        # ---------- ROUTER(S) ----------
        global_out: Optional[dict] = None
        global_ranks: Optional[torch.Tensor] = None

        if self.use_global_router:
            # compute ONCE before stack 0, reuse everywhere
            global_out = self.routers[0](self._router_input_from_feat(x_feat))
            global_ranks = global_out["ranks_cont"]          # (B,)
            # (optional) ensure shape
            if global_ranks.dim() != 1:
                global_ranks = global_ranks.view(-1)

        per_stack_out: list[dict] = []

        # ---------- BACKBONE ----------
        for s_idx, stack in enumerate(self.base.backbone):
            # choose ranks for this stack
            if self.use_global_router:
                router_out = global_out
                ranks_to_stack = global_ranks
            else:
                router_out = self.routers[s_idx](self._router_input_from_feat(x_feat))
                ranks_to_stack = router_out["ranks_cont"]
                if ranks_to_stack.dim() != 1:
                    ranks_to_stack = ranks_to_stack.view(-1)

            per_stack_out.append(router_out if router_out is not None else {})

            if not low_rank_stacks[s_idx]:
                ranks_to_stack = None

            x_pre_stack = x_feat
            for block in stack:
                if self.base.backbone_residual_in_blocks:
                    x_feat = block(x_feat, ranks=ranks_to_stack) + x_feat
                else:
                    x_feat = block(x_feat, ranks=ranks_to_stack)

            if self.base.backbone_residual_in_stacks:
                x_feat = x_feat + x_pre_stack

        # --- Classifier ---
        logits = self.base.classifier(x_feat.mean(dim=-1))

        # --- Full rank baseline (optional) ---
        with torch.no_grad():
            logits_full = self.base.forward(x_spec, ranks=None, x_is_spec=True)

        router_info = {
            "global": global_out,
            "per_stack": per_stack_out,
            "mode": "global" if self.use_global_router else "per_stack",
        }
        return logits, router_info, logits_full