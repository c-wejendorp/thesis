import torch
import torch.nn as nn
from .base_model import KWSBase
from thesis_project.models.components.routers import GRURouter
from typing import Sequence, Optional

class KWSDynamic(nn.Module):
    #TODO make a baseclass for the router
    def __init__(self, base: KWSBase, router : GRURouter, low_rank_frontend: bool = False, freze_base: bool = True) -> None:
        super().__init__()

        # --- Base ---
        self.base = base
        if freze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.router = router
        # enables low-rank in all stacks by default, low-rank frontend can be controlled separately
        self.base.toggle_low_rank(low_rank_frontend=low_rank_frontend) 

    def toggle_low_rank(
        self,
        enabled: Sequence[bool] | None = None,
        *,
        low_rank_frontend: bool = False,
    ) -> dict:
        return self.base.toggle_low_rank(enabled,low_rank_frontend=low_rank_frontend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Spectrogram ---
        x_spec = self.base.compute_spectrogram(x)  # (B, spec_bins, T)  

        # ---------- ROUTER branch ----------
        # Use the spectrogram as a sequence: (B, T, spec_bins)
        router_input = x_spec.permute(0, 2, 1)  # (B, T, spec_bins)
        ranks = self.router(router_input)

        # ---------- KWS branch ----------
        return self.base.forward(x_spec, ranks=ranks, x_is_spec=True)