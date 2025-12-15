import torch
import torch.nn as nn
from .base_model import KWSBase
from thesis_project.models.components.routers import GRURouter
from typing import Sequence, Optional

class KWSDynamic(nn.Module):
    #TODO, make base class for the router models
    def __init__(
        self,
        base: KWSBase,
        router: GRURouter,
        low_rank_frontend: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        self.base = base
        self.router = router
        self.freeze_base = freeze_base
        # create the U/S/V matrices for low-rank convs
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
        else:
            raise NotImplementedError("Unfreezing base model not implemented yet.")

        return toggle_log

    def forward(self, x: torch.Tensor):
        # --- Spectrogram ---
        x_spec = self.base.compute_spectrogram(x)  # (B, spec_bins, T)

        # ---------- ROUTER branch ----------
        router_input = x_spec.permute(0, 2, 1)     # (B, T, spec_bins)
        router_output = self.router(router_input)  # dict

        # ---------- KWS branch ----------
        classif_logits = self.base.forward(
            x_spec,
            ranks=router_output["ranks"],
            x_is_spec=True,
        )
        return classif_logits, router_output