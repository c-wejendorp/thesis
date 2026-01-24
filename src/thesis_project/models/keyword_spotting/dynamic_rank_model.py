import torch
import torch.nn as nn
from .base_model import KWSBase
from thesis_project.models.components.routers import GRURouter
from typing import Sequence, Optional
from thesis_project.models.components.layers import LowRankPointwiseConv1d

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
            # raise not implemented 
            raise NotImplemented
            for p in self.base.parameters():
                p.requires_grad_(False)
            for module in self.base.modules():
               if isinstance(module, LowRankPointwiseConv1d):
               # set grad true for S parameter.
                if module.S is not None:
                    module.S.requires_grad_(True)

            #pass
        
            #raise NotImplementedError("Unfreezing base model not implemented yet.")

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
            #ranks=router_output["ranks"],
            ranks=router_output["ranks_cont"],
            x_is_spec=True,
        )
        # ---------- KWS branch no grad full rank----------
        with torch.no_grad():
            classif_logits_full_rank = self.base.forward(
                x_spec,
                ranks=None,  # full rank
                x_is_spec=True,
            )
        return classif_logits, router_output, classif_logits_full_rank
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()   # keep BN/Dropout fixed
        return self