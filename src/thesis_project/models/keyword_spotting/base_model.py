import torch
import torch.nn as nn
from typing import Sequence, Optional
from .base_schema import KeyWordSpottingBaseConfig, SpectrogramConfig, BackboneConfig
from thesis_project.models.components.blocks import DDWS_Conv1d, SpectrogramBlock
from thesis_project.models.components.layers import LowRankPointwiseConv1d


def build_single_stack(cfg: BackboneConfig) -> nn.ModuleList:
    """Build a single stack of DDWS_Conv1d blocks based on the backbone config."""
    return nn.ModuleList([
        DDWS_Conv1d(
            **cfg.model_dump(),
            dilation=cfg.dilation_base ** i
        )
        for i in range(cfg.n_blocks_pr_stack)
    ])

def build_backbone(cfg: BackboneConfig) -> nn.ModuleList:
    """Build the full backbone consisting of multiple stacks."""
    return nn.ModuleList([
        build_single_stack(cfg)
        for _ in range(cfg.n_stacks)
    ])


class KWSBase(nn.Module):
    def __init__(self, cfg: KeyWordSpottingBaseConfig) -> None:
        """
        Args:
            cfg: Validated KeyWordSpottingBaseConfig object.
        """
        super().__init__()
        self.cfg = cfg  # keep full config for reproducibility

        # --- Spectrogram ---
        self.spectrogram = SpectrogramBlock(**cfg.spectrogram.model_dump())

        # --- Determine frontend input size ---
        self.spectrogram_bins = (
            self.spectrogram.n_mels
            if getattr(self.spectrogram, "n_mels", None) is not None
            else self.spectrogram.n_fft // 2 + 1
        )

        # --- Frontend ---
        self.frontend = LowRankPointwiseConv1d(
            in_channels=self.spectrogram_bins,
            out_channels=cfg.backbone.n_channels_ext,
            bias=True,
        )

        # --- Backbone ---
        self.backbone = build_backbone(cfg.backbone)
        self.backbone_residual_in_blocks = cfg.backbone.residual_in_blocks
        self.backbone_residual_in_stacks = cfg.backbone.residual_in_stacks

        # --- Classifier --- simple FC layer
        self.classifier = torch.nn.Linear(
            cfg.backbone.n_channels_ext,
            cfg.dataset.num_classes,
        )

        # --- Low-rank tracking ---
        self._frontend_low_rank_enabled: bool = False
        self._stacks_low_rank_enabled: list[bool] = [False] * len(self.backbone)

    def compute_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the spectrogram module.

        Args:
            x: waveform tensor of shape (B, 1, T)

        Returns:
            spectrogram: (B, spec_bins, T)
        """
        # --- Spectrogram --- INPUT: (B, n_channels_in_waveform, T)
        x_spec = self.spectrogram(x)  # (B, n_channels_in_waveform, spec_bins, T)
        return x_spec.squeeze(1)         # (B, spec_bins, T) # remove channel dim (mono)
    
    def forward(self, x: torch.Tensor, ranks: torch.Tensor | int | None = None, x_is_spec: bool = False, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass for the KeyWordSpottingModel.

        Args:
            x: waveform tensor of shape (B, 1, T) or spectrogram tensor of shape (B, spec_bins, T) if x_is_spec is True
            ranks: ranks for CPC_Conv1d layers (if any)
            

        Returns:
            logits: class logits of shape (B, num_classes)
        """
        # --- Spectrogram --- INPUT: (B, n_channels_in_waveform, T)
        if not x_is_spec:
            x = self.compute_spectrogram(x)  # (B, spec_bins, T)

        # --- Frontend ---
        low_rank_frontend = getattr(self, "_frontend_low_rank_enabled")
        x = self.frontend(x, ranks=ranks if low_rank_frontend else None)   # (B, C, T) we map the spec_bins to the number of channels expected by the backbone

        # --- Backbone ---
        low_rank_stacks = getattr(self, "_stacks_low_rank_enabled")
        for s_idx, stack in enumerate(self.backbone):       # each stack is a ModuleList
            x_pre_stack = x
            ranks_to_stack = ranks if low_rank_stacks[s_idx] else None

            for block in stack:  # type: ignore
                if self.backbone_residual_in_blocks:
                    x = block(x, ranks=ranks_to_stack) + x  # (B, C, T) and residual connection inside each block
                else:
                    x = block(x, ranks=ranks_to_stack)              # (B, C, T) and residual connection inside each block
            if self.backbone_residual_in_stacks:
                x = x + x_pre_stack  # (B, C, T) and residual connection between stacks

        # --- Classifier ---
        x = x.mean(dim=-1) #  Global pooling across time (collapse temporal dimension)
        logits = self.classifier(x) # (B, num_classes)
        if return_embedding:
            return logits, x #type: ignore
        return logits
    
    def _iter_low_rank_in_module(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, LowRankPointwiseConv1d):
                yield m
    
    def _toggle_low_rank(self, module: nn.Module, enable: bool) -> int:
        layers = list(self._iter_low_rank_in_module(module))
        for lr in layers:
            if enable and not lr.low_rank_active:
                lr.activate_low_rank()
            elif (not enable) and lr.low_rank_active:
                lr.deactivate_low_rank()
        return len(layers)
    
    def toggle_low_rank(
        self,
        enabled: Sequence[bool] | None = None,
        *,
        low_rank_frontend: bool = False,
    ) -> dict:
        n_stacks = len(self.backbone)
        if enabled is None:
            enabled = [True] * n_stacks

        if len(enabled) != n_stacks:
            raise ValueError(f"enabled must have length {n_stacks}, got {len(enabled)}")
        
        self._frontend_low_rank_enabled = bool(low_rank_frontend)
        self._stacks_low_rank_enabled = list(map(bool, enabled))

        total_layers = 0
        active_layers = 0
        per_stack = []

        if low_rank_frontend:
            n = self._toggle_low_rank(self.frontend, True)
            total_layers += n
            active_layers += n  # frontend forced on
            per_stack.append({"stack": "frontend", "low_rank_active": True})
        else:
            per_stack.append({"stack": "frontend", "low_rank_active": False})

        for i, (stack, flag) in enumerate(zip(self.backbone, enabled)):
            n = self._toggle_low_rank(stack, flag)
            total_layers += n
            if flag:
                active_layers += n
            per_stack.append({"stack": i, "low_rank_active": bool(flag)})

        return {
            "total_low_rank_layers": total_layers,
            "active_low_rank_layers": active_layers,
            "per_stack": per_stack,
        }