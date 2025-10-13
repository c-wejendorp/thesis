import torch
import torch.nn as nn
from .keyword_spotting_schema import KeyWordSpottingConfig, SpectrogramConfig, BackboneConfig
from thesis_project.models.blocks import DDWS_Conv1d, SpectrogramBlock, CPC_Conv1d

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


class KeyWordSpottingModel(nn.Module):
    def __init__(self, cfg: KeyWordSpottingConfig) -> None:
        """
        Args:
            cfg: Validated KeyWordSpottingConfig object.
        """
        super().__init__()
        self.cfg = cfg  # keep full config for reproducibility

        # --- Spectrogram ---
        self.spectrogram = SpectrogramBlock(**cfg.spectrogram.model_dump())

        # --- Backbone ---
        self.backbone = build_backbone(cfg.backbone)
        self.backbone_residual_in_blocks = cfg.backbone.residual_in_blocks
        self.backbone_residual_in_stacks = cfg.backbone.residual_in_stacks

        # --- Determine frontend input size ---
        spectrogram_bins = (
            self.spectrogram.n_mels
            if getattr(self.spectrogram, "n_mels", None) is not None
            else self.spectrogram.n_fft // 2 + 1
        )

        # --- Frontend ---
        self.frontend = CPC_Conv1d(
            in_channels=spectrogram_bins,
            out_channels=cfg.backbone.n_channels_ext,
            bias=True,
        )

        # --- Classifier ---
        self.classifier = nn.Linear(
            cfg.backbone.n_channels_ext,
            cfg.num_classes,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the KeyWordSpottingModel.

        Args:
            x: waveform tensor of shape (B, 1, T)

        Returns:
            logits: class logits of shape (B, num_classes)
        """
        # --- Spectrogram --- INPUT: (B, n_channels_in_waveform, T)
        x = self.spectrogram(x)  # (B, n_channels_in_waveform, spec_bins, T)
        x = x.squeeze(1)         # (B, spec_bins, T) # remove channel dim (mono)

        # --- Frontend ---
        x = self.frontend(x)     # (B, C, T) we map the spec_bins to the number of channels expected by the backbone

        # --- Backbone ---
        for stack in self.backbone:        # each stack is a ModuleList
            x_pre_stack = x
            for block in stack:  # type: ignore
                if self.backbone_residual_in_blocks:
                    x = block(x) + x  # (B, C, T) and residual connection inside each block
                else:
                    x = block(x)              # (B, C, T) and residual connection inside each block
            if self.backbone_residual_in_stacks:
                x = x + x_pre_stack  # (B, C, T) and residual connection between stacks

        # --- Classifier ---
        x = x.mean(dim=-1) #  Global pooling across time (collapse temporal dimension)
        logits = self.classifier(x) # (B, num_classes)
        return logits