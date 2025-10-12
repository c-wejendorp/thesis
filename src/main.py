from thesis_project.models.blocks import DDWS_Conv1d, SpectrogramBlock, CPC_Conv1d
import torch
import torch.nn as nn
from typing import Optional

from thesis_project.datasets import SpeechCommandsGoogle
from thesis_project.utils.paths import get_data_dir

spectrogram_kwargs = {
    "sample_rate": 16000,
    "n_fft": 512,
}

backbone_kwargs = {
    "n_channels_ext": 128,
    "n_channels_int": 128,
    "kernel_size": 3,
    "dropout": 0.1,
    "causal": True,
    "n_blocks_pr_stack": 3,
    "dilation_base": 2,
    "n_stacks": 1,
}

def build_single_stack(**kwargs) -> nn.ModuleList:
    required_keys = ["dilation_base", "n_blocks_pr_stack"]
    missing = [k for k in required_keys if k not in kwargs]
    if missing:
        raise ValueError(f"[build_single_stack] Missing required keys: {missing}")

    return nn.ModuleList([
        DDWS_Conv1d(**{**kwargs, "dilation": kwargs["dilation_base"] ** i})
        for i in range(kwargs["n_blocks_pr_stack"])
    ])


def build_backbone(**backbone_kwargs) -> nn.ModuleList:
    required = ["n_stacks", "n_blocks_pr_stack", "dilation_base"]
    missing = [k for k in required if k not in backbone_kwargs]
    if missing:
        raise ValueError(f"[build_backbone] Missing required keys: {missing}")

    return nn.ModuleList([
        build_single_stack(**backbone_kwargs)
        for _ in range(backbone_kwargs["n_stacks"])
    ])


class KeyWordSpottingModel(nn.Module):
    def __init__(self,
                 num_classes: int = 35,
                 spectrogram_kwargs: Optional[dict] = None,
                 backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()

        # Spectrogram
        spectrogram_kwargs = spectrogram_kwargs or {}
        self.spectrogram = SpectrogramBlock(**spectrogram_kwargs)

        # Backbone
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(**backbone_kwargs)

        # Infer frontend input channels
        if self.spectrogram.n_mels is not None:
            spectrogram_bins = self.spectrogram.n_mels
        else:
            spectrogram_bins = self.spectrogram.n_fft // 2 + 1

        # Frontend
        if "n_channels_ext" not in backbone_kwargs:
            raise ValueError("Missing 'n_channels_ext' in backbone_kwargs")
        backbone_ext_channels = backbone_kwargs["n_channels_ext"]

        self.frontend = CPC_Conv1d(spectrogram_bins, backbone_ext_channels, bias=True)

        self.classifier = nn.Linear(backbone_ext_channels, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the KeyWordSpottingModel.

        Args:
            x: waveform tensor of shape (B, 1, T)

        Returns:
            logits: class logits of shape (B, num_classes)
        """
        x = self.spectrogram(x)  # (B, spec_bins, T)
        x = self.frontend(x)     # (B, C, T) we map the spec_bins to the number of channels expected by the backbone
        for stack in self.backbone:        # each stack is a ModuleList
            for block in stack:
                x = block(x)               # (B, C, T)
        
        #  Global pooling across time (collapse temporal dimension)
        x = x.mean(dim=1)
        x = self.classifier(x)   # (B, num_classes)
        logits = self.classifier(x) # (B, num_classes)

        return logits

if __name__ == "__main__":
    model = KeyWordSpottingModel(
        num_classes=35,
        spectrogram_kwargs=spectrogram_kwargs,
        backbone_kwargs=backbone_kwargs
    )

    data_dir = get_data_dir()
    dataset = SpeechCommandsGoogle(root=str(data_dir), subset="training", download=True)

    # pass a single item through the model
    waveform, label, metadata = dataset[110]
    logits = model(waveform.unsqueeze(0))