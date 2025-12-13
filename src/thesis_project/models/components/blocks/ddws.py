import torch
import torch.nn as nn
from ..layers import LowRankPointwiseConv1d

# Code heavly inspired by Riccardo/ GN 

# DilatedDepthWiseSeparableConv1d
class DDWS_Conv1d(nn.Module):
    def __init__(
        self,
        n_channels_ext: int,
        n_channels_int: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
        use_custom_pw: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_custom_pw = use_custom_pw
        self.padding = (kernel_size - 1) * dilation if causal else dilation

        # Pointwise 1
        if use_custom_pw:
            self.pw_layer_1 = LowRankPointwiseConv1d(n_channels_ext, n_channels_int, bias=True)
        else:
            self.pw_layer_1 = nn.Conv1d(n_channels_ext, n_channels_int, 1, bias=True)

        self.act1 = nn.PReLU(num_parameters=1)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(n_channels_int)

        # Depthwise (dilated)
        self.ddw_conv = nn.Conv1d(
            n_channels_int, n_channels_int, kernel_size,
            padding=self.padding, dilation=dilation,
            groups=n_channels_int, bias=False
        )
        self.chomp = Chomp(self.padding) if causal else nn.Identity()

        self.act2 = nn.PReLU(num_parameters=1)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(n_channels_int)

        # Pointwise 2
        if use_custom_pw:
            self.pw_layer_2 = LowRankPointwiseConv1d(n_channels_int, n_channels_ext, bias=False)
        else:
            self.pw_layer_2 = nn.Conv1d(n_channels_int, n_channels_ext, 1, bias=False)

        # easy access to the low-rank layers inside this block
        self.lowrank_pw_layers = nn.ModuleList(
            [m for m in (self.pw_layer_1, self.pw_layer_2) if isinstance(m, LowRankPointwiseConv1d)]
        )

    def iter_lowrank_pw(self):
        return iter(self.lowrank_pw_layers)

    def forward(self, x: torch.Tensor, ranks=None) -> torch.Tensor:
        if self.use_custom_pw:
            x = self.pw_layer_1(x, ranks)
        else:
            x = self.pw_layer_1(x)

        x = self.act1(x)
        x = self.dropout1(x)
        x = self.norm1(x)

        x = self.ddw_conv(x)
        x = self.chomp(x)

        x = self.act2(x)
        x = self.dropout2(x)
        x = self.norm2(x)

        if self.use_custom_pw:
            x = self.pw_layer_2(x, ranks)
        else:
            x = self.pw_layer_2(x)

        return x


class Chomp(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()