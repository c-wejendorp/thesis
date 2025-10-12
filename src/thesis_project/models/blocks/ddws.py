import torch.nn as nn
import torch.nn.functional as F

from .cpc import CPC_Conv1d

# Code heavly inspired by Riccardo/ GN 

# DilatedDepthWiseSeparableConv1d
class DDWS_Conv1d(nn.Module):
    def __init__(
            self, 
            n_channels_ext,  # input/output channels
            n_channels_int,  # intermediate channels
            kernel_size,     # kernel size (for depthwise conv)
            dilation,        # dilation rate
            dropout,         # dropout rate
            causal,          # causal: no future information
            **kwargs
        ):
        super().__init__()
        self.n_channels_ext = n_channels_ext
        self.n_channels_int = n_channels_int
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.causal = causal
        self.padding = (kernel_size - 1) * dilation if causal else dilation
        ## PointWise1
        #self.pw_conv1 = nn.Conv1d(n_channels_ext, n_channels_int, 1)
        self.cpw_conv1 = CPC_Conv1d(n_channels_ext, n_channels_int, bias=True) #TODO figure out if we want bias here
        self.act1 = nn.PReLU(num_parameters=1)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(n_channels_int)
        ## DilatedDepthwise
        self.ddw_conv = nn.Conv1d(
            n_channels_int, n_channels_int, kernel_size, 
            padding=self.padding, 
            dilation=dilation, 
            groups=n_channels_int, 
            bias=False,
        )
        self.chomp = Chomp(self.padding) if causal else nn.Identity()
        self.act2 = nn.PReLU(num_parameters=1)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(n_channels_int)
        ## PointWise2
        #self.pw_conv2 = nn.Conv1d(n_channels_int, n_channels_ext, 1, bias=False)
        self.cpw_conv2 = CPC_Conv1d(n_channels_int, n_channels_ext, bias=False) #TODO figure out if we want bias here

        # add the compressed pointwise 1d conv layers (linear layers) to a module such that can be easily found and rank changed
        self.cpw_layers = nn.ModuleList([self.cpw_conv1, self.cpw_conv2])

    def forward(self, x):
        return x + self.forward_no_residual(x)
    
    def forward_no_residual(self, x):
        x = self.cpw_conv1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = self.ddw_conv(x)
        x = self.chomp(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.norm2(x)
        x = self.cpw_conv2(x)
        return x
    
# Chomp (crop input, remove causal padding)
class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., : -self.chomp_size].contiguous()