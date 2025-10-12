import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO consider making a parent class and then this a subclass specifically for svd. This would make it easier to add other compression methods later.
# Compressed Pointwise Conv1d (via Low-Rank Approximation (SVD)
class CPC_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, rank=None, bias=True):
        super().__init__()

        # Conv 1D attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_flag = bias
    
        # Full weight initialization via a temporary conv layer (kaiming uniform init) 
        conv_init = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=self.bias_flag)
        self.weight_full = nn.Parameter(conv_init.weight.data.clone())
        self.bias = nn.Parameter(conv_init.bias.data.clone()) if self.bias_flag else None # type: ignore
        del conv_init  # free memory

        # Low-rank attributes
        self.rank = rank  # None = full rank
        self.U = None
        self.V = None
        self.S = None

    def activate_low_rank(self, rank: int):
        """Compute SVD from full weight and keep top-`rank` components."""
        W = self.weight_full.detach().squeeze(-1)  # [out, in]
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        self.rank = rank
        self.U = nn.Parameter(U[:, :rank].clone())
        self.S = nn.Parameter(S[:rank].clone())
        self.V = nn.Parameter(Vt[:rank, :].clone())

    def deactivate_low_rank(self):
        """Disable low-rank mode and use full weights."""
        self.rank = None
        self.U = self.S = self.V = None

    # --- forward ---
    def forward(self, x):
        if self.rank is None:
            return F.conv1d(x, self.weight_full, self.bias)
        else:
            W_approx = (self.U @ torch.diag(self.S) @ self.V).unsqueeze(-1) # type: ignore
            return F.conv1d(x, W_approx, self.bias)