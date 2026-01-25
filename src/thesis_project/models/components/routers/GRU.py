import torch
import torch.nn as nn
from typing import Optional

class GRURouter(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 gru_hidden_dim: int, 
                 num_gru_layers: int, 
                 max_rank: int,
                 last_layer_bias_init: Optional[float] = None,
                 pool_type: Optional[str] = None,
                 pool_kernel_size: int = 2,
                 pool_stride: Optional[int] = None
                 ):
        super().__init__()
        self.max_rank = max_rank
        self.pool_type = pool_type
        
        if pool_type not in [None, 'avg', 'max']:
            raise ValueError(f"pool_type must be None, 'avg', or 'max', got {pool_type}")
        
        # Create pooling layer for smoothing along temporal dimension
        if pool_type == 'avg':
            self.pool = nn.AvgPool1d(
                kernel_size=pool_kernel_size,
                stride=pool_stride if pool_stride is not None else pool_kernel_size,
                padding=pool_kernel_size // 2
            )
        elif pool_type == 'max':
            self.pool = nn.MaxPool1d(
                kernel_size=pool_kernel_size,
                stride=pool_stride if pool_stride is not None else pool_kernel_size,
                padding=pool_kernel_size // 2
            )
        else:
            self.pool = None
        
        self.gru = nn.GRU(input_dim, gru_hidden_dim, num_layers=num_gru_layers, batch_first=True)
        self.fc_out = nn.Linear(gru_hidden_dim, 1)
        if last_layer_bias_init is not None:
            with torch.no_grad():
                self.fc_out.bias.fill_(last_layer_bias_init)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,F)

        # Apply temporal pooling for smoothing if specified
        if self.pool is not None:
            # Pool expects (B, C, L) format, we have (B, T, F)
            x = x.transpose(1, 2)  # (B, F, T)
            x = self.pool(x)  # (B, F, T')
            x = x.transpose(1, 2)  # (B, T', F)
        
        _, h_n = self.gru(x)
        h_last = h_n[-1]                 # (B,H)

        logits = self.fc_out(h_last)      # (B,1)
        logits = logits.squeeze(-1)        # (B,)
        r_normalized = torch.sigmoid(logits)
        r_cont = 1 + r_normalized * (self.max_rank - 1)
        return {"ranks_normalized": r_normalized, "ranks_cont": r_cont}