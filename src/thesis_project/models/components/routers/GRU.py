import torch
import torch.nn as nn
from typing import Optional

class GRURouter(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 gru_hidden_dim: int, 
                 num_gru_layers: int, 
                 max_rank: int,
                 num_rank_outputs: int = 1,
                 last_layer_bias_init: Optional[float] = None,
                 pool_type: Optional[str] = None,
                 pool_reduction_factor = 2,
                 ):
        super().__init__()
        self.max_rank = max_rank
        self.num_rank_outputs = num_rank_outputs
        self.pool_type = pool_type
        self.pool_reduction_factor = pool_reduction_factor
               
        # Create pooling layer for smoothing along temporal dimension
        # If pool_reduction_factor is 1, treat as no pooling regardless of pool_type
        if pool_reduction_factor == 1 or pool_type is None:
            self.pool = None  # no pooling
        elif pool_type == 'avg':
            self.pool = nn.AvgPool1d(
                kernel_size=self.pool_reduction_factor, 
                stride=self.pool_reduction_factor, 
                padding=0)
        
        elif pool_type == 'max':
            self.pool = nn.MaxPool1d(
                kernel_size=self.pool_reduction_factor,
                stride=self.pool_reduction_factor,
                padding=0
            )
        elif pool_type == 'subsample':
            # works on (B, F, T) to mimic the maxpooling behavior, so remember to transpose before/after
            self.pool = lambda x: x[:, :, ::self.pool_reduction_factor]
        else: 
            raise ValueError(f"pool_type must be None, 'avg', 'max', or 'subsample', got {pool_type}")
        
        self.gru = nn.GRU(input_dim, gru_hidden_dim, num_layers=num_gru_layers, batch_first=True)
        self.fc_out = nn.Linear(gru_hidden_dim, num_rank_outputs)
        if last_layer_bias_init is not None:
            with torch.no_grad():
                self.fc_out.bias.fill_(last_layer_bias_init)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,F)

        if self.pool is not None:
            # Do pooling, but ensure subsample and avg/max pooling reduce length by integer factor
            T = x.shape[1]
            T_cut = (T // self.pool_reduction_factor) * self.pool_reduction_factor          # drop leftover tail
            x = x[:, :T_cut, :]           # (B, T_cut, F)
            x = x.transpose(1, 2)   # (B, F, T_cut)
            x = self.pool(x)          # nn.AvgPool1d / MaxPool1d / subsample
            x = x.transpose(1, 2)     # (B, T_out, F)

        _, h_n = self.gru(x)
        h_last = h_n[-1]                 # (B,H)

        logits = self.fc_out(h_last)      # (B, num_rank_outputs)
        r_normalized = torch.sigmoid(logits)  # (B, num_rank_outputs)
        r_cont = 1 + r_normalized * (self.max_rank - 1)  # (B, num_rank_outputs)
        return {"ranks_normalized": r_normalized, "ranks_cont": r_cont}

    def compute_macs(self, sequence_length: int) -> int:
        """
        Compute the number of Multiply-Accumulate operations (MACs) for the router.

        Args:
            sequence_length: Number of timesteps T before pooling/subsampling
        """
        total_macs = 0

        # Effective sequence length after pooling/subsampling
        if self.pool_type in ("avg", "max", "subsample"):
            effective_seq_len = sequence_length // self.pool_reduction_factor
        else:
            effective_seq_len = sequence_length

        # Pooling MACs (proxy)
        if self.pool_type == "avg" or self.pool_type == "max": # max is a proxy since it doesn't have MACs but does have computational cost that scales similarly to avg pooling
            total_macs += (
            effective_seq_len
            * self.gru.input_size
            )

        # elif self.pool_type == "max":
        #     total_macs += (
        #         (self.pool_reduction_factor - 1)
        #         * effective_seq_len
        #         * self.gru.input_size
        #     )

        # GRU MACs
        total_macs += (
            3
            * (self.gru.input_size * self.gru.hidden_size
            + self.gru.hidden_size * self.gru.hidden_size)
            * effective_seq_len
        )

        if self.gru.num_layers > 1:
            total_macs += (
                3
                * (self.gru.hidden_size * self.gru.hidden_size
                + self.gru.hidden_size * self.gru.hidden_size)
                * effective_seq_len
                * (self.gru.num_layers - 1)
            )

        # Final linear layer (once)
        total_macs += self.gru.hidden_size * self.num_rank_outputs

        return total_macs