import torch
import torch.nn as nn
from typing import Optional



class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def sigmoid_ste_rank(logit: torch.Tensor, max_rank: int):
    """
    logit: (B,) or (B,1)
    returns:
        p:      (B,) or (B,1) in (0,1)
        r_cont: same shape as p, in (1, max_rank)
        r_hard: same shape as p, STE-rounded, clamped to [1, max_rank]
    """
    p = torch.sigmoid(logit)
    r_cont = 1 + p * (max_rank - 1)
    r_hard = RoundSTE.apply(r_cont).clamp(1, max_rank) #type: ignore
    return p, r_cont, r_hard

class GRURouter(nn.Module):
    def __init__(self, input_dim: int,
                 fc_hidden_dim: int, 
                 gru_hidden_dim: int, 
                 num_gru_layers: int, 
                 max_rank: int,
                 last_layer_bias_init: Optional[float] = None
                 ):
        super().__init__()
        self.max_rank = max_rank
        self.fc_in = nn.Linear(input_dim, fc_hidden_dim)
        self.gru = nn.GRU(fc_hidden_dim, gru_hidden_dim, num_layers=num_gru_layers, batch_first=True)
        self.fc_out = nn.Linear(gru_hidden_dim, 1)
        if last_layer_bias_init is not None:
            with torch.no_grad():
                self.fc_out.bias.fill_(last_layer_bias_init)


    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,F)

        x = torch.nn.functional.relu(self.fc_in(x)) #type: ignore  # (B,T,fc_hidden_dim)
        _, h_n = self.gru(x)
        h_last = h_n[-1]                 # (B,H)

        logit = self.fc_out(h_last)      # (B,1)
        logit = logit.squeeze(-1)        # (B,)

        p, r_cont, r = sigmoid_ste_rank(logit, self.max_rank) # (B,), (B,), (B,)

        return {"logit": logit, "p": p, "r_cont": r_cont, "rank": r}
    
# Example usage:
if __name__ == "__main__":
    B, T, F_bins = 8, 12, 257
    router = GRURouter(F_bins, 128, 64, 1, 64)
    out = router(torch.randn(B, T, F_bins))

    print(out["p"].min().item(), out["p"].max().item())      # should be in (0,1)
    print(out["rank"].min().item(), out["rank"].max().item())# should be in [1,64]
    print(out["rank"].unique()[:10])                         # should look integer-ish

