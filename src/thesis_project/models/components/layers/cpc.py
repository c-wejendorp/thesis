import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class LowRankPointwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv_init = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.weight_full = nn.Parameter(conv_init.weight.data.clone())  # (out,in,1)
        self.bias = nn.Parameter(conv_init.bias.data.clone()) if bias else None #type: ignore
        del conv_init

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Max rank where low-rank compute <= full compute (MACs) for kernel_size=1
        r_break_even = (in_channels * out_channels) // (in_channels + out_channels)
        self.max_useful_rank = max(1, min(in_channels, out_channels, r_break_even))

        self.U = self.S = self.V = None  # U:(out,R) S:(R,) V:(R,in)

    def activate_low_rank(self, R: int):
        R_req = int(R)
        R = max(1, min(R_req, self.max_useful_rank))

        if R < R_req:
            warnings.warn(
                f"Requested rank {R_req} exceeds max_useful_rank={self.max_useful_rank}. "
                f"Clamping to R={R}.",
                RuntimeWarning,
            )

        W = self.weight_full.detach().squeeze(-1)  # (out,in)  (detach => no grads through SVD)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        self.U = nn.Parameter(U[:, :R].clone())
        self.S = nn.Parameter(S[:R].clone())
        self.V = nn.Parameter(Vt[:R, :].clone())

    def deactivate_low_rank(self):
        self.U = self.S = self.V = None

    def forward(self, x, ranks=None):
        low_rank_ready = (self.U is not None)  # source of truth

        if ranks is not None and not low_rank_ready:
            raise RuntimeError("ranks was provided, but low-rank is not activated. Call activate_low_rank(R) first.")

        if ranks is None:
            return F.conv1d(x, self.weight_full, self.bias)

        # Low-rank path
        B = x.size(0)
        R = self.S.numel() #type: ignore

        if isinstance(ranks, int):
            ranks = torch.full((B,), ranks, device=x.device)
        ranks = torch.as_tensor(ranks, device=x.device).long().clamp(1, R)

        # fast path: all ranks equal
        if (ranks == ranks[0]).all():
            r = int(ranks[0].item())
            SV = (self.S[:r, None] * self.V[:r, :]).unsqueeze(-1)  # (r,in,1) #type: ignore
            h = F.conv1d(x, SV, bias=None)                         # (B,r,T)
            return F.conv1d(h, self.U[:, :r].unsqueeze(-1), self.bias) #type: ignore

        # general path: per-sample ranks
        SV = (self.S[:, None] * self.V).unsqueeze(-1)              # (R,in,1) #type: ignore
        h = F.conv1d(x, SV, bias=None)                             # (B,R,T)

        k = torch.arange(R, device=x.device)[None, :]
        h = h * (k < ranks[:, None]).to(h.dtype)[:, :, None]

        return F.conv1d(h, self.U.unsqueeze(-1), self.bias) #type: ignore

if __name__ == "__main__":
    torch.manual_seed(0)

    B, C_in, C_out, T = 4, 8, 6, 10
    x = torch.randn(B, C_in, T)

    layer = LowRankPointwiseConv1d(C_in, C_out)

    print("=== Full conv (no ranks) ===")
    y_full = layer(x)
    print("y_full:", y_full.shape)

    print("\n=== Passing ranks without activating low-rank (should fail) ===")
    try:
        layer(x, ranks=4)
    except RuntimeError as e:
        print("Caught expected error:", e)

    print("\n=== Activate low-rank ===")
    layer.activate_low_rank(R=5)
    print("max_useful_rank:", layer.max_useful_rank)

    print("\n=== Static rank (int) ===")
    y_r = layer(x, ranks=3)
    print("y_r:", y_r.shape)

    print("\n=== Per-sample ranks ===")
    ranks = torch.tensor([1, 2, 3, 5])
    y_dyn = layer(x, ranks=ranks)
    print("y_dyn:", y_dyn.shape)

    print("\n=== Deactivate low-rank ===")
    layer.deactivate_low_rank()

    print("\n=== Back to full conv ===")
    y_full2 = layer(x)
    print("y_full2:", y_full2.shape)

    print("\n=== Passing ranks after deactivation (should fail) ===")
    try:
        layer(x, ranks=2)
    except RuntimeError as e:
        print("Caught expected error:", e)
