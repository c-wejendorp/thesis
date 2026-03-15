import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

def ste_rank_mask(r_cont: torch.Tensor, max_rank: int, tau: float, *, device, dtype):
    """
    r_cont: (B,) or (B,1) continuous in [1, max_rank]
    returns mask: (B, max_rank) in [0,1] with STE hard forward and soft backward
    """
    r_cont = r_cont.view(-1, 1).to(device=device, dtype=dtype)  # (B,1)
    k = torch.arange(1, max_rank+1, device=device, dtype=dtype).view(1, -1)  # (1,R)

    tau = max(float(tau), 1e-6)

    soft = torch.sigmoid(((r_cont + 1e-9) - k) / tau)                 # (B,R)
    hard = (k <= r_cont).to(dtype)                             # (B,R) using <=

    mask = soft + (hard - soft).detach()                         # forward hard, backward soft
    return mask

class LowRankPointwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv_init = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.weight_full = nn.Parameter(conv_init.weight.data.clone())  # (out,in,1)
        self.bias = nn.Parameter(conv_init.bias.data.clone()) if bias else None #type: ignore
        del conv_init

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_rank = min(in_channels, out_channels)
        # Max rank where low-rank compute <= full compute (MACs) for kernel_size=1
        r_break_even = (in_channels * out_channels) // (in_channels + out_channels)
        self.max_useful_rank = max(1, min(in_channels, out_channels, r_break_even))

        self.U = self.S = self.V = None  # U:(out,R) S:(R,) V:(R,in)

    def activate_low_rank(self):
        W = self.weight_full.detach().squeeze(-1)  # (out,in)  (detach => no grads through SVD)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        self.U = nn.Parameter(U[:, :self.max_rank].clone())
        self.S = nn.Parameter(S[:self.max_rank].clone())
        self.V = nn.Parameter(Vt[:self.max_rank, :].clone())

    def deactivate_low_rank(self):
        self.U = self.S = self.V = None

    @property
    def low_rank_active(self) -> bool:
        return self.S is not None

    def forward(self, x: torch.Tensor, ranks=None) -> torch.Tensor:
        """
        x:     (B, in_ch, T)
        ranks: None            -> full conv path
            int             -> same rank for all samples (kept as float for STE)
            Tensor/array    -> per-sample continuous ranks (preferred) in [1, max_rank]
        """
        low_rank_ready = (self.U is not None)  # source of truth

        if ranks is not None and not low_rank_ready:
            raise RuntimeError(
                "ranks was provided, but low-rank is not activated. Call activate_low_rank() first."
            )

        # Full-rank path
        if ranks is None:
            return F.conv1d(x, self.weight_full, self.bias)

        # --- Low-rank path ---
        B = x.size(0)
        dtype = x.dtype  # keep ranks float and aligned with AMP/fp16 if used

        # Build continuous ranks (float, so router can receive gradients)
        if isinstance(ranks, int):
            r_cont = torch.full((B,), float(ranks), device=x.device, dtype=dtype)
        else:
            r_cont = torch.as_tensor(ranks, device=x.device, dtype=dtype)

            # Allow (B,1) or (B,) shapes
            if r_cont.dim() == 2 and r_cont.size(-1) == 1:
                r_cont = r_cont.squeeze(-1)

            if r_cont.dim() != 1 or r_cont.size(0) != B:
                raise ValueError(f"`ranks` must have shape (B,) or be an int. Got {tuple(r_cont.shape)} with B={B}.")

        # Project x into rank space using all R components
        SV = (self.S[:, None] * self.V).unsqueeze(-1)   # (R, in, 1)  # type: ignore
        h  = F.conv1d(x, SV, bias=None)                 # (B, R, T)

        # STE gating mask (forward hard, backward soft)
        mask = ste_rank_mask(
            r_cont=r_cont,            # FLOAT ranks -> gradients can flow to router
            max_rank=self.max_rank,
            tau=0.7,
            device=h.device,
            dtype=h.dtype,
        )                                               # (B, R)

        h = h * mask.unsqueeze(-1)                      # (B, R, T)
        #self.mask = mask  # for debugging
        #self.mask.retain_grad()
        # Combine rank components back to out channels
        return F.conv1d(h, self.U.unsqueeze(-1), self.bias)  # type: ignore

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
    layer.activate_low_rank()
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
