import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Any, Dict, Tuple
from dataclasses import dataclass
from thesis_project.utils.paths import create_run_folder
import numpy as np
import os
from tqdm import tqdm


@dataclass
class CrossEntropyRankLossOut:
    """
    Structured output for CrossEntropyPlusRankLoss.

    Attributes:
        loss:
            Total scalar loss used for backprop.

        ce_loss:
            Cross-entropy classification loss (detached).

        rank_loss:
            Rank / efficiency loss term (detached).

        mean_rank_normalized:
            Mean of r_normalized over the batch.
            Multiply by max_rank to get expected rank.

        rank_variance:
            Variance of r_normalized (detached).
            Useful to detect router collapse.
    """
    loss: torch.Tensor
    ce_loss: torch.Tensor
    rank_loss: torch.Tensor
    mean_rank_normalized: torch.Tensor
    rank_variance: torch.Tensor

class CrossEntropyPlusRankLoss(nn.Module):
    """
    Criterion for dynamic routing with explicit components.

    Total loss:
        L_total = L_ce
                + rank_loss_weight * L_rank
                - rank_var_weight  * Var(r_normalized)

    where:
        - L_ce   : cross-entropy classification loss
        - L_rank : rank / compute budget loss
    """

    def __init__(
        self,
        *,
        target_rank_normalized: float,
        rank_loss_weight: float,
        rank_loss_mode: Literal["batch_mean_mse", "per_sample_mse"] = "batch_mean_mse",
        rank_var_weight: float = 0.0,
        class_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            target_rank_normalized:
                Target normalized rank fraction in [0,1].
                Example: desired avg rank=16, max_rank=64 → 0.25

            rank_loss_weight:
                Weight applied to L_rank.
                Can be annealed during training.

            rank_loss_mode:
                - "batch_mean_mse":
                    (mean(r_normalized) - target)^2
                    Enforces a global compute budget.

                - "per_sample_mse":
                    mean((r_normalized - target)^2)
                    Enforces per-sample closeness to target
                    (discourages dynamic spread).

            rank_var_weight:
                Weight for variance encouragement term.
                Higher → stronger push against router collapse.

            class_weight:
                Optional class weights for cross-entropy, shape (C,).
        """
        super().__init__()

        self.target_rank_normalized = float(target_rank_normalized)
        self.rank_loss_weight = float(rank_loss_weight)
        self.rank_loss_mode = rank_loss_mode
        self.rank_var_weight = float(rank_var_weight)

        # Register CE weights so they follow device placement
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None  # type: ignore

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        r_normalized: torch.Tensor,
    ) -> CrossEntropyRankLossOut:
        """
        Args:
            logits:
                (B, C) classification logits.

            y:
                (B,) int64 class labels.

            r_normalized:
                (B,) or (B,1) continuous normalized rank in [0,1].
                IMPORTANT: pass the *continuous* value (not STE-rounded ranks).

        Returns:
            CrossEntropyRankLossOut
        """

        # --- shape normalization ---
        if r_normalized.dim() == 2 and r_normalized.size(-1) == 1:
            r_normalized = r_normalized.squeeze(-1)

        assert r_normalized.dim() == 1, \
            f"r_normalized must be (B,) or (B,1), got {tuple(r_normalized.shape)}"
        assert logits.dim() == 2, \
            f"logits must be (B,C), got {tuple(logits.shape)}"
        assert y.dim() == 1, \
            f"y must be (B,), got {tuple(y.shape)}"

        # Numerical safety (router output should already be sigmoid)
        r_normalized = r_normalized.clamp(0.0, 1.0)

        # --- L_ce: classification loss ---
        ce_loss = F.cross_entropy(logits, y, weight=self.class_weight)

        # --- L_rank: efficiency / rank loss ---
        target = torch.tensor(
            self.target_rank_normalized,
            device=r_normalized.device,
            dtype=r_normalized.dtype,
        )

        mean_r = r_normalized.mean()

        if self.rank_loss_mode == "batch_mean_mse":
            # Controls *average* compute budget
            rank_loss = (mean_r - target).pow(2)

        elif self.rank_loss_mode == "per_sample_mse":
            # Forces each sample toward target (less dynamic)
            rank_loss = (r_normalized - target).pow(2).mean()

        else:
            raise ValueError(f"Unknown rank_loss_mode={self.rank_loss_mode}")

        # --- variance encouragement (anti-collapse) ---
        rank_variance = r_normalized.var(unbiased=False)

        total_loss = (
            ce_loss
            + self.rank_loss_weight * rank_loss
            - self.rank_var_weight * rank_variance
        )

        return CrossEntropyRankLossOut(
            loss=total_loss,
            ce_loss=ce_loss.detach(),
            rank_loss=rank_loss.detach(),
            mean_rank_normalized=mean_r.detach(),
            rank_variance=rank_variance.detach(),
        )


def validate_dynamic_model(
    model,
    val_loader,
    criterion,   # CrossEntropyLoss OR CrossEntropyPlusRankLoss
    device,
    snr_values=None,
    verbose=True,
    *,
    max_rank: int = 64,   # used only to report expected rank
):
    """
    Evaluate `model` on the same validation set for different SNR values.

    Tracks (per SNR):
      - avg loss
      - avg accuracy
      - mean normalized rank (ranks_normalized in [0,1])
      - expected rank = 1 + mean_rnorm * (max_rank - 1)

    Assumes:
      - val_loader.dataset._set_snr(snr) exists
      - val_loader yields (waveforms, labels, meta)
      - model(waveforms) returns:
            classif_logits, router_output
      - router_output["ranks_normalized"] is continuous in [0,1]
    """
    if snr_values is None:
        snr_values = [None]

    model.eval()
    results = {}

    ds = val_loader.dataset

    with torch.no_grad():
        for snr in snr_values:
            # ---- SET SNR ----
            ds._set_snr(snr)

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Track mean normalized rank
            total_rnorm = 0.0

            pbar = tqdm(val_loader, desc=f"SNR={snr}", leave=False)
            for waveforms, labels, meta in pbar:
                waveforms = waveforms.to(device)
                labels = labels.to(device)

                # ---- Forward ----
                classif_logits, router_output = model(waveforms)
                r_normalized = router_output["ranks_normalized"]

                # ---- Loss ----
                # Works for both:
                #  - CrossEntropyLoss(logits, labels)
                #  - CrossEntropyPlusRankLoss(logits, labels, r_normalized)
                try:
                    loss_out = criterion(classif_logits, labels, r_normalized)
                    loss = loss_out.loss
                except TypeError:
                    loss = criterion(classif_logits, labels)

                batch_size = labels.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size

                preds = classif_logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()

                # ---- Rank stats ----
                if r_normalized.dim() == 2 and r_normalized.size(-1) == 1:
                    r_normalized = r_normalized.squeeze(-1)
                total_rnorm += r_normalized.sum().item()

                # ---- Progress ----
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100 * total_correct / total_samples:.2f}%",
                )

            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = 100.0 * total_correct / max(total_samples, 1)

            mean_rnorm = total_rnorm / max(total_samples, 1)
            expected_rank = 1.0 + mean_rnorm * (max_rank - 1)

            results[snr] = {
                "loss": avg_loss,
                "acc": avg_acc,
                "mean_rank_normalized": mean_rnorm,
                "expected_rank": expected_rank,
            }

            if verbose:
                print(
                    f"SNR={snr}: "
                    f"loss={avg_loss:.4f}, acc={avg_acc:.2f}%, "
                    f"mean_rnorm={mean_rnorm:.4f}, exp_rank={expected_rank:.2f}"
                )

    return results


def fit_dynamic_model(
    *,
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,  # CrossEntropyPlusRankLoss(...)
    device: torch.device,
    epochs: int,
    scheduler: Optional[Any] = None,
    run_dir: Optional[str] = None,
    max_rank: int = 64,  # used only for reporting expected rank
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train-only loop for your dynamic model (no validation yet), matching the base-model
    tqdm style + variable names.

    Assumptions:
      - train_loader yields (waveforms, labels, _) like your base loop
      - model(waveforms) returns either:
          (logits, r_normalized) OR {"logits": ..., "r_normalized": ...}
      - criterion(logits, labels, r_normalized) returns CrossEntropyRankLossOut with:
          .loss (scalar for backprop),
          .ce_loss, .rank_loss, .mean_rank_normalized, .rank_variance (detached tensors)

    Saving (temporary, since no validation):
      - best_model.pth: saved when *training loss* improves
      - history.npy: saved every epoch
    """

    # Folder handling (you said create_run_folder already exists—so we use it)
    if run_dir is None:
        run_dir = create_run_folder("model_runs/dynamic")

    model_path = os.path.join(run_dir, "best_model.pth")
    history_path = os.path.join(run_dir, "history.npy")

    model.to(device)

    history: Dict[str, Any] = {
        "run_dir": run_dir,
        "config": {
            "epochs": epochs,
            "max_rank": max_rank,
        },
        "epochs_log": [],
    }

    best_train_loss = float("inf")

    for epoch in range(epochs):

        # ===============================
        # ----------- TRAINING ----------
        # ===============================
        model.train()

        # Match base-model naming conventions
        train_loss = 0.0
        train_ce_loss = 0.0
        train_rank_loss = 0.0
        train_mean_rank_normalized = 0.0
        train_rank_variance = 0.0

        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for waveforms, labels, _ in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            classif_logits, router_output = model(waveforms)

            r_normalized = router_output["ranks_normalized"]

            loss_out = criterion(classif_logits, labels, r_normalized)
            loss_out.loss.backward()
            optimizer.step()

            # --- Stats ---
            batch_size = labels.size(0)
            train_total += batch_size

            # Total loss (for "best model" selection while we don't have validation)
            train_loss += loss_out.loss.item() * batch_size

            # Criterion outputs are detached already (but we cast to float to be safe)
            train_ce_loss += float(loss_out.ce_loss) * batch_size
            train_rank_loss += float(loss_out.rank_loss) * batch_size
            train_mean_rank_normalized += float(loss_out.mean_rank_normalized) * batch_size
            train_rank_variance += float(loss_out.rank_variance) * batch_size

            preds = classif_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            # For tqdm: show running expected rank (mean_rnorm * max_rank)
            running_mean_rnorm = train_mean_rank_normalized / max(train_total, 1)
            running_expected_rank = running_mean_rnorm * float(max_rank)

            pbar.set_postfix({
                "loss": f"{loss_out.loss.item():.4f}",
                "ce": f"{float(loss_out.ce_loss):.4f}",
                "rank": f"{float(loss_out.rank_loss):.4f}",
                "exp_r": f"{running_expected_rank:.1f}",
                "acc": f"{100 * train_correct / train_total:.2f}%"
            })

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss_epoch)  # or val_loss when you add validation
            else:
                scheduler.step()

        # --- Epoch aggregates (base-style) ---
        denom = max(train_total, 1)

        train_loss_epoch = train_loss / denom
        train_ce_loss_epoch = train_ce_loss / denom
        train_rank_loss_epoch = train_rank_loss / denom
        train_acc_epoch = 100.0 * train_correct / denom

        train_mean_rank_normalized_epoch = train_mean_rank_normalized / denom
        train_rank_variance_epoch = train_rank_variance / denom
        train_expected_rank_epoch = train_mean_rank_normalized_epoch * float(max_rank)



        # Save "best" based on training loss (temporary until validation exists)
        if train_loss_epoch < best_train_loss:
            best_train_loss = train_loss_epoch

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "train_metrics": {
                        "loss": train_loss_epoch,
                        "ce_loss": train_ce_loss_epoch,
                        "rank_loss": train_rank_loss_epoch,
                        "acc": train_acc_epoch,
                        "mean_rank_normalized": train_mean_rank_normalized_epoch,
                        "rank_variance": train_rank_variance_epoch,
                        "expected_rank": train_expected_rank_epoch,
                    },
                    # Helpful for reproducibility/debug:
                    "rank_loss_weight": getattr(criterion, "rank_loss_weight", None),
                    "target_rank_normalized": getattr(criterion, "target_rank_normalized", None),
                    "rank_loss_mode": getattr(criterion, "rank_loss_mode", None),
                    "rank_var_weight": getattr(criterion, "rank_var_weight", None),
                },
                model_path,
            )

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss_epoch,
            "train_ce_loss": train_ce_loss_epoch,
            "train_rank_loss": train_rank_loss_epoch,
            "train_acc": train_acc_epoch,
            "train_mean_rank_normalized": train_mean_rank_normalized_epoch,
            "train_rank_variance": train_rank_variance_epoch,
            "train_expected_rank": train_expected_rank_epoch,
            "best_train_loss": best_train_loss,
            # Track loss weights if you anneal them
            "rank_loss_weight": getattr(criterion, "rank_loss_weight", None),
            "rank_var_weight": getattr(criterion, "rank_var_weight", None),
        }
        history["epochs_log"].append(epoch_log)
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss_epoch:.4f}   CE Loss: {train_ce_loss_epoch:.4f}   Rank Loss: {train_rank_loss_epoch:.4f}")
        print(f"  Train Acc: {train_acc_epoch:.2f}%")
        print(f"  Expected Rank: {train_expected_rank_epoch:.2f} / {max_rank}   Rank Variance: {train_rank_variance_epoch:.6f}")


        # Save history each epoch (so crashes still leave logs)
        np.save(history_path, history, allow_pickle=True) # type: ignore

    return model, history

