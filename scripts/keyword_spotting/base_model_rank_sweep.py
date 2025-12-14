# %%
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union

from tqdm import tqdm
import json

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.components.routers import GRURouter
from thesis_project.utils.paths import get_data_dir
from thesis_project.datasets import SpeechCommandsGoogle
#from thesis_project.training.key_word_spotting.train_base_loop import fit_base_model, validate_model
from base_config_temp import cfg, noise_train_cfg, noise_eval_cfg

MODEL_DIR = "/Users/christoffer/Documents/Thesis/thesis_project/model_runs/base/2025-12-14_18-48-50/"
MODEL_CHECKPOINT = MODEL_DIR + "best_model.pth"
MODEL_HISTORY = np.load(MODEL_DIR + "history.npy", allow_pickle=True).item()
BATCH_SIZE = 64
STEP_FOR_RANK_SWEP_EVAL = 1

# %%
# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    pin_memory = True
    num_workers = 16
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    pin_memory = False
    num_workers = 0
    print("Using MPS")
else:
    device = torch.device("cpu")
    pin_memory = False
    print("Using CPU")

# %%
base_model = KWSBase(cfg).to(device)
base_model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))


# %%
#Datasets and dataloaders
data_dir = get_data_dir()
train_set = SpeechCommandsGoogle(root=str(data_dir), subset="training", download=True, **noise_train_cfg.model_dump())
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory = pin_memory, num_workers=num_workers)

val_set = SpeechCommandsGoogle(root=str(data_dir), subset="validation", download=True, **noise_eval_cfg.model_dump())
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory = pin_memory, num_workers=num_workers)

# %%
def validate_model(model, val_loader, criterion, device, snr_values=None, verbose=True, fixed_rank=None):
    """
    Evaluate `model` on the same validation set for different SNR values.

    Assumes:
        - val_loader.dataset._set_snr(snr) exists.
        - val_loader yields (waveform, label, meta).
    """
    if snr_values is None:
        snr_values = [None]

    model.eval()
    results = {}

    ds = val_loader.dataset  # the dataset bound to this loader

    with torch.no_grad():
        for snr in snr_values:
            # ---- SET THE SNR FOR THIS PASS ----
            ds._set_snr(snr)

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            pbar = tqdm(val_loader, desc=f"SNR={snr}", leave=False)
            for waveforms, labels, meta in pbar:
                waveforms = waveforms.to(device)
                labels = labels.to(device)

                if fixed_rank is not None:
                    rank = fixed_rank
                else:
                    rank = None
         
                logits = model(waveforms, ranks=rank)

                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()

                # live progress
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100 * total_correct / total_samples:.2f}%",
                )

            avg_loss = total_loss / total_samples
            avg_acc = 100.0 * total_correct / total_samples

            results[snr] = {"loss": avg_loss, "acc": avg_acc}

            if verbose:
                print(f"SNR={snr}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%")

    return results

# %%
def evaluate_rank_sweep(
    model: nn.Module,
    validate_fn,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_rank: int,
    snr_values,
    step: int = 4,
    stop_at_half_rank: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[Union[int, float], Dict[str, float]]]:
    """
    Sweep fixed ranks and record validation accuracy.

    Assumes validate_fn signature:
        validate_fn(
            model,
            val_loader,
            criterion,
            device,
            snr_values=...,
            verbose=...,
            fixed_rank=...
        ) -> dict {snr: {"acc": float, "loss": float}}

    Returns:
        results: dict mapping "rank_{r}" -> per-SNR metrics
    """

    results: Dict[str, Dict[Union[int, float], Dict[str, float]]] = {}

    for rank in range(step, max_rank + 1, step):
        if stop_at_half_rank and rank > (max_rank // 2):
            if verbose:
                print(f"Stopping sweep at rank {rank} (stop_at_half_rank=True) since it exceeds half max_rank={max_rank}")
            break

        per_snr = validate_fn(
            model,
            val_loader,
            criterion,
            device,
            snr_values=snr_values,
            verbose=False,
            fixed_rank=rank,
        )

        results[f"rank_{rank}"] = per_snr

        if verbose:
            print(f"\n=== Rank {rank} Val Accuracy per SNR ===")
            for snr, stats in per_snr.items():
                print(
                    f"  SNR={str(snr):>6}: "
                    f"acc={stats['acc']:.2f}%  "
                    f"loss={stats['loss']:.4f}"
                )
            print("-" * 60)

    return results

# %%
VAL_SNR_VALUES = [-5, 0, 5, 10,15,float('inf')]
# deactivate low-rank everywhere
base_model.toggle_low_rank((False,False,False))
full_rank_acc = validate_model(base_model, val_loader, nn.CrossEntropyLoss(), device, snr_values=VAL_SNR_VALUES, verbose=False, fixed_rank=None)
print(f"\n=== Full Rank Val Accuracy per SNR ===")
for snr, stats in full_rank_acc.items():
    print(f"  SNR={str(snr):>6}: acc={stats['acc']:.2f}%  loss={stats['loss']:.4f}")
print("-" * 60)

# %%
# Define combinations declaratively (good!)
combs = {
    "comb_1": {
        "stacks_low_rank": [True, True, True],
        "frontend_low_rank": False,
    },
    "comb_2": {
        "stacks_low_rank": [True, True, False],
        "frontend_low_rank": False,
    },
    "comb_3": {
        "stacks_low_rank": [True, False, False],
        "frontend_low_rank": False,
    },
}

all_results = {}
criterion = nn.CrossEntropyLoss()

for comb_name, cfg in combs.items():
    print(f"\n=== Testing combination: {comb_name} ===")
    print(f"    stacks_low_rank={cfg['stacks_low_rank']}, "
          f"frontend_low_rank={cfg['frontend_low_rank']}")

    # Toggle low-rank according to config
    base_model.toggle_low_rank(
        cfg["stacks_low_rank"],
        low_rank_frontend=cfg["frontend_low_rank"],
    )

    sweep_results = evaluate_rank_sweep(
        base_model,
        validate_model,
        val_loader,
        criterion,
        device,
        max_rank=128,
        snr_values=VAL_SNR_VALUES,
        step=STEP_FOR_RANK_SWEP_EVAL,
        stop_at_half_rank=True,
        verbose=True,
    )

    all_results[comb_name] = {
        "config": cfg,
        "sweep": sweep_results,
    }

# save the results to model directory
with open(MODEL_DIR + "low_rank_sweep_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# %%
def plot_rank_vs_accuracy_by_noise(
    results,
    title=None,
    save_path=None,
    metric="acc",            # "acc" or "loss"
    show_flops=True,         # keep the secondary y-axis like before
    full_rank=None,
):
    """
    Plot {metric} vs rank with one curve per SNR (or "noise param" key).

    Supports BOTH inputs:
      A) New dict format (your updated sweep):
         results = {
            "rank_4":  {snr: {"acc":..., "loss":...}, ...},
            "rank_8":  {...},
         }

      B) Old list format:
         results = [
            (4,  {snr: {"acc":..., "loss":...}, ...}),
            (8,  {...}),
         ]
    """

    # ---------- normalize input to list[(rank:int, per_noise:dict)] ----------
    if isinstance(results, dict):
        parsed = []
        for k, v in results.items():
            # accept "rank_4" or any string containing an int
            m = re.search(r"(\d+)", str(k))
            if m is None:
                continue
            parsed.append((int(m.group(1)), v))
        results_list = parsed
    else:
        results_list = list(results)

    # ---------- collect and sort ranks ----------
    ranks = sorted(int(r) for r, _ in results_list)
    rank_to_dict = {int(r): d for r, d in results_list}

    # ---------- gather all noise/SNR keys across ranks ----------
    all_keys = set()
    for d in rank_to_dict.values():
        for k in d.keys():
            # Try to normalize keys to floats where possible (incl. inf)
            try:
                all_keys.add(float(k))
            except Exception:
                # keep non-numeric keys as strings
                all_keys.add(str(k))

    # Sort: numeric keys first, then non-numeric
    numeric_keys = sorted([k for k in all_keys if isinstance(k, float)])
    str_keys = sorted([k for k in all_keys if isinstance(k, str)])
    noise_params = numeric_keys + str_keys

    # ---------- build series per noise key aligned to 'ranks' ----------
    series = {}  # noise_key -> list of metric values aligned with ranks
    for np_val in noise_params:
        vals = []
        for r in ranks:
            d = rank_to_dict[r]

            # keys might be stored as float or string; try both
            entry = None
            if np_val in d:
                entry = d[np_val]
            else:
                # try matching float<->string
                if isinstance(np_val, float):
                    entry = d.get(str(np_val), None)
                elif isinstance(np_val, str):
                    try:
                        entry = d.get(float(np_val), None)
                    except Exception:
                        entry = None

            if entry is None or metric not in entry:
                vals.append(np.nan)
            else:
                vals.append(entry[metric])
        series[np_val] = vals

    # ---------- relative FLOPs for PW conv ----------
    C = int(full_rank) if full_rank is not None else (max(ranks) if ranks else 1)
    flops_rel = [2 * r / C for r in ranks]  # =1 when r=C/2

    # ---------- plot ----------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for np_val, vals in series.items():
        label = f"SNR={np_val:g}" if isinstance(np_val, float) else f"{np_val}"
        ax1.plot(ranks, vals, marker="o", label=label)

    ax1.set_xlabel("Target Rank")
    ax1.set_ylabel("Validation Accuracy (%)" if metric == "acc" else "Validation Loss")
    ax1.grid(True, alpha=0.4)

    # helpful vertical reference at C//2
    ax1.axvline(x=C // 2, color="r", linestyle="--", linewidth=1, label="Max Rank // 2")

    # right axis: relative FLOPs (optional)
    if show_flops:
        ax2 = ax1.twinx()
        ax2.plot(ranks, flops_rel, linestyle="-", alpha=0.6, label="Relative PW FLOPs")
        ax2.set_ylabel("Relative PW FLOPs (vs full conv)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    # title
    if title is None:
        base = "Validation Accuracy vs. Target Rank" if metric == "acc" else "Validation Loss vs. Target Rank"
        if show_flops:
            base += " (with Relative PW FLOPs)"
        title = base
    title += f" (Max Rank: {C})"
    ax1.set_title(title)

    # save or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# %%
for comb_name, comb_data in all_results.items():
    sweep_results = comb_data["sweep"]

    plot_rank_vs_accuracy_by_noise(
        results=sweep_results,
        title=f"Validation Accuracy vs. Target Rank ({comb_name})",
        metric="acc",
        show_flops=True,
        full_rank=128,
        save_path=MODEL_DIR + f"val_acc_vs_rank_{comb_name}.png",
    )


