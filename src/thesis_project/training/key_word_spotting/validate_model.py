import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def validate_model(
    model,
    val_loader,
    criterion,
    device,
    snr_values=None,
    verbose=True,
    *,
    is_dynamic: bool = True,
    fixed_rank: int = None,
    max_rank: int = 64,
    make_rank_hist: bool = False,
    return_per_sample: bool = False,
):
    """
    Unified validation function for both base and dynamic models.
    
    Args:
        model: Model to evaluate (base or dynamic)
        val_loader: DataLoader for validation set
        criterion: Loss function (CrossEntropyLoss or CrossEntropyPlusRankLoss)
        device: Device to run on
        snr_values: List of SNR values to evaluate. If None, uses [None]
        verbose: If True, print results per SNR
        is_dynamic: If True, treat as dynamic model. If False, treat as base model.
        fixed_rank: For base models, the rank to use. Required if is_dynamic=False.
        max_rank: Maximum rank for dynamic models (used for histogram and expected rank)
        make_rank_hist: If True and is_dynamic=True, compute rank histogram
        return_per_sample: If True, return per-sample records as second return value
    
    Returns:
        results: dict {snr: {"loss": float, "acc": float, ...}}
                 For dynamic models, also includes "mean_rank_normalized", "expected_rank",
                 and optionally "rank_hist", "rank_hist_bins", "rank_hist_probs"
        per_sample_records: (if return_per_sample=True) list of dicts with sample-level data
    
    For base models:
        - model(waveforms, ranks=fixed_rank) returns logits
    For dynamic models:
        - model(waveforms) returns (logits, router_output_dict, logits_full_rank)
        - router_output_dict["per_stack"][0]["ranks_normalized"] contains normalized ranks
    """
    if snr_values is None:
        snr_values = [None]
    
    if not is_dynamic and fixed_rank is None:
        raise ValueError("fixed_rank must be provided when is_dynamic=False")

    model.eval()
    results = {}
    per_sample_records = [] if return_per_sample else None
    ds = val_loader.dataset

    with torch.no_grad():
        for snr in snr_values:
            # Set SNR
            ds.set_snr(snr)

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Dynamic model specific accumulators
            total_rnorm = 0.0 if is_dynamic else None
            total_rank_loss_weighted = 0.0 if is_dynamic else None
            
            if is_dynamic and make_rank_hist:
                raise NotImplementedError("Rank histogram tracking is not yet implemented.")
            
            num_stacks = None  # Track number of stacks
            mean_r_stacks = None  # Track per-stack mean ranks

            pbar = tqdm(val_loader, desc=f"SNR={snr}", leave=False)
            
            for waveforms, labels, meta in pbar:
                waveforms = waveforms.to(device)
                labels = labels.to(device)

                # Forward pass
                if is_dynamic:
                    # Dynamic model
                    classif_logits, router_output_dict, _ = model(waveforms)
                    r_normalized = router_output_dict["ranks_normalized"]  # Access directly, same as training
                    r_cont = router_output_dict["ranks_cont"]
                    
                    # Compute loss using same signature as training
                    loss_out = criterion(
                        classif_logits,
                        labels,
                        r_normalized,
                        rank_supervision=None  # No rank supervision during validation
                    )
                    loss = loss_out.loss
                    rank_loss_weighted = float(loss_out.rank_loss_weighted)
                else:
                    # Base model
                    classif_logits = model(waveforms, ranks=fixed_rank)
                    loss = criterion(classif_logits, labels)
                    r_normalized = None
                    r_cont = None
                    rank_loss_weighted = None

                batch_size = labels.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size

                preds = classif_logits.argmax(dim=1)
                correct_mask = preds == labels
                total_correct += correct_mask.sum().item()

                # Dynamic model: track ranks
                if is_dynamic:
                    # Ensure r_normalized is 1D (global rank) or 2D (per-stack rank)
                    if r_normalized.dim() == 2 and r_normalized.size(-1) == 1:
                        r_normalized = r_normalized.squeeze(-1)
                    if r_cont.dim() == 2 and r_cont.size(-1) == 1:
                        r_cont = r_cont.squeeze(-1)
                    
                    # Initialize mean_r_stacks on first batch
                    if mean_r_stacks is None:
                        if r_cont.dim() == 1:
                            # Global rank: single value
                            mean_r_stacks = 0.0
                        else:
                            # Per-stack: one value per stack
                            mean_r_stacks = torch.zeros(r_cont.size(1))
                    
                    # Accumulate normalized rank (for expected rank calculation)
                    r_normalized = r_normalized.clamp(0.0, 1.0)
                    
                    # For multi-stack, average across stacks first, then sum across batch
                    # For global rank, just sum across batch
                    if r_normalized.dim() == 1:
                        # Global rank: sum across batch
                        total_rnorm += r_normalized.sum().item()
                    else:
                        # Per-stack: average across stacks (dim=1), then sum across batch (dim=0)
                        total_rnorm += r_normalized.mean(dim=1).sum().item()
                    
                    total_rank_loss_weighted += rank_loss_weighted * batch_size
                    
                    # Accumulate per-stack or global mean ranks
                    if r_cont.dim() == 1:
                        # Global rank: sum across batch
                        mean_r_stacks += r_cont.sum().item()
                    else:
                        # Per-stack: sum across batch (dim=0), keep stack dimension
                        mean_r_stacks += r_cont.sum(dim=0).detach().cpu()
                
                # Per-sample records
                if return_per_sample:
                    preds_cpu = preds.detach().cpu().tolist()
                    labels_cpu = labels.detach().cpu().tolist()
                    correct_cpu = correct_mask.detach().cpu().tolist()
                    
                    if is_dynamic:
                        r_normalized_cpu = r_normalized.detach().cpu().tolist()
                        r_cont_cpu = r_cont.detach().cpu().tolist()
                    
                    for i in range(batch_size):
                        rec = {
                            "snr_eval": snr,
                            "pred": int(preds_cpu[i]),
                            "target": int(labels_cpu[i]),
                            "correct": bool(correct_cpu[i]),
                        }
                        
                        if is_dynamic:
                            rec["rank"] = float(r_cont_cpu[i])
                            rec["rank_normalized"] = float(r_normalized_cpu[i])
                        else:
                            rec["rank"] = fixed_rank
                        
                        # Add metadata if available
                        if isinstance(meta, dict):
                            for k in ["label", "utterance", "noise_type"]:
                                if k in meta:
                                    val = meta[k]
                                    if isinstance(val, (list, tuple)):
                                        rec[k] = val[i]
                                    elif torch.is_tensor(val):
                                        v = val[i]
                                        rec[k] = v.item() if v.numel() == 1 else v.detach().cpu()
                            
                            if "snr" in meta:
                                snr_val = meta["snr"]
                                if isinstance(snr_val, (list, tuple)):
                                    rec["snr_sample"] = snr_val[i]
                                elif torch.is_tensor(snr_val):
                                    rec["snr_sample"] = snr_val[i].item()
                        
                        per_sample_records.append(rec)

                # Progress bar
                pbar_dict = {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * total_correct / total_samples:.2f}%",
                }
                if is_dynamic:
                    running_mean_rnorm = total_rnorm / max(total_samples, 1)
                    running_expected_rank = 1.0 + running_mean_rnorm * (max_rank - 1)
                    pbar_dict["exp_r"] = f"{running_expected_rank:.1f}"
                
                pbar.set_postfix(pbar_dict)

            # Compute epoch aggregates
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = 100.0 * total_correct / max(total_samples, 1)
            
            snr_result = {
                "loss": avg_loss,
                "acc": avg_acc,
            }

            if is_dynamic:
                mean_rnorm = total_rnorm / max(total_samples, 1)
                expected_rank = 1.0 + mean_rnorm * (max_rank - 1)
                avg_rank_loss_weighted = total_rank_loss_weighted / max(total_samples, 1)
                snr_result["mean_rank_normalized"] = mean_rnorm
                snr_result["expected_rank"] = expected_rank
                snr_result["rank_loss_weighted"] = avg_rank_loss_weighted
                
                # Add per-stack mean ranks
                if mean_r_stacks is not None:
                    if isinstance(mean_r_stacks, torch.Tensor):
                        # Per-stack: divide by total_samples
                        snr_result["mean_r_stacks"] = (mean_r_stacks / max(total_samples, 1)).tolist()
                    else:
                        # Global: single value, expand to match expected format
                        snr_result["mean_r_stacks"] = [mean_r_stacks / max(total_samples, 1)]

            # Convert SNR to string key for consistency with training code expectations
            snr_key = f"snr_{snr}" if snr != float('inf') else "snr_inf"
            results[snr_key] = snr_result

            if verbose:
                if is_dynamic:
                    msg = f"SNR={snr}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%, "
                    msg += f"exp_rank={expected_rank:.2f}"
                    msg += f", rank_loss_weighted={avg_rank_loss_weighted:.4f}"
                    
                    # Show per-stack ranks if available
                    if "mean_r_stacks" in snr_result:
                        r_stacks_str = '[' + ','.join([f"{r:.1f}" for r in snr_result["mean_r_stacks"]]) + ']'
                        msg += f", r_stacks={r_stacks_str}"
                    
                    print(msg)
                else:
                    print(f"SNR={snr}: loss={avg_loss:.4f}, acc={avg_acc:.2f}% (rank={fixed_rank})")

    # Compute aggregate metrics across all SNRs (mean)
    if len(results) > 0:
        agg_loss = sum(r["loss"] for r in results.values()) / len(results)
        agg_acc = sum(r["acc"] for r in results.values()) / len(results)
        
        aggregate_results = {
            "loss": agg_loss,
            "acc": agg_acc,
            "per_snr_results": results,
        }
        
        if is_dynamic:
            agg_mean_rnorm = sum(r["mean_rank_normalized"] for r in results.values()) / len(results)
            agg_expected_rank = sum(r["expected_rank"] for r in results.values()) / len(results)
            agg_rank_loss_weighted = sum(r["rank_loss_weighted"] for r in results.values()) / len(results)
            aggregate_results["mean_rank_normalized"] = agg_mean_rnorm
            aggregate_results["expected_rank"] = agg_expected_rank
            aggregate_results["rank_loss_weighted"] = agg_rank_loss_weighted
    else:
        aggregate_results = {"per_snr_results": results}

    if return_per_sample:
        return aggregate_results, per_sample_records
    return aggregate_results


