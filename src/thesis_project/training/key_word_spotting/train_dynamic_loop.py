import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Any, Dict, Tuple, List
from thesis_project.training.key_word_spotting.loss_functions import DynamicRoutingLoss
from thesis_project.training.key_word_spotting.validate_model import validate_model
from thesis_project.utils.paths import create_run_folder
import numpy as np
import json
import os
from tqdm import tqdm


def fit_dynamic_model(
    *, # make all arguments keyword-only
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: DynamicRoutingLoss,  # CrossEntropyPlusRankLoss(...)
    device: torch.device,
    epochs: int,
    scheduler: Optional[Any] = None,
    run_dir: Optional[str] = None,
    max_rank: int = 64,  # used only for reporting expected rank
    val_loader=None,
    val_snr_values=[-5, 0, 5, 10, 15, float('inf')],  # SNR values to use during validation
    main_val_snr=None,  # SNR to use for best model selection; if None, use mean across all SNRs
    val_epoch: int = 1,  # how often to validate (1 = every epoch, 2 = every 2 epochs, etc.)
    enable_rank_supervision: bool = False,  # if True, find min correct rank and use as target
    rank_supervision_stable: bool = False,  # if True, require prediction to be correct at all higher ranks too
    save_epoch_checkpoints: bool = False,  # if True, save model checkpoint after each epoch
    log_every_n_steps: Optional[int] = 1,  # if set, log metrics every N steps (1 = every batch/step)
    scheduler_mode: Literal['epoch', 'step'] | None = 'step',  # when to step the scheduler
) -> Tuple[nn.Module, List[Dict[str, Any]], List[Dict[str, Any]]]:
   
    if run_dir is None:
        run_dir = create_run_folder("model_runs/dynamic")

    model_path = os.path.join(run_dir, "best_model.pth")
    epoch_log_path = os.path.join(run_dir, "history.json")
    step_log_path = os.path.join(run_dir, "step_history.json")

    model.to(device)

    epoch_logs = []
    step_logs = []
    global_step = 0
    best_train_loss = float("inf")
    best_train_acc = 0.0
    best_val_loss = float("inf")
    best_val_acc = 0.0

    # ----------- TRAINING ----------
    for epoch in range(epochs):
        model.train()
        # Match base-model naming conventions
        train_loss = 0.0
        train_ce_loss = 0.0
        train_rank_loss = 0.0
        train_rank_loss_weighted = 0.0
        train_mean_rank_normalized = 0.0
        train_rank_variance = 0.0
        train_mean_r_stacks = None  # Will be initialized on first batch

        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for waveforms, labels, _ in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            classif_logits, router_output_dict, classif_logits_full_rank = model(waveforms)

            r_normalized = router_output_dict["ranks_normalized"]  # normalized rank(s) [0,1]
            r_cont_ceiled = torch.ceil(router_output_dict["ranks_cont"])  # continuous rank, ceiled


            rank_supervision = None
            if enable_rank_supervision:
                rank_supervision = rank_supervision_logic(
                    args={
                        "model": model,
                        "waveforms": waveforms,
                        "labels": labels,
                        "r_cont_ceiled": r_cont_ceiled,
                        "max_rank": max_rank,
                        "device": device,
                        "rank_supervision_stable": rank_supervision_stable,
                    }
                )
    
            loss_out = criterion(
                classif_logits, 
                labels, 
                r_normalized, 
                rank_supervision=rank_supervision
            )

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
            train_rank_loss_weighted += float(loss_out.rank_loss_weighted) * batch_size
            train_mean_rank_normalized += float(loss_out.batch_mean_rank_normalized) * batch_size
            train_rank_variance += float(loss_out.rank_variance) * batch_size
            
            # Average r_cont_ceiled across batch dimension, keeping stack dimension
            r_stacks_batch_mean = r_cont_ceiled.detach().mean(dim=0).cpu().numpy()  # (num_stacks,) or scalar
            
            # Expand to match backbone length if using global rank
            num_stacks = len(model.base.backbone) # type: ignore
            if r_stacks_batch_mean.size != num_stacks:
                # Global rank case: expand single value to all stacks
                r_stacks_batch_mean = np.full(num_stacks, r_stacks_batch_mean.item())
            
            if train_mean_r_stacks is None:
                train_mean_r_stacks = r_stacks_batch_mean * batch_size
            else:
                train_mean_r_stacks += r_stacks_batch_mean * batch_size

            preds = classif_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            # For tqdm: show running expected rank (mean_rnorm * (max_rank - 1) + 1)
            running_mean_rnorm = train_mean_rank_normalized / max(train_total, 1)
            running_expected_rank = running_mean_rnorm * (max_rank - 1) + 1
            running_mean_r_stacks_all = (train_mean_r_stacks / max(train_total, 1)).tolist()
            r_stacks_str = '[' + ','.join([f"{r:.1f}" for r in running_mean_r_stacks_all]) + ']'

            pbar.set_postfix({
                "loss": f"{loss_out.loss.item():.4f}",
                "ce": f"{float(loss_out.ce_loss):.4f}",
                "rl_w": f"{float(loss_out.rank_loss_weighted):.4f}",
                "exp_r": f"{running_expected_rank:.1f}",
                "r_stacks": r_stacks_str,
                "acc": f"{100 * train_correct / train_total:.2f}%"
            })

            global_step += 1

            # Step-based logging
            if log_every_n_steps is not None and global_step % log_every_n_steps == 0:
                step_log = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss_out.loss.item(),
                    "ce_loss": float(loss_out.ce_loss),
                    "rank_loss": float(loss_out.rank_loss),
                    "rank_loss_weighted": float(loss_out.rank_loss_weighted),
                    "batch_mean_rank_normalized": float(loss_out.batch_mean_rank_normalized),
                    "rank_variance": float(loss_out.rank_variance),
                    "batch_acc": 100.0 * (preds == labels).sum().item() / batch_size,
                    "running_acc": 100.0 * train_correct / train_total,
                    "running_expected_rank": running_expected_rank,
                    "lr": optimizer.param_groups[0]['lr'],
                }
                step_logs.append(step_log)

            # Step-based scheduler
            if scheduler is not None and scheduler_mode == 'step':
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    pass  # ReduceLROnPlateau needs validation metric, skip per-step
                else:
                    scheduler.step()

        if scheduler is not None and scheduler_mode == 'epoch':
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss_epoch)  # or val_loss when you add validation
            else:
                scheduler.step()

        # --- Epoch aggregates (base-style) ---
        denom = max(train_total, 1)

        train_loss_epoch = train_loss / denom
        train_ce_loss_epoch = train_ce_loss / denom
        train_rank_loss_epoch = train_rank_loss / denom
        train_rank_loss_weighted_epoch = train_rank_loss_weighted / denom
        train_acc_epoch = 100.0 * train_correct / denom

        train_mean_rank_normalized_epoch = train_mean_rank_normalized / denom
        train_rank_variance_epoch = train_rank_variance / denom
        train_expected_rank_epoch = train_mean_rank_normalized_epoch * (max_rank - 1) + 1
        
        # Per-stack mean ceiled ranks
        if train_mean_r_stacks is not None:
            train_mean_r_stacks_epoch = (train_mean_r_stacks / denom).tolist()
        else:
            train_mean_r_stacks_epoch = []

        # ===============================
        # --------- VALIDATION ----------
        # ===============================
        val_results = None
        val_loss_epoch = None
        val_acc_epoch = None
        val_mean_rank_normalized_epoch = None
        val_expected_rank_epoch = None
        val_rank_loss_weighted_epoch = None
        # Check if we should validate this epoch:
        # - Every val_epoch epochs (e.g., if val_epoch=2, validate on epochs 2, 4, 6, ...)
        # - Always validate on the last epoch
        is_last_epoch = (epoch == epochs - 1)
        validate_this_epoch = (epoch + 1) % val_epoch == 0 or is_last_epoch

        if val_loader is not None and validate_this_epoch:
            val_results = validate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                snr_values=val_snr_values,
                verbose=True,
                is_dynamic=True,
                max_rank=max_rank,
                make_rank_hist=False,
                return_per_sample=False
            )
            
            # Extract aggregate metrics from validation results
            if main_val_snr is not None:
                # Use specific SNR for best model selection
                snr_key = f"snr_{main_val_snr}" if main_val_snr != float('inf') else "snr_inf"
                val_loss_epoch = val_results["per_snr_results"][snr_key]["loss"]
                val_acc_epoch = val_results["per_snr_results"][snr_key]["acc"]
                val_mean_rank_normalized_epoch = val_results["per_snr_results"][snr_key]["mean_rank_normalized"]
                val_expected_rank_epoch = val_results["per_snr_results"][snr_key]["expected_rank"]
                val_rank_loss_weighted_epoch = val_results["per_snr_results"][snr_key]["rank_loss_weighted"]
            else:
                # Use mean across all SNRs
                val_loss_epoch = val_results["loss"]
                val_acc_epoch = val_results["acc"]
                val_mean_rank_normalized_epoch = val_results["mean_rank_normalized"]
                val_expected_rank_epoch = val_results["expected_rank"]
                val_rank_loss_weighted_epoch = val_results["rank_loss_weighted"]
        
        # Build checkpoint dictionary once (used for both best model and epoch checkpoints)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "train_metrics": {
                "loss": train_loss_epoch,
                "ce_loss": train_ce_loss_epoch,
                "rank_loss": train_rank_loss_epoch,
                "rank_loss_weighted": train_rank_loss_weighted_epoch,
                "acc": train_acc_epoch,
                "mean_rank_normalized": train_mean_rank_normalized_epoch,
                "rank_variance": train_rank_variance_epoch,
                "expected_rank": train_expected_rank_epoch,
                "mean_r_stacks": train_mean_r_stacks_epoch,
            },
            "val_metrics": {
                "loss": val_loss_epoch,
                "acc": val_acc_epoch,
                "mean_rank_normalized": val_mean_rank_normalized_epoch,
                "expected_rank": val_expected_rank_epoch,
                "rank_loss_weighted": val_rank_loss_weighted_epoch,
            } if val_loader is not None else None,
            "rank_loss_weight": getattr(criterion, "rank_loss_weight", None),
            "target_rank_normalized": getattr(criterion, "target_rank_normalized", None),
        }
            
        # Always update best_train_loss (regardless of validation)
        if train_loss_epoch < best_train_loss:
            best_train_loss = train_loss_epoch
            best_train_acc = train_acc_epoch
        
        # Save "best" based on validation loss (or training loss if no validation)
        use_val_for_best = val_loader is not None and val_loss_epoch is not None
        current_metric = val_loss_epoch if use_val_for_best else train_loss_epoch
        best_metric = best_val_loss if use_val_for_best else best_train_loss

        if current_metric < best_metric: #type: ignore
            if use_val_for_best:
                best_val_loss = current_metric
                best_val_acc = val_acc_epoch  #type: ignore
                print(f"  💾 Saved best model (val_loss: {val_loss_epoch:.4f})")
            else:
                print(f"  💾 Saved best model (train_loss: {train_loss_epoch:.4f})")
            torch.save(checkpoint, model_path)

        # Save model checkpoint every epoch (if enabled)
        if save_epoch_checkpoints:
            epoch_checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, epoch_checkpoint_path)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss_epoch,
            "train_ce_loss": train_ce_loss_epoch,
            "train_rank_loss": train_rank_loss_epoch,
            "train_rank_loss_weighted": train_rank_loss_weighted_epoch,
            "train_acc": train_acc_epoch,
            "train_mean_rank_normalized": train_mean_rank_normalized_epoch,
            "train_rank_variance": train_rank_variance_epoch,
            "train_expected_rank": train_expected_rank_epoch,
            "train_mean_r_stacks": train_mean_r_stacks_epoch,
            "best_train_loss": best_train_loss,
            "best_train_acc": best_train_acc,
            # Validation metrics
            "val_loss": val_loss_epoch,
            "val_acc": val_acc_epoch,
            "val_mean_rank_normalized": val_mean_rank_normalized_epoch,
            "val_expected_rank": val_expected_rank_epoch,
            "val_rank_loss_weighted": val_rank_loss_weighted_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "val_results_all_snr": val_results,
            # Track loss weights if you anneal them
            "rank_loss_weight": getattr(criterion, "rank_loss_weight", None),
            "rank_var_weight": getattr(criterion, "rank_var_weight", None),
            "rank_target_loss_weight": getattr(criterion, "rank_target_loss_weight", None),
        }
        epoch_logs.append(epoch_log)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Train Loss: {train_loss_epoch:.4f}   CE Loss: {train_ce_loss_epoch:.4f}   Weigthed Rank Loss: {train_rank_loss_weighted_epoch:.4f}")
        print(f"  Train Acc: {train_acc_epoch:.2f}%")
        r_stacks_str = ', '.join([f"{r:.2f}" for r in train_mean_r_stacks_epoch])
        print(f"  Expected Rank: {train_expected_rank_epoch:.2f} / {max_rank}   Avg R Stacks: [{r_stacks_str}]   Rank Variance: {train_rank_variance_epoch:.6f}")
        if val_loader is not None:
            print(f"  Val Loss: {val_loss_epoch:.4f}   Val Acc: {val_acc_epoch:.2f}%")
            print(f"  Val Expected Rank: {val_expected_rank_epoch:.2f} / {max_rank}")


        # Save history each epoch (so crashes still leave logs)
        with open(epoch_log_path, 'w') as f:
            json.dump(epoch_logs, f, indent=2)
        
        # Save step logs if available
        if log_every_n_steps is not None and len(step_logs) > 0:
            with open(step_log_path, 'w') as f:
                json.dump(step_logs, f, indent=2)
    
    return model, epoch_logs, step_logs


def rank_supervision_logic(args):
            # --- Compute rank target (if enabled) ---
            # When enable_rank_supervision=True, we test ALL ranks from 1 up to
            # the router's current suggestion for each sample, and find the
            # minimum rank that gives a correct prediction. This target is then
            # used to supervise the router via the rank_loss (per_sample_mse mode).
            #
            # Gradient flow: The target is detached (fixed), but r_normalized
            # (router output) has gradients. The loss (r_normalized - target)^2
            # pushes the router toward outputting the discovered min correct rank.
    raise NotImplementedError("This function has not been implemented yet.")

            # if enable_rank_supervision:
            #     batch_size = waveforms.size(0)
                
            #     # Compute spectrogram once (reuse for all rank evaluations)
            #     with torch.no_grad():
            #         x_spec = model.base.compute_spectrogram(waveforms)  # (B, spec_bins, T)
                
            #     # Get the router's current suggestion (ceiled to integer)
            #     r_cont_ceiled_clamped = r_cont_ceiled.detach().ceil().clamp(min=1, max=max_rank).long()  # (B,) or (B,1)
            #     if r_cont_ceiled_clamped.dim() == 2:
            #         r_cont_ceiled_clamped = r_cont_ceiled_clamped.squeeze(-1)  # (B,)
                
            #     # Track minimum correct rank per sample (initialize to max_rank + 1 = "never correct")
            #     min_correct_rank = torch.full((batch_size,), max_rank + 1, dtype=torch.float32, device=device)
                
            #     # Test ALL ranks from 1 up to router's suggestion for each sample
            #     # We iterate through ranks 1 to max(r_cont_ceiled_clamped) and use masking
            #     max_router_rank = r_cont_ceiled_clamped.max().item()  # max rank we need to test
                
            #     if rank_supervision_stable:
            #         # Track correctness at each rank for each sample
            #         correctness_matrix = torch.zeros((batch_size, max_router_rank), dtype=torch.bool, device=device)
                
            #     with torch.no_grad():
            #         for rank_val in range(1, int(max_router_rank) + 1):
            #             # Create mask: which samples have router suggestion >= current rank_val
            #             # (i.e., we should test this rank for these samples)
            #             sample_mask = (r_cont_ceiled_clamped >= rank_val)  # (B,)
                        
            #             if not sample_mask.any():
            #                 continue  # no samples need this rank tested
                        
            #             # Run base model at this rank for ALL samples (simpler than masked forward)
            #             ranks_tensor = torch.full((batch_size,), rank_val, dtype=torch.long, device=device)
            #             logits_k = model.base.forward(x_spec, ranks=ranks_tensor, x_is_spec=True)
            #             preds_k = logits_k.argmax(dim=1)  # (B,)
                        
            #             # Check correctness
            #             correct_k = (preds_k == labels)  # (B,) bool
                        
            #             if rank_supervision_stable:
            #                 # Store correctness for later stability check
            #                 correctness_matrix[:, rank_val - 1] = correct_k
            #             else:
            #                 # Original behavior: just find minimum correct rank
            #                 # Update min_correct_rank where:
            #                 # 1. This sample should be tested at this rank (sample_mask)
            #                 # 2. Prediction is correct
            #                 # 3. This rank is lower than current min
            #                 should_update = sample_mask & correct_k & (rank_val < min_correct_rank)
            #                 min_correct_rank = torch.where(
            #                     should_update,
            #                     torch.tensor(rank_val, dtype=torch.float32, device=device),
            #                     min_correct_rank
            #                 )
                
            #     # If stable mode, find minimum rank where prediction is correct AND stable at all higher ranks
            #     if rank_supervision_stable:
            #         for sample_idx in range(batch_size):
            #             max_test_rank = int(r_cont_ceiled_clamped[sample_idx].item())
                        
            #             # Check each rank from 1 to max_test_rank
            #             for rank_val in range(1, max_test_rank + 1):
            #                 # Check if correct at this rank
            #                 if not correctness_matrix[sample_idx, rank_val - 1]:
            #                     continue  # not correct at this rank, skip
                            
            #                 # Check if correct at ALL higher ranks up to max_test_rank
            #                 all_higher_correct = True
            #                 for higher_rank in range(rank_val + 1, max_test_rank + 1):
            #                     if not correctness_matrix[sample_idx, higher_rank - 1]:
            #                         all_higher_correct = False
            #                         break
                            
            #                 # If stable (correct at this rank and all higher), this is our target
            #                 if all_higher_correct:
            #                     min_correct_rank[sample_idx] = float(rank_val)
            #                     break  # found the minimum stable rank
                        
            #             # If no stable rank found, leave as max_rank + 1
            #             # This will be handled below to use router's current output (no supervision)
                
            #     # Identify samples that were never correct at any tested rank
            #     never_correct_mask = (min_correct_rank > max_rank)
                
            #     # For never-correct samples, set target to router's current output (no gradient push)
            #     # For correct samples, use the min_correct_rank
            #     min_correct_rank = torch.where(
            #         never_correct_mask,
            #         r_cont_ceiled.detach().squeeze(-1) if r_cont_ceiled.dim() == 2 else r_cont_ceiled.detach(),
            #         min_correct_rank
            #     )
                
            #     # Normalize target to [0, 1] range (same as router output)
            #     # r_normalized = (rank - 1) / (max_rank - 1)
            #     target_rank_normalized = (min_correct_rank - 1) / (max_rank - 1)