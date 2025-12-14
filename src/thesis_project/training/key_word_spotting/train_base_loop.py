import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import os


def validate_model(model, val_loader, criterion, device, snr_values=None, verbose=True):
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

                logits = model(waveforms)
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



import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime


# -------------------------------------------------------------------------
# Helper: create timestamped run folder under model_runs/base/
# -------------------------------------------------------------------------
def create_run_folder(base_dir="model_runs/base"):
    """
    Creates a unique folder with timestamp inside base_dir.

    Example:
        model_runs/base/2025-12-14_18-04-11/

    Returns:
        run_dir (str): The created directory path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# -------------------------------------------------------------------------
# Training Loop (Modular + Timestamped Save Location)
# -------------------------------------------------------------------------
def fit_base_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    snr_values,
    scheduler=None,
    main_val_snr=float("inf"),
    run_dir=None,
):
    """
    Full training loop replicating your original behavior, but saving results
    under model_runs/base/TIMESTAMP folder.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        criterion: Loss function.
        device: torch.device
        epochs: number of epochs.
        snr_values: list of SNRs to evaluate during validation.
        scheduler: optional LR scheduler (supports ReduceLROnPlateau).
        main_val_snr: SNR used as the primary validation metric.
        run_dir: manually specify folder for logs; if None a TIMESTAMP folder is created.
    """

    # -------------------------------------------------------------------------
    # Set up timestamped folder
    # -------------------------------------------------------------------------
    if run_dir is None:
        run_dir = create_run_folder("model_runs/base")

    model_path = os.path.join(run_dir, "best_model.pth")
    history_path = os.path.join(run_dir, "history.npy")

    print(f"\n🔹 All outputs will be saved under: {run_dir}\n")

    # -------------------------------------------------------------------------
    # Prepare history dict
    # -------------------------------------------------------------------------
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],         # only for main_val_snr
        "val_acc": [],          # only for main_val_snr
        "val_per_snr": [],      # list of dicts
        "lr": [],
    }

    best_val_acc = -float("inf")
    # -------------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -------------------------------------------------------------------------
    for epoch in range(epochs):

        # ===============================
        # ----------- TRAINING ----------
        # ===============================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for waveforms, labels, _ in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # --- Stats ---
            batch_size = labels.size(0)
            train_total += batch_size
            train_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * train_correct / train_total:.2f}%"
            })

        train_loss_epoch = train_loss / train_total
        train_acc_epoch = 100.0 * train_correct / train_total

        # ===============================
        # ---------- VALIDATION ---------
        # ===============================
        val_results = validate_model(
            model,
            val_loader,
            criterion,
            device,
            snr_values=snr_values,
            verbose=False,
        )

        main_val = val_results[main_val_snr]
        val_loss_epoch = main_val["loss"]
        val_acc_epoch = main_val["acc"]

        # ===============================
        # -------- SCHEDULER STEP -------
        # ===============================
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss_epoch)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ===============================
        # -------- SAVE HISTORY ---------
        # ===============================
        history["train_loss"].append(train_loss_epoch)
        history["train_acc"].append(train_acc_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["val_acc"].append(val_acc_epoch)
        history["val_per_snr"].append(val_results)
        history["lr"].append(current_lr)

        print(f"\n  LR: {current_lr:.6f}")

        # ===============================
        # ----- BEST MODEL CHECKPOINT ---
        # ===============================
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            torch.save(model.state_dict(), model_path)
            print(f">>> New best model! Val Acc = {best_val_acc:.2f}% → Saved to {model_path}")

        # ===============================
        # --------- EPOCH SUMMARY -------
        # ===============================
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss_epoch:.4f}   Train Acc: {train_acc_epoch:.2f}%")
        print(f"  Val Loss (SNR={main_val_snr}): {val_loss_epoch:.4f}   Val Acc: {val_acc_epoch:.2f}%")

        print("\n=== Validation accuracy per SNR ===")
        for snr, stats in val_results.items():
            print(f"  SNR={str(snr):>6}:  acc={stats['acc']:.2f}%   loss={stats['loss']:.4f}")
        print("-" * 60)

    # -------------------------------------------------------------------------
    # SAVE TRAINING HISTORY
    # -------------------------------------------------------------------------
    np.save(history_path, history) #type: ignore
    print(f"\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"History saved to {history_path}")

    return model, history

