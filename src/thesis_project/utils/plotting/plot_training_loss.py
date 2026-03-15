import json
import os
from pathlib import Path
from typing import Optional, Literal
import matplotlib.pyplot as plt


def visualize_training_logs(
    run_dir: str,
    log_type: Literal['epoch', 'step'] = 'epoch',
    save_dir: Optional[str] = None
) -> None:
    """
    Visualize training logs from a run directory.
    
    Args:
        run_dir: Path to the run directory containing history.json and/or step_history.json
        log_type: Type of log to visualize ('epoch' or 'step')
        save_dir: Directory to save plots. If None, saves to run_dir
    """
    run_dir = Path(run_dir)
    
    # Find the appropriate log file
    if log_type == 'epoch':
        log_path = run_dir / "history.json"
    else:
        log_path = run_dir / "step_history.json"
    
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return
    
    # Load logs
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    if len(logs) == 0:
        print("No logs to visualize")
        return
    
    # Determine save directory
    if save_dir is None:
        save_dir = run_dir
    else:
        save_dir = Path(save_dir)
    
    # Extract x-axis (epoch or step)
    if log_type == 'epoch':
        x_values = [log['epoch'] for log in logs]
        x_label = 'Epoch'
        prefix = 'epoch'
    else:
        x_values = [log['step'] for log in logs]
        x_label = 'Step'
        prefix = 'step'
    
    # Check if validation exists
    has_val = any(log.get('val_loss') is not None for log in logs)
    
    # For step logs, find epoch boundaries for secondary x-axis
    epoch_boundaries = {}
    if log_type == 'step':
        current_epoch = None
        for i, log in enumerate(logs):
            epoch = log.get('epoch')
            if epoch != current_epoch:
                epoch_boundaries[log['step']] = epoch
                current_epoch = epoch
    
    # Determine key names based on log type
    if log_type == 'epoch':
        train_loss_key = 'train_loss'
        train_ce_key = 'train_ce_loss'
        train_rank_key = 'train_rank_loss_weighted'
        train_acc_key = 'train_acc'
        val_loss_key = 'val_loss'
        val_ce_key = 'val_ce_loss' if logs[0].get('val_ce_loss') is not None else None
        val_rank_key = 'val_rank_loss_weighted'
        val_acc_key = 'val_acc'
    else:  # step logs
        train_loss_key = 'loss'
        train_ce_key = 'ce_loss'
        train_rank_key = 'rank_loss_weighted'
        train_acc_key = 'running_acc'
        val_loss_key = None  # step logs don't have validation
        val_ce_key = None
        val_rank_key = None
        val_acc_key = None
        has_val = False
    
    # --- Plot 1: Training Loss Components ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_total_loss = [log[train_loss_key] for log in logs]
    train_ce_loss = [log[train_ce_key] for log in logs]
    train_rank_loss = [log[train_rank_key] for log in logs]
    
    ax.plot(x_values, train_total_loss, label='Total Loss', linewidth=2)
    ax.plot(x_values, train_ce_loss, label='CE Loss', linewidth=2)
    ax.plot(x_values, train_rank_loss, label='Weighted Rank Loss', linewidth=2)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add epoch markers for step logs
    if log_type == 'step' and epoch_boundaries:
        epoch_steps = list(epoch_boundaries.keys())
        ax.set_xticks(epoch_steps)
    
    plt.tight_layout()
    train_loss_path = save_dir / f'{prefix}_train_loss_components.png'
    plt.savefig(train_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {train_loss_path}")
    
    # --- Plot 2: Validation Loss Components (if available) ---
    if has_val and val_loss_key is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter out None values for validation
        val_x = [x for i, x in enumerate(x_values) if logs[i].get(val_loss_key) is not None]
        val_total_loss = [log[val_loss_key] for log in logs if log.get(val_loss_key) is not None]
        
        ax.plot(val_x, val_total_loss, label='Total Val Loss', linewidth=2, marker='o')
        
        # Check if component losses are available
        if val_ce_key and any(log.get(val_ce_key) is not None for log in logs):
            val_ce_loss = [log[val_ce_key] for log in logs if log.get(val_ce_key) is not None]
            ax.plot(val_x, val_ce_loss, label='Val CE Loss', linewidth=2, marker='s')
        
        if val_rank_key and any(log.get(val_rank_key) is not None for log in logs):
            val_rank_loss = [log[val_rank_key] for log in logs if log.get(val_rank_key) is not None]
            ax.plot(val_x, val_rank_loss, label='Val Weighted Rank Loss', linewidth=2, marker='^')
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add epoch markers for step logs
        if log_type == 'step' and epoch_boundaries:
            epoch_steps = list(epoch_boundaries.keys())
            ax.set_xticks(epoch_steps)
        
        plt.tight_layout()
        val_loss_path = save_dir / f'{prefix}_val_loss_components.png'
        plt.savefig(val_loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {val_loss_path}")
    
    # --- Plot 3: Train vs Val Loss and Accuracy ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss subplot
    ax1.plot(x_values, train_total_loss, label='Train Loss', linewidth=2)
    if has_val and val_loss_key is not None:
        val_x = [x for i, x in enumerate(x_values) if logs[i].get(val_loss_key) is not None]
        val_total_loss = [log[val_loss_key] for log in logs if log.get(val_loss_key) is not None]
        ax1.plot(val_x, val_total_loss, label='Val Loss', linewidth=2, marker='o')
    
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    train_acc = [log[train_acc_key] for log in logs]
    ax2.plot(x_values, train_acc, label='Train Accuracy', linewidth=2)
    if has_val and val_acc_key is not None:
        val_x = [x for i, x in enumerate(x_values) if logs[i].get(val_acc_key) is not None]
        val_acc = [log[val_acc_key] for log in logs if log.get(val_acc_key) is not None]
        ax2.plot(val_x, val_acc, label='Val Accuracy', linewidth=2, marker='o')
    
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add epoch markers for step logs
    if log_type == 'step' and epoch_boundaries:
        epoch_steps = list(epoch_boundaries.keys())
        ax1.set_xticks(epoch_steps)
        ax2.set_xticks(epoch_steps)
    
    plt.tight_layout()
    summary_path = save_dir / f'{prefix}_train_val_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_path}")
    
    # --- Plot 4: Learning Rate Schedule ---
    # Check if learning rate is available in logs
    has_lr = 'lr' in logs[0] if logs else False
    
    if has_lr:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lr_values = [log['lr'] for log in logs]
        ax.plot(x_values, lr_values, label='Learning Rate', linewidth=2, color='orange')
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')  # Use log scale for better visualization
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add epoch markers for step logs
        if log_type == 'step' and epoch_boundaries:
            epoch_steps = list(epoch_boundaries.keys())
            ax.set_xticks(epoch_steps)
        
        plt.tight_layout()
        lr_path = save_dir / f'{prefix}_learning_rate.png'
        plt.savefig(lr_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {lr_path}")
    
    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    # Example usage
    
    # Visualize epoch logs
    visualize_training_logs(
        run_dir="model_runs/dynamic/2026-02-03_18-19-34",
        log_type='step'
    )
    
    # Visualize step logs
    # visualize_training_logs(
    #     run_dir="model_runs/dynamic/2026-02-03_15-02-46",
    #     log_type='step'
    # )
    
    # Save to custom directory
    # visualize_training_logs(
    #     run_dir="model_runs/dynamic/2026-02-03_15-02-46",
    #     log_type='epoch',
    #     save_dir="custom_plots"
    # )
