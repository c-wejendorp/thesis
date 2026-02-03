import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.components.routers import GRURouter
from thesis_project.utils.paths import get_data_dir
from thesis_project.utils.compute_macs import compute_macs_base_model, compute_avg_rank_from_macs
from thesis_project.datasets import SpeechCommandsGoogle
from base_config_temp import cfg, noise_train_cfg
from thesis_project.training.key_word_spotting import fit_dynamic_model, DynamicRoutingLoss, create_dynamic_routing_loss

torch.manual_seed(42)
np.random.seed(42)

# BASE MODEL CONFIGURATION

BASE_MODEL_PATH = "model_runs/base/2025-12-14_18-48-50/best_model.pth"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")    
print(f"Using device: {DEVICE}")

base_model_state = torch.load(BASE_MODEL_PATH)
base_model = KWSBase(cfg).to(DEVICE)
base_model.load_state_dict(base_model_state)
base_model_full_rank = min(base_model.cfg.backbone.n_channels_ext, base_model.cfg.backbone.n_channels_int)
maximum_useful_rank = base_model_full_rank // 2

# ROUTER
HIDDEN_DIM = 48
NUM_GRU_LAYERS = 1
USE_GLOBAL_RANK = False
NUM_RANK_OUTPUTS = 3  # should match the number of stacks in the base model
assert (USE_GLOBAL_RANK and NUM_RANK_OUTPUTS == 1) or (not USE_GLOBAL_RANK and NUM_RANK_OUTPUTS >= 1)
# Have not implemented such that we can get ranks for the first stacks and then full rank for later stacks
assert NUM_RANK_OUTPUTS == len(base_model.backbone) or USE_GLOBAL_RANK, "NUM_RANK_OUTPUTS must match number of stacks in base model when USE_GLOBAL_RANK is False"
POOL_MODE = "subsample"  # options: None, 'avg', 'max', 'subsample'
POOL_REDUCTION_FACTOR = 2 # if pool_mode is 'subsample' this will pick every "kernel_size" element. Ignored if POOL_MODE is None

router = GRURouter(
    input_dim=base_model.frontend.out_channels,
    gru_hidden_dim=HIDDEN_DIM,
    num_gru_layers=NUM_GRU_LAYERS,
    max_rank=maximum_useful_rank,
    num_rank_outputs=NUM_RANK_OUTPUTS,
    last_layer_bias_init=3.0,
    pool_type=POOL_MODE,
    pool_reduction_factor=POOL_REDUCTION_FACTOR,
)

# DYNAMIC MODEL CONFIGURATION
FREEZE_BASE = True
LOW_RANK_FRONTEND = False

dynamic_model = KWSDynamic(
    base=base_model,
    router=router,
    use_global_rank=USE_GLOBAL_RANK,
    low_rank_frontend=LOW_RANK_FRONTEND,
    freeze_base=FREEZE_BASE
).to(DEVICE)

n_trainable = sum(p.numel() for p in dynamic_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in dynamic_model.parameters())
print(f"Total params: {total_params}")
print(f"Trainable params: {n_trainable}")

# DATASET AND LOADERS CONFIGURATION
PIN_MEMORY = False
NUM_WORKERS = 0
BATCH_SIZE = 64
USE_SUBSET = False  # whether to use a smaller subset of the dataset for quicker testing
SUBSET_SIZE = 1000  # number of samples in the subset if USE_SUBSET is True
DO_VALIDATION = False
VAL_SNR_VALUES = [-5, 0, 5, 10, 15, float('inf')]

train_set = SpeechCommandsGoogle(root=str(get_data_dir()), subset="training", download=True, **noise_train_cfg.model_dump())
if USE_SUBSET:
    train_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), SUBSET_SIZE, replace=False))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

if DO_VALIDATION:
    val_set = SpeechCommandsGoogle(root=str(get_data_dir()), subset="validation", download=True, **noise_train_cfg.model_dump())
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
else: 
    val_loader = None
    VAL_SNR_VALUES = None

# LOSS FUNCTION CONFIGURATION
RANK_LOSS_WEIGHT = 20
RANK_LOSS_MODE = "per_sample_mse"  # options: look ath the CrossEntropyPlusRankLoss class for available modes
COMPRESSED_BASE_MODEL_RANK_PR_STACK = [30,30,30] #somewhere near 30 seems to be acceptable from visual inspection
assert len(COMPRESSED_BASE_MODEL_RANK_PR_STACK) == len(base_model.backbone) # needs to match number of stacks in base model backbone
TARGET_MAC_FRACTION = 0.5 # target MACs as a fraction of the BASE model MACs with COMPRESSED_BASE_MODEL_RANK_PR_STACK

ENABLE_RANK_SUPERVISION = False
RANK_SUPERVISION_STABLE = False
# check that RANK_SUPERVISION_STABLE is not used when ENABLE_RANK_SUPERVISION is False
assert not (RANK_SUPERVISION_STABLE and not ENABLE_RANK_SUPERVISION), "RANK_SUPERVISION_STABLE doesn't make sense when ENABLE_RANK_SUPERVISION is False"


macs_base_model = compute_macs_base_model(
    base_model, 
    time_steps=63, 
    rank_pr_stack=COMPRESSED_BASE_MODEL_RANK_PR_STACK
    )
macs_router = router.compute_macs(sequence_length=63)

macs_maximum = macs_base_model - macs_router  # max MACs available for ranks to not exceed base model MACs
ranks_maximum = compute_avg_rank_from_macs(
    base_model, 
    time_steps=63, 
    macs=macs_maximum,
    num_stacks=len(base_model.backbone)
    )

macs_target = TARGET_MAC_FRACTION * macs_base_model
macs_target_budget = int(macs_target - macs_router)
ranks_target = compute_avg_rank_from_macs(
    base_model, 
    time_steps=63, 
    macs=macs_target_budget,
    num_stacks=len(base_model.backbone)
    )

avg_target_rank_normalized = (ranks_target - 1) / (maximum_useful_rank - 1)
print(f"Base model MACs: {macs_base_model}")
print(f"Router MACs: {macs_router}")
print(f"Maximum MACs for ranks: {macs_maximum}")
print(f"Maximum average rank for ranks: {ranks_maximum:.2f}")

print(f"Target MACs: {macs_target:.2f}")
print(f"MACs budget for ranks: {macs_target_budget}")
print(f"Target average rank: {ranks_target:.2f}")

criterion = create_dynamic_routing_loss(
    rank_loss_mode=RANK_LOSS_MODE,
    rank_loss_weight=RANK_LOSS_WEIGHT,
    target_rank_normalized= avg_target_rank_normalized
    )


# TRAINING CONFIGURATION
NUM_EPOCHS = 30 
#INIT_LEARNING_RATE = 1e-4
#MIN_LEARNING_RATE = 0.5e-4
INIT_LEARNING_RATE = 1e-5
MIN_LEARNING_RATE = 0.5e-5

optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=INIT_LEARNING_RATE) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=MIN_LEARNING_RATE,
    )

# Create run directory, ideally not in this file but for simplicity kept here
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#run_name = f"Some actually descriptive name"
run_dir = Path(f"model_runs/dynamic/{run_timestamp}")
#run_dir = Path(f"model_runs/dynamic/temp_run")
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run directory: {run_dir}")

# Train model
model, epoch_logs = fit_dynamic_model(
    model=dynamic_model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=DEVICE,
    epochs=NUM_EPOCHS,
    scheduler=scheduler,
    run_dir=str(run_dir),
    max_rank=maximum_useful_rank,
    val_loader=val_loader,
    val_snr_values=VAL_SNR_VALUES,
    val_epoch=1,
    enable_rank_supervision=ENABLE_RANK_SUPERVISION,
    rank_supervision_stable=RANK_SUPERVISION_STABLE,
    save_epoch_checkpoints=False,
    )

print("\nTraining completed!")
print(f"Model saved to: {run_dir}")

# Print final metrics
if epoch_logs is not None and len(epoch_logs) > 0:
    last_epoch = epoch_logs[-1]
    print(f"\nFinal metrics:")
    print(f"  Train loss: {last_epoch.get('train_loss'):.4f}")
    print(f"  Train acc: {last_epoch.get('train_acc'):.2f}%")
    print(f"  Expected rank: {last_epoch.get('train_expected_rank'):.1f}")
