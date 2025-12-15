import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic
from thesis_project.models.components.routers import GRURouter
from thesis_project.utils.paths import get_data_dir
from thesis_project.datasets import SpeechCommandsGoogle
from base_config_temp import cfg, noise_train_cfg, noise_eval_cfg
from thesis_project.training.key_word_spotting import CrossEntropyPlusRankLoss, fit_dynamic_model, validate_dynamic_model

#TODO: Should not be hardcoded here
# Training configuration
batch_size = 64
num_epochs = 10
init_learning_rate = 1e-4

maximum_useful_rank = 64
target_rank_mean = 16
target_rank_std = 4

# Scale factor from rank-space → normalized space
scale = 1.0 / (maximum_useful_rank - 1)

# Normalized mean and std
target_rank_mean_norm = (target_rank_mean - 1) * scale
target_rank_std_norm = target_rank_std * scale

# Normalized distribution in [~0, ~1]
target_rank_distribution = torch.distributions.Normal(
    loc=target_rank_mean_norm,
    scale=target_rank_std_norm
)


torch.manual_seed(42)
np.random.seed(42)

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

# ACTUALTRAINING  SETUP

# Model
base_model = KWSBase(cfg).to(device)
base_model.load_state_dict(torch.load("model_runs/base/2025-12-14_18-48-50/best_model.pth"))
router = GRURouter(
    input_dim=base_model.spectrogram_bins, #type: ignore
    fc_hidden_dim=128,
    gru_hidden_dim=64,
    num_gru_layers=1,
    max_rank=maximum_useful_rank,
    last_layer_bias_init= 4.0 # to bias towards full rank at start
    #last_layer_bias_init= None,
    ).to(device)


n_trainable_router = sum(p.numel() for p in router.parameters() if p.requires_grad)
print(f"Router trainable params: {n_trainable_router}")

model = KWSDynamic(base=base_model, router=router).to(device)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
n_total_base = sum(p.numel() for p in base_model.parameters())
print(f"Base model params: {n_total_base} (all frozen)")
print(f"Trainable params: {n_trainable}/{n_total}")

#Datasets and dataloaders
data_dir = get_data_dir()
train_set = SpeechCommandsGoogle(root=str(data_dir), subset="training", download=True, **noise_train_cfg.model_dump())

# only usee 1000 random samples for quick testing
# indices = np.random.choice(len(train_set), size=1000, replace=False)
# train_set = torch.utils.data.Subset(train_set, indices) #type: ignore

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory = pin_memory, num_workers=num_workers)

val_set = SpeechCommandsGoogle(root=str(data_dir), subset="validation", download=True, **noise_eval_cfg.model_dump())
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory = pin_memory, num_workers=num_workers)

# Loss function and optimizer
criterion = CrossEntropyPlusRankLoss(
    target_rank_normalized_distribution=target_rank_distribution,
    rank_loss_weight=1.0,
    rank_loss_mode="per_sample_mse",
    rank_var_weight=0.0,
)
optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

#TODO: potentially play around with scheduler later
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",      # we want to minimize val loss
#     factor=0.5,      # reduce LR by a factor of 0.5
#     patience=3,      # epochs with no improvement before reducing LR
# )
scheduler = None

model, history = fit_dynamic_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=num_epochs,
    scheduler=scheduler,
    run_dir=None, # automatically create timestamped folder, but can be set customly
    max_rank=maximum_useful_rank,
)