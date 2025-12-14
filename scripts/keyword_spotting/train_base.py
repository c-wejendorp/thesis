import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from thesis_project.models.keyword_spotting.base_model import KWSBase
from thesis_project.utils.paths import get_data_dir
from thesis_project.datasets import SpeechCommandsGoogle
from base_config_temp import cfg, noise_train_cfg, noise_eval_cfg
from thesis_project.training.key_word_spotting import fit_base_model

#TODO: read all configs from yaml file instead of the base_config_temp.py

#TODO: Remove this when above is fixed
# Training configuration
batch_size = 64
num_epochs = 30
init_learning_rate = 1e-3

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
model = KWSBase(cfg).to(device)

#Datasets and dataloaders
data_dir = get_data_dir()
train_set = SpeechCommandsGoogle(root=str(data_dir), subset="training", download=True, **noise_train_cfg.model_dump())
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory = pin_memory, num_workers=num_workers)

val_set = SpeechCommandsGoogle(root=str(data_dir), subset="validation", download=True, **noise_eval_cfg.model_dump())
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory = pin_memory, num_workers=num_workers)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
#TODO: potentially play around with scheduler later
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",      # we want to minimize val loss
    factor=0.5,      # reduce LR by a factor of 0.5
    patience=3,      # epochs with no improvement before reducing LR
)

#TODO: 
# Which SNRs you want to evaluate at each epoch
VAL_SNR_VALUES = [-5, 0, 5, 10,15,float('inf')]  # inf means no noise added
MAIN_VAL_SNR = float('inf')  # which SNR to use as "main" val metric (for curves)

# Train the model
model, history = fit_base_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=num_epochs,
    snr_values=VAL_SNR_VALUES,
    scheduler=scheduler,
    main_val_snr=MAIN_VAL_SNR,
    run_dir=None,  # automatically create timestamped folder
)