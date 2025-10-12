# test spectogram block
import torch
from torch.utils.data import DataLoader


from thesis_project.datasets import SpeechCommandsGoogle
from thesis_project.models.blocks.spectrogram import SpectrogramBlock
from thesis_project.utils.paths import get_data_dir

data_dir = get_data_dir()
dataset = SpeechCommandsGoogle(root=str(data_dir), subset="training", download=True)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# inspect a single item
waveform, label, metadata = dataset[110]