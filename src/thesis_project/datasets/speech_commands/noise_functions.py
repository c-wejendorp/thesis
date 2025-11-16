# audio_utils.py

import random

import torch
import torchaudio
from torch import Tensor


def load_random_noise_chunk(
    noise_paths: list[str],
    target_length: int,
    sample_rate: int,
) -> Tensor:
    """
    Load a random chunk of noise from a list of noise wav files.

    Returns:
        Tensor [C, target_length]
    """
    if not noise_paths:
        return torch.zeros(1, target_length)

    noise_path = random.choice(noise_paths)
    wav, sr = torchaudio.load(noise_path)  # [C, T]

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    num_samples = wav.shape[1]
    if num_samples <= target_length:
        pad_len = target_length - num_samples
        wav = torch.nn.functional.pad(wav, (0, pad_len))
        return wav[:, :target_length]

    start = random.randint(0, num_samples - target_length)
    end = start + target_length
    return wav[:, start:end]

#TODO make such that we can use diffrent kind of noise and also maybe do SNR calculations diffrently
def add_noise_at_snr(waveform: Tensor, snr_db: float) -> Tensor:
    """
    Add white Gaussian noise to `waveform` at the given SNR (dB).

    Args:
        waveform: Tensor [C, T]
        snr_db: desired SNR in dB

    Returns:
        Noisy waveform [C, T]
    """
    eps = 1e-8
    rms_signal = torch.sqrt(torch.mean(waveform**2) + eps)

    noise = torch.randn_like(waveform)
    rms_noise = torch.sqrt(torch.mean(noise**2) + eps)

    # snr_db = 20 * log10(rms_signal / rms_noise_scaled)
    desired_rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise_scale = desired_rms_noise / (rms_noise + eps)
    noise = noise * noise_scale

    return waveform + noise
