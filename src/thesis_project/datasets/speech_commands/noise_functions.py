# noise_functions.py

from pathlib import Path
from typing import List, Tuple
import random
from tqdm import tqdm

import torch
from torch import Tensor
import torchaudio
import torch.nn.functional as F


def _load_and_resample(path: str, sample_rate: int) -> Tensor:
    wav, sr = torchaudio.load(path)  # [C, T]
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav

def _get_chunk(wav: Tensor, start: int, target_length: int) -> Tensor:
    """
    wav: [C, T], return [C, target_length] starting at `start`,
    padding with zeros at the end if needed.
    """
    num_samples = wav.shape[1]
    end = start + target_length
    if end <= num_samples:
        return wav[:, start:end]

    pad_len = end - num_samples
    wav = F.pad(wav, (0, pad_len))
    return wav[:, start:end]

def load_random_noise_chunk(
    noise_paths: List[str],
    target_length: int,
    sample_rate: int,
    rng: random.Random = None,
) -> Tuple[Tensor, str]:
    """
    Load a random noise chunk. If rng is provided, selection is deterministic.
    """
    if rng is not None:
        noise_path = rng.choice(noise_paths)
    else:
        noise_path = random.choice(noise_paths)
    
    wav = _load_and_resample(noise_path, sample_rate)
    num_samples = wav.shape[1]
    if num_samples <= target_length:
        return _get_chunk(wav, 0, target_length), Path(noise_path).stem
    
    if rng is not None:
        start = rng.randint(0, num_samples - target_length)
    else:
        start = random.randint(0, num_samples - target_length)
    return _get_chunk(wav, start, target_length), Path(noise_path).stem


def choose_deterministic_noise_chunk(
    noise_paths: List[str],
    rng: random.Random,
    target_length: int,
    sample_rate: int,
) -> Tuple[Tensor, str]:
    """
    Deterministic version driven by a provided rng:
      - picks a file from noise_paths using rng
      - picks a start offset using rng
      - returns (noise_type, chunk [C, target_length])

    noise_type is inferred from the file name stem.
    """
    noise_path = rng.choice(noise_paths)
    wav = _load_and_resample(noise_path, sample_rate)
    num_samples = wav.shape[1]

    if num_samples <= target_length:
        start = 0
    else:
        start = rng.randint(0, num_samples - target_length)
    return _get_chunk(wav, start, target_length), Path(noise_path).stem

def attach_deterministic_noise_to_samples(
    samples: list[dict],
    noise_paths: list[str],
    target_length: int,
    sample_rate: int,
    seed: int = 12345,
) -> list[dict]:
    """
    For each sample with sample["kind"] in `kinds`, attach:
        - sample["deterministic_noise_type"]
        - sample["deterministic_noise"]

    Returns `samples` (modified in place).
    """
    assert len(noise_paths) > 0, (
        "attach_deterministic_noise_to_samples(): noise_paths is empty. "
        "Deterministic noise requires valid noise files."
    )
    rng = random.Random(seed)

    for sample in tqdm(samples, desc="Adding noise", unit="sample", total=len(samples)):
        noise_chunk, noise_type = choose_deterministic_noise_chunk(
            noise_paths=noise_paths,
            rng=rng,
            target_length=target_length,
            sample_rate=sample_rate,
        )
        sample["noise"] = noise_chunk
        sample["noise_type"] = noise_type

    return samples

def add_noise_at_snr(
    waveform: Tensor,
    noise: Tensor,
    snr_db: float,
    use_speech_region: bool = False,
    speech_threshold: float = 0.01,
    min_speech_fraction: float = 0.05,
) -> Tensor:
    """
    Add noise to `waveform` at a given SNR (in dB).

    Args:
        waveform: Tensor of shape [T] or [C, T], clean speech.
        noise:    Tensor of same shape as waveform (or broadcastable).
        snr_db:   Desired SNR in dB (signal/noise).
        use_speech_region:
            If True, estimate RMS only over regions where speech is present
            using a simple threshold on amplitude. If no speech is found,
            falls back to full-signal RMS.
        speech_threshold:
            Threshold on absolute amplitude to consider a sample as "speech".
            Assumes waveform is roughly in [-1, 1].
        min_speech_fraction:
            If the fraction of samples above threshold is below this,
            we treat it as "no speech" and fall back to full-signal RMS.

    Returns:
        noisy_waveform: waveform + scaled_noise, with noise scaled so that
                        the (RMS-based) SNR is snr_db.
    """
    # Ensure tensors are on same device / dtype
    noise = noise.to(waveform.device, waveform.dtype)

    # Handle inf / no-noise case
    if not torch.isfinite(torch.tensor(snr_db)):
        # e.g. snr_db == float("inf") → return clean (no added noise)
        return waveform

    # Flatten channel dimension for mask computation (keep time dimension)
    if waveform.dim() == 1:
        mono = waveform
    elif waveform.dim() == 2:
        # [C, T] → average over channels
        mono = waveform.mean(dim=0)
    else:
        raise ValueError(f"waveform must be 1D or 2D, got shape {waveform.shape}")

    # Decide region over which to compute RMS
    if use_speech_region:
        # Simple VAD-style mask: absolute amplitude above threshold
        speech_mask = mono.abs() > speech_threshold

        # If almost no samples are above threshold, fall back to full
        if speech_mask.float().mean().item() < min_speech_fraction:
            region_mask = torch.ones_like(mono, dtype=torch.bool)
        else:
            region_mask = speech_mask
    else:
        # Full waveform
        region_mask = torch.ones_like(mono, dtype=torch.bool)

    # Broadcast mask to match waveform/noise shape
    if waveform.dim() == 1:
        sig_region = waveform[region_mask]
        noise_region = noise[region_mask]
    else:  # [C, T]
        sig_region = waveform[:, region_mask]
        noise_region = noise[:, region_mask]

    # Compute RMS in the chosen region
    eps = 1e-12
    signal_rms = sig_region.pow(2).mean().sqrt()
    noise_rms = noise_region.pow(2).mean().sqrt() + eps  # avoid div-by-zero

    # Desired amplitude ratio from dB: SNR_dB = 20 * log10(As/An)
    snr_linear = 10.0 ** (snr_db / 20.0)

    # We want: signal_rms / (noise_rms * scale) = snr_linear
    # → scale = signal_rms / (snr_linear * noise_rms)
    scale = signal_rms / (snr_linear * noise_rms + eps)

    scaled_noise = noise * scale

    noisy_waveform = waveform + scaled_noise
    return noisy_waveform
