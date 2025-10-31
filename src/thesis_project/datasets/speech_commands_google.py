from pathlib import Path
from typing import Optional
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
from collections import defaultdict
from collections import Counter
import torch
from torch.nn.functional import pad

TARGET_LENGTH = 16000  # 1 second at 16kHz

class PadOrTrim():
    def __init__(self, max_len: int = TARGET_LENGTH, pad_value: float = 0.0,):
        self.max_len = max_len
        self.pad_value = pad_value
    
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return pad(w[:, :self.max_len], (0, max(0, self.max_len - w.shape[1])), value=self.pad_value)
    
def add_noise_at_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise with given SNR (dB)."""
    rms_x = torch.sqrt(torch.mean(x**2) + 1e-8)
    noise = torch.randn_like(x)
    rms_n = torch.sqrt(torch.mean(noise**2) + 1e-8)
    desired_rms_n = rms_x / (10 ** (snr_db / 20))
    noise = noise * (desired_rms_n / rms_n)
    return torch.clamp(x + noise, -1.0, 1.0)

class SpeechCommandsGoogle(SPEECHCOMMANDS):
    """
    Drop-in replacement for torchaudio.datasets.SPEECHCOMMANDS that:
      - accepts all original constructor args (root, subset, download, etc.)
      - adds label_to_idx / idx_to_label attributes
      - adds optional transform argument to apply to waveform on-the-fly in __getitem__

    """
    def __init__(self, *args, noise_mode: Optional[str] = None, noise_params: Optional[list[float]] = None, transform=PadOrTrim(), **kwargs):
        super().__init__(*args, **kwargs)  # forwards everything to SPEECHCOMMANDS

        # --- Noise init ---
        if bool(noise_mode) != bool(noise_params):
            raise ValueError("Both 'noise_mode' and 'noise_params' must be provided together, or neither.")

        if noise_mode is not None and noise_mode not in ['snr','std']:
            raise ValueError(f"Invalid noise_mode '{noise_mode}'. Supported modes are 'snr' and 'std'.")
        
        self.noise_mode = noise_mode if noise_mode is not None else None
        self.noise_params = list(noise_params) if noise_params is not None else None

        # --- Assign noise parameters evenly per label ---
        if self.noise_params is not None:
            if len(self.noise_params) == 0:
                raise ValueError("noise_params must not be empty if provided.")
            label_to_indices = defaultdict(list)
            for i, path in enumerate(self._walker):
                lbl = Path(path).parent.name
                label_to_indices[lbl].append(i)

            self.noise_param_per_index: Optional[list[float]] = [0.0] * len(self._walker)
            L = len(self.noise_params)
            for _, idxs in label_to_indices.items():
                for j, idx in enumerate(idxs):
                    self.noise_param_per_index[idx] = float(self.noise_params[j % L])
        else:
            self.noise_param_per_index = None
        

        # --- Transforms ---
        self.transform = transform

        # --- Label maps ---
        self.labels = sorted({Path(p).parent.name for p in self._walker})
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        # label counts
        self.label_counts = Counter()
        for path in self._walker:
            lbl = Path(path).parent.name
            self.label_counts[lbl] += 1

        # --- Assign noise parameters evenly per label ---
        if self.noise_params is not None:
            if len(self.noise_params) == 0:
                raise ValueError("noise_params must not be empty if provided.")
            label_to_indices = defaultdict(list)
            for i, path in enumerate(self._walker):
                lbl = Path(path).parent.name
                label_to_indices[lbl].append(i)

            self.noise_param_per_index: Optional[list[float]] = [0.0] * len(self._walker)
            L = len(self.noise_params)
            for _, idxs in label_to_indices.items():
                for j, idx in enumerate(idxs):
                    self.noise_param_per_index[idx] = float(self.noise_params[j % L])
        else:
            self.noise_param_per_index = None

    # function that returns the indices of all samples that has the given a noise parameter value
    def get_indices_by_noise_param(self, noise_param: float) -> Optional[list[int]]:
        if self.noise_param_per_index is None:
            raise ValueError("Dataset was not initialized with noise parameters.")
        return [i for i, v in enumerate(self.noise_param_per_index) if v == noise_param]
    
    #TODO: implement SNR mode and figure out if noise_mode should be keyword argument, which would allow to use this function independently of dataset initialization
    def add_noise_to_waveform(self, waveform: torch.Tensor, noise_param: float) -> torch.Tensor:
        if self.noise_mode == 'std':
            noise = torch.randn_like(waveform) * noise_param # simple Gaussian noise with given std
            return torch.clamp(waveform + noise, -1.0, 1.0) # clamp to valid range
        elif self.noise_mode == 'snr':
            raise NotImplementedError("SNR mode is not implemented yet.")
            return add_noise_at_snr(waveform, noise_param)
        else:
            raise ValueError("Dataset was not initialized with noise parameters.")
        

    def get_item_with_given_noise_param(self, n: int, noise_param: float) -> tuple[torch.Tensor, int, dict]:
        """
        Fetch the sample via the standard __getitem__ (which populates metadata['raw']),
        then override the noise by re-noising the CLEAN waveform stored in metadata['raw'].
        """
        # Get the standard (possibly noisy) sample; crucially, metadata['raw'] is clean.
        _, label, metadata = self.__getitem__(n)

        # Use the clean, transformed waveform as source and make a copy
        clean = metadata["raw_signal"].clone()
        # Apply the desired noise level
        noisy = self.add_noise_to_waveform(clean, noise_param)

        # Update metadata to reflect the override
        metadata["noise_param"] = noise_param  # explicit override
        return noisy, label, metadata
   

    # NOTE: If more information is needed use get_metadata from parent class.
            # When we load the data using the superclass, we get the waveform in the range [-1, 1].
            # torchaudio normalizes PCM 16-bit values (range [-32768, 32767]) to floating-point [-1, 1]
            # using a fixed linear mapping (not per-file normalization):
            #   x_min = -32768, x_max = 32767
            #   a = -1, b = 1
            #   x_float = ((x - x_min) / (x_max - x_min)) * (b - a) + a
            # in practice is just dividing by 32768.0 since -32768 / 32768 = -1 and 32767 / 32768 ~ 1.

    def __getitem__(self, n) -> tuple[torch.Tensor, int, dict]:
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        if self.transform is not None:
            waveform = self.transform(waveform)  # [C, T]

        meta_data = {
            "sample_rate": sample_rate,
            "label": label,
            "speaker_id": speaker_id,
            "utterance_number": utterance_number,
            "raw_signal": waveform.clone(), # save clean version
        }
        
        if self.noise_param_per_index is not None:
            noise_param = self.noise_param_per_index[n]
            waveform = self.add_noise_to_waveform(waveform, noise_param)

            meta_data["noise_mode"] = self.noise_mode
            meta_data["noise_param"] = noise_param
            
        return waveform, self.label_to_idx[label], meta_data

if __name__ == "__main__":
    # Example usage

    dataset = SpeechCommandsGoogle(noise_mode='std', noise_params=[0.1, 0.2, 0.3], root="./data", subset="training", download=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # inspect a single item
    wave, label, meta = dataset[1]
    #print(wave.shape)  # torch.Size([1, 16000]) # (C, T), only one channel for this dataset
    #print(label, dataset.idx_to_label[label])  # 0 'backward'
    #print(meta)  # {'sample_rate': 16000, 'label': 'backward', 'speaker_id': '0165e0e8', 'utterance_number': 0}

    for waves, labels, metabatch in loader:
        print(waves.shape)  # torch.Size([32, 1, 16000]) # (B, C, T), only one channel for this dataset
        print([dataset.idx_to_label[idx.item()] for idx in labels])  # ['yes', 'no', 'up', ...]
        #print(metabatch)  # {'sample_rate': [16000, 16000, ...], 'label': ['yes', 'no', ...], ...} 
        break