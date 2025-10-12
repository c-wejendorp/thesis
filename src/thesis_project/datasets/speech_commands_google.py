from pathlib import Path
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import pad

TARGET_LENGTH = 16000  # 1 second at 16kHz

class PadOrTrim():
    def __init__(self, max_len: int = TARGET_LENGTH, pad_value: float = 0.0,):
        self.max_len = max_len
        self.pad_value = pad_value
    
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return pad(w[:, :self.max_len], (0, max(0, self.max_len - w.shape[1])), value=self.pad_value)

class SpeechCommandsGoogle(SPEECHCOMMANDS):
    """
    Drop-in replacement for torchaudio.datasets.SPEECHCOMMANDS that:
      - accepts all original constructor args (root, subset, download, etc.)
      - adds label_to_idx / idx_to_label attributes
      - adds optional transform argument to apply to waveform on-the-fly in __getitem__

    """
    def __init__(self, *args, transform=PadOrTrim(), **kwargs):
        super().__init__(*args, **kwargs)  # forwards everything to SPEECHCOMMANDS
        self.transform = transform

        # Build label maps WITHOUT loading audio: use folder names of files in _walker
        self.labels = sorted({Path(p).parent.name for p in self._walker})
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        
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
            waveform = self.transform(waveform)  # expect shape [C, T]
        meta_data = {
            "sample_rate": sample_rate,
            "label": label,
            "speaker_id": speaker_id,
            "utterance_number": utterance_number
        }
        return waveform, self.label_to_idx[label], meta_data

if __name__ == "__main__":
    # Example usage
        
    dataset = SpeechCommandsGoogle(root="./data", subset="training", download=True)
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