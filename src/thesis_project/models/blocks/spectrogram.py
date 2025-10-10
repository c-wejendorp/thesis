import torch
import torchaudio.transforms as T
from torch import Tensor
from typing import Optional


class SpectrogramBlock(torch.nn.Module):
    """
    A configurable spectrogram (or mel-spectrogram) transformation block.

    Args:
        sample_rate (int): Audio sample rate in Hz. Default: 16000.
        n_fft (int): FFT window size. Default: 1024.
        hop_length (int): Hop (stride) length between STFT windows. Default: 256.
        n_mels (Optional[int]): Number of mel filterbanks. If None, uses linear frequency bins.
        power (Optional[float]): Exponent for magnitude:
            - 2.0 → power spectrogram (energy)
            - 1.0 → magnitude spectrogram (amplitude) 
            - None → complex STFT
        logarithm (bool): If True, converts output to log scale (dB) using the AmplitudeToDB transform. Default: True.
    """

    #TODO add **keyword option for different window functions, window lengths etc. Remember to pass to inverse transforms as well
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: Optional[int] = None,
        power: Optional[float] = 1.0,    # Default: magnitude,
        logarithm: bool = True,          # Default: log scale (dB)
    ) -> None:
        
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.logarithm = logarithm

        # Choose between regular or mel spectrogram
        if n_mels is None:
            self.spec = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                power=power,
            )
        else:
            self.spec = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=power, # pyright: ignore[reportArgumentType]
            )

        # Optional logarithmic (decibel) conversion
        if logarithm:
            # check that power is valid
            if power not in (1.0, 2.0):
                raise ValueError("Logarithmic scaling requires power to be 1.0 or 2.0")
            stype = "power" if power == 2.0 else "magnitude"
            self.db = T.AmplitudeToDB(stype=stype)
        else:
            self.db = None

        # Inverse transforms
        self.inv = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        self.inv_mel = (
            T.InverseMelScale(
                n_stft=n_fft // 2 + 1,
                n_mels=n_mels,
                sample_rate=sample_rate,
            )
            if n_mels is not None
            else None
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Compute a (mel) spectrogram or log-(mel) spectrogram from a waveform.

        Args:
            waveform (Tensor): Audio waveform tensor of shape [B, T] or [B, 1, T].

        Returns:
            Tensor: Spectrogram tensor of shape [B, F, T].
        """
        spec = self.spec(waveform)
        if self.db is not None:
            spec = self.db(spec)
        return spec
    
    def _db_to_amplitude(self, spec_db: Tensor) -> Tensor:
        """Convert dB-scaled spectrogram back to linear amplitude or power."""
        if not self.logarithm:
            return spec_db
        if self.power == 2.0:
            return torch.pow(10.0, spec_db / 10.0)
        elif self.power == 1.0:
            return torch.pow(10.0, spec_db / 20.0)
        else:
            raise ValueError("Cannot invert dB scale without a valid power value (1.0 or 2.0).")

    def inverse(self, spec: Tensor) -> Tensor:
        """
        Invert a spectrogram (or mel-spectrogram) back to waveform.
        """
        # Undo dB scaling if it was applied
        spec = self._db_to_amplitude(spec)

        # Undo mel scaling if present
        if self.n_mels is not None and self.inv_mel is not None:
            spec = self.inv_mel(spec)

        # Finally, inverse STFT
        return self.inv(spec)