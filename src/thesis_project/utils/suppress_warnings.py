# suppress_warnings.py
import warnings

def suppress_audio_warnings():
    """Suppress torchaudio migration warnings."""
    warnings.filterwarnings(
        "ignore",
        message=".*torchaudio.load_with_torchcodec.*",
        category=UserWarning,
    )