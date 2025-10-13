from .module_config import SUPPRESS_WARNINGS
from .utils.suppress_warnings import suppress_audio_warnings

if SUPPRESS_WARNINGS:
    suppress_audio_warnings()
    #TODO add to logging
    #print("[your_module] torchaudio warnings suppressed ") add to logg
#else:
    #print("[your_module] warning suppression disabled ⚠️")