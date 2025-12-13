# architectures/keyword_spotting/__init__.py
from .model import KWSBase
from .schema import KeyWordSpottingConfig, SpectrogramConfig, BackboneConfig, NoiseConfig, DatasetConfig
__all__ = ["KWSBase", "KeyWordSpottingConfig", "SpectrogramConfig", "BackboneConfig", "NoiseConfig", "DatasetConfig"]