# architectures/keyword_spotting/__init__.py
from .base_model import KWSBase
from .base_schema import KeyWordSpottingBaseConfig, SpectrogramConfig, BackboneConfig, NoiseConfig, DatasetConfig
__all__ = ["KWSBase", "KeyWordSpottingBaseConfig", "SpectrogramConfig", "BackboneConfig", "NoiseConfig", "DatasetConfig"]