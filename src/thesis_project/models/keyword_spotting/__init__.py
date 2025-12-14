# architectures/keyword_spotting/__init__.py
from .base_model import KWSBase
from .base_schema import KeyWordSpottingBaseConfig, SpectrogramConfig, BackboneConfig, NoiseConfig, DatasetConfig
from .dynamic_rank_model import KWSDynamic
__all__ = ["KWSBase","KWSDynamic", "KeyWordSpottingBaseConfig", "SpectrogramConfig", "BackboneConfig", "NoiseConfig", "DatasetConfig"]