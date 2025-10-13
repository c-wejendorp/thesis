# architectures/keyword_spotting/__init__.py
from .keyword_spotting_model import KeyWordSpottingModel
from .keyword_spotting_schema import KeyWordSpottingConfig, SpectrogramConfig, BackboneConfig
__all__ = ["KeyWordSpottingModel", "KeyWordSpottingConfig", "SpectrogramConfig", "BackboneConfig"]
