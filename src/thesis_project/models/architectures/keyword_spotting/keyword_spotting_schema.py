from pydantic import BaseModel, Field


class SpectrogramConfig(BaseModel):
    sample_rate: int
    n_fft: int


class BackboneConfig(BaseModel):
    n_channels_ext: int
    n_channels_int: int
    kernel_size: int
    dropout: float
    causal: bool
    n_blocks_pr_stack: int
    dilation_base: int
    n_stacks: int
    residual_in_blocks: bool
    residual_in_stacks: bool
    use_custom_pw: bool = Field(
        True, description="Use CPC_Conv1d for pointwise convolutions."
    )


class KeyWordSpottingConfig(BaseModel):
    spectrogram: SpectrogramConfig
    backbone: BackboneConfig
    num_classes: int = 35