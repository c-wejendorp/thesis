from pydantic import BaseModel, computed_field, Field

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

class DatasetConfig(BaseModel):
    key_words: list[str]
    use_unknown: bool
    use_silence: bool
    upsample: bool
    @computed_field
    @property
    def num_classes(self) -> int:
        return len(self.key_words) + int(self.use_unknown) + int(self.use_silence)

class NoiseConfig(BaseModel):
    add_noise: bool
    noise_prob: float
    snr: float | tuple[float, float]

class KeyWordSpottingConfig(BaseModel):
    spectrogram: SpectrogramConfig
    backbone: BackboneConfig
    dataset: DatasetConfig
    noise: NoiseConfig