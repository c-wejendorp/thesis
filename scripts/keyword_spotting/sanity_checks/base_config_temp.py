from thesis_project.datasets.speech_commands.config import ALL_KEYWORDS, CANONICAL, COMMAND_KEYWORDS_20, COMMAND_KEYWORDS_24
from thesis_project.models.keyword_spotting import KWSBase,KeyWordSpottingBaseConfig, SpectrogramConfig, BackboneConfig, NoiseConfig, DatasetConfig

#TODO: Remove this when above is fixed
spec_cfg=SpectrogramConfig(
    sample_rate=16000,
    n_fft=512,
)
backbone_cfg=BackboneConfig(
    n_channels_ext=128,
    n_channels_int=128,
    kernel_size=3,
    dropout=0.1,
    causal=True,
    n_blocks_pr_stack=3,
    dilation_base=2,
    n_stacks=3,
    residual_in_blocks=True,
    residual_in_stacks=False,
    use_custom_pw=True,
)

dataset_cfg=DatasetConfig(
    key_words=CANONICAL,
    use_unknown=True,
    use_silence=True,
    upsample=False,
)
noise_train_cfg=NoiseConfig(
    add_noise=True,
    noise_prob=0.8,
    snr=(-5, 15))

noise_eval_cfg=NoiseConfig(
    add_noise=True,
    noise_prob=0.8,
    snr=float("inf"))

cfg = KeyWordSpottingBaseConfig(
    spectrogram=spec_cfg,
    backbone=backbone_cfg,
    dataset = dataset_cfg,
    noise=noise_train_cfg,
)