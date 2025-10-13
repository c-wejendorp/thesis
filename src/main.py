from thesis_project.datasets import SpeechCommandsGoogle
from thesis_project.utils.paths import get_data_dir

from thesis_project.models.architectures.keyword_spotting import KeyWordSpottingModel, KeyWordSpottingConfig, SpectrogramConfig, BackboneConfig

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    # --- Load YAML file ---
    config_path = Path("src/thesis_project/models/architectures/keyword_spotting/keyword_spotting_config.yaml")
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # --- Parse and validate with Pydantic ---
    cfg = KeyWordSpottingConfig(
        spectrogram=SpectrogramConfig(**raw_cfg["spectrogram"]),
        backbone=BackboneConfig(**raw_cfg["backbone"]),
        num_classes=raw_cfg.get("num_classes", 35),
    )

    #print(cfg)
    model = KeyWordSpottingModel(cfg)
    data_dir = get_data_dir()
    dataset = SpeechCommandsGoogle(root=str(data_dir), subset="training", download=True)

    # pass a single item through the model
    waveform, label, metadata = dataset[110]
    logits = model(waveform.unsqueeze(0))