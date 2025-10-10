from torch.utils.data import DataLoader

from thesis_project.datasets.speech_commands_google import SpeechCommandsGoogle
from thesis_project.utils.paths import get_data_dir

dataset = SpeechCommandsGoogle(root=str(get_data_dir()), subset="training", download=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# inspect a single item
wave, label, meta = dataset[1]
#print(wave.shape)  # torch.Size([1, 16000]) # (C, T), only one channel for this dataset
#print(label, dataset.idx_to_label[label])  # 0 'backward'
#print(meta)  # {'sample_rate': 16000, 'label': 'backward', 'speaker_id': '0165e0e8', 'utterance_number': 0}

for waves, labels, metabatch in loader:
    print(waves.shape)  # torch.Size([32, 1, 16000]) # (B, C, T), only one channel for this dataset
    print([dataset.idx_to_label[idx.item()] for idx in labels])  # ['yes', 'no', 'up', ...]
    #print(metabatch)  # {'sample_rate': [16000, 16000, ...], 'label': ['yes', 'no', ...], ...} 
    break