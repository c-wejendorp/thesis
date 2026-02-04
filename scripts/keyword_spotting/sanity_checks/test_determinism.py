"""
Test that the dataset produces identical samples across multiple runs.
"""
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from pathlib import Path

from thesis_project.datasets import SpeechCommandsGoogle
from thesis_project.utils.paths import get_data_dir

# Configuration
SEED = 42

# Set global seeds at module level
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

NUM_EPOCHS = 3
BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_SAMPLES_TO_CHECK = 5000  # Check first N samples per epoch

def create_dataset_and_loader():
    """Create dataset and dataloader with fixed seeds."""
    data_dir = get_data_dir()
    
    dataset = SpeechCommandsGoogle(
        root=data_dir,
        subset="training",
        download=False,
        seed=SEED,
        add_noise=True,
        noise_prob=0.9,
        snr=(-5, 15),
    )
    
    # DataLoader with seeded generator for deterministic shuffling
    generator = torch.Generator()
    generator.manual_seed(SEED)
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )
    
    return dataset, loader

def collect_samples(loader, num_epochs, num_samples):
    """Collect samples from loader over multiple epochs."""
    all_samples = []
    
    for epoch in range(num_epochs):
        epoch_samples = []
        sample_count = 0
        
        for batch_idx, (waveforms, labels, metadata) in enumerate(loader):
            # Store batch data
            batch_data = {
                'waveforms': waveforms.clone(),
                'labels': labels.clone(),
                'snr': [metadata['snr'][i].item() for i in range(len(metadata['snr']))],
                'noise_type': metadata['noise_type'],
            }
            epoch_samples.append(batch_data)
            
            sample_count += waveforms.shape[0]
            if sample_count >= num_samples:
                break
        
        all_samples.append(epoch_samples)
        print(f"  Collected {sample_count} samples from epoch {epoch}")
    
    return all_samples

def compare_samples(samples1, samples2):
    """Compare two sets of collected samples."""
    assert len(samples1) == len(samples2), "Different number of epochs"
    
    total_mismatches = 0
    
    for epoch_idx, (epoch1, epoch2) in enumerate(zip(samples1, samples2)):
        assert len(epoch1) == len(epoch2), f"Epoch {epoch_idx}: Different number of batches"
        
        for batch_idx, (batch1, batch2) in enumerate(zip(epoch1, epoch2)):
            # Compare waveforms
            if not torch.allclose(batch1['waveforms'], batch2['waveforms'], atol=1e-6):
                total_mismatches += 1
                print(f"❌ Epoch {epoch_idx}, Batch {batch_idx}: Waveforms differ!")
                print(f"   Max diff: {(batch1['waveforms'] - batch2['waveforms']).abs().max().item()}")
            
            # Compare labels
            if not torch.equal(batch1['labels'], batch2['labels']):
                total_mismatches += 1
                print(f"❌ Epoch {epoch_idx}, Batch {batch_idx}: Labels differ!")
            
            # Compare SNR values
            if batch1['snr'] != batch2['snr']:
                total_mismatches += 1
                print(f"❌ Epoch {epoch_idx}, Batch {batch_idx}: SNR values differ!")
                print(f"   SNR1: {batch1['snr'][:5]}")
                print(f"   SNR2: {batch2['snr'][:5]}")
            
            # Compare noise types
            if batch1['noise_type'] != batch2['noise_type']:
                total_mismatches += 1
                print(f"❌ Epoch {epoch_idx}, Batch {batch_idx}: Noise types differ!")
                print(f"   Types1: {batch1['noise_type'][:5]}")
                print(f"   Types2: {batch2['noise_type'][:5]}")
    
    return total_mismatches

def main():
    print("=" * 70)
    print("Testing Dataset Determinism")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Seed: {SEED}")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Samples to check per epoch: {NUM_SAMPLES_TO_CHECK}")
    print("=" * 70)
    
    # First run
    print("\n[Run 1] Creating dataset and collecting samples...")
    dataset1, loader1 = create_dataset_and_loader()
    samples1 = collect_samples(loader1, NUM_EPOCHS, NUM_SAMPLES_TO_CHECK)
    
    # Second run
    print("\n[Run 2] Creating new dataset and collecting samples...")
    dataset2, loader2 = create_dataset_and_loader()
    samples2 = collect_samples(loader2, NUM_EPOCHS, NUM_SAMPLES_TO_CHECK)
    
    # Compare
    print("\n" + "=" * 70)
    print("Comparing samples...")
    print("=" * 70)
    mismatches = compare_samples(samples1, samples2)
    
    print("\n" + "=" * 70)
    if mismatches == 0:
        print("✅ SUCCESS: All samples match across both runs!")
        print(f"   Checked {NUM_EPOCHS} epochs with {NUM_SAMPLES_TO_CHECK} samples each")
    else:
        print(f"❌ FAILURE: Found {mismatches} mismatches")
    print("=" * 70)
    
    # Additional check: verify counter increments
    print("\n[Bonus Check] Counter values:")
    print(f"  Dataset 1 counter: {dataset1._call_counter}")
    print(f"  Dataset 2 counter: {dataset2._call_counter}")
    if dataset1._call_counter == dataset2._call_counter:
        print("  ✅ Counters match!")
    else:
        print("  ❌ Counters differ!")

if __name__ == "__main__":
    main()
