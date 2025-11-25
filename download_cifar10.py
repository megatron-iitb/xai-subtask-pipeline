#!/usr/bin/env python3
"""
Download CIFAR-10 dataset locally for offline use
Run this once when you have internet connection
"""

import os
from torchvision import datasets

# Download CIFAR-10 to a local directory
data_dir = "/home/medal/anupam.rawat/Experiment_3/data"
os.makedirs(data_dir, exist_ok=True)

print("Downloading CIFAR-10 training set...")
train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
print(f"✓ Training set downloaded: {len(train_dataset)} samples")

print("Downloading CIFAR-10 test set...")
test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)
print(f"✓ Test set downloaded: {len(test_dataset)} samples")

print(f"\n✓ CIFAR-10 dataset saved to: {data_dir}")
print(f"✓ Total size: {len(train_dataset) + len(test_dataset)} samples")
print("\nDataset structure:")
os.system(f"ls -lah {data_dir}")
