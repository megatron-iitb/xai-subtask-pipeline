#!/usr/bin/env python3
"""
Download CLIP model locally for offline use
Run this once when you have internet connection
"""

import os
from transformers import CLIPModel, CLIPProcessor

# Download CLIP to a local directory
model_dir = "/home/medal/anupam.rawat/clip_model"
os.makedirs(model_dir, exist_ok=True)

print("Downloading CLIP ViT-B/32 model...")
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

print("Saving model locally...")
model.save_pretrained(model_dir)
processor.save_pretrained(model_dir)

print(f"\n✓ CLIP model saved to: {model_dir}")
print("\nModel files:")
os.system(f"ls -lah {model_dir}")
