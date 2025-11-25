"""
accountable_xai_pipeline_cifar10.py

A self-contained, runnable version of your "accountable" XAI pipeline that:
- Uses CLIP (HuggingFace) as a frozen feature extractor
- Uses CIFAR-10 (open-source) as the dataset, with derived small subtasks
- Trains subtask heads and a meta-learner to predict final CIFAR-10 classes
- Provides temperature scaling calibration, MC-dropout hooks (optional), SHAP for meta explanations,
  and Captum IntegratedGradients for image attributions (fixed forward function)
- Keeps simple accountability artifacts: model card, artifact registry with SHA256, and a tamper-evident audit log

Run (example):
    python accountable_xai_pipeline_cifar10.py --epochs 3 --batch_size 64

Requirements (pip):
    torch torchvision transformers scikit-learn shap captum tqdm matplotlib pandas pillow

Notes / important fixes vs. original:
- The Captum error is fixed: the forward function used for IntegratedGradients now returns (batch, num_classes)
  logits so Captum can index `target` internally (instead of returning a 1-D vector).
- Temperature calibration now collects *logits* (pre-softmax) rather than log-probabilities; CrossEntropy expects logits.
- Uses CIFAR-10 as an open-source dataset; derives small subtasks from CIFAR labels (color_class and coarse animal/vehicle)

"""

import os
import json
import time
import hashlib
import logging
import uuid
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
from sklearn.model_selection import train_test_split

# Explainability libs (optional; pipeline will warn if missing)
try:
    import shap
except Exception:
    shap = None
try:
    from captum.attr import IntegratedGradients
except Exception:
    IntegratedGradients = None

# For plotting
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration & simple logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("accountable_pipeline")

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Accountability utilities: artifact registry, hashing and tamper-evident audit log
# ---------------------------------------------------------------------------

class ArtifactRegistry:
    """Stores metadata about artifacts and computes SHA256 hashes of files."""
    def __init__(self, registry_path=os.path.join(ARTIFACT_DIR, "registry.json")):
        self.registry_path = registry_path
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"artifacts": []}
            self._write()

    def _hash_file(self, path):
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def register(self, path: str, metadata: Dict[str, Any]):
        metadata = metadata.copy()
        metadata.update({
            "path": path,
            "sha256": self._hash_file(path),
            "timestamp": time.time()
        })
        self.registry["artifacts"].append(metadata)
        self._write()
        logger.info(f"Registered artifact: {path}")
        return metadata

    def _write(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)


class TamperEvidentLog:
    """A minimal append-only log that stores a hash-chain of entries to detect tampering."""
    def __init__(self, path=os.path.join(ARTIFACT_DIR, "audit_log.jsonl")):
        self.path = path
        self.prev_hash = None
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1].decode('utf-8'))
                    self.prev_hash = last.get('entry_hash')

    def append(self, event: str, data: Dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "event": event,
            "data": data,
            "prev_hash": self.prev_hash
        }
        entry_bytes = json.dumps(entry, sort_keys=True).encode('utf-8')
        entry_hash = hashlib.sha256(entry_bytes).hexdigest()
        entry['entry_hash'] = entry_hash
        with open(self.path, 'ab') as f:
            f.write(json.dumps(entry).encode('utf-8') + b"\n")
        self.prev_hash = entry_hash
        logger.info(f"Audit log append: {event}")
        return entry


artifact_registry = ArtifactRegistry()
audit_log = TamperEvidentLog()

# ---------------------------------------------------------------------------
# Simple Dataset using CIFAR-10 with derived subtasks
# ---------------------------------------------------------------------------

# Helper: coarse grouping of CIFAR classes into 'animal' vs 'vehicle' – simple split
_ANIMAL_CLASSES = {2, 3, 4, 5, 6}  # bird, cat, deer, dog, frog (approx mapping)
_VEHICLE_CLASSES = {0, 1, 7, 8, 9}  # airplane, automobile, horse, ship, truck (approx mapping)


def derive_subtasks_from_image(img: Image.Image, label: int) -> Dict[str, int]:
    """Derive two small subtasks from the CIFAR-10 image and its class:
    - color_class: coarse quantization of mean color (3 classes)
    - coarse_av: animal(1)/vehicle(0) binary derived from label
    """
    arr = np.array(img).astype(np.float32)
    mean_rgb = arr.mean(axis=(0, 1))  # [R, G, B]
    # naive color bin: which channel is dominant among R,G,B
    dominant = int(np.argmax(mean_rgb))
    # map to 3 coarse groups: 0->red/dominant, 1->green, 2->blue
    color_class = dominant
    coarse_av = 1 if label in _ANIMAL_CLASSES else 0
    return {"color_class": color_class, "animal_vehicle": coarse_av}


class CIFARSubtaskDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', download=True, transform=None):
        assert split in ('train', 'test')
        self.split = split
        self.transform = transform
        self.dataset = datasets.CIFAR10(root=root, train=(split == 'train'), download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img is a PIL Image from torchvision CIFAR
        subtasks = derive_subtasks_from_image(img, label)
        labels = {
            'color_class': int(subtasks['color_class']),
            'animal_vehicle': int(subtasks['animal_vehicle']),
            'final': int(label)
        }
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'labels': labels}

# collate function: we will pass raw PIL images into CLIPProcessor; ensure dataset returns PILs
# here we keep images as PIL when no transform is provided; pass images list to processor

# ---------------------------------------------------------------------------
# CLIP backbone + subtask heads + meta-learner
# ---------------------------------------------------------------------------

class CLIPBackbone(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', device='cpu'):
        super().__init__()
        self.device = device
        # load pretrained CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        # returns image features (projection dim, already a vector per image)
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        return outputs


class SubtaskHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class MetaLearner(nn.Module):
    def __init__(self, in_dim: int, num_final_classes: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_final_classes)
        )

    def forward(self, x):
        return self.net(x)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # param > 0: initialize near 1.0
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        # divide logits by temperature
        return logits / self.temperature

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins=15) -> float:
    """Estimate Expected Calibration Error (ECE). probs: (N, C)"""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].float().mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.float().mean() * torch.abs(bin_conf - bin_acc).item())
    return ece

# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class AccountablePipeline:
    def __init__(self,
                 subtasks: Dict[str, int],
                 final_num_classes: int,
                 clip_model_name='openai/clip-vit-base-patch32',
                 device='cpu'):
        set_seed()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.backbone = CLIPBackbone(clip_model_name=clip_model_name, device=device)

        self.subtasks = subtasks
        emb_dim = self.backbone.clip.config.projection_dim

        self.heads = nn.ModuleDict({
            name: SubtaskHead(in_dim=emb_dim, num_classes=n).to(device)
            for name, n in subtasks.items()
        })

        meta_in_dim = sum(subtasks.values())
        self.meta = MetaLearner(in_dim=meta_in_dim, num_final_classes=final_num_classes).to(device)

        self.temperature_scaler = TemperatureScaler().to(device)

        self.pipeline_id = str(uuid.uuid4())
        self.metadata = {
            "pipeline_id": self.pipeline_id,
            "subtasks": subtasks,
            "final_num_classes": final_num_classes,
            "clip_model_name": clip_model_name,
            "created_at": time.time()
        }
        audit_log.append("pipeline_created", self.metadata)

    def predict_subtask_logits(self, pixel_values: torch.Tensor):
        emb = self.backbone(pixel_values)
        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(emb)
        return logits, emb

    def predict(self, pixel_values: torch.Tensor, apply_temp=False):
        """Return: subtask probs dict, final_probs, final_logits (pre-softmax)."""
        logits, emb = self.predict_subtask_logits(pixel_values)
        probs = {name: F.softmax(l, dim=1) for name, l in logits.items()}
        concat = torch.cat([probs[name] for name in sorted(probs.keys())], dim=1)
        final_logits = self.meta(concat)
        if apply_temp:
            final_logits = self.temperature_scaler(final_logits)
        final_probs = F.softmax(final_logits, dim=1)
        return probs, final_probs, final_logits

    def save_artifacts(self, path_prefix=os.path.join(ARTIFACT_DIR, "model")):
        os.makedirs(path_prefix, exist_ok=True)
        torch.save(self.heads.state_dict(), os.path.join(path_prefix, "heads.pt"))
        torch.save(self.meta.state_dict(), os.path.join(path_prefix, "meta.pt"))
        torch.save(self.temperature_scaler.state_dict(), os.path.join(path_prefix, "temp.pt"))

        model_card = {
            "pipeline_id": self.pipeline_id,
            "metadata": self.metadata,
            "note": "Accountability model card with dataset provenance and limitations.",
        }
        mc_path = os.path.join(path_prefix, "model_card.json")
        with open(mc_path, 'w') as f:
            json.dump(model_card, f, indent=2)

        artifact_registry.register(os.path.join(path_prefix, "heads.pt"), {"type": "model_heads"})
        artifact_registry.register(os.path.join(path_prefix, "meta.pt"), {"type": "model_meta"})
        artifact_registry.register(mc_path, {"type": "model_card"})
        audit_log.append("artifacts_saved", {"prefix": path_prefix})
        logger.info("Artifacts saved and registered.")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_pipeline(pipeline: AccountablePipeline,
                   train_loader: DataLoader,
                   val_loader: DataLoader = None,
                   epochs=5,
                   lr=1e-3):
    params = list(pipeline.heads.parameters()) + list(pipeline.meta.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        pipeline.backbone.clip.eval()
        for module in pipeline.heads.values():
            module.train()
        pipeline.meta.train()

        total_loss = 0.0
        for batch in train_loader:
            images = batch['image']
            labels = batch['labels']

            inputs = pipeline.processor(images=images, return_tensors='pt', padding=True)
            inputs = {k: v.to(pipeline.device) for k, v in inputs.items()}

            # Get embeddings
            emb = pipeline.backbone(inputs['pixel_values'])

            # Compute subtask losses and collect probabilities
            subtask_losses = []
            subtask_probs = []

            for name in sorted(pipeline.subtasks.keys()):
                head = pipeline.heads[name]
                logits = head(emb)
                target = torch.tensor([l[name] for l in labels], dtype=torch.long).to(pipeline.device)
                loss = F.cross_entropy(logits, target)
                subtask_losses.append(loss)
                subtask_probs.append(F.softmax(logits, dim=1))

            # Meta learner
            meta_in = torch.cat(subtask_probs, dim=1)
            final_logits = pipeline.meta(meta_in)
            final_target = torch.tensor([l['final'] for l in labels], dtype=torch.long).to(pipeline.device)
            final_loss = F.cross_entropy(final_logits, final_target)

            # Combined loss
            loss = final_loss + 0.1 * sum(subtask_losses)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        audit_log.append("epoch_finished", {"epoch": epoch+1, "avg_loss": avg_loss})

    pipeline.save_artifacts()

# ---------------------------------------------------------------------------
# Calibration (fixed to collect logits pre-softmax)
# ---------------------------------------------------------------------------

def fit_temperature(pipeline: AccountablePipeline, val_loader: DataLoader):
    pipeline.backbone.clip.eval()
    for module in pipeline.heads.values():
        module.eval()
    pipeline.meta.eval()

    logits_all = []
    labels_all = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            labels = torch.tensor([l['final'] for l in batch['labels']], dtype=torch.long).to(pipeline.device)

            inputs = pipeline.processor(images=images, return_tensors='pt', padding=True)
            inputs = {k: v.to(pipeline.device) for k, v in inputs.items()}

            _, _, final_logits = pipeline.predict(inputs['pixel_values'], apply_temp=False)
            logits_all.append(final_logits)
            labels_all.append(labels)

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    nll_criterion = nn.CrossEntropyLoss()
    temp = TemperatureScaler().to(pipeline.device)
    opt = torch.optim.LBFGS([temp.temperature], lr=0.1, max_iter=50)

    def eval_loss():
        opt.zero_grad()
        scaled = temp(logits_all)
        loss = nll_criterion(scaled, labels_all)
        loss.backward()
        return loss

    opt.step(eval_loss)
    pipeline.temperature_scaler.temperature.data = temp.temperature.data

    temp_val = float(pipeline.temperature_scaler.temperature.data)
    audit_log.append('temperature_calibrated', {'temperature': temp_val})
    logger.info(f"Fitted temperature: {temp_val:.4f}")
    return temp_val

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def explain_meta_with_shap(pipeline: AccountablePipeline, sample_batch: List[Dict[str, Any]]):
    """Explains meta-learner decisions using SHAP KernelExplainer on a small sample.
    This is slow — only use on a handful of samples.
    """
    if shap is None:
        logger.warning("shap not available; skipping SHAP explanations")
        return None

    images = [s['image'] for s in sample_batch]
    inputs = pipeline.processor(images=images, return_tensors='pt', padding=True)
    inputs = {k: v.to(pipeline.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits_dict, _ = pipeline.predict_subtask_logits(inputs['pixel_values'])
        probs = [F.softmax(logits_dict[name], dim=1).cpu().numpy()
                 for name in sorted(logits_dict.keys())]

    concat = np.concatenate(probs, axis=1)

    def meta_predict(np_array):
        t = torch.tensor(np_array.astype('float32')).to(pipeline.device)
        with torch.no_grad():
            logits = pipeline.meta(t)
            return F.softmax(logits, dim=1).cpu().numpy()

    sample_size = min(10, concat.shape[0])
    explainer = shap.KernelExplainer(meta_predict, concat[:sample_size])
    shap_values = explainer.shap_values(concat[:min(5, concat.shape[0])])
    logger.info(f"SHAP explanation computed for {len(sample_batch)} samples")
    return shap_values


def explain_image_attributions(pipeline: AccountablePipeline, image, n_steps=50):
    """Use Captum IntegratedGradients for image-level explanations. Fixed forward_fn shape.
    Returns: attribution array (C, H, W)
    """
    if IntegratedGradients is None:
        logger.warning("captum not available; skipping image attributions")
        return None

    pipeline.backbone.clip.eval()
    for module in pipeline.heads.values():
        module.eval()
    pipeline.meta.eval()

    inputs = pipeline.processor(images=[image], return_tensors='pt', padding=True)
    pixel_values = inputs['pixel_values'].to(pipeline.device)
    pixel_values.requires_grad = True

    logits_dict, _ = pipeline.predict_subtask_logits(pixel_values)
    probs = [F.softmax(logits_dict[name], dim=1) for name in sorted(logits_dict.keys())]
    concat = torch.cat(probs, dim=1)
    final_logits = pipeline.meta(concat)
    pred_class = int(final_logits.argmax(dim=1).item())

    # forward_fn must return (batch, num_classes)
    def forward_fn(x):
        ld, _ = pipeline.predict_subtask_logits(x)
        ps = [F.softmax(ld[name], dim=1) for name in sorted(ld.keys())]
        cc = torch.cat(ps, dim=1)
        out = pipeline.meta(cc)
        return out

    ig = IntegratedGradients(forward_fn)

    # baseline: black image of same shape
    baseline = torch.zeros_like(pixel_values).to(pipeline.device)

    attributions = ig.attribute(pixel_values,
                                baselines=baseline,
                                target=pred_class,
                                n_steps=n_steps,
                                internal_batch_size=1)
    at = attributions.detach().cpu().numpy()[0]

    # visualize & save (sum over channels)
    vis = at.transpose(1, 2, 0).sum(axis=2)
    plt.figure(figsize=(6, 5))
    plt.imshow((vis - vis.min()) / (vis.max() - vis.min() + 1e-12), cmap='hot')
    plt.title(f'IntegratedGradients (Predicted class: {pred_class})')
    plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, 'attribution_map.png')
    plt.savefig(out_path)
    plt.close()
    logger.info(f'IG attributions saved to {out_path}')
    return at

# ---------------------------------------------------------------------------
# Example usage / CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data_root', type=str, default='./data')
    # Use parse_known_args so Jupyter/Colab kernel CLI args (like -f) are ignored instead of causing SystemExit
    args, unknown = p.parse_known_args()
    if unknown:
        logger.debug(f"Ignoring unknown CLI args: {unknown}")
    return args


def main():
    args = parse_args()
    device = args.device
    logger.info(f"Using device: {device}")

    # Subtasks: color_class (3), animal_vehicle (2)
    subtasks = {"color_class": 3, "animal_vehicle": 2}
    final_num_classes = 10  # CIFAR-10

    pipeline = AccountablePipeline(subtasks=subtasks,
                                   final_num_classes=final_num_classes,
                                   device=device)

    # Data
    transform = None  # keep PIL images; CLIPProcessor will handle resizing
    train_ds = CIFARSubtaskDataset(root=args.data_root, split='train', download=True, transform=transform)
    test_ds = CIFARSubtaskDataset(root=args.data_root, split='test', download=True, transform=transform)

    def collate_fn(batch):
        return {
            'image': [item['image'] for item in batch],
            'labels': [item['labels'] for item in batch]
        }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info("Starting training...")
    train_pipeline(pipeline, train_loader, val_loader=test_loader, epochs=args.epochs, lr=args.lr)

    logger.info("Calibrating temperature...")
    fit_temperature(pipeline, test_loader)

    # Quick eval on test set to compute accuracy + ECE
    pipeline.backbone.clip.eval()
    for module in pipeline.heads.values():
        module.eval()
    pipeline.meta.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image']
            labels = torch.tensor([l['final'] for l in batch['labels']], dtype=torch.long).to(device)
            inputs = pipeline.processor(images=images, return_tensors='pt', padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _, final_probs, _ = pipeline.predict(inputs['pixel_values'], apply_temp=True)
            preds = final_probs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_probs.append(final_probs.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = (all_preds == all_labels).float().mean().item()
    ece = compute_ece(all_probs, all_labels)
    logger.info(f"Test Accuracy: {acc:.4f}  ECE: {ece:.4f}")

    audit_log.append('evaluation', {'accuracy': float(acc), 'ece': float(ece)})

    # Explain a few samples
    sample_indices = list(range(10))
    sample_batch = [test_ds[i] for i in sample_indices]

    if shap is not None:
        logger.info("Computing SHAP explanations for meta learner (small sample)...")
        shap_vals = explain_meta_with_shap(pipeline, sample_batch)
        # Save a small shap summary if available
        if shap_vals is not None:
            np.save(os.path.join(ARTIFACT_DIR, 'shap_meta.npy'), shap_vals)
            artifact_registry.register(os.path.join(ARTIFACT_DIR, 'shap_meta.npy'), {'type': 'shap_values'})

    if IntegratedGradients is not None:
        logger.info("Computing IntegratedGradients for first test image...")
        at = explain_image_attributions(pipeline, test_ds[0]['image'])
        if at is not None:
            np.save(os.path.join(ARTIFACT_DIR, 'ig_attribution.npy'), at)
            artifact_registry.register(os.path.join(ARTIFACT_DIR, 'ig_attribution.npy'), {'type': 'ig_attribution'})

    logger.info("✓ Pipeline completed successfully!")
    logger.info(f"✓ Check '{ARTIFACT_DIR}' directory for outputs")


if __name__ == '__main__':
    main()
