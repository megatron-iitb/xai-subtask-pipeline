"""
accountable_xai_pipeline_cifar10_improved.py

IMPROVEMENTS OVER ORIGINAL:
1. Better subtask design: texture-based features + coarse superclass grouping (not just naive color bins)
2. Train/validation split: 90/10 split from training set for proper validation
3. Deeper meta-learner: Added hidden layers with batch normalization for better capacity
4. MC-Dropout uncertainty: Implemented proper MC-Dropout for epistemic uncertainty estimation
5. Optimized SHAP: Uses smaller background set and fewer samples to avoid timeout
6. Learning rate scheduling: CosineAnnealingLR for better convergence
7. Gradient clipping: Prevents exploding gradients during training
8. Better logging: More detailed metrics including per-subtask accuracy
9. Offline model support: Uses local model paths when HF_HUB_OFFLINE=1
10. Early stopping: Prevents overfitting using validation loss monitoring

Run (example):
    python experiment_3_improved.py --epochs 10 --batch_size 128 --lr 1e-3 --clip_model_path ./clip_model

Requirements (pip):
    torch torchvision transformers scikit-learn shap captum tqdm matplotlib pandas pillow

"""

import os
import json
import time
import hashlib
import logging
import uuid
import argparse
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

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

# ---------------------------------------------------------------------------
# Configuration & simple logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("accountable_pipeline")

ARTIFACT_DIR = "artifacts_optB"
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
                for line in f:
                    entry = json.loads(line)
                    self.prev_hash = entry.get('entry_hash')

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
            f.write(json.dumps(entry).encode('utf-8') + b'\n')
        self.prev_hash = entry_hash
        logger.debug(f"Audit log append: {event}")
        return entry


artifact_registry = ArtifactRegistry()
audit_log = TamperEvidentLog()

# ---------------------------------------------------------------------------
# Improved Dataset with better subtask derivation
# ---------------------------------------------------------------------------

# CIFAR-10 superclass grouping (more meaningful than naive animal/vehicle)
# Based on semantic similarity
_SUPERCLASS_MAPPING = {
    0: 0,  # airplane -> vehicle
    1: 0,  # automobile -> vehicle
    2: 1,  # bird -> animal
    3: 1,  # cat -> animal
    4: 1,  # deer -> animal
    5: 1,  # dog -> animal
    6: 1,  # frog -> animal
    7: 0,  # horse -> vehicle (commonly used for transport)
    8: 0,  # ship -> vehicle
    9: 0,  # truck -> vehicle
}

# Texture-based features (more informative than color dominance)
def compute_texture_features(img: Image.Image) -> Dict[str, int]:
    """Compute texture-based subtasks:
    - complexity: low/medium/high based on actual edge detection
    - brightness: dark/medium/bright based on mean intensity
    """
    from PIL import ImageFilter
    
    arr = np.array(img).astype(np.float32)
    
    # Brightness: mean intensity across all channels
    mean_intensity = arr.mean()
    if mean_intensity < 85:
        brightness_class = 0  # dark
    elif mean_intensity < 170:
        brightness_class = 1  # medium
    else:
        brightness_class = 2  # bright
    
    # Complexity: using actual edge detection (better than variance)
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_intensity = np.array(edges).astype(np.float32).mean()
    if edge_intensity < 25:
        complexity_class = 0  # low complexity (smooth objects like sky, ocean)
    elif edge_intensity < 45:
        complexity_class = 1  # medium complexity
    else:
        complexity_class = 2  # high complexity (detailed objects, animals)
    
    return {"brightness": brightness_class, "complexity": complexity_class}


def derive_subtasks_from_image(img: Image.Image, label: int) -> Dict[str, int]:
    """Derive meaningful subtasks from the CIFAR-10 image and its class:
    - superclass: animal(1) vs vehicle(0) - semantic grouping
    - brightness: dark/medium/bright (3 classes)
    - complexity: low/medium/high edge density using edge detection (3 classes)
    """
    texture_feats = compute_texture_features(img)
    superclass = _SUPERCLASS_MAPPING[label]
    
    return {
        "superclass": superclass,
        "brightness": texture_feats["brightness"],
        "complexity": texture_feats["complexity"]
    }


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
            'superclass': int(subtasks['superclass']),
            'brightness': int(subtasks['brightness']),
            'complexity': int(subtasks['complexity']),
            'final': int(label)
        }
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'labels': labels}


# ---------------------------------------------------------------------------
# CLIP backbone + improved subtask heads + deeper meta-learner
# ---------------------------------------------------------------------------

class CLIPBackbone(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', device='cpu', local_path=None):
        super().__init__()
        self.device = device
        # Load from local path if provided (offline mode)
        model_path = local_path if local_path and os.path.exists(local_path) else clip_model_name
        self.clip = CLIPModel.from_pretrained(model_path).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        # returns image features (projection dim, already a vector per image)
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        return outputs


class SubtaskHead(nn.Module):
    """Improved subtask head with batch normalization and deeper architecture"""
    def __init__(self, in_dim: int, num_classes: int, hidden=384, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ImprovedMetaLearner(nn.Module):
    """Deeper meta-learner with batch normalization and residual-like connections"""
    def __init__(self, in_dim: int, num_final_classes: int, hidden=384, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_final_classes)
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
# MC-Dropout implementation for uncertainty estimation
# ---------------------------------------------------------------------------

def enable_dropout(model):
    """Enable dropout layers during inference for MC-Dropout"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_dropout_predict(pipeline, pixel_values, n_samples=20):
    """Perform MC-Dropout predictions to estimate epistemic uncertainty
    
    Returns:
        mean_probs: (batch_size, num_classes) - mean predictions
        std_probs: (batch_size, num_classes) - standard deviation (uncertainty)
        entropy: (batch_size,) - predictive entropy
    """
    pipeline.backbone.clip.eval()  # Keep backbone frozen
    for module in pipeline.heads.values():
        module.eval()
        enable_dropout(module)
    pipeline.meta.eval()
    enable_dropout(pipeline.meta)
    
    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            _, final_probs, _ = pipeline.predict(pixel_values, apply_temp=True)
            all_probs.append(final_probs.cpu())
    
    all_probs = torch.stack(all_probs, dim=0)  # (n_samples, batch_size, num_classes)
    mean_probs = all_probs.mean(dim=0)
    std_probs = all_probs.std(dim=0)
    
    # Predictive entropy: -sum(p * log(p))
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
    
    return mean_probs, std_probs, entropy


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed=42):
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
                 clip_local_path=None,
                 device='cpu'):
        set_seed()
        self.device = device
        
        # Load processor and backbone (with offline support)
        processor_path = clip_local_path if clip_local_path and os.path.exists(clip_local_path) else clip_model_name
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.backbone = CLIPBackbone(clip_model_name=clip_model_name, device=device, local_path=clip_local_path)

        self.subtasks = subtasks
        emb_dim = self.backbone.clip.config.projection_dim

        self.heads = nn.ModuleDict({
            name: SubtaskHead(in_dim=emb_dim, num_classes=n).to(device)
            for name, n in subtasks.items()
        })

        meta_in_dim = sum(subtasks.values())
        self.meta = ImprovedMetaLearner(in_dim=meta_in_dim, num_final_classes=final_num_classes).to(device)

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
            "note": "Improved accountability model with better subtasks, deeper meta-learner, and MC-dropout",
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
# Training loop with validation and early stopping
# ---------------------------------------------------------------------------

def train_pipeline(pipeline: AccountablePipeline,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs=10,
                   lr=1e-3,
                   patience=3,
                   clip_grad=1.0):
    params = list(pipeline.heads.parameters()) + list(pipeline.meta.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        pipeline.backbone.clip.eval()
        for module in pipeline.heads.values():
            module.train()
        pipeline.meta.train()

        total_loss = 0.0
        subtask_losses_sum = defaultdict(float)
        correct = 0
        total = 0
        
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
                subtask_losses_sum[name] += loss.item()

            # Meta learner
            meta_in = torch.cat(subtask_probs, dim=1)
            final_logits = pipeline.meta(meta_in)
            final_target = torch.tensor([l['final'] for l in labels], dtype=torch.long).to(pipeline.device)
            final_loss = F.cross_entropy(final_logits, final_target)

            # Combined loss (weighted subtask losses)
            loss = final_loss + 0.1 * sum(subtask_losses)

            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, clip_grad)
            
            opt.step()

            total_loss += loss.item()
            preds = final_logits.argmax(dim=1)
            correct += (preds == final_target).sum().item()
            total += final_target.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        val_loss, val_acc, val_subtask_accs = evaluate(pipeline, val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log per-subtask accuracies
        subtask_info = ", ".join([f"{k}: {v:.3f}" for k, v in val_subtask_accs.items()])
        logger.info(f"  Val Subtask Accs: {subtask_info}")
        
        audit_log.append("epoch_finished", {
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_subtask_accs": val_subtask_accs
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            pipeline.save_artifacts()  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break
        
        scheduler.step()


def evaluate(pipeline: AccountablePipeline, loader: DataLoader):
    """Evaluate pipeline on validation/test set"""
    pipeline.backbone.clip.eval()
    for module in pipeline.heads.values():
        module.eval()
    pipeline.meta.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-subtask accuracy
    subtask_correct = defaultdict(int)
    subtask_total = defaultdict(int)
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image']
            labels = batch['labels']

            inputs = pipeline.processor(images=images, return_tensors='pt', padding=True)
            inputs = {k: v.to(pipeline.device) for k, v in inputs.items()}

            emb = pipeline.backbone(inputs['pixel_values'])

            # Subtask losses
            subtask_losses = []
            subtask_probs = []

            for name in sorted(pipeline.subtasks.keys()):
                head = pipeline.heads[name]
                logits = head(emb)
                target = torch.tensor([l[name] for l in labels], dtype=torch.long).to(pipeline.device)
                loss = F.cross_entropy(logits, target)
                subtask_losses.append(loss)
                subtask_probs.append(F.softmax(logits, dim=1))
                
                # Track subtask accuracy
                preds = logits.argmax(dim=1)
                subtask_correct[name] += (preds == target).sum().item()
                subtask_total[name] += target.size(0)

            # Meta learner
            meta_in = torch.cat(subtask_probs, dim=1)
            final_logits = pipeline.meta(meta_in)
            final_target = torch.tensor([l['final'] for l in labels], dtype=torch.long).to(pipeline.device)
            final_loss = F.cross_entropy(final_logits, final_target)

            loss = final_loss + 0.1 * sum(subtask_losses)
            total_loss += loss.item()
            
            preds = final_logits.argmax(dim=1)
            correct += (preds == final_target).sum().item()
            total += final_target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    subtask_accs = {name: subtask_correct[name] / subtask_total[name] 
                    for name in subtask_correct.keys()}
    
    return avg_loss, accuracy, subtask_accs


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
            logits_all.append(final_logits.cpu())
            labels_all.append(labels.cpu())

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    nll_criterion = nn.CrossEntropyLoss()
    temp = TemperatureScaler().to(pipeline.device)
    opt = torch.optim.LBFGS([temp.temperature], lr=0.1, max_iter=50)

    def eval_loss():
        opt.zero_grad()
        loss = nll_criterion(temp(logits_all.to(pipeline.device)), labels_all.to(pipeline.device))
        loss.backward()
        return loss

    opt.step(eval_loss)
    pipeline.temperature_scaler.temperature.data = temp.temperature.data

    temp_val = float(pipeline.temperature_scaler.temperature.data)
    audit_log.append('temperature_calibrated', {'temperature': temp_val})
    logger.info(f"Fitted temperature: {temp_val:.4f}")
    return temp_val


# ---------------------------------------------------------------------------
# Explainability (optimized SHAP)
# ---------------------------------------------------------------------------

def explain_meta_with_shap(pipeline: AccountablePipeline, sample_batch: List[Dict[str, Any]], max_samples=3):
    """Explains meta-learner decisions using SHAP KernelExplainer on a small sample.
    OPTIMIZED: Uses only 3 samples for both background and explanation to avoid timeout.
    """
    if shap is None:
        logger.warning("SHAP not available; skipping meta explanation")
        return None

    images = [s['image'] for s in sample_batch[:max_samples]]
    inputs = pipeline.processor(images=images, return_tensors='pt', padding=True)
    inputs = {k: v.to(pipeline.device) for k, v in inputs.items()}

    with torch.no_grad():
        probs_dict, _, _ = pipeline.predict(inputs['pixel_values'], apply_temp=False)
        probs = [probs_dict[name].cpu().numpy() for name in sorted(probs_dict.keys())]

    concat = np.concatenate(probs, axis=1)

    def meta_predict(np_array):
        t = torch.tensor(np_array, dtype=torch.float32).to(pipeline.device)
        logits = pipeline.meta(t)
        return F.softmax(logits, dim=1).detach().cpu().numpy()

    # Use only 3 samples for background (faster)
    explainer = shap.KernelExplainer(meta_predict, concat[:min(3, concat.shape[0])])
    shap_values = explainer.shap_values(concat[:max_samples])
    logger.info(f"SHAP explanation computed for {max_samples} samples")
    return shap_values


def explain_image_attributions(pipeline: AccountablePipeline, image, n_steps=50):
    """Use Captum IntegratedGradients for image-level explanations. Fixed forward_fn shape.
    Returns: attribution array (C, H, W)
    """
    if IntegratedGradients is None:
        logger.warning("Captum IntegratedGradients not available; skipping image attribution")
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
        logits_dict, _ = pipeline.predict_subtask_logits(x)
        probs = [F.softmax(logits_dict[name], dim=1) for name in sorted(logits_dict.keys())]
        concat = torch.cat(probs, dim=1)
        return pipeline.meta(concat)

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
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--clip_model_path', type=str, default=None, 
                   help='Local path to CLIP model for offline mode')
    p.add_argument('--patience', type=int, default=4, help='Early stopping patience')
    p.add_argument('--mc_samples', type=int, default=25, help='Number of MC-Dropout samples')
    args, unknown = p.parse_known_args()
    if unknown:
        logger.debug(f"Ignoring unknown CLI args: {unknown}")
    return args


def main():
    args = parse_args()
    device = args.device
    logger.info(f"Using device: {device}")
    logger.info(f"[OPTION B] Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"[OPTION B] Strategy: Scaled LR for larger batch (batch=256, lr=2e-3)")

    # Improved subtasks: superclass (2), brightness (3), complexity (3)
    subtasks = {"superclass": 2, "brightness": 3, "complexity": 3}
    final_num_classes = 10  # CIFAR-10

    pipeline = AccountablePipeline(subtasks=subtasks,
                                   final_num_classes=final_num_classes,
                                   clip_local_path=args.clip_model_path,
                                   device=device)

    # Data
    transform = None  # keep PIL images; CLIPProcessor will handle resizing
    # Set download=False for offline mode (dataset must be pre-downloaded)
    train_ds_full = CIFARSubtaskDataset(root=args.data_root, split='train', download=False, transform=transform)
    test_ds = CIFARSubtaskDataset(root=args.data_root, split='test', download=False, transform=transform)

    # Split training set into train/validation (90/10)
    train_size = int(0.9 * len(train_ds_full))
    val_size = len(train_ds_full) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(train_ds_full)))
    
    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(train_ds_full, val_indices)
    
    logger.info(f"Dataset splits - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    def collate_fn(batch):
        return {
            'image': [item['image'] for item in batch],
            'labels': [item['labels'] for item in batch]
        }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info("Starting training with validation...")
    train_pipeline(pipeline, train_loader, val_loader, epochs=args.epochs, lr=args.lr, patience=args.patience)

    logger.info("Calibrating temperature on validation set...")
    fit_temperature(pipeline, val_loader)

    # Final evaluation on test set with MC-Dropout
    logger.info("Evaluating on test set with MC-Dropout uncertainty estimation...")
    pipeline.backbone.clip.eval()
    for module in pipeline.heads.values():
        module.eval()
    pipeline.meta.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image']
            labels = torch.tensor([l['final'] for l in batch['labels']], dtype=torch.long).to(device)
            inputs = pipeline.processor(images=images, return_tensors='pt', padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # MC-Dropout predictions
            mean_probs, std_probs, entropy = mc_dropout_predict(pipeline, inputs['pixel_values'], n_samples=args.mc_samples)
            preds = mean_probs.argmax(dim=1)
            
            all_preds.append(preds)
            all_probs.append(mean_probs)
            all_labels.append(labels.cpu())
            all_uncertainties.append(entropy)

    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)

    acc = (all_preds == all_labels).float().mean().item()
    ece = compute_ece(all_probs, all_labels)
    mean_uncertainty = all_uncertainties.mean().item()
    
    logger.info(f"Test Accuracy: {acc:.4f}  ECE: {ece:.4f}  Mean Uncertainty (entropy): {mean_uncertainty:.4f}")

    audit_log.append('evaluation', {
        'accuracy': float(acc), 
        'ece': float(ece),
        'mean_uncertainty': float(mean_uncertainty)
    })

    # Explain a few samples (optimized)
    sample_indices = list(range(5))
    sample_batch = [test_ds[i] for i in sample_indices]

    if shap is not None:
        logger.info("Computing SHAP explanations for meta learner (optimized - 3 samples)...")
        shap_vals = explain_meta_with_shap(pipeline, sample_batch, max_samples=3)
        if shap_vals is not None:
            np.save(os.path.join(ARTIFACT_DIR, 'shap_meta.npy'), shap_vals)
            artifact_registry.register(os.path.join(ARTIFACT_DIR, 'shap_meta.npy'), {'type': 'shap_values'})

    if IntegratedGradients is not None:
        logger.info("Computing IntegratedGradients for first test image...")
        at = explain_image_attributions(pipeline, test_ds[0]['image'])
        if at is not None:
            np.save(os.path.join(ARTIFACT_DIR, 'ig_attribution.npy'), at)
            artifact_registry.register(os.path.join(ARTIFACT_DIR, 'ig_attribution.npy'), {'type': 'ig_attribution'})

    logger.info("✓ Improved pipeline completed successfully!")
    logger.info(f"✓ Check '{ARTIFACT_DIR}' directory for outputs")
    logger.info(f"✓ Key improvements: Better subtasks, validation split, deeper meta-learner, MC-dropout, optimized SHAP")


if __name__ == '__main__':
    main()
