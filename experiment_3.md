Node-by-node detailed explanation

1) Data (CIFAR-10)
What: 60,000 small 32×32 RGB images across 10 classes (train/test split provided by torchvision).
Goal: Provide open-source, reproducible data for training and evaluation.
Pipeline action: dataset yields {'image': PIL.Image, 'labels': {'color_class':..., 'animal_vehicle':..., 'final':...}}.
2) Preprocessing (CLIPProcessor)
What: Resizes, center-crops/pad if necessary, and normalizes images to CLIP's expected pixel range.
Input: list of PIL images (batch).
Output: pixel_values tensor shaped (batch_size, 3, H_clip, W_clip).
Why: CLIP is the shared feature extractor — we use its tokenizer/processor for consistent preprocessing.
3) Backbone (CLIP image encoder)
What: Pretrained CLIP image encoder (frozen weights by default) producing a fixed-dimensional embedding per image.
Output: embedding vector shape: (batch_size, emb_dim) (CLIP's projection_dim).
Why frozen: Reduce training time and enforce shared, interpretable features; prevents catastrophic drift.
4) Subtask Heads (small MLPs)
Heads: one head per subtask (here: color_class with 3 outputs, animal_vehicle with 2 outputs).
Input: emb_dim vector.
Output: logits per head (batch_size, num_classes_subtask).
Loss: cross_entropy(logits, subtask_target) used as auxiliary loss to guide embeddings.
Purpose: Break the problem into semantically meaningful subtasks to help interpretability and modularity.
5) Meta Learner (consensus layer)
Input: concatenated probabilities from all subtask heads (batch_size, sum(num_subtask_classes)).
Architecture: small MLP producing final logits for CIFAR-10 classes.
Loss: cross_entropy(final_logits, final_target) — the main supervised objective.
Training signal: Heads are trained jointly with meta learner; total loss = main loss + 0.1 * sum(aux losses).
Why use probs (not logits): the meta learner works on normalized subtask confidences — simpler and more stable.
6) Temperature Scaling (Calibration)
What: A small parametric scalar T dividing final logits to calibrate softmax outputs.
Fitting: Collect logits on a validation set, optimize T via LBFGS to minimize NLL (CrossEntropy).
Result: Better-calibrated final probabilities (lower ECE).
7) Training loop details
Which params trained: subtask heads + meta learner (backbone frozen by default).
Optimizer: Adam for heads + meta; LBFGS for temperature fitting separately.
Batching: CLIPProcessor accepts a batch of images; embeddings computed once per batch then fed to heads.
Objective: final_loss + 0.1 * sum(subtask_losses).
Logging: epoch-wise average loss appended to audit_log.
8) Evaluation metrics
Accuracy: classification accuracy on CIFAR-10 test set.
ECE (Expected Calibration Error): computed from final probabilities; useful for calibration assessment.
Where stored: evaluation metrics written to audit_log and model artifacts saved.
9) Explainability
Meta-level (why meta chose class): SHAP (KernelExplainer)
Explains how subtask probabilities contributed to final prediction.
Operates on the tabular-level (concatenated probs) — interpretable per-subtask contributions.
Image-level (which pixels mattered): Captum IntegratedGradients
Uses a forward_fn returning (batch, num_classes) logits (important fix to avoid Captum assertion).
Computes attribution maps per pixel channel; results saved as artifacts/attribution_map.png.

