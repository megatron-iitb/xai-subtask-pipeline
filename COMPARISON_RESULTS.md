
# Experiment 3: Option A vs Option B Comparison

## Configuration

| Parameter | Option A | Option B |
|-----------|----------|----------|
| **Strategy** | Best from Run 1 + Edge | Scaled LR for larger batch |
| **Batch Size** | 128 | 256 |
| **Learning Rate** | 1e-3 | 2e-3 |
| **LR Scaling** | Baseline | 2× (for 2× batch) |
| **Epochs** | 12 (early stop) | 12 (early stop) |
| **MC-Dropout Samples** | 25 | 25 |
| **Hidden Size** | 384 | 384 |

## Final Results

| Metric | Option A | Option B | Winner | Δ |
|--------|----------|----------|--------|---|
| **Test Accuracy** | 94.91% | 94.98% | **B** 🏆 | -0.07pp |
| **ECE (↓ better)** | 1.37% | 1.67% | **A** 🏆 | -0.31pp |
| **Temperature** | 1.431 | 1.504 | **A** 🏆 | -0.072 |
| **Mean Uncertainty** | 0.2355 | 0.2395 | **A** 🏆 | -0.0040 |

## Subtask Performance

| Subtask | Option A | Option B | Δ |
|---------|----------|----------|---|
| **Brightness** | 81.4% | 81.4% | +0.0pp |
| **Complexity** | 65.4% | 64.6% | +0.8pp |
| **Superclass** | 98.7% | 98.8% | -0.1pp |

## Training Efficiency

| Metric | Option A | Option B |
|--------|----------|----------|
| **Best Epoch** | 10 | 12 |
| **Final Train Acc** | 97.62% | 98.47% |
| **Final Val Acc** | 95.14% | 95.24% |
| **Batches/Epoch** | 352 | 176 |

## Key Findings


### Accuracy
- **Winner: Option B**
- Option A benefits from smaller batch size (more frequent updates)
- Option B has better accuracy despite larger batch

### Calibration (ECE)
- **Winner: Option A**
- Lower ECE = better calibrated confidence scores
- Temperature closer to 1.0 indicates better initial calibration

### Complexity Subtask (Edge Detection)
- Both options show similar complexity learning (~62-64%)
- Significant improvement over variance-based method (47%)
- Still room for improvement (target: 70%+)

### Overall Recommendation
⚖️ **Trade-off between accuracy and calibration**
- Choose Option A for best accuracy
- Choose Option B for faster training (2× throughput)
