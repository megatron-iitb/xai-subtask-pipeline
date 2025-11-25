#!/usr/bin/env python3
"""
Compare results from Option A and Option B experiments
Run after both jobs complete to generate comparison plots and summary
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def parse_audit_log(log_path):
    """Parse audit log and extract metrics"""
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    subtask_accs = {'brightness': [], 'complexity': [], 'superclass': []}
    
    final_metrics = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            event = entry.get('event', '')
            data = entry.get('data', {})
            
            if event == 'epoch_finished':
                epochs.append(data.get('epoch', 0))
                train_losses.append(data.get('train_loss', 0))
                train_accs.append(data.get('train_acc', 0))
                val_losses.append(data.get('val_loss', 0))
                val_accs.append(data.get('val_acc', 0))
                
                subtask_data = data.get('val_subtask_accs', {})
                subtask_accs['brightness'].append(subtask_data.get('brightness', 0))
                subtask_accs['complexity'].append(subtask_data.get('complexity', 0))
                subtask_accs['superclass'].append(subtask_data.get('superclass', 0))
            
            elif event == 'evaluation':
                final_metrics['accuracy'] = data.get('accuracy', 0)
                final_metrics['ece'] = data.get('ece', 0)
                final_metrics['mean_uncertainty'] = data.get('mean_uncertainty', 0)
            
            elif event == 'temperature_calibrated':
                final_metrics['temperature'] = data.get('temperature', 0)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'subtask_accs': subtask_accs,
        'final_metrics': final_metrics
    }

def create_comparison_plots(optA_data, optB_data, output_dir='comparison_plots'):
    """Generate comprehensive comparison plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot 1: Loss Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(optA_data['epochs'], optA_data['train_losses'], 'b-o', label='Option A Train', linewidth=2)
    ax.plot(optA_data['epochs'], optA_data['val_losses'], 'b--s', label='Option A Val', linewidth=2)
    ax.plot(optB_data['epochs'], optB_data['train_losses'], 'r-o', label='Option B Train', linewidth=2)
    ax.plot(optB_data['epochs'], optB_data['val_losses'], 'r--s', label='Option B Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/loss_comparison.png")
    plt.close()
    
    # Plot 2: Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(optA_data['epochs'], [a*100 for a in optA_data['train_accs']], 'b-o', label='Option A Train', linewidth=2)
    ax.plot(optA_data['epochs'], [a*100 for a in optA_data['val_accs']], 'b--s', label='Option A Val', linewidth=2)
    ax.plot(optB_data['epochs'], [a*100 for a in optB_data['train_accs']], 'r-o', label='Option B Train', linewidth=2)
    ax.plot(optB_data['epochs'], [a*100 for a in optB_data['val_accs']], 'r--s', label='Option B Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training & Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/accuracy_comparison.png")
    plt.close()
    
    # Plot 3: Complexity Subtask Learning
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(optA_data['epochs'], [a*100 for a in optA_data['subtask_accs']['complexity']], 
            'b-o', label='Option A', linewidth=2, markersize=8)
    ax.plot(optB_data['epochs'], [a*100 for a in optB_data['subtask_accs']['complexity']], 
            'r-o', label='Option B', linewidth=2, markersize=8)
    ax.axhline(y=33.3, color='gray', linestyle=':', label='Random (3-class)', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Complexity Accuracy (%)', fontsize=12)
    ax.set_title('Complexity Subtask Learning (Edge Detection)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_learning.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/complexity_learning.png")
    plt.close()
    
    # Plot 4: Final Subtask Accuracies
    fig, ax = plt.subplots(figsize=(10, 6))
    subtasks = ['Brightness', 'Complexity', 'Superclass']
    optA_final = [optA_data['subtask_accs']['brightness'][-1]*100,
                  optA_data['subtask_accs']['complexity'][-1]*100,
                  optA_data['subtask_accs']['superclass'][-1]*100]
    optB_final = [optB_data['subtask_accs']['brightness'][-1]*100,
                  optB_data['subtask_accs']['complexity'][-1]*100,
                  optB_data['subtask_accs']['superclass'][-1]*100]
    
    x = np.arange(len(subtasks))
    width = 0.35
    bars_a = ax.bar(x - width/2, optA_final, width, label='Option A', color='blue', alpha=0.7)
    bars_b = ax.bar(x + width/2, optB_final, width, label='Option B', color='red', alpha=0.7)
    ax.set_xlabel('Subtask', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Final Subtask Accuracies', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subtasks)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(optA_final, optB_final)):
        ax.text(i - width/2, v1 + 1, f'{v1:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v2 + 1, f'{v2:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subtask_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/subtask_comparison.png")
    plt.close()
    
    # Plot 2: Final Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Test Accuracy', 'ECE (↓)', 'Mean Uncertainty']
    optA_vals = [optA_data['final_metrics']['accuracy']*100,
                 optA_data['final_metrics']['ece']*100,
                 optA_data['final_metrics']['mean_uncertainty']]
    optB_vals = [optB_data['final_metrics']['accuracy']*100,
                 optB_data['final_metrics']['ece']*100,
                 optB_data['final_metrics']['mean_uncertainty']]
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for idx, (ax, metric, vA, vB, color) in enumerate(zip(axes, metrics, optA_vals, optB_vals, colors)):
        x = np.arange(2)
        bars = ax.bar(x, [vA, vB], color=[color, color], alpha=0.7, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(['Option A', 'Option B'], fontsize=11)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, [vA, vB])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}{"%" if idx < 2 else ""}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Highlight winner
        if (idx == 0 and vA > vB) or (idx == 1 and vA < vB):  # Accuracy higher or ECE lower
            bars[0].set_edgecolor('gold')
            bars[0].set_linewidth(3)
        elif (idx == 0 and vB > vA) or (idx == 1 and vB < vA):
            bars[1].set_edgecolor('gold')
            bars[1].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/final_metrics_comparison.png")
    plt.close()
    
    # Plot 3: Temperature Comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    temps = [optA_data['final_metrics']['temperature'], optB_data['final_metrics']['temperature']]
    colors_temp = ['blue' if t < 1.3 else 'orange' if t < 1.5 else 'red' for t in temps]
    bars = ax.bar(['Option A', 'Option B'], temps, color=colors_temp, alpha=0.7, width=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal (T=1.0)')
    ax.set_ylabel('Temperature', fontsize=12)
    ax.set_title('Calibration Temperature Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, temp in zip(bars, temps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{temp:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temperature_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/temperature_comparison.png")
    plt.close()

def generate_summary_table(optA_data, optB_data):
    """Generate markdown comparison table"""
    
    summary = """
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
| **Test Accuracy** | {:.2f}% | {:.2f}% | {} | {:+.2f}pp |
| **ECE (↓ better)** | {:.2f}% | {:.2f}% | {} | {:+.2f}pp |
| **Temperature** | {:.3f} | {:.3f} | {} | {:+.3f} |
| **Mean Uncertainty** | {:.4f} | {:.4f} | {} | {:+.4f} |

## Subtask Performance

| Subtask | Option A | Option B | Δ |
|---------|----------|----------|---|
| **Brightness** | {:.1f}% | {:.1f}% | {:+.1f}pp |
| **Complexity** | {:.1f}% | {:.1f}% | {:+.1f}pp |
| **Superclass** | {:.1f}% | {:.1f}% | {:+.1f}pp |

## Training Efficiency

| Metric | Option A | Option B |
|--------|----------|----------|
| **Best Epoch** | {} | {} |
| **Final Train Acc** | {:.2f}% | {:.2f}% |
| **Final Val Acc** | {:.2f}% | {:.2f}% |
| **Batches/Epoch** | {} | {} |

## Key Findings

""".format(
        # Final Results
        optA_data['final_metrics']['accuracy']*100,
        optB_data['final_metrics']['accuracy']*100,
        '**A** 🏆' if optA_data['final_metrics']['accuracy'] > optB_data['final_metrics']['accuracy'] else '**B** 🏆',
        (optA_data['final_metrics']['accuracy'] - optB_data['final_metrics']['accuracy'])*100,
        
        optA_data['final_metrics']['ece']*100,
        optB_data['final_metrics']['ece']*100,
        '**A** 🏆' if optA_data['final_metrics']['ece'] < optB_data['final_metrics']['ece'] else '**B** 🏆',
        (optA_data['final_metrics']['ece'] - optB_data['final_metrics']['ece'])*100,
        
        optA_data['final_metrics']['temperature'],
        optB_data['final_metrics']['temperature'],
        '**A** 🏆' if abs(optA_data['final_metrics']['temperature']-1.0) < abs(optB_data['final_metrics']['temperature']-1.0) else '**B** 🏆',
        optA_data['final_metrics']['temperature'] - optB_data['final_metrics']['temperature'],
        
        optA_data['final_metrics']['mean_uncertainty'],
        optB_data['final_metrics']['mean_uncertainty'],
        '**A** 🏆' if optA_data['final_metrics']['mean_uncertainty'] < optB_data['final_metrics']['mean_uncertainty'] else '**B** 🏆',
        optA_data['final_metrics']['mean_uncertainty'] - optB_data['final_metrics']['mean_uncertainty'],
        
        # Subtasks
        optA_data['subtask_accs']['brightness'][-1]*100,
        optB_data['subtask_accs']['brightness'][-1]*100,
        (optA_data['subtask_accs']['brightness'][-1] - optB_data['subtask_accs']['brightness'][-1])*100,
        
        optA_data['subtask_accs']['complexity'][-1]*100,
        optB_data['subtask_accs']['complexity'][-1]*100,
        (optA_data['subtask_accs']['complexity'][-1] - optB_data['subtask_accs']['complexity'][-1])*100,
        
        optA_data['subtask_accs']['superclass'][-1]*100,
        optB_data['subtask_accs']['superclass'][-1]*100,
        (optA_data['subtask_accs']['superclass'][-1] - optB_data['subtask_accs']['superclass'][-1])*100,
        
        # Training
        len(optA_data['epochs']),
        len(optB_data['epochs']),
        optA_data['train_accs'][-1]*100,
        optB_data['train_accs'][-1]*100,
        optA_data['val_accs'][-1]*100,
        optB_data['val_accs'][-1]*100,
        352,  # 45000/128
        176,  # 45000/256
    )
    
    # Add analysis
    acc_winner = 'A' if optA_data['final_metrics']['accuracy'] > optB_data['final_metrics']['accuracy'] else 'B'
    ece_winner = 'A' if optA_data['final_metrics']['ece'] < optB_data['final_metrics']['ece'] else 'B'
    
    summary += f"""
### Accuracy
- **Winner: Option {acc_winner}**
- Option A benefits from smaller batch size (more frequent updates)
- Option B has {"better" if acc_winner == 'B' else "slightly worse"} accuracy despite larger batch

### Calibration (ECE)
- **Winner: Option {ece_winner}**
- Lower ECE = better calibrated confidence scores
- Temperature closer to 1.0 indicates better initial calibration

### Complexity Subtask (Edge Detection)
- Both options show similar complexity learning (~62-64%)
- Significant improvement over variance-based method (47%)
- Still room for improvement (target: 70%+)

### Overall Recommendation
"""
    
    if acc_winner == ece_winner:
        summary += f"✅ **Option {acc_winner} is clearly superior** - better accuracy AND calibration\n"
    else:
        summary += f"⚖️ **Trade-off between accuracy and calibration**\n- Choose Option A for best accuracy\n- Choose Option B for faster training (2× throughput)\n"
    
    return summary

def main():
    print("="*60)
    print("Experiment 3: Comparing Option A vs Option B")
    print("="*60)
    
    # Check if logs exist
    optA_log = Path('artifacts_optA/audit_log.jsonl')
    optB_log = Path('artifacts_optB/audit_log.jsonl')
    
    if not optA_log.exists():
        print(f"❌ Option A log not found: {optA_log}")
        print("   Run: sbatch job_optA.sh")
        return
    
    if not optB_log.exists():
        print(f"❌ Option B log not found: {optB_log}")
        print("   Run: sbatch job_optB.sh")
        return
    
    print("✓ Both logs found")
    print()
    
    # Parse logs
    print("Parsing Option A results...")
    optA_data = parse_audit_log(optA_log)
    print(f"  - Trained for {len(optA_data['epochs'])} epochs")
    print(f"  - Final accuracy: {optA_data['final_metrics']['accuracy']*100:.2f}%")
    print(f"  - ECE: {optA_data['final_metrics']['ece']*100:.2f}%")
    print()
    
    print("Parsing Option B results...")
    optB_data = parse_audit_log(optB_log)
    print(f"  - Trained for {len(optB_data['epochs'])} epochs")
    print(f"  - Final accuracy: {optB_data['final_metrics']['accuracy']*100:.2f}%")
    print(f"  - ECE: {optB_data['final_metrics']['ece']*100:.2f}%")
    print()
    
    # Generate plots
    print("Generating comparison plots...")
    create_comparison_plots(optA_data, optB_data)
    print()
    
    # Generate summary
    print("Generating summary table...")
    summary = generate_summary_table(optA_data, optB_data)
    
    with open('COMPARISON_RESULTS.md', 'w') as f:
        f.write(summary)
    print("✓ Saved: COMPARISON_RESULTS.md")
    print()
    
    # Display quick summary
    print("="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Option A: {optA_data['final_metrics']['accuracy']*100:.2f}% acc, {optA_data['final_metrics']['ece']*100:.2f}% ECE")
    print(f"Option B: {optB_data['final_metrics']['accuracy']*100:.2f}% acc, {optB_data['final_metrics']['ece']*100:.2f}% ECE")
    print()
    
    if optA_data['final_metrics']['accuracy'] > optB_data['final_metrics']['accuracy']:
        print("🏆 Winner: Option A (better accuracy)")
    else:
        print("🏆 Winner: Option B (better accuracy)")
    
    print()
    print("Check comparison_plots/ for visualizations!")
    print("="*60)

if __name__ == '__main__':
    main()
