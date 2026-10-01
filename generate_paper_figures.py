import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch

# Ensure output directory exists
FIGURES_DIR = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------
# IEEE Academic Styling Configuration
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e5e5e5'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['figure.autolayout'] = True

# Colorblind-friendly palette
C_PRIMARY = '#1f77b4'  # Steel Blue
C_SECONDARY = '#ff7f0e' # Muted Orange
C_TERTIARY = '#2ca02c'  # Muted Green
C_ACCENT = '#d62728'    # Crimson Red
C_PURPLE = '#9467bd'    # Deep Purple
C_GREY = '#4a5568'      # Slate Grey

def save_fig(fig, basename):
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {basename}.png & {basename}.pdf")

# =========================================================
# FIGURE 1: Feature Representation Top-1 & Top-5 Accuracy
# =========================================================
def generate_fig1():
    labels = ['Position Only\n(126D)', 'Position + Velocity\n(252D)', 'Position + Velocity\n+ Acceleration (378D)']
    top1 = [19.59, 36.08, 27.84]
    top5 = [62.89, 70.10, 65.98]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rects1 = ax.bar(x - width/2, top1, width, label='Top-1 Accuracy', color=C_PRIMARY, edgecolor='black', linewidth=0.6)
    rects2 = ax.bar(x + width/2, top5, width, label='Top-5 Accuracy', color=C_SECONDARY, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Comparison of Frame-Level Feature Representations', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Value annotations
    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    save_fig(fig, 'fig01_feature_accuracy')

# =========================================================
# FIGURE 2: Macro-F1 Across Feature Representations
# =========================================================
def generate_fig2():
    labels = ['Position Only\n(126D)', 'Position + Velocity\n(252D)', 'Position + Velocity\n+ Acceleration (378D)']
    macro_f1 = [0.1512, 0.3203, 0.2206]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(labels, macro_f1, color=C_TERTIARY, width=0.45, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Macro-F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('Macro-F1 Across Feature Representations', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 0.40)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.4f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    save_fig(fig, 'fig02_feature_macro_f1')

# =========================================================
# FIGURE 3: Multimodal Landmark Fusion
# =========================================================
def generate_fig3():
    labels = ['Hand Only', 'Hand + Face', 'Hand + Pose', 'Hand + Face + Pose']
    top1 = [9.28, 4.12, 4.12, 6.19]
    top5 = [23.71, 13.40, 19.59, 22.68]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    rects1 = ax.bar(x - width/2, top1, width, label='Top-1 Accuracy', color=C_PRIMARY, edgecolor='black', linewidth=0.6)
    rects2 = ax.bar(x + width/2, top5, width, label='Top-5 Accuracy', color=C_SECONDARY, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Effect of Multimodal Landmark Fusion', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.set_ylim(0, 30)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    save_fig(fig, 'fig03_multimodal_accuracy')

# =========================================================
# FIGURE 4A & 4B: Motion, Visual, and Gated Multimodal
# =========================================================
def generate_fig4():
    labels = ['Motion Baseline', 'Visual Crops Only', 'Gated Multimodal']
    top1 = [9.28, 1.03, 14.43]
    top5 = [35.05, 4.12, 43.30]
    latency = [0.39, 0.10, 0.58]

    # Fig 4A: Accuracy
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rects1 = ax.bar(x - width/2, top1, width, label='Top-1 Accuracy', color=C_PRIMARY, edgecolor='black', linewidth=0.6)
    rects2 = ax.bar(x + width/2, top5, width, label='Top-5 Accuracy', color=C_SECONDARY, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Motion, Visual, and Gated Multimodal Configurations', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.set_ylim(0, 52)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    save_fig(fig, 'fig04_motion_visual_gated')

    # Fig 4B: Latency
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(labels, latency, color=C_ACCENT, width=0.45, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Inference Latency (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Measured Inference Latency of Modality Branches', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 0.75)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.2f} ms', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    save_fig(fig, 'fig04b_latency')

# =========================================================
# FIGURE 5: Neural Network Architecture Diagram
# =========================================================
def generate_fig5():
    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    ax.axis('off')

    # Layer definitions: (Title, Subtitle/Shape, Color)
    layers = [
        ("Input Sequence", "(B, 30, 252) — 126D Position + 126D Velocity", "#e2e8f0"),
        ("1D Temporal Convolution", "Conv1D(in=504, out=32, k=3, p=1) + BatchNorm + SiLU", "#bfdbfe"),
        ("Bidirectional LSTM Layer", "Hidden = 32 per dir (Output: B, 30, 64)", "#bbf7d0"),
        ("Bidirectional GRU Layer", "Hidden = 32 per dir (Output: B, 30, 64)", "#fef08a"),
        ("Temporal Attention & Pooling", "Softmax Context (64D) ⊕ Mean Pooling (64D) → (B, 128)", "#fed7aa"),
        ("Dense Feature Head", "Linear(128 → 64) + LayerNorm + ReLU + Dropout(0.30)", "#e9d5ff"),
        ("Classification Output", "Linear(64 → 97 Classes) → Softmax Predictions", "#fbcfe8")
    ]

    n_layers = len(layers)
    y_starts = np.linspace(0.92, 0.08, n_layers)

    for i, ((title, sub, color), y) in enumerate(zip(layers, y_starts)):
        # Box
        rect = matplotlib.patches.FancyBboxPatch((0.15, y - 0.04), 0.70, 0.07, boxstyle="round,pad=0.005", facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(0.50, y + 0.005, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#111827')
        ax.text(0.50, y - 0.020, sub, ha='center', va='center', fontsize=9.0, fontstyle='italic', color='#374151')

        # Arrow down
        if i < n_layers - 1:
            next_y = y_starts[i+1]
            ax.annotate('', xy=(0.50, next_y + 0.035), xytext=(0.50, y - 0.04),
                        arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5))

    ax.set_title("Proposed Neural Network Architecture\n(Conv1D + BiLSTM + BiGRU + Temporal Attention)", fontsize=13, fontweight='bold', pad=15)
    save_fig(fig, 'fig05_model_architecture')

# =========================================================
# FIGURE 6: Architectural Ablation Study
# =========================================================
def generate_fig6():
    labels = ['Baseline', 'Geometric\nFeatures', 'Gated\nMultimodal', 'Pretrained\n+ Gated']
    top1 = [16.49, 16.49, 17.53, 22.68]
    top5 = [51.55, 49.48, 51.55, 49.48]
    macro_f1 = [0.1254, 0.1081, 0.1199, 0.1801]

    # Panel A: Accuracy
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    rects1 = ax.bar(x - width/2, top1, width, label='Top-1 Accuracy', color=C_PRIMARY, edgecolor='black', linewidth=0.6)
    rects2 = ax.bar(x + width/2, top5, width, label='Top-5 Accuracy', color=C_SECONDARY, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Architectural Ablation Study — Top-1 & Top-5 Accuracy', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.set_ylim(0, 60)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    save_fig(fig, 'fig06_architecture_ablation')

    # Panel B: Macro F1
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(labels, macro_f1, color=C_PURPLE, width=0.45, edgecolor='black', linewidth=0.6)
    ax.set_ylabel('Macro-F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('Architectural Ablation Study — Macro-F1', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 0.24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    save_fig(fig, 'fig06b_architecture_macro_f1')

# =========================================================
# FIGURE 7: Transfer Learning Comparison
# =========================================================
def generate_fig7():
    labels = ['ImageNet-Pretrained CNN', 'Word-Level Pretrained CNN']
    top1 = [9.28, 19.59]
    top5 = [32.99, 49.48]
    latency = [0.34, 0.46]

    # Fig 7A: Accuracy
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    rects1 = ax.bar(x - width/2, top1, width, label='Top-1 Accuracy', color=C_PRIMARY, edgecolor='black', linewidth=0.6)
    rects2 = ax.bar(x + width/2, top5, width, label='Top-5 Accuracy', color=C_SECONDARY, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Transfer Learning Comparison (Visual Backbone)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.set_ylim(0, 60)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    save_fig(fig, 'fig07_transfer_learning')

    # Fig 7B: Latency
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(labels, latency, color=C_GREY, width=0.4, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Inference Latency (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Latency of Pretrained CNN Backbones', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 0.60)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.2f} ms', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    save_fig(fig, 'fig07b_transfer_learning_latency')

# =========================================================
# FIGURE 8: Training Convergence Curves (Real History)
# =========================================================
def generate_fig8():
    history_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/models/experiments/normalized_smoothed_252/training_history.json"
    if not os.path.exists(history_path):
        history_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/models/training_history.json"

    with open(history_path, 'r') as f:
        hist = json.load(f)

    epochs = range(1, len(hist['train_loss']) + 1)
    train_acc = [a * 100.0 for a in hist['train_acc']]
    val_acc = [a * 100.0 for a in hist['val_acc']]
    train_loss = hist['train_loss']
    val_loss = hist['val_loss']

    # Fig 8A: Accuracy
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, train_acc, label='Training Accuracy', color=C_PRIMARY, linewidth=2.0)
    ax.plot(epochs, val_acc, label='Validation Accuracy', color=C_SECONDARY, linewidth=2.0, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Training Convergence — Accuracy vs Epoch', fontsize=12, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    save_fig(fig, 'fig08_training_validation_accuracy')

    # Fig 8B: Loss
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, train_loss, label='Training Loss', color=C_PRIMARY, linewidth=2.0)
    ax.plot(epochs, val_loss, label='Validation Loss', color=C_ACCENT, linewidth=2.0, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax.set_title('Training Convergence — Loss vs Epoch', fontsize=12, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    save_fig(fig, 'fig08_training_validation_loss')

# =========================================================
# FIGURE 9: 97x97 Confusion Matrix (From Model Evaluation)
# =========================================================
def generate_fig9():
    from src.model import BiLSTM_GRU_Model
    from src.train import safe_dataset_split

    X_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/data/x_hand_features_252.npy"
    y_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/data/y_labels.npy"
    model_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/models/best_model_252.pth"

    X = np.load(X_path)
    y = np.load(y_path)
    (X_tr, y_tr, _), (X_val, y_val, _), (X_test, y_test, _) = safe_dataset_split(X, y, test_size=0.15, val_size=0.15, random_state=42)

    model = BiLSTM_GRU_Model(input_dim=252, hidden_dim=32, num_classes=101, dropout=0.30)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).numpy()

    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Normalized Classification Ratio')
    ax.set_title('Normalized Confusion Matrix (97-Class ISL Test Set)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Sentence Class ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Sentence Class ID', fontsize=11, fontweight='bold')
    save_fig(fig, 'fig09_confusion_matrix')

# =========================================================
# FIGURE 10: Top-1 Accuracy Across All Configurations (Table VIII)
# =========================================================
def generate_fig10():
    data = [
        ("252D Position + Velocity", 36.08),
        ("378D Position + Velocity + Acceleration", 27.84),
        ("Pretrained + Gated Multimodal", 22.68),
        ("126D Position Only", 19.59),
        ("Word-Pretrained CNN", 19.59),
        ("Gated Multimodal (LM + Visual)", 17.53),
        ("Geometric Features (280D)", 16.49),
        ("Baseline Recurrent (252D)", 16.49),
        ("Gated Multimodal (Hand-Crop)", 14.43),
        ("Motion Baseline", 9.28),
        ("Default ImageNet CNN", 9.28),
        ("Hand Only LM", 9.28),
        ("Hand + Face + Pose LM", 6.19),
        ("Hand + Face / Hand + Pose LM", 4.12),
        ("Visual Crops Only", 1.03)
    ]
    # Sort descending
    data = sorted(data, key=lambda x: x[1])

    names = [d[0] for d in data]
    accs = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    bars = ax.barh(names, accs, color=C_PRIMARY, edgecolor='black', linewidth=0.5, height=0.65)

    ax.set_xlabel('Top-1 Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Top-1 Accuracy Across All Evaluated Configurations (Table VIII)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(0, 42)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.6, bar.get_y() + bar.get_height()/2.0, f'{w:.2f}%', ha='left', va='center', fontsize=8.5, fontweight='bold')

    save_fig(fig, 'fig10_all_experiments_accuracy')

# =========================================================
# FIGURE 11: Inference Latency of Evaluated Configurations
# =========================================================
def generate_fig11():
    configs = [
        'Visual Crops Only',
        'ImageNet CNN',
        'Motion Baseline',
        'Word-Pretrained CNN',
        'Gated Multimodal'
    ]
    latency = [0.10, 0.34, 0.39, 0.46, 0.58]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(configs, latency, color=C_GREY, width=0.45, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Inference Latency (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Latency of Evaluated Configurations', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 0.70)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.2f} ms', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    save_fig(fig, 'fig11_latency')

# =========================================================
# FIGURE 12: Real-Time Processing Budget & Latency Ranges
# =========================================================
def generate_fig12():
    categories = [
        'Model Inference Latency',
        '30-FPS Frame Budget',
        'MediaPipe Landmark Extraction'
    ]
    # Range representation: (min, max, mean/mid)
    mins = [0.10, 33.33, 46.60]
    maxs = [0.58, 33.33, 95.00]
    mids = [0.34, 33.33, 70.80]

    y_pos = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    
    # Horizontal range bars
    for i in range(len(categories)):
        if mins[i] == maxs[i]:
            ax.plot([mins[i]], [y_pos[i]], marker='o', color=C_PRIMARY, markersize=10)
            ax.text(mins[i] + 2.0, y_pos[i], f'Fixed: {mins[i]:.2f} ms', va='center', fontsize=9.5, fontweight='bold', color=C_PRIMARY)
        else:
            ax.plot([mins[i], maxs[i]], [y_pos[i], y_pos[i]], color=C_ACCENT if i==2 else C_TERTIARY, linewidth=6.0, alpha=0.85)
            ax.plot(mids[i], y_pos[i], marker='|', color='black', markersize=12, markeredgewidth=2)
            ax.text(maxs[i] + 2.0, y_pos[i], f'Range: {mins[i]:.1f} – {maxs[i]:.1f} ms', va='center', fontsize=9.5, fontweight='bold', color='#111827')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (ms / frame)', fontsize=11, fontweight='bold')
    ax.set_title('Recognition Latency vs 30-FPS Real-Time Budget', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(0, 115)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    save_fig(fig, 'fig12_realtime_budget')

# =========================================================
# FIGURE 13: Unseen-Signer Generalisation
# =========================================================
def generate_fig13():
    evals = ['Main Reported Test Set\n(97 Samples)', 'Unseen-Signer Test\n(20 Unseen Videos)']
    accs = [36.08, 10.00]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(evals, accs, color=[C_PRIMARY, C_ACCENT], width=0.45, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Unseen-Signer Generalisation Evaluation', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bars[0].set_label('Held-out Test Split')
    bars[1].set_label('Unseen Signer Evaluation')

    ax.text(bars[0].get_x() + bars[0].get_width()/2.0, accs[0] + 1.2, f'{accs[0]:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.text(bars[1].get_x() + bars[1].get_width()/2.0, accs[1] + 1.2, f'{accs[1]:.1f}%\n(2 / 20 correct)', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=C_ACCENT)

    save_fig(fig, 'fig13_unseen_signer')

# =========================================================
# OPTIONAL FIGURE 14: Temporal Attention Weights Visualization
# =========================================================
def generate_fig14():
    from src.model import BiLSTM_GRU_Model
    from src.train import safe_dataset_split

    X_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/data/x_hand_features_252.npy"
    y_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/data/y_labels.npy"
    model_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/models/best_model_252.pth"

    X = np.load(X_path)
    y = np.load(y_path)
    _, _, (X_test, y_test, _) = safe_dataset_split(X, y, test_size=0.15, val_size=0.15, random_state=42)

    model = BiLSTM_GRU_Model(input_dim=252, hidden_dim=32, num_classes=101, dropout=0.30)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    weights_captured = []
    def attn_hook(module, input, output):
        # output is (context, weights)
        weights_captured.append(output[1].detach().cpu())

    handle = model.attention.register_forward_hook(attn_hook)
    with torch.no_grad():
        inputs = torch.tensor(X_test[:3], dtype=torch.float32)
        logits = model(inputs)
    handle.remove()

    weights_np = weights_captured[0].squeeze(-1).numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i in range(3):
        ax.plot(range(1, 31), weights_np[i], label=f'Sample Sequence {i+1}', linewidth=2.0)

    ax.set_xlabel('Sequence Frame Index (t = 1 ... 30)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Temporal Attention Weight', fontsize=11, fontweight='bold')
    ax.set_title('Temporal Attention Weight Distribution Across Execution Frames', fontsize=12, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    save_fig(fig, 'fig14_temporal_attention')

# =========================================================
# OPTIONAL FIGURE 15: t-SNE Embedding Visualization
# =========================================================
def generate_fig15():
    from src.model import BiLSTM_GRU_Model
    from src.train import safe_dataset_split

    X_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/data/x_hand_features_252.npy"
    y_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/data/y_labels.npy"
    model_path = "/Users/apple/Desktop/SIGNAI ALL MODELS/Combine_model/models/best_model_252.pth"

    X = np.load(X_path)
    y = np.load(y_path)
    _, _, (X_test, y_test, _) = safe_dataset_split(X, y, test_size=0.15, val_size=0.15, random_state=42)

    model = BiLSTM_GRU_Model(input_dim=252, hidden_dim=32, num_classes=101, dropout=0.30)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    embs_captured = []
    def fc_hook(module, input, output):
        # input[0] is the 128D concatenated representation before fc1!
        embs_captured.append(input[0].detach().cpu())

    handle = model.fc1.register_forward_hook(fc_hook)
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        logits = model(inputs)
    handle.remove()

    dense128 = embs_captured[0].numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    emb2d = tsne.fit_transform(dense128)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=y_test, cmap='tab20', alpha=0.85, edgecolors='k', linewidths=0.5, s=60)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11, fontweight='bold')
    ax.set_title('t-SNE Embeddings of Learned Representation (128D -> 2D)', fontsize=12, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    save_fig(fig, 'fig15_tsne_embeddings')

if __name__ == '__main__':
    print("Generating publication figures...")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
    generate_fig9()
    generate_fig10()
    generate_fig11()
    generate_fig12()
    generate_fig13()
    generate_fig14()
    generate_fig15()
    print("All 15 publication figures successfully generated in results/figures/")
