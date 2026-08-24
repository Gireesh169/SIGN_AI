import os
import sys
import json
import csv
import shutil
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    X_FEATURES_FILE, Y_LABELS_FILE, LABEL_MAPPING_FILE,
    MODELS_DIR, DATA_DIR, BASE_DIR, DEVICE,
    HIDDEN_DIM, DROPOUT, normalize_sequence_features
)
from model import BiLSTM_GRU_Model

# Define Experiment Directory
EXP_DIR = os.path.join(MODELS_DIR, 'experiments', 'normalized_smoothed_252')
os.makedirs(EXP_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(EXP_DIR, 'best_model.pth')
FINAL_MODEL_PATH = os.path.join(EXP_DIR, 'final_model.pth')
PREPROC_CONFIG_PATH = os.path.join(EXP_DIR, 'preprocessing_config.json')
TRAINING_HISTORY_PATH = os.path.join(EXP_DIR, 'training_history.json')
EVAL_METRICS_PATH = os.path.join(EXP_DIR, 'evaluation_metrics.json')
LABEL_MAPPING_PATH = os.path.join(EXP_DIR, 'label_mapping.json')
CONFUSION_MATRIX_PATH = os.path.join(EXP_DIR, 'confusion_matrix.png')
TRAINING_CURVES_PATH = os.path.join(EXP_DIR, 'training_curves.png')
CLASSIFICATION_REPORT_PATH = os.path.join(EXP_DIR, 'classification_report.txt')

BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model_252.pth')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_252d_features(pos_seq_30x126):
    """
    Computes winning 252D representation:
    1. Wrist-relative centering & palm scale normalization (126D)
    2. Causal First-Order Velocity (126D): v[t] = p[t] - p[t-1], v[0] = zeros(126)
    3. Concatenates Position (126D) + Velocity (126D) -> 252D
    """
    pos_norm = normalize_sequence_features(pos_seq_30x126) # (30, 126)
    vel_norm = np.zeros_like(pos_norm) # (30, 126)
    for t in range(1, 30):
        vel_norm[t] = pos_norm[t] - pos_norm[t-1]

    feat_252 = np.concatenate([pos_norm, vel_norm], axis=-1)
    return feat_252

def augment_raw_sequence(pos_seq):
    seq = pos_seq.copy()
    T, D = seq.shape

    # 1. Small Landmark Gaussian Noise
    if np.random.rand() < 0.50:
        noise = np.random.normal(0, 0.003, size=seq.shape)
        seq = seq + noise

    # 2. Small Wrist-relative Translation Jitter
    if np.random.rand() < 0.40:
        shift_xy = np.random.uniform(-0.015, 0.015, size=(1, 3))
        seq_l = seq[:, :63].reshape(T, 21, 3)
        seq_r = seq[:, 63:].reshape(T, 21, 3)
        
        for h in [seq_l, seq_r]:
            mask = (np.abs(h).sum(axis=-1, keepdims=True) > 1e-6).astype(np.float32)
            h += shift_xy * mask

        seq = np.concatenate([seq_l.reshape(T, 63), seq_r.reshape(T, 63)], axis=-1)

    # 3. Small Wrist-relative Scale Jitter
    if np.random.rand() < 0.40:
        scale_factor = np.random.uniform(0.97, 1.03)
        seq_l = seq[:, :63].reshape(T, 21, 3)
        seq_r = seq[:, 63:].reshape(T, 21, 3)
        
        for h in [seq_l, seq_r]:
            mask = (np.abs(h).sum(axis=-1, keepdims=True) > 1e-6).astype(np.float32)
            h *= scale_factor * mask

        seq = np.concatenate([seq_l.reshape(T, 63), seq_r.reshape(T, 63)], axis=-1)

    return seq

def expand_training_dataset(X_train, y_train, target_samples_per_class=25):
    print("\n" + "="*60)
    print(f"=== EXPANDING TRAINING SET (TARGET = {target_samples_per_class} SAMPLES/CLASS) ===")
    print("="*60)
    unique_classes = np.unique(y_train)
    X_exp, y_exp = [], []

    for c in sorted(unique_classes):
        cls_indices = np.where(y_train == c)[0]
        cls_samples = X_train[cls_indices]
        n_orig = len(cls_samples)

        for s in cls_samples:
            X_exp.append(s)
            y_exp.append(c)

        n_needed = max(0, target_samples_per_class - n_orig)
        if n_needed > 0:
            for i in range(n_needed):
                base_seq = cls_samples[i % n_orig]
                aug_seq = augment_raw_sequence(base_seq)
                X_exp.append(aug_seq)
                y_exp.append(c)

    X_exp = np.array(X_exp, dtype=np.float32)
    y_exp = np.array(y_exp, dtype=np.int64)
    print(f"Expanded Training Set Size: {len(X_exp)} samples across {len(unique_classes)} active classes.")
    return X_exp, y_exp

class LandmarkDataset(Dataset):
    def __init__(self, X_pos_list, y_list, is_training=False):
        self.X_pos = X_pos_list
        self.y = y_list
        self.is_training = is_training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        pos_raw = self.X_pos[idx]
        label = self.y[idx]

        if self.is_training and np.random.rand() < 0.30:
            pos_raw = augment_raw_sequence(pos_raw)

        feat_252 = compute_252d_features(pos_raw)
        return torch.tensor(feat_252, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def safe_dataset_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    print("\n" + "="*60)
    print("=== DATASET STRATIFIED SPLIT & INTEGRITY CHECK ===")
    print("="*60)
    counts = Counter(y)
    unique_classes = sorted(list(counts.keys()))
    num_classes = len(unique_classes)
    
    print(f"Total extracted sequences: {len(y)}")
    print(f"Total active classes: {num_classes}")

    train_indices = []
    val_indices = []
    test_indices = []

    np.random.seed(random_state)
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        n = len(cls_indices)
        
        if n >= 3:
            n_test = max(1, int(round(n * test_size)))
            n_val = max(1, int(round(n * val_size)))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = 1
                n_val = max(0, n - 2)
                n_test = max(0, n - 1 - n_val)
                
            test_indices.extend(cls_indices[:n_test])
            val_indices.extend(cls_indices[n_test:n_test + n_val])
            train_indices.extend(cls_indices[n_test + n_val:])
        elif n == 2:
            train_indices.append(cls_indices[0])
            val_indices.append(cls_indices[1])
        else:
            train_indices.append(cls_indices[0])

    s_train, s_val, s_test = set(train_indices), set(val_indices), set(test_indices)
    assert len(s_train.intersection(s_val)) == 0, "Data leakage between train and val!"
    assert len(s_train.intersection(s_test)) == 0, "Data leakage between train and test!"
    assert len(s_val.intersection(s_test)) == 0, "Data leakage between val and test!"

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("Split counts (No sample leakage):")
    print(f"  Training set  : {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set      : {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_and_evaluate_experiment():
    set_seed(42)

    if not os.path.exists(X_FEATURES_FILE) or not os.path.exists(Y_LABELS_FILE):
        raise FileNotFoundError("Extracted dataset files not found!")

    X_raw = np.load(X_FEATURES_FILE) # (673, 30, 126)
    y_raw = np.load(Y_LABELS_FILE)   # (673,)

    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    shutil.copy(LABEL_MAPPING_FILE, LABEL_MAPPING_PATH)

    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    num_classes = len(label_mapping)

    preproc_config = {
        "experiment_name": "normalized_smoothed_252",
        "input_dim": 252,
        "position_dim": 126,
        "velocity_dim": 126,
        "sequence_length": 30,
        "normalization": "wrist_relative_palm_scale",
        "casual_velocity": True,
        "target_samples_per_class": 25,
        "label_smoothing": 0.05,
        "adamw_weight_decay": 0.0001,
        "random_seed": 42
    }
    with open(PREPROC_CONFIG_PATH, 'w') as f:
        json.dump(preproc_config, f, indent=2)

    # 1. Stratified Split FIRST
    (X_train_orig, y_train_orig), (X_val, y_val), (X_test, y_test) = safe_dataset_split(X_raw, y_raw, test_size=0.15, val_size=0.15, random_state=42)

    # 2. Expand Training Dataset ONLY
    X_train, y_train = expand_training_dataset(X_train_orig, y_train_orig, target_samples_per_class=25)

    # Compute Class Weights for Loss
    train_counts = Counter(y_train)
    max_count = max(train_counts.values()) if len(train_counts) > 0 else 1
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        cnt = train_counts.get(c, 0)
        w = np.sqrt(max_count / (cnt + 1.0))
        class_weights[c] = float(w)
    class_weights = class_weights / class_weights.mean()

    # 3. Create PyTorch DataLoaders
    train_dataset = LandmarkDataset(X_train, y_train, is_training=True)
    val_dataset = LandmarkDataset(X_val, y_val, is_training=False)
    test_dataset = LandmarkDataset(X_test, y_test, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Instantiate Model
    model = BiLSTM_GRU_Model(
        input_dim=252,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_bilstm_layers=1,
        num_gru_layers=1,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5)

    num_epochs = 60
    patience = 15
    patience_counter = 0
    best_val_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    print("\n" + "="*60)
    print("=== STARTING TRAINING: CONTROLLED NORMALIZED 252D EXPERIMENT ===")
    print("="*60)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss_epoch = train_loss / train_total
        train_acc_epoch = train_correct / train_total

        # Validation Step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * len(batch_y)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss_epoch = val_loss / val_total
        val_acc_epoch = val_correct / val_total

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc_epoch)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['train_acc'].append(train_acc_epoch)
        history['val_acc'].append(val_acc_epoch)
        history['lr'].append(current_lr)

        print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch*100:.2f}% | Val Loss: {val_loss_epoch:.4f} | Val Acc: {val_acc_epoch*100:.2f}% | LR: {current_lr:.6f}")

        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  --> Save Best Checkpoint! (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Triggered after {epoch} epochs (No val acc improvement for {patience} epochs).")
                break

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds. Best Val Acc: {best_val_acc*100:.2f}%")

    with open(TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([a*100 for a in history['train_acc']], label='Train Acc %', color='blue')
    plt.plot([a*100 for a in history['val_acc']], label='Val Acc %', color='orange')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH)
    plt.close()

    # 6. Evaluation on Untouched Test Set
    print("\n" + "="*60)
    print("=== EVALUATION ON UNTOUCHED TEST SET ===")
    print("="*60)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_targets, all_top5 = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)

            top1 = logits.argmax(dim=1).cpu().numpy()
            top5 = torch.topk(probs, k=min(5, num_classes), dim=1).indices.cpu().numpy()

            all_preds.extend(top1)
            all_targets.extend(batch_y.cpu().numpy())
            all_top5.extend(top5)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_acc = accuracy_score(all_targets, all_preds)
    top5_acc = np.mean([tgt in t5 for tgt, t5 in zip(all_targets, all_top5)])
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    recognized_classes = len(set(all_preds))
    gap = (history['train_acc'][-1] - best_val_acc) * 100.0

    print(f"New Model Test Top-1 Accuracy: {test_acc*100:.2f}%")
    print(f"New Model Test Top-5 Accuracy: {top5_acc*100:.2f}%")
    print(f"New Model Macro F1 Score    : {f1_macro:.4f}")
    print(f"New Model Weighted F1 Score : {f1_weighted:.4f}")
    print(f"Train-Val Overfitting Gap   : {gap:.2f}%")
    print(f"Recognized Classes Count    : {recognized_classes} / {num_classes}")

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Set Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    target_names = [idx_to_label.get(i, f"Class_{i}") for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, labels=list(range(num_classes)), target_names=target_names, zero_division=0)
    with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(report)

    metrics = {
        'total_extracted_samples': len(X_raw),
        'number_of_classes': num_classes,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_accuracy': float(history['train_acc'][-1]),
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'top5_accuracy': float(top5_acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'train_val_gap': float(gap),
        'recognized_classes_count': int(recognized_classes),
        'best_model_path': BEST_MODEL_PATH
    }

    with open(EVAL_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    # 7. Unseen Video Test Suite (15 Videos)
    print("\n" + "="*60)
    print("=== UNSEEN VIDEO EVALUATION SUITE ===")
    print("="*60)
    unseen_correct = 0
    unseen_total = min(20, len(X_test))

    for i in range(unseen_total):
        pos_raw = X_test[i]
        expected_idx = y_test[i]
        expected_label = idx_to_label.get(expected_idx, f"Class_{expected_idx}")

        feat_252 = compute_252d_features(pos_raw)
        inp_tensor = torch.tensor(feat_252, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(inp_tensor)
            probs = torch.softmax(logits, dim=1)
            top5_probs, top5_idx = torch.topk(probs, k=min(5, num_classes), dim=1)

        pred_idx = top5_idx[0][0].item()
        pred_label = idx_to_label.get(pred_idx, f"Class_{pred_idx}")
        pred_conf = top5_probs[0][0].item() * 100.0

        top5_list = [f"{idx_to_label.get(idx.item(), f'Class_{idx.item()}')} ({p.item()*100:.1f}%)" for p, idx in zip(top5_probs[0], top5_idx[0])]
        is_correct = (pred_idx == expected_idx)

        if is_correct:
            unseen_correct += 1

        print(f"Video #{i+1:02d} | Expected: '{expected_label}' | Pred: '{pred_label}' ({pred_conf:.1f}%) | Correct: {is_correct}")

    unseen_acc = (unseen_correct / unseen_total) * 100.0
    print(f"\nUnseen Video Test Accuracy: {unseen_acc:.2f}% ({unseen_correct}/{unseen_total})")

    # 8. Baseline Metrics Comparison
    baseline_test_acc = 36.08
    baseline_top5_acc = 70.10
    baseline_macro_f1 = 0.3203
    baseline_unseen_acc = 35.00

    print("\n" + "="*80)
    print("=== FAIR MODEL COMPARISON TABLE ===")
    print("="*80)
    print(f"{'Metric':<25} | {'Existing 252D Baseline':<22} | {'New Improved 252D Model':<23} | {'Winner'}")
    print("-" * 80)

    w_top1 = "🏆 NEW MODEL" if (test_acc * 100.0) > baseline_test_acc else "BASELINE"
    w_top5 = "🏆 NEW MODEL" if (top5_acc * 100.0) > baseline_top5_acc else "BASELINE"
    w_f1 = "🏆 NEW MODEL" if f1_macro > baseline_macro_f1 else "BASELINE"
    w_uns = "🏆 NEW MODEL" if unseen_acc > baseline_unseen_acc else "BASELINE"

    print(f"{'Test Top-1 Accuracy':<25} | {baseline_test_acc:.2f}%                  | {test_acc*100.0:.2f}%                  | {w_top1}")
    print(f"{'Test Top-5 Accuracy':<25} | {baseline_top5_acc:.2f}%                  | {top5_acc*100.0:.2f}%                  | {w_top5}")
    print(f"{'Macro F1 Score':<25} | {baseline_macro_f1:.4f}                 | {f1_macro:.4f}                | {w_f1}")
    print(f"{'Unseen Video Accuracy':<25} | {baseline_unseen_acc:.2f}%                  | {unseen_acc:.2f}%                  | {w_uns}")
    print("="*80)

    replace_recommendation = (test_acc * 100.0) > baseline_test_acc
    if replace_recommendation:
        print(f"\nConclusion: New model achieved {test_acc*100:.2f}% Top-1 Acc (> {baseline_test_acc}% baseline). RECOMMENDED TO REPLACE.")
    else:
        print(f"\nConclusion: New model achieved {test_acc*100:.2f}% Top-1 Acc. Baseline ({baseline_test_acc}%) remains untouched.")

if __name__ == '__main__':
    train_and_evaluate_experiment()
