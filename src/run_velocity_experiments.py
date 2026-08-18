import os
import sys
import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    X_FEATURES_FILE, Y_LABELS_FILE, LABEL_MAPPING_FILE, DATA_DIR,
    MODELS_DIR, BASE_DIR, RANDOM_SEED, DEVICE, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT, HIDDEN_DIM, normalize_sequence_features
)
from model import BiLSTM_GRU_Model
from train import augment_landmark_sequence, safe_dataset_split

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CSV_PATH = os.path.join(DATA_DIR, 'dataset_video_status.csv')
EXP_METRICS_PATH = os.path.join(MODELS_DIR, 'velocity_experiment_metrics.json')
EXP_COMPARISON_PLOT_PATH = os.path.join(RESULTS_DIR, 'velocity_experiment_comparison.png')

MODEL_126_PATH = os.path.join(MODELS_DIR, 'best_model_126.pth')
MODEL_252_PATH = os.path.join(MODELS_DIR, 'best_model_252.pth')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_temporal_velocity(seq_arr):
    """
    Computes first-order causal velocity:
    V[0] = 0
    V[t] = P[t] - P[t-1] for t >= 1
    No future frames (t+1) are used.
    Input shape: (N, T, 126) or (T, 126)
    Output shape: (N, T, 126) or (T, 126)
    """
    is_3d = (seq_arr.ndim == 3)
    X_in = seq_arr if is_3d else seq_arr[np.newaxis, ...]
    N, T, D = X_in.shape
    
    vel = np.zeros_like(X_in)
    for t in range(1, T):
        vel[:, t] = X_in[:, t] - X_in[:, t-1]
        
    return vel if is_3d else vel[0]

def expand_training_dataset_mode(X_train, y_train, is_mode_b=False, target_samples_per_class=25):
    """
    Expands training set to target_samples_per_class for every active class.
    If is_mode_b is True, computes 252-D (Position + Velocity) features for each expanded sequence.
    Validation and test sets MUST NOT be passed to this function.
    """
    unique_classes = np.unique(y_train)
    X_exp = []
    y_exp = []
    
    for c in sorted(unique_classes):
        cls_indices = np.where(y_train == c)[0]
        cls_samples = X_train[cls_indices] # (n_orig, 30, 126)
        n_orig = len(cls_samples)
        
        for s in cls_samples:
            if is_mode_b:
                v = compute_temporal_velocity(s)
                feat = np.concatenate([s, v], axis=-1) # (30, 252)
            else:
                feat = s # (30, 126)
            X_exp.append(feat)
            y_exp.append(c)
            
        n_needed = max(0, target_samples_per_class - n_orig)
        if n_needed > 0:
            for i in range(n_needed):
                base_seq = cls_samples[i % n_orig] # (30, 126)
                aug_pos = augment_landmark_sequence(torch.tensor(base_seq, dtype=torch.float32)).numpy() # (30, 126)
                if is_mode_b:
                    v = compute_temporal_velocity(aug_pos)
                    feat = np.concatenate([aug_pos, v], axis=-1) # (30, 252)
                else:
                    feat = aug_pos # (30, 126)
                X_exp.append(feat)
                y_exp.append(c)
                
    return np.array(X_exp, dtype=np.float32), np.array(y_exp, dtype=np.int64)

class DynamicDataset(Dataset):
    def __init__(self, x_data, y_data, is_train=True, is_mode_b=False):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
        self.is_train = is_train
        self.is_mode_b = is_mode_b

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        seq = self.x[idx].clone()
        label = self.y[idx]

        if self.is_train:
            if self.is_mode_b:
                # Split pos (0:126) and vel (126:252)
                pos = seq[:, :126]
                aug_pos = augment_landmark_sequence(pos)
                aug_vel = compute_temporal_velocity(aug_pos.numpy())
                seq = torch.cat([aug_pos, torch.tensor(aug_vel, dtype=torch.float32)], dim=-1)
            else:
                seq = augment_landmark_sequence(seq)

        return seq, label

def compute_top_k_accuracy(outputs, targets, k=5):
    with torch.no_grad():
        topk_preds = torch.topk(outputs, k=min(k, outputs.shape[1]), dim=1).indices
        targets_expanded = targets.view(-1, 1).expand_as(topk_preds)
        correct = (topk_preds == targets_expanded).any(dim=1).float()
        return correct.mean().item()

def train_and_eval_model(feature_dim, X_train, y_train, X_val, y_val, X_test, y_test, num_classes, model_save_path, exp_name):
    set_seed(RANDOM_SEED)
    is_mode_b = (feature_dim == 252)

    train_ds = DynamicDataset(X_train, y_train, is_train=True, is_mode_b=is_mode_b)
    val_ds = DynamicDataset(X_val, y_val, is_train=False, is_mode_b=is_mode_b)
    test_ds = DynamicDataset(X_test, y_test, is_train=False, is_mode_b=is_mode_b)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTM_GRU_Model(
        input_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_bilstm_layers=1,
        num_gru_layers=1,
        dropout=DROPOUT
    ).to(DEVICE)

    # Class weights for loss
    train_counts = Counter(y_train)
    max_count = max(train_counts.values()) if len(train_counts) > 0 else 1
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        cnt = train_counts.get(c, 0)
        class_weights[c] = float(np.sqrt(max_count / (cnt + 1.0)))
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_loss = float('inf')
    best_val_f1 = -1.0
    early_stopping_patience = 8
    no_improve_epochs = 0

    print(f"\nStarting {exp_name} (Feature Dim={feature_dim}, train_size={len(X_train)}) for up to {NUM_EPOCHS} epochs...")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_epoch_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        epoch_time = time.time() - start_epoch_time

        scheduler.step(epoch_val_loss)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)

        if epoch_val_loss < best_val_loss or epoch_val_f1 > best_val_f1:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
            if epoch_val_f1 > best_val_f1:
                best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), model_save_path)
            no_improve_epochs = 0
            best_str = " (Best model saved)"
        else:
            no_improve_epochs += 1
            best_str = ""

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s{best_str}")

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best model for evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu())
            all_targets.extend(batch_y.numpy())

    all_probs = torch.cat(all_probs, dim=0) if len(all_probs) > 0 else torch.zeros((0, num_classes))
    all_targets_t = torch.tensor(all_targets, dtype=torch.long)

    test_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
    top5_acc = compute_top_k_accuracy(all_probs, all_targets_t, k=5) if len(all_targets) > 0 else 0.0
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0) if len(all_targets) > 0 else 0.0
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0) if len(all_targets) > 0 else 0.0

    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    gap = final_train_acc - final_val_acc

    metrics = {
        'exp_name': exp_name,
        'feature_dim': feature_dim,
        'train_accuracy': float(final_train_acc),
        'val_accuracy': float(final_val_acc),
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'top5_accuracy': float(top5_acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'train_val_gap': float(gap),
        'history': history,
        'model': model
    }

    return metrics

def run_controlled_velocity_experiments():
    start_total_time = time.time()
    set_seed(RANDOM_SEED)

    print("="*60)
    print("=== CONTROLLED EXPERIMENT: 126 POSITION vs 252 POSITION+VELOCITY ===")
    print("="*60)

    # 1. Load Raw Landmark Features (673, 30, 126)
    x_raw = np.load(X_FEATURES_FILE)
    y = np.load(Y_LABELS_FILE)

    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    num_classes = len(label_mapping)

    # 2. Independent Hand Normalization (Wrist-centered, Palm scale normalized)
    X_pos = normalize_sequence_features(x_raw) # (673, 30, 126)

    # 3. Compute Causal First-Order Velocity (673, 30, 126)
    X_vel = compute_temporal_velocity(X_pos) # (673, 30, 126)

    # Mode B Combined Feature Array (673, 30, 252)
    X_pos_vel = np.concatenate([X_pos, X_vel], axis=-1)

    print(f"Total dataset samples       : {len(y)}")
    print(f"Mode A (Position) shape      : {X_pos.shape}")
    print(f"Mode B (Pos + Velocity) shape: {X_pos_vel.shape}")
    print(f"Active 97 classes count      : {num_classes}")

    # 4. Stratified Dataset Split FIRST (70% train, 15% val, 15% test with seed 42)
    (X_pos_tr_orig, y_tr_orig, train_idxs), (X_pos_val, y_val, val_idxs), (X_pos_test, y_test, test_idxs) = safe_dataset_split(X_pos, y, random_state=42)

    # Get Mode B split using EXACT SAME split indices
    X_pos_vel_tr_orig = X_pos_vel[train_idxs]
    X_pos_vel_val = X_pos_vel[val_idxs]
    X_pos_vel_test = X_pos_vel[test_idxs]

    # Verify identical labels and lengths
    assert (y_val == y[val_idxs]).all() and (y_test == y[test_idxs]).all()

    print("\nDataset Split Complete (Zero Leakage, Identical for both Experiments):")
    print(f"  Training split original: {len(y_tr_orig)} samples")
    print(f"  Validation split       : {len(y_val)} samples (Untouched)")
    print(f"  Test split             : {len(y_test)} samples (Untouched)")

    # 5. Expand Training Dataset ONLY (Target = 25 sequences per active class = 2,425 total)
    X_pos_tr_exp, y_train_A = expand_training_dataset_mode(X_pos_tr_orig, y_tr_orig, is_mode_b=False, target_samples_per_class=25)
    X_pos_vel_tr_exp, y_train_B = expand_training_dataset_mode(X_pos_tr_orig, y_tr_orig, is_mode_b=True, target_samples_per_class=25)

    assert (y_train_A == y_train_B).all()
    print(f"Expanded Training Dataset : {len(y_train_A)} sequences (25 samples/class)")

    # =========================================================================
    # EXPERIMENT A: MODE A (126 POSITION FEATURES)
    # =========================================================================
    res_A = train_and_eval_model(
        feature_dim=126,
        X_train=X_pos_tr_exp, y_train=y_train_A,
        X_val=X_pos_val, y_val=y_val,
        X_test=X_pos_test, y_test=y_test,
        num_classes=num_classes,
        model_save_path=MODEL_126_PATH,
        exp_name="Experiment A (126 Position Features)"
    )

    # =========================================================================
    # EXPERIMENT B: MODE B (252 POSITION + VELOCITY FEATURES)
    # =========================================================================
    res_B = train_and_eval_model(
        feature_dim=252,
        X_train=X_pos_vel_tr_exp, y_train=y_train_B,
        X_val=X_pos_vel_val, y_val=y_val,
        X_test=X_pos_vel_test, y_test=y_test,
        num_classes=num_classes,
        model_save_path=MODEL_252_PATH,
        exp_name="Experiment B (252 Position + Velocity Features)"
    )

    # =========================================================================
    # 20 UNSEEN TEST VIDEO PREDICTIONS FOR BOTH EXPERIMENTS (SAME 20 TEST SAMPLES)
    # =========================================================================
    print("\n" + "="*60)
    print("=== 20 UNSEEN TEST VIDEO INFERENCE VERIFICATION COMPARISON ===")
    print("="*60)

    # Load video status CSV for file names
    valid_samples = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] == 'processed' and row['class_name'] in set(label_mapping.values()):
                    valid_samples.append(row)

    np.random.seed(RANDOM_SEED)
    sample_picks_20 = np.random.choice(test_idxs, size=min(20, len(test_idxs)), replace=False)

    model_A = res_A['model']
    model_B = res_B['model']
    model_A.eval()
    model_B.eval()

    unseen_correct_A = 0
    unseen_correct_B = 0

    print(f"{'Video #':<8} | {'Video Name':<20} | {'Expected Sentence':<30} | {'Exp A Prediction (126D)':<30} | {'Status A':<10} | {'Exp B Prediction (252D)':<30} | {'Status B':<10}")
    print("-" * 150)

    unseen_log = []

    for i, idx in enumerate(sample_picks_20, start=1):
        meta = valid_samples[idx] if idx < len(valid_samples) else {'class_name': idx_to_label[y[idx]], 'video_name': f'test_{idx}'}
        expected_str = meta['class_name']
        v_name = meta['video_name']

        # Predict Model A (126D)
        sample_x_A = torch.tensor(X_pos[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_A = model_A(sample_x_A)
            pred_idx_A = torch.argmax(out_A, dim=1).item()
            conf_A = torch.softmax(out_A, dim=1)[0, pred_idx_A].item() * 100.0
        pred_str_A = idx_to_label.get(pred_idx_A, str(pred_idx_A))
        match_A = (expected_str.lower() == pred_str_A.lower())
        if match_A: unseen_correct_A += 1

        # Predict Model B (252D)
        sample_x_B = torch.tensor(X_pos_vel[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_B = model_B(sample_x_B)
            pred_idx_B = torch.argmax(out_B, dim=1).item()
            conf_B = torch.softmax(out_B, dim=1)[0, pred_idx_B].item() * 100.0
        pred_str_B = idx_to_label.get(pred_idx_B, str(pred_idx_B))
        match_B = (expected_str.lower() == pred_str_B.lower())
        if match_B: unseen_correct_B += 1

        status_A = "CORRECT" if match_A else "MISMATCH"
        status_B = "CORRECT" if match_B else "MISMATCH"

        print(f"#{i:02d}      | {v_name[:18]:<20} | {expected_str[:28]:<30} | {pred_str_A[:28]:<30} ({conf_A:.1f}%) | {status_A:<10} | {pred_str_B[:28]:<30} ({conf_B:.1f}%) | {status_B:<10}")

        unseen_log.append({
            'sample_num': i,
            'video_name': v_name,
            'expected': expected_str,
            'pred_A_126d': pred_str_A,
            'conf_A': conf_A,
            'match_A': match_A,
            'pred_B_252d': pred_str_B,
            'conf_B': conf_B,
            'match_B': match_B
        })

    unseen_acc_A = (unseen_correct_A / len(sample_picks_20)) * 100.0
    unseen_acc_B = (unseen_correct_B / len(sample_picks_20)) * 100.0

    print("="*150)
    print(f"Unseen 20-Video Accuracy Summary:")
    print(f"  Experiment A (126D Position Only)     : {unseen_correct_A} / 20 correct ({unseen_acc_A:.2f}%)")
    print(f"  Experiment B (252D Position+Velocity) : {unseen_correct_B} / 20 correct ({unseen_acc_B:.2f}%)")
    print("="*150 + "\n")

    # Select Winning Model based on Test & Unseen performance
    if res_B['test_accuracy'] > res_A['test_accuracy'] or unseen_acc_B > unseen_acc_A:
        winning_exp = "Experiment B (252 Position + Velocity Features)"
        winning_model_path = MODEL_252_PATH
    else:
        winning_exp = "Experiment A (126 Position Features)"
        winning_model_path = MODEL_126_PATH

    # Plot Comparison Curves
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(res_A['history']['val_loss'], label='Exp A (126D Position) Val Loss', color='blue')
    plt.plot(res_B['history']['val_loss'], label='Exp B (252D Pos+Vel) Val Loss', color='orange')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(res_A['history']['val_acc'], label='Exp A (126D Position) Val Acc', color='blue')
    plt.plot(res_B['history']['val_acc'], label='Exp B (252D Pos+Vel) Val Acc', color='orange')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(EXP_COMPARISON_PLOT_PATH)
    plt.close()

    # Save Experiment Metrics JSON
    metrics_summary = {
        'exp_A_126d': {
            'test_accuracy': res_A['test_accuracy'],
            'top5_accuracy': res_A['top5_accuracy'],
            'macro_f1': res_A['macro_f1'],
            'weighted_f1': res_A['weighted_f1'],
            'val_accuracy': res_A['val_accuracy'],
            'best_val_accuracy': res_A['best_val_accuracy'],
            'train_val_gap': res_A['train_val_gap'],
            'unseen_20_acc': unseen_acc_A,
            'model_path': MODEL_126_PATH
        },
        'exp_B_252d': {
            'test_accuracy': res_B['test_accuracy'],
            'top5_accuracy': res_B['top5_accuracy'],
            'macro_f1': res_B['macro_f1'],
            'weighted_f1': res_B['weighted_f1'],
            'val_accuracy': res_B['val_accuracy'],
            'best_val_accuracy': res_B['best_val_accuracy'],
            'train_val_gap': res_B['train_val_gap'],
            'unseen_20_acc': unseen_acc_B,
            'model_path': MODEL_252_PATH
        },
        'winning_experiment': winning_exp,
        'winning_model_path': winning_model_path,
        'unseen_test_log': unseen_log
    }

    with open(EXP_METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    total_time = time.time() - start_total_time

    print("="*80)
    print("=== CONTROLLED EXPERIMENT COMPARISON SUMMARY REPORT ===")
    print("="*80)
    print(f"{'Metric':<30} | {'Experiment A (126D Position)':<25} | {'Experiment B (252D Pos+Vel)':<25} | {'Better Model':<15}")
    print("-" * 100)
    print(f"{'Test Accuracy (Top-1)':<30} | {res_A['test_accuracy']*100:.2f}%{'':<18} | {res_B['test_accuracy']*100:.2f}%{'':<18} | {'Exp B' if res_B['test_accuracy'] > res_A['test_accuracy'] else 'Exp A'}")
    print(f"{'Top-5 Accuracy':<30} | {res_A['top5_accuracy']*100:.2f}%{'':<18} | {res_B['top5_accuracy']*100:.2f}%{'':<18} | {'Exp B' if res_B['top5_accuracy'] > res_A['top5_accuracy'] else 'Exp A'}")
    print(f"{'Macro F1 Score':<30} | {res_A['macro_f1']:.4f}{'':<19} | {res_B['macro_f1']:.4f}{'':<19} | {'Exp B' if res_B['macro_f1'] > res_A['macro_f1'] else 'Exp A'}")
    print(f"{'Weighted F1 Score':<30} | {res_A['weighted_f1']:.4f}{'':<19} | {res_B['weighted_f1']:.4f}{'':<19} | {'Exp B' if res_B['weighted_f1'] > res_A['weighted_f1'] else 'Exp A'}")
    print(f"{'Best Validation Accuracy':<30} | {res_A['best_val_accuracy']*100:.2f}%{'':<18} | {res_B['best_val_accuracy']*100:.2f}%{'':<18} | {'Exp B' if res_B['best_val_accuracy'] > res_A['best_val_accuracy'] else 'Exp A'}")
    print(f"{'Train-Validation Gap':<30} | {res_A['train_val_gap']*100:.2f}%{'':<18} | {res_B['train_val_gap']*100:.2f}%{'':<18} | {'Exp B' if res_B['train_val_gap'] < res_A['train_val_gap'] else 'Exp A'}")
    print(f"{'20-Video Unseen Accuracy':<30} | {unseen_acc_A:.2f}%{'':<18} | {unseen_acc_B:.2f}%{'':<18} | {'Exp B' if unseen_acc_B > unseen_acc_A else 'Exp A'}")
    print("="*100)
    print(f"\nWINNING EXPERIMENT: {winning_exp}")
    print(f"Total Experiment Time: {total_time:.2f} seconds")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_controlled_velocity_experiments()
