import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    X_FEATURES_FILE, Y_LABELS_FILE, LABEL_MAPPING_FILE, FAILED_VIDEOS_FILE, VIDEOS_DIR,
    MODELS_DIR, DATA_DIR, BASE_DIR, MODEL_PATH, BEST_MODEL_PATH, EVAL_METRICS_PATH,
    CONFUSION_MATRIX_PATH, TRAINING_HISTORY_PATH,
    FEATURE_DIM, SEQUENCE_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, HIDDEN_DIM, NUM_BILSTM_LAYERS, NUM_GRU_LAYERS, DROPOUT, RANDOM_SEED, DEVICE, normalize_sequence_features
)
from model import BiLSTM_GRU_Model

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TRAINING_HISTORY_JSON_PATH = os.path.join(MODELS_DIR, 'training_history.json')
RESULTS_CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
RESULTS_REPORT_PATH = os.path.join(RESULTS_DIR, 'classification_report.txt')
RESULTS_CURVES_PATH = os.path.join(RESULTS_DIR, 'training_curves.png')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_valid_sample_metadata():
    """
    Map each extracted sample to its exact video path and file name matching label_mapping.json.
    """
    CSV_PATH = os.path.join(DATA_DIR, 'dataset_video_status.csv')
    if not os.path.exists(CSV_PATH):
        return []

    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    active_97_classes = set(label_mapping.values())

    valid_samples = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'processed' and row['class_name'] in active_97_classes:
                valid_samples.append({'class': row['class_name'], 'video_path': row['video_path'], 'video_name': row['video_name']})
    return valid_samples

def safe_dataset_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    print("\n" + "="*60)
    print("=== DATA QUALITY & STRATIFIED SPLIT ANALYSIS ===")
    print("="*60)
    counts = Counter(y)
    unique_classes = sorted(list(counts.keys()))
    num_classes = len(unique_classes)
    
    print(f"Total extracted samples: {len(y)}")
    print(f"Total active classes present: {num_classes}")

    min_samples = min(counts.values()) if len(counts) > 0 else 0
    print(f"Minimum samples per class: {min_samples}")

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

    # Verify zero overlap between split sets
    s_train, s_val, s_test = set(train_indices), set(val_indices), set(test_indices)
    assert len(s_train.intersection(s_val)) == 0, "Data leakage detected between train and validation!"
    assert len(s_train.intersection(s_test)) == 0, "Data leakage detected between train and test!"
    assert len(s_val.intersection(s_test)) == 0, "Data leakage detected between validation and test!"

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("\nSplit verification complete (No leakage detected):")
    print(f"  Training samples  : {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test samples      : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    return (X_train, y_train, train_indices), (X_val, y_val, val_indices), (X_test, y_test, test_indices)

def augment_landmark_sequence(seq):
    """
    Stronger training-only sequence augmentation pipeline combining 8 realistic variants:
    1. Small landmark Gaussian noise
    2. Wrist-relative translation jitter
    3. Scale jitter
    4. Small 2D rotation jitter around wrist/palm
    5. Temporal speed warping (stretching/compression)
    6. Random temporal shift
    7. Random frame dropout with linear temporal interpolation
    8. Small feature masking (keypoint/hand dropout)
    """
    if isinstance(seq, np.ndarray):
        seq = torch.tensor(seq, dtype=torch.float32)
    
    T, D = seq.shape
    seq_aug = seq.clone()

    # 1. Random Temporal Shift
    if np.random.rand() < 0.5:
        shift = np.random.randint(-4, 5)
        if shift != 0:
            seq_aug = torch.roll(seq_aug, shifts=shift, dims=0)

    # 2. Temporal Speed Warping (Stretching / Compression)
    if np.random.rand() < 0.6:
        warp_factor = np.random.uniform(0.80, 1.25)
        orig_t = np.linspace(0, T - 1, T)
        warped_t = np.clip(orig_t * warp_factor, 0, T - 1)
        seq_np = seq_aug.numpy()
        warped_seq = np.zeros_like(seq_np)
        for d in range(D):
            warped_seq[:, d] = np.interp(orig_t, warped_t, seq_np[:, d])
        seq_aug = torch.tensor(warped_seq, dtype=torch.float32)

    # 3. Random Frame Dropout followed by Linear Interpolation
    if np.random.rand() < 0.5:
        drop_rate = np.random.uniform(0.10, 0.20)
        num_drop = max(1, int(T * drop_rate))
        drop_indices = set(np.random.choice(T, size=num_drop, replace=False))
        valid_indices = [i for i in range(T) if i not in drop_indices]
        if len(valid_indices) >= 5:
            seq_np = seq_aug.numpy()
            interpolated_seq = np.zeros_like(seq_np)
            grid_all = np.arange(T)
            for d in range(D):
                interpolated_seq[:, d] = np.interp(grid_all, valid_indices, seq_np[valid_indices, d])
            seq_aug = torch.tensor(interpolated_seq, dtype=torch.float32)

    # 4. Spatial Augmentation (Noise, Translation, Scale, Rotation)
    left_hand = seq_aug[:, :63].view(T, 21, 3)
    right_hand = seq_aug[:, 63:].view(T, 21, 3)

    scale = np.random.uniform(0.88, 1.12)
    translation = torch.tensor(np.random.uniform(-0.04, 0.04, size=(1, 1, 3)), dtype=torch.float32)

    angle_deg = np.random.uniform(-15.0, 15.0)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_mat = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)

    for h in [left_hand, right_hand]:
        mask = (torch.abs(h).sum(dim=-1, keepdim=True) > 1e-6).float()
        if mask.sum() > 0:
            # 4a. Rotation around wrist landmark 0
            wrist_pos = h[:, 0:1, :2].clone()
            xy = h[:, :, :2] - wrist_pos
            xy_rot = torch.matmul(xy, rot_mat.T)
            h[:, :, :2] = (xy_rot + wrist_pos)

            # 4b. Scale & Translation
            h_aug = h * scale + translation
            
            # 4c. Small Landmark Gaussian Noise
            noise = torch.randn_like(h_aug) * np.random.uniform(0.005, 0.012)
            h_aug = h_aug + noise
            
            h.copy_(h_aug * mask)

    # 5. Small Feature Masking (Hand / Keypoint Dropout)
    if np.random.rand() < 0.3:
        kp_drop = np.random.choice(21, size=np.random.randint(1, 3), replace=False)
        for kp in kp_drop:
            if np.random.rand() < 0.5:
                left_hand[:, kp, :] = 0.0
            else:
                right_hand[:, kp, :] = 0.0

    seq_aug = torch.cat([left_hand.view(T, 63), right_hand.view(T, 63)], dim=1)
    return seq_aug

def expand_training_dataset(X_train, y_train, target_samples_per_class=25):
    """
    Expands training set to target_samples_per_class for every active class.
    Logs augmented sequence counts created per class.
    Validation and test sets MUST NOT be passed to this function.
    """
    print("\n" + "="*60)
    print("=== EXPANDING TRAINING DATASET (TARGET = 25 SAMPLES / CLASS) ===")
    print("="*60)
    
    unique_classes = np.unique(y_train)
    X_expanded = []
    y_expanded = []
    
    aug_counts_per_class = {}
    
    for c in sorted(unique_classes):
        cls_indices = np.where(y_train == c)[0]
        cls_samples = X_train[cls_indices]
        n_orig = len(cls_samples)
        
        # Add original training sequences
        for s in cls_samples:
            X_expanded.append(s)
            y_expanded.append(c)
            
        n_needed = max(0, target_samples_per_class - n_orig)
        n_created = 0
        
        if n_needed > 0:
            for i in range(n_needed):
                base_seq = cls_samples[i % n_orig]
                aug_seq = augment_landmark_sequence(torch.tensor(base_seq, dtype=torch.float32)).numpy()
                X_expanded.append(aug_seq)
                y_expanded.append(c)
                n_created += 1
                
        aug_counts_per_class[int(c)] = n_created
        
    X_expanded = np.array(X_expanded, dtype=np.float32)
    y_expanded = np.array(y_expanded, dtype=np.int64)
    
    print(f"Total training sequences after expansion: {len(X_expanded)} (Active classes: {len(unique_classes)})")
    print(f"Target samples per class: {target_samples_per_class}")
    print("Augmented sequences created per class log (first 10 classes sample):")
    for c in sorted(list(aug_counts_per_class.keys()))[:10]:
        n_orig = (y_train == c).sum()
        print(f"  Class [{c:03d}]: {n_orig} original + {aug_counts_per_class[c]} augmented = {n_orig + aug_counts_per_class[c]} samples")
    print("="*60 + "\n")
    
    return X_expanded, y_expanded, aug_counts_per_class

class SignDataset(Dataset):
    def __init__(self, x_data, y_data, is_train=True):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
        self.is_train = is_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        seq = self.x[idx].clone()
        label = self.y[idx]
        if self.is_train:
            seq = augment_landmark_sequence(seq)
        return seq, label

def compute_top_k_accuracy(outputs, targets, k=5):
    with torch.no_grad():
        topk_preds = torch.topk(outputs, k=min(k, outputs.shape[1]), dim=1).indices
        targets_expanded = targets.view(-1, 1).expand_as(topk_preds)
        correct = (topk_preds == targets_expanded).any(dim=1).float()
        return correct.mean().item()

def run_training():
    set_seed(RANDOM_SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(X_FEATURES_FILE) or not os.path.exists(Y_LABELS_FILE):
        raise FileNotFoundError(f"Feature files {X_FEATURES_FILE} and {Y_LABELS_FILE} not found.")

    X_raw = np.load(X_FEATURES_FILE)
    y = np.load(Y_LABELS_FILE)
    valid_samples = get_valid_sample_metadata()

    if len(X_raw) == 0 or len(y) == 0:
        raise ValueError("Loaded dataset is empty. Cannot train model.")

    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    print(f"Loaded raw dataset: X shape = {X_raw.shape}, y shape = {y.shape}, classes = {num_classes}")

    # Relative wrist & palm scale landmark normalization
    X = normalize_sequence_features(X_raw)
    print("Wrist centering & palm scale normalization applied to all sequences.")

    # Stratified split FIRST (70% train, 15% val, 15% test)
    (X_train_orig, y_train_orig, train_idxs), (X_val, y_val, val_idxs), (X_test, y_test, test_idxs) = safe_dataset_split(X, y)

    # Expand training dataset ONLY (target = 25 sequences per active class)
    X_train, y_train, aug_counts = expand_training_dataset(X_train_orig, y_train_orig, target_samples_per_class=25)

    # Compute class weights for loss
    train_counts = Counter(y_train)
    max_count = max(train_counts.values()) if len(train_counts) > 0 else 1
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        cnt = train_counts.get(c, 0)
        w = np.sqrt(max_count / (cnt + 1.0))
        class_weights[c] = float(w)
    class_weights = class_weights / class_weights.mean()

    train_ds = SignDataset(X_train, y_train, is_train=True)
    val_ds = SignDataset(X_val, y_val, is_train=False) if len(X_val) > 0 else train_ds
    test_ds = SignDataset(X_test, y_test, is_train=False) if len(X_test) > 0 else val_ds

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTM_GRU_Model(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_bilstm_layers=NUM_BILSTM_LAYERS,
        num_gru_layers=NUM_GRU_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    early_stopping_patience = 8
    no_improve_epochs = 0

    print(f"\nStarting Augmented Training (hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, batch_size={BATCH_SIZE}, train_size={len(X_train)}) for up to {NUM_EPOCHS} epochs on device: {DEVICE}...")

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

        # Save best model based on validation loss or validation macro F1
        if epoch_val_loss < best_val_loss or epoch_val_f1 > best_val_f1:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
            if epoch_val_f1 > best_val_f1:
                best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improve_epochs = 0
            best_str = " (Best model saved)"
        else:
            no_improve_epochs += 1
            best_str = ""

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s{best_str}")

        if no_improve_epochs >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (Validation metrics unimproved for {early_stopping_patience} epochs).")
            break

    # Save final model
    torch.save(model.state_dict(), MODEL_PATH)

    # Save training history JSON
    with open(TRAINING_HISTORY_JSON_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    # Load best model for evaluation
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
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

    report_str = classification_report(all_targets, all_preds, zero_division=0)

    per_class_correct = Counter()
    per_class_total = Counter(all_targets)
    for target, pred in zip(all_targets, all_preds):
        if target == pred:
            per_class_correct[target] += 1

    recognized_classes_count = len(per_class_correct)
    worst_classes_info = []
    for cls in set(all_targets):
        c_tot = per_class_total[cls]
        c_cor = per_class_correct[cls]
        c_acc = c_cor / c_tot if c_tot > 0 else 0.0
        worst_classes_info.append((cls, idx_to_label.get(cls, str(cls)), c_acc, c_cor, c_tot))
    
    worst_classes_info.sort(key=lambda item: (item[2], item[4]))

    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc_val = max(history['val_acc'])
    gap = final_train_acc - final_val_acc

    print("\n" + "="*60)
    print("=== FINAL AUGMENTED MODEL TEST EVALUATION ===")
    print("="*60)
    print(f"Train Accuracy            : {final_train_acc * 100:.2f}%")
    print(f"Validation Accuracy (Best): {best_val_acc_val * 100:.2f}% (Final: {final_val_acc * 100:.2f}%)")
    print(f"Test Accuracy (Top-1)     : {test_acc * 100:.2f}%")
    print(f"Top-5 Accuracy            : {top5_acc * 100:.2f}%")
    print(f"Macro F1 Score            : {f1_macro:.4f}")
    print(f"Weighted F1 Score         : {f1_weighted:.4f}")
    print(f"Train-Validation Gap      : {gap * 100:.2f}%")
    print(f"Classes Correctly Recognized: {recognized_classes_count} / {len(set(all_targets))} active test classes")
    print("="*60 + "\n")

    # Save Classification Report
    with open(os.path.join(MODELS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report_str)
    with open(RESULTS_REPORT_PATH, 'w') as f:
        f.write(report_str)

    # Copy label_mapping.json into models/
    shutil.copy(LABEL_MAPPING_FILE, os.path.join(MODELS_DIR, 'label_mapping.json'))

    # Save metrics JSON
    metrics = {
        'total_extracted_samples': len(X),
        'number_of_classes': num_classes,
        'feature_shape': list(X.shape),
        'train_samples_expanded': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_accuracy': float(final_train_acc),
        'val_accuracy': float(final_val_acc),
        'best_val_accuracy': float(best_val_acc_val),
        'test_accuracy': float(test_acc),
        'top5_accuracy': float(top5_acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'train_val_gap': float(gap),
        'recognized_classes_count': int(recognized_classes_count),
        'model_path': MODEL_PATH,
        'best_model_path': BEST_MODEL_PATH
    }

    with open(EVAL_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves (Strong Sequence Augmentation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH)
    plt.savefig(RESULTS_CURVES_PATH)
    plt.close()

    # Plot Confusion Matrix
    if len(all_targets) > 0:
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.savefig(RESULTS_CONFUSION_MATRIX_PATH)
        plt.close()

    # =========================================================================
    # INFERENCE TEST ON AT LEAST 10 COMPLETELY UNSEEN TEST VIDEOS
    # =========================================================================
    print("\n" + "="*60)
    print("=== AUTOMATIC TEST SET INFERENCE VERIFICATION (10 UNSEEN SAMPLES) ===")
    print("="*60)

    np.random.seed(RANDOM_SEED)
    sample_picks = np.random.choice(test_idxs, size=min(10, len(test_idxs)), replace=False)

    num_correct = 0
    for i, idx in enumerate(sample_picks, start=1):
        sample_meta = valid_samples[idx] if idx < len(valid_samples) else {'class': idx_to_label[y[idx]], 'video_name': f'test_sample_{idx}'}
        expected_sentence = sample_meta['class']
        video_name = sample_meta['video_name']
        
        sample_x = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(sample_x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        predicted_sentence = idx_to_label.get(pred_idx.item(), f"Class_{pred_idx.item()}")
        confidence = conf.item() * 100.0
        is_correct = (expected_sentence.lower() == predicted_sentence.lower())
        if is_correct:
            num_correct += 1

        print(f"Test Video #{i:02d} [{video_name}]:")
        print(f"  Expected sentence : {expected_sentence}")
        print(f"  Predicted sentence: {predicted_sentence}")
        print(f"  Confidence        : {confidence:.2f}%")
        print(f"  Status            : {'MATCH [CORRECT]' if is_correct else 'MISMATCH [INCORRECT]'}")
        print("-" * 50)

    print(f"Unseen Test Summary: {num_correct} / {len(sample_picks)} test videos correctly predicted ({num_correct/len(sample_picks)*100:.1f}% accuracy)")
    print("="*60 + "\n")

if __name__ == '__main__':
    run_training()
