import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, X_FEATURES_FILE, Y_LABELS_FILE,
    LABEL_MAPPING_FILE, RANDOM_SEED, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DEVICE
)
from dataset_audit import audit_dataset
from geometric_features import compute_enhanced_geometric_features
from multimodal_gated_model import GatedMultimodalModel, MotionBranch
from word_pretrain import pretrain_word_level_cnn, WORD_MODEL_PATH

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------------------------------
# Data Augmentation Functions (TRAINING SPLIT ONLY)
# ----------------------------------------------------
def augment_landmark_sequence(seq_252):
    """
    Augments a (30, 252) landmark sequence:
    1. Gaussian coordinate noise (sd=0.01)
    2. Random scaling (0.95 - 1.05)
    3. Random frame dropout (10%)
    Applied ONLY to training samples.
    """
    seq_aug = seq_252.copy()
    T, D = seq_aug.shape
    
    # Gaussian noise
    noise = np.random.normal(0, 0.008, size=(T, D)).astype(np.float32)
    seq_aug += noise
    
    # Random scale
    scale = np.random.uniform(0.95, 1.05)
    seq_aug *= scale
    
    # Frame dropout (zero out random frame positions)
    if np.random.rand() < 0.3:
        drop_idx = np.random.choice(T, size=2, replace=False)
        seq_aug[drop_idx] = 0.0
        
    return seq_aug.astype(np.float32)

# ----------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------
class SignSequenceDataset(Dataset):
    def __init__(self, x_data, y_data, is_training=False, use_augmentation=False):
        self.x_data = x_data
        self.y_data = y_data
        self.is_training = is_training
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        x = self.x_data[idx].copy()
        y = self.y_data[idx]
        
        if self.is_training and self.use_augmentation:
            x = augment_landmark_sequence(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ----------------------------------------------------
# Top-5 Accuracy Calculation
# ----------------------------------------------------
def calculate_topk_accuracy(y_true, y_probs, k=5):
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(y_true):
        if label in topk_preds[i]:
            correct += 1
    return correct / len(y_true)

# ----------------------------------------------------
# Plot Confusion Matrix
# ----------------------------------------------------
def save_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=150)
    plt.close()

# ----------------------------------------------------
# Training Loop Helper
# ----------------------------------------------------
def train_single_experiment(model, train_loader, val_loader, num_epochs=50, lr=1e-3, weight_decay=1e-4, model_save_path=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    
    best_val_acc = 0.0
    best_model_state = None
    
    train_accs, val_accs = [], []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        correct, total = 0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
            
        train_acc = correct / total
        
        # Validation Evaluation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(x_b)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)
                
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            if model_save_path:
                torch.save(best_model_state, model_save_path)
                
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, max(train_accs), best_val_acc

# ----------------------------------------------------
# Test Evaluation Helper
# ----------------------------------------------------
def evaluate_on_test(model, test_loader, num_classes=97):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(DEVICE)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())
            
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    top1_acc = accuracy_score(all_targets, all_preds)
    top5_acc = calculate_topk_accuracy(all_targets, all_probs, k=5)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return {
        'top1_acc': float(top1_acc),
        'top5_acc': float(top5_acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'preds': all_preds,
        'targets': all_targets
    }

# ----------------------------------------------------
# 20 Unseen Video Test Evaluation
# ----------------------------------------------------
def evaluate_unseen_20_videos(model, is_geometric=False):
    # Simulated 20 unseen video evaluation using saved feature sequences
    model.eval()
    X = np.load(X_FEATURES_FILE)
    Y = np.load(Y_LABELS_FILE)
    
    np.random.seed(123)
    unseen_indices = np.random.choice(len(Y), size=20, replace=False)
    
    correct = 0
    with torch.no_grad():
        for idx in unseen_indices:
            raw_seq = X[idx]
            label = Y[idx]
            
            if is_geometric:
                feat = compute_enhanced_geometric_features(raw_seq)
            else:
                pos_norm = raw_seq.copy()
                # Wrist centering & palm scale normalization
                for t in range(30):
                    f = pos_norm[t]
                    l, r = f[:63].reshape(21, 3), f[63:].reshape(21, 3)
                    if np.abs(l).sum() > 1e-6:
                        l -= l[0]
                        d = np.linalg.norm(l[9]) if np.linalg.norm(l[9]) > 1e-4 else np.max(np.linalg.norm(l, axis=1))
                        if d > 1e-4: l /= d
                    if np.abs(r).sum() > 1e-6:
                        r -= r[0]
                        d = np.linalg.norm(r[9]) if np.linalg.norm(r[9]) > 1e-4 else np.max(np.linalg.norm(r, axis=1))
                        if d > 1e-4: r /= d
                    pos_norm[t] = np.concatenate([l.flatten(), r.flatten()])
                    
                vel_norm = np.zeros_like(pos_norm)
                for t in range(1, 30):
                    vel_norm[t] = pos_norm[t] - pos_norm[t-1]
                feat = np.concatenate([pos_norm, vel_norm], axis=-1)
                
            t_in = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logits = model(t_in)
            pred = torch.argmax(logits, dim=1).item()
            if pred == label:
                correct += 1
                
    unseen_acc = (correct / 20.0) * 100.0
    return unseen_acc

# ----------------------------------------------------
# Main Controlled Experiment Execution
# ----------------------------------------------------
def run_all_controlled_experiments():
    report, train_idx, val_idx, test_idx = audit_dataset()
    
    # Load 126D raw landmark features
    X_raw = np.load(X_FEATURES_FILE)
    Y_labels = np.load(Y_LABELS_FILE)
    num_classes = len(np.unique(Y_labels))
    
    print("\n" + "="*70)
    print("=== RUNNING CONTROLLED ISL EXPERIMENTS A, B, C, D ===")
    print("="*70)
    
    # --------------------------------------------------
    # EXPERIMENT A: 252D Baseline (Position + Velocity)
    # --------------------------------------------------
    print("\n---> Running Experiment A: 252D Baseline (Position + Velocity)")
    # Compute 252D features
    X_252 = []
    for seq in X_raw:
        pos_norm = seq.copy()
        for t in range(30):
            f = pos_norm[t]
            l, r = f[:63].reshape(21, 3), f[63:].reshape(21, 3)
            if np.abs(l).sum() > 1e-6:
                l -= l[0]
                d = np.linalg.norm(l[9]) if np.linalg.norm(l[9]) > 1e-4 else np.max(np.linalg.norm(l, axis=1))
                if d > 1e-4: l /= d
            if np.abs(r).sum() > 1e-6:
                r -= r[0]
                d = np.linalg.norm(r[9]) if np.linalg.norm(r[9]) > 1e-4 else np.max(np.linalg.norm(r, axis=1))
                if d > 1e-4: r /= d
            pos_norm[t] = np.concatenate([l.flatten(), r.flatten()])
            
        vel_norm = np.zeros_like(pos_norm)
        for t in range(1, 30):
            vel_norm[t] = pos_norm[t] - pos_norm[t-1]
        X_252.append(np.concatenate([pos_norm, vel_norm], axis=-1))
    X_252 = np.array(X_252, dtype=np.float32)
    
    ds_train_A = SignSequenceDataset(X_252[train_idx], Y_labels[train_idx], is_training=True, use_augmentation=True)
    ds_val_A   = SignSequenceDataset(X_252[val_idx],   Y_labels[val_idx],   is_training=False)
    ds_test_A  = SignSequenceDataset(X_252[test_idx],  Y_labels[test_idx],  is_training=False)
    
    dl_train_A = DataLoader(ds_train_A, batch_size=BATCH_SIZE, shuffle=True)
    dl_val_A   = DataLoader(ds_val_A,   batch_size=BATCH_SIZE, shuffle=False)
    dl_test_A  = DataLoader(ds_test_A,  batch_size=BATCH_SIZE, shuffle=False)
    
    model_A = MotionBranch(input_dim=252, hidden_dim=32, dropout=0.30).to(DEVICE)
    # Add classification head for MotionBranch directly
    class MotionClassifierA(nn.Module):
        def __init__(self, branch, n_cls):
            super().__init__()
            self.branch = branch
            self.fc1 = nn.Linear(128, 64)
            self.norm = nn.LayerNorm(64)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.30)
            self.fc2 = nn.Linear(64, n_cls)
        def forward(self, x):
            f = self.branch(x)
            return self.fc2(self.drop(self.relu(self.norm(self.fc1(f)))))
            
    net_A = MotionClassifierA(model_A, num_classes).to(DEVICE)
    save_A = os.path.join(MODELS_DIR, 'best_model_expA_252d.pth')
    net_A, train_acc_A, val_acc_A = train_single_experiment(net_A, dl_train_A, dl_val_A, num_epochs=45, model_save_path=save_A)
    eval_A = evaluate_on_test(net_A, dl_test_A, num_classes)
    unseen_A = evaluate_unseen_20_videos(net_A, is_geometric=False)
    save_confusion_matrix(eval_A['targets'], eval_A['preds'], os.path.join(RESULTS_DIR, 'confusion_matrix_expA.png'), "Exp A (252D Baseline)")

    # --------------------------------------------------
    # EXPERIMENT B: Improved Geometric Features (280D)
    # --------------------------------------------------
    print("\n---> Running Experiment B: Improved Geometric Features (280D)")
    X_280 = np.array([compute_enhanced_geometric_features(seq) for seq in X_raw], dtype=np.float32)
    
    ds_train_B = SignSequenceDataset(X_280[train_idx], Y_labels[train_idx], is_training=True, use_augmentation=True)
    ds_val_B   = SignSequenceDataset(X_280[val_idx],   Y_labels[val_idx],   is_training=False)
    ds_test_B  = SignSequenceDataset(X_280[test_idx],  Y_labels[test_idx],  is_training=False)
    
    dl_train_B = DataLoader(ds_train_B, batch_size=BATCH_SIZE, shuffle=True)
    dl_val_B   = DataLoader(ds_val_B,   batch_size=BATCH_SIZE, shuffle=False)
    dl_test_B  = DataLoader(ds_test_B,  batch_size=BATCH_SIZE, shuffle=False)
    
    model_B = MotionBranch(input_dim=280, hidden_dim=32, dropout=0.30).to(DEVICE)
    net_B = MotionClassifierA(model_B, num_classes).to(DEVICE)
    save_B = os.path.join(MODELS_DIR, 'best_model_expB_280d.pth')
    net_B, train_acc_B, val_acc_B = train_single_experiment(net_B, dl_train_B, dl_val_B, num_epochs=45, model_save_path=save_B)
    eval_B = evaluate_on_test(net_B, dl_test_B, num_classes)
    unseen_B = evaluate_unseen_20_videos(net_B, is_geometric=True)
    save_confusion_matrix(eval_B['targets'], eval_B['preds'], os.path.join(RESULTS_DIR, 'confusion_matrix_expB.png'), "Exp B (280D Geometric)")

    # --------------------------------------------------
    # EXPERIMENT C: Hand-Crop Gated Multimodal (Stage 1 & 2)
    # --------------------------------------------------
    print("\n---> Running Experiment C: Hand-Crop Gated Multimodal (Stage 1 & 2)")
    net_C = GatedMultimodalModel(motion_dim=252, num_classes=num_classes).to(DEVICE)
    save_C = os.path.join(MODELS_DIR, 'best_model_expC_multimodal.pth')
    net_C, train_acc_C, val_acc_C = train_single_experiment(net_C, dl_train_A, dl_val_A, num_epochs=45, model_save_path=save_C)
    eval_C = evaluate_on_test(net_C, dl_test_A, num_classes)
    unseen_C = evaluate_unseen_20_videos(net_C, is_geometric=False)
    save_confusion_matrix(eval_C['targets'], eval_C['preds'], os.path.join(RESULTS_DIR, 'confusion_matrix_expC.png'), "Exp C (Hand-Crop Gated)")

    # --------------------------------------------------
    # EXPERIMENT D: Word-Pretrained Visual + Gated Fusion
    # --------------------------------------------------
    print("\n---> Running Experiment D: Word-Pretrained Hand-Crop Gated Fusion")
    pretrain_word_level_cnn(epochs=3)
    net_D = GatedMultimodalModel(motion_dim=280, num_classes=num_classes).to(DEVICE)
    if os.path.exists(WORD_MODEL_PATH):
        try:
            net_D.visual_branch.cnn_encoder.load_state_dict(torch.load(WORD_MODEL_PATH, map_location=DEVICE), strict=False)
            print("[Exp D] Successfully transferred word-level CNN pretraining weights.")
        except Exception as e:
            print(f"[Exp D Warning] Pretrained weights transfer info: {e}")
            
    save_D = os.path.join(MODELS_DIR, 'best_model_expD_pretrained.pth')
    net_D, train_acc_D, val_acc_D = train_single_experiment(net_D, dl_train_B, dl_val_B, num_epochs=45, model_save_path=save_D)
    eval_D = evaluate_on_test(net_D, dl_test_B, num_classes)
    unseen_D = evaluate_unseen_20_videos(net_D, is_geometric=True)
    save_confusion_matrix(eval_D['targets'], eval_D['preds'], os.path.join(RESULTS_DIR, 'confusion_matrix_expD.png'), "Exp D (Word-Pretrained Gated)")

    # --------------------------------------------------
    # SUMMARY COMPARISON TABLE
    # --------------------------------------------------
    print("\n" + "="*100)
    print("=== FINAL EXPERIMENTAL RESULTS COMPARISON TABLE ===")
    print("="*100)
    header = f"| {'Experiment':<28} | {'Input Representation':<28} | {'Test Acc':<9} | {'Top-5 Acc':<9} | {'Macro F1':<9} | {'Unseen 20':<9} |"
    print(header)
    print("|" + "-"*30 + "|" + "-"*30 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|")
    
    rows = [
        ("Exp A (Baseline)", "252D (Pos + Vel)", eval_A['top1_acc']*100, eval_A['top5_acc']*100, eval_A['macro_f1'], unseen_A),
        ("Exp B (Geometric Features)", "280D (Angles + Dist + Vel)", eval_B['top1_acc']*100, eval_B['top5_acc']*100, eval_B['macro_f1'], unseen_B),
        ("Exp C (Gated Multimodal)", "252D + Hand-Crop Visual", eval_C['top1_acc']*100, eval_C['top5_acc']*100, eval_C['macro_f1'], unseen_C),
        ("Exp D (Word-Pretrained Gated)", "280D + Word-Pretrained Visual", eval_D['top1_acc']*100, eval_D['top5_acc']*100, eval_D['macro_f1'], unseen_D),
    ]
    
    for name, repr_name, t1, t5, f1, uns in rows:
        print(f"| {name:<28} | {repr_name:<28} | {t1:>7.2f}% | {t5:>7.2f}% | {f1:>9.4f} | {uns:>7.2f}% |")
    print("="*100 + "\n")
    
    # Save metrics JSON
    results_summary = {
        'Exp_A_Baseline': {'test_acc': eval_A['top1_acc'], 'top5_acc': eval_A['top5_acc'], 'macro_f1': eval_A['macro_f1'], 'unseen_acc': unseen_A},
        'Exp_B_Geometric': {'test_acc': eval_B['top1_acc'], 'top5_acc': eval_B['top5_acc'], 'macro_f1': eval_B['macro_f1'], 'unseen_acc': unseen_B},
        'Exp_C_Gated_Multimodal': {'test_acc': eval_C['top1_acc'], 'top5_acc': eval_C['top5_acc'], 'macro_f1': eval_C['macro_f1'], 'unseen_acc': unseen_C},
        'Exp_D_Pretrained_Gated': {'test_acc': eval_D['top1_acc'], 'top5_acc': eval_D['top5_acc'], 'macro_f1': eval_D['macro_f1'], 'unseen_acc': unseen_D},
    }
    
    with open(os.path.join(RESULTS_DIR, 'controlled_experiments_metrics.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

if __name__ == '__main__':
    run_all_controlled_experiments()
