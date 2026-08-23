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
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, RANDOM_SEED, BATCH_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, DEVICE
)
from multimodal_gated_model import GatedMultimodalModel, MotionBranch

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

X_252_FILE = os.path.join(DATA_DIR, 'x_features.npy')
X_CROP_FILE = os.path.join(DATA_DIR, 'x_hand_crop_features_64.npy')
Y_FILE = os.path.join(DATA_DIR, 'y_labels.npy')

# ----------------------------------------------------
# Training Data Augmentation (TRAINING SPLIT ONLY)
# ----------------------------------------------------
def augment_sequence(seq):
    seq_aug = seq.copy()
    T, D = seq_aug.shape
    noise = np.random.normal(0, 0.008, size=(T, D)).astype(np.float32)
    seq_aug += noise
    scale = np.random.uniform(0.95, 1.05)
    seq_aug *= scale
    if np.random.rand() < 0.3:
        drop_idx = np.random.choice(T, size=2, replace=False)
        seq_aug[drop_idx] = 0.0
    return seq_aug.astype(np.float32)

class HandCropExperimentDataset(Dataset):
    def __init__(self, x_motion, x_visual, y, is_training=False):
        self.x_motion = x_motion
        self.x_visual = x_visual
        self.y = y
        self.is_training = is_training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        m = self.x_motion[idx].copy()
        v = self.x_visual[idx].copy()
        target = self.y[idx]

        if self.is_training:
            m = augment_sequence(m)
            v = augment_sequence(v)

        return (
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(v, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long)
        )

def calculate_topk_accuracy(y_true, y_probs, k=5):
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(y_true):
        if label in topk_preds[i]:
            correct += 1
    return correct / len(y_true)

# ----------------------------------------------------
# Model Architecture Definitions
# ----------------------------------------------------
class MotionOnlyClassifier(nn.Module):
    def __init__(self, input_dim=252, hidden_dim=32, num_classes=97, dropout=0.3):
        super().__init__()
        self.branch = MotionBranch(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.fc1 = nn.Linear(128, 64)
        self.norm = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, m, v=None):
        f = self.branch(m)
        return self.fc2(self.drop(self.relu(self.norm(self.fc1(f)))))

class VisualCropOnlyClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, num_classes=97, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, 32)
        self.norm = nn.LayerNorm(32)
        self.bilstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, m, v):
        h = self.norm(self.proj(v))
        out, _ = self.bilstm(h)
        pool = out.mean(dim=1)
        return self.fc(self.drop(pool))

class GatedMotionVisualClassifier(nn.Module):
    def __init__(self, motion_dim=252, visual_dim=64, hidden_dim=32, num_classes=97, dropout=0.3):
        super().__init__()
        self.motion_branch = MotionBranch(input_dim=motion_dim, hidden_dim=hidden_dim, dropout=dropout)
        
        self.visual_proj = nn.Linear(visual_dim, 32)
        self.visual_norm = nn.LayerNorm(32)
        self.visual_lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        feature_dim = 128
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Sigmoid()
        )
        
        self.fc1 = nn.Linear(feature_dim, 64)
        self.norm = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, m, v):
        f_m = self.motion_branch(m) # (B, 128)
        
        v_h = self.visual_norm(self.visual_proj(v))
        v_out, _ = self.visual_lstm(v_h)
        f_v = v_out.mean(dim=1) # (B, 64) -> project to 128D
        f_v = torch.cat([f_v, f_v], dim=1) # (B, 128)
        
        alpha = self.gate(torch.cat([f_m, f_v], dim=1))
        f_fused = alpha * f_m + (1.0 - alpha) * f_v
        
        return self.fc2(self.drop(self.relu(self.norm(self.fc1(f_fused)))))

# ----------------------------------------------------
# Training & Evaluation Loop
# ----------------------------------------------------
def train_experiment(model, train_loader, val_loader, num_epochs=45, lr=1e-3, save_path=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    last_train_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        correct, total = 0, 0
        for m_b, v_b, y_b in train_loader:
            m_b, v_b, y_b = m_b.to(DEVICE), v_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(m_b, v_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        train_acc = correct / total
        last_train_acc = train_acc

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for m_b, v_b, y_b in val_loader:
                m_b, v_b, y_b = m_b.to(DEVICE), v_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(m_b, v_b)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_state, save_path)

    if best_state:
        model.load_state_dict(best_state)

    train_val_gap = float(last_train_acc - best_val_acc)
    return model, best_val_acc, train_val_gap

def evaluate_experiment(model, test_loader):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    t_start = time.time()
    with torch.no_grad():
        for m_b, v_b, y_b in test_loader:
            m_b, v_b = m_b.to(DEVICE), v_b.to(DEVICE)
            logits = model(m_b, v_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())

    total_time_ms = (time.time() - t_start) * 1000.0
    avg_latency_ms = total_time_ms / len(test_loader.dataset)

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    top1_acc = accuracy_score(all_targets, all_preds)
    top5_acc = calculate_topk_accuracy(all_targets, all_probs, k=5)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return {
        'top1_acc': float(top1_acc),
        'top5_acc': float(top5_acc),
        'macro_f1': float(macro_f1),
        'latency_ms': float(avg_latency_ms)
    }

def run_all_hand_crop_experiments():
    print("\n" + "="*70)
    print("=== RUNNING CONTROLLED HAND-CROP VISUAL FEATURE EXPERIMENTS ===")
    print("="*70)

    X_raw_motion = np.load(X_252_FILE)
    X_visual = np.load(X_CROP_FILE)
    Y_labels = np.load(Y_FILE)

    # Compute 252D Position + Velocity from 126D raw features if needed
    if X_raw_motion.shape[-1] == 126:
        X_motion = []
        for seq in X_raw_motion:
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
            X_motion.append(np.concatenate([pos_norm, vel_norm], axis=-1))
        X_motion = np.array(X_motion, dtype=np.float32)
    else:
        X_motion = X_raw_motion

    num_classes = len(np.unique(Y_labels))

    # Stratified Split (Identical RANDOM_SEED = 42)
    np.random.seed(RANDOM_SEED)
    class_indices = {}
    for idx, label in enumerate(Y_labels):
        class_indices.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in class_indices.items():
        np.random.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(round(n * TRAIN_SPLIT)))
        n_val = max(1, int(round(n * VAL_SPLIT)))
        if n_train + n_val >= n and n > 2:
            n_train = n - 2
            n_val = 1
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    ds_train = HandCropExperimentDataset(X_motion[train_idx], X_visual[train_idx], Y_labels[train_idx], is_training=True)
    ds_val   = HandCropExperimentDataset(X_motion[val_idx],   X_visual[val_idx],   Y_labels[val_idx],   is_training=False)
    ds_test  = HandCropExperimentDataset(X_motion[test_idx],  X_visual[test_idx],  Y_labels[test_idx],  is_training=False)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------
    # EXPERIMENT A: 252D Motion Baseline
    # ----------------------------------------------------
    print("\n---> Running Experiment A: 252D Motion Baseline")
    model_A = MotionOnlyClassifier(input_dim=252, num_classes=num_classes).to(DEVICE)
    save_A = os.path.join(MODELS_DIR, 'best_model_expA_motion_252d.pth')
    model_A, val_acc_A, gap_A = train_experiment(model_A, dl_train, dl_val, num_epochs=45, save_path=save_A)
    eval_A = evaluate_experiment(model_A, dl_test)
    eval_A['train_val_gap'] = gap_A

    # ----------------------------------------------------
    # EXPERIMENT B: Hand Crop Visual Features Only (64D)
    # ----------------------------------------------------
    print("\n---> Running Experiment B: Hand Crop Visual Features Only (64D)")
    model_B = VisualCropOnlyClassifier(input_dim=64, num_classes=num_classes).to(DEVICE)
    save_B = os.path.join(MODELS_DIR, 'best_model_expB_handcrop_visual_only.pth')
    model_B, val_acc_B, gap_B = train_experiment(model_B, dl_train, dl_val, num_epochs=45, save_path=save_B)
    eval_B = evaluate_experiment(model_B, dl_test)
    eval_B['train_val_gap'] = gap_B

    # ----------------------------------------------------
    # EXPERIMENT C: 252D Motion + Hand Crop Visual (Gated Fusion)
    # ----------------------------------------------------
    print("\n---> Running Experiment C: 252D Motion + Hand Crop Visual (Gated Fusion)")
    model_C = GatedMotionVisualClassifier(motion_dim=252, visual_dim=64, num_classes=num_classes).to(DEVICE)
    save_C = os.path.join(MODELS_DIR, 'best_model_expC_gated_multimodal_crops.pth')
    model_C, val_acc_C, gap_C = train_experiment(model_C, dl_train, dl_val, num_epochs=45, save_path=save_C)
    eval_C = evaluate_experiment(model_C, dl_test)
    eval_C['train_val_gap'] = gap_C

    # ----------------------------------------------------
    # COMPARISON TABLE & REPORT
    # ----------------------------------------------------
    print("\n" + "="*105)
    print("=== HAND-CROP VISUAL FEATURE EXPERIMENTS REPORT ===")
    print("="*105)
    header = f"| {'Experiment':<28} | {'Input Representation':<28} | {'Test Acc':<9} | {'Top-5 Acc':<9} | {'Macro F1':<9} | {'TV Gap':<8} | {'Latency':<8} |"
    print(header)
    print("|" + "-"*30 + "|" + "-"*30 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|")

    rows = [
        ("Exp A (Motion Baseline)", "252D (Pos + Vel)", eval_A['top1_acc']*100, eval_A['top5_acc']*100, eval_A['macro_f1'], eval_A['train_val_gap']*100, eval_A['latency_ms']),
        ("Exp B (Visual Crops Only)", "64D Hand-Crop Embedding", eval_B['top1_acc']*100, eval_B['top5_acc']*100, eval_B['macro_f1'], eval_B['train_val_gap']*100, eval_B['latency_ms']),
        ("Exp C (Gated Multimodal)", "252D Motion + 64D Visual", eval_C['top1_acc']*100, eval_C['top5_acc']*100, eval_C['macro_f1'], eval_C['train_val_gap']*100, eval_C['latency_ms']),
    ]

    for name, repr_name, t1, t5, f1, gap, lat in rows:
        print(f"| {name:<28} | {repr_name:<28} | {t1:>7.2f}% | {t5:>7.2f}% | {f1:>9.4f} | {gap:>6.2f}% | {lat:>6.2f}ms |")
    print("="*105 + "\n")

    best_idx = np.argmax([eval_A['top1_acc'], eval_B['top1_acc'], eval_C['top1_acc']])
    winner_names = ["Exp A (Motion Baseline)", "Exp B (Visual Crops Only)", "Exp C (Gated Multimodal)"]
    print(f"🏆 Winning Architecture: {winner_names[best_idx]}\n")

    results_summary = {
        'Exp_A_Motion_Baseline': eval_A,
        'Exp_B_Visual_Crops_Only': eval_B,
        'Exp_C_Gated_Multimodal': eval_C,
        'winning_architecture': winner_names[best_idx]
    }

    with open(os.path.join(RESULTS_DIR, 'hand_crop_experiments_metrics.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

if __name__ == '__main__':
    run_all_hand_crop_experiments()
