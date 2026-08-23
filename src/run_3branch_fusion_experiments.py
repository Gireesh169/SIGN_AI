import os
import sys
import json
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
from modality_gated_model import ModalityGatedFusionModel

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

HAND_FILE = os.path.join(DATA_DIR, 'x_hand_features_252.npy')
FACE_FILE = os.path.join(DATA_DIR, 'x_face_features_compact.npy')
POSE_FILE = os.path.join(DATA_DIR, 'x_pose_features_compact.npy')
Y_FILE = os.path.join(DATA_DIR, 'y_labels.npy')

# ----------------------------------------------------
# Data Augmentation (TRAINING SPLIT ONLY)
# ----------------------------------------------------
def augment_hand_sequence(seq):
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

class MultimodalDataset(Dataset):
    def __init__(self, x_hand, x_face, x_pose, y, is_training=False):
        self.x_hand = x_hand
        self.x_face = x_face
        self.x_pose = x_pose
        self.y = y
        self.is_training = is_training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        h = self.x_hand[idx].copy()
        f = self.x_face[idx].copy()
        p = self.x_pose[idx].copy()
        target = self.y[idx]

        if self.is_training:
            h = augment_hand_sequence(h)

        return (
            torch.tensor(h, dtype=torch.float32),
            torch.tensor(f, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long)
        )

def calculate_topk_accuracy(y_true, y_probs, k=5):
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(y_true):
        if label in topk_preds[i]:
            correct += 1
    return correct / len(y_true)

def train_fusion_experiment(model, train_loader, val_loader, num_epochs=45, lr=1e-3, weight_decay=1e-4, save_path=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        correct, total = 0, 0
        for h_b, f_b, p_b, y_b in train_loader:
            h_b, f_b, p_b, y_b = h_b.to(DEVICE), f_b.to(DEVICE), p_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(h_b, f_b, p_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for h_b, f_b, p_b, y_b in val_loader:
                h_b, f_b, p_b, y_b = h_b.to(DEVICE), f_b.to(DEVICE), p_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(h_b, f_b, p_b)
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

    return model, best_val_acc

def evaluate_fusion_experiment(model, test_loader):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for h_b, f_b, p_b, y_b in test_loader:
            h_b, f_b, p_b = h_b.to(DEVICE), f_b.to(DEVICE), p_b.to(DEVICE)
            logits = model(h_b, f_b, p_b)
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
        'weighted_f1': float(weighted_f1)
    }

def run_all_fusion_experiments():
    print("\n" + "="*70)
    print("=== STARTING 3-BRANCH MODALITY GATED FUSION EXPERIMENTS ===")
    print("="*70)

    # Load 3-branch features
    X_hand = np.load(HAND_FILE)
    X_face = np.load(FACE_FILE)
    X_pose = np.load(POSE_FILE)
    Y_labels = np.load(Y_FILE)

    num_samples = len(Y_labels)
    num_classes = len(np.unique(Y_labels))

    # Stratified Split (Exact same split seed)
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

    ds_train = MultimodalDataset(X_hand[train_idx], X_face[train_idx], X_pose[train_idx], Y_labels[train_idx], is_training=True)
    ds_val   = MultimodalDataset(X_hand[val_idx],   X_face[val_idx],   X_pose[val_idx],   Y_labels[val_idx],   is_training=False)
    ds_test  = MultimodalDataset(X_hand[test_idx],  X_face[test_idx],  X_pose[test_idx],  Y_labels[test_idx],  is_training=False)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------
    # EXPERIMENT A: Hand Only
    # ----------------------------------------------------
    print("\n---> Running Experiment A: Hand-Only Baseline (252D)")
    model_A = ModalityGatedFusionModel(use_face=False, use_pose=False, num_classes=num_classes).to(DEVICE)
    save_A = os.path.join(MODELS_DIR, 'best_model_multimodal_hand_only.pth')
    model_A, val_acc_A = train_fusion_experiment(model_A, dl_train, dl_val, num_epochs=45, save_path=save_A)
    eval_A = evaluate_fusion_experiment(model_A, dl_test)

    # ----------------------------------------------------
    # EXPERIMENT B: Hand + Face Gated
    # ----------------------------------------------------
    print("\n---> Running Experiment B: Hand + Face Gated (252D + 16D)")
    model_B = ModalityGatedFusionModel(use_face=True, use_pose=False, num_classes=num_classes).to(DEVICE)
    save_B = os.path.join(MODELS_DIR, 'best_model_multimodal_hand_face.pth')
    model_B, val_acc_B = train_fusion_experiment(model_B, dl_train, dl_val, num_epochs=45, save_path=save_B)
    eval_B = evaluate_fusion_experiment(model_B, dl_test)

    # ----------------------------------------------------
    # EXPERIMENT C: Hand + Pose Gated
    # ----------------------------------------------------
    print("\n---> Running Experiment C: Hand + Pose Gated (252D + 24D)")
    model_C = ModalityGatedFusionModel(use_face=False, use_pose=True, num_classes=num_classes).to(DEVICE)
    save_C = os.path.join(MODELS_DIR, 'best_model_multimodal_hand_pose.pth')
    model_C, val_acc_C = train_fusion_experiment(model_C, dl_train, dl_val, num_epochs=45, save_path=save_C)
    eval_C = evaluate_fusion_experiment(model_C, dl_test)

    # ----------------------------------------------------
    # EXPERIMENT D: Hand + Face + Pose Gated
    # ----------------------------------------------------
    print("\n---> Running Experiment D: Hand + Face + Pose Gated (252D + 16D + 24D)")
    model_D = ModalityGatedFusionModel(use_face=True, use_pose=True, num_classes=num_classes).to(DEVICE)
    save_D = os.path.join(MODELS_DIR, 'best_model_multimodal_hand_face_pose.pth')
    model_D, val_acc_D = train_fusion_experiment(model_D, dl_train, dl_val, num_epochs=45, save_path=save_D)
    eval_D = evaluate_fusion_experiment(model_D, dl_test)

    # ----------------------------------------------------
    # COMPARISON TABLE & REPORT
    # ----------------------------------------------------
    print("\n" + "="*105)
    print("=== 3-BRANCH MODALITY GATED FUSION EXPERIMENT RESULTS ===")
    print("="*105)
    header = f"| {'Model':<25} | {'Input Modalities':<28} | {'Test Acc':<9} | {'Top-5 Acc':<9} | {'Macro F1':<9} | {'Weighted F1':<11} |"
    print(header)
    print("|" + "-"*27 + "|" + "-"*30 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*13 + "|")

    rows = [
        ("Hand Only (Exp A)", "Hand (252D)", eval_A['top1_acc']*100, eval_A['top5_acc']*100, eval_A['macro_f1'], eval_A['weighted_f1']),
        ("Hand + Face (Exp B)", "Hand (252D) + Face (16D)", eval_B['top1_acc']*100, eval_B['top5_acc']*100, eval_B['macro_f1'], eval_B['weighted_f1']),
        ("Hand + Pose (Exp C)", "Hand (252D) + Pose (24D)", eval_C['top1_acc']*100, eval_C['top5_acc']*100, eval_C['macro_f1'], eval_C['weighted_f1']),
        ("Hand + Face + Pose (Exp D)", "Hand (252D) + Face + Pose", eval_D['top1_acc']*100, eval_D['top5_acc']*100, eval_D['macro_f1'], eval_D['weighted_f1']),
    ]

    for name, inputs, t1, t5, mf1, wf1 in rows:
        print(f"| {name:<25} | {inputs:<28} | {t1:>7.2f}% | {t5:>7.2f}% | {mf1:>9.4f} | {wf1:>11.4f} |")
    print("="*105 + "\n")

    best_exp_idx = np.argmax([eval_A['top1_acc'], eval_B['top1_acc'], eval_C['top1_acc'], eval_D['top1_acc']])
    winner_names = ["Hand Only (Exp A)", "Hand + Face (Exp B)", "Hand + Pose (Exp C)", "Hand + Face + Pose (Exp D)"]
    print(f"🏆 Winning Architecture: {winner_names[best_exp_idx]}\n")

    results_summary = {
        'Exp_A_Hand_Only': eval_A,
        'Exp_B_Hand_Face': eval_B,
        'Exp_C_Hand_Pose': eval_C,
        'Exp_D_Hand_Face_Pose': eval_D,
        'winning_architecture': winner_names[best_exp_idx]
    }

    with open(os.path.join(RESULTS_DIR, '3branch_fusion_metrics.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

if __name__ == '__main__':
    run_all_fusion_experiments()
