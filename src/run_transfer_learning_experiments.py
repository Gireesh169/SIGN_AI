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
from run_hand_crop_experiments import (
    HandCropExperimentDataset, GatedMotionVisualClassifier,
    train_experiment, evaluate_experiment
)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

X_252_FILE = os.path.join(DATA_DIR, 'x_features.npy')
X_DEFAULT_CROP_FILE = os.path.join(DATA_DIR, 'x_hand_crop_features_64.npy')
X_WORD_PRETRAINED_FILE = os.path.join(DATA_DIR, 'x_word_pretrained_crop_features_64.npy')
Y_FILE = os.path.join(DATA_DIR, 'y_labels.npy')

def run_transfer_learning_benchmark():
    print("\n" + "="*70)
    print("=== STARTING TRANSFER LEARNING BENCHMARK EXPERIMENTS ===")
    print("="*70)

    X_raw_motion = np.load(X_252_FILE)
    X_default_visual = np.load(X_DEFAULT_CROP_FILE)
    X_word_visual = np.load(X_WORD_PRETRAINED_FILE)
    Y_labels = np.load(Y_FILE)

    # Compute 252D Position + Velocity features
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

    # ----------------------------------------------------
    # EXPERIMENT A: Default ImageNet MobileNetV3 + Gated Fusion
    # ----------------------------------------------------
    print("\n---> Running Exp A: Default ImageNet MobileNetV3 + Gated Fusion")
    ds_train_A = HandCropExperimentDataset(X_motion[train_idx], X_default_visual[train_idx], Y_labels[train_idx], is_training=True)
    ds_val_A   = HandCropExperimentDataset(X_motion[val_idx],   X_default_visual[val_idx],   Y_labels[val_idx],   is_training=False)
    ds_test_A  = HandCropExperimentDataset(X_motion[test_idx],  X_default_visual[test_idx],  Y_labels[test_idx],  is_training=False)

    dl_train_A = DataLoader(ds_train_A, batch_size=BATCH_SIZE, shuffle=True)
    dl_val_A   = DataLoader(ds_val_A,   batch_size=BATCH_SIZE, shuffle=False)
    dl_test_A  = DataLoader(ds_test_A,  batch_size=BATCH_SIZE, shuffle=False)

    model_A = GatedMotionVisualClassifier(motion_dim=252, visual_dim=64, num_classes=num_classes).to(DEVICE)
    save_A = os.path.join(MODELS_DIR, 'best_model_expA_default_cnn.pth')
    model_A, val_acc_A, gap_A = train_experiment(model_A, dl_train_A, dl_val_A, num_epochs=45, save_path=save_A)
    eval_A = evaluate_experiment(model_A, dl_test_A)
    eval_A['train_val_gap'] = gap_A

    # ----------------------------------------------------
    # EXPERIMENT B: Word-Level Fine-Tuned MobileNetV3 + Gated Fusion
    # ----------------------------------------------------
    print("\n---> Running Exp B: Word-Level Fine-Tuned MobileNetV3 + Gated Fusion")
    ds_train_B = HandCropExperimentDataset(X_motion[train_idx], X_word_visual[train_idx], Y_labels[train_idx], is_training=True)
    ds_val_B   = HandCropExperimentDataset(X_motion[val_idx],   X_word_visual[val_idx],   Y_labels[val_idx],   is_training=False)
    ds_test_B  = HandCropExperimentDataset(X_motion[test_idx],  X_word_visual[test_idx],  Y_labels[test_idx],  is_training=False)

    dl_train_B = DataLoader(ds_train_B, batch_size=BATCH_SIZE, shuffle=True)
    dl_val_B   = DataLoader(ds_val_B,   batch_size=BATCH_SIZE, shuffle=False)
    dl_test_B  = DataLoader(ds_test_B,  batch_size=BATCH_SIZE, shuffle=False)

    model_B = GatedMotionVisualClassifier(motion_dim=252, visual_dim=64, num_classes=num_classes).to(DEVICE)
    save_B = os.path.join(MODELS_DIR, 'best_model_expB_word_pretrained.pth')
    model_B, val_acc_B, gap_B = train_experiment(model_B, dl_train_B, dl_val_B, num_epochs=45, save_path=save_B)
    eval_B = evaluate_experiment(model_B, dl_test_B)
    eval_B['train_val_gap'] = gap_B

    # ----------------------------------------------------
    # COMPARISON TABLE & REPORT
    # ----------------------------------------------------
    print("\n" + "="*105)
    print("=== TRANSFER LEARNING EXPERIMENT RESULTS REPORT ===")
    print("="*105)
    header = f"| {'Experiment':<28} | {'Visual CNN Backbone':<28} | {'Test Acc':<9} | {'Top-5 Acc':<9} | {'Macro F1':<9} | {'TV Gap':<8} |"
    print(header)
    print("|" + "-"*30 + "|" + "-"*30 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|")

    rows = [
        ("Exp A (Default ImageNet CNN)", "Default ImageNet MobileNetV3", eval_A['top1_acc']*100, eval_A['top5_acc']*100, eval_A['macro_f1'], eval_A['train_val_gap']*100),
        ("Exp B (Word-Pretrained CNN)", "Word-Level Fine-Tuned MobileNet", eval_B['top1_acc']*100, eval_B['top5_acc']*100, eval_B['macro_f1'], eval_B['train_val_gap']*100),
    ]

    for name, backbone_name, t1, t5, f1, gap in rows:
        print(f"| {name:<28} | {backbone_name:<28} | {t1:>7.2f}% | {t5:>7.2f}% | {f1:>9.4f} | {gap:>6.2f}% |")
    print("="*105 + "\n")

    best_idx = np.argmax([eval_A['top1_acc'], eval_B['top1_acc']])
    winner_names = ["Exp A (Default ImageNet CNN)", "Exp B (Word-Pretrained CNN)"]
    print(f"🏆 Winning Transfer Learning Architecture: {winner_names[best_idx]}\n")

    results_summary = {
        'Exp_A_Default_ImageNet_CNN': eval_A,
        'Exp_B_Word_Pretrained_CNN': eval_B,
        'winning_architecture': winner_names[best_idx]
    }

    with open(os.path.join(RESULTS_DIR, 'transfer_learning_experiments_metrics.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

if __name__ == '__main__':
    run_transfer_learning_benchmark()
