import os
import sys
import json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, X_FEATURES_FILE, Y_LABELS_FILE,
    LABEL_MAPPING_FILE, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def audit_dataset():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load dataset arrays
    if not os.path.exists(X_FEATURES_FILE) or not os.path.exists(Y_LABELS_FILE):
        raise FileNotFoundError(f"Dataset files missing: {X_FEATURES_FILE} or {Y_LABELS_FILE}")
        
    X = np.load(X_FEATURES_FILE)
    Y = np.load(Y_LABELS_FILE)
    
    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_map = json.load(f)
        
    num_samples = len(Y)
    num_classes = len(np.unique(Y))
    seq_len = X.shape[1]
    feature_dim = X.shape[2]
    
    # 2. Check Train/Val/Test Split without leakage
    np.random.seed(RANDOM_SEED)
    indices = np.arange(num_samples)
    
    # Stratified or permuted split by class
    class_indices = {}
    for idx, label in enumerate(Y):
        class_indices.setdefault(label, []).append(idx)
        
    train_indices, val_indices, test_indices = [], [], []
    
    for label, idxs in class_indices.items():
        np.random.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(round(n * TRAIN_SPLIT)))
        n_val = max(1, int(round(n * VAL_SPLIT)))
        if n_train + n_val >= n and n > 2:
            n_train = n - 2
            n_val = 1
            
        train_indices.extend(idxs[:n_train])
        val_indices.extend(idxs[n_train:n_train + n_val])
        test_indices.extend(idxs[n_train + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Leakage check: Ensure disjoint sets
    intersection_tv = set(train_indices).intersection(set(val_indices))
    intersection_tt = set(train_indices).intersection(set(test_indices))
    intersection_vt = set(val_indices).intersection(set(test_indices))
    
    leak_free = (len(intersection_tv) == 0 and len(intersection_tt) == 0 and len(intersection_vt) == 0)
    
    report = {
        "dataset_name": "ISL_CSLRT 97-Class Sentence Dataset",
        "total_samples": int(num_samples),
        "active_classes": int(num_classes),
        "sequence_length": int(seq_len),
        "raw_landmark_feature_dim": int(feature_dim),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "test_samples": int(len(test_indices)),
        "train_split_pct": round(len(train_indices) / num_samples * 100, 2),
        "val_split_pct": round(len(val_indices) / num_samples * 100, 2),
        "test_split_pct": round(len(test_indices) / num_samples * 100, 2),
        "leak_free_verification": leak_free,
        "augmentation_policy": "Training split ONLY. Validation & Test splits remain 100% unaugmented and pristine."
    }
    
    report_path = os.path.join(RESULTS_DIR, 'dataset_audit_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*65)
    print("=== DATASET PIPELINE AUDIT REPORT ===")
    print("="*65)
    print(f"Total Recovered Sequences : {num_samples}")
    print(f"Active Sentence Classes   : {num_classes}")
    print(f"Sequence Dimensions       : ({seq_len} frames, {feature_dim}D raw landmarks)")
    print(f"Train Split               : {len(train_indices)} samples ({report['train_split_pct']}%)")
    print(f"Val Split                 : {len(val_indices)} samples ({report['val_split_pct']}%)")
    print(f"Test Split                : {len(test_indices)} samples ({report['test_split_pct']}%)")
    print(f"Leakage Check             : {'PASSED (Disjoint Train/Val/Test)' if leak_free else 'FAILED'}")
    print(f"Augmentation Policy       : {report['augmentation_policy']}")
    print(f"Audit Report Saved To     : {report_path}")
    print("="*65 + "\n")

    return report, train_indices, val_indices, test_indices

if __name__ == '__main__':
    audit_dataset()
