import os
import sys
import json
import csv
import cv2
import numpy as np
import mediapipe as mp
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VIDEOS_DIR, FRAMES_DIR, DATA_DIR, BASE_DIR,
    X_FEATURES_FILE, Y_LABELS_FILE, LABEL_MAPPING_FILE,
    SEQUENCE_LENGTH, FEATURE_DIM
)

def rebuild_clean_dataset():
    CSV_PATH = os.path.join(DATA_DIR, 'dataset_video_status.csv')
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Run dataset_verification.py first.")

    with open(CSV_PATH) as f:
        valid_records = [row for row in csv.DictReader(f) if row['status'] == 'processed']

    with open(LABEL_MAPPING_FILE) as f:
        full_mapping = json.load(f)
    full_idx_to_label = {int(k): v for k, v in full_mapping.items()}
    full_label_to_idx = {v: int(k) for k, v in full_mapping.items()}

    print(f"Rebuilding features for {len(valid_records)} verified valid records...")

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    x_list = []
    y_raw_list = []

    for i, rec in enumerate(valid_records, start=1):
        v_path = rec['video_path']
        decoder = rec['decoder_used']
        frames = []

        if decoder == 'opencv' and os.path.exists(v_path):
            cap = cv2.VideoCapture(v_path)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None: break
                    frames.append(frame)
                cap.release()

        if len(frames) == 0:
            cls_name = rec['class_name']
            v_name = rec['video_name']
            cls_vid_dir = os.path.dirname(v_path)
            c_frm_dir = os.path.join(FRAMES_DIR, cls_name)

            if os.path.exists(c_frm_dir) and os.path.exists(cls_vid_dir):
                c_vids = sorted([f for f in os.listdir(cls_vid_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV'))])
                idx = c_vids.index(v_name) + 1 if v_name in c_vids else 1
                frame_folders = sorted([d for d in os.listdir(c_frm_dir) if os.path.isdir(os.path.join(c_frm_dir, d)) and not d.startswith('.')], key=lambda x: int(x) if x.isdigit() else x)

                target_sf = os.path.join(c_frm_dir, str(idx))
                if not os.path.exists(target_sf) and (idx - 1) < len(frame_folders):
                    target_sf = os.path.join(c_frm_dir, frame_folders[idx - 1])

                if os.path.exists(target_sf):
                    imgs = sorted([f for f in os.listdir(target_sf) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    for img_f in imgs:
                        frm = cv2.imread(os.path.join(target_sf, img_f))
                        if frm is not None:
                            frames.append(frm)

        if len(frames) == 0:
            continue

        # Uniform sampling to 30 frames
        if len(frames) >= SEQUENCE_LENGTH:
            indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            sampled_frames = [frames[idx] for idx in indices]
        else:
            indices = np.round(np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH)).astype(int)
            sampled_frames = [frames[idx] for idx in indices]

        sequence_features = []
        for frame in sampled_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            left_hand = np.zeros(63, dtype=np.float32)
            right_hand = np.zeros(63, dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    coords = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
                    if label == 'Left':
                        left_hand = np.array(coords, dtype=np.float32)
                    else:
                        right_hand = np.array(coords, dtype=np.float32)

            frame_features = np.concatenate([left_hand, right_hand])
            sequence_features.append(frame_features)

        feat_arr = np.array(sequence_features, dtype=np.float32)
        if feat_arr.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
            c_idx = full_label_to_idx[rec['class_name']]
            x_list.append(feat_arr)
            y_raw_list.append(c_idx)

        if i % 100 == 0 or i == len(valid_records):
            print(f"Extracted {i}/{len(valid_records)} samples...")

    x_all = np.array(x_list, dtype=np.float32)
    y_raw = np.array(y_raw_list, dtype=np.int64)

    print(f"Extraction complete: Total valid sample tensors = {len(x_all)}")

    # Class sample count analysis
    counts = Counter(y_raw)
    all_classes = set(range(101))
    active_classes = set([c for c, cnt in counts.items() if cnt > 0])
    missing_classes = sorted(list(all_classes - active_classes))
    low_sample_classes = sorted([c for c, cnt in counts.items() if 0 < cnt < 5])
    active_97_classes = sorted([c for c, cnt in counts.items() if cnt >= 5])

    # Filter dataset to 97 active classes (>= 5 samples)
    keep_indices = [idx for idx, label in enumerate(y_raw) if label in active_97_classes]
    x_filtered = x_all[keep_indices]
    y_raw_filtered = y_raw[keep_indices]

    # Map to contiguous 0..96 class index mapping
    old_to_new_idx = {old_i: new_i for new_i, old_i in enumerate(active_97_classes)}
    y_97 = np.array([old_to_new_idx[label] for label in y_raw_filtered], dtype=np.int64)

    idx_97_to_label = {new_i: full_idx_to_label[old_i] for new_i, old_i in enumerate(active_97_classes)}

    # Save cleaned features
    np.save(X_FEATURES_FILE, x_filtered)
    np.save(Y_LABELS_FILE, y_97)

    # Save 97-class label mapping for training
    with open(os.path.join(BASE_DIR, 'models', 'label_mapping.json'), 'w') as f:
        json.dump(idx_97_to_label, f, indent=2)

    # Save excluded classes JSON
    excluded_info = {
        'missing_classes_0_samples': [{'class_idx': c, 'class_name': full_idx_to_label[c]} for c in missing_classes],
        'low_sample_classes_under_5_samples': [{'class_idx': c, 'class_name': full_idx_to_label[c], 'sample_count': counts[c]} for c in low_sample_classes],
        'total_active_training_classes': len(active_97_classes)
    }

    with open(os.path.join(DATA_DIR, 'excluded_low_sample_classes.json'), 'w') as f:
        json.dump(excluded_info, f, indent=2)

    counts_97 = Counter(y_97)
    min_s = min(counts_97.values())
    max_s = max(counts_97.values())

    print("\n" + "="*60)
    print("=== BEFORE TRAINING VERIFIED DATASET REPORT ===")
    print("="*60)
    print(f"Verified samples           : {len(x_filtered)}")
    print(f"Number of classes          : {len(active_97_classes)}")
    print(f"Minimum samples/class      : {min_s}")
    print(f"Maximum samples/class      : {max_s}")
    print(f"Excluded low-sample classes: {len(low_sample_classes)} ({', '.join([full_idx_to_label[c] for c in low_sample_classes])})")
    print(f"Missing classes            : {len(missing_classes)} ({', '.join([full_idx_to_label[c] for c in missing_classes])})")
    print(f"Final feature shape        : {x_filtered.shape}")
    print("="*60 + "\n")

    return x_filtered.shape, len(active_97_classes), len(x_filtered)

if __name__ == '__main__':
    rebuild_clean_dataset()
