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

def build_dataset_97():
    CSV_PATH = os.path.join(DATA_DIR, 'dataset_video_status.csv')
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")

    with open(CSV_PATH) as f:
        proc_records = [row for row in csv.DictReader(f) if row['status'] == 'processed']

    with open(LABEL_MAPPING_FILE) as f:
        full_mapping = json.load(f)
    full_idx_to_label = {int(k): v for k, v in full_mapping.items()}
    full_label_to_idx = {v: int(k) for k, v in full_mapping.items()}

    c_counts = Counter([r['class_name'] for r in proc_records])
    
    # 97 active training classes (>= 5 samples)
    active_97_names = sorted([name for name in c_counts if c_counts[name] >= 5])
    low_sample_names = sorted([name for name in c_counts if c_counts[name] < 5])
    missing_names = sorted([name for name in full_label_to_idx if name not in c_counts])

    # Mapping for 97 classes (0..96)
    label_to_idx_97 = {name: i for i, name in enumerate(active_97_names)}
    idx_97_to_label = {i: name for i, name in enumerate(active_97_names)}

    valid_97_records = [r for r in proc_records if r['class_name'] in active_97_names]

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    x_list = []
    y_list = []

    print(f"Building clean 97-class dataset ({len(valid_97_records)} samples)...")

    for i, rec in enumerate(valid_97_records, start=1):
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
            x_list.append(feat_arr)
            y_list.append(label_to_idx_97[rec['class_name']])

        if i % 100 == 0 or i == len(valid_97_records):
            print(f"Extracted {i}/{len(valid_97_records)} samples...")

    x_arr = np.array(x_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)

    # Save clean features
    np.save(X_FEATURES_FILE, x_arr)
    np.save(Y_LABELS_FILE, y_arr)

    # Save models/label_mapping.json (97 classes)
    with open(os.path.join(BASE_DIR, 'models', 'label_mapping.json'), 'w') as f:
        json.dump(idx_97_to_label, f, indent=2)

    # Save data/excluded_low_sample_classes.json
    excluded_info = {
        'missing_classes_0_samples': [{'class_idx': full_label_to_idx[name], 'class_name': name} for name in missing_names],
        'low_sample_classes_under_5_samples': [{'class_idx': full_label_to_idx[name], 'class_name': name, 'sample_count': c_counts[name]} for name in low_sample_names],
        'total_active_training_classes': len(active_97_names)
    }

    with open(os.path.join(DATA_DIR, 'excluded_low_sample_classes.json'), 'w') as f:
        json.dump(excluded_info, f, indent=2)

    counts_97 = Counter(y_arr)
    min_s = min(counts_97.values())
    max_s = max(counts_97.values())

    print("\n" + "="*60)
    print("=== BEFORE TRAINING VERIFIED DATASET REPORT ===")
    print("="*60)
    print(f"Verified samples           : {len(x_arr)}")
    print(f"Number of classes          : {len(active_97_names)}")
    print(f"Minimum samples/class      : {min_s}")
    print(f"Maximum samples/class      : {max_s}")
    print(f"Excluded low-sample classes: {len(low_sample_names)} ({', '.join(low_sample_names)})")
    print(f"Missing classes            : {len(missing_names)} ({', '.join(missing_names)})")
    print(f"Final feature shape        : {x_arr.shape}")
    print("="*60 + "\n")

    return x_arr.shape

if __name__ == '__main__':
    build_dataset_97()
