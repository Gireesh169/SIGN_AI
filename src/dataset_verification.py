import os
import sys
import json
import csv
import numpy as np
import cv2
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VIDEOS_DIR, FRAMES_DIR, DATA_DIR, BASE_DIR, X_FEATURES_FILE, Y_LABELS_FILE, LABEL_MAPPING_FILE

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CSV_PATH = os.path.join(DATA_DIR, 'dataset_video_status.csv')
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, 'dataset_video_status.csv')

def run_dataset_audit():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(VIDEOS_DIR):
        raise FileNotFoundError(f"Videos directory {VIDEOS_DIR} not found.")

    classes = sorted([d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d)) and not d.startswith('.')])
    
    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    label_to_idx = {v: int(k) for k, v in label_mapping.items()}

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV')

    x_current = np.load(X_FEATURES_FILE) if os.path.exists(X_FEATURES_FILE) else np.zeros((0, 30, 126), dtype=np.float32)
    y_current = np.load(Y_LABELS_FILE) if os.path.exists(Y_LABELS_FILE) else np.zeros((0,), dtype=np.int64)

    records = []
    total_original_videos = 0
    recovered_samples = 0
    unrecoverable_videos = 0
    missing_videos = 0

    class_sample_counts = Counter()
    unrecoverable_by_class = Counter()

    fieldnames = ['class_name', 'class_idx', 'video_name', 'video_path', 'status', 'decoder_used', 'frame_count', 'notes']

    for cls in classes:
        cls_idx = label_to_idx.get(cls, -1)
        cls_video_dir = os.path.join(VIDEOS_DIR, cls)
        cls_frames_dir = os.path.join(FRAMES_DIR, cls) if os.path.exists(FRAMES_DIR) else None

        vids = sorted([f for f in os.listdir(cls_video_dir) if f.lower().endswith(video_extensions)])
        
        frame_folders = []
        if cls_frames_dir and os.path.exists(cls_frames_dir):
            frame_folders = sorted([d for d in os.listdir(cls_frames_dir) if os.path.isdir(os.path.join(cls_frames_dir, d)) and not d.startswith('.')], key=lambda x: int(x) if x.isdigit() else x)

        for idx, v_name in enumerate(vids, start=1):
            total_original_videos += 1
            v_path = os.path.join(cls_video_dir, v_name)

            status = 'failed'
            decoder = 'none'
            f_count = 0
            notes = ''

            # 1. OpenCV check
            cap = cv2.VideoCapture(v_path)
            is_open = cap.isOpened()
            if is_open:
                f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                status = 'processed'
                decoder = 'opencv'
                cap.release()
            else:
                cap.release()

            # 2. Frames fallback check if video decoders failed
            if status == 'failed' and len(frame_folders) > 0:
                target_sf = os.path.join(cls_frames_dir, str(idx))
                if not os.path.exists(target_sf) and (idx - 1) < len(frame_folders):
                    target_sf = os.path.join(cls_frames_dir, frame_folders[idx - 1])

                if os.path.exists(target_sf):
                    imgs = [f for f in os.listdir(target_sf) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(imgs) > 0:
                        status = 'processed'
                        decoder = 'frames_fallback'
                        f_count = len(imgs)
                        notes = f"Recovered via frame images in {os.path.basename(target_sf)}"

            if status == 'failed':
                notes = "Corrupt video container (moov atom missing) & no frame images available"

            if status == 'processed':
                recovered_samples += 1
                class_sample_counts[cls_idx] += 1
            else:
                unrecoverable_videos += 1
                unrecoverable_by_class[cls_idx] += 1

            records.append({
                'class_name': cls,
                'class_idx': cls_idx,
                'video_name': v_name,
                'video_path': v_path,
                'status': status,
                'decoder_used': decoder,
                'frame_count': f_count,
                'notes': notes
            })

    # Save dataset_video_status.csv in data/ and results/
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    with open(RESULTS_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Class analysis
    all_class_indices = set(range(len(classes)))
    active_class_indices = set([c for c, count in class_sample_counts.items() if count > 0])
    missing_class_indices = all_class_indices - active_class_indices

    missing_classes_info = [(c, idx_to_label.get(c, f"Class_{c}")) for c in sorted(list(missing_class_indices))]
    low_sample_classes_info = [(c, idx_to_label.get(c, f"Class_{c}"), class_sample_counts[c]) for c in sorted(list(active_class_indices)) if class_sample_counts[c] < 5]

    sample_counts_list = [class_sample_counts[c] for c in active_class_indices]
    min_samples = min(sample_counts_list) if sample_counts_list else 0
    max_samples = max(sample_counts_list) if sample_counts_list else 0
    avg_samples = float(np.mean(sample_counts_list)) if sample_counts_list else 0.0

    print("\n" + "="*60)
    print("=== DATASET VERIFICATION & RECOVERY REPORT ===")
    print("="*60)
    print(f"Total original videos      : {total_original_videos}")
    print(f"Total recovered samples    : {recovered_samples}")
    print(f"Total unrecoverable videos : {unrecoverable_videos}")
    print(f"Total active classes       : {len(active_class_indices)}")
    print(f"Missing classes            : {len(missing_classes_info)}")
    print(f"Minimum samples/class      : {min_samples}")
    print(f"Maximum samples/class      : {max_samples}")
    print(f"Average samples/class      : {avg_samples:.2f}")

    print("\n--- Missing Classes (0 Recoverable Samples) ---")
    if missing_classes_info:
        for idx, cls_name in missing_classes_info:
            print(f"  Class [{idx:03d}] '{cls_name}' (All {unrecoverable_by_class[idx]} video files corrupted, 0 frames)")
    else:
        print("  None (All 101 classes represented)")

    print(f"\n--- Low-Sample Classes (< 5 Samples, count={len(low_sample_classes_info)}) ---")
    for idx, cls_name, cnt in low_sample_classes_info:
        print(f"  Class [{idx:03d}] '{cls_name}': {cnt} samples")

    print("\nCSV status report saved to:")
    print(f"  - {CSV_PATH}")
    print(f"  - {RESULTS_CSV_PATH}")
    print("="*60 + "\n")

    return {
        'total_original_videos': total_original_videos,
        'total_recovered_samples': recovered_samples,
        'total_unrecoverable_videos': unrecoverable_videos,
        'total_active_classes': len(active_class_indices),
        'missing_classes_count': len(missing_classes_info),
        'min_samples_per_class': min_samples,
        'max_samples_per_class': max_samples,
        'avg_samples_per_class': avg_samples,
        'feature_shape': list(x_current.shape)
    }

if __name__ == '__main__':
    run_dataset_audit()
