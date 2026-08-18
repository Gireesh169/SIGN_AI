import os
import sys
import json
import csv
import cv2
import time
import imageio
import threading
import numpy as np
import mediapipe as mp
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VIDEOS_DIR, FRAMES_DIR, DATA_DIR,
    PROGRESS_FILE, PARTIAL_X_FILE, PARTIAL_Y_FILE,
    X_FEATURES_FILE, Y_LABELS_FILE, LABEL_MAPPING_FILE, FAILED_VIDEOS_FILE,
    SEQUENCE_LENGTH, FEATURE_DIM
)

thread_local = threading.local()

def get_thread_hands():
    if not hasattr(thread_local, 'detector'):
        mp_hands = mp.solutions.hands
        thread_local.detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    return thread_local.detector

def read_video_frames(v_path, frames_dir=None, sample_idx=1):
    """
    Attempts to read frames from a video file with robust multi-backend fallback:
    1. OpenCV VideoCapture
    2. ImageIO (ffmpeg backend)
    3. Pre-extracted frame images directory (if available)
    
    Returns (frames, metadata, failure_info)
    """
    fps = 0.0
    w = 0
    h = 0
    
    # 1. Try OpenCV VideoCapture
    cap = cv2.VideoCapture(v_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        
        cv_frames = []
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    break
                continue
            consecutive_failures = 0
            cv_frames.append(frame)
        cap.release()
        
        if len(cv_frames) > 0:
            return cv_frames, {'frame_count': len(cv_frames), 'fps': fps, 'width': w, 'height': h, 'decoder': 'opencv'}, None

    # 2. Try ImageIO fallback if OpenCV failed
    iio_frames = []
    iio_err_msg = ""
    try:
        reader = imageio.get_reader(v_path, format='ffmpeg')
        try:
            meta = reader.get_meta_data()
            if fps == 0.0:
                fps = meta.get('fps', 0.0)
        except Exception:
            pass
            
        for f in reader:
            iio_frames.append(f)
        reader.close()
        
        if len(iio_frames) > 0:
            h_iio, w_iio = iio_frames[0].shape[:2]
            if w == 0: w = w_iio
            if h == 0: h = h_iio
            # Convert ImageIO RGB to BGR for consistency
            iio_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in iio_frames]
            return iio_bgr, {'frame_count': len(iio_bgr), 'fps': fps, 'width': w, 'height': h, 'decoder': 'imageio'}, None
    except Exception as e:
        iio_err_msg = str(e)

    # 3. Try Pre-extracted frames fallback
    if frames_dir and os.path.exists(frames_dir):
        target_folder = os.path.join(frames_dir, str(sample_idx))
        if not os.path.exists(target_folder):
            subdirs = [os.path.join(frames_dir, d) for d in os.listdir(frames_dir) 
                       if os.path.isdir(os.path.join(frames_dir, d)) and not d.startswith('.')]
            subdirs.sort(key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else os.path.basename(p))
            if 0 < sample_idx <= len(subdirs):
                target_folder = subdirs[sample_idx - 1]
            elif len(subdirs) > 0:
                target_folder = subdirs[0]
            else:
                target_folder = None
                
        if target_folder and os.path.exists(target_folder):
            img_files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            img_files.sort()
            frame_list = []
            for img_f in img_files:
                img_path = os.path.join(target_folder, img_f)
                frm = cv2.imread(img_path)
                if frm is not None:
                    frame_list.append(frm)
            if len(frame_list) > 0:
                h_f, w_f = frame_list[0].shape[:2]
                return frame_list, {'frame_count': len(frame_list), 'fps': fps, 'width': w_f, 'height': h_f, 'decoder': 'frames_fallback'}, None

    # All decoders failed
    err_msg = f"Failed OpenCV decode. ImageIO error: {iio_err_msg or 'Unknown decode error'}"
    failure_info = {
        'error_type': 'VideoDecodeError',
        'error_message': err_msg,
        'frame_count': 0,
        'fps': fps,
        'width': w,
        'height': h
    }
    return [], None, failure_info

def process_landmarks_from_frames(frames):
    """
    Uniformly samples or temporal-interpolates frames to exactly SEQUENCE_LENGTH (30) frames.
    Extracts MediaPipe left & right hand landmarks (63 coords per hand, total 126 per frame).
    Missing hands are padded with 63 zeros.
    """
    if len(frames) == 0:
        return None

    # Temporal sampling / interpolation to exactly 30 frames
    if len(frames) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
        sampled_frames = [frames[i] for i in indices]
    else:
        # Interpolate / repeat frames uniformly to reach exactly 30 frames
        indices = np.round(np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH)).astype(int)
        sampled_frames = [frames[i] for i in indices]

    detector = get_thread_hands()
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

    return np.array(sequence_features, dtype=np.float32)

def process_single_sample(sample):
    try:
        frames, meta, fail_info = read_video_frames(
            sample['video_path'],
            frames_dir=sample['frames_dir'],
            sample_idx=sample['sample_idx']
        )
        
        if fail_info is not None or len(frames) == 0:
            if fail_info is None:
                fail_info = {
                    'error_type': 'EmptyVideoError',
                    'error_message': 'No frames extracted from video',
                    'frame_count': 0,
                    'fps': 0.0,
                    'width': 0,
                    'height': 0
                }
            return sample['sample_id'], None, sample['class_idx'], {
                'video_path': sample['video_path'],
                'class_name': sample['class'],
                'error_type': fail_info['error_type'],
                'error_message': fail_info['error_message'],
                'frame_count': fail_info['frame_count'],
                'fps': fail_info['fps'],
                'width': fail_info['width'],
                'height': fail_info['height']
            }

        seq_features = process_landmarks_from_frames(frames)
        if seq_features is not None and seq_features.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
            return sample['sample_id'], seq_features, sample['class_idx'], None
        else:
            shape_str = seq_features.shape if seq_features is not None else "None"
            return sample['sample_id'], None, sample['class_idx'], {
                'video_path': sample['video_path'],
                'class_name': sample['class'],
                'error_type': 'InvalidShapeError',
                'error_message': f'Extracted feature shape {shape_str} != ({SEQUENCE_LENGTH}, {FEATURE_DIM})',
                'frame_count': meta['frame_count'],
                'fps': meta['fps'],
                'width': meta['width'],
                'height': meta['height']
            }
    except Exception as e:
        return sample['sample_id'], None, sample['class_idx'], {
            'video_path': sample['video_path'],
            'class_name': sample['class'],
            'error_type': 'UnexpectedProcessingError',
            'error_message': str(e),
            'frame_count': 0,
            'fps': 0.0,
            'width': 0,
            'height': 0
        }

def run_feature_extraction(max_workers=4):
    os.makedirs(DATA_DIR, exist_ok=True)

    classes = [d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d)) and not d.startswith('.')]
    classes.sort()
    label_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_label = {i: cls_name for i, cls_name in enumerate(classes)}

    with open(LABEL_MAPPING_FILE, 'w') as f:
        json.dump(idx_to_label, f, indent=2)

    processed_samples = set()
    x_data = []
    y_data = []

    # Reset or load checkpoint
    FAILED_LOGS_FILE = os.path.join(DATA_DIR, 'failed_logs_partial.json')
    if os.path.exists(PROGRESS_FILE) and os.path.exists(PARTIAL_X_FILE) and os.path.exists(PARTIAL_Y_FILE):
        try:
            x_data = list(np.load(PARTIAL_X_FILE))
            y_data = list(np.load(PARTIAL_Y_FILE))
            if os.path.exists(FAILED_LOGS_FILE):
                with open(FAILED_LOGS_FILE, 'r') as f:
                    failed_logs = json.load(f)
            else:
                failed_logs = []
            with open(PROGRESS_FILE, 'r') as f:
                processed_samples = set(json.load(f))
            print(f"Resuming feature extraction from checkpoint: {len(processed_samples)} samples in progress, {len(x_data)} features extracted.")
        except Exception as e:
            print(f"Error loading checkpoint, starting fresh: {e}")
            processed_samples = set()
            x_data = []
            y_data = []
            failed_logs = []
    else:
        processed_samples = set()
        x_data = []
        y_data = []
        failed_logs = []



    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV')

    all_samples = []
    for cls in classes:
        cls_video_dir = os.path.join(VIDEOS_DIR, cls)
        cls_frames_dir = os.path.join(FRAMES_DIR, cls) if os.path.exists(FRAMES_DIR) else None
        vids = [f for f in os.listdir(cls_video_dir) if f.lower().endswith(video_exts)]
        vids.sort()
        for idx, v_name in enumerate(vids, start=1):
            v_path = os.path.join(cls_video_dir, v_name)
            all_samples.append({
                'class': cls,
                'class_idx': label_to_idx[cls],
                'video_path': v_path,
                'sample_idx': idx,
                'frames_dir': cls_frames_dir,
                'sample_id': f"{cls}/{v_name}"
            })

    unprocessed_samples = [s for s in all_samples if s['sample_id'] not in processed_samples]
    print(f"Total samples: {len(all_samples)} | Already extracted: {len(processed_samples)} | Remaining: {len(unprocessed_samples)}")

    lock = threading.Lock()
    completed_count = 0

    fieldnames = ['video_path', 'class_name', 'error_type', 'error_message', 'frame_count', 'fps', 'width', 'height']
    
    def save_all_outputs(x_list, y_list, fails_list):
        x_arr = np.array(x_list, dtype=np.float32) if len(x_list) > 0 else np.zeros((0, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int64) if len(y_list) > 0 else np.zeros((0,), dtype=np.int64)
        
        np.save(PARTIAL_X_FILE, x_arr)
        np.save(PARTIAL_Y_FILE, y_arr)
        np.save(X_FEATURES_FILE, x_arr)
        np.save(Y_LABELS_FILE, y_arr)
        
        with open(FAILED_LOGS_FILE, 'w') as f:
            json.dump(fails_list, f)
            
        with open(FAILED_VIDEOS_FILE, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in fails_list:
                writer.writerow(row)
        return x_arr, y_arr

    if len(unprocessed_samples) > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_sample, s) for s in unprocessed_samples]
            for future in tqdm(futures, desc=f"Extracting MediaPipe Features ({max_workers}x parallel)"):
                sid, feat, cls_idx, fail_info = future.result()
                with lock:
                    processed_samples.add(sid)
                    if feat is not None:
                        x_data.append(feat)
                        y_data.append(cls_idx)
                    elif fail_info is not None:
                        failed_logs.append(fail_info)

                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == len(unprocessed_samples):
                        with open(PROGRESS_FILE, 'w') as f:
                            json.dump(list(processed_samples), f)
                        save_all_outputs(x_data, y_data, failed_logs)

    x_arr, y_arr = save_all_outputs(x_data, y_data, failed_logs)

    total_count = len(all_samples)
    success_count = len(x_data)
    failed_count = len(failed_logs)
    success_rate = (success_count / total_count * 100.0) if total_count > 0 else 0.0

    # Calculate per-class sample counts
    class_counts = Counter(y_data)
    samples_per_class_list = [class_counts[i] for i in range(len(classes))]
    min_samples = min(samples_per_class_list) if samples_per_class_list else 0
    max_samples = max(samples_per_class_list) if samples_per_class_list else 0

    print("\n========================================")
    print("FEATURE EXTRACTION REPORT")
    print("========================================")
    print(f"Total videos: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Number of classes: {len(classes)}")
    print(f"Samples per class: min={min_samples}, max={max_samples}, avg={np.mean(samples_per_class_list):.1f}")
    print(f"Minimum samples/class: {min_samples}")
    print(f"Maximum samples/class: {max_samples}")
    print("========================================\n")

    # Print top 20 most common failure reasons
    if failed_logs:
        failure_reasons = Counter(f["error_message"] for f in failed_logs)
        print("Top Failure Reasons:")
        for reason, count in failure_reasons.most_common(20):
            print(f"  [{count}x] {reason}")
    else:
        print("Top Failure Reasons: None (0 failures)")

if __name__ == '__main__':
    run_feature_extraction()



