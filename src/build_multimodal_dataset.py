import os
import sys
import json
import csv
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, BASE_DIR, FRAMES_DIR, VIDEOS_DIR,
    X_FEATURES_FILE, Y_LABELS_FILE, LABEL_MAPPING_FILE,
    SEQUENCE_LENGTH, FEATURE_DIM
)

def build_multimodal_dataset_and_embeddings():
    start_time = time.time()
    print("="*60)
    print("=== STEP 1 & 3: MULTIMODAL DATA ALIGNMENT & FEATURE CACHING ===")
    print("="*60)

    CSV_PATH = os.path.join(DATA_DIR, 'dataset_video_status.csv')
    MANIFEST_PATH = os.path.join(DATA_DIR, 'multimodal_manifest.csv')
    EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'rgb_embeddings.npy')

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")

    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    active_97_classes = set(label_mapping.values())
    label_to_idx = {name: int(k) for k, name in label_mapping.items()}
    # Load 97-class landmark dataset (674, 30, 126)
    x_landmarks = np.load(X_FEATURES_FILE)
    y_labels = np.load(Y_LABELS_FILE)

    # Read processed valid records matching label_mapping.json
    valid_records = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'processed' and row['class_name'] in active_97_classes:
                valid_records.append(row)

    # Slice x_landmarks and y_labels to match valid records
    min_len = min(len(valid_records), len(x_landmarks))
    valid_records = valid_records[:min_len]
    x_landmarks = x_landmarks[:min_len]
    y_labels = y_labels[:min_len]

    # Save aligned landmark features
    np.save(X_FEATURES_FILE, x_landmarks)
    np.save(Y_LABELS_FILE, y_labels)

    print(f"Aligned dataset: {len(valid_records)} records, x_landmarks shape: {x_landmarks.shape}, y_labels shape: {y_labels.shape}")

    # Initialize pretrained MobileNetV3-Small for feature extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading pretrained MobileNetV3-Small feature extractor on device: {device}...")
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        mobilenet = models.mobilenet_v3_small(pretrained=True)
    
    # Remove final classification layer to get 576-dimensional feature vector
    mobilenet.classifier = nn.Identity()
    mobilenet.eval().to(device)

    # Image preprocessing pipeline (224x224, ImageNet normalization)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    manifest_rows = []
    rgb_embeddings_list = []

    print(f"\nExtracting MobileNetV3 visual embeddings for {len(valid_records)} sequences...")

    for i, rec in enumerate(valid_records):
        v_path = rec['video_path']
        cls_name = rec['class_name']
        v_name = rec['video_name']
        label_id = label_to_idx[cls_name]
        
        frames = []
        decoder = rec['decoder_used']

        # 1. Try reading pre-extracted frame subdirectories
        cls_vid_dir = os.path.dirname(v_path)
        c_frm_dir = os.path.join(FRAMES_DIR, cls_name)
        target_sf = ""

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

        # 2. Fallback to OpenCV video decoding
        if len(frames) == 0 and os.path.exists(v_path):
            cap = cv2.VideoCapture(v_path)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None: break
                    frames.append(frame)
                cap.release()

        num_orig_frames = len(frames)
        if num_orig_frames == 0:
            # Fallback black frames if corrupt
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(SEQUENCE_LENGTH)]
            num_orig_frames = SEQUENCE_LENGTH

        # Uniform temporal sampling to exactly 30 frames
        if len(frames) >= SEQUENCE_LENGTH:
            indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            sampled_frames = [frames[idx] for idx in indices]
        else:
            indices = np.round(np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH)).astype(int)
            sampled_frames = [frames[idx] for idx in indices]

        # Convert frames to RGB PyTorch batch tensor (30, 3, 224, 224)
        tensor_batch = []
        for frm in sampled_frames:
            frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            frm_resized = cv2.resize(frm_rgb, (224, 224))
            t_frm = preprocess(frm_resized)
            tensor_batch.append(t_frm)

        input_batch = torch.stack(tensor_batch).to(device) # (30, 3, 224, 224)

        # Extract MobileNetV3 features
        with torch.no_grad():
            features = mobilenet(input_batch) # (30, 576)
            if features.dim() > 2:
                features = torch.flatten(features, 1)

        seq_embedding = features.cpu().numpy().astype(np.float32) # (30, 576)
        rgb_embeddings_list.append(seq_embedding)

        # Record manifest entry
        manifest_rows.append({
            'video_id': f"vid_{i:03d}",
            'sentence': cls_name,
            'label_id': label_id,
            'landmark_path': X_FEATURES_FILE,
            'frame_folder': target_sf if target_sf else v_path,
            'num_rgb_frames': num_orig_frames,
            'landmark_shape': f"30x126"
        })

        if (i + 1) % 100 == 0 or (i + 1) == len(valid_records):
            print(f"Extracted features for {i + 1}/{len(valid_records)} sequences...")

    rgb_embeddings = np.array(rgb_embeddings_list, dtype=np.float32) # (674, 30, 576)

    # Save rgb_embeddings.npy
    np.save(EMBEDDINGS_PATH, rgb_embeddings)
    print(f"\nSaved cached visual embeddings to: {EMBEDDINGS_PATH}")
    print(f"RGB Embeddings Shape: {rgb_embeddings.shape}")

    # Save multimodal_manifest.csv
    manifest_fields = ['video_id', 'sentence', 'label_id', 'landmark_path', 'frame_folder', 'num_rgb_frames', 'landmark_shape']
    with open(MANIFEST_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved multimodal manifest to: {MANIFEST_PATH}")
    
    elapsed_time = time.time() - start_time

    print("\n" + "="*60)
    print("=== DATASET ALIGNMENT & FEATURE CACHING REPORT ===")
    print("="*60)
    print(f"Total paired sequences    : {len(manifest_rows)}")
    print(f"RGB Embeddings Shape      : {rgb_embeddings.shape}")
    print(f"Landmarks Shape           : {x_landmarks.shape}")
    print(f"Total Active Classes      : {len(active_97_classes)}")
    print(f"Feature Extraction Time   : {elapsed_time:.2f} seconds")
    print("="*60 + "\n")

    return rgb_embeddings.shape, len(manifest_rows)

if __name__ == '__main__':
    build_multimodal_dataset_and_embeddings()
