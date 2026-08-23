import os
import sys
import json
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, DATA_DIR, VIDEOS_DIR, FRAMES_DIR, LABEL_MAPPING_FILE, DEVICE
from hand_crop_extractor import HandCropCNNEncoder, extract_hand_crops_sequence

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

HAND_CROPS_FILE = os.path.join(DATA_DIR, 'x_hand_crop_features_64.npy')
RAW_126_FILE = os.path.join(DATA_DIR, 'x_hand_features_252.npy') # Raw landmarks for bounding box

def extract_hand_crop_embeddings():
    print("\n" + "="*70)
    print("=== EXTRACTING 64D HAND-CROP MOBILENETV3 VISUAL EMBEDDINGS ===")
    print("="*70)

    # Initialize MobileNetV3-Small Hand Crop Encoder
    encoder = HandCropCNNEncoder(embed_dim=64).to(DEVICE)
    encoder.eval()

    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    label_to_idx = {v: int(k) for k, v in label_mapping.items()}

    classes = sorted([d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d)) and not d.startswith('.')])
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV')

    all_visual_embeddings = []
    processed_count = 0

    for cls in classes:
        cls_idx = label_to_idx.get(cls, -1)
        if cls_idx == -1: continue

        cls_video_dir = os.path.join(VIDEOS_DIR, cls)
        cls_frames_dir = os.path.join(FRAMES_DIR, cls) if os.path.exists(FRAMES_DIR) else None

        vids = sorted([f for f in os.listdir(cls_video_dir) if f.lower().endswith(video_extensions)])
        frame_folders = []
        if cls_frames_dir and os.path.exists(cls_frames_dir):
            frame_folders = sorted([d for d in os.listdir(cls_frames_dir) if os.path.isdir(os.path.join(cls_frames_dir, d)) and not d.startswith('.')], key=lambda x: int(x) if x.isdigit() else x)

        for idx, v_name in enumerate(vids, start=1):
            v_path = os.path.join(cls_video_dir, v_name)
            frames = []

            # 1. Read video frames via OpenCV
            cap = cv2.VideoCapture(v_path)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None: break
                    frames.append(frame)
                cap.release()

            # 2. Fallback to pre-extracted RGB frames if video decoders fail
            if len(frames) == 0 and len(frame_folders) > 0:
                target_sf = os.path.join(cls_frames_dir, str(idx))
                if not os.path.exists(target_sf) and (idx - 1) < len(frame_folders):
                    target_sf = os.path.join(cls_frames_dir, frame_folders[idx - 1])
                if os.path.exists(target_sf):
                    img_files = sorted([f for f in os.listdir(target_sf) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    for img_f in img_files:
                        img_path = os.path.join(target_sf, img_f)
                        img = cv2.imread(img_path)
                        if img is not None: frames.append(img)

            if len(frames) == 0:
                continue

            # Sample exactly 30 frames uniformly
            total_f = len(frames)
            indices = np.linspace(0, total_f - 1, 30, dtype=int)
            sampled_frames = [frames[i] for i in indices]

            # Generate dummy/center bounding box coords for MobileNetV3 crop extractor
            # Standard crop centered on upper-body / hands region
            dummy_landmarks_30x126 = np.zeros((30, 126), dtype=np.float32)
            for t in range(30):
                # Left hand center estimation
                dummy_landmarks_30x126[t, :63:3] = 0.35
                dummy_landmarks_30x126[t, 1:63:3] = 0.50
                # Right hand center estimation
                dummy_landmarks_30x126[t, 63::3] = 0.65
                dummy_landmarks_30x126[t, 64::3] = 0.50

            # Extract hand crop PyTorch tensor (30, 2, 3, 112, 112)
            crops_seq_t = extract_hand_crops_sequence(sampled_frames, dummy_landmarks_30x126, crop_size=112)
            crops_batch = crops_seq_t.unsqueeze(0).to(DEVICE) # (1, 30, 2, 3, 112, 112)

            with torch.no_grad():
                embed_64 = encoder(crops_batch).squeeze(0).cpu().numpy() # (30, 64)

            all_visual_embeddings.append(embed_64)
            processed_count += 1

    all_visual_embeddings = np.array(all_visual_embeddings, dtype=np.float32)
    np.save(HAND_CROPS_FILE, all_visual_embeddings)

    print("\n" + "="*70)
    print("=== HAND-CROP VISUAL EMBEDDING EXTRACTION COMPLETE ===")
    print("="*70)
    print(f"Total Sequences Processed: {processed_count}")
    print(f"Visual Embedding Shape   : {all_visual_embeddings.shape} (Expected: (673, 30, 64))")
    print(f"Saved Array File         : {HAND_CROPS_FILE}")
    print("="*70 + "\n")

if __name__ == '__main__':
    extract_hand_crop_embeddings()
