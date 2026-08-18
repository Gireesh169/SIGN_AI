import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODELS_DIR, DATA_DIR, BASE_DIR, LABEL_MAPPING_FILE, DEVICE, HIDDEN_DIM, DROPOUT,
    normalize_sequence_features
)
from model import BiLSTM_GRU_Model

MODEL_252_PATH = os.path.join(MODELS_DIR, 'best_model_252.pth')

def run_video_file_inference(video_path, model_path=MODEL_252_PATH, label_map_path=LABEL_MAPPING_FILE):
    start_time = time.time()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Load Label Map
    with open(label_map_path, 'r') as f:
        raw_map = json.load(f)
    idx_to_label = {int(k): v for k, v in raw_map.items()}
    num_classes = len(idx_to_label)

    # Load Model
    model = BiLSTM_GRU_Model(
        input_dim=252,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_bilstm_layers=1,
        num_gru_layers=1,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Extract Video Frames with robust fallback (OpenCV -> ImageIO -> Frames directory)
    abs_video_path = os.path.abspath(video_path)
    cap = cv2.VideoCapture(abs_video_path)
    frames = []
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()

    if len(frames) == 0:
        try:
            import imageio
            reader = imageio.get_reader(abs_video_path, format='ffmpeg')
            for frame_rgb in reader:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
            reader.close()
        except Exception:
            pass

    if len(frames) == 0:
        # Fallback to Frames_Sentence_Level if path points to a sentence folder or video basename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        parent_folder = os.path.basename(os.path.dirname(video_path))
        
        # Extract digits if present in parentheses e.g. "thank you so much (1)" -> "1"
        import re
        digit_match = re.search(r'\((\d+)\)', base_name)
        folder_num = digit_match.group(1) if digit_match else base_name

        candidate_dirs = [
            os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus', 'Frames_Sentence_Level', parent_folder, folder_num),
            os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus', 'Frames_Sentence_Level', parent_folder, base_name),
            os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus', 'Frames_Sentence_Level', parent_folder, '1')
        ]

        for frames_dir in candidate_dirs:
            if os.path.exists(frames_dir):
                img_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                for img_f in img_files:
                    img_p = os.path.join(frames_dir, img_f)
                    img = cv2.imread(img_p)
                    if img is not None:
                        frames.append(img)
                if len(frames) > 0:
                    break

    if len(frames) == 0:
        raise ValueError(f"Could not read frames from video file or fallback directory: {video_path}")

    # Uniform 30 frame sampling
    seq_len = 30
    if len(frames) >= seq_len:
        indices = np.linspace(0, len(frames) - 1, seq_len, dtype=int)
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = list(frames)
        while len(sampled_frames) < seq_len:
            sampled_frames.append(frames[-1])

    # Extract MediaPipe Landmarks (Left: 0:63, Right: 63:126)
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    sequence_landmarks = []
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

        sequence_landmarks.append(np.concatenate([left_hand, right_hand]))

    detector.close()

    # 1. Wrist Centering & Palm Scale Normalization
    pos_arr = np.array(sequence_landmarks, dtype=np.float32) # (30, 126)
    pos_norm = normalize_sequence_features(pos_arr) # (30, 126)

    # 2. Causal Velocity Calculation (30, 126)
    vel_norm = np.zeros_like(pos_norm)
    for t in range(1, seq_len):
        vel_norm[t] = pos_norm[t] - pos_norm[t-1]

    # 3. Concatenate Position + Velocity -> (30, 252)
    feat_252 = np.concatenate([pos_norm, vel_norm], axis=-1)

    # 4. PyTorch Model Evaluation
    input_tensor = torch.tensor(feat_252, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_sentence = idx_to_label.get(pred_idx.item(), f"Class_{pred_idx.item()}")
    confidence_pct = float(conf.item() * 100.0)
    elapsed_sec = time.time() - start_time

    print("="*60)
    print("=== 252D VIDEO FILE INFERENCE REPORT ===")
    print("="*60)
    print(f"Video Path       : {video_path}")
    print(f"Total Video Frames: {len(frames)}")
    print(f"Sampled Sequence  : 30 x 252D")
    print(f"Predicted Sentence: {pred_sentence}")
    print(f"Confidence        : {confidence_pct:.2f}%")
    print(f"Processing Time   : {elapsed_sec:.2f} seconds")
    print("="*60 + "\n")

    return {
        'video_path': video_path,
        'prediction': pred_sentence,
        'confidence': confidence_pct,
        'processing_time_sec': elapsed_sec
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 252D ISL sentence recognition on a video file.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to input sign video file")
    args = parser.parse_args()

    run_video_file_inference(args.video_path)
