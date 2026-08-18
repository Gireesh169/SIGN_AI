import os
import sys
import time
import json
import numpy as np
import torch
import cv2
import mediapipe as mp
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODELS_DIR, LABEL_MAPPING_FILE, DEVICE, HIDDEN_DIM, DROPOUT,
    normalize_sequence_features
)
from model import BiLSTM_GRU_Model
from realtime_engine import TTSEngineWorker

MODEL_252_PATH = os.path.join(MODELS_DIR, 'best_model_252.pth')

def run_desktop_webcam_inference():
    """
    Desktop Computer Camera Real-Time ISL Sentence Recognition
    - MediaPipe processing on UN-FLIPPED original frames for 100% accurate Left/Right hand ordering
    - Visual mirror flip ONLY for display
    - 252D Position + Causal Velocity Model (`best_model_252.pth`)
    - Dynamic Top-1 Sentence & Confidence Display
    """
    if not os.path.exists(LABEL_MAPPING_FILE):
        print(f"Error: {LABEL_MAPPING_FILE} not found.")
        return

    with open(LABEL_MAPPING_FILE, 'r') as f:
        raw_map = json.load(f)
    idx_to_label = {int(k): v for k, v in raw_map.items()}
    num_classes = len(idx_to_label)

    if not os.path.exists(MODEL_252_PATH):
        print(f"Error: {MODEL_252_PATH} not found.")
        return

    # Load 252D PyTorch Model
    model = BiLSTM_GRU_Model(
        input_dim=252,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_bilstm_layers=1,
        num_gru_layers=1,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_252_PATH, map_location=DEVICE))
    model.eval()

    # Initialize Non-blocking Background TTS Worker
    tts = TTSEngineWorker()

    # Open Computer Webcam (Camera index 0, 1, or 2)
    cap = None
    for cam_idx in range(4):
        c = cv2.VideoCapture(cam_idx)
        if c.isOpened():
            ret, tmp = c.read()
            if ret and tmp is not None:
                cap = c
                print(f"Successfully opened computer camera at index {cam_idx}!")
                break
            c.release()

    if cap is None or not cap.isOpened():
        print("Error: Could not access computer camera.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    sequence_length = 30
    pos_buffer = deque(maxlen=sequence_length)
    prediction_history = deque(maxlen=5)

    current_sentence = "Waiting for hands..."
    current_confidence = 0.0
    confidence_threshold = 0.20 # 20% minimum threshold for 97 classes
    required_agreement = 3

    fps = 0.0
    start_fps_time = time.time()
    frame_count = 0
    last_inference_time_ms = 0.0

    print("\n" + "="*70)
    print("=== DESKTOP COMPUTER CAMERA 252D REAL-TIME RECOGNITION ===")
    print("="*70)
    print("Press 'q' or ESC in the video window to stop.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t_now = time.time()
            frame_count += 1
            if t_now - start_fps_time >= 1.0:
                fps = frame_count / (t_now - start_fps_time)
                frame_count = 0
                start_fps_time = t_now

            # CRITICAL FIX: Process MediaPipe on UN-FLIPPED frame to preserve true Left/Right handedness!
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            left_hand = np.zeros(63, dtype=np.float32)
            right_hand = np.zeros(63, dtype=np.float32)
            hands_count = 0

            # Create display frame and draw hand landmarks BEFORE mirror flip
            display_frame = frame.copy()

            if results.multi_hand_landmarks and results.multi_handedness:
                hands_count = len(results.multi_hand_landmarks)
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    label = handedness.classification[0].label # 'Left' or 'Right'
                    coords = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
                    if label == 'Left':
                        left_hand = np.array(coords, dtype=np.float32)
                    else:
                        right_hand = np.array(coords, dtype=np.float32)

            # Mirror flip display frame AFTER landmark drawing for natural user experience
            display_frame = cv2.flip(display_frame, 1)

            raw_126 = np.concatenate([left_hand, right_hand])

            if hands_count > 0:
                pos_buffer.append(raw_126)
            else:
                if len(pos_buffer) > 0:
                    pos_buffer.popleft()
                if len(pos_buffer) == 0:
                    current_sentence = "Waiting for hands..."
                    current_confidence = 0.0
                    prediction_history.clear()

            # 252D Real-time Inference on 30-frame rolling window
            if len(pos_buffer) == sequence_length:
                t_inf_start = time.time()
                pos_arr = np.array(pos_buffer)
                pos_norm = normalize_sequence_features(pos_arr)

                # Causal velocity calculation
                vel_norm = np.zeros_like(pos_norm)
                for t_idx in range(1, sequence_length):
                    vel_norm[t_idx] = pos_norm[t_idx] - pos_norm[t_idx-1]

                feat_252 = np.concatenate([pos_norm, vel_norm], axis=-1)
                input_tensor = torch.tensor(feat_252, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)

                pred_label = idx_to_label.get(pred_idx.item(), f"Class_{pred_idx.item()}")
                conf_val = float(conf.item())
                last_inference_time_ms = (time.time() - t_inf_start) * 1000.0

                prediction_history.append((pred_label, conf_val))

                labels = [item[0] for item in prediction_history]
                confs = [item[1] for item in prediction_history]
                label_counts = Counter(labels)
                most_common_label, count = label_counts.most_common(1)[0]
                avg_conf = np.mean(confs)

                # Update Display Sentence Dynamically
                current_sentence = most_common_label
                current_confidence = float(avg_conf * 100.0)

                # Trigger Audio Speech when confident and agreed
                if count >= required_agreement and avg_conf >= confidence_threshold:
                    tts.speak_async(most_common_label)

                print(f"[DESKTOP CAM DEBUG] FPS: {fps:.1f} | Latency: {last_inference_time_ms:.2f}ms | Hands: {hands_count} | Buffer: {len(pos_buffer)}/30 | Pred: {current_sentence} ({current_confidence:.1f}%)")

            # Draw HUD Overlay
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 75), (15, 20, 25), -1)
            cv2.line(display_frame, (0, 75), (display_frame.shape[1], 75), (50, 150, 255), 2)

            cv2.putText(display_frame, f"Recognized: {current_sentence}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Confidence: {current_confidence:.1f}% | FPS: {fps:.1f} | Latency: {last_inference_time_ms:.1f}ms", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("ISL Real-Time Recognition (Computer Camera)", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    print("Desktop Computer Camera inference stopped.")

if __name__ == '__main__':
    run_desktop_webcam_inference()
