import os
import sys
import time
import json
import queue
import threading
import subprocess
import re
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

MODEL_252_PATH = os.path.join(MODELS_DIR, 'best_model_252.pth')
PREPROC_CONFIG_PATH = os.path.join(MODELS_DIR, 'preprocessing_config.json')

class TTSEngineWorker:
    """
    Dedicated non-blocking background Text-To-Speech worker thread using Queue & subprocess.
    Safely sanitizes text strings to prevent shell syntax errors on sentences like 'i am (age)'.
    """
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.last_spoken_sentence = ""
        self.is_speaking = False
        self.lock = threading.Lock()
        self.stopped = False

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def speak_async(self, sentence: str):
        if not sentence or sentence.strip() == "":
            return
        
        sentence_clean = sentence.strip()
        with self.lock:
            # Duplicate Speech Prevention
            if sentence_clean == self.last_spoken_sentence:
                return
            self.last_spoken_sentence = sentence_clean

        self.speech_queue.put(sentence_clean)

    def reset_last_spoken(self):
        with self.lock:
            self.last_spoken_sentence = ""

    def get_speaking_status(self) -> bool:
        with self.lock:
            return self.is_speaking or not self.speech_queue.empty()

    def _worker_loop(self):
        while not self.stopped:
            try:
                raw_sentence = self.speech_queue.get(timeout=0.2)
                with self.lock:
                    self.is_speaking = True

                # Sanitize text: Remove non-alphanumeric chars (parentheses, quotes, etc.)
                clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', raw_sentence).strip()

                if clean_text:
                    if sys.platform == 'darwin':
                        # Use subprocess.run to pass args safely without shell syntax errors
                        subprocess.run(['say', clean_text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.say(clean_text)
                            engine.runAndWait()
                        except Exception:
                            pass

                with self.lock:
                    self.is_speaking = False
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS ERROR] Speech worker exception: {e}")
                with self.lock:
                    self.is_speaking = False

    def stop(self):
        self.stopped = True

class RealTimeInferenceEngine:
    """
    Real-Time 252D ISL Sentence Recognition Engine
    - 30-frame rolling sliding window buffer
    - MediaPipe handedness (Left: 0..62, Right: 63..125)
    - Independent wrist-centering & palm-scale normalization
    - Causal first-order velocity calculation
    - Temporal prediction smoothing
    - Safe non-blocking background TTS worker
    """
    def __init__(self, model_path=MODEL_252_PATH, label_map_path=LABEL_MAPPING_FILE):
        self.device = DEVICE
        self.sequence_length = 30
        self.feature_dim = 252

        # Smoothing Parameters
        self.smoothing_window_size = 5
        self.confidence_threshold = 0.20
        self.required_agreement = 3

        # Initialize MediaPipe Hands Detector
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        # Load Label Mapping
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label mapping file not found at: {label_map_path}")
        with open(label_map_path, 'r') as f:
            raw_map = json.load(f)
        self.idx_to_label = {int(k): v for k, v in raw_map.items()}
        self.num_classes = len(self.idx_to_label)

        # Load PyTorch 252D Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found at: {model_path}")
        
        self.model = BiLSTM_GRU_Model(
            input_dim=self.feature_dim,
            hidden_dim=HIDDEN_DIM,
            num_classes=self.num_classes,
            num_bilstm_layers=1,
            num_gru_layers=1,
            dropout=DROPOUT
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialize TTS Worker
        self.tts = TTSEngineWorker()

        # Session Buffers & State
        self.pos_buffer = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=self.smoothing_window_size)
        self.current_stable_sentence = "Waiting for hands..."
        self.current_confidence = 0.0
        self.last_inference_time_ms = 0.0
        self.last_inference_timestamp = 0.0
        self.min_inference_interval_sec = 0.09
        self.frame_count = 0
        self.start_fps_time = time.time()
        self.current_fps = 0.0

    def extract_frame_landmarks(self, frame_bgr):
        """
        Extracts 126D raw landmark features using MediaPipe handedness on UN-FLIPPED frames.
        Left hand -> indices 0:63
        Right hand -> indices 63:126
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)

        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)
        hands_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_count = len(results.multi_hand_landmarks)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label # 'Left' or 'Right'
                coords = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
                if label == 'Left':
                    left_hand = np.array(coords, dtype=np.float32)
                else:
                    right_hand = np.array(coords, dtype=np.float32)

        raw_126 = np.concatenate([left_hand, right_hand])
        return raw_126, hands_count

    def process_frame(self, frame_bgr):
        """
        Processes a single camera frame safely:
        1. Extract 126D landmarks with MediaPipe handedness.
        2. Append to rolling position buffer.
        3. When buffer is 30 frames, run 252D PyTorch model inference.
        4. Smooth predictions & trigger safe non-blocking background TTS.
        """
        t_now = time.time()
        self.frame_count += 1
        if t_now - self.start_fps_time >= 1.0:
            self.current_fps = self.frame_count / (t_now - self.start_fps_time)
            self.frame_count = 0
            self.start_fps_time = t_now

        raw_126, hands_count = self.extract_frame_landmarks(frame_bgr)

        if hands_count > 0:
            self.pos_buffer.append(raw_126)
        else:
            if len(self.pos_buffer) > 0:
                self.pos_buffer.popleft()
            if len(self.pos_buffer) == 0:
                self.current_stable_sentence = "No hands detected"
                self.current_confidence = 0.0
                self.prediction_history.clear()

        ran_inference = False

        if len(self.pos_buffer) == self.sequence_length and (t_now - self.last_inference_timestamp) >= self.min_inference_interval_sec:
            t_inf_start = time.time()
            try:
                # 1. Normalize 30x126 position sequence
                pos_arr = np.array(self.pos_buffer) # (30, 126)
                pos_norm = normalize_sequence_features(pos_arr) # (30, 126)

                # 2. Compute Causal First-Order Velocity (30, 126)
                vel_norm = np.zeros_like(pos_norm)
                for t in range(1, self.sequence_length):
                    vel_norm[t] = pos_norm[t] - pos_norm[t-1]

                # 3. Concatenate Position + Velocity -> (30, 252)
                feat_252 = np.concatenate([pos_norm, vel_norm], axis=-1)

                # 4. PyTorch Inference
                input_tensor = torch.tensor(feat_252, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)

                pred_label = self.idx_to_label.get(pred_idx.item(), f"Class_{pred_idx.item()}")
                confidence_val = float(conf.item())

                self.last_inference_time_ms = (time.time() - t_inf_start) * 1000.0
                self.last_inference_timestamp = t_now
                ran_inference = True

                # 5. Temporal Smoothing & Dynamic Top-1 Prediction
                self.prediction_history.append((pred_label, confidence_val))
                
                labels = [item[0] for item in self.prediction_history]
                confs = [item[1] for item in self.prediction_history]

                label_counts = Counter(labels)
                most_common_label, count = label_counts.most_common(1)[0]
                avg_conf = float(np.mean(confs))

                self.current_stable_sentence = most_common_label
                self.current_confidence = round(avg_conf * 100.0, 2)

                # Trigger Background Speech when confident and agreed over window
                if count >= self.required_agreement and avg_conf >= self.confidence_threshold:
                    self.tts.speak_async(most_common_label)

                print(f"[ENGINE DEBUG] FPS: {self.current_fps:.1f} | Latency: {self.last_inference_time_ms:.2f}ms | Hands: {hands_count} | Buffer: {len(self.pos_buffer)}/30 | Pred: {pred_label} ({confidence_val*100:.1f}%) | Stable: {self.current_stable_sentence}")

            except Exception as inf_err:
                print(f"[ENGINE ERROR] Inference processing exception: {inf_err}")

        return {
            'hands_detected': hands_count,
            'buffer_size': len(self.pos_buffer),
            'buffer_capacity': self.sequence_length,
            'stable_sentence': self.current_stable_sentence,
            'confidence': round(self.current_confidence, 2),
            'is_speaking': self.tts.get_speaking_status(),
            'fps': round(self.current_fps, 1),
            'inference_time_ms': round(self.last_inference_time_ms, 2),
            'ran_inference': ran_inference
        }

    def clear_buffer(self):
        self.pos_buffer.clear()
        self.prediction_history.clear()
        self.current_stable_sentence = "Buffer cleared"
        self.current_confidence = 0.0
        self.tts.reset_last_spoken()

    def release(self):
        if self.hands_detector:
            self.hands_detector.close()
        if self.tts:
            self.tts.stop()
