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
from image_enhancer import adaptive_enhance_frame, ENABLE_LOW_LIGHT_ENHANCEMENT

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

# =========================================================================
# ADJUSTABLE STABILIZATION THRESHOLDS & STABILITY PARAMETERS (HIGH-SPEED REAL-TIME)
# =========================================================================
SLIDING_WINDOW_SIZE = 30       # Landmark sequence length per inference window
HISTORY_MAX_LEN = 5           # Size of rolling prediction history buffer
REQUIRED_VOTES = 2            # Minimum matching predictions required for temporal consensus
CONFIDENCE_THRESHOLD = 0.35   # Minimum Top-1 probability (35%) to accept a prediction
PROB_MARGIN_THRESHOLD = 0.08  # Minimum difference between Top-1 and Top-2 probabilities
SPEECH_COOLDOWN_SEC = 2.5     # Cooldown interval (in seconds) between spoken sentences
UNCERTAIN_FALLBACK_TEXT = "Not confident. Please repeat the sign."

def evaluate_stabilized_prediction(top_preds, prediction_history):
    if not top_preds or len(top_preds) == 0:
        return UNCERTAIN_FALLBACK_TEXT, 0.0, False, True, 0.0, 0
        
    top1_pred = top_preds[0]['sentence']
    top1_conf = top_preds[0]['confidence'] / 100.0  # Fraction 0.0 .. 1.0
    
    top2_conf = (top_preds[1]['confidence'] / 100.0) if len(top_preds) > 1 else 0.0
    prob_margin = top1_conf - top2_conf
    
    is_conf_valid = top1_conf >= CONFIDENCE_THRESHOLD
    is_margin_valid = prob_margin >= PROB_MARGIN_THRESHOLD
    
    # Store candidate in history if confident and distinct
    if is_conf_valid and is_margin_valid:
        prediction_history.append(top1_pred)
    else:
        prediction_history.append("UNCERTAIN")
        
    # Majority Temporal Voting
    valid_votes = [item for item in prediction_history if item != "UNCERTAIN"]
    if len(valid_votes) > 0:
        counts = Counter(valid_votes)
        most_common_label, vote_count = counts.most_common(1)[0]
    else:
        most_common_label, vote_count = UNCERTAIN_FALLBACK_TEXT, 0
        
    has_consensus = (vote_count >= REQUIRED_VOTES)
    is_uncertain = (not is_conf_valid) or (not is_margin_valid)
    is_stable = has_consensus and (not is_uncertain)
    
    predicted_text = most_common_label if is_stable else UNCERTAIN_FALLBACK_TEXT
    
    return (
        predicted_text,
        round(top1_conf * 100.0, 2),
        is_stable,
        is_uncertain,
        round(prob_margin * 100.0, 2),
        vote_count
    )

class RealTimeInferenceEngine:
    def __init__(self, model_path=MODEL_252_PATH, label_map_path=LABEL_MAPPING_FILE):
        self.device = DEVICE
        self.sequence_length = SLIDING_WINDOW_SIZE
        self.feature_dim = 252

        # Initialize High-Speed MediaPipe Hands Detector (model_complexity=0 for 3x speedup)
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
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

        print(f"[ENGINE STARTUP] Model:          {os.path.basename(model_path)}")
        print(f"[ENGINE STARTUP] Label mapping:  {label_map_path}")
        print(f"[ENGINE STARTUP] Classes:        {self.num_classes}")
        print(f"[ENGINE STARTUP] Inference engine: READY")

        # Initialize TTS Worker
        self.tts = TTSEngineWorker()

        # Session Buffers & Multi-Person Tracking State
        self.pos_buffer = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=HISTORY_MAX_LEN)
        self.top_predictions = []
        self.current_predicted_text = "Capturing sign... 0 / 30 frames"
        self.current_confidence = 0.0
        self.current_stable = False
        self.current_uncertain = True
        self.current_prob_margin = 0.0
        self.current_vote_count = 0
        self.last_spoken_time = 0.0
        
        self.tracked_person_center = None
        self.people_count = 0
        self.signer_status = "No signer detected"

        self.last_inference_time_ms = 0.0
        self.last_inference_timestamp = 0.0
        self.min_inference_interval_sec = 0.02  # 20ms interval (up to 50 FPS)
        self.frame_count = 0
        self.start_fps_time = time.time()
        self.current_fps = 0.0

    def extract_frame_landmarks(self, frame_bgr):
        """
        Extracts 126D raw landmark features for the Primary Active Signer using MediaPipe handedness.
        Prevents landmark mixing when multiple people appear in camera view.
        Applies adaptive low-light enhancement when dark or low contrast frames are detected.
        """
        proc_frame, was_enhanced = adaptive_enhance_frame(frame_bgr)
        if was_enhanced and not getattr(self, '_logged_enhancement', False):
            print("[ENHANCER] Low-light frame detected -> Adaptive CLAHE & Gamma applied")
            self._logged_enhancement = True
        elif not was_enhanced:
            self._logged_enhancement = False

        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)

        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)
        hands_count = 0
        self.people_count = 0
        selected_signer_center = None

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_count = len(results.multi_hand_landmarks)

            if hands_count == 1:
                self.people_count = 1
                lm = results.multi_hand_landmarks[0].landmark
                selected_signer_center = (lm[9].x, lm[9].y)
                self.signer_status = "Single signer detected"
            elif hands_count >= 2:
                # Spatial clustering check based on inter-wrist distance
                lm1 = results.multi_hand_landmarks[0].landmark
                lm2 = results.multi_hand_landmarks[1].landmark
                wrist_dist = np.sqrt((lm1[0].x - lm2[0].x)**2 + (lm1[0].y - lm2[0].y)**2)

                if wrist_dist <= 0.45:
                    self.people_count = 1
                    selected_signer_center = ((lm1[9].x + lm2[9].x)/2.0, (lm1[9].y + lm2[9].y)/2.0)
                    self.signer_status = "Single signer detected"
                else:
                    self.people_count = 2
                    self.signer_status = "Multiple people detected — tracking selected signer"

                    c1 = (lm1[9].x, lm1[9].y)
                    c2 = (lm2[9].x, lm2[9].y)

                    score1 = 1.0 - (abs(c1[0] - 0.5) + abs(c1[1] - 0.5))
                    score2 = 1.0 - (abs(c2[0] - 0.5) + abs(c2[1] - 0.5))

                    if self.tracked_person_center is not None:
                        d1 = np.sqrt((c1[0] - self.tracked_person_center[0])**2 + (c1[1] - self.tracked_person_center[1])**2)
                        d2 = np.sqrt((c2[0] - self.tracked_person_center[0])**2 + (c2[1] - self.tracked_person_center[1])**2)
                        score1 += (1.0 - d1) * 2.0
                        score2 += (1.0 - d2) * 2.0

                    selected_signer_center = c1 if score1 >= score2 else c2

            # Check if tracked active signer switched abruptly (> 0.35 screen distance)
            if selected_signer_center is not None and self.tracked_person_center is not None:
                shift_dist = np.sqrt((selected_signer_center[0] - self.tracked_person_center[0])**2 + (selected_signer_center[1] - self.tracked_person_center[1])**2)
                if shift_dist > 0.35:
                    print("[ENGINE TRACKING] Active signer changed — Resetting 30-frame sequence buffer")
                    self.pos_buffer.clear()
                    self.prediction_history.clear()

            if selected_signer_center is not None:
                self.tracked_person_center = selected_signer_center

            # Process landmarks strictly for the primary selected signer
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lm = hand_landmarks.landmark
                hand_center = (lm[9].x, lm[9].y)

                # Skip secondary person hands
                if self.people_count >= 2 and selected_signer_center is not None:
                    dist_to_selected = np.sqrt((hand_center[0] - selected_signer_center[0])**2 + (hand_center[1] - selected_signer_center[1])**2)
                    if dist_to_selected > 0.35:
                        continue

                label = handedness.classification[0].label # 'Left' or 'Right'
                coords = [val for lm_pt in hand_landmarks.landmark for val in (lm_pt.x, lm_pt.y, lm_pt.z)]
                if label == 'Left':
                    left_hand = np.array(coords, dtype=np.float32)
                else:
                    right_hand = np.array(coords, dtype=np.float32)
        else:
            self.people_count = 0
            self.signer_status = "No signer detected"

        raw_126 = np.concatenate([left_hand, right_hand])
        return raw_126, hands_count

    def process_frame(self, frame_bgr):
        """
        Processes a single camera frame safely:
        1. Extract 126D landmarks with MediaPipe handedness.
        2. Append to rolling position buffer.
        3. When buffer is 30 frames, run 252D PyTorch model inference.
        4. Apply Prediction Stabilization System & trigger non-blocking TTS.
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
                self.current_predicted_text = "No hands detected"
                self.current_confidence = 0.0
                self.current_stable = False
                self.current_uncertain = True
                self.prediction_history.clear()

        ran_inference = False
        buf_len = len(self.pos_buffer)

        if buf_len < self.sequence_length and hands_count > 0:
            self.current_predicted_text = f"Capturing sign... {buf_len} / 30 frames"

        if buf_len == self.sequence_length and (t_now - self.last_inference_timestamp) >= self.min_inference_interval_sec:
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
                    top_k_probs, top_k_indices = torch.topk(probs, min(5, self.num_classes), dim=1)

                top_preds = []
                for p, idx in zip(top_k_probs[0], top_k_indices[0]):
                    lbl = self.idx_to_label.get(idx.item(), f"Class_{idx.item()}")
                    top_preds.append({
                        'sentence': lbl,
                        'confidence': round(float(p.item()) * 100.0, 2)
                    })

                self.top_predictions = top_preds
                self.last_inference_time_ms = (time.time() - t_inf_start) * 1000.0
                self.last_inference_timestamp = t_now
                ran_inference = True

                # 5. Prediction Stabilization System & Margin Checking
                p_text, p_conf, p_stable, p_uncertain, p_margin, p_votes = evaluate_stabilized_prediction(
                    top_preds, self.prediction_history
                )
                
                self.current_predicted_text = p_text
                self.current_confidence = p_conf
                self.current_stable = p_stable
                self.current_uncertain = p_uncertain
                self.current_prob_margin = p_margin
                self.current_vote_count = p_votes

                # 6. Trigger Speech output only when stable, certain, and cooldown expired
                if p_stable and not p_uncertain and (t_now - self.last_spoken_time >= SPEECH_COOLDOWN_SEC):
                    self.tts.speak_async(p_text)
                    self.last_spoken_time = t_now

                print(f"[STABILIZER] Text: '{p_text}' | Conf: {p_conf}% | Margin: {p_margin}% | Stable: {p_stable} | Uncertain: {p_uncertain} | Votes: {p_votes}/{REQUIRED_VOTES}")

            except Exception as inf_err:
                print(f"[ENGINE ERROR] Inference processing exception: {inf_err}")

        return {
            'predicted_text': self.current_predicted_text,
            'confidence': self.current_confidence,
            'top_predictions': getattr(self, 'top_predictions', []),
            'stable': self.current_stable,
            'uncertain': self.current_uncertain,
            'prob_margin': self.current_prob_margin,
            'vote_count': self.current_vote_count,
            'hands_detected': hands_count,
            'people_count': self.people_count,
            'signer_status': self.signer_status,
            'buffer_size': buf_len,
            'buffer_capacity': self.sequence_length,
            'sequence_ready': (buf_len == 30),
            'backend_received_shape': f"(1, {buf_len}, 252)",
            'pos_features_dim': 126,
            'vel_features_dim': 126,
            'combined_features_dim': 252,
            'model_loaded': 'best_model_252.pth',
            'is_speaking': self.tts.get_speaking_status(),
            'fps': round(self.current_fps, 1),
            'inference_time_ms': round(self.last_inference_time_ms, 2),
            'ran_inference': ran_inference
        }

    def predict_sequence(self, raw_sequence_30x126):
        """
        Runs 252D inference on a pre-collected (30, 126) or (30, 252) array with Stabilization.
        """
        t_start = time.time()
        arr = np.array(raw_sequence_30x126, dtype=np.float32)
        if arr.shape == (30, 126):
            pos_norm = normalize_sequence_features(arr)
            vel_norm = np.zeros_like(pos_norm)
            for t in range(1, 30):
                vel_norm[t] = pos_norm[t] - pos_norm[t-1]
            feat_252 = np.concatenate([pos_norm, vel_norm], axis=-1)
        elif arr.shape == (30, 252):
            feat_252 = arr
        else:
            raise ValueError(f"Expected shape (30, 126) or (30, 252), got {arr.shape}")

        input_tensor = torch.tensor(feat_252, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            top_k_probs, top_k_indices = torch.topk(probs, min(3, self.num_classes), dim=1)

        top_preds = []
        for p, idx in zip(top_k_probs[0], top_k_indices[0]):
            lbl = self.idx_to_label.get(idx.item(), f"Class_{idx.item()}")
            top_preds.append({
                'sentence': lbl,
                'confidence': round(float(p.item()) * 100.0, 2)
            })

        p_text, p_conf, p_stable, p_uncertain, p_margin, p_votes = evaluate_stabilized_prediction(
            top_preds, self.prediction_history
        )

        t_ms = (time.time() - t_start) * 1000.0
        return {
            'predicted_text': p_text,
            'confidence': p_conf,
            'top_predictions': top_preds,
            'stable': p_stable,
            'uncertain': p_uncertain,
            'prob_margin': p_margin,
            'vote_count': p_votes,
            'feature_shape': [30, 252],
            'inference_time_ms': round(t_ms, 2)
        }

    def clear_buffer(self):
        self.pos_buffer.clear()
        self.prediction_history.clear()
        self.current_predicted_text = UNCERTAIN_FALLBACK_TEXT
        self.current_confidence = 0.0
        self.current_stable = False
        self.current_uncertain = True
        self.tts.reset_last_spoken()

    def release(self):
        if self.hands_detector:
            self.hands_detector.close()
        if self.tts:
            self.tts.stop()

