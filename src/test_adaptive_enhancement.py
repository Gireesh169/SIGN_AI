import os
import sys
import time
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR
from realtime_engine import RealTimeInferenceEngine
from image_enhancer import adaptive_enhance_frame, measure_frame_quality

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_adaptive_enhancement_benchmark():
    print("\n" + "="*70)
    print("=== STARTING ADAPTIVE LOW-LIGHT ENHANCEMENT BENCHMARK ===")
    print("="*70)

    # Load real extracted RGB frames from Frames_Sentence_Level
    frames_dir = os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus', 'Frames_Sentence_Level', 'what happened', '1')
    if not os.path.exists(frames_dir):
        # Fallback to any available frame folder in Frames_Sentence_Level
        all_cls = [d for d in os.listdir(os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus', 'Frames_Sentence_Level')) if not d.startswith('.')]
        if all_cls:
            cls_p = os.path.join(BASE_DIR, 'ISL_CSLRT_Corpus', 'Frames_Sentence_Level', all_cls[0])
            sub_p = [d for d in os.listdir(cls_p) if not d.startswith('.')][0]
            frames_dir = os.path.join(cls_p, sub_p)

    test_frames = []
    if os.path.exists(frames_dir):
        img_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:50]
        for img_f in img_files:
            img_path = os.path.join(frames_dir, img_f)
            img = cv2.imread(img_path)
            if img is not None:
                # Dim frame to low-light intensity (30% original brightness)
                dimmed = (img.astype(np.float32) * 0.30).astype(np.uint8)
                test_frames.append(dimmed)

    print(f"Loaded {len(test_frames)} real dimmed low-light test frames from {frames_dir}.")

    engine = RealTimeInferenceEngine()

    # ----------------------------------------------------
    # SETUP A: Original Low-Light Frames (Unenhanced)
    # ----------------------------------------------------
    print("\n---> Running Setup A: Original Low-Light Frames (Unenhanced)")
    t0 = time.time()
    hands_detected_A = 0
    missing_hands_A = 0

    for frame in test_frames:
        # Pass directly without enhancement
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = engine.hands_detector.process(rgb)
        if res.multi_hand_landmarks and len(res.multi_hand_landmarks) > 0:
            hands_detected_A += 1
        else:
            missing_hands_A += 1

    time_A = (time.time() - t0) * 1000.0 / len(test_frames)
    detection_rate_A = (hands_detected_A / len(test_frames)) * 100.0
    missing_rate_A = (missing_hands_A / len(test_frames)) * 100.0

    # ----------------------------------------------------
    # SETUP B: Adaptive Enhanced Low-Light Frames
    # ----------------------------------------------------
    print("\n---> Running Setup B: Adaptive Enhanced Frames (Gamma + CLAHE)")
    t0 = time.time()
    hands_detected_B = 0
    missing_hands_B = 0
    enhanced_count = 0

    for frame in test_frames:
        enhanced_frame, was_enhanced = adaptive_enhance_frame(frame)
        if was_enhanced: enhanced_count += 1
        rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        res = engine.hands_detector.process(rgb)
        if res.multi_hand_landmarks and len(res.multi_hand_landmarks) > 0:
            hands_detected_B += 1
        else:
            missing_hands_B += 1

    time_B = (time.time() - t0) * 1000.0 / len(test_frames)
    detection_rate_B = (hands_detected_B / len(test_frames)) * 100.0
    missing_rate_B = (missing_hands_B / len(test_frames)) * 100.0

    engine.release()

    # ----------------------------------------------------
    # BENCHMARK REPORT
    # ----------------------------------------------------
    print("\n" + "="*85)
    print("=== ADAPTIVE LOW-LIGHT ENHANCEMENT BENCHMARK REPORT ===")
    print("="*85)
    header = f"| {'Setup / Pipeline':<28} | {'Detection Rate':<16} | {'Missing Hand Rate':<18} | {'Avg Latency':<12} |"
    print(header)
    print("|" + "-"*30 + "|" + "-"*18 + "|" + "-"*20 + "|" + "-"*14 + "|")

    print(f"| {'Setup A: Original Frames':<28} | {detection_rate_A:>14.1f}% | {missing_rate_A:>16.1f}% | {time_A:>10.2f} ms |")
    print(f"| {'Setup B: Adaptive Enhanced':<28} | {detection_rate_B:>14.1f}% | {missing_rate_B:>16.1f}% | {time_B:>10.2f} ms |")
    print("="*85 + "\n")

    report = {
        'setup_A_original': {
            'detection_rate_pct': round(detection_rate_A, 2),
            'missing_hand_rate_pct': round(missing_rate_A, 2),
            'avg_latency_ms': round(time_A, 2)
        },
        'setup_B_adaptive_enhanced': {
            'detection_rate_pct': round(detection_rate_B, 2),
            'missing_hand_rate_pct': round(missing_rate_B, 2),
            'avg_latency_ms': round(time_B, 2),
            'frames_enhanced_pct': round((enhanced_count / len(test_frames)) * 100.0, 2)
        }
    }

    report_path = os.path.join(RESULTS_DIR, 'adaptive_enhancement_metrics.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Benchmark Report Saved To: {report_path}\n")

if __name__ == '__main__':
    run_adaptive_enhancement_benchmark()
