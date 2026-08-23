import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_DIR, DATA_DIR, VIDEOS_DIR, FRAMES_DIR, LABEL_MAPPING_FILE,
    SEQUENCE_LENGTH, normalize_sequence_features
)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Output File Paths (New non-overwriting files)
HAND_FILE = os.path.join(DATA_DIR, 'x_hand_features_252.npy')
FACE_FILE = os.path.join(DATA_DIR, 'x_face_features_compact.npy')
POSE_FILE = os.path.join(DATA_DIR, 'x_pose_features_compact.npy')
Y_MULTI_FILE = os.path.join(DATA_DIR, 'y_labels_multimodal.npy')
REPORT_FILE = os.path.join(RESULTS_DIR, 'multimodal_features_report.json')

def extract_face_compact_features(face_landmarks):
    """
    Extracts a compact 16D facial expression & orientation feature vector.
    If face_landmarks is None, returns 16D zeros.
    """
    if face_landmarks is None:
        return np.zeros(16, dtype=np.float32), False
        
    lms = face_landmarks.landmark
    # 1. Eyebrow movement (4D)
    # Left eyebrow 70 vs Left eye 159; Right eyebrow 300 vs Right eye 386
    left_eyebrow = np.array([lms[70].x, lms[70].y, lms[70].z])
    left_eye_top = np.array([lms[159].x, lms[159].y, lms[159].z])
    right_eyebrow = np.array([lms[300].x, lms[300].y, lms[300].z])
    right_eye_top = np.array([lms[386].x, lms[386].y, lms[386].z])
    
    eb_left_dist = np.linalg.norm(left_eyebrow - left_eye_top)
    eb_right_dist = np.linalg.norm(right_eyebrow - right_eye_top)
    eyebrow_feats = np.array([eb_left_dist, eb_right_dist, left_eyebrow[1], right_eyebrow[1]], dtype=np.float32)
    
    # 2. Eye openness (4D)
    # Left eye vertical 159 vs 145, horizontal 33 vs 133
    eye_left_v = np.linalg.norm(np.array([lms[159].x, lms[159].y]) - np.array([lms[145].x, lms[145].y]))
    eye_left_h = np.linalg.norm(np.array([lms[33].x, lms[33].y]) - np.array([lms[133].x, lms[133].y]))
    eye_left_ratio = eye_left_v / (eye_left_h + 1e-6)
    
    eye_right_v = np.linalg.norm(np.array([lms[386].x, lms[386].y]) - np.array([lms[374].x, lms[374].y]))
    eye_right_h = np.linalg.norm(np.array([lms[362].x, lms[362].y]) - np.array([lms[263].x, lms[263].y]))
    eye_right_ratio = eye_right_v / (eye_right_h + 1e-6)
    eye_feats = np.array([eye_left_v, eye_left_ratio, eye_right_v, eye_right_ratio], dtype=np.float32)
    
    # 3. Mouth openness & lip shape (4D)
    # Upper lip 13 vs Lower lip 14, Left corner 61 vs Right corner 291
    lip_top = np.array([lms[13].x, lms[13].y, lms[13].z])
    lip_bottom = np.array([lms[14].x, lms[14].y, lms[14].z])
    mouth_left = np.array([lms[61].x, lms[61].y, lms[61].z])
    mouth_right = np.array([lms[291].x, lms[291].y, lms[291].z])
    
    mouth_height = np.linalg.norm(lip_top - lip_bottom)
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    mouth_ratio = mouth_height / (mouth_width + 1e-6)
    mouth_feats = np.array([mouth_height, mouth_width, mouth_ratio, lip_top[1] - lip_bottom[1]], dtype=np.float32)
    
    # 4. Head orientation (4D)
    nose = np.array([lms[1].x, lms[1].y, lms[1].z])
    chin = np.array([lms[152].x, lms[152].y, lms[152].z])
    head_vec = chin - nose
    head_feats = np.array([head_vec[0], head_vec[1], head_vec[2], np.linalg.norm(head_vec)], dtype=np.float32)
    
    face_16 = np.concatenate([eyebrow_feats, eye_feats, mouth_feats, head_feats])
    return face_16, True

def extract_pose_compact_features(pose_landmarks):
    """
    Extracts a compact 24D upper-body pose feature vector.
    Shoulders (11,12), Elbows (13,14), Wrists (15,16), Hips (23,24).
    If pose_landmarks is None, returns 24D zeros.
    """
    if pose_landmarks is None:
        return np.zeros(24, dtype=np.float32), False
        
    lms = pose_landmarks.landmark
    # Target upper-body indices
    target_indices = [11, 12, 13, 14, 15, 16, 23, 24]
    pts = []
    for idx in target_indices:
        pts.append([lms[idx].x, lms[idx].y, lms[idx].z])
    pts = np.array(pts, dtype=np.float32) # (8, 3) -> 24D coords
    
    # Shoulder center normalization
    shoulder_center = (pts[0] + pts[1]) / 2.0
    pts_norm = pts - shoulder_center
    
    # Shoulder width scale normalization
    shoulder_width = np.linalg.norm(pts[0] - pts[1])
    if shoulder_width > 1e-4:
        pts_norm /= shoulder_width
        
    pose_24 = pts_norm.flatten() # 24D
    return pose_24, True

def run_3branch_extraction():
    print("\n" + "="*70)
    print("=== STARTING 3-BRANCH MEDIAPIPE MULTIMODAL FEATURE EXTRACTION ===")
    print("="*70)
    
    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    label_to_idx = {v: int(k) for k, v in label_mapping.items()}
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    classes = sorted([d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d)) and not d.startswith('.')])
    
    all_hand_seqs = []
    all_face_seqs = []
    all_pose_seqs = []
    all_labels = []
    
    total_frames_processed = 0
    missing_hand_frames = 0
    missing_face_frames = 0
    missing_pose_frames = 0
    valid_sequences_count = 0
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV')
    
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
            
            # Read video frames via OpenCV
            cap = cv2.VideoCapture(v_path)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None: break
                    frames.append(frame)
                cap.release()
                
            # Fallback to pre-extracted RGB frame folder if video decoders fail
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
                
            # Uniformly sample exactly 30 frames
            total_f = len(frames)
            indices = np.linspace(0, total_f - 1, 30, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            
            hand_126_seq = []
            face_16_seq = []
            pose_24_seq = []
            
            for frame_bgr in sampled_frames:
                total_frames_processed += 1
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                # 1. Hand Landmarks (126D)
                left_hand = np.zeros(63, dtype=np.float32)
                right_hand = np.zeros(63, dtype=np.float32)
                has_hand = False
                
                if results.left_hand_landmarks:
                    coords = [val for lm in results.left_hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
                    left_hand = np.array(coords, dtype=np.float32)
                    has_hand = True
                if results.right_hand_landmarks:
                    coords = [val for lm in results.right_hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
                    right_hand = np.array(coords, dtype=np.float32)
                    has_hand = True
                    
                if not has_hand: missing_hand_frames += 1
                raw_126 = np.concatenate([left_hand, right_hand])
                hand_126_seq.append(raw_126)
                
                # 2. Face Landmarks (16D)
                face_16, has_face = extract_face_compact_features(results.face_landmarks)
                if not has_face: missing_face_frames += 1
                face_16_seq.append(face_16)
                
                # 3. Pose Landmarks (24D)
                pose_24, has_pose = extract_pose_compact_features(results.pose_landmarks)
                if not has_pose: missing_pose_frames += 1
                pose_24_seq.append(pose_24)
                
            # Convert arrays
            hand_arr = np.array(hand_126_seq, dtype=np.float32) # (30, 126)
            face_arr = np.array(face_16_seq, dtype=np.float32)   # (30, 16)
            pose_arr = np.array(pose_24_seq, dtype=np.float32)   # (30, 24)
            
            # Compute 252D Hand Features (126D position + 126D velocity)
            pos_norm = normalize_sequence_features(hand_arr)
            vel_norm = np.zeros_like(pos_norm)
            for t in range(1, 30):
                vel_norm[t] = pos_norm[t] - pos_norm[t-1]
            hand_252 = np.concatenate([pos_norm, vel_norm], axis=-1) # (30, 252)
            
            all_hand_seqs.append(hand_252)
            all_face_seqs.append(face_arr)
            all_pose_seqs.append(pose_arr)
            all_labels.append(cls_idx)
            valid_sequences_count += 1

    holistic.close()
    
    all_hand_seqs = np.array(all_hand_seqs, dtype=np.float32)
    all_face_seqs = np.array(all_face_seqs, dtype=np.float32)
    all_pose_seqs = np.array(all_pose_seqs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    # Save files
    np.save(HAND_FILE, all_hand_seqs)
    np.save(FACE_FILE, all_face_seqs)
    np.save(POSE_FILE, all_pose_seqs)
    np.save(Y_MULTI_FILE, all_labels)
    
    report = {
        "hand_feature_dimension": int(all_hand_seqs.shape[2]),
        "face_feature_dimension": int(all_face_seqs.shape[2]),
        "pose_feature_dimension": int(all_pose_seqs.shape[2]),
        "number_of_valid_sequences": int(valid_sequences_count),
        "total_frames_processed": int(total_frames_processed),
        "missing_landmark_statistics": {
            "missing_hand_frames": int(missing_hand_frames),
            "missing_hand_pct": round((missing_hand_frames / max(1, total_frames_processed)) * 100.0, 2),
            "missing_face_frames": int(missing_face_frames),
            "missing_face_pct": round((missing_face_frames / max(1, total_frames_processed)) * 100.0, 2),
            "missing_pose_frames": int(missing_pose_frames),
            "missing_pose_pct": round((missing_pose_frames / max(1, total_frames_processed)) * 100.0, 2)
        },
        "saved_files": {
            "hand_features": HAND_FILE,
            "face_features": FACE_FILE,
            "pose_features": POSE_FILE,
            "labels": Y_MULTI_FILE
        }
    }
    
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*70)
    print("=== 3-BRANCH MEDIAPIPE MULTIMODAL FEATURE EXTRACTION REPORT ===")
    print("="*70)
    print(f"Hand Feature Dimension    : {report['hand_feature_dimension']}D (30 x 252)")
    print(f"Face Feature Dimension    : {report['face_feature_dimension']}D (30 x 16)")
    print(f"Pose Feature Dimension    : {report['pose_feature_dimension']}D (30 x 24)")
    print(f"Number of Valid Sequences : {report['number_of_valid_sequences']}")
    print(f"Missing Hand Frames       : {report['missing_landmark_statistics']['missing_hand_pct']}%")
    print(f"Missing Face Frames       : {report['missing_landmark_statistics']['missing_face_pct']}%")
    print(f"Missing Pose Frames       : {report['missing_landmark_statistics']['missing_pose_pct']}%")
    print(f"Feature Report Saved To   : {REPORT_FILE}")
    print("="*70 + "\n")

if __name__ == '__main__':
    run_3branch_extraction()
