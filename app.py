import os
import sys
import json
import base64
import time
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.config import (
    BASE_DIR, MODELS_DIR
)
from src.realtime_engine import RealTimeInferenceEngine
from src.video_inference import run_video_file_inference

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global Real-Time Inference Engine & Camera Stream
engine = None
camera_capture = None
latest_status = {
    'hands_detected': 0,
    'buffer_size': 0,
    'buffer_capacity': 30,
    'stable_sentence': 'Waiting for camera...',
    'confidence': 0.0,
    'is_speaking': False,
    'fps': 0.0,
    'inference_time_ms': 0.0
}

def get_engine():
    global engine
    if engine is None:
        engine = RealTimeInferenceEngine()
    return engine

def get_camera():
    global camera_capture
    if camera_capture is None or not camera_capture.isOpened():
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, tmp = cap.read()
                if ret and tmp is not None:
                    camera_capture = cap
                    print(f"Successfully opened server-side computer camera index {idx}")
                    break
                cap.release()
    return camera_capture

def generate_mjpeg_stream():
    global latest_status
    eng = get_engine()
    cap = get_camera()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    if cap is None or not cap.isOpened():
        latest_status['stable_sentence'] = "Camera Hardware Unavailable"
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No Camera Hardware Found", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands:
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # 1. MediaPipe landmark extraction on UN-FLIPPED frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                display_frame = frame.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 2. Mirror flip display frame AFTER drawing landmarks for user view
                display_frame = cv2.flip(display_frame, 1)

                # 3. Process 252D ML inference engine safely (Exception guarded)
                try:
                    status = eng.process_frame(frame)
                    latest_status.update(status)
                except Exception as engine_err:
                    print(f"[STREAM ENGINE ERROR]: {engine_err}")

                # 4. Draw HUD Overlay on video stream
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 60), (15, 20, 25), -1)
                cv2.putText(display_frame, f"Recognized: {latest_status['stable_sentence']}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Confidence: {latest_status['confidence']}% | FPS: {latest_status['fps']} | Latency: {latest_status['inference_time_ms']}ms", (15, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                ret_code, jpeg = cv2.imencode('.jpg', display_frame)
                if not ret_code:
                    continue

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as stream_err:
                print(f"[STREAM LOOP ERROR]: {stream_err}")
                time.sleep(0.01)

@app.route('/')
def index():
    metrics_path = os.path.join(MODELS_DIR, 'velocity_experiment_metrics.json')
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    return render_template('index.html', metrics=metrics)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera_status', methods=['GET'])
def camera_status_endpoint():
    global latest_status
    return jsonify(latest_status)

@app.route('/api/process_frame', methods=['POST'])
def process_frame_endpoint():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image frame provided', 'status': 'failed'}), 400

    img_data = data['image']
    if ',' in img_data:
        img_data = img_data.split(',')[1]

    try:
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Failed to decode image frame: {e}', 'status': 'failed'}), 400

    if frame is None:
        return jsonify({'error': 'Decoded image frame is empty', 'status': 'failed'}), 400

    eng = get_engine()
    res = eng.process_frame(frame)
    res['status'] = 'success'
    latest_status.update(res)

    return jsonify(res)

@app.route('/api/clear', methods=['POST'])
def clear_endpoint():
    global latest_status
    eng = get_engine()
    eng.clear_buffer()
    latest_status['stable_sentence'] = "Buffer cleared"
    latest_status['confidence'] = 0.0
    return jsonify({'status': 'success', 'message': 'Buffer cleared successfully.'})

@app.route('/api/predict_video', methods=['POST'])
def predict_video_endpoint():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided', 'status': 'failed'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected video file', 'status': 'failed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = run_video_file_inference(filepath)
        return jsonify({
            'status': 'success',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'processing_time_sec': result['processing_time_sec']
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_path = os.path.join(MODELS_DIR, 'velocity_experiment_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Metrics not available'}), 404

if __name__ == '__main__':
    get_engine()
    get_camera()
    print("Flask app running at http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
