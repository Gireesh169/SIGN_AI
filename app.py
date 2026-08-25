import os
import sys
import json
import base64
import time
import logging
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from flask_socketio import SocketIO, emit, join_room as socket_join_room, leave_room as socket_leave_room
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Structured logging — INFO level, consistent format
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sign_ai')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.config import (
    BASE_DIR, MODELS_DIR
)
from src.realtime_engine import RealTimeInferenceEngine
from src.video_inference import run_video_file_inference
from src.database import ClassroomDatabase
from src.conversation_memory import ConversationMemoryStore

# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# SECRET_KEY — read from environment; never commit real keys
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'development-only-secret-change-in-production')

# Upload folder — auto-created; limited to 16 MB per request
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Ensure uploads directory exists at startup
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info('Upload directory ready: %s', app.config['UPLOAD_FOLDER'])

# ---------------------------------------------------------------------------
# Flask-SocketIO — CORS origin configurable via environment variable
# ---------------------------------------------------------------------------
cors_origins = os.environ.get('SOCKETIO_CORS_ORIGINS', '*')
socketio = SocketIO(app, cors_allowed_origins=cors_origins, async_mode='eventlet')

# ---------------------------------------------------------------------------
# Database & conversation store — schema auto-initialized on construction
# ---------------------------------------------------------------------------
db = ClassroomDatabase()
conversation_store = ConversationMemoryStore()
logger.info('Database initialized successfully')

# ---------------------------------------------------------------------------
# Cache-control headers for all responses
# ---------------------------------------------------------------------------
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ---------------------------------------------------------------------------
# Global Real-Time Inference Engine — loaded ONCE (lazy singleton)
# model.eval() and torch.no_grad() are used inside RealTimeInferenceEngine
# ---------------------------------------------------------------------------
_engine = None

def get_engine():
    """
    Returns the singleton RealTimeInferenceEngine.
    Loads best_model_252.pth exactly once at first call.
    Raises on model file not found — logged as CRITICAL.
    """
    global _engine
    if _engine is None:
        try:
            _engine = RealTimeInferenceEngine()
            logger.info('Model loaded successfully: best_model_252.pth')
        except FileNotFoundError as e:
            logger.critical('Model file not found — inference unavailable: %s', e)
            raise
        except Exception as e:
            logger.critical('Failed to load inference engine: %s', e)
            raise
    return _engine

# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------
@app.route('/health')
def health_check():
    """
    GET /health — Returns service status and model load state.
    Safe: does not expose internal paths or secrets.
    """
    model_loaded = _engine is not None
    if model_loaded:
        return jsonify({
            'status': 'ok',
            'service': 'sign-ai',
            'model_loaded': True
        }), 200
    else:
        return jsonify({
            'status': 'degraded',
            'service': 'sign-ai',
            'model_loaded': False
        }), 200

# =========================================================================
# CLASSROOM PAGE ROUTES
# =========================================================================
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/create-room')
def create_room_page():
    return render_template('create_room.html')

@app.route('/api/create_room', methods=['POST'])
def create_room_endpoint():
    teacher_name = request.form.get('teacher_name', '').strip()
    room_name = request.form.get('room_name', '').strip()
    subject = request.form.get('subject', '').strip()
    description = request.form.get('description', '')
    max_students = int(request.form.get('max_students', 30))
    # Do NOT log password
    password = request.form.get('password', '')

    room = db.create_room(teacher_name, room_name, subject, description, max_students, password)
    logger.info('Room created: %s by teacher: %s (subject: %s)', room['room_code'], teacher_name, subject)
    return redirect(url_for('teacher_dashboard_page', room_code=room['room_code']))

@app.route('/join-room')
def join_room_page():
    error = request.args.get('error')
    return render_template('join_room.html', error=error)

@app.route('/api/join_room', methods=['POST'])
def join_room_endpoint():
    student_name = request.form.get('student_name', '').strip()
    room_code = request.form.get('room_code', '').strip().upper()
    # Do NOT log password
    password = request.form.get('password', '')

    room = db.get_room(room_code)
    if not room:
        return redirect(url_for('join_room_page', error="Invalid Room Code. Please check and try again."))

    if room['password'] and room['password'] != password:
        return redirect(url_for('join_room_page', error="Incorrect Room Password."))

    db.join_room(room_code, student_name)
    logger.info('Student joined room: %s (room: %s)', student_name, room_code)
    return redirect(url_for('student_dashboard_page', room_code=room_code, student_name=student_name))

@app.route('/teacher/<room_code>')
def teacher_dashboard_page(room_code):
    room = db.get_room(room_code)
    if not room:
        return redirect(url_for('index_page'))
    members = db.get_room_members(room_code)
    questions = db.get_questions_by_room(room_code)
    pending_requests = db.get_student_requests(room_code, status="pending")
    return render_template('teacher_dashboard.html', room=room, members=members, questions=questions, pending_requests=pending_requests)

@app.route('/student/<room_code>')
def student_dashboard_page(room_code):
    student_name = request.args.get('student_name', 'Student')
    room = db.get_room(room_code)
    if not room:
        return redirect(url_for('index_page'))
    latest_question = db.get_latest_question(room_code)
    activity_logs = db.get_student_activity_logs(room_code, student_name)
    return render_template('student_dashboard.html', room=room, student_name=student_name, latest_question=latest_question, activity_logs=activity_logs)

@app.route('/history')
def history_page():
    room_code = request.args.get('room_code')
    student_name = request.args.get('student_name')
    logs = db.get_filtered_history(room_code=room_code, student_name=student_name)
    return render_template('history.html', logs=logs)

@app.route('/report/<room_code>')
def report_page(room_code):
    room = db.get_room(room_code)
    if not room:
        return redirect(url_for('index_page'))
    members = db.get_room_members(room_code)
    questions = db.get_questions_by_room(room_code)
    sign_logs = db.get_filtered_history(room_code=room_code)
    return render_template('report.html', room=room, members=members, questions=questions, sign_logs=sign_logs)

# =========================================================================
# QUESTION & ANSWER REST APIS
# =========================================================================
@app.route('/api/latest_question/<room_code>')
def latest_question_endpoint(room_code):
    q = db.get_latest_question(room_code)
    return jsonify(q or {})

@app.route('/api/question_responses/<question_id>')
def question_responses_endpoint(question_id):
    responses = db.get_responses_by_question(question_id)
    return jsonify({'responses': responses})

@app.route('/api/student_activity/<room_code>/<student_name>')
def student_activity_endpoint(room_code, student_name):
    logs = db.get_student_activity_logs(room_code, student_name)
    return jsonify({'activity_logs': logs})

# =========================================================================
# REAL-TIME SOCKET.IO EVENT HANDLERS
# =========================================================================
@socketio.on('connect')
def handle_connect():
    logger.info('Socket.IO client connected: %s', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Socket.IO client disconnected: %s', request.sid)

@socketio.on('join_room_event')
def handle_join_room_event(data):
    room_code = data.get('room_code')
    user_name = data.get('user_name')
    role = data.get('role', 'student')
    if room_code:
        socket_join_room(room_code)
        logger.info('Socket.IO: %s "%s" joined room %s', role, user_name, room_code)
        if role == 'student':
            emit('student_joined_event', {'student_name': user_name, 'room_code': room_code}, to=room_code)

@socketio.on('teacher_speech_event')
def handle_teacher_speech_event(data):
    room_code = data.get('room_code')
    message = data.get('message')
    teacher_name = data.get('teacher_name', 'Teacher')

    if room_code and message:
        # Create Question Entry in DB & get question_id
        q_data = db.add_question(room_code, teacher_name, message)
        logger.info('Teacher speech broadcast: room=%s teacher=%s', room_code, teacher_name)
        emit('teacher_question_broadcast', q_data, to=room_code)

@socketio.on('student_answer_event')
def handle_student_answer_event(data):
    question_id = data.get('question_id')
    room_code = data.get('room_code')
    student_name = data.get('student_name')
    answer = data.get('answer')
    confidence = data.get('confidence', 1.0)
    status = data.get('status', 'ANSWERED')

    if question_id and room_code and student_name and answer:
        resp_data = db.submit_or_update_student_response(question_id, room_code, student_name, answer, confidence, status)
        logger.info('Student answer received: room=%s student=%s', room_code, student_name)
        emit('student_answer_broadcast', resp_data, to=room_code)

@socketio.on('student_status_event')
def handle_student_status_event(data):
    room_code = data.get('room_code')
    if room_code:
        emit('student_status_broadcast', data, to=room_code)

@app.route('/api/resolve_request', methods=['POST'])
def resolve_request_endpoint():
    data = request.json or {}
    req_id = data.get('request_id')
    if req_id:
        db.resolve_student_request(req_id)
        return jsonify({'status': 'success'})
    return jsonify({'error': 'Invalid request_id'}), 400

# =========================================================================
# REAL-TIME SIGN RECOGNITION API
# =========================================================================

@app.route('/predict_sequence', methods=['POST'])
def predict_sequence_endpoint():
    """
    POST /predict_sequence — Accepts a pre-extracted 30-frame × 126-feature landmark
    sequence from the browser (MediaPipe already ran client-side), runs BiLSTM+GRU
    inference, and returns prediction results.

    Expected JSON body:
        { "sequence": [[f0, f1, ..., f125], ..., [f0, ..., f125]] }   # shape (30, 126)

    Error handling:
      400 — Missing/invalid sequence payload
      500 — Inference engine failure (safe message only, no traceback exposed)
    """
    logger.info('predict_sequence request from %s', request.remote_addr)

    # --- Input validation ---
    data = request.json or {}
    sequence = data.get('sequence', None)

    if sequence is None:
        logger.warning('predict_sequence: missing "sequence" field')
        return jsonify({
            'status': 'failed',
            'error': 'Missing "sequence" field. Expected {"sequence": [[...126 values...] x 30]}'
        }), 400

    if not isinstance(sequence, list):
        logger.warning('predict_sequence: sequence is not a list, got %s', type(sequence).__name__)
        return jsonify({'status': 'failed', 'error': 'Invalid sequence payload: must be a JSON array'}), 400

    if len(sequence) != 30:
        logger.warning('predict_sequence: expected 30 frames, got %d', len(sequence))
        return jsonify({
            'status': 'failed',
            'error': f'Invalid sequence length: expected 30 frames, got {len(sequence)}'
        }), 400

    try:
        arr = np.array(sequence, dtype=np.float32)
        if arr.shape != (30, 126):
            raise ValueError(f'Shape mismatch: expected (30, 126), got {arr.shape}')
    except Exception as shape_err:
        logger.warning('predict_sequence: shape validation failed: %s', shape_err)
        return jsonify({'status': 'failed', 'error': f'Invalid sequence shape: {shape_err}'}), 400

    # --- Inference ---
    try:
        eng = get_engine()
        result = eng.predict_sequence(arr)
    except FileNotFoundError:
        logger.error('Inference engine unavailable: model file not found')
        return jsonify({'error': 'Inference service unavailable', 'status': 'failed'}), 500
    except Exception as inf_err:
        logger.error('predict_sequence inference error: %s', inf_err, exc_info=True)
        return jsonify({'error': 'Inference failed. Please try again.', 'status': 'failed'}), 500

    logger.info(
        'predict_sequence result: text=%r conf=%.1f stable=%s uncertain=%s',
        result.get('predicted_text', ''),
        result.get('confidence', 0.0),
        result.get('stable'),
        result.get('uncertain')
    )

    return jsonify({
        'status': 'success',
        'predicted_text': result.get('predicted_text', ''),
        'confidence': result.get('confidence', 0.0),
        'top_predictions': result.get('top_predictions', []),
        'stable': result.get('stable', False),
        'uncertain': result.get('uncertain', True),
        'people_count': 0,
        'signer_status': 'Frontend MediaPipe',
        'buffer_size': 30,
        'sequence_ready': True,
        'backend_received_shape': '(1, 30, 252)',
        'model_loaded': 'best_model_252.pth',
        'fps': 0.0
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    POST /predict — Accepts a base64-encoded image, runs BiLSTM+GRU inference,
    returns prediction results.

    Error handling:
      400 — No image provided or image could not be decoded
      500 — Inference engine failure (safe message only, no traceback exposed)

    NOTE: The student dashboard uses /predict_sequence instead, which accepts
    pre-extracted MediaPipe landmark sequences directly from the browser.
    """
    logger.info('Prediction request received from %s', request.remote_addr)

    # --- Input validation ---
    data = request.json or {}
    image_data = data.get('image', None)
    if not image_data:
        logger.warning('Prediction request missing image field (keys present: %s)', list(data.keys()))
        return jsonify({'error': 'No image provided', 'status': 'failed',
                        'hint': 'Use /predict_sequence to send MediaPipe landmark sequences directly'}), 400

    # --- Decode base64 image ---
    try:
        if isinstance(image_data, str) and ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('cv2.imdecode returned None')
    except Exception as decode_err:
        logger.warning('Image decode failed: %s', decode_err)
        return jsonify({'error': 'Invalid or corrupted image data', 'status': 'failed'}), 400

    # --- Inference ---
    try:
        eng = get_engine()
        result = eng.process_frame(frame)
    except FileNotFoundError:
        logger.error('Inference engine unavailable: model file not found')
        return jsonify({'error': 'Inference service unavailable', 'status': 'failed'}), 500
    except Exception as inf_err:
        logger.error('Inference error: %s', inf_err, exc_info=True)
        return jsonify({'error': 'Inference failed. Please try again.', 'status': 'failed'}), 500

    return jsonify({
        'status': 'success',
        'predicted_text': result.get('predicted_text', ''),
        'confidence': result.get('confidence', 0.0),
        'top_predictions': result.get('top_predictions', []),
        'stable': result.get('stable', False),
        'uncertain': result.get('uncertain', True),
        'people_count': result.get('people_count', 0),
        'signer_status': result.get('signer_status', 'No signer detected'),
        'buffer_size': result.get('buffer_size', 0),
        'sequence_ready': result.get('sequence_ready', False),
        'backend_received_shape': result.get('backend_received_shape', '(1, 0, 252)'),
        'model_loaded': result.get('model_loaded', 'best_model_252.pth'),
        'fps': result.get('fps', 0.0)
    })

# =========================================================================
# APPLICATION ENTRY POINT
# =========================================================================
if __name__ == '__main__':
    # Pre-load model at startup so first request is instant
    logger.info('=== SIGN-AI Smart Inclusive Classroom Platform ===')
    logger.info('Loading inference engine at startup...')
    try:
        get_engine()
    except Exception:
        logger.warning('Inference engine failed to load at startup — /predict will be unavailable until resolved')

    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'
    logger.info('Starting server on %s:%d', host, port)
    logger.info('Service URLs:')
    logger.info('  Local:  http://127.0.0.1:%d', port)
    logger.info('  Health: http://127.0.0.1:%d/health', port)
    socketio.run(app, host=host, port=port, debug=False)
