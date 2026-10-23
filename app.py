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
from src.translation_engine import translate, get_supported_languages, SUPPORTED_LANGUAGES
from src.offline_sync import get_sync_manager
from src.curriculum_assistant import generate_bilingual_lesson, list_curriculum_options
from src.worksheet_generator import generate_bilingual_worksheet, get_worksheet_by_id
from src.flashcard_generator import generate_flashcards, get_available_flashcard_packs

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
    try:
        eng = get_engine()
        model_loaded = eng is not None
    except Exception as e:
        logger.warning('Health check: engine not initialized: %s', e)
        model_loaded = False

    return jsonify({
        'status': 'ok' if model_loaded else 'degraded',
        'service': 'sign-ai',
        'model_loaded': model_loaded
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
        return redirect(url_for('join_room_page', error="Room not found. Please check the Room Code."))
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
# VERNACULAR SMART CLASSROOM REST APIS & OFFLINE SYNC
# =========================================================================
@app.route('/api/languages', methods=['GET'])
def get_languages_endpoint():
    """Returns dictionary of supported languages and metadata."""
    return jsonify({
        'status': 'success',
        'languages': get_supported_languages(),
        'default_language': 'sat'
    })

@app.route('/api/translate', methods=['POST'])
def translate_endpoint():
    """
    POST /api/translate — Real-time Vernacular Translation.
    Body: { "text": "...", "source_language": "hi", "target_language": "sat" }
    """
    data = request.json or {}
    text = data.get('text', '').strip()
    src_lang = data.get('source_language', 'hi').strip()
    target_lang = data.get('target_language', 'sat').strip()

    if not text:
        return jsonify({'error': 'Missing text parameter', 'status': 'failed'}), 400

    result = translate(text, src_lang, target_lang)
    return jsonify(result)

@app.route('/api/sync/status', methods=['GET'])
def sync_status_endpoint():
    """GET /api/sync/status — Returns offline readiness & sync details."""
    mgr = get_sync_manager()
    return jsonify(mgr.get_status())

@app.route('/api/sync/trigger', methods=['POST'])
def sync_trigger_endpoint():
    """POST /api/sync/trigger — Synchronizes curriculum, worksheets, and flashcards for offline mode."""
    mgr = get_sync_manager()
    res = mgr.trigger_synchronization()
    return jsonify(res)

@app.route('/api/curriculum/options', methods=['GET'])
def curriculum_options_endpoint():
    """GET /api/curriculum/options — Returns available grades, subjects, and topics."""
    return jsonify(list_curriculum_options())

@app.route('/api/curriculum/generate_lesson', methods=['POST'])
def generate_lesson_endpoint():
    """POST /api/curriculum/generate_lesson — Generates bilingual lesson plan."""
    data = request.json or {}
    grade = data.get('grade', 'class_1')
    subject = data.get('subject', 'fln_numeracy')
    topic_id = data.get('topic_id', 'c1_num_1_10')
    target_lang = data.get('target_language', 'sat')
    
    lesson = generate_bilingual_lesson(grade, subject, topic_id, target_lang)
    return jsonify(lesson)

@app.route('/api/worksheet/generate', methods=['POST'])
def generate_worksheet_endpoint():
    """POST /api/worksheet/generate — Generates bilingual worksheet."""
    data = request.json or {}
    grade = data.get('grade', 'Class 1')
    subject = data.get('subject', 'Numeracy')
    topic = data.get('topic', 'Numbers 1-10')
    target_lang = data.get('target_language', 'sat')
    difficulty = data.get('difficulty', 'Medium')

    worksheet = generate_bilingual_worksheet(grade, subject, topic, target_lang, difficulty)
    return jsonify(worksheet)

@app.route('/worksheet/print/<worksheet_id>')
def print_worksheet_page(worksheet_id):
    """GET /worksheet/print/<worksheet_id> — Renders printable bilingual worksheet HTML."""
    worksheet = get_worksheet_by_id(worksheet_id)
    if not worksheet:
        # Generate temporary worksheet if missing
        worksheet = generate_bilingual_worksheet("Class 1", "Numeracy", "Numbers 1-10", "sat", "Medium")
    return render_template('worksheet_print.html', worksheet=worksheet)

@app.route('/api/flashcards/packs', methods=['GET'])
def flashcard_packs_endpoint():
    """GET /api/flashcards/packs — Returns available flashcard categories."""
    return jsonify({'packs': get_available_flashcard_packs()})

@app.route('/api/flashcards/generate', methods=['POST'])
def generate_flashcards_endpoint():
    """POST /api/flashcards/generate — Generates visual flashcards with audio support."""
    data = request.json or {}
    pack_id = data.get('pack_id', 'classroom_objects')
    target_lang = data.get('target_language', 'sat')
    
    cards = generate_flashcards(pack_id, target_lang)
    return jsonify(cards)

# =========================================================================
# REAL-TIME SOCKET.IO EVENT HANDLERS
# =========================================================================
active_socket_sessions = {}

@socketio.on('connect')
def handle_connect():
    logger.info('Socket.IO client connected: %s', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid_info = active_socket_sessions.pop(request.sid, None)
    if sid_info:
        room_code = sid_info.get('room_code')
        user_name = sid_info.get('student_name')
        role = sid_info.get('role')
        logger.info('Socket.IO client disconnected: %s (%s "%s" in room %s)', request.sid, role, user_name, room_code)
        if role == 'student' and room_code and user_name:
            db.update_member_status(room_code, user_name, 'offline')
            emit('student_status_broadcast', {
                'student_name': user_name,
                'room_code': room_code,
                'status': 'offline',
                'preferred_language': sid_info.get('preferred_language', 'en')
            }, to=room_code)
    else:
        logger.info('Socket.IO client disconnected: %s', request.sid)

@socketio.on('join_room_event')
def handle_join_room_event(data):
    room_code = data.get('room_code')
    user_name = data.get('user_name')
    role = data.get('role', 'student')
    pref_lang = data.get('preferred_language', 'en')
    if room_code:
        socket_join_room(room_code)
        if user_name:
            active_socket_sessions[request.sid] = {
                'room_code': room_code,
                'student_name': user_name,
                'role': role,
                'preferred_language': pref_lang
            }
        logger.info('Socket.IO: %s "%s" joined room %s (lang: %s)', role, user_name, room_code, pref_lang)
        if role == 'student' and user_name:
            db.join_room(room_code, user_name, pref_lang)
            db.update_member_status(room_code, user_name, 'online')
            emit('student_joined_event', {
                'student_name': user_name,
                'room_code': room_code,
                'preferred_language': pref_lang,
                'status': 'online'
            }, to=room_code)
            emit('student_status_broadcast', {
                'student_name': user_name,
                'room_code': room_code,
                'status': 'online',
                'preferred_language': pref_lang
            }, to=room_code)

@socketio.on('teacher_speech_event')
def handle_teacher_speech_event(data):
    start_t = time.perf_counter()
    room_code = data.get('room_code')
    message = data.get('message', '').strip()
    teacher_name = data.get('teacher_name', 'Teacher')
    
    room = db.get_room(room_code) if room_code else None
    src_lang = data.get('source_language') or (room.get('teacher_language') if room else 'en')

    if room_code and message:
        # Create Question Entry in DB & get question_id
        q_data = db.add_question(room_code, teacher_name, message)
        
        # Determine fallback translation for global broadcast
        default_target = data.get('target_language', 'sat')
        tr_res = translate(message, src_lang, default_target)
        translated_text = tr_res.get('translated_text', message)
        measured_latency = round((time.perf_counter() - start_t) * 1000.0 + tr_res.get('latency_ms', 0.0), 1)
        
        logger.info('Teacher speech broadcast: room=%s teacher=%s src_lang=%s (lat=%.1fms)', room_code, teacher_name, src_lang, measured_latency)

        broadcast_payload = {
            'message_id': q_data['question_id'],
            'question_id': q_data['question_id'],
            'room_id': room_code,
            'room_code': room_code,
            'original_text': message,
            'question_text': message,
            'source_language': src_lang,
            'target_language': default_target,
            'translated_text': translated_text,
            'latency_ms': measured_latency,
            'backend_used': tr_res.get('backend_used', 'offline_canonical_dictionary'),
            'teacher_name': teacher_name,
            'timestamp': q_data['timestamp'],
            'created_at': q_data['created_at']
        }
        emit('teacher_question_broadcast', broadcast_payload, to=room_code)

@socketio.on('student_answer_event')
def handle_student_answer_event(data):
    start_t = time.perf_counter()
    question_id = data.get('question_id')
    room_code = data.get('room_code') or data.get('room_id')
    student_name = data.get('student_name')
    response_mode = data.get('response_mode', 'speech')  # 'speech', 'sign', 'text'
    original_pred = data.get('original_prediction') or data.get('answer', '')
    translated_text = data.get('translated_text', '')
    student_lang = data.get('target_language') or data.get('source_language') or 'en'
    confidence = data.get('confidence', 1.0)
    status = data.get('status', 'ANSWERED')

    room = db.get_room(room_code) if room_code else None
    teacher_lang = room.get('teacher_language', 'en') if room else 'en'

    # Translate student answer into Teacher Language for Teacher Dashboard
    if original_pred and not translated_text:
        src_for_trans = 'en' if response_mode == 'sign' else student_lang
        tr_res = translate(original_pred, src_for_trans, teacher_lang)
        translated_text = tr_res.get('translated_text', original_pred)
        measured_latency = round((time.perf_counter() - start_t) * 1000.0 + tr_res.get('latency_ms', 0.0), 1)
    else:
        measured_latency = round((time.perf_counter() - start_t) * 1000.0, 1)

    if question_id and room_code and student_name and original_pred:
        resp_data = db.submit_or_update_student_response(
            question_id=question_id,
            room_code=room_code,
            student_name=student_name,
            answer=original_pred,
            confidence=confidence,
            status=status,
            target_language=teacher_lang,
            translated_text=translated_text,
            original_prediction=original_pred
        )
        logger.info('Student answer received (%s): room=%s student=%s (lat=%.1fms)', response_mode, room_code, student_name, measured_latency)

        broadcast_payload = {
            'question_id': question_id,
            'room_id': room_code,
            'room_code': room_code,
            'student_name': student_name,
            'response_mode': response_mode,
            'original_prediction': original_pred,
            'predicted_answer': original_pred,
            'translated_text': translated_text,
            'teacher_language': teacher_lang,
            'target_language': teacher_lang,
            'student_language': student_lang,
            'confidence': float(confidence),
            'latency_ms': measured_latency,
            'status': status,
            'is_update': resp_data.get('is_update', False),
            'timestamp': resp_data.get('timestamp')
        }
        emit('student_answer_broadcast', broadcast_payload, to=room_code)

@socketio.on('set_language_event')
def handle_set_language_event(data):
    room_code = data.get('room_code')
    student_name = data.get('student_name')
    target_lang = data.get('target_language', 'en')
    if room_code and student_name:
        db.update_member_language(room_code, student_name, target_lang)
        logger.info('Language updated for %s in room %s: %s', student_name, room_code, target_lang)

@socketio.on('set_teacher_language_event')
def handle_set_teacher_language_event(data):
    room_code = data.get('room_code')
    teacher_lang = data.get('teacher_language', 'en')
    if room_code:
        db.update_teacher_language(room_code, teacher_lang)
        logger.info('Teacher language updated for room %s: %s', room_code, teacher_lang)
        emit('teacher_language_updated_broadcast', {
            'room_code': room_code,
            'teacher_language': teacher_lang
        }, to=room_code)

@socketio.on('student_status_event')
def handle_student_status_event(data):
    room_code = data.get('room_code')
    if room_code:
        emit('student_status_broadcast', data, to=room_code)

# ---------------------------------------------------------------------------
# RAISE HAND SOCKET & REST HANDLERS
# ---------------------------------------------------------------------------
@socketio.on('raise_hand_event')
def handle_raise_hand_event(data):
    room_code = data.get('room_code')
    student_name = data.get('student_name', 'Student')
    if room_code and student_name:
        hand_data = db.raise_hand(room_code, student_name)
        notif = db.create_notification(room_code, 'teacher', 'raise_hand', f'✋ {student_name} raised their hand', student_name)
        logger.info('Raise hand event: room=%s student=%s', room_code, student_name)
        emit('raise_hand_broadcast', hand_data, to=room_code)
        emit('notification_broadcast', notif, to=room_code)

@socketio.on('raise_hand_action_event')
def handle_raise_hand_action_event(data):
    hand_id = data.get('hand_id')
    status = data.get('status', 'acknowledged') # 'acknowledged', 'dismissed', 'answered'
    room_code = data.get('room_code')
    if hand_id and room_code:
        db.update_raise_hand_status(hand_id, status)
        logger.info('Raise hand action: hand_id=%s status=%s', hand_id, status)
        emit('raise_hand_action_broadcast', {'hand_id': hand_id, 'status': status, 'room_code': room_code}, to=room_code)

@app.route('/api/raise_hands/<room_code>')
def get_active_raise_hands_endpoint(room_code):
    hands = db.get_active_raise_hands(room_code)
    return jsonify({'raise_hands': hands})

# ---------------------------------------------------------------------------
# ANNOUNCEMENTS SOCKET & REST HANDLERS
# ---------------------------------------------------------------------------
@socketio.on('teacher_announcement_event')
def handle_teacher_announcement_event(data):
    room_code = data.get('room_code')
    teacher_name = data.get('teacher_name', 'Teacher')
    title = data.get('title', '').strip()
    content = data.get('content', '').strip()
    priority = data.get('priority', 'normal')
    is_pinned = data.get('is_pinned', 0)

    if room_code and title and content:
        ann = db.create_announcement(room_code, teacher_name, title, content, priority, is_pinned)
        notif = db.create_notification(room_code, 'student', 'announcement', f'📢 Announcement: {title}')
        logger.info('Announcement created: room=%s title=%s priority=%s', room_code, title, priority)
        emit('announcement_broadcast', ann, to=room_code)
        emit('notification_broadcast', notif, to=room_code)

@socketio.on('pin_announcement_event')
def handle_pin_announcement_event(data):
    ann_id = data.get('ann_id')
    room_code = data.get('room_code')
    if ann_id and room_code:
        db.toggle_pin_announcement(ann_id)
        emit('pin_announcement_broadcast', {'ann_id': ann_id, 'room_code': room_code}, to=room_code)

@app.route('/api/announcements/<room_code>')
def get_announcements_endpoint(room_code):
    announcements = db.get_announcements(room_code)
    return jsonify({'announcements': announcements})

# ---------------------------------------------------------------------------
# QUESTION CONTROL & RESPONSE REVIEW APIs
# ---------------------------------------------------------------------------
@socketio.on('question_control_event')
def handle_question_control_event(data):
    question_id = data.get('question_id')
    room_code = data.get('room_code')
    action = data.get('action') # 'close', 'reopen', 'repeat', 'timer'
    timer_seconds = data.get('timer_seconds', 0)

    if question_id and room_code and action:
        if action == 'close':
            db.update_question_status(question_id, 'closed')
        elif action == 'reopen':
            db.update_question_status(question_id, 'active')
        
        logger.info('Question control: qid=%s action=%s timer=%s', question_id, action, timer_seconds)
        emit('question_control_broadcast', {
            'question_id': question_id,
            'room_code': room_code,
            'action': action,
            'timer_seconds': timer_seconds
        }, to=room_code)

@app.route('/api/response/review', methods=['POST'])
def review_response_endpoint():
    data = request.json or {}
    resp_id = data.get('response_id')
    reviewed = data.get('reviewed', 1)
    if resp_id:
        db.mark_response_reviewed(resp_id, reviewed)
        return jsonify({'status': 'success'})
    return jsonify({'error': 'Invalid response_id'}), 400

# ---------------------------------------------------------------------------
# QUIZ & ASSESSMENT REST & SOCKET APIS
# ---------------------------------------------------------------------------
@app.route('/api/quiz/create', methods=['POST'])
def create_quiz_endpoint():
    data = request.json or {}
    room_code = data.get('room_code')
    teacher_name = data.get('teacher_name', 'Teacher')
    title = data.get('title', '').strip()
    description = data.get('description', '').strip()
    duration_minutes = int(data.get('duration_minutes', 10))
    questions_input = data.get('questions', [])

    if not room_code or not title:
        return jsonify({'error': 'Missing room_code or quiz title', 'status': 'failed'}), 400

    quiz = db.create_quiz(room_code, teacher_name, title, description, duration_minutes)
    for q in questions_input:
        q_text = q.get('question_text', '').strip()
        q_type = q.get('question_type', 'mcq')
        opts = json.dumps(q.get('options', []))
        ans = q.get('correct_answer', '').strip()
        pts = int(q.get('points', 1))
        if q_text:
            db.add_quiz_question(quiz['quiz_id'], q_text, q_type, opts, ans, pts)

    notif = db.create_notification(room_code, 'student', 'quiz', f'📝 New Quiz Available: {title}')
    socketio.emit('quiz_created_broadcast', quiz, to=room_code)
    socketio.emit('notification_broadcast', notif, to=room_code)
    return jsonify({'status': 'success', 'quiz': quiz})

@app.route('/api/quizzes/<room_code>')
def get_quizzes_endpoint(room_code):
    quizzes = db.get_quizzes_by_room(room_code)
    return jsonify({'quizzes': quizzes})

@app.route('/api/quiz/<quiz_id>')
def get_quiz_detail_endpoint(quiz_id):
    quiz = db.get_quiz_with_questions(quiz_id)
    if not quiz:
        return jsonify({'error': 'Quiz not found'}), 404
    return jsonify(quiz)

@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz_endpoint():
    data = request.json or {}
    quiz_id = data.get('quiz_id')
    question_id = data.get('question_id')
    room_code = data.get('room_code')
    student_name = data.get('student_name')
    response_text = data.get('response', '').strip()
    response_mode = data.get('response_mode', 'text')
    confidence = float(data.get('confidence', 1.0))
    correct_answer = data.get('correct_answer', '').strip()

    if not quiz_id or not question_id or not student_name or not response_text:
        return jsonify({'error': 'Missing required fields', 'status': 'failed'}), 400

    is_correct = 1 if (correct_answer and response_text.lower() == correct_answer.lower()) else 0
    score = 1 if is_correct else 0

    resp_id = db.submit_quiz_response(quiz_id, question_id, room_code, student_name, response_text, response_mode, confidence, is_correct, score)
    return jsonify({'status': 'success', 'response_id': resp_id, 'is_correct': is_correct, 'score': score})

@app.route('/api/quiz/results/<quiz_id>')
def get_quiz_results_endpoint(quiz_id):
    results = db.get_quiz_results(quiz_id)
    return jsonify({'results': results})

# ---------------------------------------------------------------------------
# CLASSROOM CONTENT MANAGEMENT REST APIS
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'pdf', 'ppt', 'pptx', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/content/upload', methods=['POST'])
def upload_content_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file payload provided', 'status': 'failed'}), 400

    file = request.files['file']
    room_code = request.form.get('room_code', '').strip()
    teacher_name = request.form.get('teacher_name', 'Teacher').strip()
    subject = request.form.get('subject', 'General').strip()
    module_name = request.form.get('module_name', 'Module 1').strip()
    lesson_title = request.form.get('lesson_title', 'Lesson 1').strip()

    if not file or file.filename == '':
        return jsonify({'error': 'No file selected', 'status': 'failed'}), 400

    if not room_code:
        return jsonify({'error': 'Missing room_code', 'status': 'failed'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}', 'status': 'failed'}), 400

    filename = secure_filename(file.filename)
    timestamp_prefix = int(time.time())
    unique_filename = f"{timestamp_prefix}_{filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    file.save(save_path)
    file_size = os.path.getsize(save_path)
    file_ext = filename.rsplit('.', 1)[1].lower()
    web_file_path = f"/static/uploads/{unique_filename}"

    content_item = db.add_classroom_content(
        room_code=room_code,
        teacher_name=teacher_name,
        subject=subject,
        module_name=module_name,
        lesson_title=lesson_title,
        file_name=filename,
        file_path=web_file_path,
        file_type=file_ext,
        file_size=file_size
    )

    notif = db.create_notification(room_code, 'student', 'content', f'📁 New Content Uploaded: {lesson_title} ({filename})')
    socketio.emit('content_uploaded_broadcast', content_item, to=room_code)
    socketio.emit('notification_broadcast', notif, to=room_code)

    return jsonify({'status': 'success', 'content': content_item})

@app.route('/api/content/<room_code>')
def get_classroom_content_endpoint(room_code):
    content_list = db.get_classroom_content(room_code)
    return jsonify({'content': content_list})

# ---------------------------------------------------------------------------
# CLASSROOM ANALYTICS & STUDENT PERFORMANCE REST APIS
# ---------------------------------------------------------------------------
@app.route('/api/analytics/<room_code>')
def get_classroom_analytics_endpoint(room_code):
    analytics = db.get_room_analytics(room_code)
    return jsonify(analytics)

@app.route('/api/student_performance/<room_code>/<student_name>')
def get_student_performance_endpoint(room_code, student_name):
    perf = db.get_student_performance(room_code, student_name)
    return jsonify(perf)

@app.route('/analytics/<room_code>')
def analytics_page(room_code):
    room = db.get_room(room_code)
    if not room:
        return redirect(url_for('index_page'))
    analytics = db.get_room_analytics(room_code)
    return render_template('analytics.html', room=room, analytics=analytics)

# ---------------------------------------------------------------------------
# USER ROLES & ADMIN MANAGEMENT
# ---------------------------------------------------------------------------
@app.route('/admin')
def admin_dashboard_page():
    users = db.get_all_users()
    return render_template('admin.html', users=users)

@app.route('/api/admin/system_health')
def admin_system_health():
    try:
        eng = get_engine()
        model_ready = eng is not None
    except Exception:
        model_ready = False

    return jsonify({
        'status': 'healthy' if model_ready else 'degraded',
        'model_loaded': 'best_model_252.pth' if model_ready else None,
        'torch_device': str(DEVICE),
        'db_path': db.db_path,
        'uptime_status': 'online'
    })

@app.route('/api/admin/create_user', methods=['POST'])
def admin_create_user():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'student').strip()
    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password required', 'status': 'failed'}), 400

    # In production use bcrypt/scrypt; simple string hash for baseline extension
    pass_hash = base64.b64encode(password.encode()).decode()
    user = db.create_user(username, pass_hash, role, full_name, email)
    if not user:
        return jsonify({'error': 'Username already exists', 'status': 'failed'}), 400

    return jsonify({'status': 'success', 'user': user})

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
