import os
import sys
import json
import base64
import time
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from flask_socketio import SocketIO, emit, join_room as socket_join_room, leave_room as socket_leave_room
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.config import (
    BASE_DIR, MODELS_DIR
)
from src.realtime_engine import RealTimeInferenceEngine
from src.video_inference import run_video_file_inference
from src.database import ClassroomDatabase
from src.conversation_memory import ConversationMemoryStore

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign_ai_classroom_secret_2026'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
db = ClassroomDatabase()
conversation_store = ConversationMemoryStore()

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Global Real-Time Inference Engine
engine = None
def get_engine():
    global engine
    if engine is None:
        engine = RealTimeInferenceEngine()
    return engine

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
    teacher_name = request.form.get('teacher_name')
    room_name = request.form.get('room_name')
    subject = request.form.get('subject')
    description = request.form.get('description', '')
    max_students = int(request.form.get('max_students', 30))
    password = request.form.get('password', '')

    room = db.create_room(teacher_name, room_name, subject, description, max_students, password)
    return redirect(url_for('teacher_dashboard_page', room_code=room['room_code']))

@app.route('/join-room')
def join_room_page():
    error = request.args.get('error')
    return render_template('join_room.html', error=error)

@app.route('/api/join_room', methods=['POST'])
def join_room_endpoint():
    student_name = request.form.get('student_name')
    room_code = request.form.get('room_code', '').strip().upper()
    password = request.form.get('password', '')

    room = db.get_room(room_code)
    if not room:
        return redirect(url_for('join_room_page', error="Invalid Room Code. Please check and try again."))

    if room['password'] and room['password'] != password:
        return redirect(url_for('join_room_page', error="Incorrect Room Password."))

    db.join_room(room_code, student_name)
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
@socketio.on('join_room_event')
def handle_join_room_event(data):
    room_code = data.get('room_code')
    user_name = data.get('user_name')
    role = data.get('role', 'student')
    if room_code:
        socket_join_room(room_code)
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
# REAL-TIME SIGN RECOGNITION API (/predict)
# =========================================================================
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.json or {}
        image_data = data.get('image', None)
        if not image_data:
            return jsonify({'error': 'No image provided', 'status': 'failed'}), 400

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        eng = get_engine()
        result = eng.process_frame(frame)

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

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    get_engine()
    print("🚀 Smart Inclusive Classroom Platform running at http://127.0.0.1:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
