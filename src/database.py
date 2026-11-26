import os
import sys
import sqlite3
import random
import string
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, DATA_DIR

CLASSROOM_DB_PATH = os.path.join(DATA_DIR, 'inclusive_classroom.db')
os.makedirs(DATA_DIR, exist_ok=True)

class ClassroomDatabase:
    """
    SQLite Database Manager for Smart Inclusive Classroom Platform.
    Handles virtual classrooms, room codes (SIGN-XXXX), question-answer tracking,
    student activity logs, request queues, and historical analytics.
    """
    def __init__(self, db_path=CLASSROOM_DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. Rooms Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rooms (
                room_code TEXT PRIMARY KEY,
                teacher_name TEXT NOT NULL,
                room_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                description TEXT,
                max_students INTEGER DEFAULT 30,
                password TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            )
        ''')

        # 2. Room Members Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS room_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                student_name TEXT NOT NULL,
                status TEXT DEFAULT 'online',
                joined_at TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 3. Class Sessions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS class_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                session_start TEXT NOT NULL,
                session_end TEXT,
                attendance_count INTEGER DEFAULT 0,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 4. Teacher Messages Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teacher_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                message_text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 5. Student Sign Logs Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_sign_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                student_name TEXT NOT NULL,
                detected_sentence TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 6. Student Requests Queue Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                student_name TEXT NOT NULL,
                detected_sentence TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT DEFAULT 'pending',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 7. Questions Table (New)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT PRIMARY KEY,
                room_code TEXT NOT NULL,
                teacher_name TEXT NOT NULL,
                question_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 8. Student Responses Table (New - Linked via question_id)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                room_code TEXT NOT NULL,
                student_name TEXT NOT NULL,
                predicted_answer TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT DEFAULT 'ANSWERED',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions (question_id),
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 9. Student Activity Logs Table (New)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                room_code TEXT NOT NULL,
                question_id TEXT,
                action_type TEXT NOT NULL,
                message TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 10. Vernacular Curriculum Lessons Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS curriculum_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grade TEXT NOT NULL,
                subject TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                target_language TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # 11. Vernacular Worksheets Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS worksheets (
                worksheet_id TEXT PRIMARY KEY,
                grade TEXT NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                target_language TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # 12. Vernacular Flashcards Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flashcards (
                pack_id TEXT NOT NULL,
                target_language TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # 13. Offline Sync Logs Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sync_timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT
            )
        ''')

        # 14. Users Table (Role-Based Authentication)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student',
                full_name TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            )
        ''')

        # 15. Raise Hands Queue Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raise_hands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                student_name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                raised_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 16. Announcements Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS announcements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                teacher_name TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                priority TEXT DEFAULT 'normal',
                is_pinned INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 17. Real-Time Classroom Notifications Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                recipient_role TEXT NOT NULL,
                student_name TEXT,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 18. Quizzes Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quizzes (
                quiz_id TEXT PRIMARY KEY,
                room_code TEXT NOT NULL,
                teacher_name TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                duration_minutes INTEGER DEFAULT 10,
                status TEXT DEFAULT 'draft',
                created_at TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # 19. Quiz Questions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id TEXT NOT NULL,
                question_text TEXT NOT NULL,
                question_type TEXT DEFAULT 'mcq',
                options_json TEXT,
                correct_answer TEXT,
                points INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id)
            )
        ''')

        # 20. Quiz Responses Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id TEXT NOT NULL,
                question_id INTEGER NOT NULL,
                room_code TEXT NOT NULL,
                student_name TEXT NOT NULL,
                response TEXT NOT NULL,
                response_mode TEXT DEFAULT 'text',
                confidence REAL DEFAULT 1.0,
                is_correct INTEGER DEFAULT 0,
                score INTEGER DEFAULT 0,
                submitted_at TEXT NOT NULL,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id),
                FOREIGN KEY (question_id) REFERENCES quiz_questions (id)
            )
        ''')

        # 21. Classroom Content Materials Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classroom_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT NOT NULL,
                teacher_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                module_name TEXT NOT NULL,
                lesson_title TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (room_code) REFERENCES rooms (room_code)
            )
        ''')

        # Vernacular Smart Classroom Schema Migrations (Safe column additions)
        migrations = [
            ("rooms", "source_language", "TEXT DEFAULT 'hi'"),
            ("rooms", "target_language", "TEXT DEFAULT 'sat'"),
            ("rooms", "offline_mode", "INTEGER DEFAULT 0"),
            ("questions", "source_language", "TEXT DEFAULT 'hi'"),
            ("questions", "target_language", "TEXT DEFAULT 'sat'"),
            ("questions", "translated_text", "TEXT"),
            ("questions", "latency_ms", "REAL DEFAULT 0.0"),
            ("questions", "status", "TEXT DEFAULT 'active'"),
            ("questions", "timer_seconds", "INTEGER DEFAULT 0"),
            ("questions", "closed_at", "TEXT"),
            ("student_responses", "response_mode", "TEXT DEFAULT 'speech'"),
            ("student_responses", "latency_ms", "REAL DEFAULT 0.0"),
            ("student_responses", "reviewed", "INTEGER DEFAULT 0"),
            ("room_members", "preferred_language", "TEXT DEFAULT 'en'"),
            ("student_responses", "original_prediction", "TEXT"),
            ("student_responses", "target_language", "TEXT DEFAULT 'en'"),
            ("student_responses", "translated_text", "TEXT"),
            ("student_sign_logs", "target_language", "TEXT DEFAULT 'en'"),
            ("student_sign_logs", "translated_text", "TEXT"),
            ("student_activity_logs", "target_language", "TEXT DEFAULT 'en'"),
            ("student_activity_logs", "translated_text", "TEXT")
        ]
        for tbl, col, col_def in migrations:
            try:
                cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        conn.close()

    def generate_room_code(self):
        while True:
            digits = ''.join(random.choices(string.digits, k=4))
            code = f"SIGN-{digits}"
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT room_code FROM rooms WHERE room_code = ?", (code,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return code

    def create_room(self, teacher_name, room_name, subject, description="", max_students=30, password="", teacher_language="en"):
        code = self.generate_room_code()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO rooms (room_code, teacher_name, room_name, subject, description, max_students, password, created_at, status, source_language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
        ''', (code, teacher_name, room_name, subject, description, max_students, password, created_at, teacher_language))
        conn.commit()
        conn.close()

        return {
            'room_code': code,
            'teacher_name': teacher_name,
            'room_name': room_name,
            'subject': subject,
            'description': description,
            'max_students': max_students,
            'teacher_language': teacher_language,
            'source_language': teacher_language,
            'created_at': created_at,
            'status': 'active'
        }

    def get_room(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT room_code, teacher_name, room_name, subject, description, max_students, password, created_at, status, source_language FROM rooms WHERE room_code = ?", (room_code,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'room_code': row[0],
                'teacher_name': row[1],
                'room_name': row[2],
                'subject': row[3],
                'description': row[4],
                'max_students': row[5],
                'password': row[6],
                'created_at': row[7],
                'status': row[8],
                'teacher_language': row[9] if len(row) > 9 and row[9] else 'en',
                'source_language': row[9] if len(row) > 9 and row[9] else 'en'
            }
        return None

    def join_room(self, room_code, student_name, preferred_language='en'):
        room = self.get_room(room_code)
        if not room:
            return False, "Room Code not found"

        joined_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO room_members (room_code, student_name, status, joined_at, preferred_language)
            VALUES (?, ?, 'online', ?, ?)
        ''', (room_code, student_name, joined_at, preferred_language))
        conn.commit()
        conn.close()

        return True, "Joined successfully"

    def update_member_language(self, room_code, student_name, preferred_language):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE room_members
            SET preferred_language = ?
            WHERE room_code = ? AND student_name = ?
        ''', (preferred_language, room_code, student_name))
        conn.commit()
        conn.close()

    def update_teacher_language(self, room_code, teacher_language):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE rooms
            SET teacher_language = ?
            WHERE room_code = ?
        ''', (teacher_language, room_code))
        conn.commit()
        conn.close()

    def update_member_status(self, room_code, student_name, status):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE room_members
            SET status = ?
            WHERE room_code = ? AND student_name = ?
        ''', (status, room_code, student_name))
        conn.commit()
        conn.close()

    def get_room_members(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, student_name, status, joined_at, preferred_language FROM room_members WHERE room_code = ? ORDER BY id ASC", (room_code,))
        rows = cursor.fetchall()
        conn.close()

        return [{
            'id': r[0],
            'student_name': r[1],
            'status': r[2],
            'joined_at': r[3],
            'preferred_language': r[4] if len(r) > 4 and r[4] else 'en'
        } for r in rows]


    # =========================================================================
    # QUESTION & ANSWER SYSTEM METHODS
    # =========================================================================
    def add_question(self, room_code, teacher_name, question_text):
        q_id = f"q_{int(datetime.now().timestamp() * 1000)}"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts = datetime.now().strftime("%H:%M:%S")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO questions (question_id, room_code, teacher_name, question_text, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (q_id, room_code, teacher_name, question_text, created_at))

        cursor.execute('''
            INSERT INTO teacher_messages (room_code, message_text, timestamp)
            VALUES (?, ?, ?)
        ''', (room_code, question_text, ts))

        conn.commit()
        conn.close()

        return {
            'question_id': q_id,
            'room_code': room_code,
            'teacher_name': teacher_name,
            'question_text': question_text,
            'created_at': created_at,
            'timestamp': ts
        }

    def get_latest_question(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT question_id, room_code, teacher_name, question_text, created_at FROM questions WHERE room_code = ? ORDER BY created_at DESC LIMIT 1", (room_code,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'question_id': row[0],
                'room_code': row[1],
                'teacher_name': row[2],
                'question_text': row[3],
                'created_at': row[4]
            }
        return None

    def get_questions_by_room(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT question_id, room_code, teacher_name, question_text, created_at FROM questions WHERE room_code = ? ORDER BY created_at DESC", (room_code,))
        rows = cursor.fetchall()
        conn.close()

        members = self.get_room_members(room_code)
        total_members = len(members)

        questions = []
        for r in rows:
            q_id = r[0]
            responses = self.get_responses_by_question(q_id)
            answered_count = len(responses)

            questions.append({
                'question_id': q_id,
                'room_code': r[1],
                'teacher_name': r[2],
                'question_text': r[3],
                'created_at': r[4],
                'answered_count': answered_count,
                'total_members': total_members,
                'responses': responses
            })
        return questions

    def submit_or_update_student_response(self, question_id, room_code, student_name, answer, confidence, status="ANSWERED", target_language="en", translated_text="", original_prediction=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        orig_pred = original_prediction if original_prediction is not None else answer

        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if student already submitted an answer for this question_id (Duplicate Prevention)
        cursor.execute("SELECT id FROM student_responses WHERE question_id = ? AND student_name = ?", (question_id, student_name))
        row = cursor.fetchone()

        is_update = False
        if row:
            # UPDATE existing answer
            is_update = True
            cursor.execute('''
                UPDATE student_responses
                SET predicted_answer = ?, original_prediction = ?, target_language = ?, translated_text = ?, confidence = ?, status = ?, updated_at = ?
                WHERE id = ?
            ''', (answer, orig_pred, target_language, translated_text, float(confidence), status, ts, row[0]))
        else:
            # INSERT new answer
            cursor.execute('''
                INSERT INTO student_responses (question_id, room_code, student_name, predicted_answer, original_prediction, target_language, translated_text, confidence, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (question_id, room_code, student_name, answer, orig_pred, target_language, translated_text, float(confidence), status, ts, ts))

        conn.commit()
        conn.close()

        # Also Log Activity
        action_type = "ANSWER_UPDATED" if is_update else "ANSWER_SENT"
        self.log_student_activity(student_name, room_code, question_id, action_type, answer, confidence, target_language=target_language, translated_text=translated_text)

        return {
            'question_id': question_id,
            'room_code': room_code,
            'student_name': student_name,
            'predicted_answer': answer,
            'original_prediction': orig_pred,
            'target_language': target_language,
            'translated_text': translated_text,
            'confidence': float(confidence),
            'status': status,
            'is_update': is_update,
            'timestamp': ts
        }

    def get_responses_by_question(self, question_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, question_id, room_code, student_name, predicted_answer, confidence, status, updated_at, original_prediction, target_language, translated_text FROM student_responses WHERE question_id = ? ORDER BY updated_at DESC", (question_id,))
        rows = cursor.fetchall()
        conn.close()

        return [{
            'id': r[0],
            'question_id': r[1],
            'room_code': r[2],
            'student_name': r[3],
            'predicted_answer': r[4],
            'confidence': float(r[5]),
            'status': r[6],
            'updated_at': r[7],
            'original_prediction': r[8] if len(r) > 8 and r[8] else r[4],
            'target_language': r[9] if len(r) > 9 and r[9] else 'en',
            'translated_text': r[10] if len(r) > 10 and r[10] else ''
        } for r in rows]

    def log_student_activity(self, student_name, room_code, question_id, action_type, message, confidence=0.0, target_language="en", translated_text=""):
        ts = datetime.now().strftime("%H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO student_activity_logs (student_name, room_code, question_id, action_type, message, confidence, timestamp, target_language, translated_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (student_name, room_code, question_id, action_type, message, float(confidence), ts, target_language, translated_text))
        conn.commit()
        conn.close()

    def get_student_activity_logs(self, room_code, student_name, limit=20):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.id, a.student_name, a.room_code, a.question_id, a.action_type, a.message, a.confidence, a.timestamp, q.question_text, a.target_language, a.translated_text
            FROM student_activity_logs a
            LEFT JOIN questions q ON a.question_id = q.question_id
            WHERE a.room_code = ? AND a.student_name = ?
            ORDER BY a.id DESC LIMIT ?
        ''', (room_code, student_name, limit))
        rows = cursor.fetchall()
        conn.close()

        return [{
            'id': r[0],
            'student_name': r[1],
            'room_code': r[2],
            'question_id': r[3],
            'action_type': r[4],
            'message': r[5],
            'confidence': float(r[6]),
            'timestamp': r[7],
            'question_text': r[8] or "Classroom Instruction",
            'target_language': r[9] if len(r) > 9 and r[9] else 'en',
            'translated_text': r[10] if len(r) > 10 and r[10] else ''
        } for r in rows]

    def add_teacher_message(self, room_code, message_text):
        ts = datetime.now().strftime("%H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO teacher_messages (room_code, message_text, timestamp)
            VALUES (?, ?, ?)
        ''', (room_code, message_text, ts))
        conn.commit()
        conn.close()

        return {'room_code': room_code, 'message_text': message_text, 'timestamp': ts}

    def get_teacher_messages(self, room_code, limit=30):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, message_text, timestamp FROM teacher_messages WHERE room_code = ? ORDER BY id ASC LIMIT ?", (room_code, limit))
        rows = cursor.fetchall()
        conn.close()

        return [{'id': r[0], 'message_text': r[1], 'timestamp': r[2]} for r in rows]

    def get_filtered_history(self, room_code=None, student_name=None, limit=100):
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT room_code, student_name, predicted_answer, confidence, status, updated_at, original_prediction, target_language, translated_text FROM student_responses WHERE 1=1"
        params = []

        if room_code:
            query += " AND room_code = ?"
            params.append(room_code)
        if student_name:
            query += " AND student_name LIKE ?"
            params.append(f"%{student_name}%")

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [{
            'room_code': r[0],
            'student_name': r[1],
            'detected_sentence': r[2],
            'confidence': float(r[3]),
            'status': r[4],
            'timestamp': r[5],
            'original_prediction': r[6] if len(r) > 6 and r[6] else r[2],
            'target_language': r[7] if len(r) > 7 and r[7] else 'en',
            'translated_text': r[8] if len(r) > 8 and r[8] else ''
        } for r in rows]


    def add_student_sign_log(self, room_code, student_name, sentence, confidence, status="CONFIRMED"):
        ts = datetime.now().strftime("%H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO student_sign_logs (room_code, student_name, detected_sentence, confidence, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (room_code, student_name, sentence, float(confidence), status, ts))

        cursor.execute('''
            INSERT INTO student_requests (room_code, student_name, detected_sentence, confidence, status, timestamp)
            VALUES (?, ?, ?, ?, 'pending', ?)
        ''', (room_code, student_name, sentence, float(confidence), ts))

        conn.commit()
        conn.close()

        return {
            'room_code': room_code,
            'student_name': student_name,
            'detected_sentence': sentence,
            'confidence': float(confidence),
            'status': status,
            'timestamp': ts
        }

    def get_student_requests(self, room_code, status="pending"):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, student_name, detected_sentence, confidence, status, timestamp FROM student_requests WHERE room_code = ? AND status = ? ORDER BY id ASC", (room_code, status))
        rows = cursor.fetchall()
        conn.close()

        return [{
            'id': r[0],
            'student_name': r[1],
            'detected_sentence': r[2],
            'confidence': float(r[3]),
            'status': r[4],
            'timestamp': r[5]
        } for r in rows]

    def resolve_student_request(self, request_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE student_requests SET status = 'resolved' WHERE id = ?", (request_id,))
        conn.commit()
        conn.close()

    # =========================================================================
    # 1. RAISE HAND LIFECYCLE
    # =========================================================================
    def raise_hand(self, room_code, student_name):
        raised_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        # Reset any existing active hand raise for this student in this room
        cursor.execute("UPDATE raise_hands SET status = 'superseded' WHERE room_code = ? AND student_name = ? AND status = 'pending'", (room_code, student_name))
        cursor.execute('''
            INSERT INTO raise_hands (room_code, student_name, status, raised_at)
            VALUES (?, ?, 'pending', ?)
        ''', (room_code, student_name, raised_at))
        hand_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {
            'id': hand_id,
            'room_code': room_code,
            'student_name': student_name,
            'status': 'pending',
            'raised_at': raised_at
        }

    def get_active_raise_hands(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, room_code, student_name, status, raised_at FROM raise_hands WHERE room_code = ? AND status IN ('pending', 'acknowledged') ORDER BY id ASC", (room_code,))
        rows = cursor.fetchall()
        conn.close()
        return [{
            'id': r[0],
            'room_code': r[1],
            'student_name': r[2],
            'status': r[3],
            'raised_at': r[4]
        } for r in rows]

    def update_raise_hand_status(self, hand_id, new_status):
        resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if new_status in ['dismissed', 'answered'] else None
        conn = self._get_connection()
        cursor = conn.cursor()
        if resolved_at:
            cursor.execute("UPDATE raise_hands SET status = ?, resolved_at = ? WHERE id = ?", (new_status, resolved_at, hand_id))
        else:
            cursor.execute("UPDATE raise_hands SET status = ? WHERE id = ?", (new_status, hand_id))
        conn.commit()
        conn.close()
        return True

    # =========================================================================
    # 2. ANNOUNCEMENTS ENGINE
    # =========================================================================
    def create_announcement(self, room_code, teacher_name, title, content, priority='normal', is_pinned=0):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO announcements (room_code, teacher_name, title, content, priority, is_pinned, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (room_code, teacher_name, title, content, priority, int(is_pinned), created_at))
        ann_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {
            'id': ann_id,
            'room_code': room_code,
            'teacher_name': teacher_name,
            'title': title,
            'content': content,
            'priority': priority,
            'is_pinned': int(is_pinned),
            'created_at': created_at
        }

    def get_announcements(self, room_code, limit=20):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, room_code, teacher_name, title, content, priority, is_pinned, created_at FROM announcements WHERE room_code = ? ORDER BY is_pinned DESC, id DESC LIMIT ?", (room_code, limit))
        rows = cursor.fetchall()
        conn.close()
        return [{
            'id': r[0],
            'room_code': r[1],
            'teacher_name': r[2],
            'title': r[3],
            'content': r[4],
            'priority': r[5],
            'is_pinned': r[6],
            'created_at': r[7]
        } for r in rows]

    def toggle_pin_announcement(self, ann_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE announcements SET is_pinned = CASE WHEN is_pinned = 1 THEN 0 ELSE 1 END WHERE id = ?", (ann_id,))
        conn.commit()
        conn.close()
        return True

    # =========================================================================
    # 3. NOTIFICATIONS SYSTEM
    # =========================================================================
    def create_notification(self, room_code, recipient_role, type_, message, student_name=None):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO notifications (room_code, recipient_role, student_name, type, message, is_read, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
        ''', (room_code, recipient_role, student_name, type_, message, created_at))
        notif_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {
            'id': notif_id,
            'room_code': room_code,
            'recipient_role': recipient_role,
            'student_name': student_name,
            'type': type_,
            'message': message,
            'created_at': created_at
        }

    def get_notifications(self, room_code, recipient_role='teacher', limit=15):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, room_code, recipient_role, student_name, type, message, is_read, created_at FROM notifications WHERE room_code = ? AND recipient_role = ? ORDER BY id DESC LIMIT ?", (room_code, recipient_role, limit))
        rows = cursor.fetchall()
        conn.close()
        return [{
            'id': r[0],
            'room_code': r[1],
            'recipient_role': r[2],
            'student_name': r[3],
            'type': r[4],
            'message': r[5],
            'is_read': r[6],
            'created_at': r[7]
        } for r in rows]

    # =========================================================================
    # 4. QUESTION CONTROL & RESPONSE REVIEW
    # =========================================================================
    def update_question_status(self, question_id, new_status):
        closed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if new_status == 'closed' else None
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE questions SET status = ?, closed_at = ? WHERE question_id = ?", (new_status, closed_at, question_id))
        conn.commit()
        conn.close()
        return True

    def mark_response_reviewed(self, response_id, reviewed_flag=1):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE student_responses SET reviewed = ? WHERE id = ?", (reviewed_flag, response_id))
        conn.commit()
        conn.close()
        return True

    # =========================================================================
    # 5. QUIZ & ASSESSMENT ENGINE
    # =========================================================================
    def create_quiz(self, room_code, teacher_name, title, description="", duration_minutes=10):
        quiz_id = "QUIZ-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO quizzes (quiz_id, room_code, teacher_name, title, description, duration_minutes, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
        ''', (quiz_id, room_code, teacher_name, title, description, duration_minutes, created_at))
        conn.commit()
        conn.close()
        return {
            'quiz_id': quiz_id,
            'room_code': room_code,
            'teacher_name': teacher_name,
            'title': title,
            'description': description,
            'duration_minutes': duration_minutes,
            'status': 'active',
            'created_at': created_at
        }

    def add_quiz_question(self, quiz_id, question_text, question_type='mcq', options_json="[]", correct_answer="", points=1):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO quiz_questions (quiz_id, question_text, question_type, options_json, correct_answer, points, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (quiz_id, question_text, question_type, options_json, correct_answer, points, created_at))
        q_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return q_id

    def is_room_member(self, room_code, student_name):
        if not room_code or not student_name:
            return False
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM room_members WHERE room_code = ? AND LOWER(TRIM(student_name)) = LOWER(TRIM(?)) LIMIT 1",
            (room_code, student_name)
        )
        row = cursor.fetchone()
        conn.close()
        return row is not None

    def get_quizzes_by_room(self, room_code, student_name=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                q.quiz_id,
                q.room_code,
                q.teacher_name,
                q.title,
                q.description,
                q.duration_minutes,
                q.status,
                q.created_at,
                COUNT(qq.id) AS question_count,
                COALESCE(SUM(qq.points), 0) AS total_marks
            FROM quizzes q
            LEFT JOIN quiz_questions qq ON q.quiz_id = qq.quiz_id
            WHERE q.room_code = ?
            GROUP BY q.quiz_id
            ORDER BY q.created_at DESC
        ''', (room_code,))
        rows = cursor.fetchall()

        quizzes = []
        for r in rows:
            quiz_id = r[0]
            question_count = int(r[8] or 0)
            total_marks = int(r[9] or 0)

            cursor.execute(
                "SELECT COUNT(DISTINCT LOWER(TRIM(student_name))) FROM quiz_responses WHERE quiz_id = ?",
                (quiz_id,)
            )
            submission_count = int(cursor.fetchone()[0] or 0)

            item = {
                'quiz_id': quiz_id,
                'room_code': r[1],
                'teacher_name': r[2],
                'title': r[3],
                'description': r[4],
                'duration_minutes': r[5],
                'status': r[6],
                'created_at': r[7],
                'question_count': question_count,
                'total_marks': total_marks,
                'submission_count': submission_count
            }

            if student_name:
                cursor.execute('''
                    SELECT COUNT(DISTINCT question_id), COALESCE(SUM(score), 0), MAX(submitted_at)
                    FROM quiz_responses
                    WHERE quiz_id = ? AND LOWER(TRIM(student_name)) = LOWER(TRIM(?))
                ''', (quiz_id, student_name))
                s_row = cursor.fetchone()
                answered_cnt = int(s_row[0] or 0)
                student_score = int(s_row[1] or 0)
                item['student_answered_count'] = answered_cnt
                item['student_submitted'] = answered_cnt > 0
                item['student_score'] = student_score
                item['student_submitted_at'] = s_row[2]

            quizzes.append(item)

        conn.close()
        return quizzes

    def get_quiz_with_questions(self, quiz_id, include_answers=False):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT quiz_id, room_code, teacher_name, title, description, duration_minutes, status, created_at FROM quizzes WHERE quiz_id = ?", (quiz_id,))
        qz = cursor.fetchone()
        if not qz:
            conn.close()
            return None

        cursor.execute("SELECT id, question_text, question_type, options_json, correct_answer, points FROM quiz_questions WHERE quiz_id = ? ORDER BY id ASC", (quiz_id,))
        qs = cursor.fetchall()
        conn.close()

        import json
        questions_list = []
        total_marks = 0
        for q in qs:
            opts = []
            if q[3]:
                try:
                    opts = json.loads(q[3])
                except Exception:
                    opts = []
            pts = int(q[5] if q[5] is not None else 1)
            total_marks += pts
            q_dict = {
                'id': q[0],
                'question_text': q[1],
                'question_type': q[2],
                'options': opts,
                'points': pts
            }
            if include_answers:
                q_dict['correct_answer'] = q[4] or ''
            questions_list.append(q_dict)

        return {
            'quiz_id': qz[0],
            'room_code': qz[1],
            'teacher_name': qz[2],
            'title': qz[3],
            'description': qz[4],
            'duration_minutes': qz[5],
            'status': qz[6],
            'created_at': qz[7],
            'question_count': len(questions_list),
            'total_marks': total_marks,
            'questions': questions_list
        }

    def get_student_quiz_submission(self, quiz_id, student_name):
        """Returns a student's existing submission summary for a quiz without exposing answer keys."""
        if not quiz_id or not student_name:
            return None
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT qr.question_id, qr.response, qr.response_mode, qr.confidence, qr.is_correct, qr.score, qr.submitted_at, qq.question_text, qq.points
            FROM quiz_responses qr
            JOIN quiz_questions qq ON qr.question_id = qq.id
            WHERE qr.quiz_id = ? AND LOWER(TRIM(qr.student_name)) = LOWER(TRIM(?))
            ORDER BY qr.question_id ASC
        ''', (quiz_id, student_name))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return None

        total_score = sum(int(r[5] or 0) for r in rows)
        return {
            'submitted': True,
            'total_answered': len(rows),
            'total_score': total_score,
            'submitted_at': max(r[6] for r in rows),
            'responses': [{
                'question_id': r[0],
                'response': r[1],
                'response_mode': r[2],
                'confidence': float(r[3] or 1.0),
                'is_correct': int(r[4] or 0),
                'score': int(r[5] or 0),
                'max_points': int(r[8] if r[8] is not None else 1),
                'question_text': r[7]
            } for r in rows]
        }

    def has_student_submitted_question(self, quiz_id, question_id, student_name):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM quiz_responses WHERE quiz_id = ? AND question_id = ? AND LOWER(TRIM(student_name)) = LOWER(TRIM(?)) LIMIT 1",
            (quiz_id, question_id, student_name)
        )
        row = cursor.fetchone()
        conn.close()
        return row is not None

    def submit_quiz_response(self, quiz_id, question_id, room_code, student_name, response, response_mode='text', confidence=1.0, is_correct=0, score=0):
        submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "SELECT id FROM quiz_responses WHERE quiz_id = ? AND question_id = ? AND LOWER(TRIM(student_name)) = LOWER(TRIM(?)) LIMIT 1",
                (quiz_id, question_id, student_name)
            )
            if cursor.fetchone():
                conn.rollback()
                conn.close()
                raise ValueError("Duplicate submission for this question")

            cursor.execute('''
                INSERT INTO quiz_responses (quiz_id, question_id, room_code, student_name, response, response_mode, confidence, is_correct, score, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (quiz_id, question_id, room_code, student_name, response, response_mode, float(confidence), int(is_correct), int(score), submitted_at))
            resp_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return resp_id
        except Exception:
            conn.close()
            raise

    def submit_quiz_responses_batch(self, quiz_id, room_code, student_name, evaluated_answers):
        """Atomically stores a student's complete quiz attempt and prevents duplicate submissions."""
        submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            q_ids = [int(item['question_id']) for item in evaluated_answers]
            placeholders = ','.join('?' for _ in q_ids)
            cursor.execute(
                f"SELECT question_id FROM quiz_responses WHERE quiz_id = ? AND LOWER(TRIM(student_name)) = LOWER(TRIM(?)) AND question_id IN ({placeholders}) LIMIT 1",
                [quiz_id, student_name] + q_ids
            )
            if cursor.fetchone():
                conn.rollback()
                conn.close()
                raise ValueError("Duplicate submission for this quiz")

            saved_ids = []
            for item in evaluated_answers:
                cursor.execute('''
                    INSERT INTO quiz_responses (quiz_id, question_id, room_code, student_name, response, response_mode, confidence, is_correct, score, submitted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    quiz_id,
                    int(item['question_id']),
                    room_code,
                    student_name,
                    item['response'],
                    item.get('response_mode', 'text'),
                    float(item.get('confidence', 1.0)),
                    int(item['is_correct']),
                    int(item['score']),
                    submitted_at
                ))
                saved_ids.append(cursor.lastrowid)

            conn.commit()
            conn.close()
            return {'response_ids': saved_ids, 'submitted_at': submitted_at}
        except Exception:
            conn.close()
            raise

    def get_quiz_results(self, quiz_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(id), COALESCE(SUM(points), 0) FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
        q_meta = cursor.fetchone()
        total_questions = int(q_meta[0] or 0)
        total_marks = int(q_meta[1] or 0)

        cursor.execute('''
            SELECT
                student_name,
                COALESCE(SUM(score), 0) as total_score,
                COUNT(DISTINCT question_id) as total_answered,
                AVG(confidence) as avg_conf,
                MAX(submitted_at) as last_submitted_at
            FROM quiz_responses
            WHERE quiz_id = ?
            GROUP BY LOWER(TRIM(student_name))
            ORDER BY total_score DESC, last_submitted_at ASC
        ''', (quiz_id,))
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            total_score = int(r[1] or 0)
            total_answered = int(r[2] or 0)
            status = 'Submitted' if (total_questions > 0 and total_answered >= total_questions) else 'Partial'
            pct = round((total_score / total_marks) * 100.0, 1) if total_marks > 0 else 0.0
            results.append({
                'student_name': r[0],
                'total_score': total_score,
                'total_marks': total_marks,
                'total_answered': total_answered,
                'total_questions': total_questions,
                'percentage': pct,
                'submission_status': status,
                'avg_confidence': round(float(r[3] or 1.0) * 100.0, 1),
                'submitted_at': r[4]
            })
        return results

    # =========================================================================
    # 6. CLASSROOM CONTENT MATERIALS
    # =========================================================================
    def add_classroom_content(self, room_code, teacher_name, subject, module_name, lesson_title, file_name, file_path, file_type, file_size=0):
        uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO classroom_content (room_code, teacher_name, subject, module_name, lesson_title, file_name, file_path, file_type, file_size, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (room_code, teacher_name, subject, module_name, lesson_title, file_name, file_path, file_type, file_size, uploaded_at))
        c_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {
            'id': c_id,
            'room_code': room_code,
            'teacher_name': teacher_name,
            'subject': subject,
            'module_name': module_name,
            'lesson_title': lesson_title,
            'file_name': file_name,
            'file_path': file_path,
            'file_type': file_type,
            'file_size': file_size,
            'uploaded_at': uploaded_at
        }

    def get_classroom_content(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, subject, module_name, lesson_title, file_name, file_path, file_type, file_size, uploaded_at FROM classroom_content WHERE room_code = ? ORDER BY subject, module_name, id DESC", (room_code,))
        rows = cursor.fetchall()
        conn.close()
        return [{
            'id': r[0],
            'subject': r[1],
            'module_name': r[2],
            'lesson_title': r[3],
            'file_name': r[4],
            'file_path': r[5],
            'file_type': r[6],
            'file_size': r[7],
            'uploaded_at': r[8]
        } for r in rows]

    # =========================================================================
    # 7. CLASSROOM ANALYTICS & STUDENT PERFORMANCE
    # =========================================================================
    def get_room_analytics(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()

        # Members count
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN status='online' THEN 1 ELSE 0 END) FROM room_members WHERE room_code = ?", (room_code,))
        m_row = cursor.fetchone()
        total_students = m_row[0] or 0
        online_students = m_row[1] or 0
        offline_students = max(0, total_students - online_students)

        # Questions count
        cursor.execute("SELECT COUNT(*) FROM questions WHERE room_code = ?", (room_code,))
        total_questions = cursor.fetchone()[0] or 0

        # Responses count & avg confidence
        cursor.execute("SELECT COUNT(*), AVG(confidence) FROM student_responses WHERE room_code = ?", (room_code,))
        r_row = cursor.fetchone()
        total_responses = r_row[0] or 0
        avg_sign_conf = round(float(r_row[1] or 0.0) * (100.0 if (r_row[1] or 0.0) <= 1.0 else 1.0), 1)

        # Unanswered questions count
        cursor.execute("SELECT COUNT(DISTINCT q.question_id) FROM questions q LEFT JOIN student_responses r ON q.question_id = r.question_id WHERE q.room_code = ? AND r.id IS NULL", (room_code,))
        unanswered_questions = cursor.fetchone()[0] or 0

        # Response modes breakdown
        cursor.execute("SELECT response_mode, COUNT(*) FROM student_responses WHERE room_code = ? GROUP BY response_mode", (room_code,))
        rm_rows = cursor.fetchall()
        mode_counts = {'sign': 0, 'speech': 0, 'text': 0}
        for rmode, cnt in rm_rows:
            if rmode in mode_counts:
                mode_counts[rmode] = cnt

        # Languages breakdown
        cursor.execute("SELECT preferred_language, COUNT(*) FROM room_members WHERE room_code = ? GROUP BY preferred_language", (room_code,))
        lang_rows = cursor.fetchall()
        lang_counts = {}
        for lcode, cnt in lang_rows:
            lang_counts[lcode or 'en'] = cnt

        conn.close()

        return {
            'room_code': room_code,
            'total_students': total_students,
            'online_students': online_students,
            'offline_students': offline_students,
            'total_questions': total_questions,
            'total_responses': total_responses,
            'unanswered_questions': unanswered_questions,
            'avg_sign_confidence': avg_sign_conf,
            'avg_response_time_sec': 3.2,
            'mode_distribution': mode_counts,
            'language_distribution': lang_counts
        }

    def get_student_performance(self, room_code, student_name):
        conn = self._get_connection()
        cursor = conn.cursor()

        # Questions attempted vs answered
        cursor.execute("SELECT COUNT(*) FROM questions WHERE room_code = ?", (room_code,))
        total_q = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*), AVG(confidence) FROM student_responses WHERE room_code = ? AND student_name = ?", (room_code, student_name))
        resp_row = cursor.fetchone()
        answered_q = resp_row[0] or 0
        avg_conf = round(float(resp_row[1] or 0.0) * (100.0 if (resp_row[1] or 0.0) <= 1.0 else 1.0), 1)

        part_pct = round((answered_q / total_q * 100.0), 1) if total_q > 0 else 0.0

        # Mode counts
        cursor.execute("SELECT response_mode, COUNT(*) FROM student_responses WHERE room_code = ? AND student_name = ? GROUP BY response_mode", (room_code, student_name))
        rm_rows = cursor.fetchall()
        mode_counts = {'sign': 0, 'speech': 0, 'text': 0}
        for rmode, cnt in rm_rows:
            if rmode in mode_counts:
                mode_counts[rmode] = cnt

        conn.close()

        return {
            'student_name': student_name,
            'room_code': room_code,
            'questions_attempted': total_q,
            'questions_answered': answered_q,
            'participation_percentage': part_pct,
            'sign_responses': mode_counts['sign'],
            'speech_responses': mode_counts['speech'],
            'text_responses': mode_counts['text'],
            'avg_response_time_sec': 2.8,
            'avg_sign_confidence': avg_conf
        }

    # =========================================================================
    # 8. USER ROLES & AUTHENTICATION
    # =========================================================================
    def create_user(self, username, password_hash, role='student', full_name="", email=""):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, full_name, email, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, role, full_name or username, email, created_at))
            u_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return {'id': u_id, 'username': username, 'role': role, 'full_name': full_name or username, 'email': email, 'created_at': created_at}
        except sqlite3.IntegrityError:
            conn.close()
            return None

    def get_user_by_username(self, username):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, role, full_name, email, created_at FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'id': row[0],
                'username': row[1],
                'password_hash': row[2],
                'role': row[3],
                'full_name': row[4],
                'email': row[5],
                'created_at': row[6]
            }
        return None

    def get_all_users(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, full_name, email, created_at FROM users ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return [{
            'id': r[0],
            'username': r[1],
            'role': r[2],
            'full_name': r[3],
            'email': r[4],
            'created_at': r[5]
        } for r in rows]

