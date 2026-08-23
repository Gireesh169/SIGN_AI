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

    def create_room(self, teacher_name, room_name, subject, description="", max_students=30, password=""):
        code = self.generate_room_code()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO rooms (room_code, teacher_name, room_name, subject, description, max_students, password, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
        ''', (code, teacher_name, room_name, subject, description, max_students, password, created_at))
        conn.commit()
        conn.close()

        return {
            'room_code': code,
            'teacher_name': teacher_name,
            'room_name': room_name,
            'subject': subject,
            'description': description,
            'max_students': max_students,
            'created_at': created_at,
            'status': 'active'
        }

    def get_room(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT room_code, teacher_name, room_name, subject, description, max_students, password, created_at, status FROM rooms WHERE room_code = ?", (room_code,))
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
                'status': row[8]
            }
        return None

    def join_room(self, room_code, student_name):
        room = self.get_room(room_code)
        if not room:
            return False, "Room Code not found"

        joined_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO room_members (room_code, student_name, status, joined_at)
            VALUES (?, ?, 'online', ?)
        ''', (room_code, student_name, joined_at))
        conn.commit()
        conn.close()

        return True, "Joined successfully"

    def get_room_members(self, room_code):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, student_name, status, joined_at FROM room_members WHERE room_code = ? ORDER BY id ASC", (room_code,))
        rows = cursor.fetchall()
        conn.close()

        return [{'id': r[0], 'student_name': r[1], 'status': r[2], 'joined_at': r[3]} for r in rows]

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

    def submit_or_update_student_response(self, question_id, room_code, student_name, answer, confidence, status="ANSWERED"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                SET predicted_answer = ?, confidence = ?, status = ?, updated_at = ?
                WHERE id = ?
            ''', (answer, float(confidence), status, ts, row[0]))
        else:
            # INSERT new answer
            cursor.execute('''
                INSERT INTO student_responses (question_id, room_code, student_name, predicted_answer, confidence, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (question_id, room_code, student_name, answer, float(confidence), status, ts, ts))

        conn.commit()
        conn.close()

        # Also Log Activity
        action_type = "ANSWER_UPDATED" if is_update else "ANSWER_SENT"
        self.log_student_activity(student_name, room_code, question_id, action_type, answer, confidence)

        return {
            'question_id': question_id,
            'room_code': room_code,
            'student_name': student_name,
            'predicted_answer': answer,
            'confidence': float(confidence),
            'status': status,
            'is_update': is_update,
            'timestamp': ts
        }

    def get_responses_by_question(self, question_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, question_id, room_code, student_name, predicted_answer, confidence, status, updated_at FROM student_responses WHERE question_id = ? ORDER BY updated_at DESC", (question_id,))
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
            'updated_at': r[7]
        } for r in rows]

    def log_student_activity(self, student_name, room_code, question_id, action_type, message, confidence=0.0):
        ts = datetime.now().strftime("%H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO student_activity_logs (student_name, room_code, question_id, action_type, message, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (student_name, room_code, question_id, action_type, message, float(confidence), ts))
        conn.commit()
        conn.close()

    def get_student_activity_logs(self, room_code, student_name, limit=20):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.id, a.student_name, a.room_code, a.question_id, a.action_type, a.message, a.confidence, a.timestamp, q.question_text
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
            'question_text': r[8] or "Classroom Instruction"
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

        query = "SELECT room_code, student_name, predicted_answer, confidence, status, updated_at FROM student_responses WHERE 1=1"
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
            'timestamp': r[5]
        } for r in rows]
