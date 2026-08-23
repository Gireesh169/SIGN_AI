import os
import sys
import time
import sqlite3
from datetime import datetime
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, DATA_DIR

DB_PATH = os.path.join(DATA_DIR, 'conversation_memory.db')
os.makedirs(DATA_DIR, exist_ok=True)

class ShortTermPredictionMemory:
    """
    LAYER 1: Short-Term Prediction Memory Buffer.
    Stores recent frame predictions: [prediction, confidence, timestamp].
    Used for temporal smoothing, voting, and avoiding repeated speech.
    Resets per camera session.
    """
    def __init__(self, buffer_size=7):
        self.buffer = deque(maxlen=buffer_size)
        self.last_accepted_text = None
        self.last_accepted_time = 0.0

    def add_prediction(self, prediction_text, confidence, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.buffer.append({
            'prediction': prediction_text,
            'confidence': float(confidence),
            'timestamp': float(timestamp)
        })

    def get_stable_prediction(self, min_confidence=0.40, min_votes=3, min_margin=0.10, prob_margin=0.15):
        """
        Calculates stable prediction via temporal voting.
        Returns (stable_text, confidence, is_stable).
        """
        if len(self.buffer) < min_votes:
            return None, 0.0, False

        preds = [item['prediction'] for item in self.buffer]
        confs = [item['confidence'] for item in self.buffer]

        counter = Counter(preds)
        most_common_pred, votes = counter.most_common(1)[0]
        avg_conf = float(np.mean([item['confidence'] for item in self.buffer if item['prediction'] == most_common_pred]))

        if votes >= min_votes and avg_conf >= min_confidence and prob_margin >= min_margin:
            return most_common_pred, avg_conf, True
        return None, avg_conf, False

    def clear(self):
        self.buffer.clear()
        self.last_accepted_text = None
        self.last_accepted_time = 0.0

class ConversationMemoryStore:
    """
    LAYER 2: Conversation Memory Store.
    Stores accepted two-way communications between Teacher and Student.
    Session-based in memory + persistent SQLite database backing.
    """
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def add_message(self, sender, message, confidence=1.0, timestamp_str=None):
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime("%H:%M:%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversation_history (sender, message, timestamp, confidence)
            VALUES (?, ?, ?, ?)
        ''', (sender, message, timestamp_str, float(confidence)))
        conn.commit()
        conn.close()

        return {
            'sender': sender,
            'message': message,
            'timestamp': timestamp_str,
            'confidence': float(confidence)
        }

    def get_history(self, limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sender, message, timestamp, confidence FROM conversation_history
            ORDER BY id ASC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()

        history = []
        for r in rows:
            history.append({
                'sender': r[0],
                'message': r[1],
                'timestamp': r[2],
                'confidence': float(r[3])
            })
        return history

    def clear_history(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM conversation_history')
        conn.commit()
        conn.close()

import numpy as np # Helper import for numpy mean calculation

if __name__ == '__main__':
    store = ConversationMemoryStore()
    store.add_message("Teacher", "Open your book", 1.0)
    store.add_message("Student", "i do not understand", 0.88)
    store.add_message("Teacher", "I will explain again", 1.0)

    hist = store.get_history()
    print(f"Conversation Memory Store Loaded ({len(hist)} messages):")
    for msg in hist:
        print(f"[{msg['timestamp']}] {msg['sender']}: {msg['message']} (Conf: {msg['confidence']:.2f})")
