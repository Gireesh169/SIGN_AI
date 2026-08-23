import os
import sys
import unittest
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import ClassroomDatabase

class TestClassroomPlatform(unittest.TestCase):
    def setUp(self):
        self.test_db_path = os.path.join(os.path.dirname(__file__), 'test_inclusive_classroom.db')
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.db = ClassroomDatabase(db_path=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_room_creation_and_joining(self):
        # 1. Create Room
        room = self.db.create_room("Prof. Sharma", "Computer Vision 101", "CSE-AI", "Sign Language Class", 30)
        self.assertIsNotNone(room['room_code'])
        self.assertTrue(room['room_code'].startswith('SIGN-'))
        print(f"Generated Room Code: {room['room_code']}")

        # 2. Join Room
        success, msg = self.db.join_room(room['room_code'], "Rahul")
        self.assertTrue(success)
        
        members = self.db.get_room_members(room['room_code'])
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]['student_name'], "Rahul")
        print("✅ Room Creation and Student Joining Test Passed!")

    def test_teacher_speech_and_student_sign_logs(self):
        room = self.db.create_room("Prof. Sharma", "AI Class", "CSE")
        room_code = room['room_code']

        # Add Teacher Speech Broadcast
        msg = self.db.add_teacher_message(room_code, "Please open your notebook.")
        self.assertEqual(msg['message_text'], "Please open your notebook.")

        # Add Student Sign Log
        log = self.db.add_student_sign_log(room_code, "Rahul", "I have a question", 0.91)
        self.assertEqual(log['detected_sentence'], "I have a question")

        # Verify Student Request Queue
        requests = self.db.get_student_requests(room_code, status="pending")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['student_name'], "Rahul")

        # Resolve Request
        self.db.resolve_student_request(requests[0]['id'])
        pending_after = self.db.get_student_requests(room_code, status="pending")
        self.assertEqual(len(pending_after), 0)
        print("✅ Teacher Speech & Student Sign Queue Test Passed!")

if __name__ == '__main__':
    print("="*70)
    print("=== RUNNING SMART INCLUSIVE CLASSROOM PLATFORM UNIT TESTS ===")
    print("="*70)
    unittest.main()
