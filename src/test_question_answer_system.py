import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import ClassroomDatabase

class TestQuestionAnswerSystem(unittest.TestCase):
    def setUp(self):
        self.test_db_path = os.path.join(os.path.dirname(__file__), 'test_qa_classroom.db')
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.db = ClassroomDatabase(db_path=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_question_creation_and_linking(self):
        print("\n---> Testing Teacher Question Creation & Unique Question ID")
        room = self.db.create_room("Prof. Sharma", "AI Class", "CSE-AI")
        room_code = room['room_code']

        self.db.join_room(room_code, "Gireesh")
        self.db.join_room(room_code, "Ravi")

        # 1. Teacher asks question
        q1 = self.db.add_question(room_code, "Prof. Sharma", "What is your name?")
        self.assertIsNotNone(q1['question_id'])
        self.assertTrue(q1['question_id'].startswith("q_"))
        print(f"✅ Created Question ID: {q1['question_id']} - '{q1['question_text']}'")

        # 2. Student Gireesh submits answer
        resp1 = self.db.submit_or_update_student_response(q1['question_id'], room_code, "Gireesh", "I am Gireesh", 0.82)
        self.assertFalse(resp1['is_update'])
        self.assertEqual(resp1['predicted_answer'], "I am Gireesh")

        # 3. Student Gireesh UPDATES answer (Duplicate Prevention)
        resp2 = self.db.submit_or_update_student_response(q1['question_id'], room_code, "Gireesh", "My name is Gireesh", 0.88)
        self.assertTrue(resp2['is_update'])
        self.assertEqual(resp2['predicted_answer'], "My name is Gireesh")

        # 4. Check Question Stats
        q_list = self.db.get_questions_by_room(room_code)
        self.assertEqual(len(q_list), 1)
        self.assertEqual(q_list[0]['answered_count'], 1)
        self.assertEqual(q_list[0]['total_members'], 2)
        print("✅ Question Response Stats: 1 / 2 Students Answered")

        # 5. Check Student Activity Logs
        logs = self.db.get_student_activity_logs(room_code, "Gireesh")
        self.assertGreaterEqual(len(logs), 2)
        self.assertEqual(logs[0]['action_type'], "ANSWER_UPDATED")
        print("✅ Student Activity Logs Verified Successfully!")

if __name__ == '__main__':
    print("="*70)
    print("=== RUNNING QUESTION, ANSWER, & ACTIVITY LOG SYSTEM UNIT TESTS ===")
    print("="*70)
    unittest.main()
