import os
import sys
import unittest
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import ClassroomDatabase

class TestProductionPlatform(unittest.TestCase):
    def setUp(self):
        # Use a temporary test database file
        self.test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inclusive_classroom.db')
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.db = ClassroomDatabase(db_path=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_room_creation_and_members(self):
        room = self.db.create_room('Dr. Smith', 'Physics 101', 'Physics', 'Intro to Physics', 30, '', 'en')
        self.assertIsNotNone(room['room_code'])
        self.assertTrue(room['room_code'].startswith('SIGN-'))

        ok, msg = self.db.join_room(room['room_code'], 'Rahul', 'hi')
        self.assertTrue(ok)

        members = self.db.get_room_members(room['room_code'])
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]['student_name'], 'Rahul')
        self.assertEqual(members[0]['preferred_language'], 'hi')

    def test_raise_hand_lifecycle(self):
        room = self.db.create_room('Prof. Verma', 'Chemistry', 'Chemistry')
        code = room['room_code']
        self.db.join_room(code, 'Priya')

        hand = self.db.raise_hand(code, 'Priya')
        self.assertEqual(hand['student_name'], 'Priya')
        self.assertEqual(hand['status'], 'pending')

        active_hands = self.db.get_active_raise_hands(code)
        self.assertEqual(len(active_hands), 1)

        self.db.update_raise_hand_status(hand['id'], 'answered')
        active_hands_after = self.db.get_active_raise_hands(code)
        self.assertEqual(len(active_hands_after), 0)

    def test_announcements_engine(self):
        room = self.db.create_room('Teacher A', 'Math 101', 'Math')
        code = room['room_code']

        ann = self.db.create_announcement(code, 'Teacher A', 'Exam Date', 'Exam is next Monday', 'important', 1)
        self.assertEqual(ann['title'], 'Exam Date')
        self.assertEqual(ann['is_pinned'], 1)

        announcements = self.db.get_announcements(code)
        self.assertEqual(len(announcements), 1)

        self.db.toggle_pin_announcement(ann['id'])
        updated = self.db.get_announcements(code)
        self.assertEqual(updated[0]['is_pinned'], 0)

    def test_quiz_and_assessment(self):
        room = self.db.create_room('Teacher B', 'Biology', 'Biology')
        code = room['room_code']

        quiz = self.db.create_quiz(code, 'Teacher B', 'Cell Biology Quiz', 'Quiz on Cell Structure', 15)
        self.assertTrue(quiz['quiz_id'].startswith('QUIZ-'))

        q1_id = self.db.add_quiz_question(quiz['quiz_id'], 'What is power house of cell?', 'mcq', json.dumps(['Mitochondria', 'Nucleus']), 'Mitochondria', 1)
        self.assertIsNotNone(q1_id)

        quiz_detail = self.db.get_quiz_with_questions(quiz['quiz_id'])
        self.assertEqual(len(quiz_detail['questions']), 1)
        self.assertEqual(quiz_detail['questions'][0]['question_text'], 'What is power house of cell?')

        resp_id = self.db.submit_quiz_response(quiz['quiz_id'], q1_id, code, 'Rahul', 'Mitochondria', 'text', 1.0, 1, 1)
        self.assertIsNotNone(resp_id)

        results = self.db.get_quiz_results(quiz['quiz_id'])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['student_name'], 'Rahul')
        self.assertEqual(results[0]['total_score'], 1)

    def test_classroom_content_upload(self):
        room = self.db.create_room('Teacher C', 'English 101', 'English')
        code = room['room_code']

        c = self.db.add_classroom_content(code, 'Teacher C', 'English', 'Module 1', 'Grammar Notes', 'notes.pdf', '/static/uploads/notes.pdf', 'pdf', 1024)
        self.assertEqual(c['file_name'], 'notes.pdf')

        content_list = self.db.get_classroom_content(code)
        self.assertEqual(len(content_list), 1)

    def test_classroom_analytics_and_performance(self):
        room = self.db.create_room('Teacher D', 'Science 101', 'Science')
        code = room['room_code']
        self.db.join_room(code, 'Amit', 'hi')

        q_data = self.db.add_question(code, 'Teacher D', 'What is H2O?')
        self.db.submit_or_update_student_response(q_data['question_id'], code, 'Amit', 'WATER', 0.95, 'ANSWERED')

        analytics = self.db.get_room_analytics(code)
        self.assertEqual(analytics['total_students'], 1)
        self.assertEqual(analytics['total_questions'], 1)
        self.assertEqual(analytics['total_responses'], 1)

        perf = self.db.get_student_performance(code, 'Amit')
        self.assertEqual(perf['student_name'], 'Amit')
        self.assertEqual(perf['questions_answered'], 1)

    def test_user_roles_and_admin(self):
        u1 = self.db.create_user('admin_user', 'passhash1', 'admin', 'Admin User', 'admin@school.com')
        self.assertIsNotNone(u1)
        self.assertEqual(u1['role'], 'admin')

        u2 = self.db.get_user_by_username('admin_user')
        self.assertEqual(u2['full_name'], 'Admin User')

        all_users = self.db.get_all_users()
        self.assertTrue(len(all_users) >= 1)

if __name__ == '__main__':
    unittest.main()
