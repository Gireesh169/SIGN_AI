import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app_module
from src.database import ClassroomDatabase


class TestQuizWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.temp_dir.name, 'test_quiz_workflow.db')
        self.original_db = flask_app_module.db
        self.db = ClassroomDatabase(db_path=self.test_db_path)
        flask_app_module.db = self.db
        flask_app_module.app.config['TESTING'] = True
        self.client = flask_app_module.app.test_client()

    def tearDown(self):
        flask_app_module.db = self.original_db
        self.temp_dir.cleanup()

    def _create_room_and_student(self, teacher_name='Prof. Sharma', student_name='Rahul'):
        room = self.db.create_room(teacher_name, 'Science 101', 'Science', 'Classroom', 30, '', 'en')
        self.db.join_room(room['room_code'], student_name, 'en')
        return room['room_code']

    def test_end_to_end_teacher_create_student_take_and_results(self):
        # 1. Teacher creates classroom
        res = self.client.post('/api/create_room', data={
            'teacher_name': 'Prof. Sharma',
            'room_name': 'Physics 101',
            'subject': 'Physics',
            'description': 'Mechanics',
            'max_students': '30',
            'password': ''
        })
        self.assertEqual(res.status_code, 302)
        room_code = res.headers['Location'].rstrip('/').split('/')[-1]

        # 2. Teacher creates quiz with MCQ, Text, and Sign questions
        create_res = self.client.post('/api/quiz/create', json={
            'room_code': room_code,
            'teacher_name': 'Prof. Sharma',
            'title': 'Force & Motion Quiz',
            'description': 'Answer all 3 questions',
            'duration_minutes': 15,
            'questions': [
                {
                    'question_text': 'What is the SI unit of force?',
                    'question_type': 'mcq',
                    'options': ['Newton', 'Joule', 'Watt'],
                    'correct_answer': 'Newton',
                    'points': 2
                },
                {
                    'question_text': 'Name the force pulling objects toward Earth.',
                    'question_type': 'text',
                    'options': [],
                    'correct_answer': 'Gravity',
                    'points': 3
                },
                {
                    'question_text': 'Sign the greeting word HELLO.',
                    'question_type': 'sign',
                    'options': [],
                    'correct_answer': 'HELLO',
                    'points': 5
                }
            ]
        })
        self.assertEqual(create_res.status_code, 200)
        create_data = create_res.get_json()
        self.assertEqual(create_data['status'], 'success')
        quiz_id = create_data['quiz']['quiz_id']
        self.assertEqual(create_data['quiz']['question_count'], 3)
        self.assertEqual(create_data['quiz']['total_marks'], 10)

        # 3. Student joins the same classroom
        join_res = self.client.post('/api/join_room', data={
            'student_name': 'Rahul',
            'room_code': room_code,
            'password': ''
        })
        self.assertEqual(join_res.status_code, 302)

        # 4. Student sees published quiz in classroom list
        list_res = self.client.get(f'/api/quizzes/{room_code}?student_name=Rahul')
        self.assertEqual(list_res.status_code, 200)
        quizzes = list_res.get_json()['quizzes']
        self.assertEqual(len(quizzes), 1)
        self.assertEqual(quizzes[0]['quiz_id'], quiz_id)
        self.assertEqual(quizzes[0]['question_count'], 3)
        self.assertEqual(quizzes[0]['total_marks'], 10)
        self.assertFalse(quizzes[0]['student_submitted'])

        # 5. Student opens quiz -> verify questions/options returned and NO correct_answer leaked
        detail_res = self.client.get(f'/api/quiz/{quiz_id}?room_code={room_code}&student_name=Rahul')
        self.assertEqual(detail_res.status_code, 200)
        quiz_detail = detail_res.get_json()
        self.assertEqual(quiz_detail['title'], 'Force & Motion Quiz')
        self.assertEqual(len(quiz_detail['questions']), 3)
        for q in quiz_detail['questions']:
            self.assertNotIn('correct_answer', q)

        q1, q2, q3 = quiz_detail['questions']
        self.assertEqual(q1['options'], ['Newton', 'Joule', 'Watt'])

        # 6. Student submits quiz (with tampered client correct_answer/score to verify server ignores it)
        submit_res = self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'room_code': room_code,
            'student_name': 'Rahul',
            'correct_answer': 'WRONG_IGNORE_ME',
            'score': 999,
            'answers': [
                {'question_id': q1['id'], 'response': 'Newton', 'response_mode': 'mcq', 'confidence': 1.0},
                {'question_id': q2['id'], 'response': '  gravity ', 'response_mode': 'text', 'confidence': 1.0},
                {'question_id': q3['id'], 'response': 'THANKS', 'response_mode': 'sign', 'confidence': 0.88}
            ]
        })
        self.assertEqual(submit_res.status_code, 200)
        submit_data = submit_res.get_json()
        self.assertEqual(submit_data['status'], 'success')
        self.assertEqual(submit_data['submission_status'], 'Submitted')
        self.assertEqual(submit_data['total_score'], 5)  # 2 + 3 + 0
        self.assertEqual(submit_data['max_score'], 10)
        self.assertEqual(submit_data['total_answered'], 3)
        self.assertEqual(submit_data['total_questions'], 3)
        self.assertEqual(submit_data['percentage'], 50.0)
        for item in submit_data['results']:
            self.assertNotIn('correct_answer', item)

        # 7. Duplicate submission is rejected
        dup_res = self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'room_code': room_code,
            'student_name': 'Rahul',
            'answers': [
                {'question_id': q1['id'], 'response': 'Newton', 'response_mode': 'mcq'},
                {'question_id': q2['id'], 'response': 'Gravity', 'response_mode': 'text'},
                {'question_id': q3['id'], 'response': 'HELLO', 'response_mode': 'sign'}
            ]
        })
        self.assertEqual(dup_res.status_code, 409)
        self.assertEqual(dup_res.get_json()['status'], 'duplicate')

        # 8. Teacher views results
        results_res = self.client.get(f'/api/quiz/results/{quiz_id}')
        self.assertEqual(results_res.status_code, 200)
        results_data = results_res.get_json()
        self.assertEqual(results_data['quiz']['total_marks'], 10)
        self.assertEqual(len(results_data['results']), 1)
        row = results_data['results'][0]
        self.assertEqual(row['student_name'], 'Rahul')
        self.assertEqual(row['total_score'], 5)
        self.assertEqual(row['total_marks'], 10)
        self.assertEqual(row['total_answered'], 3)
        self.assertEqual(row['total_questions'], 3)
        self.assertEqual(row['submission_status'], 'Submitted')

        # 9. Student quiz list reflects completed submission
        list_after = self.client.get(f'/api/quizzes/{room_code}?student_name=Rahul').get_json()['quizzes']
        self.assertTrue(list_after[0]['student_submitted'])
        self.assertEqual(list_after[0]['student_score'], 5)

    def test_sign_response_correct_grading_and_single_submit_duplicate(self):
        room_code = self._create_room_and_student(student_name='Priya')
        create_res = self.client.post('/api/quiz/create', json={
            'room_code': room_code,
            'teacher_name': 'Prof. Sharma',
            'title': 'Sign Quiz',
            'duration_minutes': 5,
            'questions': [
                {
                    'question_text': 'Perform the sign for Thank You',
                    'question_type': 'sign',
                    'options': [],
                    'correct_answer': 'thank you',
                    'points': 4
                }
            ]
        })
        quiz_id = create_res.get_json()['quiz']['quiz_id']
        q_id = self.client.get(f'/api/quiz/{quiz_id}').get_json()['questions'][0]['id']

        res = self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'question_id': q_id,
            'room_code': room_code,
            'student_name': 'Priya',
            'response': 'THANK YOU',
            'response_mode': 'sign',
            'confidence': 0.92,
            'correct_answer': 'WRONG_CLIENT_VALUE'
        })
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body['is_correct'], 1)
        self.assertEqual(body['score'], 4)
        self.assertEqual(body['max_points'], 4)

        dup = self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'question_id': q_id,
            'room_code': room_code,
            'student_name': 'Priya',
            'response': 'THANK YOU',
            'response_mode': 'sign'
        })
        self.assertEqual(dup.status_code, 409)

    def test_validation_and_security_error_cases(self):
        room_a = self._create_room_and_student(student_name='Amit')
        room_b = self._create_room_and_student(student_name='Neha')

        # Invalid quiz creations
        self.assertEqual(self.client.post('/api/quiz/create', json={'room_code': room_a, 'title': '', 'questions': []}).status_code, 400)
        self.assertEqual(self.client.post('/api/quiz/create', json={'room_code': room_a, 'title': 'Empty Qs', 'questions': []}).status_code, 400)
        self.assertEqual(self.client.post('/api/quiz/create', json={
            'room_code': room_a,
            'title': 'Bad MCQ',
            'questions': [{'question_text': 'Q1', 'question_type': 'mcq', 'options': ['OnlyOne'], 'correct_answer': 'OnlyOne'}]
        }).status_code, 400)
        self.assertEqual(self.client.post('/api/quiz/create', json={
            'room_code': room_a,
            'title': 'Mismatch MCQ',
            'questions': [{'question_text': 'Q1', 'question_type': 'mcq', 'options': ['A', 'B'], 'correct_answer': 'C'}]
        }).status_code, 400)
        self.assertEqual(self.client.post('/api/quiz/create', json={
            'room_code': 'SIGN-0000',
            'title': 'Missing Room',
            'questions': [{'question_text': 'Q1', 'question_type': 'text', 'correct_answer': 'x'}]
        }).status_code, 404)

        # Valid 2-question quiz in room_a
        quiz_id = self.client.post('/api/quiz/create', json={
            'room_code': room_a,
            'title': 'Validation Quiz',
            'questions': [
                {'question_text': 'Q1', 'question_type': 'mcq', 'options': ['A', 'B'], 'correct_answer': 'A', 'points': 1},
                {'question_text': 'Q2', 'question_type': 'text', 'correct_answer': 'B', 'points': 2}
            ]
        }).get_json()['quiz']['quiz_id']
        qs = self.client.get(f'/api/quiz/{quiz_id}').get_json()['questions']

        # Missing quiz / results -> 404
        self.assertEqual(self.client.get('/api/quiz/QUIZ-NOTFOUND').status_code, 404)
        self.assertEqual(self.client.get('/api/quiz/results/QUIZ-NOTFOUND').status_code, 404)

        # Unenrolled student -> 403
        self.assertEqual(self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'room_code': room_a,
            'student_name': 'Intruder',
            'answers': [{'question_id': qs[0]['id'], 'response': 'A'}, {'question_id': qs[1]['id'], 'response': 'B'}]
        }).status_code, 403)

        # Cross-room submission -> 403
        self.assertEqual(self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'room_code': room_b,
            'student_name': 'Neha',
            'answers': [{'question_id': qs[0]['id'], 'response': 'A'}, {'question_id': qs[1]['id'], 'response': 'B'}]
        }).status_code, 403)

        # Missing required question in batch -> 400
        partial_res = self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'room_code': room_a,
            'student_name': 'Amit',
            'answers': [{'question_id': qs[0]['id'], 'response': 'A'}]
        })
        self.assertEqual(partial_res.status_code, 400)
        self.assertIn(qs[1]['id'], partial_res.get_json()['missing_question_ids'])

        # Empty answer -> 400
        self.assertEqual(self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'room_code': room_a,
            'student_name': 'Amit',
            'answers': [{'question_id': qs[0]['id'], 'response': '   '}, {'question_id': qs[1]['id'], 'response': 'B'}]
        }).status_code, 400)

        # Foreign question_id -> 400
        self.assertEqual(self.client.post('/api/quiz/submit', json={
            'quiz_id': quiz_id,
            'room_code': room_a,
            'student_name': 'Amit',
            'answers': [{'question_id': 999999, 'response': 'A'}]
        }).status_code, 400)

    def test_dashboards_render_without_alert_stub(self):
        room_code = self._create_room_and_student(student_name='Rahul')

        teacher_html = self.client.get(f'/teacher/{room_code}').get_data(as_text=True)
        self.assertIn('addQuizQuestion', teacher_html)
        self.assertIn('teacher-quiz-results-container', teacher_html)

        student_html = self.client.get(f'/student/{room_code}?student_name=Rahul').get_data(as_text=True)
        self.assertIn('quiz-taking-view', student_html)
        self.assertIn('submitQuizAttempt', student_html)
        self.assertIn('startQuizSignCamera', student_html)
        self.assertNotIn('alert(`Starting Quiz:', student_html)

        # Preserve existing pages
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/history').status_code, 200)
        self.assertEqual(self.client.get(f'/report/{room_code}').status_code, 200)
        self.assertEqual(self.client.get(f'/analytics/{room_code}').status_code, 200)


if __name__ == '__main__':
    unittest.main()
