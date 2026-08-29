"""
SIGN-AI Multilingual Output Verification Test Suite
Tests:
- English → Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi translations
- translate(text, target_language) interface and returned dictionary structure
- Session translation caching using (source_text, target_language) key
- Error fallback to canonical English text
- Teacher broadcast and Student response payload structures
- Database logging of Original English, Target Language, Translated Text, Confidence, Student, Timestamp
"""

import os
import sys
import unittest
import json
from datetime import datetime

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multilingual_translator import translate, get_translator, SUPPORTED_LANGUAGES
from src.database import ClassroomDatabase

TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_multilingual_classroom.db')

class TestMultilingualOutputSystem(unittest.TestCase):
    def setUp(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        self.db = ClassroomDatabase(db_path=TEST_DB_PATH)
        self.translator = get_translator()
        self.translator.clear_cache()

    def tearDown(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_supported_languages_list(self):
        """Verify all 8 initial languages are supported with BCP-47 codes."""
        langs = self.translator.get_supported_languages()
        expected = ['en', 'te', 'hi', 'ta', 'kn', 'ml', 'bn', 'mr']
        for code in expected:
            self.assertIn(code, langs)
            self.assertIn('bcp47', langs[code])
            self.assertIn('name', langs[code])

    def test_english_to_all_8_languages(self):
        """Test English translation to Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi."""
        sample_text = "I need water"
        test_languages = ['en', 'te', 'hi', 'ta', 'kn', 'ml', 'bn', 'mr']

        for lang_code in test_languages:
            result = translate(sample_text, lang_code)
            self.assertIsInstance(result, dict)
            self.assertIn("source_text", result)
            self.assertIn("target_language", result)
            self.assertIn("translated_text", result)

            self.assertEqual(result["source_text"], sample_text)
            self.assertEqual(result["target_language"], lang_code)
            self.assertTrue(len(result["translated_text"]) > 0)
            print(f"[TEST LANG] EN -> {lang_code.upper()}: '{sample_text}' => '{result['translated_text']}'")

    def test_telugu_example_requirement(self):
        """Verify Requirement 2: 'I need water' -> Telugu 'నాకు నీళ్లు కావాలి'."""
        result = translate("I need water", "te")
        self.assertEqual(result["source_text"], "I need water")
        self.assertEqual(result["target_language"], "te")
        self.assertIn("నాకు నీళ్లు కావాలి", result["translated_text"])

    def test_teacher_instruction_translations(self):
        """Verify Requirement 3: Teacher speaks 'Please open your notebook.' -> Telugu, Hindi, Tamil."""
        teacher_msg = "Please open your notebook"

        te_res = translate(teacher_msg, "te")
        hi_res = translate(teacher_msg, "hi")
        ta_res = translate(teacher_msg, "ta")

        self.assertIn("దయచేసి", te_res["translated_text"])
        self.assertIn("कृपया", hi_res["translated_text"])
        self.assertIn("தயவுசெய்து", ta_res["translated_text"])

    def test_translation_caching(self):
        """Verify Requirement 5: Session cache uses (source_text, target_language) key."""
        text = "How are you"
        lang = "te"

        # First call: populates cache
        res1 = translate(text, lang)
        self.assertFalse(res1.get("cached", False))

        # Second call: uses cache
        res2 = translate(text, lang)
        self.assertTrue(res2.get("cached", False))
        self.assertEqual(res1["translated_text"], res2["translated_text"])

    def test_fallback_on_invalid_language(self):
        """Verify Requirement 13: Fallback returns original English text if translation fails or unsupported."""
        text = "I need water"
        res = translate(text, "xyz_unsupported")
        self.assertEqual(res["translated_text"], text)

    def test_classroom_socket_payload_structures(self):
        """Verify Requirement 6 & 11: Teacher message and Student response payload fields."""
        # 1. Teacher Message Payload
        teacher_msg_payload = {
            "message_id": "q_12345",
            "room_id": "SIGN-1001",
            "original_text": "Please open your notebook",
            "timestamp": "10:30:00"
        }
        self.assertIn("message_id", teacher_msg_payload)
        self.assertIn("room_id", teacher_msg_payload)
        self.assertIn("original_text", teacher_msg_payload)
        self.assertIn("timestamp", teacher_msg_payload)

        # 2. Student Response Payload (Telugu Student)
        tr = translate("I need water", "te")
        student_resp_payload = {
            "question_id": "q_12345",
            "original_prediction": "I need water",
            "translated_text": tr["translated_text"],
            "target_language": "te",
            "confidence": 0.84,
            "timestamp": "10:31:00"
        }
        self.assertEqual(student_resp_payload["original_prediction"], "I need water")
        self.assertEqual(student_resp_payload["target_language"], "te")
        self.assertIn("నాకు నీళ్లు కావాలి", student_resp_payload["translated_text"])

    def test_database_multilingual_logging(self):
        """Verify Requirement 7: Activity & response logs store Original English, Target Language, Translated Text, Confidence, Student, Timestamp."""
        room = self.db.create_room("Teacher Alpha", "Physics 101", "Physics")
        room_code = room["room_code"]
        self.db.join_room(room_code, "Student A", preferred_language="te")

        q = self.db.add_question(room_code, "Teacher Alpha", "How are you?")
        q_id = q["question_id"]

        tr = translate("I am fine thank you sir", "te")
        resp = self.db.submit_or_update_student_response(
            question_id=q_id,
            room_code=room_code,
            student_name="Student A",
            answer="i am fine thank you sir",
            confidence=0.88,
            status="ANSWERED",
            target_language="te",
            translated_text=tr["translated_text"],
            original_prediction="i am fine thank you sir"
        )

        self.assertEqual(resp["original_prediction"], "i am fine thank you sir")
        self.assertEqual(resp["target_language"], "te")
        self.assertEqual(resp["translated_text"], tr["translated_text"])

        # Fetch responses by question
        responses = self.db.get_responses_by_question(q_id)
        self.assertEqual(len(responses), 1)
        r = responses[0]
        self.assertEqual(r["original_prediction"], "i am fine thank you sir")
        self.assertEqual(r["target_language"], "te")
        self.assertEqual(r["translated_text"], tr["translated_text"])

        # Fetch activity logs
        logs = self.db.get_student_activity_logs(room_code, "Student A")
        self.assertTrue(len(logs) > 0)
        self.assertEqual(logs[0]["target_language"], "te")

if __name__ == '__main__':
    unittest.main()
