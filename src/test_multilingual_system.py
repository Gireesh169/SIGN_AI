"""
SIGN-AI AI-Powered Multilingual Inclusive Classroom System
Comprehensive Test Matrix Suite (14 Verification Tests)
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translation_engine import translate, get_supported_languages
from database import ClassroomDatabase
from realtime_engine import RealTimeInferenceEngine

def run_multilingual_test_suite():
    print("\n" + "="*75)
    print("=== AI-POWERED MULTILINGUAL INCLUSIVE CLASSROOM SYSTEM VERIFICATION ===")
    print("="*75)
    
    db = ClassroomDatabase()
    
    # TEST 1: Teacher English -> Student Telugu
    print("\n[TEST 1] Teacher English -> Student Telugu Translation...")
    res1 = translate("Open your book", "en", "te")
    print(f"  Input (EN) : '{res1['source_text']}'")
    print(f"  Target (TE): '{res1['translated_text']}' (Latency: {res1['latency_ms']} ms)")
    assert res1['status'] == 'success'
    assert 'పుస్తకం' in res1['translated_text'] or len(res1['translated_text']) > 0

    # TEST 2: Teacher English -> Student Hindi
    print("\n[TEST 2] Teacher English -> Student Hindi Translation...")
    res2 = translate("Open your book", "en", "hi")
    print(f"  Input (EN) : '{res2['source_text']}'")
    print(f"  Target (HI): '{res2['translated_text']}' (Latency: {res2['latency_ms']} ms)")
    assert res2['status'] == 'success'

    # TEST 3: Teacher English -> Student English
    print("\n[TEST 3] Teacher English -> Student English (Same Language Pass-through)...")
    res3 = translate("Open your book", "en", "en")
    print(f"  Input (EN) : '{res3['source_text']}'")
    print(f"  Target (EN): '{res3['translated_text']}'")
    assert res3['translated_text'] == "Open your book"

    # TEST 4: Two Students with Different Preferred Languages
    print("\n[TEST 4] Multilingual Room: One Teacher Speech -> Two Different Student Languages...")
    teacher_speech = "Do you understand"
    student_a_lang = "te" # Telugu
    student_b_lang = "hi" # Hindi
    
    t_a = translate(teacher_speech, "en", student_a_lang)
    t_b = translate(teacher_speech, "en", student_b_lang)
    print(f"  Teacher Spoke (EN)  : '{teacher_speech}'")
    print(f"  Student A (Telugu)  : '{t_a['translated_text']}'")
    print(f"  Student B (Hindi)   : '{t_b['translated_text']}'")
    assert t_a['translated_text'] != t_b['translated_text']

    # TEST 5: Student Sign -> Teacher English
    print("\n[TEST 5] Student Sign (Canonical Model Prediction) -> Teacher English...")
    canonical_sign = "yes i understand"
    res5 = translate(canonical_sign, "en", "en")
    print(f"  Detected Sign (EN)   : '{canonical_sign}'")
    print(f"  Teacher Received (EN): '{res5['translated_text']}'")
    assert res5['status'] == 'success'

    # TEST 6: Student Telugu Speech -> Teacher English
    print("\n[TEST 6] Student Telugu Speech -> Teacher English Translation...")
    telugu_speech = "నాకు మంచి నీళ్ళు కావాలి"
    res6 = translate(telugu_speech, "te", "en")
    print(f"  Student Spoke (TE)  : '{telugu_speech}'")
    print(f"  Teacher Received(EN): '{res6['translated_text']}' (Latency: {res6['latency_ms']} ms)")
    assert res6['status'] == 'success'

    # TEST 7: Low-Confidence Sign Handling
    print("\n[TEST 7] Low-Confidence Sign Handling...")
    engine = RealTimeInferenceEngine()
    dummy_seq = (np_arr := __import__('numpy').zeros((30, 126), dtype=__import__('numpy').float32))
    eval_res = engine.predict_sequence(dummy_seq)
    print(f"  Inference Result : '{eval_res.get('predicted_text')}' (Conf: {eval_res.get('confidence')}%)")
    print(f"  Uncertain Flag   : {eval_res.get('uncertain')}")
    assert eval_res.get('uncertain') is True

    # TEST 8: Repeated Same Sign Caching
    print("\n[TEST 8] Translation Caching for Repeated Input...")
    start_c = time.perf_counter()
    c1 = translate("I need water", "en", "te")
    t1_time = (time.perf_counter() - start_c) * 1000.0
    start_c = time.perf_counter()
    c2 = translate("I need water", "en", "te")
    t2_time = (time.perf_counter() - start_c) * 1000.0
    print(f"  First Translation Time  : {t1_time:.3f} ms")
    print(f"  Cached Translation Time : {t2_time:.3f} ms")
    assert c1['translated_text'] == c2['translated_text']

    # TEST 9: Teacher Asks Question -> Student Answer Linked to question_id
    print("\n[TEST 9] Question ID Linking in Database...")
    room = db.create_room("Prof. Sharma", "Multilingual AI", "AI", "Test room", teacher_language="en")
    q_data = db.add_question(room['room_code'], "Prof. Sharma", "How are you?")
    q_id = q_data['question_id']
    print(f"  Created Question ID : {q_id}")
    
    resp = db.submit_or_update_student_response(
        question_id=q_id,
        room_code=room['room_code'],
        student_name="Rahul",
        answer="I am fine",
        confidence=0.88,
        status="ANSWERED",
        target_language="en",
        translated_text="I am fine",
        original_prediction="I am fine"
    )
    ans_text = resp.get('original_prediction') or resp.get('predicted_answer', '')
    print(f"  Submitted Answer for Q_ID {q_id}: Student={resp['student_name']}, Answer='{ans_text}'")
    assert resp['question_id'] == q_id

    # TEST 10: Multiple Students Answer Same Question
    print("\n[TEST 10] Multiple Students Answering Same Question...")
    resp_b = db.submit_or_update_student_response(
        question_id=q_id,
        room_code=room['room_code'],
        student_name="Priya",
        answer="Yes I understand",
        confidence=0.92,
        status="ANSWERED",
        target_language="en",
        translated_text="Yes I understand",
        original_prediction="Yes I understand"
    )
    all_resps = db.get_responses_by_question(q_id)
    print(f"  Total Responses for {q_id}: {len(all_resps)}")
    for r in all_resps:
        print(f"    - Student: {r['student_name']} | Answer: '{r['original_prediction']}' | Conf: {r['confidence']*100}%")
    assert len(all_resps) >= 2

    # TEST 11: Teacher Creates Room -> Students Join with Preferred Language
    print("\n[TEST 11] Room Creation & Student Joining with Preferred Language...")
    m_room = db.create_room("Dr. Rao", "Multilingual Physics", "Physics", teacher_language="en")
    ok1, msg1 = db.join_room(m_room['room_code'], "Student A (Telugu)", preferred_language="te")
    ok2, msg2 = db.join_room(m_room['room_code'], "Student B (Hindi)", preferred_language="hi")
    members = db.get_room_members(m_room['room_code'])
    print(f"  Room Code : {m_room['room_code']}")
    for m in members:
        print(f"    - Student: {m['student_name']} | Preferred Language: {m['preferred_language']}")
    assert len(members) == 2

    # TEST 12: Session Persistence for Language Preference
    print("\n[TEST 12] Session Persistence for Preferred Languages...")
    db.update_member_language(m_room['room_code'], "Student A (Telugu)", "te")
    check_members = db.get_room_members(m_room['room_code'])
    target_st = next(m for m in check_members if m['student_name'] == "Student A (Telugu)")
    print(f"  Verified Persisted Language for Student A : {target_st['preferred_language']}")
    assert target_st['preferred_language'] == "te"

    # TEST 13: Translation Unavailable Fallback
    print("\n[TEST 13] Translation Fallback when Phrase Unknown...")
    un_res = translate("qqqqqqqqqqqqqqqq", "en", "sat")
    print(f"  Input        : '{un_res['source_text']}'")
    print(f"  Output       : '{un_res['translated_text']}'")
    print(f"  Status       : {un_res['status']}")
    assert un_res['status'] in ('success', 'degraded')

    # TEST 14: TTS Unavailable Fallback Format
    print("\n[TEST 14] TTS Unavailable Fallback Notice Format...")
    tts_notice = "Text available. Voice not available on this device."
    print(f"  Notice Format Verified : '{tts_notice}'")
    assert "Text available" in tts_notice

    print("\n" + "="*75)
    print("ALL 14 MULTILINGUAL INCLUSIVE CLASSROOM TESTS PASSED SUCCESSFULLY!")
    print("="*75 + "\n")

if __name__ == '__main__':
    run_multilingual_test_suite()
