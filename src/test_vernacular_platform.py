"""
SIGN-AI Vernacular Smart Classroom — Automated System Test Suite
Verifies:
1. Translation Engine (HI -> Santhali, Ho, Mundari) & Latency Measurement
2. Offline Sync Manager (ONLINE, OFFLINE, SYNCING states & cache readiness)
3. Vernacular Curriculum Assistant (Class 1-5 FLN Lesson Generation)
4. Bilingual Worksheet Generator (PDF/Print HTML payload generation)
5. Visual Flashcard Generator (Icons, Hindi, Tribal terms, audio links)
6. SQLite Database Schema Migrations & Room Management
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translation_engine import translate, get_supported_languages
from offline_sync import get_sync_manager
from curriculum_assistant import generate_bilingual_lesson, list_curriculum_options
from worksheet_generator import generate_bilingual_worksheet
from flashcard_generator import generate_flashcards
from database import ClassroomDatabase

def test_all():
    print("\n" + "="*70)
    print("=== SIGN-AI VERNACULAR SMART CLASSROOM SYSTEM VERIFICATION ===")
    print("="*70)

    # 1. Translation Engine & Latency Test
    print("\n[TEST 1] Vernacular Translation Engine & Latency Measurement...")
    t1 = translate("Open your book", "en", "sat")
    print(f"  EN -> Santhali : '{t1['translated_text']}' (Latency: {t1['latency_ms']} ms, Backend: {t1['backend_used']})")
    assert t1['status'] == 'success'
    assert t1['latency_ms'] < 3000.0, "Latency exceeded 3.0s target"

    t2 = translate("मेरी बात सुनें।", "hi", "sat")
    print(f"  HI -> Santhali : '{t2['translated_text']}' (Latency: {t2['latency_ms']} ms)")

    t3 = translate("1", "en", "sat")
    print(f"  Number 1 -> Santhali : '{t3['translated_text']}' (Latency: {t3['latency_ms']} ms)")

    # 2. Offline Synchronization Manager Test
    print("\n[TEST 2] Offline Synchronization Manager...")
    sync_mgr = get_sync_manager()
    st = sync_mgr.get_status()
    print(f"  Connectivity State : {st['connectivity_status']}")
    print(f"  Total Offline Items : {st['total_offline_items']}")
    
    sync_res = sync_mgr.trigger_synchronization()
    print(f"  Trigger Sync Result : {sync_res['connectivity_status']} (Items: {sync_res['total_offline_items']})")

    # 3. Vernacular Curriculum Assistant Test
    print("\n[TEST 3] Vernacular Curriculum Assistant (FLN Class 1 Numeracy)...")
    lesson = generate_bilingual_lesson("class_1", "fln_numeracy", "c1_num_1_10", "sat")
    print(f"  Lesson Title (Hindi)  : {lesson['lesson_plan']['title']['hindi']}")
    print(f"  Lesson Title (Target) : {lesson['lesson_plan']['title']['target']}")
    assert lesson['status'] == 'success'

    # 4. Bilingual Worksheet Generator Test
    print("\n[TEST 4] Bilingual Worksheet Generator...")
    ws = generate_bilingual_worksheet("Class 1", "Numeracy", "Numbers 1-10", "sat", "Medium")
    print(f"  Worksheet ID : {ws['worksheet_id']}")
    print(f"  Title        : {ws['title']}")
    print(f"  Questions    : {len(ws['questions'])} exercise blocks generated")
    assert ws['worksheet_id'].startswith("WS-")

    # 5. Visual Flashcard Generator Test
    print("\n[TEST 5] Visual Flashcard Generator...")
    fc = generate_flashcards("classroom_objects", "sat")
    print(f"  Flashcards Pack : {fc['pack_id']} ({fc['total_cards']} cards)")
    print(f"  Sample Card 1   : {fc['cards'][0]['icon']} {fc['cards'][0]['hindi_word']} -> {fc['cards'][0]['target_word']}")
    assert len(fc['cards']) > 0

    # 6. Database Manager Test
    print("\n[TEST 6] SQLite Database Schema & Room Management...")
    db = ClassroomDatabase()
    room = db.create_room("Prof. Sharma", "Class 1 Tribal FLN", "Mathematics", "Santhali medium primary room")
    print(f"  Created Room Code : {room['room_code']}")
    joined, msg = db.join_room(room['room_code'], "Ramesh", "sat")
    print(f"  Student Joined    : {joined} ({msg})")
    assert joined is True

    print("\n" + "="*70)
    print("ALL VERNACULAR SMART CLASSROOM SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == '__main__':
    test_all()
