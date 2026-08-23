import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation_memory import ShortTermPredictionMemory, ConversationMemoryStore

def test_short_term_prediction_memory():
    print("\n---> Testing Layer 1: Short-Term Prediction Memory Buffer")
    st_mem = ShortTermPredictionMemory(buffer_size=5)

    # 1. Add noisy predictions
    st_mem.add_prediction("what happened", 0.85)
    st_mem.add_prediction("what happened", 0.88)
    st_mem.add_prediction("what do you do", 0.42)
    st_mem.add_prediction("what happened", 0.91)
    st_mem.add_prediction("what happened", 0.87)

    stable_pred, conf, is_stable = st_mem.get_stable_prediction(min_votes=3, min_confidence=0.40)
    print(f"Prediction Buffer Consensus -> Stable Pred: '{stable_pred}', Conf: {conf:.2f}, Is Stable: {is_stable}")
    assert is_stable == True, "Expected prediction buffer consensus to accept 'what happened'"
    assert stable_pred == "what happened", "Expected consensus prediction to be 'what happened'"
    print("✅ Layer 1 Short-Term Prediction Memory Buffer Test Passed!")

def test_conversation_memory_store():
    print("\n---> Testing Layer 2: Conversation Memory Store (SQLite Persistence)")
    db_test_path = os.path.join(os.path.dirname(__file__), 'test_conversation_memory.db')
    if os.path.exists(db_test_path):
        os.remove(db_test_path)

    store = ConversationMemoryStore(db_path=db_test_path)
    store.clear_history()

    # Add messages
    m1 = store.add_message("Teacher", "Open your book", 1.0)
    m2 = store.add_message("Student", "i do not understand", 0.89)
    m3 = store.add_message("Teacher", "I will explain again", 1.0)

    history = store.get_history()
    print(f"Conversation History Length: {len(history)}")
    assert len(history) == 3, f"Expected 3 messages, got {len(history)}"

    assert history[0]['sender'] == "Teacher" and history[0]['message'] == "Open your book"
    assert history[1]['sender'] == "Student" and history[1]['message'] == "i do not understand"
    assert history[2]['sender'] == "Teacher" and history[2]['message'] == "I will explain again"

    if os.path.exists(db_test_path):
        os.remove(db_test_path)

    print("✅ Layer 2 Conversation Memory Store SQLite Test Passed!")

if __name__ == '__main__':
    print("="*70)
    print("=== RUNNING CONVERSATION MEMORY & PREDICTION BUFFER TEST SUITE ===")
    print("="*70)
    test_short_term_prediction_memory()
    test_conversation_memory_store()
    print("\n" + "="*70)
    print("=== ALL CONVERSATION MEMORY TESTS PASSED SUCCESSFULLY ===")
    print("="*70 + "\n")
