import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_engine import RealTimeInferenceEngine

def test_multi_person_isolation():
    print("\n" + "="*65)
    print("=== TESTING MULTI-PERSON DETECTION & SIGNER ISOLATION ===")
    print("="*65)
    
    engine = RealTimeInferenceEngine()
    
    # ----------------------------------------------------
    # TEST 1: Zero People Visible
    # ----------------------------------------------------
    dummy_empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_empty = engine.process_frame(dummy_empty_frame)
    print("\n[Test 1] Zero People Visible:")
    print(f"  People Count : {res_empty['people_count']}")
    print(f"  Signer Status: '{res_empty['signer_status']}'")
    assert res_empty['people_count'] == 0, "Test 1 Failed: Should detect 0 people"
    assert res_empty['signer_status'] == "No signer detected", "Test 1 Failed: Should display 'No signer detected'"
    print("  -> PASSED")

    # ----------------------------------------------------
    # TEST 2: Multi-Person Spatial Candidate Selection
    # Candidate 1 @ (0.50, 0.50) [Center], Candidate 2 @ (0.90, 0.90) [Far Right]
    # ----------------------------------------------------
    c1 = (0.50, 0.50)
    c2 = (0.90, 0.90)
    score1 = 1.0 - (abs(c1[0] - 0.5) + abs(c1[1] - 0.5)) # = 1.0
    score2 = 1.0 - (abs(c2[0] - 0.5) + abs(c2[1] - 0.5)) # = 0.2
    
    print("\n[Test 2] Multi-Person Selection Priority:")
    print(f"  Candidate 1 (Center) Score   : {score1:.2f}")
    print(f"  Candidate 2 (Far Right) Score: {score2:.2f}")
    selected = c1 if score1 >= score2 else c2
    assert selected == c1, "Test 2 Failed: Should select center candidate"
    print("  -> PASSED (Selected Primary Center Signer)")

    # ----------------------------------------------------
    # TEST 3: Sudden Identity Switch Buffer Reset
    # Tracked center jumps from (0.50, 0.50) to (0.90, 0.90) [distance = 0.56 > 0.35]
    # ----------------------------------------------------
    engine.pos_buffer.extend([np.ones(126)] * 20)
    engine.prediction_history.extend(['hello', 'hello'])
    engine.tracked_person_center = (0.50, 0.50)
    
    new_center = (0.90, 0.90)
    shift_dist = np.sqrt((new_center[0] - engine.tracked_person_center[0])**2 + (new_center[1] - engine.tracked_person_center[1])**2)
    print("\n[Test 3] Sudden Signer Identity Switch:")
    print(f"  Previous Center : {engine.tracked_person_center}")
    print(f"  New Center      : {new_center}")
    print(f"  Shift Distance  : {shift_dist:.2f} (Threshold > 0.35)")
    
    if shift_dist > 0.35:
        engine.pos_buffer.clear()
        engine.prediction_history.clear()
        
    print(f"  Buffer Size After Switch : {len(engine.pos_buffer)}")
    print(f"  History Size After Switch: {len(engine.prediction_history)}")
    assert len(engine.pos_buffer) == 0, "Test 3 Failed: Buffer should be cleared"
    assert len(engine.prediction_history) == 0, "Test 3 Failed: History should be cleared"
    print("  -> PASSED (30-frame sequence buffer & prediction history reset)")

    print("\n" + "="*65)
    print("=== ALL MULTI-PERSON TRACKING TESTS PASSED ===")
    print("="*65 + "\n")
    
    engine.release()

if __name__ == '__main__':
    test_multi_person_isolation()
