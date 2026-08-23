import os
import sys
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_engine import (
    evaluate_stabilized_prediction,
    CONFIDENCE_THRESHOLD,
    PROB_MARGIN_THRESHOLD,
    REQUIRED_VOTES,
    UNCERTAIN_FALLBACK_TEXT
)

def test_stabilization_logic():
    print("\n" + "="*65)
    print("=== TESTING REAL-TIME PREDICTION STABILIZATION LOGIC ===")
    print("="*65)
    
    # ----------------------------------------------------
    # TEST 1: Top-1 vs Top-2 Margin Check (Uncertain case)
    # Top 1: 42%, Top 2: 39% (Margin = 3% < 10%)
    # ----------------------------------------------------
    history1 = deque(maxlen=7)
    top_preds_close = [
        {'sentence': 'what happened', 'confidence': 42.0},
        {'sentence': 'what do you do', 'confidence': 39.0},
        {'sentence': 'where are you from', 'confidence': 19.0}
    ]
    p_text, p_conf, p_stable, p_uncertain, p_margin, p_votes = evaluate_stabilized_prediction(top_preds_close, history1)
    print("\n[Test 1] Close Top-1/Top-2 Margin (42% vs 39%):")
    print(f"  Predicted Text : '{p_text}'")
    print(f"  Confidence     : {p_conf}%")
    print(f"  Prob Margin    : {p_margin}% (Threshold >= {PROB_MARGIN_THRESHOLD*100}%)")
    print(f"  Stable         : {p_stable}")
    print(f"  Uncertain      : {p_uncertain}")
    assert p_uncertain == True, "Test 1 Failed: Should be marked uncertain"
    assert p_text == UNCERTAIN_FALLBACK_TEXT, "Test 1 Failed: Should fallback to default message"
    print("  -> PASSED")

    # ----------------------------------------------------
    # TEST 2: Low Confidence Check (Uncertain case)
    # Top 1: 35% (< 40%)
    # ----------------------------------------------------
    history2 = deque(maxlen=7)
    top_preds_low = [
        {'sentence': 'let him take time', 'confidence': 35.0},
        {'sentence': 'serve the food', 'confidence': 20.0},
        {'sentence': 'take care of yourself', 'confidence': 15.0}
    ]
    p_text, p_conf, p_stable, p_uncertain, p_margin, p_votes = evaluate_stabilized_prediction(top_preds_low, history2)
    print("\n[Test 2] Low Top-1 Confidence (35% < 40%):")
    print(f"  Predicted Text : '{p_text}'")
    print(f"  Uncertain      : {p_uncertain}")
    assert p_uncertain == True, "Test 2 Failed: Should be marked uncertain"
    print("  -> PASSED")

    # ----------------------------------------------------
    # TEST 3: Temporal Majority Consensus (Stable case)
    # 3 consecutive windows of 'what happened' (84% conf, 76% margin)
    # ----------------------------------------------------
    history3 = deque(maxlen=7)
    top_preds_stable = [
        {'sentence': 'what happened', 'confidence': 84.0},
        {'sentence': 'what do you do', 'confidence': 8.0},
        {'sentence': 'where are you from', 'confidence': 5.0}
    ]
    
    # Push 3 windows of 'what happened'
    for i in range(3):
        p_text, p_conf, p_stable, p_uncertain, p_margin, p_votes = evaluate_stabilized_prediction(top_preds_stable, history3)
        
    print("\n[Test 3] 3 Windows Consensus ('what happened' @ 84% conf):")
    print(f"  Predicted Text : '{p_text}'")
    print(f"  Confidence     : {p_conf}%")
    print(f"  Prob Margin    : {p_margin}%")
    print(f"  Vote Count     : {p_votes}/{REQUIRED_VOTES}")
    print(f"  Stable         : {p_stable}")
    print(f"  Uncertain      : {p_uncertain}")
    assert p_stable == True, "Test 3 Failed: Should be marked stable"
    assert p_uncertain == False, "Test 3 Failed: Should not be uncertain"
    assert p_text == "what happened", "Test 3 Failed: Should return 'what happened'"
    print("  -> PASSED")

    print("\n" + "="*65)
    print("=== ALL STABILIZATION TESTS PASSED SUCCESSFULLY ===")
    print("="*65 + "\n")

if __name__ == '__main__':
    test_stabilization_logic()
