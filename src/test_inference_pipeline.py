import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS_DIR
from realtime_engine import RealTimeInferenceEngine

class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.engine = RealTimeInferenceEngine()

    def test_model_and_label_mapping_verification(self):
        print("\n---> Verifying Model & Label Mapping Startup Parameters")
        self.assertEqual(self.engine.feature_dim, 252)
        self.assertEqual(self.engine.sequence_length, 30)
        self.assertEqual(self.engine.num_classes, 101)
        self.assertIsNotNone(self.engine.model)
        print(f"✅ Loaded Model: best_model_252.pth | Feature Dim: 252 | Classes: 101")

    def test_252d_sequence_inference(self):
        print("\n---> Verifying 252D Position + Velocity 30-Frame Sequence Pipeline")

        # Create dummy (30, 126) position landmark sequence
        dummy_pos_30x126 = np.zeros((30, 126), dtype=np.float32)
        for t in range(30):
            # Left hand wrist & palm landmarks
            dummy_pos_30x126[t, :63:3] = 0.35 + (t * 0.002)
            dummy_pos_30x126[t, 1:63:3] = 0.50
            # Right hand wrist & palm landmarks
            dummy_pos_30x126[t, 63::3] = 0.65 - (t * 0.002)
            dummy_pos_30x126[t, 64::3] = 0.50

        # Predict sequence
        result = self.engine.predict_sequence(dummy_pos_30x126)

        self.assertIn('predicted_text', result)
        self.assertIn('confidence', result)
        self.assertIn('top_predictions', result)
        self.assertGreaterEqual(len(result['top_predictions']), 1)

        print(f"Top Prediction: '{result['top_predictions'][0]['sentence']}' | Conf: {result['top_predictions'][0]['confidence']}%")
        print("Top 5 Predictions Breakdown:")
        for i, tp in enumerate(result['top_predictions'][:5], 1):
            print(f"  {i}. {tp['sentence']} — {tp['confidence']}%")

        print("✅ 252D 30-Frame Inference Pipeline Test Passed!")

if __name__ == '__main__':
    print("="*70)
    print("=== RUNNING INFERENCE PIPELINE DEBUG & VERIFICATION SUITE ===")
    print("="*70)
    unittest.main()
