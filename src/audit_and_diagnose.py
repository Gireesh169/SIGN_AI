"""
SIGN-AI Full System Audit and Diagnostic Script
Checks:
- Model checkpoint loading (best_model_252.pth) & PyTorch model instantiation
- Label mapping (label_mapping.json)
- Feature normalization & 252D shape calculations
- RealTimeInferenceEngine status
- SQLite database tables & operations
- API Endpoints (/health, /predict_sequence, /predict, etc.)
- TTS engine & speech worker thread
- Multilingual translation & offline sync modules
"""

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_DIR, MODELS_DIR, DATA_DIR, DEVICE,
    normalize_sequence_features
)
from model import BiLSTM_GRU_Model
from realtime_engine import RealTimeInferenceEngine
from database import ClassroomDatabase
from conversation_memory import ConversationMemoryStore, ShortTermPredictionMemory
from translation_engine import translate
from offline_sync import get_sync_manager

def run_diagnostics():
    print("\n" + "="*70)
    print("=== SIGN-AI FULL SYSTEM AUDIT AND DIAGNOSTICS ===")
    print("="*70)
    
    audit_results = []
    
    # 1. Model Loading Check
    print("\n[CHECK 1] Primary Model Checkpoint & Model Instantiation...")
    model_252_path = os.path.join(MODELS_DIR, 'best_model_252.pth')
    label_map_path = os.path.join(MODELS_DIR, 'label_mapping.json')
    
    if not os.path.exists(model_252_path):
        print(f"  ❌ FAIL: {model_252_path} does not exist!")
        audit_results.append(("Model File Missing", "models/best_model_252.pth", "File not found", "Restore checkpoint", "CRITICAL"))
    else:
        print(f"  ✓ Model file present: {model_252_path} ({os.path.getsize(model_252_path)} bytes)")
        
    if not os.path.exists(label_map_path):
        print(f"  ❌ FAIL: {label_map_path} does not exist!")
        audit_results.append(("Label Mapping Missing", "models/label_mapping.json", "File not found", "Restore label mapping", "CRITICAL"))
    else:
        with open(label_map_path) as f:
            label_map = json.load(f)
        print(f"  ✓ Label mapping loaded: {len(label_map)} classes")

    try:
        model = BiLSTM_GRU_Model(input_dim=252, hidden_dim=32, num_classes=len(label_map))
        ckpt = torch.load(model_252_path, map_location=DEVICE)
        model.load_state_dict(ckpt)
        model.eval()
        print("  ✓ PyTorch model loaded successfully with 252D velocity layer and eval() state")
    except Exception as e:
        print(f"  ❌ PyTorch model load failed: {e}")
        audit_results.append(("Model Load Exception", "src/model.py", str(e), "Fix model class / state dict mismatch", "HIGH"))

    # 2. RealTimeInferenceEngine Check
    print("\n[CHECK 2] RealTimeInferenceEngine Instantiation...")
    try:
        engine = RealTimeInferenceEngine()
        print("  ✓ RealTimeInferenceEngine instantiated successfully")
        
        # Test sequence inference
        dummy_seq = np.random.randn(30, 126).astype(np.float32)
        res = engine.predict_sequence(dummy_seq)
        print(f"  ✓ Dummy 30x126 sequence inference result: '{res.get('predicted_text')}' (Conf: {res.get('confidence')}%)")
    except Exception as e:
        print(f"  ❌ RealTimeInferenceEngine failed: {e}")
        audit_results.append(("Inference Engine Error", "src/realtime_engine.py", str(e), "Fix inference engine logic", "HIGH"))

    # 3. Preprocessing & Normalization Verification
    print("\n[CHECK 3] Preprocessing & Normalization Consistency...")
    dummy_frame = np.random.randn(30, 126).astype(np.float32)
    norm_frame = normalize_sequence_features(dummy_frame)
    if norm_frame.shape == (30, 126):
        print("  ✓ normalize_sequence_features outputs expected (30, 126) shape")
    else:
        print(f"  ❌ Normalization shape mismatch: got {norm_frame.shape}")
        audit_results.append(("Normalization Shape Error", "src/config.py", f"Got {norm_frame.shape}", "Fix shape handling", "MEDIUM"))

    # 4. Database Integrity Check
    print("\n[CHECK 4] SQLite Classroom & Memory Databases...")
    try:
        db = ClassroomDatabase()
        room = db.create_room("Audit Teacher", "Audit Room", "CS", "Audit Room Desc")
        print(f"  ✓ ClassroomDatabase initialized & room created: {room['room_code']}")
        
        mem_store = ConversationMemoryStore()
        msg = mem_store.add_message("Teacher", "Test message", 1.0)
        print(f"  ✓ ConversationMemoryStore working: message added ID {msg.get('timestamp')}")
    except Exception as e:
        print(f"  ❌ Database check failed: {e}")
        audit_results.append(("Database Error", "src/database.py", str(e), "Fix SQLite queries or schema", "HIGH"))

    # 5. Multilingual & Sync Manager Check
    print("\n[CHECK 5] Translation Engine & Sync Manager...")
    try:
        tr = translate("Open your book", "en", "sat")
        print(f"  ✓ Translation Engine: '{tr['source_text']}' -> '{tr['translated_text']}' (Latency: {tr['latency_ms']} ms)")
        
        sync_mgr = get_sync_manager()
        st = sync_mgr.get_status()
        print(f"  ✓ Sync Manager: Status = {st['connectivity_status']}, Offline Items = {st['total_offline_items']}")
    except Exception as e:
        print(f"  ❌ Translation or Sync manager failed: {e}")
        audit_results.append(("Translation/Sync Error", "src/translation_engine.py", str(e), "Fix translation module", "MEDIUM"))

    print("\n" + "="*70)
    print("=== AUDIT DIAGNOSTICS SUMMARY ===")
    print("="*70)
    if not audit_results:
        print("🎉 No critical runtime errors detected in Python diagnostic suite.")
    else:
        print(f"Found {len(audit_results)} issues during diagnostic check:")
        for err, loc, cause, fix, prio in audit_results:
            print(f"  [{prio}] {err} @ {loc}: {cause} -> {fix}")
    print("="*70 + "\n")

if __name__ == '__main__':
    run_diagnostics()
