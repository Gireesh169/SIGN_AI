# 🤟 SIGN_AI: Real-Time Indian Sign Language (ISL) Sentence Recognition

**SIGN_AI** is a state-of-the-art, real-time Indian Sign Language (ISL) sentence recognition system powered by a **252D Position + Causal Velocity Landmark Model** and a **BiLSTM + GRU** deep neural network architecture.

The project features **MediaPipe Hand Tracking**, **independent wrist-centering and palm-scale normalization**, **rolling 30-frame temporal sliding windows**, **temporal prediction hysteresis smoothing**, **non-blocking background text-to-speech (TTS)**, and both **Desktop OpenCV** and **Flask Web Dashboard** interfaces.

---

## 📊 Model Performance & Benchmarks

| Metric | Score / Benchmark |
| :--- | :--- |
| **Active Sentence Classes** | **97 ISL Classes** |
| **Top-1 Test Accuracy** | **36.08%** |
| **Top-5 Test Accuracy** | **70.10%** |
| **Macro F1 Score** | **0.3203** |
| **Weighted F1 Score** | **0.3203** |
| **Inference Latency** | **~4.5 ms / prediction** |
| **Real-Time Camera FPS** | **~30 FPS** |

---

## 🌟 Key Features

- **252D Kinematic Landmark Feature Vector**:
  - `0:63`: Left Hand normalized 3D landmark coordinates ($21 \text{ points} \times 3 \text{ coords}$).
  - `63:126`: Right Hand normalized 3D landmark coordinates ($21 \text{ points} \times 3 \text{ coords}$).
  - `126:252`: Causal First-Order Velocity features ($V[t] = P[t] - P[t-1]$ for $t \ge 1$, $V[0] = 0$).
- **True MediaPipe Handedness**: Un-flipped camera landmark detection ensures 100% consistent Left/Right hand ordering matching training data.
- **Rolling 30-Frame Sliding Window Buffer**: Continuous streaming predictions ($1\text{--}30, 2\text{--}31, 3\text{--}32\dots$) with zero buffer clearing stalls.
- **Temporal Prediction Hysteresis Smoothing**: 5-frame majority voting window requiring agreement and confidence gating before sentence emission.
- **Non-Blocking Background Text-To-Speech (TTS)**: Dedicated worker thread using `queue.Queue` and `subprocess` with duplicate speech prevention and text sanitization (handles sentences with parentheses or punctuation seamlessly).
- **Dual Execution Interfaces**:
  1. **Flask Web Application Dashboard**: Zero-permission server-side OpenCV video streaming (`/video_feed`), live HUD overlay, and video file upload testing.
  2. **Native OpenCV Desktop Camera App**: Ultra-fast desktop window with skeleton visualization and HUD metrics.
  3. **Offline Video File Inference CLI**: CLI script supporting video file processing.
- **Automated Test Suite**: 10 real-time scenario tests verifying buffer mechanics, stability, non-blocking camera loops, low-confidence gating, and system restart.

---

## 📁 Repository Structure

```text
Combine_model/
├── app.py                      # Flask Web Dashboard & MJPEG Video Streaming Server
├── requirements.txt            # Python Dependency Manifest
├── README.md                   # Project Documentation
├── preprocessing_config.json   # Feature Preprocessing Configuration
│
├── models/
│   ├── best_model_252.pth      # Winning 252D PyTorch Model Weights
│   ├── preprocessing_config.json
│   └── label_mapping.json      # 97-Class Label Mapping Manifest
│
├── src/
│   ├── realtime_engine.py      # Real-Time 252D Inference Engine & TTS Worker
│   ├── webcam_inference.py     # Native OpenCV Desktop Computer Camera Application
│   ├── video_inference.py      # Offline Video File Inference CLI Script
│   ├── test_inference_scenarios.py # Automated 10-Scenario Test Suite
│   ├── model.py                # BiLSTM + GRU Neural Network Architecture
│   ├── train.py                # Model Training Pipeline
│   └── config.py               # Path Constants & Landmark Normalization Utilities
│
├── templates/
│   └── index.html              # Modern Web Dashboard Frontend Interface
│
└── ISL_CSLRT_Corpus/           # Dataset Corpus (Videos & Sentence Frames)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Ensure you have **Python 3.9+** installed on your system.

```bash
# Clone the repository
git clone https://github.com/Gireesh169/SIGN_AI.git
cd SIGN_AI

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 2. Running the Project

#### Mode A: Flask Web Dashboard (Recommended)

Start the web server to access live camera stream and video file upload testing in your browser:

```bash
python app.py
```

Open your browser at **`http://127.0.0.1:5001`**.

**Features**:
- Server-side OpenCV live camera feed (`/video_feed`) — no browser camera permission prompts needed!
- Live hand count, 30-frame buffer indicator, recognized sentence, confidence progress bar, and speaking audio indicator.
- **Upload Video File** button to test any sign language video file.

---

#### Mode B: Native Desktop Computer Camera Window

Launch the standalone desktop camera application using OpenCV:

```bash
python src/webcam_inference.py
```

**Features**:
- Opens your local computer camera directly in a high-performance native window.
- Visual hand skeleton connections drawn in real time.
- Top HUD bar showing **Recognized Sentence**, **Confidence Score (%)**, **FPS (~30)**, and **Inference Latency (~4.5 ms)**.
- Press **`q`** or **`ESC`** in the window to quit.

---

#### Mode C: Offline Video File Inference CLI

Run inference on any pre-recorded sign language video file:

```bash
python src/video_inference.py --video_path "ISL_CSLRT_Corpus/Videos_Sentence_Level/thank you so much/thank you so much (1).MP4"
```

---

## 🧪 Running Automated Test Suite

Run the 10-scenario real-time automated test suite to verify pipeline integrity:

```bash
python src/test_inference_scenarios.py
```

**Verified Scenarios**:
1. `[No Hands]` — Handles missing hands cleanly.
2. `[Buffer Filling]` — Fills rolling 30-frame deque buffer correctly.
3. `[Stable Prediction]` — Validates high-confidence prediction emissions.
4. `[Duplicate Speech Stop]` — Prevents repeated TTS speech for identical sentences.
5. `[Sentence A -> B]` — Smooth transition between distinct recognized signs.
6. `[Low Confidence Gating]` — Displays low-confidence status without false speech.
7. `[Noisy Switch Prevent]` — Prevents rapid switching during transient noise.
8. `[Non-Blocking Camera]` — Verifies camera frame rate remains unblocked during inference/TTS.
9. `[Resource Release]` — Verifies buffer clearing mechanics.
10. `[System Restart]` — Confirms state reset cleanly.

---

## 🧠 Neural Architecture & Technical Details

```text
Input Sequence (30, 252)
       │
       ▼
   BiLSTM Layer (Hidden Dim: 128, Bidirectional)
       │
       ▼
    GRU Layer (Hidden Dim: 128)
       │
       ▼
  Dropout (0.3) & Dense Head
       │
       ▼
Output Logits (97 Active ISL Classes)
```

- **Feature Preprocessing**: Hand landmarks are centered to landmark `0` (Wrist) and divided by distance to landmark `9` (Middle MCP) for hand scale invariance.
- **Causal Velocity**: Velocity $V[t] = P[t] - P[t-1]$ for $t \ge 1$ captures gesture kinetics without using future frames.

---

## 📄 License & Acknowledgments

This project is built as part of research into real-time Indian Sign Language Recognition using deep kinetic temporal models.
Corpus dataset: **ISL CSLRT Corpus**.
