# VANTAGE

### Multimodal Behavioral Intelligence Platform for Human Readiness Analysis

**VANTAGE** is an AI-powered real-time platform that estimates **human confidence as a behavioral stability construct** — not an emotional or psychological state. By analyzing **gaze steadiness** and **gesture consistency**, VANTAGE computes a live **Behavioral Readiness Index** using interpretable, explainable methods.

Designed for interviews, presentation training, and performance preparation — built with privacy, ethics, and transparency at its core.

---

## 🚨 Problem Statement

Confidence is critical in communication, leadership, and interviews. Yet existing AI tools fail in three key ways:

1. **Emotion ≠ Confidence** — Facial emotion detection does not measure readiness or decisiveness
2. **Subjective Evaluation** — Human judgment introduces inconsistency and bias
3. **Black-box Models** — Opaque outputs reduce trust and real-world adoption

Confidence is fundamentally **behavioral**, measurable through:
- Stability of gaze over time
- Smoothness and control of physical movement
- Absence of nervous, repetitive micro-patterns

There is currently **no lightweight, real-time system** that measures confidence objectively without inferring emotion, personality, or mental state.

---

## 💡 Solution

VANTAGE introduces a **Multimodal Behavioral Fusion Framework** built on two observable dimensions:

| Modality | Signal Measured | Score |
|---|---|---|
| 👁 Visual Stability | Iris deviation from neutral gaze baseline | 0 – 100 |
| ✋ Gesture Firmness | Wrist jerk + sustained tremble detection | 0 – 100 |

These are fused into a single **Behavioral Confidence Index** with a human-readable label: `High / Moderate / Low`.

---

## ✨ Key Features

- 🎥 Real-time webcam analysis at 25–30 FPS
- 🧠 Two-phase gaze model: calibration → deviation scoring
- ✋ Two-layer hand model: instant jerk + sustained tremble detection
- 📊 Weighted multimodal score fusion
- ⚡ Fast and lightweight — no GPU required
- 🧩 Fully modular architecture (`eye_module`, `hand_module`, `fusion`)
- 🧭 Ethical, non-judgmental design — no emotion or personality labeling

---

## 🏗 System Architecture

```
Webcam Feed
     ↓
┌────────────────────────────────────────────┐
│              main.py (orchestrator)         │
│                                            │
│   ┌─────────────┐    ┌──────────────────┐  │
│   │ eye_module  │    │  hand_module     │  │
│   │             │    │                  │  │
│   │ Phase 1:    │    │ Layer 1: Instant  │  │
│   │  Calibrate  │    │   jerk score     │  │
│   │ Phase 2:    │    │ Layer 2: Sustained│  │
│   │  Deviation  │    │   tremble detect │  │
│   └──────┬──────┘    └────────┬─────────┘  │
│          └────────┬───────────┘            │
│               fusion.py                    │
│         Weighted confidence score          │
└────────────────────────────────────────────┘
     ↓
Live HUD overlay on video frame
```

---

## 🔬 How It Works

### 👁 Visual Stability (`eye_module.py`)

Uses **MediaPipe FaceLandmarker** with iris center landmarks (indices 468 and 473 — the actual eyeball centers, not eyelid contours).

**Phase 1 — Calibration (~2 seconds)**

The user looks at the screen. 60 frames of head-relative iris positions are averaged into a personal neutral gaze baseline. A progress bar is shown on screen during this phase.

**Phase 2 — Deviation Scoring**

Each frame, the iris position is expressed relative to the nose-tip anchor (removes head translation). Euclidean distance from the calibration baseline is mapped to a 0–100 score:

```
distance ≤ 10 px     →  score = 100   (flat tolerance zone)
10 px < dist < 40 px →  score ramps linearly 100 → 0
distance ≥ 40 px     →  score = 0     (fully looking away)
```

Pressing `R` recalibrates the baseline mid-session.

---

### ✋ Gesture Firmness (`hand_module.py`)

Uses **MediaPipe HandLandmarker** to track wrist position each frame.

**Layer 1 — Instant Jerk Score**

```
velocity     = distance(wrist_t, wrist_{t-1})
jerk         = |velocity_t − velocity_{t-1}|
effective    = max(0, jerk − dead_zone)      ← removes tracking noise
smoothed     = mean(last 5 frames)            ← prevents spike overreaction
score        = 100 × exp(−smoothed / 8.0)     ← gradual exponential falloff
```

A deliberate-gesture bonus (+5 pts) is applied when velocity is high but jerk is low, rewarding smooth and intentional movement.

**Layer 2 — Sustained Tremble Detection**

```
Each frame:
  jerk > threshold  →  tremble_counter + 1  (max = 150 ≈ 5 seconds at 30 fps)
  jerk ≤ threshold  →  tremble_counter − 1

tremble_ratio = counter / 150

ratio < 0.40          →  no penalty
ratio between 0.40–0.70  →  penalty ramps linearly 0 → 45 pts
ratio ≥ 0.70          →  −45 pts  (persistent trembling)
```

A **single sudden movement does not lower the score.** Only behavior sustained across multiple seconds is penalized, making the system behaviorally intelligent rather than frame-reactive.

---

### 🔀 Score Fusion (`fusion.py`)

```
Confidence = (0.6 × eye_score + 0.4 × hand_score)
```

Gaze is weighted more heavily as it is a more stable signal. Both weights are tunable constants.

| Score | Label |
|---|---|
| ≥ 80 | High |
| 55 – 79 | Moderate |
| < 55 | Low |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | MediaPipe FaceLandmarker, HandLandmarker |
| Video Capture | OpenCV |
| Numerical Processing | NumPy |
| Language | Python 3.10+ |
| Package Manager | uv |

---

## 🧭 Ethical & Responsible AI

VANTAGE explicitly does **not**:
- Diagnose psychological or mental health states
- Infer intelligence, personality, or character
- Replace human evaluators or inform hiring decisions

It measures **observable behavioral signals only**, using fully transparent mathematical formulas. Every score component is explainable and traceable to a specific signal.

---

## 🚀 Installation & Setup

**Requirements:** Python 3.10+, a webcam

```bash
git clone https://github.com/your-username/vantage.git
cd vantage

# Using pip
pip install opencv-python mediapipe numpy requests

# Or using uv (recommended)
uv sync
```

**Run:**

```bash
python main.py
```

MediaPipe models (`face_landmarker.task`, `hand_landmarker.task`) are downloaded automatically on first run (~6 MB total).

**Controls:**

| Key | Action |
|---|---|
| `R` | Recalibrate gaze baseline |
| `ESC` | Exit |

**Camera:** Defaults to index `0` (built-in webcam). Change `CAMERA_INDEX` at the top of `main.py` for external cameras or DroidCam.

---

## 📁 Project Structure

```
vantage/
├── main.py               ← Orchestration loop and HUD rendering
├── eye_module.py         ← GazeTracker — calibration + iris deviation scoring
├── hand_module.py        ← GestureFirmness — jerk + sustained tremble detection
├── fusion.py             ← Weighted multimodal score fusion
├── test_camera.py        ← Standalone single-file prototype (reference)
├── face_landmarker.task  ← MediaPipe face model (auto-downloaded)
├── hand_landmarker.task  ← MediaPipe hand model (auto-downloaded)
└── pyproject.toml
```

---

## 🏆 Hackathon Information

- **Event:** Dev Season of Code (DSOC) 2026
- **Theme:** AI / Machine Learning for Social Good
- **Built during hackathon:** ✅ Yes

---

## 🔮 Future Scope

- 🎤 **Audio modality** — speech hesitation, pause structure, filler word detection
- 🌐 **Web deployment** — FastAPI backend + React dashboard
- 📱 **Mobile support** — on-device inference via WASM or TFLite
- 🔒 **Full local processing** — no data ever leaves the device
- 📈 **Session history** — trend tracking and progress over time
- 🧪 **Personalized baselines** — model adapts to individual behavioral patterns

---
