# VANTAGE

### Multimodal Behavioral Intelligence Platform for Human Readiness Analysis

**VANTAGE** is an AI-powered real-time platform that estimates **human confidence as a behavioral stability construct** — not an emotional or psychological state. By analyzing **gaze steadiness**, **speech patterns**, **gesture consistency**, and **face engagement**, VANTAGE computes a live **Behavioral Readiness Index** using interpretable, explainable methods.

Designed for interviews, presentation training, and performance preparation — built with privacy, ethics, and transparency at its core.

---

## Problem Statement

Confidence is critical in communication, leadership, and interviews. Yet existing AI tools fail in three key ways:

1. **Emotion ≠ Confidence** — Facial emotion detection does not measure readiness or decisiveness
2. **Subjective Evaluation** — Human judgment introduces inconsistency and bias
3. **Black-box Models** — Opaque outputs reduce trust and real-world adoption

Confidence is fundamentally **behavioral**, measurable through:
- Stability of gaze over time
- Smoothness and control of physical movement
- Continuity and decisiveness of speech
- Absence of nervous, repetitive micro-patterns

There is currently **no lightweight, real-time system** that measures confidence objectively without inferring emotion, personality, or mental state.

---

## Solution

VANTAGE introduces a **Multimodal Behavioral Fusion Framework** built on four observable dimensions:

| Modality | Signal Measured | Weight |
|---|---|---|
| 👁 Gaze Stability | Iris deviation from neutral baseline | 0.35 |
| 🎙 Speech Decisiveness | Pause / hesitation detection via RMS energy | 0.30 |
| ✋ Gesture Firmness | Wrist jerk + sustained tremble detection | 0.20 |
| 🧠 Face Engagement | Sustained face hiding / occlusion | 0.15 |

These are fused into a single **Behavioral Confidence Index** with a five-tier label: `Commanding / Confident / Moderate / Hesitant / Low Confidence`.

When the microphone is muted, weights adapt automatically (3-signal mode: 0.50 / 0.30 / 0.20).

---

## Key Features

- Real-time webcam analysis at 25–30 FPS
- Two-phase gaze model: calibration → deviation scoring
- Two-layer hand model: instant jerk + sustained tremble detection
- Amplitude-based speech hesitation detection (no speech-to-text)
- Sustained face-hiding engagement tracker
- Weighted multimodal score fusion with adaptive mic-mute handling
- Session recording with JSON persistence and improvement tips
- Streamlit web dashboard with SVG gauges and live signal bars
- Cloud deployment support via snapshot-based analysis
- Fast and lightweight — no GPU required
- Fully modular architecture with clean separation of concerns
- Ethical, non-judgmental design — no emotion or personality labeling

---

## System Architecture

```
Webcam / Mic Feed
       ↓
┌──────────────────────────────────────────────────────────────┐
│                    app.py (Streamlit UI)                      │
│                                                              │
│   ┌──────────────────┐                                       │
│   │ core/             │                                       │
│   │  camera_manager   │ ← singleton capture thread            │
│   │  confidence_engine│ ← orchestrates all modules            │
│   │  session_manager  │ ← records + persists sessions         │
│   │  engagement_module│ ← face visibility tracking            │
│   │  fusion           │ ← weighted score combination          │
│   └──────────────────┘                                       │
│                                                              │
│   ┌──────────────────┐  ┌──────────────────┐                 │
│   │ vision/           │  │ audio/            │                 │
│   │  eye_module       │  │  speech_module    │                 │
│   │  hand_module      │  └──────────────────┘                 │
│   │  frame_analyzer   │                                       │
│   └──────────────────┘                                       │
└──────────────────────────────────────────────────────────────┘
       ↓
  Live Dashboard + Session History
```

---

## How It Works

### 👁 Gaze Stability (`vision/eye_module.py`)

Uses **MediaPipe FaceLandmarker** with iris center landmarks (indices 468 and 473).

**Phase 1 — Calibration (~2 seconds)**

60 frames of head-relative iris positions are averaged into a personal neutral gaze baseline.

**Phase 2 — Deviation Scoring**

Each frame, iris position relative to the nose-tip anchor is compared to the baseline:

```
distance ≤ 10 px     →  score = 100   (flat tolerance zone)
10 px < dist < 40 px →  score ramps linearly 100 → 0
distance ≥ 40 px     →  score = 0     (fully looking away)
```

### 🎙 Speech Decisiveness (`audio/speech_module.py`)

Pure amplitude-based analysis — no speech-to-text, no NLP. Monitors microphone RMS energy in real time within a 6-second rolling window:

- **Continuity ratio** — fraction of window that was active speech
- **Pause density** — completed pauses per 10-second window
- **Live silence penalty** — immediate reaction during ongoing pauses
- **Mic mute detection** — automatically returns `None` when mic is off

### ✋ Gesture Firmness (`vision/hand_module.py`)

Uses **MediaPipe HandLandmarker** to track wrist position each frame.

**Layer 1 — Instant Jerk Score**: Dead zone → moving average → exponential falloff. Deliberate-gesture bonus for smooth intentional movement.

**Layer 2 — Sustained Tremble**: Rolling counter over ~5 seconds. Only persistent trembling (>2 s) triggers a penalty, making the system behaviorally intelligent rather than frame-reactive.

### 🧠 Engagement (`core/engagement_module.py`)

Detects face hiding via:
1. No face detected → occluded
2. Hand landmark near nose tip → hand-over-face

Only sustained hiding (>2 s) penalises the score. Brief touches are ignored.

### 🔀 Score Fusion (`core/fusion.py`)

```
Confidence = 0.35 × eye + 0.30 × speech + 0.20 × hand + 0.15 × engagement
```

When mic is muted: `0.50 × eye + 0.30 × hand + 0.20 × engagement`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | MediaPipe FaceLandmarker, HandLandmarker |
| Audio Analysis | sounddevice (RMS energy) |
| Video Capture | OpenCV |
| Web UI | Streamlit |
| Numerical Processing | NumPy |
| Language | Python 3.10+ |

---

## Ethical & Responsible AI

VANTAGE explicitly does **not**:
- Diagnose psychological or mental health states
- Infer intelligence, personality, or character
- Replace human evaluators or inform hiring decisions

It measures **observable behavioral signals only**, using fully transparent mathematical formulas. Every score component is explainable and traceable to a specific signal.

---

## Installation & Setup

**Requirements:** Python 3.10+, a webcam, a microphone (optional)

```bash
git clone https://github.com/your-username/vantage.git
cd vantage
pip install -r requirements.txt
```

**Run:**

```bash
streamlit run app.py
```

MediaPipe models (`face_landmarker.task`, `hand_landmarker.task`) are downloaded automatically on first run (~6 MB total) into the `models/` directory.

---

## Project Structure

```
vantage/
├── app.py                          ← Streamlit entry point & UI
├── requirements.txt                ← Python dependencies
├── packages.txt                    ← System packages (Streamlit Cloud)
├── README.md
│
├── core/
│   ├── camera_manager.py           ← Singleton camera capture thread
│   ├── confidence_engine.py        ← Background processing engine
│   ├── engagement_module.py        ← Face visibility tracking
│   ├── fusion.py                   ← Weighted multimodal score fusion
│   └── session_manager.py          ← Session recording & persistence
│
├── vision/
│   ├── eye_module.py               ← Gaze stability tracker
│   ├── hand_module.py              ← Gesture firmness tracker
│   └── frame_analyzer.py           ← Single-frame cloud analyzer
│
├── audio/
│   └── speech_module.py            ← Speech hesitation detector
│
├── models/
│   ├── face_landmarker.task        ← MediaPipe face model (auto-downloaded)
│   └── hand_landmarker.task        ← MediaPipe hand model (auto-downloaded)
│
├── data/
│   └── history.json                ← Session history (auto-generated)
│
└── utils/
    └── config.py                   ← Centralized paths & constants
```

---

## Hackathon Information

- **Event:** Dev Season of Code (DSOC) 2026
- **Theme:** AI / Machine Learning for Social Good
- **Built during hackathon:** Yes

---
