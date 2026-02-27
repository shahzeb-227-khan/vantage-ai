"""
hand_module.py — Gesture firmness tracker (Phase 3)

Tracks wrist velocity and jerk (rate of velocity change) to score how
controlled and deliberate hand movements are.

  Still hands          → firmness ~100
  Slow, smooth moves   → firmness 70–90
  Shaky / rapid jerk   → firmness 0–40

Uses the MediaPipe Tasks API (compatible with mediapipe 0.10+).
Model is downloaded automatically on first run.
"""

import os
import cv2
import requests
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque

# ---------- Model ----------
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def _ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand landmarker model...")
        r = requests.get(HAND_MODEL_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(HAND_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("Hand model ready.")

# ---------- Hand landmark indices ----------
WRIST     = 0
INDEX_TIP = 8

# Connections for manual skeleton drawing (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class GestureFirmness:
    """
    Single-hand firmness scorer based on wrist jerk.

    Algorithm:
        1. Track wrist position each frame.
        2. Velocity  = distance between consecutive wrist positions.
        3. Jerk      = |velocity_t − velocity_{t−1}|
           (high jerk = nervous / sudden movements)
        4. Map jerk to 0–100 score.
        5. Apply EMA smoothing for stable display.
    """

    def __init__(
        self,
        history_len: int = 30,
        max_jerk: float  = 15.0,   # px/frame jerk that maps to score 0
        ema_alpha: float = 0.35,
    ):
        self.max_jerk  = max_jerk
        self.ema_alpha = ema_alpha

        self._pos_history = deque(maxlen=history_len)
        self._vel_history = deque(maxlen=history_len)
        self._score: float = 100.0
        self._hand_visible = False
        self._last_result  = None   # cache latest result for draw()

        _ensure_hand_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def score(self) -> float:
        return self._score

    @property
    def hand_visible(self) -> bool:
        return self._hand_visible

    def process(self, mp_image: mp.Image, timestamp_ms: int, w: int, h: int) -> float:
        """Run detection, update firmness score. Returns current score 0–100."""
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        self._last_result = result

        if result.hand_landmarks:
            self._hand_visible = True
            hand = result.hand_landmarks[0]

            # Wrist position in pixels
            wrist = np.array([hand[WRIST].x * w, hand[WRIST].y * h])
            self._pos_history.append(wrist)

            if len(self._pos_history) >= 2:
                vel = np.linalg.norm(
                    self._pos_history[-1] - self._pos_history[-2]
                )
                self._vel_history.append(vel)

                if len(self._vel_history) >= 2:
                    jerk = abs(self._vel_history[-1] - self._vel_history[-2])
                    raw  = max(0.0, 100.0 * (1.0 - jerk / self.max_jerk))
                    self._score = (
                        self.ema_alpha * raw
                        + (1.0 - self.ema_alpha) * self._score
                    )
        else:
            self._hand_visible = False
            # No hand → drift toward 50 (absence ≠ nervousness)
            self._score = self.ema_alpha * 50.0 + (1.0 - self.ema_alpha) * self._score

        return self._score

    def draw(self, bgr_frame: np.ndarray, w: int, h: int) -> None:
        """Draw hand skeleton onto bgr_frame in-place using the last detection result."""
        if self._last_result is None or not self._last_result.hand_landmarks:
            return
        for hand in self._last_result.hand_landmarks:
            pts = {
                i: (int(lm.x * w), int(lm.y * h))
                for i, lm in enumerate(hand)
            }
            # Draw connections
            for a, b in HAND_CONNECTIONS:
                cv2.line(bgr_frame, pts[a], pts[b], (0, 220, 120), 2)
            # Draw joints
            for pt in pts.values():
                cv2.circle(bgr_frame, pt, 4, (255, 255, 255), -1)

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
