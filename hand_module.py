"""
hand_module.py — Gesture firmness tracker (Phase 3)

Two-layer scoring:
  1. Instant jerk score  — reacts to abrupt single-frame movements
  2. Sustained tremble   — detects oscillation persisting for ~5 seconds

Final score = instant_score penalised by tremble_ratio.

Expected behaviour:
  Hands relaxed              → ~100
  Smooth deliberate gesture  → 80–95
  Single sudden snap         → brief dip, recovers in < 1 s
  Trembling for 5 s          → sustained drop to 30–50
  Continuous fidget          → sustained low score

Uses the MediaPipe Tasks API (compatible with mediapipe 0.10+).
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
WRIST = 0

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
    Two-layer hand firmness scorer.

    Layer 1 — Instant jerk (per-frame):
        • Dead zone filters tracking noise
        • 5-frame moving average smooths spikes
        • Exponential falloff maps smoothed jerk → instant_score

    Layer 2 — Sustained tremble (time-window):
        • BUFFER_SIZE frames (~5 s at 30 fps) rolling window
        • Each frame: if jerk > TREMBLE_THRESHOLD → counter +1, else −1
        • tremble_ratio = counter / BUFFER_SIZE  (0.0 – 1.0)
        • Only penalises when ratio > TREMBLE_ONSET (default 0.4 = ~2 s sustained)
        • Penalty scales up to TREMBLE_MAX_PENALTY at TREMBLE_FULL (ratio 0.7)

    Final score = instant_score − tremble_penalty, clamped to [0, 100].
    """

    # --- Instant jerk tuning ---
    JERK_DEAD_ZONE   = 4.0   # px/frame: noise floor, ignored
    JERK_SCALE       = 8.0   # px/frame: exponential decay constant
    JERK_SMOOTH_LEN  = 5     # frames:   moving-average window
    VEL_DELIBERATE   = 12.0  # px/frame: velocity for deliberate-gesture bonus
    DELIBERATE_BONUS = 5.0   # score pts

    # --- Sustained tremble tuning ---
    BUFFER_SIZE        = 150   # frames ≈ 5 s at 30 fps
    TREMBLE_THRESHOLD  = 3.0   # px/frame effective jerk that counts as a tremble frame
    TREMBLE_ONSET      = 0.40  # tremble_ratio at which penalty starts (40 % of window)
    TREMBLE_FULL       = 0.70  # tremble_ratio at which max penalty is applied
    TREMBLE_MAX_PENALTY = 45.0 # max score points deducted by sustained tremble

    # --- Display ---
    EMA_ALPHA = 0.30

    def __init__(self):
        self._pos_history   = deque(maxlen=30)
        self._vel_history   = deque(maxlen=30)
        self._jerk_smooth   = deque(maxlen=self.JERK_SMOOTH_LEN)

        # Sustained tremble counter (0 … BUFFER_SIZE)
        self._tremble_counter: float = 0.0

        self._score: float = 100.0
        self._hand_visible = False
        self._last_result  = None

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
    def last_hand_result(self):
        """Raw HandLandmarker result from last detection (for engagement module)."""
        return self._last_result

    @property
    def hand_visible(self) -> bool:
        return self._hand_visible

    @property
    def tremble_ratio(self) -> float:
        """0.0 = no sustained tremble, 1.0 = constant tremble."""
        return self._tremble_counter / self.BUFFER_SIZE

    def process(self, mp_image: mp.Image, timestamp_ms: int, w: int, h: int) -> float:
        """Run detection, update firmness score. Returns current score 0–100."""
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        self._last_result = result

        if result.hand_landmarks:
            self._hand_visible = True
            hand  = result.hand_landmarks[0]
            wrist = np.array([hand[WRIST].x * w, hand[WRIST].y * h])
            self._pos_history.append(wrist)

            if len(self._pos_history) >= 2:
                vel = float(np.linalg.norm(
                    self._pos_history[-1] - self._pos_history[-2]
                ))
                self._vel_history.append(vel)

                if len(self._vel_history) >= 2:
                    raw_jerk       = abs(self._vel_history[-1] - self._vel_history[-2])
                    effective_jerk = max(0.0, raw_jerk - self.JERK_DEAD_ZONE)

                    # --- Layer 1: instant score ---
                    self._jerk_smooth.append(effective_jerk)
                    smoothed_jerk = float(np.mean(self._jerk_smooth))
                    instant_score = 100.0 * np.exp(-smoothed_jerk / self.JERK_SCALE)

                    # Deliberate gesture bonus
                    if vel > self.VEL_DELIBERATE and smoothed_jerk < self.JERK_DEAD_ZONE:
                        instant_score = min(100.0, instant_score + self.DELIBERATE_BONUS)

                    # --- Layer 2: sustained tremble counter ---
                    # +1 if this frame is a tremble frame, −1 if stable
                    if effective_jerk > self.TREMBLE_THRESHOLD:
                        self._tremble_counter = min(
                            self.BUFFER_SIZE,
                            self._tremble_counter + 1.0
                        )
                    else:
                        self._tremble_counter = max(
                            0.0,
                            self._tremble_counter - 1.0
                        )

                    # --- Tremble penalty (only after sustained onset) ---
                    ratio = self.tremble_ratio
                    if ratio <= self.TREMBLE_ONSET:
                        penalty = 0.0
                    elif ratio >= self.TREMBLE_FULL:
                        penalty = self.TREMBLE_MAX_PENALTY
                    else:
                        # Linear ramp between onset and full
                        t = (ratio - self.TREMBLE_ONSET) / (self.TREMBLE_FULL - self.TREMBLE_ONSET)
                        penalty = t * self.TREMBLE_MAX_PENALTY

                    # --- Combine & smooth ---
                    raw_final = max(0.0, instant_score - penalty)
                    self._score = float(
                        self.EMA_ALPHA * raw_final
                        + (1.0 - self.EMA_ALPHA) * self._score
                    )
        else:
            self._hand_visible = False
            # Tremble counter decays when hand is absent
            self._tremble_counter = max(0.0, self._tremble_counter - 1.0)
            # Score drifts toward 50
            self._score = float(
                self.EMA_ALPHA * 50.0 + (1.0 - self.EMA_ALPHA) * self._score
            )

        return self._score

    def draw(self, bgr_frame: np.ndarray, w: int, h: int) -> None:
        """Draw hand skeleton and tremble indicator onto bgr_frame in-place."""
        if self._last_result is None or not self._last_result.hand_landmarks:
            return
        for hand in self._last_result.hand_landmarks:
            pts = {
                i: (int(lm.x * w), int(lm.y * h))
                for i, lm in enumerate(hand)
            }
            # Skeleton colour shifts red when trembling
            ratio      = self.tremble_ratio
            skel_color = (
                int(ratio * 80),
                int(220 * (1 - ratio)),
                int(ratio * 200),
            )
            for a, b in HAND_CONNECTIONS:
                cv2.line(bgr_frame, pts[a], pts[b], skel_color, 2)
            for pt in pts.values():
                cv2.circle(bgr_frame, pt, 4, (255, 255, 255), -1)

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
