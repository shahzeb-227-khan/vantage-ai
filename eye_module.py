"""
eye_module.py — Gaze stability tracker (Phase 2)

Uses MediaPipe FaceLandmarker iris centers relative to the nose-tip anchor.
Calibrates a neutral baseline on startup, then scores ongoing deviation.
"""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import requests
import os

# ---------- Model ----------
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        r = requests.get(MODEL_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("Face model ready.")

# ---------- Landmark indices ----------
LEFT_IRIS  = 468   # actual eyeball center (moves with gaze)
RIGHT_IRIS = 473
NOSE_IDX   = 1     # stable head anchor to cancel translation


class GazeTracker:
    """
    Two-phase gaze stability scorer.

    Phase 1 — Calibration:
        Collect CALIBRATION_FRAMES of head-relative iris offsets while the
        user looks at the screen.  Their average becomes the neutral baseline.

    Phase 2 — Scoring:
        Each frame, compute euclidean distance of each iris from the baseline.
        A flat tolerance zone gives a full 100 score for micro-movements.
        Beyond that, score ramps linearly to 0 at GAZE_MAX_DIST pixels.
    """

    def __init__(
        self,
        calibration_frames: int = 60,
        gaze_tolerance: float   = 10.0,
        gaze_max_dist: float    = 40.0,
        ema_alpha: float        = 0.35,
    ):
        self.calibration_frames = calibration_frames
        self.gaze_tolerance     = gaze_tolerance
        self.gaze_max_dist      = gaze_max_dist
        self.ema_alpha          = ema_alpha

        self._calib_left:  list = []
        self._calib_right: list = []
        self._baseline_left:  np.ndarray | None = None
        self._baseline_right: np.ndarray | None = None
        self._score: float = 100.0

        _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._baseline_left is not None

    @property
    def calibration_progress(self) -> float:
        """0.0 → 1.0 fraction of calibration completed."""
        return min(1.0, len(self._calib_left) / self.calibration_frames)

    @property
    def score(self) -> float:
        return self._score

    def process(self, mp_image: mp.Image, timestamp_ms: int, w: int, h: int) -> float:
        """Run detection and update score. Returns current score 0–100."""
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                self._update(landmarks, w, h)
        else:
            # No face: let score decay gradually
            self._score *= (1.0 - self.ema_alpha)

        return self._score

    def close(self):
        self._detector.close()

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update(self, landmarks, w: int, h: int):
        # Iris centers in pixel space
        left_px  = np.array([landmarks[LEFT_IRIS].x  * w, landmarks[LEFT_IRIS].y  * h])
        right_px = np.array([landmarks[RIGHT_IRIS].x * w, landmarks[RIGHT_IRIS].y * h])

        # Head-relative offsets (remove head translation)
        nose_px  = np.array([landmarks[NOSE_IDX].x * w, landmarks[NOSE_IDX].y * h])
        left_rel  = left_px  - nose_px
        right_rel = right_px - nose_px

        if not self.is_calibrated:
            # Phase 1 — collect baseline samples
            self._calib_left.append(left_rel)
            self._calib_right.append(right_rel)
            if len(self._calib_left) >= self.calibration_frames:
                self._baseline_left  = np.mean(self._calib_left,  axis=0)
                self._baseline_right = np.mean(self._calib_right, axis=0)
        else:
            # Phase 2 — measure deviation from neutral gaze
            dist_l = np.linalg.norm(left_rel  - self._baseline_left)
            dist_r = np.linalg.norm(right_rel - self._baseline_right)
            avg    = (dist_l + dist_r) / 2.0

            if avg <= self.gaze_tolerance:
                raw = 100.0
            else:
                span = self.gaze_max_dist - self.gaze_tolerance
                raw  = max(0.0, 100.0 * (1.0 - (avg - self.gaze_tolerance) / span))

            self._score = float(self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._score)
