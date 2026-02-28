"""
vision/eye_module.py — Gaze stability tracker.

Uses MediaPipe FaceLandmarker iris centers relative to the nose-tip anchor.
Calibrates a neutral baseline on startup, then scores ongoing deviation.

Two-phase model:
    Phase 1 — Calibration: collect CALIBRATION_FRAMES of head-relative iris
              offsets while the user looks at the screen.
    Phase 2 — Scoring: euclidean distance from baseline mapped to 0-100.
"""

import os

import numpy as np
import mediapipe as mp
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.config import FACE_MODEL_PATH, FACE_MODEL_URL

# Landmark indices
LEFT_IRIS = 468
RIGHT_IRIS = 473
NOSE_IDX = 1


def _ensure_model() -> None:
    """Download the face landmarker model if not already present."""
    if not os.path.exists(FACE_MODEL_PATH):
        os.makedirs(os.path.dirname(FACE_MODEL_PATH), exist_ok=True)
        r = requests.get(FACE_MODEL_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(FACE_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


class GazeTracker:
    """Two-phase gaze stability scorer.

    Phase 1 — Calibration:
        Collect ``calibration_frames`` of head-relative iris offsets.
        Their average becomes the neutral baseline.

    Phase 2 — Scoring:
        Euclidean distance of each iris from the baseline is mapped
        through a tolerance zone to a 0-100 score.
    """

    def __init__(
        self,
        calibration_frames: int = 60,
        gaze_tolerance: float = 10.0,
        gaze_max_dist: float = 40.0,
        ema_alpha: float = 0.35,
    ) -> None:
        self.calibration_frames = calibration_frames
        self.gaze_tolerance = gaze_tolerance
        self.gaze_max_dist = gaze_max_dist
        self.ema_alpha = ema_alpha

        self._calib_left: list = []
        self._calib_right: list = []
        self._baseline_left: np.ndarray | None = None
        self._baseline_right: np.ndarray | None = None
        self._score: float = 100.0
        self._last_face_landmarks = None

        _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_calibrated(self) -> bool:
        """True once the neutral gaze baseline has been established."""
        return self._baseline_left is not None

    @property
    def calibration_progress(self) -> float:
        """0.0 to 1.0 fraction of calibration completed."""
        return min(1.0, len(self._calib_left) / self.calibration_frames)

    @property
    def score(self) -> float:
        """Current gaze stability score 0-100."""
        return self._score

    @property
    def last_face_landmarks(self):
        """Raw face_landmarks list from last detection (used by engagement)."""
        return self._last_face_landmarks

    def process(self, mp_image: mp.Image, timestamp_ms: int, w: int, h: int) -> float:
        """Run detection and update score. Returns current score 0-100."""
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        self._last_face_landmarks = result.face_landmarks

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                self._update(landmarks, w, h)
        else:
            self._score *= 1.0 - self.ema_alpha

        return self._score

    def close(self) -> None:
        """Release the MediaPipe detector resources."""
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _update(self, landmarks, w: int, h: int) -> None:
        """Update calibration or score from a single set of face landmarks."""
        left_px = np.array([landmarks[LEFT_IRIS].x * w, landmarks[LEFT_IRIS].y * h])
        right_px = np.array([landmarks[RIGHT_IRIS].x * w, landmarks[RIGHT_IRIS].y * h])
        nose_px = np.array([landmarks[NOSE_IDX].x * w, landmarks[NOSE_IDX].y * h])

        left_rel = left_px - nose_px
        right_rel = right_px - nose_px

        if not self.is_calibrated:
            self._calib_left.append(left_rel)
            self._calib_right.append(right_rel)
            if len(self._calib_left) >= self.calibration_frames:
                self._baseline_left = np.mean(self._calib_left, axis=0)
                self._baseline_right = np.mean(self._calib_right, axis=0)
        else:
            dist_l = np.linalg.norm(left_rel - self._baseline_left)
            dist_r = np.linalg.norm(right_rel - self._baseline_right)
            avg = (dist_l + dist_r) / 2.0

            if avg <= self.gaze_tolerance:
                raw = 100.0
            else:
                span = self.gaze_max_dist - self.gaze_tolerance
                raw = max(0.0, 100.0 * (1.0 - (avg - self.gaze_tolerance) / span))

            self._score = float(
                self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._score
            )
