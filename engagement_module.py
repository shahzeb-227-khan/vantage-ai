"""
engagement_module.py — Face Visibility & Behavioral Engagement Tracker

Detects when the user hides / occludes their face for a sustained period
and produces an engagement_score (0–100) that penalises only persistent
hiding, not brief face-touches.

Detection sources
─────────────────
1. No face detected at all  →  occluded frame
2. Any hand landmark within PROXIMITY_THRESHOLD (normalized coords) of the
   nose tip  →  hand-over-face frame

Temporal logic
──────────────
• A rolling "hidden counter" increments on occluded frames and decrements
  on clear frames, clamped to [0, BUFFER_SIZE] (≈ 5 s at 30 fps).
• Penalty only activates once hidden_ratio > HIDDEN_ONSET (~2 s sustained).
• At hidden_ratio == 1.0 the score hits FLOOR (not 0 — loss of tracking
  alone shouldn't fully zero the confidence).
• EMA alpha 0.2 gives a slow, gradual recovery after the face reappears.

The module is fully deterministic — no ML inference, no extra models.
"""

import math
from collections import deque

# ── Key landmark indices (MediaPipe canonical face mesh) ──────────────────
NOSE_IDX  = 1     # nose tip — stable reference for proximity detection
MOUTH_IDX = 13    # upper-lip center
LEYE_IDX  = 33    # left eye outer canthus
REYE_IDX  = 362   # right eye outer canthus


class EngagementTracker:
    """
    Sustained face-hiding detector producing an engagement_score 0–100.

    API:
        tracker = EngagementTracker()
        score   = tracker.process(face_landmarks_list, hand_landmarks_list)
        ratio   = tracker.hidden_ratio   # 0.0 = fully visible, 1.0 = constantly hidden

    Arguments to process():
        face_landmarks_list  – result.face_landmarks from FaceLandmarker
                               (list of NormalizedLandmark lists, or empty)
        hand_landmarks_list  – result.hand_landmarks from HandLandmarker
                               (list of NormalizedLandmark lists, or None/empty)
    """

    # ── Tunable constants ─────────────────────────────────────────────────
    BUFFER_SIZE        = 150    # frames ≈ 5 s at 30 fps
    PROXIMITY_THRESHOLD = 0.12  # normalized distance; hand landmark within
                                # this radius of nose → potential occlusion
    HIDDEN_ONSET       = 0.40   # ratio at which penalty begins  (~2 s of hiding)
    SCORE_FLOOR        = 20.0   # minimum engagement score (even fully hidden
                                # person may still be present; don't zero out)
    EMA_ALPHA          = 0.20   # slow smoothing — recovery should feel gradual

    def __init__(self):
        self._hidden_counter: float = 0.0
        self._score: float = 100.0

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def score(self) -> float:
        return self._score

    @property
    def hidden_ratio(self) -> float:
        """0.0 = fully visible every frame, 1.0 = hidden every frame."""
        return self._hidden_counter / self.BUFFER_SIZE

    def process(self, face_landmarks_list, hand_landmarks_list) -> float:
        """
        Update engagement score.

        Parameters
        ----------
        face_landmarks_list : list  (from FaceLandmarker result.face_landmarks)
        hand_landmarks_list : list | None  (from HandLandmarker result.hand_landmarks)

        Returns
        -------
        float : engagement_score 0–100
        """
        hidden = self._is_hidden(face_landmarks_list, hand_landmarks_list)

        # ── Sustained counter ──────────────────────────────────────────────
        # +1 per hidden frame, −1 per clear frame, clamped to [0, BUFFER_SIZE]
        if hidden:
            self._hidden_counter = min(self.BUFFER_SIZE, self._hidden_counter + 1.0)
        else:
            self._hidden_counter = max(0.0, self._hidden_counter - 1.0)

        ratio = self.hidden_ratio

        # ── Target score ──────────────────────────────────────────────────
        if ratio <= self.HIDDEN_ONSET:
            # Brief / no hiding — full engagement
            target = 100.0
        else:
            # Linear ramp from onset → floor as ratio → 1.0
            t      = (ratio - self.HIDDEN_ONSET) / (1.0 - self.HIDDEN_ONSET)
            target = max(self.SCORE_FLOOR, 100.0 - (100.0 - self.SCORE_FLOOR) * t)

        # ── EMA smoothing ─────────────────────────────────────────────────
        self._score = float(
            (1.0 - self.EMA_ALPHA) * self._score
            + self.EMA_ALPHA * target
        )
        return self._score

    # ── Internal helpers ──────────────────────────────────────────────────

    def _is_hidden(self, face_landmarks_list, hand_landmarks_list) -> bool:
        """
        Return True when the face appears occluded this frame.

        Logic:
          1. No face landmarks at all → hidden.
          2. Any hand landmark within PROXIMITY_THRESHOLD (normalised) of
             the nose tip → hand-over-face → hidden.
        """
        # ── 1. No face ────────────────────────────────────────────────────
        if not face_landmarks_list:
            return True

        face = face_landmarks_list[0]
        nose = face[NOSE_IDX]

        # ── 2. Hand proximity to nose ─────────────────────────────────────
        if hand_landmarks_list:
            for hand in hand_landmarks_list:
                for lm in hand:
                    dist = math.sqrt(
                        (lm.x - nose.x) ** 2
                        + (lm.y - nose.y) ** 2
                    )
                    if dist < self.PROXIMITY_THRESHOLD:
                        return True

        return False
