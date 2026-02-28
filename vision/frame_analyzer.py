"""
vision/frame_analyzer.py — Single-frame analysis for Streamlit Cloud.

Processes individual camera snapshots (from st.camera_input) through the
Vantage detection pipeline without requiring continuous video capture or
microphone access.

Cloud constraints handled:
    - No cv2.VideoCapture — frames arrive from st.camera_input.
    - No microphone — speech_score is always None.
    - Fast calibration — 5 frames instead of 60.
"""

import time

import mediapipe as mp
import numpy as np

from vision.eye_module import GazeTracker
from vision.hand_module import GestureFirmness
from core.engagement_module import EngagementTracker
from core.fusion import fuse_scores, score_label


class FrameAnalyzer:
    """Stateful single-frame processor for cloud deployment."""

    def __init__(self, calibration_frames: int = 5) -> None:
        self._gaze = GazeTracker(calibration_frames=calibration_frames)
        self._gesture = GestureFirmness()
        self._engagement = EngagementTracker()
        self._frame_count = 0
        self._ts_base = int(time.time() * 1000)
        self._confidences: list[float] = []

    def process(self, rgb_frame: np.ndarray) -> dict:
        """Analyse one RGB frame and return a metrics dictionary.

        Args:
            rgb_frame: shape (H, W, 3), dtype uint8, RGB order.

        Returns:
            Dict with keys matching ConfidenceEngine.get_latest() so that
            SessionManager.record() and the UI helpers work unchanged.
        """
        self._frame_count += 1
        h, w = rgb_frame.shape[:2]

        timestamp_ms = self._ts_base + self._frame_count * 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        eye_score = self._gaze.process(mp_image, timestamp_ms, w, h)
        hand_score = self._gesture.process(mp_image, timestamp_ms, w, h)
        engagement_score = self._engagement.process(
            self._gaze.last_face_landmarks,
            (
                self._gesture.last_hand_result.hand_landmarks
                if self._gesture.last_hand_result
                else None
            ),
        )

        speech_score = None

        if self._gaze.is_calibrated:
            confidence = fuse_scores(
                eye_score, speech_score, hand_score, engagement_score
            )
        else:
            confidence = 0.0

        self._confidences.append(confidence)
        recent = self._confidences[-15:]
        avg = sum(recent) / len(recent) if recent else confidence

        return dict(
            confidence=confidence,
            state=score_label(confidence),
            insight="",
            eye_score=eye_score,
            speech_score=speech_score,
            hand_score=hand_score,
            engagement_score=engagement_score,
            avg5=avg,
            trend="\u2192",
            fps=0,
            is_calibrated=self._gaze.is_calibrated,
            calibration_pct=self._gaze.calibration_progress,
        )

    def close(self) -> None:
        """Release detector resources."""
        self._gaze.close()
        self._gesture.close()
