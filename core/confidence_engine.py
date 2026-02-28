"""
core/confidence_engine.py — Background processing engine for Vantage.

Runs all detection modules in a background thread and exposes the latest
annotated frame + metric dict through thread-safe accessors.

The engine does **not** own the camera — it receives frames from a
``CameraManager`` instance, which is the sole owner of cv2.VideoCapture.

Usage::

    from core.camera_manager import CameraManager

    cam = CameraManager.get_instance()
    cam.start()
    engine = ConfidenceEngine(cam)
    engine.start()
    result = engine.get_latest()   # -> dict | None
    engine.stop()
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from core.camera_manager import CameraManager
from vision.eye_module import GazeTracker
from vision.hand_module import GestureFirmness
from core.engagement_module import EngagementTracker
from audio.speech_module import SpeechAnalyzer
from core.fusion import fuse_scores, score_label


class ConfidenceEngine:
    """Runs the full Vantage detection pipeline in a background thread."""

    def __init__(self, camera: CameraManager) -> None:
        self._camera = camera
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._recalibrate_flag = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """Start background processing. Returns True on first result."""
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        for _ in range(30):
            time.sleep(0.1)
            if self._latest is not None:
                return True
        return self._latest is not None

    def stop(self) -> None:
        """Stop the background thread (camera is NOT released)."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._latest = None

    def get_latest(self) -> dict | None:
        """Return the most recent result dict (thread-safe)."""
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def recalibrate(self) -> None:
        """Signal the engine to reset gaze calibration on next tick."""
        with self._lock:
            self._recalibrate_flag = True

    # ------------------------------------------------------------------ #
    #  Internal loop                                                       #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Main processing loop executed in a background thread."""
        self._recalibrate_flag = False

        speech = SpeechAnalyzer()
        speech.start()

        try:
            with GazeTracker() as gaze, GestureFirmness() as gesture:
                engagement = EngagementTracker()

                face_hidden_start = None
                speech_silence_start = None
                was_speaking = False
                confidence_history: deque[tuple[float, float]] = deque()
                prev_time = time.time()

                while self._running:
                    frame = self._camera.get_frame()
                    if frame is None:
                        time.sleep(0.005)
                        continue

                    with self._lock:
                        if self._recalibrate_flag:
                            gaze.__init__()
                            self._recalibrate_flag = False

                    # Frame is already flipped by CameraManager.
                    h, w, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    timestamp_ms = int(time.time() * 1000)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb_frame
                    )

                    eye_score = gaze.process(mp_image, timestamp_ms, w, h)
                    hand_score = gesture.process(mp_image, timestamp_ms, w, h)
                    engagement_score = engagement.process(
                        gaze.last_face_landmarks,
                        (
                            gesture.last_hand_result.hand_landmarks
                            if gesture.last_hand_result
                            else None
                        ),
                    )
                    speech_score = speech.get_score()

                    if gaze.is_calibrated:
                        confidence = fuse_scores(
                            eye_score, speech_score, hand_score, engagement_score
                        )
                    else:
                        confidence = 0.0

                    curr_time = time.time()
                    fps = 1.0 / max(curr_time - prev_time, 1e-6)
                    prev_time = curr_time

                    # Temporal memory
                    confidence_history.append((curr_time, confidence))
                    while (
                        confidence_history
                        and confidence_history[0][0] < curr_time - 5.0
                    ):
                        confidence_history.popleft()
                    vals = [v for _, v in confidence_history]
                    avg5 = sum(vals) / len(vals) if vals else confidence
                    if len(vals) >= 4:
                        mid = len(vals) // 2
                        fh = sum(vals[:mid]) / mid
                        sh = sum(vals[mid:]) / (len(vals) - mid)
                        diff = sh - fh
                        trend = (
                            "\u2191" if diff > 3 else ("\u2193" if diff < -3 else "\u2192")
                        )
                    else:
                        trend = "\u2192"

                    # Behavioral rules
                    insight = ""
                    if gaze.is_calibrated:
                        if not gaze.last_face_landmarks:
                            if face_hidden_start is None:
                                face_hidden_start = curr_time
                            elif curr_time - face_hidden_start >= 4.0:
                                confidence = max(0.0, confidence * 0.6)
                                insight = "Avoidance detected"
                        else:
                            face_hidden_start = None

                        if speech_score is not None:
                            if speech_score > 50:
                                was_speaking = True
                                speech_silence_start = None
                            elif was_speaking:
                                if speech_silence_start is None:
                                    speech_silence_start = curr_time
                                elif curr_time - speech_silence_start >= 2.0:
                                    confidence = max(0.0, confidence - 10)
                                    if not insight:
                                        insight = "Hesitation detected"
                            else:
                                speech_silence_start = None

                        if (
                            eye_score > 80
                            and speech_score is not None
                            and speech_score > 75
                        ):
                            confidence = min(100.0, confidence + 5)
                            if not insight:
                                insight = "High composure"

                    gesture.draw(frame, w, h)

                    result = dict(
                        frame=frame.copy(),
                        confidence=confidence,
                        state=score_label(confidence),
                        insight=insight,
                        eye_score=eye_score,
                        speech_score=speech_score,
                        hand_score=hand_score,
                        engagement_score=engagement_score,
                        avg5=avg5,
                        trend=trend,
                        fps=fps,
                        is_calibrated=gaze.is_calibrated,
                        calibration_pct=gaze.calibration_progress,
                    )

                    with self._lock:
                        self._latest = result

        finally:
            speech.stop()
