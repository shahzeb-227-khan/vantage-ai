"""
confidence_engine.py — Black-box engine for Vantage

Runs all detection modules in a background thread and exposes the latest
annotated frame + metric dict through thread-safe accessors.

Usage:
    engine = ConfidenceEngine()
    engine.start()
    ...
    result = engine.get_latest()   # -> dict | None
    engine.stop()

`get_latest()` returns a dict with keys:
    frame           : np.ndarray (BGR, annotated)
    confidence      : float   0-100
    state           : str     e.g. "Confident"
    insight         : str     behavioral flag text
    eye_score       : float
    speech_score    : float | None  (None = mic muted)
    hand_score      : float
    engagement_score: float
    avg5            : float   5-second rolling average
    trend           : str     "↑" / "↓" / "→"
    fps             : float
    is_calibrated   : bool
    calibration_pct : float  0-1
"""

import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from eye_module        import GazeTracker
from hand_module       import GestureFirmness
from engagement_module import EngagementTracker
from speech_module     import SpeechAnalyzer
from fusion            import fuse_scores, score_label

CAMERA_INDEX = 0


class ConfidenceEngine:
    """Runs the full Vantage detection pipeline in a background thread."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self._camera_index = camera_index
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: dict | None = None
        # _recalibrate_flag must exist before start() so recalibrate() never
        # raises AttributeError if called before the thread initialises it.
        self._recalibrate_flag = False
        # _cap is stored here so stop() can force-release the camera as a
        # safety net when join() times out (e.g. frozen cap.read on Windows).
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """Open camera and start background processing. Returns True on success."""
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait up to 3 s for the first frame
        for _ in range(30):
            time.sleep(0.1)
            if self._latest is not None:
                return True
        return self._latest is not None

    def stop(self) -> None:
        """Stop the background thread and guarantee camera release."""
        self._running = False
        if self._thread:
            # Give the run-loop up to 3 s to exit cleanly via its finally block.
            self._thread.join(timeout=3.0)
            self._thread = None
        # Safety net: if join() timed out the thread may still hold the device.
        # Force-release so the next engine.start() can open the camera.
        with self._lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self._latest = None

    def get_latest(self) -> dict | None:
        """Return the most recent result dict (thread-safe). None if not ready."""
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
        self._recalibrate_flag = False

        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            c = cv2.VideoCapture(self._camera_index, backend)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    cap = c
                    break
            c.release()

        if cap is None:
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Publish cap reference so stop() can force-release if join() times out.
        with self._lock:
            self._cap = cap

        speech = SpeechAnalyzer()
        speech.start()

        try:
            with GazeTracker() as gaze, GestureFirmness() as gesture:
                engagement = EngagementTracker()

                face_hidden_start    = None
                speech_silence_start = None
                was_speaking         = False
                confidence_history: deque[tuple[float, float]] = deque()

                prev_time = time.time()

                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    # Check recalibrate flag
                    with self._lock:
                        if self._recalibrate_flag:
                            gaze.__init__()
                            self._recalibrate_flag = False

                    frame     = cv2.flip(frame, 1)
                    h, w, _   = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    timestamp_ms = int(time.time() * 1000)
                    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    eye_score        = gaze.process(mp_image, timestamp_ms, w, h)
                    hand_score       = gesture.process(mp_image, timestamp_ms, w, h)
                    engagement_score = engagement.process(
                        gaze.last_face_landmarks,
                        gesture.last_hand_result.hand_landmarks
                            if gesture.last_hand_result else None,
                    )
                    speech_score = speech.get_score()

                    # Fused confidence
                    if gaze.is_calibrated:
                        confidence = fuse_scores(eye_score, speech_score,
                                                 hand_score, engagement_score)
                    else:
                        confidence = 0.0

                    curr_time = time.time()
                    fps = 1.0 / max(curr_time - prev_time, 1e-6)
                    prev_time = curr_time

                    # Temporal memory
                    confidence_history.append((curr_time, confidence))
                    while confidence_history and confidence_history[0][0] < curr_time - 5.0:
                        confidence_history.popleft()
                    vals = [v for _, v in confidence_history]
                    avg5 = sum(vals) / len(vals) if vals else confidence
                    if len(vals) >= 4:
                        mid  = len(vals) // 2
                        fh   = sum(vals[:mid]) / mid
                        sh   = sum(vals[mid:]) / (len(vals) - mid)
                        diff = sh - fh
                        trend = "↑" if diff > 3 else ("↓" if diff < -3 else "→")
                    else:
                        trend = "→"

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

                        if (eye_score > 80
                                and speech_score is not None
                                and speech_score > 75):
                            confidence = min(100.0, confidence + 5)
                            if not insight:
                                insight = "High composure"

                    # Draw skeleton on frame
                    gesture.draw(frame, w, h)

                    result = dict(
                        frame            = frame.copy(),
                        confidence       = confidence,
                        state            = score_label(confidence),
                        insight          = insight,
                        eye_score        = eye_score,
                        speech_score     = speech_score,
                        hand_score       = hand_score,
                        engagement_score = engagement_score,
                        avg5             = avg5,
                        trend            = trend,
                        fps              = fps,
                        is_calibrated    = gaze.is_calibrated,
                        calibration_pct  = gaze.calibration_progress,
                    )

                    with self._lock:
                        self._latest = result

        finally:
            speech.stop()
            cap.release()
            # Clear the shared reference — stop()'s safety net no longer needed.
            with self._lock:
                self._cap = None
