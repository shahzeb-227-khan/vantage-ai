"""
core/camera_manager.py — Singleton camera capture for Vantage.

Opens the webcam exactly once and reads frames in a dedicated daemon
thread.  All other modules receive frames as numpy arrays — they never
touch cv2.VideoCapture themselves.

Usage::

    cam = CameraManager.get_instance()
    cam.start()                  # idempotent
    frame = cam.get_frame()      # latest BGR numpy array, or None
    cam.stop()                   # release & join
"""

import threading
import time

import cv2
import numpy as np

from utils.config import CAMERA_INDEX

# Maximum retries when the camera is busy / fails to open.
_MAX_OPEN_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds between retries


class CameraManager:
    """Thread-safe singleton that owns the sole cv2.VideoCapture."""

    _instance: "CameraManager | None" = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Singleton accessor                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_instance(cls, camera_index: int = CAMERA_INDEX) -> "CameraManager":
        """Return the single CameraManager (create on first call)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(camera_index)
            return cls._instance

    # ------------------------------------------------------------------ #
    #  Init (private — use get_instance)                                   #
    # ------------------------------------------------------------------ #

    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._open_error: str = ""

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        """True while the capture thread is active."""
        return self._running

    @property
    def open_error(self) -> str:
        """Non-empty string describing the last open failure, if any."""
        return self._open_error

    def start(self) -> bool:
        """Open camera (with retries) and start capture thread.

        Returns True on success.  Idempotent — calling start() while
        already running is a no-op that returns True.
        """
        if self._running:
            return True

        self._open_error = ""
        cap = self._try_open()
        if cap is None:
            return False

        with self._lock:
            self._cap = cap
            self._frame = None

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Signal the capture thread to exit and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        with self._lock:
            cap = self._cap
            self._cap = None
            self._frame = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def get_frame(self) -> np.ndarray | None:
        """Return the most recent BGR frame, or None if unavailable."""
        with self._lock:
            return self._frame

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _try_open(self) -> cv2.VideoCapture | None:
        """Try to open the camera with retries and multiple backends."""
        for attempt in range(_MAX_OPEN_RETRIES):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                c = cv2.VideoCapture(self._camera_index, backend)
                if c.isOpened():
                    ret, _ = c.read()
                    if ret:
                        # Reduce internal buffer — we always want the newest frame.
                        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        c.set(cv2.CAP_PROP_FPS, 30)
                        return c
                c.release()

            if attempt < _MAX_OPEN_RETRIES - 1:
                time.sleep(_RETRY_DELAY)

        self._open_error = (
            "Camera is busy or unavailable after "
            f"{_MAX_OPEN_RETRIES} attempts."
        )
        return None

    def _capture_loop(self) -> None:
        """Daemon thread: continuously grab the newest frame."""
        while self._running:
            with self._lock:
                cap = self._cap
            if cap is None:
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            # Mirror for natural interaction and store.
            frame = cv2.flip(frame, 1)

            with self._lock:
                self._frame = frame
