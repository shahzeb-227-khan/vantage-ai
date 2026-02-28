"""
speech_module.py — Real-Time Speech Pause / Hesitation Detection

Detects pauses and hesitation patterns using microphone RMS energy analysis.
No speech-to-text, no NLP — pure amplitude-based silence detection.

Public interface:
    analyzer = SpeechAnalyzer()
    analyzer.start()
    score = analyzer.get_score()   # float 0–100, or None when mic is muted
    muted = analyzer.is_muted      # True when mic has been silent > MIC_MUTE_DURATION
    analyzer.stop()

Scoring expectations:
    Continuous speaking        → 85–100
    Occasional natural pauses  → 70–85
    Multiple long pauses       → 40–70
    Very hesitant / silent     → <40
"""

import math
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd


class SpeechAnalyzer:
    # ------------------------------------------------------------------ #
    #  Tunable parameters                                                  #
    # ------------------------------------------------------------------ #
    SAMPLE_RATE        = 16_000   # Hz
    BLOCK_DURATION     = 0.2      # seconds per audio block
    SILENCE_THRESHOLD  = 0.010    # RMS below this = silence  (raise if too sensitive)
    PAUSE_DURATION     = 1.5      # seconds of continuous silence → counts as a pause
    WINDOW_SECONDS     = 6.0      # rolling analysis window (shorter = more responsive)
    EMA_ALPHA          = 0.35     # smoothing: 0.35 * new + 0.65 * old
    MIC_MUTE_THRESHOLD = 0.005    # RMS so low it must be a muted/absent mic
    MIC_MUTE_DURATION  = 3.0      # seconds below threshold before flagging as muted
    # ------------------------------------------------------------------ #

    def __init__(self):
        self._block_size    = int(self.SAMPLE_RATE * self.BLOCK_DURATION)
        self._block_time    = self._block_size / self.SAMPLE_RATE

        # Rolling window: deque of (timestamp, rms) pairs
        self._window: deque[tuple[float, float]] = deque()

        # Pause tracking
        self._current_silence = 0.0   # seconds of consecutive silence so far
        self._pauses_in_window: deque[float] = deque()  # timestamps of completed pauses

        self._speech_score = 100.0
        self._running      = False
        self._lock         = threading.Lock()

        # Mic mute detection
        self._mute_timer = 0.0    # seconds of continuous near-zero amplitude
        self._mic_muted  = False

        self._stream          = None
        self._analysis_thread = None

        # Latest RMS fed from audio callback
        self._latest_rms = 0.0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Open microphone stream and start background analysis thread."""
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate = self.SAMPLE_RATE,
            blocksize  = self._block_size,
            channels   = 1,
            callback   = self._audio_callback,
        )
        self._stream.start()

        self._analysis_thread = threading.Thread(
            target = self._analyze_loop, daemon = True
        )
        self._analysis_thread.start()

    def stop(self) -> None:
        """Stop the microphone stream and background thread."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_score(self) -> float | None:
        """Return Speech Decisiveness Score (0–100), or None when mic is muted."""
        with self._lock:
            if self._mic_muted:
                return None
            return round(self._speech_score, 1)

    @property
    def is_muted(self) -> bool:
        """True when the microphone appears to be muted or disconnected."""
        with self._lock:
            return self._mic_muted

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Called by sounddevice on every audio block (runs in audio thread)."""
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        with self._lock:
            self._latest_rms = rms

    def _analyze_loop(self) -> None:
        """Background thread: consume latest RMS every block and update score."""
        while self._running:
            time.sleep(self._block_time)

            with self._lock:
                rms = self._latest_rms
                now = time.time()

                # ---- Append to rolling window ----
                self._window.append((now, rms))

                # ---- Evict samples older than WINDOW_SECONDS ----
                cutoff = now - self.WINDOW_SECONDS
                while self._window and self._window[0][0] < cutoff:
                    self._window.popleft()

                # ---- Evict old pause timestamps ----
                while self._pauses_in_window and self._pauses_in_window[0] < cutoff:
                    self._pauses_in_window.popleft()

                # ---- Mic mute detection ----
                if rms < self.MIC_MUTE_THRESHOLD:
                    self._mute_timer += self._block_time
                else:
                    self._mute_timer = 0.0
                self._mic_muted = self._mute_timer >= self.MIC_MUTE_DURATION

                # ---- Silence / speech tracking (only when mic is active) ----
                if not self._mic_muted:
                    if rms < self.SILENCE_THRESHOLD:
                        self._current_silence += self._block_time
                    else:
                        if self._current_silence >= self.PAUSE_DURATION:
                            # A completed meaningful pause just ended
                            self._pauses_in_window.append(now)
                        self._current_silence = 0.0

                # ---- Compute raw score from rolling window ----
                raw = self._compute_raw_score(now)

                # ---- EMA smoothing (only update while mic is active) ----
                if not self._mic_muted:
                    self._speech_score = (
                        (1 - self.EMA_ALPHA) * self._speech_score
                        + self.EMA_ALPHA     * raw
                    )

    def _compute_raw_score(self, now: float) -> float:
        """
        Derive a 0–100 score from the current rolling window.

        Two factors:
          1. continuity_ratio — fraction of window that was speech (not silence)
          2. pause_density    — pauses per 10-second window
        """
        if not self._window:
            return self._speech_score   # hold last value if no data

        # Continuity: count blocks above threshold
        speech_blocks  = sum(1 for _, rms in self._window if rms >= self.SILENCE_THRESHOLD)
        total_blocks   = len(self._window)
        continuity     = speech_blocks / total_blocks          # 0–1

        # Pause density: pauses per window (normalised to 10 s)
        pause_count    = len(self._pauses_in_window)
        elapsed        = min(now - self._window[0][0], self.WINDOW_SECONDS)
        density        = pause_count / max(1.0, elapsed / 10.0)

        # Exponential penalty for completed pauses
        pause_score    = 100.0 * math.exp(-density * 0.8)

        # Live silence penalty — reacts immediately during an ongoing pause.
        # Ramps from 0 → full penalty over 3 seconds of continuous silence.
        live_penalty = min(1.0, self._current_silence / 3.0)
        live_score   = 100.0 * (1.0 - live_penalty)

        # Weighted combination
        raw = (
            0.50 * continuity * 100.0
            + 0.30 * pause_score
            + 0.20 * live_score
        )

        return max(0.0, min(100.0, raw))
