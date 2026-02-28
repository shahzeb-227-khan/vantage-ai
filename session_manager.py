"""
session_manager.py — Local JSON session storage for Vantage

Usage:
    mgr = SessionManager()
    mgr.start_session()
    mgr.record(metrics_dict)   # called every frame
    summary = mgr.end_session()
    history = mgr.load_history()
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path


HISTORY_FILE = Path(__file__).parent / "history.json"


class SessionManager:
    def __init__(self):
        self._start_time: float | None = None
        self._confidences: list[float] = []
        self._active = False

    # ------------------------------------------------------------------ #
    #  Session lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def start_session(self) -> None:
        self._start_time = time.time()
        self._confidences = []
        self._active = True

    def record(self, metrics: dict) -> None:
        """Call once per frame while session is active."""
        if not self._active:
            return
        c = metrics.get("confidence", 0.0)
        if metrics.get("is_calibrated", False):
            self._confidences.append(float(c))

    def end_session(self) -> dict | None:
        """Stop recording, compute summary, persist to history.json, return summary."""
        if not self._active or self._start_time is None:
            return None
        self._active = False

        duration = time.time() - self._start_time
        vals = self._confidences

        if not vals:
            return None

        avg_conf = sum(vals) / len(vals)
        max_conf = max(vals)
        min_conf = min(vals)
        pct_high = 100.0 * sum(1 for v in vals if v >= 75) / len(vals)
        pct_low  = 100.0 * sum(1 for v in vals if v < 50)  / len(vals)

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(duration, 1),
            "avg_confidence": round(avg_conf, 1),
            "max_confidence": round(max_conf, 1),
            "min_confidence": round(min_conf, 1),
            "pct_above_75": round(pct_high, 1),
            "pct_below_50": round(pct_low, 1),
            "frame_count": len(vals),
        }

        self._persist(summary)
        return summary

    # ------------------------------------------------------------------ #
    #  History access                                                      #
    # ------------------------------------------------------------------ #

    def load_history(self) -> list[dict]:
        """Return list of past session summaries, newest first."""
        if not HISTORY_FILE.exists():
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(reversed(data)) if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _persist(self, summary: dict) -> None:
        existing: list[dict] = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(summary)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)


# ------------------------------------------------------------------ #
#  Rule-based improvement tips                                        #
# ------------------------------------------------------------------ #

def generate_tips(summary: dict) -> list[str]:
    """Return a list of actionable tips based on summary stats."""
    tips = []
    avg  = summary.get("avg_confidence", 100)
    low  = summary.get("pct_below_50", 0)
    high = summary.get("pct_above_75", 0)

    if avg < 60:
        tips.append("🗣 Your overall confidence was low. Practice speaking at a steady pace with fewer pauses.")
    if avg >= 80:
        tips.append("✅ Excellent overall confidence. Keep maintaining this level in real interviews.")

    if low > 30:
        tips.append("⚠️ You spent significant time below 50% confidence. Work on reducing long silences and hesitation.")

    if high < 40:
        tips.append("📈 Less than 40% of your session was high-confidence. Focus on projecting certainty in your answers.")
    elif high > 70:
        tips.append("🌟 Great job — over 70% of your session was in the high-confidence zone.")

    # Fallback if no specific tip triggered
    if not tips:
        tips.append("👍 Solid session. Continue practising to maintain consistency under pressure.")

    return tips
