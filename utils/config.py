"""
utils/config.py — Centralized paths and constants for Vantage.

All path references resolve relative to the project root so that
imports work identically regardless of which module is the entry point.
"""

import os
from pathlib import Path

# Project root directory (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model file paths
FACE_MODEL_PATH = str(PROJECT_ROOT / "models" / "face_landmarker.task")
HAND_MODEL_PATH = str(PROJECT_ROOT / "models" / "hand_landmarker.task")

# Model download URLs
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Session history file
HISTORY_FILE = PROJECT_ROOT / "data" / "history.json"

# Default camera index
CAMERA_INDEX = 0

# Cloud detection
IS_CLOUD = os.path.exists("/mount/src")
