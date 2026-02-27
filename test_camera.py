import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import requests
import os
import numpy as np
from collections import deque

# ---------- Model download (runs once) ----------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~1.8 MB)...")
    response = requests.get(MODEL_URL, stream=True, timeout=30)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model ready.")

# ---------- Detector setup ----------
options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Iris center landmarks — actual eyeball positions, not eyelid contour averages.
LEFT_IRIS  = 468
RIGHT_IRIS = 473

# Nose tip used as head anchor to make iris positions head-relative.
NOSE_IDX = 1

# ---------- Gaze baseline calibration ----------
# Collect iris offsets for the first N frames while the user looks at the
# screen, then fix that as the "neutral" gaze zone.  Any deviation from
# this zone — not jitter — is what lowers the confidence score.
CALIBRATION_FRAMES = 60      # ~2 s at 30 fps
GAZE_TOLERANCE     = 10.0    # px: deviation inside this radius → score 100
GAZE_MAX_DIST      = 40.0    # px: deviation at or beyond this → score 0
EMA_ALPHA          = 0.35    # smoothing for display (higher = more responsive)

calib_buffer_left  = []      # accumulates relative offsets during calibration
calib_buffer_right = []
baseline_left      = None    # fixed after calibration
baseline_right     = None
smoothed_stability = 100.0   # display value, EMA-smoothed

# ---------- Camera ----------
# DroidCam registers as index 1 on Windows (built-in webcam is 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

prev_time = 0
fps_log_time = 0

try:
    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert to RGB and wrap for Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # detect_for_video needs a strictly-increasing ms timestamp
            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            h, w, _ = frame.shape

            if result.face_landmarks:
                for landmarks in result.face_landmarks:

                    # --- Iris centers (pixel coords) ---
                    left_center  = np.array([landmarks[LEFT_IRIS].x  * w,
                                             landmarks[LEFT_IRIS].y  * h])
                    right_center = np.array([landmarks[RIGHT_IRIS].x * w,
                                             landmarks[RIGHT_IRIS].y * h])

                    # --- Head-relative iris offsets (removes head translation) ---
                    face_ref = np.array([landmarks[NOSE_IDX].x * w,
                                         landmarks[NOSE_IDX].y * h])
                    left_rel  = left_center  - face_ref
                    right_rel = right_center - face_ref

                    # --- Phase 1: Calibration ---
                    # Accumulate neutral-gaze samples while the user looks at
                    # the screen.  Once we have enough, lock in the baseline.
                    if baseline_left is None:
                        calib_buffer_left.append(left_rel)
                        calib_buffer_right.append(right_rel)

                        if len(calib_buffer_left) >= CALIBRATION_FRAMES:
                            baseline_left  = np.mean(calib_buffer_left,  axis=0)
                            baseline_right = np.mean(calib_buffer_right, axis=0)

                    # --- Phase 2: Deviation scoring ---
                    # Measure how far each iris has drifted from its neutral
                    # position.  Small drift (within tolerance) → score 100.
                    # Drift beyond GAZE_MAX_DIST → score 0. Linear in between.
                    else:
                        dist_left  = np.linalg.norm(left_rel  - baseline_left)
                        dist_right = np.linalg.norm(right_rel - baseline_right)
                        avg_dist   = (dist_left + dist_right) / 2.0

                        # Flat zone: movements within tolerance are ignored
                        if avg_dist <= GAZE_TOLERANCE:
                            raw_score = 100.0
                        else:
                            # Linear ramp from tolerance edge (100) to max dist (0)
                            raw_score = max(
                                0.0,
                                100.0 * (1.0 - (avg_dist - GAZE_TOLERANCE)
                                               / (GAZE_MAX_DIST - GAZE_TOLERANCE))
                            )

                        # EMA smoothing for display
                        smoothed_stability = (EMA_ALPHA * raw_score
                                              + (1.0 - EMA_ALPHA) * smoothed_stability)

            else:
                # No face detected — let history decay toward 0 gradually
                smoothed_stability = (1.0 - EMA_ALPHA) * smoothed_stability

            # FPS counter
            curr_time = time.time()
            delta = curr_time - prev_time
            fps = 1 / delta if delta > 0 else 0
            prev_time = curr_time

            # Print FPS to terminal once per second
            if curr_time - fps_log_time >= 1.0:
                print(f"FPS: {int(fps)}", flush=True)
                fps_log_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # --- Calibration overlay ---
            # Show a progress bar and message until baseline is locked.
            if baseline_left is None:
                n = len(calib_buffer_left)
                pct = n / CALIBRATION_FRAMES
                cv2.putText(frame, f"Calibrating... look at screen",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2)
                bar_x, bar_y, bar_w, bar_h = 20, 95, 300, 10
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_w * pct), bar_y + bar_h),
                              (0, 200, 255), -1)
            else:
                # --- Draw stability score ---
                score = int(smoothed_stability)

                # Colour gradient: green (high) → yellow-orange (mid) → red (low)
                if score >= 70:
                    s_color = (0, 255, 0)       # green
                elif score >= 40:
                    s_color = (0, 200, 255)     # yellow-orange
                else:
                    s_color = (0, 0, 255)       # red

                cv2.putText(frame, f"Eye Stability: {score}%", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, s_color, 2)

                # Thin progress bar under the text
                bar_x, bar_y, bar_w, bar_h = 20, 95, 300, 12
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                              (60, 60, 60), -1)                              # background
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_w * score / 100), bar_y + bar_h),
                              s_color, -1)                                   # filled portion

            cv2.imshow("Vantage - Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF
            # ESC key OR window X button both exit cleanly
            if key == 27 or cv2.getWindowProperty("Vantage - Camera Test", cv2.WND_PROP_VISIBLE) < 1:
                break

except KeyboardInterrupt:
    pass  # Ctrl+C exits cleanly without a traceback
finally:
    cap.release()
    cv2.destroyAllWindows()