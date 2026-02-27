"""
main.py — Vantage: Multi-modal Confidence AI
============================================
Orchestrates:
  • GazeTracker   (eye_module)   — iris-based gaze stability
  • GestureFirmness (hand_module) — wrist jerk-based hand firmness
  • fuse_scores   (fusion)       — weighted confidence score

Controls:
  ESC or close window → exit
  R key              → recalibrate gaze baseline
"""

import cv2
import mediapipe as mp
import time

from eye_module  import GazeTracker
from hand_module import GestureFirmness
from fusion      import fuse_scores, score_label, score_color

WINDOW = "Vantage — Confidence AI"

# Camera index: 0 = built-in webcam, 1 = DroidCam (typical on Windows)
CAMERA_INDEX = 0


def draw_score_bar(frame, label: str, score: float, y: int, color):
    """Draw a labelled score + progress bar at vertical position y."""
    cv2.putText(frame, f"{label}: {int(score)}%",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    bx, by, bw, bh = 20, y + 8, 280, 10
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by),
                  (bx + int(bw * score / 100), by + bh), color, -1)


def main():
    # Try different backends until one works
    cap = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        c = cv2.VideoCapture(CAMERA_INDEX, backend)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                cap = c
                print(f"Camera opened with backend {backend}")
                break
        c.release()

    if cap is None:
        print("Cannot open camera. Make sure no other app is using it.")
        return

    # Force a reasonable resolution so DroidCam doesn't return blank frames
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0.0
    fps_log_time = 0.0

    with GazeTracker() as gaze, GestureFirmness() as gesture:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            h, w, _   = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- Eye tracking ----
            timestamp_ms = int(time.time() * 1000)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            eye_score    = gaze.process(mp_image, timestamp_ms, w, h)

            # ---- Hand tracking ----
            hand_score = gesture.process(mp_image, timestamp_ms, w, h)

            # ---- Fused confidence ----
            if gaze.is_calibrated:
                confidence = fuse_scores(eye_score, hand_score)
            else:
                confidence = 0.0   # no score until baseline is locked

            # ---- FPS ----
            curr_time = time.time()
            fps       = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time

            if curr_time - fps_log_time >= 1.0:
                print(f"FPS: {int(fps)}", flush=True)
                fps_log_time = curr_time

            # ---- Draw hand skeleton ----
            gesture.draw(frame, w, h)

            # ---- HUD ----
            cv2.putText(frame, f"FPS: {int(fps)}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if not gaze.is_calibrated:
                # Show calibration progress
                pct = gaze.calibration_progress
                cv2.putText(frame, "Calibrating — look at screen",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 200, 255), 2)
                cv2.rectangle(frame, (20, 85), (300, 95), (60, 60, 60), -1)
                cv2.rectangle(frame, (20, 85),
                              (20 + int(280 * pct), 95), (0, 200, 255), -1)
            else:
                # Individual modality scores
                eye_color  = score_color(eye_score)
                hand_color = score_color(hand_score)
                conf_color = score_color(confidence)

                draw_score_bar(frame, "Gaze Stability",   eye_score,  70,  eye_color)
                draw_score_bar(frame, "Gesture Firmness", hand_score, 105, hand_color)

                # Fused confidence (larger, bottom)
                label = score_label(confidence)
                cv2.putText(frame, f"Confidence: {int(confidence)}% [{label}]",
                            (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.85, conf_color, 2)
                bx, by, bw, bh = 20, 165, 280, 14
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
                cv2.rectangle(frame, (bx, by),
                              (bx + int(bw * confidence / 100), by + bh), conf_color, -1)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break
            if key == ord('r') or key == ord('R'):
                # Hard reset gaze baseline — useful to recalibrate mid-session
                gaze.__init__()
                print("Gaze recalibrated.")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
