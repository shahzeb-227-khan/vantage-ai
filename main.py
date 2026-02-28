"""
main.py — Vantage: Multi-modal Confidence AI
============================================
Orchestrates:
  • GazeTracker       (eye_module)        — iris-based gaze stability      [0.35]
  • SpeechAnalyzer    (speech_module)     — pause / hesitation detection   [0.30]
  • GestureFirmness   (hand_module)       — wrist jerk + tremble detection [0.20]
  • EngagementTracker (engagement_module) — face-hiding detection          [0.15]
  • fuse_scores       (fusion)            — weighted 4-signal confidence score

Controls:
  ESC or close window → exit
  R key               → recalibrate gaze baseline
"""

import cv2
import math
import mediapipe as mp
import numpy as np
import time
from collections import deque

from eye_module        import GazeTracker
from hand_module       import GestureFirmness
from engagement_module import EngagementTracker
from speech_module     import SpeechAnalyzer
from fusion            import fuse_scores, score_label, score_color

WINDOW = "Vantage — Confidence AI"

# Camera index: 0 = built-in webcam, 1 = DroidCam (typical on Windows)
CAMERA_INDEX = 0


PANEL_W = 340   # width of right-side info panel


def draw_mini_bar(panel, label: str, score: float, y: int, color, muted: bool = False):
    """Draw a compact labelled bar inside the right panel."""
    cv2.putText(panel, label, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    bx, by, bw, bh = 16, y + 5, PANEL_W - 32, 8
    cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (55, 55, 65), -1)
    if not muted:
        cv2.rectangle(panel, (bx, by), (bx + int(bw * score / 100), by + bh), color, -1)
    val_txt = "MUTED" if muted else f"{int(score)}%"
    cv2.putText(panel, val_txt, (bx + bw + 4, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color if not muted else (0, 200, 255), 1)


def draw_circular_meter(panel, cx: int, cy: int, radius: int,
                        score: float, color, label: str):
    """
    Draw a speedometer-style arc meter.
    Arc spans 240 degrees (150° → 390° clockwise in cv2 convention).
    """
    # Background track
    cv2.ellipse(panel, (cx, cy), (radius, radius), 0, 150, 390,
                (55, 55, 65), 14, cv2.LINE_AA)
    # Foreground arc
    end_angle = 150 + 240 * score / 100
    if score > 0:
        cv2.ellipse(panel, (cx, cy), (radius, radius), 0, 150, end_angle,
                    color, 14, cv2.LINE_AA)
    # Score text
    score_txt = str(int(score))
    (tw, th), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(panel, score_txt,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    # State label below score
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(panel, label,
                (cx - lw // 2, cy + th // 2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


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

    speech = SpeechAnalyzer()
    speech.start()

    try:
        with GazeTracker() as gaze, GestureFirmness() as gesture:
            engagement = EngagementTracker()

            # Behavioral context state
            face_hidden_start    = None
            speech_silence_start = None
            was_speaking         = False
            behavioral_flag      = ""

            # Temporal memory — rolling 5-second confidence history
            confidence_history: deque[tuple[float, float]] = deque()  # (timestamp, value)

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

                # ---- Engagement (face visibility) ----
                engagement_score = engagement.process(
                    gaze.last_face_landmarks,
                    gesture.last_hand_result.hand_landmarks
                        if gesture.last_hand_result else None,
                )

                # ---- Speech decisiveness ----
                speech_score = speech.get_score()

                # ---- Fused confidence ----
                if gaze.is_calibrated:
                    confidence = fuse_scores(eye_score, speech_score, hand_score, engagement_score)
                else:
                    confidence = 0.0

                # ---- FPS ----
                curr_time = time.time()
                fps       = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
                prev_time = curr_time

                if curr_time - fps_log_time >= 1.0:
                    print(f"FPS: {int(fps)}", flush=True)
                    fps_log_time = curr_time

                # ---- Temporal memory ----
                confidence_history.append((curr_time, confidence))
                # Evict entries older than 5 seconds
                while confidence_history and confidence_history[0][0] < curr_time - 5.0:
                    confidence_history.popleft()
                # Compute 5s average and trend arrow
                vals = [v for _, v in confidence_history]
                avg5 = sum(vals) / len(vals) if vals else confidence
                if len(vals) >= 4:
                    mid = len(vals) // 2
                    first_half = sum(vals[:mid]) / mid
                    second_half = sum(vals[mid:]) / (len(vals) - mid)
                    diff = second_half - first_half
                    trend = "\u2191" if diff > 3 else ("\u2193" if diff < -3 else "\u2192")
                else:
                    trend = "\u2192"

                # ---- Behavioral context rules ----
                behavioral_flag = ""
                if gaze.is_calibrated:

                    # Rule 1 — Face hidden > 4 seconds → Avoidance
                    if not gaze.last_face_landmarks:
                        if face_hidden_start is None:
                            face_hidden_start = curr_time
                        elif curr_time - face_hidden_start >= 4.0:
                            confidence = max(0.0, confidence * 0.6)
                            behavioral_flag = "Avoidance detected"
                    else:
                        face_hidden_start = None

                    # Rule 2 — Was speaking, then silence > 2s → Hesitation
                    if speech_score is not None:
                        if speech_score > 50:
                            was_speaking = True
                            speech_silence_start = None
                        elif was_speaking:
                            if speech_silence_start is None:
                                speech_silence_start = curr_time
                            elif curr_time - speech_silence_start >= 2.0:
                                confidence = max(0.0, confidence - 10)
                                if not behavioral_flag:
                                    behavioral_flag = "Hesitation detected"
                        else:
                            speech_silence_start = None

                    # Rule 3 — Strong gaze + smooth speech → High composure bonus
                    if (eye_score > 80
                            and speech_score is not None
                            and speech_score > 75):
                        confidence = min(100.0, confidence + 5)
                        if not behavioral_flag:
                            behavioral_flag = "High composure"

                # ---- Draw hand skeleton ----
                gesture.draw(frame, w, h)

                # ---- Build composite canvas (camera | right panel) ----
                canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
                canvas[:, :w] = frame

                panel = canvas[:, w:]
                panel[:] = (28, 28, 38)

                # FPS (camera side, unobtrusive)
                cv2.putText(canvas, f"FPS: {int(fps)}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

                # Panel title
                cv2.putText(panel, "VANTAGE",
                            (PANEL_W // 2 - 52, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
                cv2.line(panel, (16, 40), (PANEL_W - 16, 40), (60, 60, 75), 1)

                if not gaze.is_calibrated:
                    pct = gaze.calibration_progress
                    cv2.putText(panel, "Calibrating...",
                                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
                    cv2.rectangle(panel, (16, 90), (PANEL_W - 16, 102), (55, 55, 65), -1)
                    cv2.rectangle(panel, (16, 90),
                                  (16 + int((PANEL_W - 32) * pct), 102), (0, 200, 255), -1)
                else:
                    conf_color   = score_color(confidence)
                    eye_color    = score_color(eye_score)
                    hand_color   = score_color(hand_score)
                    speech_color = score_color(speech_score) if speech_score is not None else (0, 200, 255)
                    label        = score_label(confidence)

                    # Circular confidence meter
                    draw_circular_meter(panel, PANEL_W // 2, 148, 82,
                                        confidence, conf_color, label)

                    # 5-sec average + trend arrow
                    avg_txt = f"5s avg: {int(avg5)}%  {trend}"
                    (aw, _), _ = cv2.getTextSize(avg_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.putText(panel, avg_txt,
                                (PANEL_W // 2 - aw // 2, 248),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

                    cv2.line(panel, (16, 260), (PANEL_W - 16, 260), (60, 60, 75), 1)

                    # Mini signal bars
                    draw_mini_bar(panel, "Eye Stability",  eye_score,        280, eye_color)
                    draw_mini_bar(panel, "Speech",         speech_score if speech_score is not None else 0,
                                  308, speech_color, muted=(speech_score is None))
                    draw_mini_bar(panel, "Hand Stability", hand_score,       336, hand_color)
                    draw_mini_bar(panel, "Engagement",     engagement_score, 364,
                                  score_color(engagement_score))

                    cv2.line(panel, (16, 382), (PANEL_W - 16, 382), (60, 60, 75), 1)

                    # Behavioral insight
                    if behavioral_flag:
                        if behavioral_flag == "High composure":
                            flag_color = (0, 220, 0)
                        elif behavioral_flag == "Hesitation detected":
                            flag_color = (0, 200, 255)
                        else:
                            flag_color = (0, 0, 255)
                        cv2.putText(panel, "Insight:",
                                    (16, 402), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.46, (150, 150, 150), 1, cv2.LINE_AA)
                        cv2.putText(panel, behavioral_flag,
                                    (16, 424), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.56, flag_color, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(panel, "Insight: Nominal",
                                    (16, 402), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.46, (80, 80, 80), 1, cv2.LINE_AA)

                    # Footer
                    cv2.putText(panel, "R = recalibrate  |  ESC = quit",
                                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.36, (70, 70, 70), 1)

                cv2.imshow(WINDOW, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:   # ESC
                    break
                if key == ord('r') or key == ord('R'):
                    gaze.__init__()
                    print("Gaze recalibrated.")
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        speech.stop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
