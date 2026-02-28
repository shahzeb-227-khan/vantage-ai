"""
core/fusion.py — Multi-modal score fusion.

Combines all behavioral modality scores into a single Confidence Score.
When speech is muted/unavailable (speech_score=None) the weights adapt
automatically so the other signals are not diluted.

Normal (4-signal) weights:
    eye_score    — gaze stability        (0.35)
    speech_score — speech continuity     (0.30)
    hand_score   — hand stability        (0.20)
    engagement   — face engagement       (0.15)

Adaptive (mic muted, 3-signal) weights:
    eye_score    — gaze stability        (0.50)
    hand_score   — hand stability        (0.30)
    engagement   — face engagement       (0.20)
"""


def fuse_scores(
    eye_score: float,
    speech_score: float | None,
    hand_score: float,
    engagement: float,
) -> float:
    """Return weighted fusion of modality scores (0-100).

    Args:
        eye_score:    0-100 from GazeTracker.
        speech_score: 0-100 from SpeechAnalyzer, or None when mic is muted.
        hand_score:   0-100 from GestureFirmness.
        engagement:   0-100 from EngagementTracker.

    Returns:
        Fused confidence score 0-100.
    """
    if speech_score is None:
        return 0.50 * eye_score + 0.30 * hand_score + 0.20 * engagement
    return (
        0.35 * eye_score
        + 0.30 * speech_score
        + 0.20 * hand_score
        + 0.15 * engagement
    )


def score_label(score: float) -> str:
    """Return a five-tier confidence state label for display."""
    if score >= 85:
        return "Commanding"
    if score >= 70:
        return "Confident"
    if score >= 55:
        return "Moderate"
    if score >= 40:
        return "Hesitant"
    return "Low Confidence"


def score_color(score: float) -> tuple[int, int, int]:
    """Return a BGR colour matched to the five confidence tiers."""
    if score >= 85:
        return (0, 255, 120)
    if score >= 70:
        return (0, 210, 0)
    if score >= 55:
        return (0, 200, 255)
    if score >= 40:
        return (0, 120, 255)
    return (0, 0, 255)
