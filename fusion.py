"""
fusion.py — Multi-modal score fusion (Phase 4)

Combines individual modality scores into a single "Confidence Score".
Weights can be tuned per use-case; defaults favour gaze slightly.
"""


def fuse_scores(
    eye_score:  float,
    hand_score: float,
    eye_weight:  float = 0.6,
    hand_weight: float = 0.4,
) -> float:
    """
    Weighted average of gaze stability and gesture firmness.

    Args:
        eye_score:   0–100, from GazeTracker
        hand_score:  0–100, from GestureFirmness
        eye_weight:  relative importance of gaze (default 0.6)
        hand_weight: relative importance of gesture (default 0.4)

    Returns:
        Fused confidence score 0–100.
    """
    total = eye_weight + hand_weight
    return (eye_weight * eye_score + hand_weight * hand_score) / total


def score_label(score: float) -> str:
    """Human-readable confidence tier for display."""
    if score >= 80:
        return "High"
    elif score >= 55:
        return "Moderate"
    else:
        return "Low"


def score_color(score: float) -> tuple[int, int, int]:
    """BGR colour used to render the score on-screen."""
    if score >= 70:
        return (0, 255, 0)      # green
    elif score >= 40:
        return (0, 200, 255)    # yellow-orange
    else:
        return (0, 0, 255)      # red
