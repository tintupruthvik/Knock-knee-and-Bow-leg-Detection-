"""
classification.py
-----------------
Rule-based classification system:

Uses joint angles, Q-angles, and normalised distances derived
from the vector-maths in angle_calculation.py.

Classification labels:
    "Normal"        – alignment within healthy range
    "Knock Knees"   – genu valgum (knees close / touching, ankles apart)
    "Bow Legs"      – genu varum  (knees apart, ankles close / touching)
"""

from dataclasses import dataclass
from typing import Optional, List

from angle_calculation import AngleMetrics


# ──────────────────────────────────────────────
# Constants / thresholds
# ──────────────────────────────────────────────

LABEL_NORMAL      = "Normal"
LABEL_KNOCK_KNEES = "Knock Knees"
LABEL_BOW_LEGS    = "Bow Legs"
LABELS = [LABEL_NORMAL, LABEL_KNOCK_KNEES, LABEL_BOW_LEGS]

MIN_KNEE_ANGLE_FOR_ANALYSIS = 145.0


# ──────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────

@dataclass
class ClassificationResult:
    label: str                       # "Normal" | "Knock Knees" | "Bow Legs"
    confidence: float                # 0.0–1.0
    method: str                      # always "rule-based"
    proba: Optional[List[float]]     # always None
    warnings: List[str]              # non-fatal diagnostic messages


# ──────────────────────────────────────────────
# Rule-based classifier
# ──────────────────────────────────────────────

class RuleBasedClassifier:
    """
    Deterministic classifier that combines three independent signals:

      1. Knee-ankle ratio  – most direct measure of gap disparity
      2. Average Q-angle   – indicates medial/lateral knee displacement
      3. Raw gap comparison – ankle vs knee horizontal distance

    Voting: each signal casts a vote; majority wins with confidence
    proportional to agreement level.
    """

    def classify(self, m: AngleMetrics) -> ClassificationResult:
        warnings: List[str] = []

        # ── Quality gate ───────────────────────
        avg_knee = (m.left_knee_angle + m.right_knee_angle) / 2.0
        if avg_knee < MIN_KNEE_ANGLE_FOR_ANALYSIS:
            warnings.append(
                f"Knees appear bent (avg {avg_knee:.0f}°); result less reliable"
            )

        if abs(m.left_knee_angle - m.right_knee_angle) > 20:
            warnings.append("Asymmetric knee angles; check posture")

        score_knock = 0
        score_bow   = 0

        # 1. Knee-ankle ratio (strongest signal)
        if m.knee_ankle_ratio > 0.03:
            score_knock += 3
        elif m.knee_ankle_ratio < -0.03:
            score_bow += 3

        # 2. Q-angle (medium signal)
        avg_q = (m.left_q_angle + m.right_q_angle) / 2.0
        if avg_q > 2.0:
            score_knock += 1.5
        elif avg_q < -2.0:
            score_bow += 1.5

        # 3. Raw gap: ankle wider than knee = knock knees
        if m.ankle_gap > m.knee_gap + 0.03:
            score_knock += 2
        elif m.knee_gap > m.ankle_gap + 0.03:
            score_bow += 2

        # ── Final decision ─────────────────────
        if score_knock > score_bow and score_knock >= 2:
            label = LABEL_KNOCK_KNEES
        elif score_bow > score_knock and score_bow >= 2:
            label = LABEL_BOW_LEGS
        else:
            label = LABEL_NORMAL

        # ── Confidence calculation ─────────────
        confidence = max(score_knock, score_bow) / (score_knock + score_bow + 1e-6)
        confidence = min(confidence, 1.0)

        # Symmetry penalty
        knee_diff = abs(m.left_knee_angle - m.right_knee_angle)
        if knee_diff > 15:
            confidence *= 0.7

        return ClassificationResult(
            label=label,
            confidence=float(confidence),
            method="rule-based",
            proba=None,
            warnings=warnings,
        )


# ──────────────────────────────────────────────
# Temporal smoother (for video stability)
# ──────────────────────────────────────────────

class TemporalSmoother:
    """
    Reduces flickering in video by requiring a label to appear in
    the majority of a rolling window before committing to it.
    """

    def __init__(self, window: int = 10):
        self.window  = window
        self._buffer: List[str] = []

    def update(self, result: ClassificationResult) -> ClassificationResult:
        """Feed a new result and return the temporally-smoothed result."""
        self._buffer.append(result.label)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        counts = {L: self._buffer.count(L) for L in LABELS}
        smoothed_label = max(counts, key=counts.__getitem__)
        smoothed_conf  = counts[smoothed_label] / len(self._buffer)

        return ClassificationResult(
            label=smoothed_label,
            confidence=smoothed_conf,
            method=result.method + "+smoothed",
            proba=None,
            warnings=result.warnings,
        )

    def reset(self):
        self._buffer.clear()
