"""
angle_calculation.py
--------------------
Pure-math module: computes joint angles and normalised distances
from leg keypoints extracted by pose_detection.py.

All calculations use the vector angle formula:
    angle = arccos( (A · B) / (|A| |B|) )

Distances are normalised by leg length (hip-to-ankle) so results
are body-size independent.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from pose_detection import LegKeypoints


# ──────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────

@dataclass
class AngleMetrics:
    """
    All computed geometric metrics for one frame.

    Angles are in degrees (0–180).
    Distances are expressed as a fraction of average leg length.
    """
    # Knee angles (angle at the knee joint formed by hip–knee–ankle)
    left_knee_angle:  float
    right_knee_angle: float

    # Hip–knee–ankle alignment angle (deviation from straight line, degrees)
    # 180° = perfectly straight leg; less → more bent / angled
    left_leg_alignment:  float
    right_leg_alignment: float

    # Q-angle proxy: lateral displacement of knee relative to hip-ankle line
    # Positive → knee displaced inward (valgus), Negative → outward (varus)
    left_q_angle:  float   # degrees
    right_q_angle: float   # degrees

    # Normalised inter-joint distances (fraction of avg leg length)
    knee_gap:   float   # horizontal distance between knees
    ankle_gap:  float   # horizontal distance between ankles

    # Raw pixel distances (for display)
    knee_gap_px:  float
    ankle_gap_px: float

    # Leg lengths (pixels) for reference
    left_leg_length:  float
    right_leg_length: float

    # Signed: negative means knees are wider than ankles (bow-legs)
    #         positive means ankles are wider than knees (knock-knees)
    knee_ankle_ratio: float   # (knee_gap - ankle_gap) / avg_leg_length


# ──────────────────────────────────────────────
# Core maths helpers
# ──────────────────────────────────────────────

def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector from point a to point b (2-D, ignores z)."""
    return (b - a)[:2]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle (degrees) between two 2-D vectors using the dot-product formula:
        angle = arccos( (v1 · v2) / (|v1| |v2|) )

    Clamps the cosine to [-1, 1] to handle floating-point edge cases.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """
    Angle at the knee joint (hip–knee–ankle).
    A perfectly straight leg → 180°; a deeply bent knee → small angle.
    """
    v_up   = _vec(knee, hip)    # knee → hip
    v_down = _vec(knee, ankle)  # knee → ankle
    return _angle_between(v_up, v_down)


def _q_angle_proxy(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    side: str,
) -> float:
    """
    Simplified Q-angle: signed angle between the thigh vector and the
    mechanical axis (hip-to-ankle).

    Convention (matches clinical view, camera-facing person):
        LEFT  leg: positive → knee displaced medially (inward) → valgus
        RIGHT leg: positive → knee displaced medially (inward) → valgus

    Uses the 2-D cross product sign to determine which side the knee sits
    relative to the hip-ankle line.
    """
    # Hip-to-ankle mechanical axis
    axis = _vec(hip, ankle)          # hip → ankle
    # Hip-to-knee (thigh direction)
    thigh = _vec(hip, knee)          # hip → knee

    angle_deg = _angle_between(axis, thigh)

    # Cross product z-component gives sign (left-hand vs right-hand turn)
    cross_z = axis[0] * thigh[1] - axis[1] * thigh[0]

    # For LEFT leg facing camera: knee inward = cross_z > 0 → valgus (+)
    # For RIGHT leg facing camera: knee inward = cross_z < 0 → valgus (+)
    if side == "left":
        signed = angle_deg if cross_z > 0 else -angle_deg
    else:  # right
        signed = angle_deg if cross_z < 0 else -angle_deg

    return float(signed)


def _leg_length(hip: np.ndarray, ankle: np.ndarray) -> float:
    """Euclidean distance hip-to-ankle (pixels)."""
    return float(np.linalg.norm((ankle - hip)[:2]))


def _horiz_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Horizontal (x-axis) distance between two joints (pixels, always ≥0)."""
    return float(abs(a[0] - b[0]))


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def compute_metrics(kp: LegKeypoints) -> AngleMetrics:
    """
    Compute all angle and distance metrics from a LegKeypoints instance.

    Parameters
    ----------
    kp : LegKeypoints
        Joint pixel coordinates extracted by PoseDetector.

    Returns
    -------
    AngleMetrics
        Fully populated metrics dataclass.
    """
    # ── Knee angles ────────────────────────────
    left_knee_angle  = _knee_angle(kp.left_hip,  kp.left_knee,  kp.left_ankle)
    right_knee_angle = _knee_angle(kp.right_hip, kp.right_knee, kp.right_ankle)

    # Leg alignment = same as knee angle (180° = perfectly straight)
    left_leg_align  = left_knee_angle
    right_leg_align = right_knee_angle

    # ── Q-angle proxies ────────────────────────
    left_q  = _q_angle_proxy(kp.left_hip,  kp.left_knee,  kp.left_ankle,  "left")
    right_q = _q_angle_proxy(kp.right_hip, kp.right_knee, kp.right_ankle, "right")

    # ── Leg lengths for normalisation ─────────
    left_len  = _leg_length(kp.left_hip,  kp.left_ankle)
    right_len = _leg_length(kp.right_hip, kp.right_ankle)
    avg_len   = (left_len + right_len) / 2.0 + 1e-6   # avoid /0

    # ── Raw inter-joint horizontal distances ──
    knee_gap_px  = _horiz_dist(kp.left_knee,  kp.right_knee)
    ankle_gap_px = _horiz_dist(kp.left_ankle, kp.right_ankle)

    # ── Normalise by leg length ────────────────
    knee_gap_norm  = knee_gap_px  / avg_len
    ankle_gap_norm = ankle_gap_px / avg_len

    # ── Knee–ankle ratio ───────────────────────
    # > 0 → ankles wider → knock-knees
    # < 0 → knees wider  → bow-legs
    knee_ankle_ratio = (ankle_gap_px - knee_gap_px) / avg_len

    return AngleMetrics(
        left_knee_angle=left_knee_angle,
        right_knee_angle=right_knee_angle,
        left_leg_alignment=left_leg_align,
        right_leg_alignment=right_leg_align,
        left_q_angle=left_q,
        right_q_angle=right_q,
        knee_gap=knee_gap_norm,
        ankle_gap=ankle_gap_norm,
        knee_gap_px=knee_gap_px,
        ankle_gap_px=ankle_gap_px,
        left_leg_length=left_len,
        right_leg_length=right_len,
        knee_ankle_ratio=knee_ankle_ratio,
    )


def format_metrics(m: AngleMetrics) -> dict:
    """Return a dict of human-readable metric strings for HUD display."""
    return {
        "Left knee angle":   f"{m.left_knee_angle:.1f}°",
        "Right knee angle":  f"{m.right_knee_angle:.1f}°",
        "Left Q-angle":      f"{m.left_q_angle:.1f}°",
        "Right Q-angle":     f"{m.right_q_angle:.1f}°",
        "Knee gap (norm)":   f"{m.knee_gap:.3f}",
        "Ankle gap (norm)":  f"{m.ankle_gap:.3f}",
        "Knee-ankle ratio":  f"{m.knee_ankle_ratio:+.3f}",
    }
