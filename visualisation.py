"""
visualisation.py
----------------
All OpenCV drawing helpers:
  - Joint-to-joint lines (leg skeleton)
  - Angle arc + label at knee
  - HUD panel (classification result, confidence, metrics) — TOP RIGHT
  - Status / rejection banner
"""

import cv2
import numpy as np
import math
from typing import Optional

from pose_detection import LegKeypoints, PoseResult
from angle_calculation import AngleMetrics, format_metrics
from classification import ClassificationResult, LABEL_NORMAL, LABEL_KNOCK_KNEES, LABEL_BOW_LEGS


# ──────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────

C_GREEN   = (50,  220,  80)
C_RED     = (40,   60, 230)
C_ORANGE  = (30,  155, 240)
C_CYAN    = (220, 210,  40)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,    0)
C_YELLOW  = (0,   220, 240)
C_BLUE    = (230, 120,  30)

RESULT_COLORS = {
    LABEL_NORMAL:      C_GREEN,
    LABEL_KNOCK_KNEES: C_ORANGE,
    LABEL_BOW_LEGS:    C_RED,
    "Unknown":         C_CYAN,
}

# Output display size — image is upscaled to this for a bigger window
DISPLAY_WIDTH  = 1000
DISPLAY_HEIGHT = 750


# ──────────────────────────────────────────────
# Utility drawing helpers
# ──────────────────────────────────────────────

def _pt(arr: np.ndarray) -> tuple:
    """Convert (x,y,...) array to integer (x,y) tuple for OpenCV."""
    return (int(arr[0]), int(arr[1]))


def _draw_thick_line(
    img: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    color: tuple,
    thickness: int = 3,
) -> None:
    cv2.line(img, _pt(p1), _pt(p2), color, thickness, cv2.LINE_AA)


def _draw_angle_arc(
    img: np.ndarray,
    vertex: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    angle_deg: float,
    color: tuple,
    radius: int = 28,
) -> None:
    """Draw a small arc at `vertex` showing the angle between p1-vertex-p2."""
    v1 = (p1[:2] - vertex[:2])
    v2 = (p2[:2] - vertex[:2])

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-3 or n2 < 1e-3:
        return

    start_angle = math.degrees(math.atan2(v1[1], v1[0]))
    end_angle   = math.degrees(math.atan2(v2[1], v2[0]))

    diff = (end_angle - start_angle + 360) % 360
    if diff > 180:
        start_angle, end_angle = end_angle, start_angle

    cv2.ellipse(
        img, _pt(vertex),
        (radius, radius), 0,
        start_angle, end_angle,
        color, 2, cv2.LINE_AA
    )

    mid_angle_rad = math.radians((start_angle + end_angle) / 2)
    lx = int(vertex[0] + (radius + 18) * math.cos(mid_angle_rad))
    ly = int(vertex[1] + (radius + 18) * math.sin(mid_angle_rad))
    cv2.putText(
        img, f"{angle_deg:.0f}deg",          # fixed: no degree symbol
        (lx - 15, ly + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )


# ──────────────────────────────────────────────
# Leg skeleton overlay
# ──────────────────────────────────────────────

def draw_leg_skeleton(
    img: np.ndarray,
    kp: LegKeypoints,
    result_color: tuple = C_GREEN,
) -> None:
    segments = [
        (kp.left_hip,   kp.left_knee),
        (kp.left_knee,  kp.left_ankle),
        (kp.right_hip,  kp.right_knee),
        (kp.right_knee, kp.right_ankle),
        (kp.left_hip,   kp.right_hip),
        (kp.left_ankle, kp.right_ankle),
    ]

    for a, b in segments:
        _draw_thick_line(img, a, b, result_color, thickness=3)

    for joint in [kp.left_hip, kp.left_knee, kp.left_ankle,
                  kp.right_hip, kp.right_knee, kp.right_ankle]:
        cv2.circle(img, _pt(joint), 7, C_WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, _pt(joint), 7, result_color, 2, cv2.LINE_AA)

    for knee in [kp.left_knee, kp.right_knee]:
        cv2.circle(img, _pt(knee), 10, result_color, 2, cv2.LINE_AA)


def draw_angle_annotations(
    img: np.ndarray,
    kp: LegKeypoints,
    metrics: AngleMetrics,
    result_color: tuple = C_GREEN,
) -> None:
    """Draw knee angle arcs and labels."""
    _draw_angle_arc(
        img, kp.left_knee,  kp.left_hip,  kp.left_ankle,
        metrics.left_knee_angle,  result_color
    )
    _draw_angle_arc(
        img, kp.right_knee, kp.right_hip, kp.right_ankle,
        metrics.right_knee_angle, result_color
    )


# ──────────────────────────────────────────────
# HUD panel  — TOP RIGHT, larger
# ──────────────────────────────────────────────

def draw_hud(
    img: np.ndarray,
    clf_result: ClassificationResult,
    metrics: Optional[AngleMetrics] = None,
    show_metrics: bool = True,
) -> None:
    """
    Draw a large semi-transparent HUD panel in the TOP-RIGHT corner showing:
    - Classification label (big, coloured)
    - Confidence bar
    - All key metrics
    """
    h, w = img.shape[:2]

    # Compact panel
    panel_w = 240
    panel_h = 240 if (show_metrics and metrics) else 110
    panel   = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (15, 15, 15)

    color = RESULT_COLORS.get(clf_result.label, C_CYAN)

    # ── Label ─────────────────────────────────
    cv2.putText(
        panel, clf_result.label,
        (10, 34), cv2.FONT_HERSHEY_DUPLEX, 0.85,
        color, 2, cv2.LINE_AA
    )

    # ── Confidence text ────────────────────────
    cv2.putText(
        panel, f"Confidence: {clf_result.confidence * 100:.0f}%",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        C_WHITE, 1, cv2.LINE_AA
    )

    # ── Confidence bar ─────────────────────────
    bar_x  = 10
    bar_y  = 62
    bar_h  = 8
    bar_w  = int((panel_w - 20) * clf_result.confidence)
    cv2.rectangle(panel, (bar_x, bar_y), (panel_w - 10, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color, -1)

    # ── Method tag ────────────────────────────
    cv2.putText(
        panel, f"[{clf_result.method}]",
        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        (140, 140, 140), 1, cv2.LINE_AA
    )

    # ── Metrics ───────────────────────────────
    if show_metrics and metrics:
        metric_dict = format_metrics(metrics)
        y_pos = 102
        for key, val in metric_dict.items():
            val = val.replace("°", "deg")
            cv2.putText(
                panel, f"{key}: {val}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 210, 180), 1, cv2.LINE_AA
            )
            y_pos += 20

    # ── Paste panel into TOP-RIGHT of frame ───
    alpha  = 0.82
    margin = 10
    x0 = w - panel_w - margin
    y0 = margin
    roi = img[y0:y0 + panel_h, x0:x0 + panel_w]
    if roi.shape[:2] == panel.shape[:2]:
        blended = cv2.addWeighted(roi, 1 - alpha, panel, alpha, 0)
        img[y0:y0 + panel_h, x0:x0 + panel_w] = blended


# ──────────────────────────────────────────────
# Banners & status bar
# ──────────────────────────────────────────────

def draw_rejection_banner(img: np.ndarray, reason: str) -> None:
    """Draw a semi-transparent amber warning banner at the bottom."""
    h, w = img.shape[:2]
    banner_h = 44
    y0       = h - banner_h

    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 120, 200), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(
        img, f"  WARNING: {reason}",
        (10, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        C_WHITE, 1, cv2.LINE_AA
    )


def draw_status_bar(img: np.ndarray, text: str, color: tuple = C_GREEN) -> None:
    """Slim status bar at the very top of the frame."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(
        img, text,
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        color, 1, cv2.LINE_AA
    )


def draw_controls_help(img: np.ndarray) -> None:
    """Bottom-right corner hint."""
    h, w = img.shape[:2]
    cv2.putText(
        img, "Press any key to exit",
        (w - 220, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (160, 160, 160), 1, cv2.LINE_AA
    )


# ──────────────────────────────────────────────
# Frame upscaler  (called in run_image)
# ──────────────────────────────────────────────

def upscale_for_display(img: np.ndarray,
                        target_w: int = DISPLAY_WIDTH,
                        target_h: int = DISPLAY_HEIGHT) -> np.ndarray:
    """
    Resize the annotated frame to a larger display size while keeping
    the aspect ratio. Letterbox with black bars if needed.
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off  = (target_h - new_h) // 2
    x_off  = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas
