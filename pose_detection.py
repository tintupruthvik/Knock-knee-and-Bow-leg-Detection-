"""
pose_detection.py
-----------------
Compatible with MediaPipe 0.10.30+ (Tasks API only).
Uses mediapipe.tasks.python.vision.PoseLandmarker.
"""

import cv2
import numpy as np
import urllib.request
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

# ──────────────────────────────────────────────
# Download model file if not present
# ──────────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[PoseDetector] Downloading pose model (~5 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[PoseDetector] Model saved → {MODEL_PATH}")

# ──────────────────────────────────────────────
# Landmark indices (BlazePose 33-point schema)
# ──────────────────────────────────────────────

LANDMARK_IDX = {
    "nose":            0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}

POSE_CONNECTIONS_FALLBACK = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

MIN_VISIBILITY = 0.55
REQUIRED_JOINTS = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
]

# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class LegKeypoints:
    left_hip:    np.ndarray
    left_knee:   np.ndarray
    left_ankle:  np.ndarray
    right_hip:   np.ndarray
    right_knee:  np.ndarray
    right_ankle: np.ndarray
    visibility:  Dict[str, float]

@dataclass
class PoseResult:
    keypoints:        Optional[LegKeypoints]
    raw_landmarks:    Optional[object]
    valid:            bool
    rejection_reason: str

# ──────────────────────────────────────────────
# PoseDetector
# ──────────────────────────────────────────────

class PoseDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        _ensure_model()

        # Map complexity 0/1/2 to lite/full/heavy model names
        # We always download 'full'; complexity param is noted but one model used
        running_mode = (
            mp_vision.RunningMode.IMAGE
            if static_image_mode
            else mp_vision.RunningMode.VIDEO
        )

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.static_image_mode = static_image_mode
        self._frame_ts = 0  # millisecond timestamp for VIDEO mode

    # ── Core processing ───────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.static_image_mode:
            result = self.landmarker.detect(mp_image)
        else:
            self._frame_ts += 33  # ~30 fps assumed
            result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return PoseResult(None, None, False, "No person detected in frame")

        lm = result.pose_landmarks[0]   # first (and only) person

        def pt(name: str) -> np.ndarray:
            l = lm[LANDMARK_IDX[name]]
            return np.array([l.x * w, l.y * h, l.z * w], dtype=np.float32)

        def vis(name: str) -> float:
            v = lm[LANDMARK_IDX[name]].visibility
            return float(v) if v is not None else 0.0

        # Visibility check
        visibility = {j: vis(j) for j in REQUIRED_JOINTS}
        low = [j for j, v in visibility.items() if v < MIN_VISIBILITY]
        if low:
            return PoseResult(
                None, lm, False,
                f"Low visibility: {', '.join(low)}"
            )

        kp = LegKeypoints(
            left_hip=pt("left_hip"),
            left_knee=pt("left_knee"),
            left_ankle=pt("left_ankle"),
            right_hip=pt("right_hip"),
            right_knee=pt("right_knee"),
            right_ankle=pt("right_ankle"),
            visibility=visibility,
        )

        ok, reason = self._validate_stance(kp, w, h)
        if not ok:
            return PoseResult(kp, lm, False, reason)

        return PoseResult(kp, lm, True, "")

    def _validate_stance(self, kp, frame_w, frame_h) -> Tuple[bool, str]:
        for side, hip, knee, ankle in [
            ("left",  kp.left_hip,  kp.left_knee,  kp.left_ankle),
            ("right", kp.right_hip, kp.right_knee, kp.right_ankle),
        ]:
            if not (hip[1] < knee[1] < ankle[1]):
                return False, f"Not standing upright ({side} leg order wrong)"

        hip_slope = abs(kp.left_hip[1] - kp.right_hip[1])
        hip_width = abs(kp.left_hip[0] - kp.right_hip[0]) + 1e-6
        if hip_slope / hip_width > 0.4:
            return False, "Person appears tilted sideways"

        ankle_y_norm = max(kp.left_ankle[1], kp.right_ankle[1]) / frame_h
        if ankle_y_norm < 0.55:
            return False, "Full legs not visible — move camera back"

        hip_y_norm = min(kp.left_hip[1], kp.right_hip[1]) / frame_h
        if hip_y_norm > 0.65:
            return False, "Only lower body visible — move camera back"

        return True, ""

    # ── Drawing helpers ────────────────────────

    def draw_skeleton(self, frame: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        out = frame.copy()
        lm = pose_result.raw_landmarks
        if lm is None:
            return out

        h, w = frame.shape[:2]

        # Draw connections
        for a_idx, b_idx in POSE_CONNECTIONS_FALLBACK:
            a = lm[a_idx]
            b = lm[b_idx]
            p1 = (int(a.x * w), int(a.y * h))
            p2 = (int(b.x * w), int(b.y * h))
            cv2.line(out, p1, p2, (0, 180, 255), 2, cv2.LINE_AA)

        # Draw joints
        for l in lm:
            cx, cy = int(l.x * w), int(l.y * h)
            cv2.circle(out, (cx, cy), 4, (0, 255, 200), -1, cv2.LINE_AA)

        return out

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
