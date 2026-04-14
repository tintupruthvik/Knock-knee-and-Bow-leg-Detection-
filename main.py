"""
main.py
-------
Entry point for the Leg Alignment Analysis System (Rule-Based Classifier).

Usage
-----
# Single image (webcam / video removed for simplicity)
python main.py --input path/to/image.jpg

# Save annotated output
python main.py --input path/to/image.jpg --output annotated.jpg

# Hide the metrics panel
python main.py --input path/to/image.jpg --no-metrics

Options
-------
  --input FILE           Path to image file
  --complexity {0,1,2}   MediaPipe model complexity (default: 1)
  --no-metrics           Hide metrics panel in HUD
  --output FILE          Save annotated output image
"""

import argparse
import sys
import numpy as np
import cv2

from pose_detection    import PoseDetector
from angle_calculation import compute_metrics
from classification    import (
    RuleBasedClassifier,
    TemporalSmoother,
    ClassificationResult,
)
from visualisation import (
    draw_leg_skeleton, draw_angle_annotations,
    draw_hud, draw_rejection_banner,
    draw_status_bar, draw_controls_help,
    upscale_for_display,
    RESULT_COLORS,
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leg Alignment Analyser — knock knees / bow legs / normal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--input", type=str, required=True,
                   help="Path to input image")
    p.add_argument("--complexity", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--output", type=str, default=None,
                   help="Save annotated output image")

    return p.parse_args()


# ──────────────────────────────────────────────
# Frame processor  (core pipeline)
# ──────────────────────────────────────────────

class FrameProcessor:
    """
    Runs the full pipeline on a single BGR frame:
      pose detection → metrics → classification → visualisation.

    Returns the annotated frame.
    """

    def __init__(self, detector, classifier, smoother, show_metrics: bool = True):
        self.detector     = detector
        self.classifier   = classifier
        self.smoother     = smoother
        self.show_metrics = show_metrics

    def process(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()

        # ── Pose detection ─────────────────────
        pose_result = self.detector.process_frame(frame)

        # Always draw the MediaPipe skeleton if landmarks exist
        if pose_result.raw_landmarks is not None:
            out = self.detector.draw_skeleton(out, pose_result)

        if not pose_result.valid:
            draw_rejection_banner(out, pose_result.rejection_reason)
            draw_status_bar(out, "Analysing… waiting for valid pose",
                            color=(0, 180, 220))
            draw_controls_help(out)
            return out

        # ── Angle metrics ──────────────────────
        metrics = compute_metrics(pose_result.keypoints)

        # ── Classification (rule-based only) ───
        raw_result = self.classifier.classify(metrics)
        result     = self.smoother.update(raw_result)

        result_color = RESULT_COLORS.get(result.label, (200, 200, 200))

        # ── Visualisation overlays ─────────────
        draw_leg_skeleton(out, pose_result.keypoints, result_color)
        draw_angle_annotations(out, pose_result.keypoints, metrics, result_color)
        draw_hud(out, result, metrics if self.show_metrics else None,
                 show_metrics=self.show_metrics)

        # Warnings
        if result.warnings:
            draw_rejection_banner(out, " | ".join(result.warnings))

        label_text = f"Result: {result.label}  ({result.confidence*100:.0f}%)"
        draw_status_bar(out, label_text, color=result_color)
        draw_controls_help(out)

        return out


# ──────────────────────────────────────────────
# Image mode
# ──────────────────────────────────────────────

def run_image(args: argparse.Namespace, processor: FrameProcessor) -> None:
    import os
    frame = cv2.imread(args.input)
    if frame is None:
        sys.exit(f"[ERROR] Cannot read image: {args.input}")

    annotated = processor.process(frame)

    # Always save output
    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(args.input)
        out_path = base + "_annotated" + (ext if ext else ".jpg")
    cv2.imwrite(out_path, annotated)
    print(f"[INFO] Saved annotated image -> {out_path}")

    # Upscale for a bigger display window
    display_frame = upscale_for_display(annotated)

    try:
        cv2.imshow("Leg Alignment Analysis", display_frame)
        print("[INFO] Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("[INFO] No display available - output saved to file only.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    classifier = RuleBasedClassifier()
    print(f"[INFO] Classifier: RuleBasedClassifier")

    detector = PoseDetector(
        static_image_mode=True,
        model_complexity=args.complexity,
    )

    smoother = TemporalSmoother(window=8)

    processor = FrameProcessor(
        detector=detector,
        classifier=classifier,
        smoother=smoother,
        show_metrics=not args.no_metrics,
    )

    try:
        run_image(args, processor)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
