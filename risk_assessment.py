"""Experimental risk assessment for Person, Fire, and Smoke detections.

This is a decision-support prototype, not a safety controller. A low risk
result means only that no supported hazard class was confirmed by this model.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
from ultralytics import YOLO


CLASS_NAMES = ("Person", "Fire", "Smoke")
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "yolo11n_v2_best.pt"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionContext:
    class_name: str
    confidence: float
    relative_size: str
    position: str


@dataclass(frozen=True)
class RiskAssessment:
    hazard_level: str
    rescue_priority: str
    confirmed_classes: tuple[str, ...]
    candidates: tuple[str, ...]
    frame_counts: tuple[tuple[str, int], ...]
    detections: tuple[DetectionContext, ...]


class RiskAssessor:
    """Apply confidence and short-term persistence rules to detector outputs."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        window_frames: int = 5,
        confirm_frames: int = 3,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if window_frames < 1:
            raise ValueError("window_frames must be at least 1")
        if not 1 <= confirm_frames <= window_frames:
            raise ValueError("confirm_frames must be between 1 and window_frames")

        self.confidence_threshold = confidence_threshold
        self.confirm_frames = confirm_frames
        self._history: deque[dict[str, bool]] = deque(maxlen=window_frames)

    def update(
        self,
        detections: Sequence[Detection],
        image_width: int,
        image_height: int,
    ) -> RiskAssessment:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")

        accepted = [
            detection
            for detection in detections
            if detection.class_name in CLASS_NAMES
            and detection.confidence >= self.confidence_threshold
        ]
        present = {name: False for name in CLASS_NAMES}
        contexts = []
        for detection in accepted:
            present[detection.class_name] = True
            contexts.append(
                _make_context(detection, image_width=image_width, image_height=image_height)
            )
        self._history.append(present)

        frame_counts = {
            name: sum(frame[name] for frame in self._history)
            for name in CLASS_NAMES
        }
        confirmed = tuple(
            name for name in CLASS_NAMES
            if frame_counts[name] >= self.confirm_frames
        )
        candidates = tuple(
            name for name in CLASS_NAMES
            if present[name] and name not in confirmed
        )

        if "Fire" in confirmed:
            hazard_level = "HIGH"
        elif "Smoke" in confirmed:
            hazard_level = "MODERATE"
        else:
            hazard_level = "LOW"

        if "Person" in confirmed:
            rescue_priority = "PERSON PRESENT"
        elif "Person" in candidates:
            rescue_priority = "PERSON CANDIDATE"
        else:
            rescue_priority = "NONE CONFIRMED"

        return RiskAssessment(
            hazard_level=hazard_level,
            rescue_priority=rescue_priority,
            confirmed_classes=confirmed,
            candidates=candidates,
            frame_counts=tuple((name, frame_counts[name]) for name in CLASS_NAMES),
            detections=tuple(contexts),
        )


def _make_context(
    detection: Detection,
    image_width: int,
    image_height: int,
) -> DetectionContext:
    x1, y1, x2, y2 = detection.xyxy
    box_width = max(0.0, min(x2, image_width) - max(x1, 0.0))
    box_height = max(0.0, min(y2, image_height) - max(y1, 0.0))
    relative_area = (box_width * box_height) / (image_width * image_height)

    if relative_area < 0.05:
        relative_size = "small"
    elif relative_area < 0.20:
        relative_size = "medium"
    else:
        relative_size = "large"

    center_x = (x1 + x2) / 2 / image_width
    center_y = (y1 + y2) / 2 / image_height
    horizontal = "left" if center_x < 1 / 3 else "right" if center_x >= 2 / 3 else "center"
    vertical = "top" if center_y < 1 / 3 else "bottom" if center_y >= 2 / 3 else "middle"

    # Geometry is context only; without calibration it must not change urgency.
    return DetectionContext(
        class_name=detection.class_name,
        confidence=detection.confidence,
        relative_size=relative_size,
        position=f"{horizontal}/{vertical}",
    )


def _parse_source(source: str) -> str | int:
    return int(source) if source.isdecimal() else source


def _source_fps(source: str | int) -> float:
    if isinstance(source, int):
        return 20.0

    capture = cv2.VideoCapture(source)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return fps if fps > 0 else 20.0


def _draw_assessment(frame, assessment: RiskAssessment) -> None:
    lines = [
        f"HAZARD: {assessment.hazard_level} (not a safety guarantee)",
        f"RESCUE PRIORITY: {assessment.rescue_priority}",
    ]
    if assessment.confirmed_classes:
        lines.append("CONFIRMED: " + ", ".join(assessment.confirmed_classes))
    if assessment.candidates:
        lines.append("CANDIDATE: " + ", ".join(assessment.candidates))
    for detection in assessment.detections[:4]:
        lines.append(
            f"{detection.class_name} {detection.confidence:.2f} | "
            f"{detection.relative_size} | {detection.position}"
        )

    line_height = 22
    panel_height = 12 + line_height * len(lines)
    cv2.rectangle(frame, (8, 8), (570, panel_height), (25, 25, 25), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 30 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    source = _parse_source(args.source)
    model = YOLO(str(model_path))
    fps = _source_fps(source)
    assessor = RiskAssessor(
        confidence_threshold=args.confidence,
        window_frames=args.window_frames,
        confirm_frames=args.confirm_frames,
    )

    source_name = f"camera_{source}" if isinstance(source, int) else Path(source).stem
    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent
        / "runs"
        / "risk_assessment"
        / f"{source_name}_risk.avi"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    previous_status = None
    confirmed_frame_counts: Counter[str] = Counter()
    frames_processed = 0
    try:
        for result in model.predict(
            source=source,
            stream=True,
            conf=args.confidence,
            verbose=False,
        ):
            frames_processed += 1
            height, width = result.orig_shape
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    detections.append(
                        Detection(
                            class_name=str(model.names[class_id]),
                            confidence=float(box.conf.item()),
                            xyxy=tuple(float(value) for value in box.xyxy[0].tolist()),
                        )
                    )

            assessment = assessor.update(detections, image_width=width, image_height=height)
            confirmed_frame_counts.update(assessment.confirmed_classes)
            status = (
                assessment.hazard_level,
                assessment.rescue_priority,
                assessment.confirmed_classes,
                assessment.candidates,
            )
            if status != previous_status:
                print(
                    f"Frame {frames_processed}: hazard={assessment.hazard_level}, "
                    f"rescue={assessment.rescue_priority}, "
                    f"confirmed={assessment.confirmed_classes or 'none'}, "
                    f"candidates={assessment.candidates or 'none'}"
                )
                previous_status = status

            frame = result.plot()
            _draw_assessment(frame, assessment)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"XVID"),
                    fps if fps > 0 else 20.0,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise OSError(f"Could not create output video: {output_path}")
            writer.write(frame)

            if args.show:
                cv2.imshow("Risk assessment prototype", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frames_processed == 0:
        raise RuntimeError(f"No frames were read from source: {args.source}")

    print(f"Frames processed: {frames_processed}")
    for name in CLASS_NAMES:
        print(
            f"{name}: confirmed in {confirmed_frame_counts[name]} "
            f"frames ({confirmed_frame_counts[name] / frames_processed:.1%})"
        )
    print(f"Annotated output: {output_path.resolve()}")
    print("Interpret detections as experimental decision support, not proof of safety.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO detections through the experimental risk-assessment rules."
    )
    parser.add_argument("--source", required=True, help="Video path or camera index (for example: 0)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained YOLO .pt model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence for assessment")
    parser.add_argument("--window-frames", type=int, default=5, help="Persistence window size")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Frames required to confirm a class")
    parser.add_argument("--output", help="Annotated AVI output path")
    parser.add_argument("--show", action="store_true", help="Display the processed frames; press Q to stop")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
