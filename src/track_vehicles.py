
from pathlib import Path
from collections import Counter

import csv
import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
INPUT_VIDEO = ROOT / "data" / "input" / "road_video.mp4"
OUTPUT_DIR = ROOT / "data" / "output"
TRACKER_CONFIG = ROOT / "bytetrack_road.yaml"
MODEL_PATH = ROOT / "yolo26n.pt"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIDENCE = 0.15
IMAGE_SIZE = 640
LINE_HEIGHT_RATIO = 0.50
LINE_TOLERANCE = 8

DIRECTIONS = ("top_to_bottom", "bottom_to_top")


def create_tracker_config():
    TRACKER_CONFIG.write_text(
        """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 60
match_thresh: 0.80
fuse_score: True
""",
        encoding="utf-8",
    )


def analyze_video(input_path, output_video, output_csv):
    input_path = Path(input_path)
    output_video = Path(output_video)
    output_csv = Path(output_csv)

    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    create_tracker_config()

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\nRoad Vision Agent | Device: {device}")
    print(f"Input video: {input_path}")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    writer = None
    frame_number = 0
    fps = 0.0
    width = 0
    height = 0

    # Each track remembers its last stable side of the line.
    previous_side = {}

    # A track may count once in each direction.
    counted_events = set()
    vehicle_counts = Counter()
    direction_counts = Counter()
    crossing_events = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid video properties.")

        line_y = int(height * LINE_HEIGHT_RATIO)

        print(f"Video dimensions: {width}x{height}")
        print(f"FPS: {fps:.2f} | Total frames: {total_frames}")
        print(f"Counting line Y: {line_y}")

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError("Could not create output video.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1

            result = model.track(
                source=frame,
                persist=True,
                tracker=str(TRACKER_CONFIG),
                classes=list(VEHICLE_CLASSES.keys()),
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]

            # Draw the counting line.
            cv2.line(
                frame,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                3,
            )

            boxes = result.boxes
            has_ids = (
                boxes is not None
                and boxes.id is not None
                and len(boxes) > 0
            )

            if has_ids:
                coordinates = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for box, class_id, track_id, confidence in zip(
                    coordinates, class_ids, track_ids, confidences
                ):
                    vehicle = VEHICLE_CLASSES.get(class_id)
                    if vehicle is None:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    # Bottom-center anchor for road traffic.
                    anchor_x = (x1 + x2) // 2
                    anchor_y = y2

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(
                        frame,
                        (anchor_x, anchor_y),
                        5,
                        (255, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        f"{vehicle} ID:{track_id} {confidence:.2f}",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # Determine which side of the line the anchor is on.
                    if anchor_y < line_y - LINE_TOLERANCE:
                        current_side = -1
                    elif anchor_y > line_y + LINE_TOLERANCE:
                        current_side = 1
                    else:
                        # Stay inside the tolerance band without changing
                        # the last stable side.
                        continue

                    old_side = previous_side.get(track_id)

                    if old_side is not None and old_side != current_side:
                        if old_side == -1 and current_side == 1:
                            direction = "top_to_bottom"
                        elif old_side == 1 and current_side == -1:
                            direction = "bottom_to_top"
                        else:
                            direction = None

                        event_key = (track_id, direction)

                        if (
                            direction is not None
                            and event_key not in counted_events
                        ):
                            counted_events.add(event_key)
                            vehicle_counts[vehicle] += 1
                            direction_counts[direction] += 1

                            crossing_events.append({
                                "frame": frame_number,
                                "timestamp_seconds": round(
                                    frame_number / fps, 2
                                ),
                                "track_id": track_id,
                                "vehicle_type": vehicle,
                                "confidence": round(float(confidence), 4),
                                "direction": direction,
                            })

                            print(
                                f"COUNTED: {vehicle} | ID {track_id} | "
                                f"{direction} | Frame {frame_number}"
                            )

                    previous_side[track_id] = current_side

            # Overlay uses exactly the same event counters as the CSV/API.
            top_count = direction_counts["top_to_bottom"]
            bottom_count = direction_counts["bottom_to_top"]
            total_count = top_count + bottom_count

            cv2.rectangle(
                frame, (10, 10), (460, 145), (0, 0, 0), -1
            )
            overlay_lines = [
                f"Total crossings: {total_count}",
                f"Top-to-bottom: {top_count}",
                f"Bottom-to-top: {bottom_count}",
                (
                    f"Cars:{vehicle_counts['car']} "
                    f"Trucks:{vehicle_counts['truck']}"
                ),
                (
                    f"Bikes:{vehicle_counts['motorcycle']} "
                    f"Buses:{vehicle_counts['bus']}"
                ),
            ]

            for index, text in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    text,
                    (20, 35 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            writer.write(frame)

            if frame_number % 100 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} frames | "
                    f"Crossings: {total_count}"
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    columns = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "vehicle_type",
        "confidence",
        "direction",
    ]

    with output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        csv_writer = csv.DictWriter(file, fieldnames=columns)
        csv_writer.writeheader()
        csv_writer.writerows(crossing_events)

    counts = {
        vehicle: int(vehicle_counts[vehicle])
        for vehicle in VEHICLE_CLASSES.values()
    }

    summary = {
        "total_counted": len(crossing_events),
        "vehicle_counts": counts,
        "top_to_bottom_count": int(
            direction_counts["top_to_bottom"]
        ),
        "bottom_to_top_count": int(
            direction_counts["bottom_to_top"]
        ),
        "crossing_events": crossing_events,
        "frames_processed": frame_number,
        "fps": round(fps, 2),
        "line_y": line_y,
        "device": "cuda" if device == "0" else "cpu",
        "annotated_video": str(output_video),
        "crossing_report": str(output_csv),
    }

    print("\nCrossing Counts")
    print("-" * 35)
    for vehicle, count in counts.items():
        print(f"{vehicle.capitalize():12s}: {count}")

    print(f"Top to bottom: {summary['top_to_bottom_count']}")
    print(f"Bottom to top: {summary['bottom_to_top_count']}")
    print(f"Total crossings: {summary['total_counted']}")
    print(f"Annotated video: {output_video}")
    print(f"Crossing report: {output_csv}")

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analyze_video(
        INPUT_VIDEO,
        OUTPUT_DIR / "road_tracked.mp4",
        OUTPUT_DIR / "vehicle_crossings.csv",
    )


if __name__ == "__main__":
    main()
