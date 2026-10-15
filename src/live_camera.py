from collections import Counter
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

from track_vehicles import (
    VEHICLE_CLASSES,
    CONFIDENCE,
    IMAGE_SIZE,
    LINE_HEIGHT_RATIO,
    LINE_TOLERANCE,
    create_tracker_config,
)

ROOT = Path(__file__).resolve().parent.parent
TRACKER_CONFIG = ROOT / "bytetrack_road.yaml"


def main():
    device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Road Vision Agent Live Camera | Device: {device}")
    print("Press Q in the camera window to quit.")

    create_tracker_config()
    model = YOLO("yolo26n.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions "
            "and close other applications using the camera."
        )

    previous_side = {}
    counted_ids = set()
    counts = Counter()

    frames_processed = 0
    fps_started = time.perf_counter()
    measured_fps = 0.0

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Unable to read a frame from the webcam.")
                break

            height, width = frame.shape[:2]
            line_y = int(height * LINE_HEIGHT_RATIO)

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

            # Horizontal counting line.
            cv2.line(
                frame,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                2,
            )

            boxes = result.boxes

            if (
                boxes is not None
                and boxes.id is not None
                and len(boxes) > 0
            ):
                coordinates = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for box, class_id, track_id, confidence in zip(
                    coordinates,
                    class_ids,
                    track_ids,
                    confidences,
                ):
                    x1, y1, x2, y2 = map(int, box)
                    vehicle = VEHICLE_CLASSES.get(class_id)

                    if vehicle is None:
                        continue

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
                        4,
                        (255, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        frame,
                        f"{vehicle} ID:{track_id} {confidence:.2f}",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                    if anchor_y < line_y - LINE_TOLERANCE:
                        current_side = -1
                    elif anchor_y > line_y + LINE_TOLERANCE:
                        current_side = 1
                    else:
                        current_side = 0

                    old_side = previous_side.get(track_id)

                    crossed = (
                        old_side == -1
                        and current_side == 1
                        and track_id not in counted_ids
                    )

                    if crossed:
                        counted_ids.add(track_id)
                        counts[vehicle] += 1
                        print(
                            f"COUNTED: {vehicle}, ID={track_id}, "
                            f"total={len(counted_ids)}"
                        )

                    # Preserve the last stable side near the line.
                    if current_side != 0:
                        previous_side[track_id] = current_side

            frames_processed += 1
            elapsed = time.perf_counter() - fps_started

            if elapsed >= 1.0:
                measured_fps = frames_processed / elapsed
                frames_processed = 0
                fps_started = time.perf_counter()

            # Dashboard overlay.
            cv2.rectangle(
                frame,
                (10, 10),
                (360, 145),
                (0, 0, 0),
                -1,
            )

            overlay = [
                f"Total crossed: {len(counted_ids)}",
                f"Cars: {counts['car']}  Trucks: {counts['truck']}",
                f"Bikes: {counts['motorcycle']}  Buses: {counts['bus']}",
                f"Processing FPS: {measured_fps:.1f}",
            ]

            for index, text in enumerate(overlay):
                cv2.putText(
                    frame,
                    text,
                    (20, 38 + index * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Road Vision Agent - Live Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nLive session totals:")
    for vehicle in VEHICLE_CLASSES.values():
        print(f"{vehicle}: {counts[vehicle]}")
    print(f"Total crossings: {len(counted_ids)}")


if __name__ == "__main__":
    main()
