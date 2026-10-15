
from pathlib import Path
from collections import Counter

import torch
from ultralytics import YOLO


# Project directory
ROOT = Path(__file__).resolve().parent.parent

# Input and output image paths
INPUT_IMAGE = ROOT / "data" / "input" / "road.jpg"
OUTPUT_DIR = ROOT / "data" / "output"
OUTPUT_IMAGE = OUTPUT_DIR / "road_detected.jpg"

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def main():
    # Check if the input image exists
    if not INPUT_IMAGE.exists():
        print(f"ERROR: Image not found: {INPUT_IMAGE}")
        print("Place a road image at data/input/road.jpg")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select GPU if available, otherwise use CPU
    device = "0" if torch.cuda.is_available() else "cpu"

    print("Road Vision Agent: Vehicle Detection")
    print("-" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    # Load pretrained YOLO model
    print("\nLoading YOLO model...")
    model = YOLO("yolo26n.pt")

    # Detect vehicles in the image
    print("Detecting vehicles...")

    results = model.predict(
        source=str(INPUT_IMAGE),
        device=device,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.40,
        verbose=False,
    )

    result = results[0]
    counts = Counter()

    print("\nDetected vehicles")
    print("-" * 40)

    # Read detected classes and confidence scores
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        vehicle_name = VEHICLE_CLASSES[class_id]

        counts[vehicle_name] += 1

        print(
            f"{vehicle_name.capitalize():12s} "
            f"Confidence: {confidence:.2%}"
        )

    # Save the image with bounding boxes
    result.save(filename=str(OUTPUT_IMAGE))

    # Display vehicle totals
    print("\nVehicle Count Summary")
    print("-" * 40)

    for vehicle_name in VEHICLE_CLASSES.values():
        print(f"{vehicle_name.capitalize():12s}: {counts[vehicle_name]}")

    total = sum(counts.values())
    print("-" * 40)
    print(f"{'Total':12s}: {total}")

    print(f"\nAnnotated image saved to:\n{OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
