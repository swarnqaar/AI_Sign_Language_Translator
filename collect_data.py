"""
collect_data.py  –  Landmark Feature Extractor
================================================
Run this script BEFORE training to build your dataset.

Usage:
    python collect_data.py --gesture A --samples 200
    python collect_data.py --gesture Hello --samples 200

What it does:
  1. Opens the webcam.
  2. Runs MediaPipe Hands on every frame.
  3. Extracts the (x, y, z) coordinates of all 21 landmarks per hand
     (flattened to a 126-dimensional vector for 2 hands).
  4. Appends each vector + the gesture label to Sign_Data/sign_language_data.csv.

If fewer than 2 hands are visible, missing hand landmarks are filled with NaN
(the KNNImputer in DataTransformation handles this later).

Requirements:
    pip install mediapipe opencv-python
"""

import argparse
import csv
import os
import time

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import numpy as np


OUTPUT_CSV = os.path.join("Sign_Data", "sign_language_data.csv")
NUM_LANDMARKS = 21
NUM_HANDS = 2
COORDS = ["x", "y", "z"]

# Build column headers: lh_0_x, lh_0_y, lh_0_z, … rh_20_z, label
COLUMNS = (
    [f"lh_{i}_{c}" for i in range(NUM_LANDMARKS) for c in COORDS]
    + [f"rh_{i}_{c}" for i in range(NUM_LANDMARKS) for c in COORDS]
    + ["label"]
)


def extract_landmarks(hand_landmarks_list, handedness_list):
    """
    Returns a flat list of 126 floats (left hand + right hand).
    Missing hands are represented as NaN values.
    """
    left = [np.nan] * (NUM_LANDMARKS * 3)
    right = [np.nan] * (NUM_LANDMARKS * 3)

    for idx, (hand_lm, handedness) in enumerate(
        zip(hand_landmarks_list, handedness_list)
    ):
        coords = []
        for lm in hand_lm.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        label = handedness.classification[0].label  # "Left" or "Right"
        if label == "Left":
            left = coords
        else:
            right = coords

    return left + right


def main(gesture: str, num_samples: int):
    os.makedirs("Sign_Data", exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    collected = 0
    print(f"\nCollecting {num_samples} samples for gesture '{gesture}'.")
    print("Press SPACE to start capturing, Q to quit early.\n")

    capturing = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands, open(OUTPUT_CSV, "a", newline="") as csvfile:

        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(COLUMNS)

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS
                    )

            status = (
                f"Capturing: {collected}/{num_samples}"
                if capturing
                else "Press SPACE to start"
            )
            cv2.putText(
                frame, f"Gesture: {gesture}  |  {status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imshow("Sign Language Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                capturing = True
            elif key == ord("q"):
                break

            if capturing and results.multi_hand_landmarks:
                row = extract_landmarks(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                )
                writer.writerow(row + [gesture])
                collected += 1
                time.sleep(0.05)  # ~20 fps capture rate

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {collected} samples for '{gesture}' saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sign language landmark data")
    parser.add_argument("--gesture", required=True, help="Gesture label, e.g. 'A'")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples")
    args = parser.parse_args()
    main(args.gesture, args.samples)
