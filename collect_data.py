import argparse
import csv
import os
import sys
import time
import platform
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# ── Version check ────────────────────────────────────────────────────────────
print(f"MediaPipe version: {mp.__version__}")
print(f"OpenCV version   : {cv2.__version__}")

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_CSV    = os.path.join("Sign_Data", "sign_language_data.csv")
NUM_LANDMARKS = 21
COORDS        = ["x", "y", "z"]

ALL_GESTURES = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "Hello","Thank_You","Yes","No","Please",
    "Sorry","Help","More","Done","Good",
]

COLUMNS = (
    [f"lh_{i}_{c}" for i in range(NUM_LANDMARKS) for c in COORDS]
  + [f"rh_{i}_{c}" for i in range(NUM_LANDMARKS) for c in COORDS]
  + ["label"]
)

# ── Download model file for new MediaPipe API ────────────────────────────────
def download_model():
    model_path = "hand_landmarker.task"
    if os.path.exists(model_path):
        return model_path
    print("Downloading hand landmark model (~9 MB)...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded OK")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from:")
        print(url)
        print(f"Save as '{model_path}' in your project folder.")
        sys.exit(1)
    return model_path

# ── Create hand landmarker using NEW mediapipe tasks API ─────────────────────
def create_landmarker():
    model_path = download_model()
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)

# ── Extract landmarks from result ────────────────────────────────────────────
def extract_landmarks(detection_result):
    left  = [np.nan] * 63
    right = [np.nan] * 63

    if not detection_result.hand_landmarks:
        return left + right

    for hand_lm, handedness in zip(
        detection_result.hand_landmarks,
        detection_result.handedness
    ):
        coords = []
        for lm in hand_lm:
            coords.extend([lm.x, lm.y, lm.z])
        label = handedness[0].category_name
        if label == "Left":
            left = coords
        else:
            right = coords

    return left + right

# ── Draw skeleton on frame ───────────────────────────────────────────────────
def draw_skeleton(frame, detection_result):
    if not detection_result.hand_landmarks:
        return frame
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for hand_lm in detection_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)
        for pt in pts:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            cv2.circle(frame, pt, 5, (255, 255, 255), 1)
    return frame

# ── Draw HUD ─────────────────────────────────────────────────────────────────
def draw_hud(frame, gesture, collected, total_target, capturing, hand_found):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture}",
                (15, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 100), 2)

    if capturing:
        status = f"Recording: {collected} / {total_target}"
        col    = (100, 255, 100)
    else:
        status = "Press SPACE to start"
        col    = (200, 200, 200)
    cv2.putText(frame, status, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

    # Progress bar
    bx, by, bw, bh = 15, 105, w - 30, 14
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    if total_target > 0 and collected > 0:
        fill = int(bw * min(collected / total_target, 1.0))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (50, 220, 100), -1)
    pct = int(collected / total_target * 100) if total_target > 0 else 0
    cv2.putText(frame, f"{pct}%", (bx + bw // 2 - 18, by + bh - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Hand dot
    dot_col  = (50, 220, 50) if hand_found else (50, 50, 220)
    dot_text = "Hand: YES" if hand_found else "Hand: NO"
    cv2.circle(frame, (w - 22, 22), 11, dot_col, -1)
    cv2.putText(frame, dot_text, (w - 115, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, dot_col, 1)

    # Bottom hint
    cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "SPACE=Start   P=Pause   Q=Quit & Save",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return frame

# ── Camera helpers ────────────────────────────────────────────────────────────
def find_camera():
    print("\nScanning for cameras...")
    for i in range(4):
        print(f"  Trying {i}...", end=" ", flush=True)
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        cap = cv2.VideoCapture(i, backend) if backend else cv2.VideoCapture(i)
        if not cap.isOpened():
            print("not found"); cap.release(); continue
        ret, frame = cap.read()
        if not ret or frame is None or frame.mean() < 1:
            print("no image"); cap.release(); continue
        h2, w2 = frame.shape[:2]
        print(f"OK ({w2}x{h2})")
        cap.release()
        return i
    return None

def open_camera(idx):
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
    cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, _ = cap.read()
    return cap if ret else None

# ── CSV helpers ───────────────────────────────────────────────────────────────
def count_existing(gesture):
    if not os.path.exists(OUTPUT_CSV):
        return 0
    count = 0
    try:
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("label", "").strip() == gesture:
                    count += 1
    except Exception:
        pass
    return count

def has_header():
    if not os.path.exists(OUTPUT_CSV):
        return False
    try:
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            return "lh_0_x" in f.readline()
    except Exception:
        return False

# ── Main collection ───────────────────────────────────────────────────────────
def collect_gesture(gesture, num_samples, camera_idx=0):
    existing  = count_existing(gesture)
    remaining = num_samples - existing

    if remaining <= 0:
        print(f"Already have {existing}/{num_samples} for '{gesture}'. Skipping.")
        return True

    print(f"\n{'='*50}")
    print(f"  Gesture : {gesture}")
    print(f"  Target  : {num_samples}  Have: {existing}  Need: {remaining}")
    print(f"{'='*50}")

    os.makedirs("Sign_Data", exist_ok=True)

    cap = open_camera(camera_idx)
    if cap is None:
        print(f"Cannot open camera {camera_idx}. Try --camera 1")
        return False

    print("Loading hand detector (new MediaPipe Tasks API)...")
    landmarker = create_landmarker()
    print("Hand detector ready.\n")
    print("  1. Show your hand in front of the camera")
    print("  2. Wait for GREEN dot (hand detected)")
    print("  3. Press SPACE to start recording\n")

    need_header  = not has_header()
    csv_file     = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer       = csv.writer(csv_file)
    if need_header:
        writer.writerow(COLUMNS)

    collected  = 0
    capturing  = False
    paused     = False

    try:
        while collected < remaining:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)

            # Run detection using new Tasks API
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            hand_found = len(result.hand_landmarks) > 0

            # Draw skeleton
            frame = draw_skeleton(frame, result)

            # Draw HUD
            frame = draw_hud(
                frame, gesture,
                existing + collected, num_samples,
                capturing and not paused,
                hand_found,
            )

            cv2.imshow("Sign Language Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                capturing = True
                paused    = False
                print("  Recording started...")
            elif key == ord("p"):
                paused = not paused
                print(f"  {'Paused' if paused else 'Resumed'}")
            elif key in (ord("q"), 27):
                print(f"  Stopped. {collected} samples saved.")
                break

            # Save sample
            if capturing and not paused and hand_found:
                row = extract_landmarks(result)
                writer.writerow(row + [gesture])
                csv_file.flush()
                collected += 1
                if collected % 50 == 0:
                    print(f"  {existing + collected}/{num_samples} saved...")
                time.sleep(0.05)

    except KeyboardInterrupt:
        print(f"\nStopped. {collected} samples saved.")
    finally:
        landmarker.close()
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()

    total = existing + collected
    print(f"\n{'='*50}")
    print(f"  Done! {collected} new samples collected.")
    print(f"  Total for '{gesture}': {total}/{num_samples}")
    print(f"  Saved: {os.path.abspath(OUTPUT_CSV)}")
    print(f"{'='*50}\n")
    return True

# ── Verify CSV ────────────────────────────────────────────────────────────────
def verify_csv():
    if not os.path.exists(OUTPUT_CSV):
        print("No CSV found. Collect data first.")
        return
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    print(f"\nTotal rows   : {len(df)}")
    print(f"Columns      : {len(df.columns)} (need 127)")
    print(f"\nSamples per gesture:")
    for g in ALL_GESTURES:
        c    = (df["label"] == g).sum()
        icon = "OK" if c >= 200 else ("LOW" if c > 0 else "---")
        print(f"  [{icon}] {g:15s}: {c}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", type=str, default=None)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--camera",  type=int, default=None)
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--verify",  action="store_true")
    parser.add_argument("--list",    action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify_csv(); sys.exit(0)

    if args.list:
        existing = {}
        if os.path.exists(OUTPUT_CSV):
            import pandas as pd
            try:
                df = pd.read_csv(OUTPUT_CSV)
                existing = df["label"].value_counts().to_dict()
            except Exception:
                pass
        for g in ALL_GESTURES:
            print(f"  {g:15s}: {existing.get(g, 0)} samples")
        sys.exit(0)

    cam = args.camera
    if cam is None:
        cam = find_camera()
        if cam is None:
            print("No camera found. Try: python collect_data.py --gesture A --camera 1")
            sys.exit(1)

    if args.all:
        for i, g in enumerate(ALL_GESTURES, 1):
            print(f"\n[{i}/{len(ALL_GESTURES)}] Gesture: {g}")
            collect_gesture(g, args.samples, cam)
            time.sleep(1)
        verify_csv()
    elif args.gesture:
        collect_gesture(args.gesture, args.samples, cam)
    else:
        print("Usage: python collect_data.py --gesture A --samples 300")

