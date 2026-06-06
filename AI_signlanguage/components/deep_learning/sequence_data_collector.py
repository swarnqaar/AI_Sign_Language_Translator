"""
sequence_data_collector.py
===========================
Collects 30-frame gesture sequences for CNN+LSTM training.
Works with MediaPipe 0.10+ (new Tasks API).

USAGE:
  python sequence_data_collector.py
  python sequence_data_collector.py --gesture A
  python sequence_data_collector.py --gesture A --sequences 100
"""

import os
import sys
import time
import argparse
import platform
import urllib.request
import numpy as np
import cv2
import mediapipe as mp

# ── Silence warnings ──────────────────────────────────────────────────────────
os.environ["GLOG_minloglevel"]    = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print(f"MediaPipe : {mp.__version__}")
print(f"OpenCV    : {cv2.__version__}")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join("Sign_Data", "sequences")
SEQUENCE_LENGTH = 30    # frames per clip
NUM_SEQUENCES   = 100   # clips per gesture class

ALL_GESTURES = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "Hello","Thank_You","Yes","No","Please",
    "Sorry","Help","More","Done","Good",
]

# ── Download MediaPipe model ──────────────────────────────────────────────────
def download_model():
    model_path = "hand_landmarker.task"
    if os.path.exists(model_path):
        return model_path
    print("\nDownloading hand landmark model (~9 MB)...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded OK\n")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Download manually from:")
        print(url)
        print(f"Save as '{model_path}' in your project folder.")
        sys.exit(1)
    return model_path

# ── Create landmarker ─────────────────────────────────────────────────────────
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

# ── Extract 126-dim landmark vector ──────────────────────────────────────────
def extract_keypoints(detection_result):
    left  = np.zeros(63)
    right = np.zeros(63)
    if detection_result.hand_landmarks:
        for hand_lm, handedness in zip(
            detection_result.hand_landmarks,
            detection_result.handedness
        ):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm]
            ).flatten()
            label = handedness[0].category_name
            if label == "Left":
                left = coords
            else:
                right = coords
    return np.concatenate([left, right])

# ── Draw skeleton ─────────────────────────────────────────────────────────────
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

# ── Open camera ───────────────────────────────────────────────────────────────
def open_camera(idx=0):
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

# ── Find camera ───────────────────────────────────────────────────────────────
def find_camera():
    for i in range(4):
        print(f"  Trying camera {i}...", end=" ", flush=True)
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        cap = cv2.VideoCapture(i, backend) if backend else cv2.VideoCapture(i)
        if not cap.isOpened():
            print("not found"); cap.release(); continue
        ret, frame = cap.read()
        if not ret or frame is None or frame.mean() < 1:
            print("no image"); cap.release(); continue
        h, w = frame.shape[:2]
        print(f"OK ({w}x{h})")
        cap.release()
        return i
    return None

# ── Create folder structure ───────────────────────────────────────────────────
def create_folders(gestures, num_sequences):
    for gesture in gestures:
        for seq in range(num_sequences):
            os.makedirs(
                os.path.join(DATA_PATH, gesture, str(seq)),
                exist_ok=True
            )

# ── Count existing sequences ──────────────────────────────────────────────────
def count_existing(gesture, num_sequences):
    count = 0
    for seq in range(num_sequences):
        seq_dir = os.path.join(DATA_PATH, gesture, str(seq))
        if os.path.isdir(seq_dir):
            frames = [f for f in os.listdir(seq_dir) if f.endswith(".npy")]
            if len(frames) == SEQUENCE_LENGTH:
                count += 1
    return count

# ── Draw HUD ──────────────────────────────────────────────────────────────────
def draw_hud(frame, gesture, seq_idx, num_seq, frame_num, seq_len, state, hand_found):
    h, w = frame.shape[:2]

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Gesture name
    cv2.putText(frame, f"Gesture: {gesture}",
                (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 100), 2)

    # Sequence progress
    cv2.putText(frame, f"Sequence: {seq_idx + 1} / {num_seq}",
                (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

    # State message
    state_colours = {
        "waiting":    (200, 200, 200),
        "countdown":  (255, 165,   0),
        "recording":  (100, 255, 100),
        "done":       (100, 255, 100),
    }
    cv2.putText(frame, state,
                (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                state_colours.get(state, (255,255,255)), 2)

    # Frame progress bar (only during recording)
    if frame_num > 0:
        bx, by, bw, bh = 15, 115, w - 30, 10
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        fill = int(bw * (frame_num / seq_len))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (50, 220, 100), -1)

    # Hand detection dot
    dot_col  = (50, 220, 50) if hand_found else (50, 50, 220)
    dot_text = "Hand: YES" if hand_found else "Hand: NO"
    cv2.circle(frame, (w - 22, 22), 11, dot_col, -1)
    cv2.putText(frame, dot_text, (w - 115, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, dot_col, 1)

    # Bottom hint
    cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "SPACE=Start next   Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    return frame

# ── Main collection function ──────────────────────────────────────────────────
def collect_sequences(gesture=None, num_sequences=NUM_SEQUENCES, camera_idx=0):
    """
    Collects gesture sequences for CNN+LSTM training.

    For each sequence:
      1. Shows a 3-second countdown
      2. Records exactly 30 frames
      3. Saves each frame as a .npy file

    Output:
      Sign_Data/sequences/A/0/0.npy ... 29.npy
      Sign_Data/sequences/A/1/0.npy ... 29.npy
      ...
    """

    gestures = [gesture] if gesture else ALL_GESTURES
    create_folders(gestures, num_sequences)

    # Open camera
    cap = open_camera(camera_idx)
    if cap is None:
        print(f"Cannot open camera {camera_idx}. Try --camera 1")
        return

    # Load hand detector
    print("\nLoading hand detector...")
    landmarker = create_landmarker()
    print("Hand detector ready.")

    print("\n" + "="*50)
    print(f"  Collecting sequences for: {gestures}")
    print(f"  Sequences per gesture   : {num_sequences}")
    print(f"  Frames per sequence     : {SEQUENCE_LENGTH}")
    print("="*50)
    print("\n  Press SPACE to start each sequence")
    print("  Press Q to quit and save progress\n")

    try:
        for gesture_name in gestures:
            existing = count_existing(gesture_name, num_sequences)
            print(f"\nGesture: {gesture_name}  "
                  f"(existing: {existing}/{num_sequences})")

            for seq_idx in range(num_sequences):

                # Skip already collected sequences
                seq_dir    = os.path.join(DATA_PATH, gesture_name, str(seq_idx))
                npy_files  = [f for f in os.listdir(seq_dir) if f.endswith(".npy")]
                if len(npy_files) == SEQUENCE_LENGTH:
                    continue

                # ── Wait for SPACE ────────────────────────────────────────────
                waiting = True
                while waiting:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)

                    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result   = landmarker.detect(mp_image)

                    frame = draw_skeleton(frame, result)
                    frame = draw_hud(
                        frame, gesture_name, seq_idx, num_sequences,
                        0, SEQUENCE_LENGTH,
                        f"Press SPACE — Sequence {seq_idx+1}/{num_sequences}",
                        len(result.hand_landmarks) > 0,
                    )
                    cv2.imshow("Sequence Collector", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        waiting = False
                    elif key in (ord("q"), 27):
                        print("\nQuit. Progress saved.")
                        cap.release()
                        landmarker.close()
                        cv2.destroyAllWindows()
                        return

                # ── Countdown 3-2-1 ──────────────────────────────────────────
                for count in range(3, 0, -1):
                    deadline = time.time() + 1.0
                    while time.time() < deadline:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        frame = cv2.flip(frame, 1)

                        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result   = landmarker.detect(mp_image)

                        frame = draw_skeleton(frame, result)
                        frame = draw_hud(
                            frame, gesture_name, seq_idx, num_sequences,
                            0, SEQUENCE_LENGTH,
                            f"Starting in {count}...",
                            len(result.hand_landmarks) > 0,
                        )
                        cv2.imshow("Sequence Collector", frame)
                        cv2.waitKey(1)

                # ── Record 30 frames ──────────────────────────────────────────
                print(f"  Recording sequence {seq_idx+1}/{num_sequences}...",
                      end=" ", flush=True)

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)

                    frame = cv2.flip(frame, 1)

                    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result   = landmarker.detect(mp_image)

                    # Extract and save keypoints
                    keypoints = extract_keypoints(result)
                    npy_path  = os.path.join(
                        DATA_PATH, gesture_name, str(seq_idx), str(frame_num)
                    )
                    np.save(npy_path, keypoints)

                    # Show frame
                    frame = draw_skeleton(frame, result)
                    frame = draw_hud(
                        frame, gesture_name, seq_idx, num_sequences,
                        frame_num + 1, SEQUENCE_LENGTH,
                        f"Recording frame {frame_num+1}/{SEQUENCE_LENGTH}",
                        len(result.hand_landmarks) > 0,
                    )
                    cv2.imshow("Sequence Collector", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\nQuit mid-sequence.")
                        cap.release()
                        landmarker.close()
                        cv2.destroyAllWindows()
                        return

                print("done")

            print(f"Gesture '{gesture_name}' complete!")

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("  All sequences collected!")
    print(f"  Saved to: {os.path.abspath(DATA_PATH)}")
    print("="*50)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture",   type=str, default=None,
                        help="Single gesture to collect e.g. A")
    parser.add_argument("--sequences", type=int, default=NUM_SEQUENCES,
                        help="Number of sequences per gesture (default 100)")
    parser.add_argument("--camera",    type=int, default=None,
                        help="Camera index (default: auto-detect)")
    args = parser.parse_args()

    # Find camera
    cam = args.camera
    if cam is None:
        print("Scanning for cameras...")
        cam = find_camera()
        if cam is None:
            print("No camera found. Try: python sequence_data_collector.py --camera 1")
            sys.exit(1)

    collect_sequences(
        gesture      = args.gesture,
        num_sequences = args.sequences,
        camera_idx   = cam,
    )