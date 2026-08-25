"""
Real-Time Inference Engine
===========================
Runs the trained CNN+LSTM model on a LIVE webcam feed.

How it works:
  1. A rolling buffer of the last 30 frames is maintained.
  2. Every frame, MediaPipe extracts 126 landmark values.
  3. Once the buffer is full, the model predicts the gesture.
  4. Predictions are smoothed using a majority-vote window (last 5 preds).
  5. Detected gestures are appended to a sentence builder.
  6. pyttsx3 reads the sentence aloud.

Key design decisions:
  - Buffer size = 30 matches training sequence length exactly.
  - Confidence threshold (default 0.6) prevents junk predictions
    when no hand is visible or the gesture is ambiguous.
  - Sentence is cleared with SPACE key; spoken with ENTER key.
"""

import os
import sys
import collections
import time
import pickle
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional

import tensorflow as tf

from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.utils.speech_utils.tts_engine import speak_text

mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

SEQUENCE_LENGTH     = 30
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW    = 5       # majority vote over last N predictions
FINAL_MODEL_DIR     = "final_model"


def extract_keypoints(results) -> np.ndarray:
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.multi_hand_landmarks:
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
            ).flatten()
            if handedness.classification[0].label == "Left":
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])


def draw_overlay(
    frame, sentence: list, predictions: list,
    current_gesture: str, confidence: float,
    fps: float,
):
    """Draw a clean, informative HUD on the webcam frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Current gesture + confidence bar
    if current_gesture and confidence >= CONFIDENCE_THRESHOLD:
        bar_w = int(confidence * 250)
        cv2.rectangle(frame, (10, 90), (10 + bar_w, 110), (0, 220, 120), -1)
        cv2.rectangle(frame, (10, 90), (260, 110), (200, 200, 200), 1)
        cv2.putText(
            frame,
            f"{current_gesture}  ({confidence*100:.0f}%)",
            (10, 82), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 150), 2,
        )
    else:
        cv2.putText(
            frame, "Detecting...", (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
        )

    # FPS
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (w - 110, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
    )

    # Sentence at bottom
    sentence_str = " ".join(sentence[-8:])  # show last 8 words
    cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, f"Sentence: {sentence_str}",
        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 100), 2,
    )

    # Controls legend
    cv2.putText(
        frame,
        "SPACE=clear  ENTER=speak  BACKSPACE=undo  Q=quit",
        (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
    )


class RealTimeInference:
    """
    Manages the rolling buffer, model, label encoder, and main loop.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None,
        sequence_len: int = SEQUENCE_LENGTH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        model_path = model_path or os.path.join(FINAL_MODEL_DIR, "deep_model.keras")
        label_encoder_path = (
            label_encoder_path
            or os.path.join(FINAL_MODEL_DIR, "label_encoder_dl.pkl")
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Deep model not found at {model_path}. "
                "Run deep_model_trainer.py first."
            )

        self.model = tf.keras.models.load_model(model_path)
        with open(label_encoder_path, "rb") as f:
            self.le = pickle.load(f)

        self.sequence_len = sequence_len
        self.confidence_threshold = confidence_threshold
        self.sequence_buffer = collections.deque(maxlen=sequence_len)
        self.prediction_history = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.sentence = []
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.5   # seconds between adding same gesture

        logging.info("RealTimeInference engine initialised.")

    def predict_gesture(self) -> tuple[str, float]:
        """Run inference on the current buffer. Returns (gesture_label, confidence)."""
        if len(self.sequence_buffer) < self.sequence_len:
            return "", 0.0

        seq = np.expand_dims(
            np.array(self.sequence_buffer, dtype=np.float32), axis=0
        )  # (1, 30, 126)
        probs = self.model.predict(seq, verbose=0)[0]
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        label = self.le.inverse_transform([class_idx])[0]
        return label, confidence

    def get_smoothed_prediction(self, label: str, confidence: float) -> str:
        """Majority vote over the last SMOOTHING_WINDOW predictions."""
        if confidence >= self.confidence_threshold:
            self.prediction_history.append(label)
        if not self.prediction_history:
            return ""
        most_common = collections.Counter(self.prediction_history).most_common(1)[0]
        return most_common[0] if most_common[1] >= (SMOOTHING_WINDOW // 2 + 1) else ""

    def maybe_add_to_sentence(self, gesture: str):
        """Add gesture to sentence builder, respecting cooldown."""
        if not gesture:
            return
        now = time.time()
        if (
            gesture != self.last_gesture
            or (now - self.last_gesture_time) > self.gesture_cooldown
        ):
            self.sentence.append(gesture)
            self.last_gesture = gesture
            self.last_gesture_time = now

    def run(self):
        """Main webcam loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_counter = collections.deque(maxlen=30)
        current_gesture, current_conf = "", 0.0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        ) as hands:

            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Draw hand skeleton
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                            mp_draw_style.get_default_hand_landmarks_style(),
                            mp_draw_style.get_default_hand_connections_style(),
                        )

                # Update rolling buffer
                kp = extract_keypoints(results)
                self.sequence_buffer.append(kp)

                # Predict every frame (model is fast enough at ~30fps)
                label, conf = self.predict_gesture()
                smoothed = self.get_smoothed_prediction(label, conf)
                if smoothed:
                    current_gesture = smoothed
                    current_conf    = conf
                    self.maybe_add_to_sentence(smoothed)

                # HUD
                fps_counter.append(time.time() - t0)
                fps = 1.0 / (sum(fps_counter) / len(fps_counter))
                draw_overlay(
                    frame, self.sentence, list(self.prediction_history),
                    current_gesture, current_conf, fps,
                )

                cv2.imshow("AI Sign Language Translator", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:          # SPACE — clear sentence
                    self.sentence.clear()
                    current_gesture, current_conf = "", 0.0
                elif key == 13:          # ENTER — speak sentence
                    if self.sentence:
                        text = " ".join(self.sentence)
                        logging.info(f"Speaking: {text}")
                        speak_text(text)
                elif key == 8:           # BACKSPACE — undo last word
                    if self.sentence:
                        self.sentence.pop()

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Real-time inference session ended.")


if __name__ == "__main__":
    engine = RealTimeInference()
    engine.run()
