"""
Live Translator Page
====================
Displays a real-time webcam feed with gesture detection.
Uses OpenCV + MediaPipe in a background thread; Streamlit renders the frames.

NOTE: Full webcam access requires running the app locally.
      In cloud deployments, use the /live API endpoint instead.
"""

import streamlit as st
import numpy as np
import threading
import time
import os
import sys

st.set_page_config(page_title="Live Translator", page_icon="🎥", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Grotesk&display=swap');
  h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
  body, p, div { font-family: 'Space Grotesk', sans-serif; }
  .pred-badge {
      display: inline-block;
      background: linear-gradient(90deg, #7c3aed, #2563eb);
      color: white;
      font-size: 2.5rem;
      font-weight: 800;
      padding: 0.4rem 1.5rem;
      border-radius: 14px;
      letter-spacing: 0.08em;
      margin-bottom: 0.5rem;
  }
  .conf-bar-wrap { background: rgba(255,255,255,0.1); border-radius: 100px; height: 10px; }
  .conf-bar { background: linear-gradient(90deg, #34d399, #60a5fa); border-radius: 100px; height: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Sign Language Translator")
st.caption("Point your webcam at your hand and start signing.")

# ── Session state ────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "current_gesture" not in st.session_state:
    st.session_state.current_gesture = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

# ── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    frame_placeholder = st.empty()

with right:
    st.markdown("#### 🔮 Current Detection")
    gesture_placeholder    = st.empty()
    confidence_placeholder = st.empty()

    st.markdown("---")
    st.markdown("#### 📝 Sentence Builder")
    sentence_placeholder = st.empty()

    st.markdown("---")
    speak_col, clear_col = st.columns(2)
    speak_btn = speak_col.button("🔊 Speak", use_container_width=True)
    clear_btn = clear_col.button("🗑️ Clear", use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⌨️ Controls")
    st.info("**Space** = Clear  |  **Enter** = Speak  |  **Backspace** = Undo")

    st.markdown("---")
    st.markdown("#### ℹ️ Model Info")
    model_path = os.path.join("final_model", "deep_model.keras")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        st.success(f"✅ Deep model loaded ({size_mb:.1f} MB)")
    else:
        st.warning("⚠️ No trained model found. Please train first.")

# ── Start / Stop ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
start_btn = col1.button("▶ Start Camera", type="primary",
                        disabled=st.session_state.running, use_container_width=True)
stop_btn  = col2.button("⏹ Stop Camera",
                        disabled=not st.session_state.running, use_container_width=True)

if start_btn:
    st.session_state.running = True
    st.rerun()
if stop_btn:
    st.session_state.running = False
    st.rerun()
if clear_btn:
    st.session_state.sentence = []
    st.rerun()
if speak_btn and st.session_state.sentence:
    from AI_signlanguage.utils.speech_utils.tts_engine import text_to_audio_bytes
    text = " ".join(st.session_state.sentence)
    audio_bytes = text_to_audio_bytes(text)
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)

# ── Live feed loop ────────────────────────────────────────────────────────────
if st.session_state.running:
    try:
        import cv2
        import mediapipe as mp
        import collections
        import pickle
        import tensorflow as tf

        mp_hands     = mp.solutions.hands
        mp_drawing   = mp.solutions.drawing_utils
        SEQUENCE_LEN = 30
        THRESHOLD    = 0.60

        model_path = os.path.join("final_model", "deep_model.keras")
        le_path    = os.path.join("final_model", "label_encoder_dl.pkl")

        if not os.path.exists(model_path):
            st.error("Model not found. Train the deep model first via the Train page.")
            st.stop()

        model = tf.keras.models.load_model(model_path)
        with open(le_path, "rb") as f:
            le = pickle.load(f)

        cap    = cv2.VideoCapture(0)
        buffer = collections.deque(maxlen=SEQUENCE_LEN)
        pred_history = collections.deque(maxlen=5)

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands.process(rgb)

                if res.multi_hand_landmarks:
                    for hl in res.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                lh = np.zeros(63); rh = np.zeros(63)
                if res.multi_hand_landmarks:
                    for hl, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                        c = np.array([[l.x, l.y, l.z] for l in hl.landmark]).flatten()
                        if hd.classification[0].label == "Left": lh = c
                        else: rh = c
                buffer.append(np.concatenate([lh, rh]))

                gesture, conf = "", 0.0
                if len(buffer) == SEQUENCE_LEN:
                    seq  = np.expand_dims(np.array(buffer, dtype=np.float32), 0)
                    prob = model.predict(seq, verbose=0)[0]
                    idx  = np.argmax(prob)
                    conf = float(prob[idx])
                    if conf >= THRESHOLD:
                        gesture = le.inverse_transform([idx])[0]
                        pred_history.append(gesture)
                        if len(pred_history) == 5:
                            from collections import Counter
                            common = Counter(pred_history).most_common(1)[0]
                            if common[1] >= 3:
                                st.session_state.current_gesture = common[0]
                                st.session_state.confidence = conf

                # Render frame
                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True,
                )

                # Update right panel
                g = st.session_state.current_gesture
                c = st.session_state.confidence
                gesture_placeholder.markdown(
                    f'<div class="pred-badge">{g or "—"}</div>', unsafe_allow_html=True
                )
                if c > 0:
                    bar_w = int(c * 100)
                    confidence_placeholder.markdown(
                        f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{bar_w}%"></div></div>'
                        f'<p style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:4px">{c*100:.1f}% confidence</p>',
                        unsafe_allow_html=True,
                    )

                sentence_placeholder.markdown(
                    f"**{' '.join(st.session_state.sentence[-10:]) or '(start signing)'}**"
                )

                time.sleep(0.03)   # ~30fps throttle

        cap.release()

    except ImportError as e:
        st.error(f"Missing dependency: {e}. Make sure mediapipe, opencv, tensorflow are installed.")
    except Exception as e:
        st.error(f"Camera error: {e}")
else:
    frame_placeholder.markdown("""
    <div style="height:400px;display:flex;align-items:center;justify-content:center;
    background:rgba(255,255,255,0.03);border-radius:16px;border:2px dashed rgba(255,255,255,0.15);">
      <div style="text-align:center;color:rgba(255,255,255,0.4)">
        <div style="font-size:3rem">📷</div>
        <div style="margin-top:1rem">Click <b>Start Camera</b> to begin</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
