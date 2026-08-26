"""
Architecture Explanation Page
"""

import streamlit as st

st.set_page_config(page_title="Architecture", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Grotesk&display=swap');
  h1,h2,h3{font-family:'Syne',sans-serif!important;}
  body,p,div{font-family:'Space Grotesk',sans-serif;}
  .arch-box {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 14px;
      padding: 1.5rem 2rem;
      margin-bottom: 1.5rem;
  }
  .step-num {
      display: inline-block;
      background: linear-gradient(135deg,#7c3aed,#2563eb);
      color:white;
      width:32px;height:32px;
      border-radius:50%;
      text-align:center;
      line-height:32px;
      font-weight:700;
      margin-right:10px;
  }
  table { width:100%; border-collapse:collapse; }
  th { background:rgba(124,58,237,0.2); padding:8px 12px; text-align:left; border-radius:6px; }
  td { padding:8px 12px; border-bottom:1px solid rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ System Architecture")
st.caption("End-to-end technical explanation of the AI Sign Language Translator")

# ── Data Flow ────────────────────────────────────────────────────────────────
st.markdown("## 📊 Data Flow")

st.markdown("""
<div class="arch-box">
<pre style="color:#a5f3fc;font-size:0.85rem;line-height:1.8">
Webcam Frame (1280×720 RGB)
        │
        ▼
MediaPipe Hands ──────────────────────────────────────────────────────────
  • Detects up to 2 hands                                                  │
  • Returns 21 landmarks per hand: (x, y, z) in normalised image coords   │
  • Left hand → indices  0–62  (21 × 3 = 63 values)                      │
  • Right hand → indices 63–125 (21 × 3 = 63 values)                     │
  • Missing hand → zero-padded                                             │
        │                                                                  │
        ▼                                                                  │
Rolling Buffer: deque(maxlen=30)  ← keeps last 30 frames                 │
        │                                                                  │
        ▼ (when buffer is full)                                           │
CNN+LSTM Model                                                             │
  Input:  (1, 30, 126)  — batch=1, seq_len=30, features=126              │
  ├─ TimeDistributed(Dense 256 → BN → Drop 0.3)   per-frame encoding     │
  ├─ TimeDistributed(Dense 128 → BN → Drop 0.3)                          │
  ├─ BiLSTM(128, return_seq=True) + BN            temporal learning       │
  ├─ BiLSTM(64,  return_seq=False) + BN                                   │
  ├─ Dense(128, ReLU) + Dropout(0.4)                                      │
  └─ Dense(36, Softmax)           ← 36 gesture classes                    │
        │                                                                  │
        ▼                                                                  │
Softmax probabilities (36 values, sum=1.0)                                │
        │                                                                  │
        ▼ (if max_prob ≥ 0.60 threshold)                                  │
Majority Vote Smoother (last 5 predictions, need ≥3 same)                │
        │                                                                  │
        ▼                                                                  │
Sentence Builder → "Hello my name is..."                                  │
        │                                                                  │
        ▼                                                                  │
pyttsx3 / gTTS Speech Output                                              │
</pre>
</div>
""", unsafe_allow_html=True)

# ── Model Comparison ─────────────────────────────────────────────────────────
st.markdown("## 🧠 Model Comparison")

st.markdown("""
| Model | Best For | Accuracy | Speed | Handles Motion |
|-------|----------|----------|-------|----------------|
| **CNN + LSTM** ⭐ | Dynamic gestures (all ASL) | ~90% | ~30fps | ✅ Yes |
| Transformer | Complex long sequences | ~88% | ~25fps | ✅ Yes |
| Pure CNN | Static image gestures | ~85% | ~60fps | ❌ No |
| Random Forest | Landmark CSV, no GPU | ~80% | ~1000fps | ❌ No |
| SVC | Landmark CSV, small data | ~78% | ~500fps | ❌ No |
""")

st.info("**Recommendation:** Use CNN+LSTM for production. Use Random Forest/SVC for rapid prototyping without GPU.")

# ── MediaPipe Landmarks ───────────────────────────────────────────────────────
st.markdown("## ✋ MediaPipe Hand Landmarks")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    MediaPipe returns **21 landmarks** per hand, each with **(x, y, z)**:
    - **0**: Wrist
    - **1–4**: Thumb (CMC → Tip)
    - **5–8**: Index finger
    - **9–12**: Middle finger
    - **13–16**: Ring finger
    - **17–20**: Pinky finger

    Coordinates are **normalised** to image size [0, 1].
    The **z** coordinate represents depth (distance from camera plane).
    """)
with col2:
    st.markdown("""
    ```
    Feature vector structure (126 dims):
    ┌─────────────────────┬─────────────────────┐
    │   Left Hand (0-62)  │  Right Hand (63-125) │
    ├──────┬──────┬───────┼──────┬──────┬────────┤
    │ lh_0_x│lh_0_y│lh_0_z│ rh_0_x│rh_0_y│rh_0_z │
    │ lh_1_x│lh_1_y│lh_1_z│ rh_1_x│rh_1_y│rh_1_z │
    │  ...  │  ...  │  ... │  ...  │  ...  │  ...  │
    │lh_20_x│lh_20_y│lh_20_z│rh_20_x│rh_20_y│rh_20_z│
    └───────┴───────┴──────┴───────┴───────┴───────┘
    ```
    """)

# ── Production Pipeline ───────────────────────────────────────────────────────
st.markdown("## 🚀 Production ML Pipeline")
st.markdown("""
<div class="arch-box">
<span class="step-num">1</span> <b>Data Ingestion</b> — MongoDB Atlas → Feature Store CSV → Stratified Train/Test Split<br><br>
<span class="step-num">2</span> <b>Data Validation</b> — Schema check (127 columns) + Kolmogorov-Smirnov drift detection<br><br>
<span class="step-num">3</span> <b>Data Transformation</b> — KNNImputer (k=5, distance-weighted) + StandardScaler → .npy arrays<br><br>
<span class="step-num">4</span> <b>Model Training</b> — GridSearchCV (RF, GBM, SVC, KNN, LR) + MLflow/DagsHub logging<br><br>
<span class="step-num">5</span> <b>S3 Sync</b> — Artifacts + final model pushed to AWS S3 bucket<br><br>
<span class="step-num">6</span> <b>CI/CD</b> — GitHub Actions → Docker → AWS ECR → EC2 self-hosted runner
</div>
""", unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("## 🛠️ Technology Stack")
st.markdown("""
| Component | Technology | Why |
|-----------|-----------|-----|
| Hand Detection | MediaPipe Hands | Real-time, accurate, free |
| Deep Learning | TensorFlow/Keras | CNN+LSTM, production-ready |
| Classical ML | scikit-learn | GridSearchCV, preprocessing |
| Data Store | MongoDB Atlas | Cloud-hosted, JSON documents |
| Experiment Tracking | MLflow + DagsHub | Metric logging, model registry |
| API | FastAPI | Async, auto-docs, type-safe |
| UI | Streamlit | Rapid ML app development |
| Speech | pyttsx3 + gTTS | Offline + online TTS |
| Containerisation | Docker | Reproducible deployments |
| Cloud Storage | AWS S3 | Artifact persistence |
| CI/CD | GitHub Actions + ECR | Automated build + deploy |
""")
