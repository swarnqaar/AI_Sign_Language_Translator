"""
AI Sign Language Translator — Streamlit Application
=====================================================
Multi-page Streamlit app with:
  Page 1 (Home):     Project overview, supported gestures
  Page 2 (Live):     Real-time webcam gesture recognition
  Page 3 (Upload):   Batch CSV landmark inference
  Page 4 (Train):    Trigger training pipeline from UI
  Page 5 (About):    Architecture explanation

Run:
    streamlit run streamlit_app/Home.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }

    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #a78bfa; }

    .gesture-pill {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167,139,250,0.4);
        color: #c4b5fd;
        border-radius: 100px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }

    .stat-box {
        text-align: center;
        padding: 1.5rem;
        background: rgba(96, 165, 250, 0.08);
        border-radius: 12px;
        border: 1px solid rgba(96,165,250,0.2);
    }
    .stat-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
        margin-top: 4px;
    }

    /* Hide Streamlit default header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Hero Section ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <div class="hero-title">🤟 AI Sign Language Translator</div>
  <div class="hero-subtitle">
    Real-time gesture recognition using MediaPipe + CNN+LSTM deep learning.<br>
    Translate ASL / ISL gestures to text and speech — bridging communication gaps.
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Stats Row ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="stat-box">
        <div class="stat-number">36</div>
        <div class="stat-label">Gesture Classes</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="stat-box">
        <div class="stat-number">126</div>
        <div class="stat-label">Landmark Features</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="stat-box">
        <div class="stat-number">30</div>
        <div class="stat-label">Frames / Sequence</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="stat-box">
        <div class="stat-number">~90%</div>
        <div class="stat-label">Target Accuracy</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Feature Cards ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="feature-card">
        <h3>🎥 Real-Time Recognition</h3>
        <p style="color:rgba(255,255,255,0.65)">
        Uses your webcam + MediaPipe Hands to extract 21 hand landmarks per frame.
        A rolling 30-frame buffer feeds the CNN+LSTM model for temporal gesture detection.
        Predictions are smoothed with majority voting to eliminate jitter.
        </p>
    </div>

    <div class="feature-card">
        <h3>🧠 CNN + LSTM Architecture</h3>
        <p style="color:rgba(255,255,255,0.65)">
        <b>Why CNN+LSTM?</b> Sign language involves motion —
        letters like J and Z require trajectory tracking, not just a static snapshot.
        The CNN encodes each frame's spatial landmark configuration;
        the LSTM learns how the hand moves across the sequence.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="feature-card">
        <h3>🔊 Speech Output</h3>
        <p style="color:rgba(255,255,255,0.65)">
        Detected gestures are assembled into a sentence.
        Press <b>Enter</b> to have the sentence read aloud via pyttsx3 (offline)
        or Google TTS (online, multi-language including Hindi/Bengali).
        </p>
    </div>

    <div class="feature-card">
        <h3>🚀 Production Pipeline</h3>
        <p style="color:rgba(255,255,255,0.65)">
        Full ML pipeline: MongoDB ingestion → KS drift validation →
        KNN imputation + StandardScaler → GridSearchCV model selection →
        MLflow experiment tracking → S3 artifact sync →
        Docker + AWS EC2 deployment via GitHub Actions CI/CD.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Supported Gestures ─────────────────────────────────────────────────────
st.markdown("### 🤙 Supported Gestures")

from AI_signlanguage.constant.training_pipeline import SIGN_CLASSES
pills = "".join(f'<span class="gesture-pill">{g}</span>' for g in SIGN_CLASSES)
st.markdown(f'<div style="margin-top:0.5rem">{pills}</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ─── Quick Start ────────────────────────────────────────────────────────────
st.markdown("### ⚡ Quick Start")
st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect gesture sequences (run for each class)
python -m signlanguage.components.deep_learning.sequence_data_collector

# 3. Train the CNN+LSTM model
python train_deep.py

# 4. Launch the Streamlit app
streamlit run streamlit_app/Home.py

# 5. Or start the FastAPI server
python app.py
""", language="bash")


# ─── Architecture Diagram ───────────────────────────────────────────────────
st.markdown("### 🏗️ System Architecture")
st.markdown("""
```
Webcam Feed
    │
    ▼
MediaPipe Hands ──► 21 landmarks × (x,y,z) × 2 hands = 126 features/frame
    │
    ▼
Rolling Buffer (30 frames)
    │
    ▼
CNN+LSTM Model
  ├─ TimeDistributed Dense ──► per-frame spatial encoding
  ├─ Bidirectional LSTM ×2  ──► temporal pattern learning
  └─ Dense + Softmax        ──► gesture class probabilities
    │
    ▼
Majority Vote Smoother (last 5 predictions)
    │
    ▼
Sentence Builder ──► "Hello my name is..."
    │
    ▼
Text Display + pyttsx3/gTTS Speech Output
```
""")
