"""
Train Model Page
=================
Allows triggering the training pipeline from the Streamlit UI.
Shows live logs and training metrics.
"""

import streamlit as st
import os, sys, time, subprocess

st.set_page_config(page_title="Train Model", page_icon="🏋️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700&family=Space+Grotesk&display=swap');
  h1,h2,h3{font-family:'Syne',sans-serif!important;}
  body,p,div{font-family:'Space Grotesk',sans-serif;}
  .log-box {
      background: #0d1117;
      color: #39d353;
      border-radius: 10px;
      padding: 1rem;
      font-family: 'Courier New', monospace;
      font-size: 0.8rem;
      height: 400px;
      overflow-y: auto;
      border: 1px solid rgba(255,255,255,0.1);
  }
</style>
""", unsafe_allow_html=True)

st.title("🏋️ Model Training Dashboard")

tab1, tab2 = st.tabs(["🧠 Deep Learning (CNN+LSTM)", "📊 Classical ML Pipeline"])

with tab1:
    st.markdown("""
    **CNN+LSTM Deep Learning Training**

    Prerequisites:
    1. Collect sequence data with `sequence_data_collector.py` (min 100 sequences/class)
    2. Click **Start Training** below

    The model will train for up to 100 epochs with early stopping.
    """)

    col1, col2, col3 = st.columns(3)
    epochs     = col1.slider("Max Epochs", 20, 200, 100, 10)
    batch_size = col2.selectbox("Batch Size", [16, 32, 64], index=1)
    model_type = col3.selectbox("Model Type", ["cnn_lstm", "transformer"])

    if st.button("▶ Start Deep Training", type="primary"):
        st.info("Training started… This may take several minutes.")
        log_box = st.empty()

        try:
            from AI_signlanguage.components.deep_learning.deep_model_trainer import (
                train_deep_model,
            )
            logs = []
            # Redirect stdout/stderr to capture logs
            import io, contextlib
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model, history, acc = train_deep_model(
                    epochs=epochs, batch_size=batch_size, model_type=model_type
                )
            logs.append(f.getvalue())
            log_box.markdown(
                f'<div class="log-box">{"<br>".join(logs)}</div>',
                unsafe_allow_html=True,
            )
            st.success(f"✅ Training complete! Test Accuracy: **{acc*100:.2f}%**")

            # Show training curve if saved
            curve_path = os.path.join("models", "cnn_lstm", "training_history.png")
            if os.path.exists(curve_path):
                st.image(curve_path, caption="Training History", use_column_width=True)

            cm_path = os.path.join("models", "cnn_lstm", "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix", use_column_width=True)

        except Exception as e:
            st.error(f"Training failed: {e}")

with tab2:
    st.markdown("""
    **Classical ML Pipeline (sklearn)**
    Runs the full ingestion → validation → transformation → GridSearchCV training.

    Prerequisites: Data must be available in MongoDB (run `push_data.py` first).
    """)

    if st.button("▶ Start Classical Training", type="primary"):
        try:
            from signlanguage.pipeline.training_pipeline import TrainingPipeline
            with st.spinner("Running full training pipeline…"):
                pipeline = TrainingPipeline()
                artifact = pipeline.run_pipeline()
            st.success("✅ Classical ML pipeline completed!")
            st.json({
                "trained_model": artifact.trained_model_file_path,
                "train_f1": round(artifact.train_metric_artifact.f1_score, 4),
                "test_f1":  round(artifact.test_metric_artifact.f1_score, 4),
                "train_acc": round(artifact.train_metric_artifact.accuracy, 4),
                "test_acc":  round(artifact.test_metric_artifact.accuracy, 4),
            })
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

    st.markdown("---")
    st.markdown("#### 📁 Model Artifacts")
    final_model_dir = "final_model"
    if os.path.isdir(final_model_dir):
        for fname in os.listdir(final_model_dir):
            fpath = os.path.join(final_model_dir, fname)
            size  = os.path.getsize(fpath) / 1e6
            st.markdown(f"- `{fname}` — {size:.2f} MB")
    else:
        st.info("No models trained yet.")
