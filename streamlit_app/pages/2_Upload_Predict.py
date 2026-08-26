"""
Upload & Predict Page
======================
Accepts a CSV of pre-extracted MediaPipe landmark features
and returns gesture predictions with a downloadable CSV.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, pickle

st.set_page_config(page_title="Upload & Predict", page_icon="📂", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700&family=Space+Grotesk&display=swap');
  h1,h2,h3{font-family:'Syne',sans-serif!important;}
  body,p,div{font-family:'Space Grotesk',sans-serif;}
  .result-table td { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("📂 Upload CSV for Prediction")
st.markdown("""
Upload a CSV where each row is one gesture frame with **126 landmark features**
(`lh_0_x, lh_0_y, lh_0_z, ... rh_20_x, rh_20_y, rh_20_z`).
The model returns a `predicted_gesture` column.
""")

uploaded = st.file_uploader("Upload landmark CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(5), use_container_width=True)

    predict_btn = st.button("🔮 Run Prediction", type="primary")
    if predict_btn:
        try:
            # Use deep model if available, else sklearn estimator
            deep_model_path = os.path.join("final_model", "deep_model.keras")
            sklearn_model_path = os.path.join("final_model", "model.pkl")
            le_deep_path = os.path.join("final_model", "label_encoder_dl.pkl")
            le_sklearn_path = os.path.join("final_model", "label_encoder.pkl")
            preprocessor_path = os.path.join("final_model", "preprocessor.pkl")

            feat_df = df.drop(columns=["label"], errors="ignore")

            if os.path.exists(deep_model_path) and os.path.exists(le_deep_path):
                import tensorflow as tf
                model = tf.keras.models.load_model(deep_model_path)
                with open(le_deep_path, "rb") as f:
                    le = pickle.load(f)

                # For single-frame CSV, replicate frame 30 times to form a sequence
                X = feat_df.values.astype(np.float32)
                sequences = np.stack([X] * 30, axis=1)  # (N, 30, 126)
                probs = model.predict(sequences, verbose=0)
                y_pred = np.argmax(probs, axis=1)
                labels = le.inverse_transform(y_pred)
                confidences = np.max(probs, axis=1)

                df["predicted_gesture"] = labels
                df["confidence_%"] = (confidences * 100).round(1)
                st.success(f"✅ Predictions complete using CNN+LSTM model")

            elif os.path.exists(sklearn_model_path) and os.path.exists(preprocessor_path):
                with open(preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
                with open(sklearn_model_path, "rb") as f:
                    model = pickle.load(f)
                with open(le_sklearn_path, "rb") as f:
                    le = pickle.load(f)

                X_transformed = preprocessor.transform(feat_df)
                y_pred_encoded = model.predict(X_transformed)
                labels = le.inverse_transform(y_pred_encoded.astype(int))
                df["predicted_gesture"] = labels
                st.success("✅ Predictions complete using sklearn model")
            else:
                st.error("No trained model found. Please train the model first.")
                st.stop()

            st.dataframe(df, use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="sign_language_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
