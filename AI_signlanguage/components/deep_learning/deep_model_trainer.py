"""
Deep Learning Model Trainer
=============================
Loads the sequence dataset from disk, builds the CNN+LSTM model,
trains it with callbacks, evaluates on the test split, and saves:
  - models/cnn_lstm/best_model.h5      ← best weights (val_accuracy)
  - models/cnn_lstm/final_model.keras  ← final saved model
  - final_model/deep_model.keras       ← copy for the API to load
  - final_model/label_encoder_dl.pkl   ← label mapping

Training split: 80% train / 20% test (stratified by gesture class).
"""

import os
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")          # headless — no display needed on server
import matplotlib.pyplot as plt
import seaborn as sns

from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.constant.training_pipeline import SIGN_CLASSES
from AI_signlanguage.components.deep_learning.CNN_LSTM_model import (
    build_cnn_lstm_model,
    get_callbacks,
)

DATA_PATH       = os.path.join("Sign_Data", "sequences")
SEQUENCE_LENGTH = 30
MODEL_SAVE_PATH = os.path.join("models", "cnn_lstm", "best_model.h5")
FINAL_MODEL_DIR = "final_model"


# -----------------------------------------------------------------------
# Dataset loader
# -----------------------------------------------------------------------
def load_sequence_dataset():
    """
    Walks Sign_Data/sequences/<gesture>/<seq_idx>/<frame>.npy
    and builds:
      X: (N, 30, 126) — sequences
      y: (N,)         — integer class labels
    """
    sequences, labels = [], []

    for gesture in SIGN_CLASSES:
        gesture_dir = os.path.join(DATA_PATH, gesture)
        if not os.path.isdir(gesture_dir):
            logging.warning(f"No data folder found for gesture: {gesture}")
            continue

        for seq_idx in sorted(os.listdir(gesture_dir)):
            seq_dir = os.path.join(gesture_dir, seq_idx)
            if not os.path.isdir(seq_dir):
                continue

            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(seq_dir, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    window.append(np.load(frame_path))
                else:
                    window.append(np.zeros(126))   # pad missing frames

            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(gesture)

    X = np.array(sequences, dtype=np.float32)  # (N, 30, 126)
    return X, np.array(labels)


# -----------------------------------------------------------------------
# Plotting helpers (saved to disk, not displayed)
# -----------------------------------------------------------------------
def plot_training_history(history, save_dir: str = "models/cnn_lstm"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],    label="Train Acc",  color="#4CAF50")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",   color="#2196F3")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", color="#F44336")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#FF9800")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    plt.close()
    logging.info("Training history plot saved.")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir: str = "models/cnn_lstm"):
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(12, len(class_names)), max(10, len(class_names) - 2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix — Sign Language CNN+LSTM")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    logging.info("Confusion matrix saved.")


# -----------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------
def train_deep_model(
    epochs: int = 100,
    batch_size: int = 32,
    sequence_len: int = SEQUENCE_LENGTH,
    model_type: str = "cnn_lstm",   # "cnn_lstm" | "transformer"
):
    """
    Full training workflow:
      1. Load sequences from disk
      2. Encode labels
      3. Train/test split (stratified)
      4. Build CNN+LSTM model
      5. Train with callbacks
      6. Evaluate and generate reports
      7. Save model + label encoder
    """
    try:
        logging.info("=== Starting Deep Learning Training ===")

        # 1. Load data
        X, y_raw = load_sequence_dataset()
        if len(X) == 0:
            raise ValueError(
                "No sequence data found. Run sequence_data_collector.py first."
            )
        logging.info(f"Dataset loaded: X={X.shape}, classes={np.unique(y_raw)}")

        # 2. Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        num_classes = len(le.classes_)
        logging.info(f"Classes ({num_classes}): {list(le.classes_)}")

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logging.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

        # 4. Build model
        if model_type == "transformer":
            from AI_signlanguage.components.deep_learning.CNN_LSTM_model import (
                build_transformer_model,
            )
            model = build_transformer_model(
                sequence_len=sequence_len,
                num_classes=num_classes,
            )
        else:
            model = build_cnn_lstm_model(
                sequence_len=sequence_len,
                num_classes=num_classes,
            )

        # 5. Train
        callbacks = get_callbacks(MODEL_SAVE_PATH)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # 6. Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        logging.info(f"\nClassification Report:\n{report}")
        print(report)

        plot_training_history(history)
        plot_confusion_matrix(y_test, y_pred, le.classes_)

        # 7. Save
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        final_keras = os.path.join(FINAL_MODEL_DIR, "deep_model.keras")
        model.save(final_keras)
        logging.info(f"Deep model saved: {final_keras}")

        le_path = os.path.join(FINAL_MODEL_DIR, "label_encoder_dl.pkl")
        with open(le_path, "wb") as f:
            pickle.dump(le, f)
        logging.info(f"Label encoder saved: {le_path}")

        return model, history, test_acc

    except Exception as e:
        raise SignLanguageException(e, sys)


if __name__ == "__main__":
    train_deep_model(epochs=100, batch_size=32)
