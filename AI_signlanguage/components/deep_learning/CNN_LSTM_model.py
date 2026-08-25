"""
CNN + LSTM Model Architecture
==============================
This is the RECOMMENDED model for sign language recognition.

WHY CNN + LSTM?
  - Sign language involves MOTION (temporal sequences of frames)
  - CNN extracts spatial features from each frame (finger positions)
  - LSTM learns the temporal pattern ACROSS frames (how gesture unfolds)
  - Together they capture both shape AND movement

Architecture Flow:
  Input: (batch, sequence_len=30, 126 landmarks)
         ↓
  TimeDistributed(Dense → BatchNorm → Dropout)   ← acts like CNN on each frame
         ↓
  LSTM(128, return_sequences=True)               ← learns temporal patterns
         ↓
  LSTM(64, return_sequences=False)               ← compresses to fixed vector
         ↓
  Dense(128, ReLU) + Dropout(0.4)
         ↓
  Dense(num_classes, Softmax)                    ← gesture prediction

For pure static gestures (images only):
  Use the CNN branch (build_cnn_model below).

For dynamic gestures / full sign language sentences:
  Use the CNN+LSTM branch (build_cnn_lstm_model below).
"""

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization,
    Conv2D, MaxPooling2D, Flatten, TimeDistributed,
    GlobalAveragePooling2D, Input, Bidirectional,
    Conv1D, MaxPooling1D, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2

from AI_signlanguage.exception.exception import SignLanguageException
from AI_signlanguage.logging.logger import logging
from AI_signlanguage.constant.training_pipeline import NUM_CLASSES, FEATURE_SIZE, SIGN_CLASSES


# -----------------------------------------------------------------------
# 1.  CNN + LSTM  — RECOMMENDED for dynamic / video gestures
# -----------------------------------------------------------------------
def build_cnn_lstm_model(
    sequence_len: int = 30,
    feature_size: int = FEATURE_SIZE,   # 126
    num_classes: int = NUM_CLASSES,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Input shape: (batch, sequence_len=30, feature_size=126)
      - sequence_len = number of consecutive frames per sample
      - feature_size = 21 landmarks × 3 coords × 2 hands

    The TimeDistributed wrapper applies the same Dense layers to EVERY
    frame independently before feeding the sequence into the LSTMs.
    This mimics a lightweight CNN feature extractor per frame.
    """
    model = Sequential(name="CNN_LSTM_SignLanguage")

    # ── Frame-level feature extraction (TimeDistributed acts like CNN) ──
    model.add(Input(shape=(sequence_len, feature_size)))
    model.add(TimeDistributed(Dense(256, activation="relu"), name="td_dense_1"))
    model.add(TimeDistributed(BatchNormalization(), name="td_bn_1"))
    model.add(TimeDistributed(Dropout(0.3), name="td_drop_1"))

    model.add(TimeDistributed(Dense(128, activation="relu"), name="td_dense_2"))
    model.add(TimeDistributed(BatchNormalization(), name="td_bn_2"))
    model.add(TimeDistributed(Dropout(0.3), name="td_drop_2"))

    # ── Temporal learning with stacked Bidirectional LSTMs ──
    model.add(Bidirectional(
        LSTM(lstm_units_1, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    ))
    model.add(BatchNormalization(name="bn_lstm_1"))

    model.add(Bidirectional(
        LSTM(lstm_units_2, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2"
    ))
    model.add(BatchNormalization(name="bn_lstm_2"))

    # ── Classification head ──
    model.add(Dense(dense_units, activation="relu",
                    kernel_regularizer=l2(1e-4), name="dense_1"))
    model.add(Dropout(dropout_rate, name="dropout_final"))
    model.add(Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logging.info(f"CNN+LSTM model built. Parameters: {model.count_params():,}")
    model.summary()
    return model


# -----------------------------------------------------------------------
# 2.  Pure CNN — for static image-based gestures
# -----------------------------------------------------------------------
def build_cnn_model(
    input_shape: tuple = (64, 64, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Classic ConvNet for classifying single gesture images.
    Input: (batch, H=64, W=64, C=3) RGB image.

    Architecture:
      Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → MaxPool
      → Flatten → Dense(256) → Dropout → Softmax
    """
    model = Sequential(name="CNN_SignLanguage")

    # Block 1
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                     input_shape=input_shape, name="conv1"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", name="conv4"))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())

    # Classification head
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logging.info(f"CNN model built. Parameters: {model.count_params():,}")
    return model


# -----------------------------------------------------------------------
# 3.  Transformer-based model — Advanced / state-of-the-art
# -----------------------------------------------------------------------
def build_transformer_model(
    sequence_len: int = 30,
    feature_size: int = FEATURE_SIZE,
    num_classes: int = NUM_CLASSES,
    d_model: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_transformer_blocks: int = 2,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Self-attention transformer encoder for gesture sequence classification.
    Captures long-range dependencies between hand positions across frames.
    """
    inputs = Input(shape=(sequence_len, feature_size), name="input")

    # Project to d_model dimensions
    x = Dense(d_model, activation="relu", name="input_projection")(inputs)

    # Transformer encoder blocks
    for i in range(num_transformer_blocks):
        # Multi-head self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            name=f"mha_{i}"
        )(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(
            x + attn_output
        )

        # Feed-forward
        ff = Dense(ff_dim, activation="relu", name=f"ff1_{i}")(x)
        ff = Dense(d_model, name=f"ff2_{i}")(ff)
        ff = Dropout(dropout_rate)(ff)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(
            x + ff
        )

    # Global average pooling + classifier
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="Transformer_SignLanguage")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logging.info(f"Transformer model built. Parameters: {model.count_params():,}")
    return model


# -----------------------------------------------------------------------
# 4.  Training helpers
# -----------------------------------------------------------------------
def get_callbacks(model_save_path: str, log_dir: str = "logs/fit") -> list:
    """
    Standard callback suite:
      - EarlyStopping: stop if val_loss doesn't improve for 15 epochs
      - ModelCheckpoint: save best weights
      - ReduceLROnPlateau: halve LR when val_loss plateaus
      - TensorBoard: training curves
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]
