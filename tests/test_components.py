"""
Unit tests for core components.
Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from AI_signlanguage.utils.speech_utils.sentence_builder import SentenceBuilder
from AI_signlanguage.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)


# ────────────────────────────────────────────────
# SentenceBuilder tests
# ────────────────────────────────────────────────
class TestSentenceBuilder:
    def test_add_gesture(self):
        sb = SentenceBuilder(cooldown_seconds=0)
        sb.add_gesture("HELLO")
        assert sb.tokens == ["HELLO"]

    def test_deduplication_within_cooldown(self):
        sb = SentenceBuilder(cooldown_seconds=999)
        sb.add_gesture("A")
        sb.add_gesture("A")   # same gesture within cooldown → suppressed
        assert len(sb.tokens) == 1

    def test_deduplication_after_cooldown(self):
        sb = SentenceBuilder(cooldown_seconds=0)
        sb.add_gesture("A")
        sb.add_gesture("A")   # cooldown=0 → both added
        assert len(sb.tokens) == 2

    def test_get_sentence_capitalisation(self):
        sb = SentenceBuilder(cooldown_seconds=0)
        sb.add_gesture("HELLO")
        sb.add_gesture("WORLD")
        sentence = sb.get_sentence()
        assert sentence[0].isupper()   # first letter capital

    def test_undo(self):
        sb = SentenceBuilder(cooldown_seconds=0)
        sb.add_gesture("A")
        sb.add_gesture("B")
        removed = sb.undo()
        assert removed == "B"
        assert sb.tokens == ["A"]

    def test_clear(self):
        sb = SentenceBuilder(cooldown_seconds=0)
        sb.add_gesture("A")
        sb.clear()
        assert sb.tokens == []

    def test_max_words(self):
        sb = SentenceBuilder(cooldown_seconds=0, max_words=3)
        for letter in ["A", "B", "C", "D"]:
            sb.add_gesture(letter)
        assert len(sb.tokens) == 3


# ────────────────────────────────────────────────
# Classification metric tests
# ────────────────────────────────────────────────
class TestClassificationMetric:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        metric = get_classification_score(y, y)
        assert metric.accuracy == pytest.approx(1.0)
        assert metric.f1_score == pytest.approx(1.0)

    def test_wrong_predictions(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        metric = get_classification_score(y_true, y_pred)
        assert metric.accuracy == pytest.approx(0.0)

    def test_metric_range(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])
        metric = get_classification_score(y_true, y_pred)
        assert 0.0 <= metric.accuracy <= 1.0
        assert 0.0 <= metric.f1_score <= 1.0
        assert 0.0 <= metric.precision_score <= 1.0
        assert 0.0 <= metric.recall_score <= 1.0


# ────────────────────────────────────────────────
# Model architecture tests (requires tensorflow)
# ────────────────────────────────────────────────
class TestModelArchitecture:
    def test_cnn_lstm_output_shape(self):
        try:
            from AI_signlanguage.components.deep_learning.CNN_LSTM_model import (
                build_cnn_lstm_model,
            )
            model = build_cnn_lstm_model(
                sequence_len=30, feature_size=126, num_classes=36
            )
            # Dummy batch of 2 sequences
            x = np.random.rand(2, 30, 126).astype(np.float32)
            preds = model.predict(x, verbose=0)
            assert preds.shape == (2, 36)
            assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5)  # softmax sums to 1
        except ImportError:
            pytest.skip("TensorFlow not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
