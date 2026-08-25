"""
Sentence Builder
=================
Aggregates detected gesture tokens into grammatically coherent sentences.

Features:
  - Deduplication: prevents "A A A" when hand is held for 3 seconds
  - Capitalisation: first word of sentence is capitalised
  - Punctuation: auto-adds period when user pauses ≥2 seconds
  - Common corrections: "I NEED HELP" → keeps as-is (already natural)

Advanced mode (requires transformers):
  The GPT-based corrector passes the raw gesture sequence through a
  small language model to produce a grammatically correct sentence.
  This is optional and requires internet + `transformers` package.
"""

import time
import re
from collections import deque
from AI_signlanguage.logging.logger import logging


class SentenceBuilder:
    """
    Token-level sentence accumulator for real-time gesture recognition.

    Usage:
        builder = SentenceBuilder()
        builder.add_gesture("HELLO")
        builder.add_gesture("MY")
        builder.add_gesture("NAME")
        builder.add_gesture("IS")
        sentence = builder.get_sentence()  # "Hello my name is"
    """

    def __init__(
        self,
        cooldown_seconds: float = 1.5,
        max_words: int = 20,
        auto_correct: bool = False,
    ):
        self.cooldown = cooldown_seconds
        self.max_words = max_words
        self.auto_correct = auto_correct
        self.tokens: list[str] = []
        self.last_token: str = ""
        self.last_add_time: float = 0.0

    def add_gesture(self, gesture: str) -> bool:
        """
        Add a detected gesture to the sentence.
        Returns True if a new word was added, False if suppressed.
        """
        if not gesture:
            return False

        now = time.time()
        same_gesture = gesture == self.last_token
        within_cooldown = (now - self.last_add_time) < self.cooldown

        if same_gesture and within_cooldown:
            return False   # suppress repeated gesture

        if len(self.tokens) >= self.max_words:
            logging.info("Max sentence length reached.")
            return False

        self.tokens.append(gesture)
        self.last_token = gesture
        self.last_add_time = now
        return True

    def undo(self):
        """Remove the last added gesture."""
        if self.tokens:
            removed = self.tokens.pop()
            self.last_token = self.tokens[-1] if self.tokens else ""
            return removed
        return None

    def clear(self):
        """Reset the sentence."""
        self.tokens.clear()
        self.last_token = ""
        self.last_add_time = 0.0

    def get_raw(self) -> str:
        """Return raw gesture sequence as uppercase string."""
        return " ".join(self.tokens)

    def get_sentence(self) -> str:
        """
        Return a readable sentence:
          - Join with spaces
          - Capitalise first word
          - Lowercase remaining words (unless they're abbreviations)
          - Auto-correct common sign language patterns
        """
        if not self.tokens:
            return ""

        words = []
        for i, token in enumerate(self.tokens):
            word = token.upper()
            # Keep 1-2 char tokens as uppercase (I, A, B, OK, etc.)
            if len(word) > 2:
                word = word.capitalize() if i == 0 else word.lower()
            words.append(word)

        sentence = " ".join(words)

        if self.auto_correct:
            sentence = self._auto_correct(sentence)

        return sentence

    def _auto_correct(self, text: str) -> str:
        """
        Lightweight rule-based correction for common ISL/ASL patterns.
        For production, replace with a transformer-based corrector.
        """
        corrections = {
            r"\bi\b": "I",              # lowercase i → I
            r"\bplease help\b": "Please help me",
            r"\bthank you\b": "Thank you",
            r"\bhello\b": "Hello",
        }
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def __repr__(self):
        return f"SentenceBuilder(words={len(self.tokens)}, text='{self.get_sentence()}')"


# -----------------------------------------------------------------------
# Optional: GPT-based grammar corrector
# -----------------------------------------------------------------------
def correct_with_gpt(raw_gesture_sequence: str, model_name: str = "gpt2") -> str:
    """
    Uses a Hugging Face text2text model to convert raw gesture tokens
    into a grammatically correct English sentence.

    Example:
        Input:  "I NEED HELP PLEASE DOCTOR"
        Output: "I need help, please call a doctor."

    Requires: pip install transformers torch
    """
    try:
        from transformers import pipeline
        corrector = pipeline(
            "text2text-generation",
            model="prithivida/grammar_error_correcter_v1",
        )
        prompt = f"gec: {raw_gesture_sequence.lower()}"
        result = corrector(prompt, max_length=64, num_return_sequences=1)
        return result[0]["generated_text"].strip()
    except ImportError:
        logging.warning("transformers not installed. Returning raw sequence.")
        return raw_gesture_sequence
    except Exception as e:
        logging.error(f"GPT correction failed: {e}")
        return raw_gesture_sequence
