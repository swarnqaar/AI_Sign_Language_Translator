"""
Text-to-Speech Engine
======================
Provides cross-platform speech output with two backends:
  1. pyttsx3  — offline, no API key needed, works on Windows/macOS/Linux
  2. gTTS     — Google TTS, requires internet, better voice quality

Usage:
    from signlanguage.utils.speech_utils.tts_engine import speak_text
    speak_text("Hello, how are you?")
    speak_text("Hello", engine="gtts", lang="en")   # Google TTS
    speak_text("Namaste", engine="gtts", lang="hi") # Hindi
"""

import os
import sys
import tempfile
from AI_signlanguage.logging.logger import logging


def speak_text(text: str, engine: str = "pyttsx3", lang: str = "en"):
    """
    Speak the given text aloud.

    Parameters
    ----------
    text    : str   — text to speak
    engine  : str   — "pyttsx3" (offline) or "gtts" (Google, online)
    lang    : str   — language code for gTTS (e.g., "en", "hi", "bn")
    """
    if not text or not text.strip():
        return

    try:
        if engine == "pyttsx3":
            _speak_pyttsx3(text)
        elif engine == "gtts":
            _speak_gtts(text, lang)
        else:
            logging.warning(f"Unknown TTS engine '{engine}', falling back to pyttsx3.")
            _speak_pyttsx3(text)
    except Exception as e:
        logging.error(f"TTS error: {e}")


def _speak_pyttsx3(text: str):
    """
    Offline TTS via pyttsx3.
    Runs in blocking mode (waits until speech completes).
    For non-blocking, run in a thread.
    """
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)    # words per minute (150 = natural pace)
    engine.setProperty("volume", 0.9)  # 0.0–1.0

    # Prefer a female voice if available
    voices = engine.getProperty("voices")
    for voice in voices:
        if "female" in voice.name.lower() or "zira" in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break

    engine.say(text)
    engine.runAndWait()


def _speak_gtts(text: str, lang: str = "en"):
    """
    Online TTS via Google Text-to-Speech (gTTS).
    Saves to a temp MP3 and plays it with playsound or mpg123.
    """
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang, slow=False)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
        tts.save(tmp_path)

    # Play the audio file
    try:
        import playsound
        playsound.playsound(tmp_path)
    except ImportError:
        # Fallback: use system command
        if sys.platform == "win32":
            os.system(f'start /wait "" "{tmp_path}"')
        elif sys.platform == "darwin":
            os.system(f"afplay {tmp_path}")
        else:
            os.system(f"mpg123 -q {tmp_path}")

    os.unlink(tmp_path)


def text_to_audio_bytes(text: str, lang: str = "en") -> bytes:
    """
    Returns MP3 audio as bytes (useful for Streamlit st.audio).
    """
    from gtts import gTTS
    import io

    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
