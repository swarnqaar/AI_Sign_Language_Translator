"""
live_translate.py  —  Standalone CLI real-time translator
==========================================================
Launches the webcam translation loop directly without Streamlit.

Usage:
    python live_translate.py
    python live_translate.py --confidence 0.7
    python live_translate.py --tts gtts --lang hi    # Hindi speech output
"""

import argparse
from AI_signlanguage.components.deep_learning.realtime_inference import RealTimeInference
from AI_signlanguage.logging.logger import logging


def main():
    parser = argparse.ArgumentParser(description="AI Sign Language Real-Time Translator")
    parser.add_argument("--model",      default="final_model/deep_model.keras")
    parser.add_argument("--encoder",    default="final_model/label_encoder_dl.pkl")
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--tts",        default="pyttsx3", choices=["pyttsx3", "gtts"])
    parser.add_argument("--lang",       default="en", help="Language code for gTTS")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  🤟  AI Sign Language Translator  —  Live Mode")
    print("="*55)
    print(f"  Model       : {args.model}")
    print(f"  Confidence  : {args.confidence*100:.0f}%")
    print(f"  TTS Engine  : {args.tts} ({args.lang})")
    print("="*55)
    print("  SPACE=clear sentence | ENTER=speak | Q=quit")
    print("="*55 + "\n")

    engine = RealTimeInference(
        model_path=args.model,
        label_encoder_path=args.encoder,
        confidence_threshold=args.confidence,
    )
    engine.run()


if __name__ == "__main__":
    main()
