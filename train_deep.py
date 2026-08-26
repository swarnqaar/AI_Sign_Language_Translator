"""
train_deep.py  —  CLI entry-point for CNN+LSTM training
=======================================================
Usage:
    python train_deep.py                      # defaults: cnn_lstm, 100 epochs
    python train_deep.py --model transformer  # use transformer architecture
    python train_deep.py --epochs 50 --batch 64
"""

import argparse
from AI_signlanguage.components.deep_learning.deep_model_trainer import train_deep_model
from AI_signlanguage.logging.logger import logging


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN+LSTM sign language model"
    )
    parser.add_argument("--model", default="cnn_lstm",
                        choices=["cnn_lstm", "transformer"],
                        help="Model architecture to train")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=32)
    args = parser.parse_args()

    logging.info(
        f"Starting deep training: model={args.model}, "
        f"epochs={args.epochs}, batch={args.batch}"
    )
    model, history, acc = train_deep_model(
        epochs=args.epochs,
        batch_size=args.batch,
        model_type=args.model,
    )
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Test Accuracy : {acc*100:.2f}%")
    print(f"Model saved   : final_model/deep_model.keras")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
