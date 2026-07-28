"""
Trains the TensorFlow document classifier and persists the model, vectorizer
config, and label encoder to the paths configured in .env / Settings.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --dataset data/dataset/document_classification_synthetic.csv --epochs 20
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ml.dataset import load_dataset
from app.ml.training import train_and_save_classifier

logger = get_logger(__name__)


def main():
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)

    parser = argparse.ArgumentParser(description="Train the document domain classifier.")
    parser.add_argument("--dataset", default="data/dataset/document_classification.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    settings.ensure_directories()

    logger.info("Loading dataset from %s", args.dataset)
    texts, labels = load_dataset(args.dataset)

    logger.info("Training classifier (epochs=%d, batch_size=%d)...", args.epochs, args.batch_size)
    result = train_and_save_classifier(
        texts=texts,
        labels=labels,
        model_path=settings.MODEL_PATH,
        vectorizer_path=settings.VECTORIZER_PATH,
        label_encoder_path=settings.LABEL_ENCODER_PATH,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n=== Training complete ===")
    print(f"Classes: {result['classes']}")
    print(f"Validation loss: {result['val_loss']:.4f}")
    print(f"Validation accuracy: {result['val_accuracy']:.4f}")
    print(f"Model saved to: {settings.MODEL_PATH}")
    print(f"Label encoder saved to: {settings.LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
