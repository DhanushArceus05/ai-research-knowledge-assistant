"""
Dataset loading and preprocessing utilities for the document classifier.
"""
from typing import Tuple, List
import csv

from app.core.logging import get_logger

logger = get_logger(__name__)

CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing",
]


def load_dataset(csv_path: str) -> Tuple[List[str], List[str]]:
    """Loads a labelled CSV dataset with columns: text,category."""
    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("category") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)

    if not texts:
        raise ValueError(f"No usable rows found in dataset: {csv_path}")

    logger.info("Loaded %d labelled samples from %s", len(texts), csv_path)
    return texts, labels
