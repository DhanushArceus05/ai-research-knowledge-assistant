"""
TensorFlow/Keras training pipeline for the document domain classifier.

Architecture: TextVectorization -> Embedding -> GlobalAveragePooling1D -> Dense(ReLU)
-> Dropout -> Dense(Softmax).
"""
from typing import List
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.core.logging import get_logger

logger = get_logger(__name__)

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 64


def build_model(num_classes: int, train_texts: List[str]):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    vectorize_layer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    vectorize_layer.adapt(train_texts)

    model = models.Sequential([
        layers.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, vectorize_layer


def train_and_save_classifier(
    texts: List[str],
    labels: List[str],
    model_path: str,
    vectorizer_path: str,
    label_encoder_path: str,
    epochs: int = 15,
    batch_size: int = 8,
) -> dict:
    """Trains the classifier end-to-end and persists model + label encoder to disk."""

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42,
        stratify=encoded_labels if len(set(encoded_labels)) > 1 else None,
    )

    model, vectorize_layer = build_model(num_classes, train_texts)

    import tensorflow as tf

    history = model.fit(
        tf.constant(train_texts, dtype=tf.string), np.array(train_labels),
        validation_data=(tf.constant(val_texts, dtype=tf.string), np.array(val_labels)),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    eval_results = model.evaluate(
        tf.constant(val_texts, dtype=tf.string), np.array(val_labels), verbose=0
    )

    model.save(model_path)

    # Persist the vectorizer's vocabulary + config so it can be rebuilt at inference time
    vectorizer_config = {
        "config": vectorize_layer.get_config(),
        "vocabulary": vectorize_layer.get_vocabulary(),
    }
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer_config, f)

    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    logger.info("Model trained and saved to %s", model_path)

    return {
        "num_classes": num_classes,
        "classes": list(label_encoder.classes_),
        "val_loss": float(eval_results[0]),
        "val_accuracy": float(eval_results[1]),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
