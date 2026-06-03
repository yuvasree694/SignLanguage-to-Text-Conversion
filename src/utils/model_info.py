"""Read input size and class labels from a saved Keras model."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tensorflow import keras

from src.utils.config_loader import resolve_path

logger = logging.getLogger("asl_system")


def get_model_input_size(model: keras.Model) -> Tuple[int, int]:
    """Return (width, height) the model expects for cv2.resize."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    # Keras: (batch, height, width, channels)
    height, width = shape[1], shape[2]
    return (width, height)


def get_model_num_classes(model: keras.Model) -> int:
    shape = model.output_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[-1])


def load_class_labels(model: keras.Model, config: Dict[str, Any], model_path: str) -> List[str]:
    """Match label list to model output size (metadata, config, or truncated)."""
    n = get_model_num_classes(model)
    config_labels = config["classes"]["all"]

    meta_paths = [resolve_path("data/processed/metadata.json")]
    for meta_path in meta_paths:
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            labels = meta.get("classes", [])
            if len(labels) == n:
                logger.info("Using %d class labels from %s", n, meta_path)
                return labels

    if len(config_labels) == n:
        return config_labels

    if len(config_labels) > n:
        logger.warning(
            "Model has %d classes but config has %d — using first %d labels. Retrain recommended.",
            n, len(config_labels), n,
        )
        return config_labels[:n]

    raise ValueError(
        f"Model expects {n} classes but config only has {len(config_labels)}. "
        "Re-run prepare_dataset.py and train_model.py."
    )
