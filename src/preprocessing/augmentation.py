"""Image augmentation pipeline using Keras ImageDataGenerator."""

from typing import Any, Dict, Tuple

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_augmentation_generator(config: Dict[str, Any]) -> ImageDataGenerator:
    """Create ImageDataGenerator with augmentation parameters from config."""
    aug = config["augmentation"]
    return ImageDataGenerator(
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        zoom_range=aug["zoom_range"],
        horizontal_flip=aug["horizontal_flip"],
        brightness_range=tuple(aug["brightness_range"]),
        fill_mode=aug["fill_mode"],
        rescale=1.0 / 255.0 if config["preprocessing"]["normalize"] else None,
    )


def build_validation_generator(config: Dict[str, Any]) -> ImageDataGenerator:
    """Create generator for validation/test (no augmentation)."""
    rescale = 1.0 / 255.0 if config["preprocessing"]["normalize"] else None
    return ImageDataGenerator(rescale=rescale)


def augment_batch(
    generator: ImageDataGenerator,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentation to a batch of images."""
    gen = generator.flow(images, labels, batch_size=batch_size, shuffle=False)
    return next(gen)
