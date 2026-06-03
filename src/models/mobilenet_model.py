"""Transfer learning model for accurate real-hand ASL recognition."""

from typing import Any, Dict, Optional

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def build_mobilenet_model(
    config: Dict[str, Any],
    fine_tune: bool = True,
) -> keras.Model:
    """MobileNetV2 backbone — much more accurate on real webcam gestures than a small CNN."""
    model_cfg = config["model"]
    img_h = config["image"]["height"]
    img_w = config["image"]["width"]
    num_classes = config["num_classes"]

    base = MobileNetV2(
        input_shape=(img_h, img_w, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(model_cfg.get("dense_units", [256])[0], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(model_cfg.get("dropout", 0.4))(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=base.input, outputs=outputs, name="asl_mobilenet_v2")

    if fine_tune:
        fine_tune_at = model_cfg.get("fine_tune_at", len(base.layers) - 30)
        for layer in base.layers[fine_tune_at:]:
            layer.trainable = True

    return model
