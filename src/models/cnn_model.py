"""Configurable CNN architecture for ASL classification."""

from typing import Any, Dict, List, Optional

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.models.mobilenet_model import build_mobilenet_model


def build_model(
    config: Dict[str, Any],
    architecture_override: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Build model based on config model.type (mobilenet_v2 or cnn)."""
    model_type = config.get("model", {}).get("type", "cnn")
    if model_type == "mobilenet_v2":
        return build_mobilenet_model(config)
    return build_cnn_model(config, architecture_override)


def build_cnn_model(
    config: Dict[str, Any],
    architecture_override: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """
    Build CNN model from configuration.

    Architecture: Conv blocks (Conv2D + BatchNorm + MaxPool) -> Dense -> Softmax
    """
    model_cfg = architecture_override or config["model"]
    img_h = config["image"]["height"]
    img_w = config["image"]["width"]
    channels = config["image"]["channels"]
    num_classes = config["num_classes"]

    inputs = keras.Input(shape=(img_h, img_w, channels), name="input_image")
    x = inputs

    l2 = regularizers.l2(model_cfg.get("l2_regularization", 0.001))

    for i, conv_cfg in enumerate(model_cfg["conv_layers"]):
        x = layers.Conv2D(
            filters=conv_cfg["filters"],
            kernel_size=conv_cfg["kernel_size"],
            padding="same",
            kernel_regularizer=l2,
            name=f"conv_{i+1}",
        )(x)

        if conv_cfg.get("batch_norm", True):
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

        activation = conv_cfg.get("activation", "relu")
        if activation == "leaky_relu":
            x = layers.LeakyReLU(name=f"act_{i+1}")(x)
        else:
            x = layers.Activation(activation, name=f"act_{i+1}")(x)
        x = layers.MaxPooling2D(
            pool_size=model_cfg.get("pool_size", 2),
            name=f"pool_{i+1}",
        )(x)
        x = layers.Dropout(model_cfg.get("dropout", 0.3) * 0.5, name=f"dropout_conv_{i+1}")(x)

    x = layers.Flatten(name="flatten")(x)

    dense_units = model_cfg.get("dense_units_cnn", model_cfg.get("dense_units", [512, 256]))
    dropout_rate = model_cfg.get("dropout_cnn", model_cfg.get("dropout", 0.5))
    for j, units in enumerate(dense_units):
        x = layers.Dense(units, kernel_regularizer=l2, name=f"dense_{j+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{j+1}")(x)
        x = layers.Activation("relu", name=f"act_dense_{j+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_dense_{j+1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_cfg.get("name", "asl_cnn"))
    return model


def compile_model(
    model: keras.Model,
    config: Dict[str, Any],
    learning_rate: Optional[float] = None,
    optimizer_name: Optional[str] = None,
) -> keras.Model:
    """Compile model with optimizer and loss from config."""
    train_cfg = config["training"]
    lr = learning_rate or train_cfg["learning_rate"]
    opt_name = (optimizer_name or train_cfg["optimizer"]).lower()

    optimizers = {
        "adam": keras.optimizers.Adam(learning_rate=lr),
        "sgd": keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=lr),
        "adagrad": keras.optimizers.Adagrad(learning_rate=lr),
    }
    optimizer = optimizers.get(opt_name, keras.optimizers.Adam(learning_rate=lr))

    model.compile(
        optimizer=optimizer,
        loss=train_cfg["loss"],
        metrics=["accuracy"],
    )
    return model


def get_model_summary(model: keras.Model) -> str:
    """Return model summary as string."""
    lines: List[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)
