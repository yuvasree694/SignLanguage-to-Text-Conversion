"""Model training with callbacks and augmentation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.models.cnn_model import build_cnn_model, build_model, compile_model, get_model_summary
from src.preprocessing.augmentation import build_augmentation_generator
from src.utils.config_loader import resolve_path

logger = logging.getLogger("asl_system")


class TrainingMetricsCallback(Callback):
    """Log epoch metrics to logger."""

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        logs = logs or {}
        logger.info(
            "Epoch %d - loss: %.4f, acc: %.4f, val_loss: %.4f, val_acc: %.4f",
            epoch + 1,
            logs.get("loss", 0),
            logs.get("accuracy", 0),
            logs.get("val_loss", 0),
            logs.get("val_accuracy", 0),
        )


class ASLTrainer:
    """Train CNN model with augmentation and callbacks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

    def build_and_compile(
        self,
        architecture_override: Optional[Dict[str, Any]] = None,
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
    ) -> keras.Model:
        if architecture_override:
            self.model = build_cnn_model(self.config, architecture_override)
        else:
            self.model = build_model(self.config)
        self.model = compile_model(self.model, self.config, learning_rate, optimizer)
        logger.info("Model built:\n%s", get_model_summary(self.model))
        return self.model

    def _build_callbacks(self, output_dir: Path) -> list:
        train_cfg = self.config["training"]
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=train_cfg["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=train_cfg["reduce_lr_factor"],
                patience=train_cfg["reduce_lr_patience"],
                min_lr=train_cfg["min_lr"],
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(output_dir / "best_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            CSVLogger(str(output_dir / "training_log.csv")),
            TensorBoard(log_dir=str(output_dir / "tensorboard")),
            TrainingMetricsCallback(),
        ]
        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        use_augmentation: bool = True,
    ) -> keras.callbacks.History:
        if self.model is None:
            self.build_and_compile()

        output_dir = resolve_path(self.config["paths"]["models"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        batch_size = self.config["dataset"]["batch_size"]
        epochs = self.config["training"]["epochs"]

        if use_augmentation:
            aug_gen = build_augmentation_generator(self.config)
            train_gen = aug_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
            steps = len(X_train) // batch_size
            history = self.model.fit(
                train_gen,
                steps_per_epoch=max(steps, 1),
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=self._build_callbacks(run_dir),
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=self._build_callbacks(run_dir),
            )

        self.history = history
        self.model.save(run_dir / "final_model.keras")

        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)

        logger.info("Training complete. Models saved to %s", run_dir)
        return history

    def load_model(self, model_path: str) -> keras.Model:
        path = resolve_path(model_path)
        self.model = keras.models.load_model(str(path))
        logger.info("Loaded model from %s", path)
        return self.model
