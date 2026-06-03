"""Model evaluation: metrics, confusion matrix, and visualization."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from src.utils.config_loader import resolve_path

logger = logging.getLogger("asl_system")


class ASLEvaluator:
    """Evaluate trained CNN on test set with comprehensive metrics."""

    def __init__(self, config: Dict[str, Any], class_labels: Optional[List[str]] = None):
        self.config = config
        self.class_labels = class_labels or config["classes"]["all"]

    def evaluate(
        self,
        model: keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Run evaluation and return metrics dictionary."""
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        report = classification_report(
            y_true, y_pred, target_names=self.class_labels, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        logger.info(
            "Test Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
        )
        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        figsize: Tuple[int, int] = (14, 12),
    ) -> None:
        """Save confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
        )
        plt.title("Confusion Matrix - ASL Classification")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        path = resolve_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Confusion matrix saved to %s", path)

    def plot_loss_curves(
        self,
        history: Dict[str, List[float]],
        output_path: str,
    ) -> None:
        """Plot training and validation loss/accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history["loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        acc_key = "accuracy" if "accuracy" in history else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
        axes[1].plot(history[acc_key], label="Train Accuracy")
        axes[1].plot(history[val_acc_key], label="Val Accuracy")
        axes[1].set_title("Accuracy Curves")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = resolve_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Loss curves saved to %s", path)

    def save_report(self, metrics: Dict[str, Any], output_path: str) -> None:
        """Save evaluation metrics to JSON."""
        path = resolve_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation report saved to %s", path)

    def full_evaluation(
        self,
        model: keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        history: Optional[Dict[str, List[float]]] = None,
        output_dir: str = "outputs/evaluation",
    ) -> Dict[str, Any]:
        """Run full evaluation pipeline with plots."""
        metrics = self.evaluate(model, X_test, y_test)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

        out = resolve_path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.plot_confusion_matrix(y_true, y_pred, str(out / "confusion_matrix.png"))
        if history:
            self.plot_loss_curves(history, str(out / "loss_curves.png"))
        self.save_report(metrics, str(out / "evaluation_report.json"))

        return metrics
