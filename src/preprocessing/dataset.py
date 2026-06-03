"""Dataset loading, preparation, and splitting."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src.preprocessing.image_pipeline import preprocess_for_recognition
from src.utils.config_loader import resolve_path

logger = logging.getLogger("asl_system")

# Folders to skip when importing Kaggle ASL Alphabet
SKIP_FOLDERS = {"space", "nothing", "del", "delete", "nothing1", "nothing2", "nothing3", "nothing4"}


class ASLDataset:
    """Load and prepare ASL image dataset from directory structure."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.img_width = config["image"]["width"]
        self.img_height = config["image"]["height"]
        self.classes = config["classes"]["all"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        self.target_size = (self.img_width, self.img_height)
        self.normalize = config["preprocessing"]["normalize"]
        self.max_per_class = config["dataset"].get("max_samples_per_class")

    def load_from_directory(
        self,
        data_dir: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from folder-per-class structure.

        Expected layout:
            data/raw/A/, data/raw/B/, ...
        """
        data_path = resolve_path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_path}")

        images: List[np.ndarray] = []
        labels: List[str] = []

        for class_name in sorted(os.listdir(data_path)):
            class_dir = data_path / class_name
            if not class_dir.is_dir():
                continue

            label = class_name.upper()
            if label in SKIP_FOLDERS or label not in self.classes:
                continue

            files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            ]
            if self.max_per_class and len(files) > self.max_per_class:
                random.seed(self.config["dataset"]["random_seed"])
                files = random.sample(files, self.max_per_class)

            for img_file in files:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                processed = preprocess_for_recognition(
                    img,
                    target_size=self.target_size,
                    normalize=self.normalize,
                )
                images.append(processed)
                labels.append(label)

        if not images:
            raise ValueError(
                f"No images found in {data_path}. "
                "Download the ASL Alphabet dataset — see README."
            )

        X = np.array(images, dtype=np.float32)
        y = self.label_encoder.transform(labels)
        logger.info("Loaded %d images, %d classes", len(X), len(set(labels)))
        return X, y

    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split into train, validation, and test sets."""
        ds = self.config["dataset"]
        seed = ds["random_seed"]
        test_size = ds["test_split"]
        val_size = ds["val_split"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=seed, stratify=y_train
        )

        num_classes = self.config["num_classes"]
        return (
            X_train, X_val, X_test,
            to_categorical(y_train, num_classes),
            to_categorical(y_val, num_classes),
            to_categorical(y_test, num_classes),
        )

    def save_processed(
        self,
        output_dir: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        out = resolve_path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "X_train.npy", X_train)
        np.save(out / "X_val.npy", X_val)
        np.save(out / "X_test.npy", X_test)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "y_val.npy", y_val)
        np.save(out / "y_test.npy", y_test)

        metadata = {
            "classes": self.classes,
            "num_classes": self.config["num_classes"],
            "image_shape": [self.img_height, self.img_width, self.config["image"]["channels"]],
            "model_type": self.config["model"].get("type", "cnn"),
            "real_data": True,
        }
        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved processed dataset to %s", out)

    def load_processed(self, data_dir: str) -> Tuple[np.ndarray, ...]:
        path = resolve_path(data_dir)
        return (
            np.load(path / "X_train.npy"),
            np.load(path / "X_val.npy"),
            np.load(path / "X_test.npy"),
            np.load(path / "y_train.npy"),
            np.load(path / "y_val.npy"),
            np.load(path / "y_test.npy"),
        )

    @staticmethod
    def is_synthetic_dataset(data_dir: str) -> bool:
        """Detect demo synthetic data (colored blobs, not real hands)."""
        path = resolve_path(data_dir)
        for class_dir in path.iterdir():
            if not class_dir.is_dir():
                continue
            for img_file in list(class_dir.glob("*.jpg"))[:3]:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if float(np.std(hsv[:, :, 0])) > 40 and float(np.mean(img)) > 100:
                    return True
        return False
