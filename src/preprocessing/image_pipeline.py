"""Unified image preprocessing for training and webcam inference."""

from typing import Optional, Tuple

import cv2
import numpy as np


def composite_on_black(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Place masked hand region on black background (matches ASL Alphabet dataset style)."""
    black = np.zeros_like(frame)
    mask_3ch = cv2.merge([mask, mask, mask])
    return np.where(mask_3ch > 0, frame, black)


def center_square_crop(image: np.ndarray, padding_ratio: float = 0.15) -> np.ndarray:
    """Crop to square around content with padding."""
    if image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        h, w = image.shape[:2]
        side = max(h, w)
        cy, cx = h // 2, w // 2
        half = side // 2
        y1 = max(0, cy - half)
        x1 = max(0, cx - half)
        return image[y1:y1 + side, x1:x1 + side]

    x, y, w, h = cv2.boundingRect(coords)
    pad = int(padding_ratio * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    cropped = image[y1:y2, x1:x2]

    ch, cw = cropped.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off:y_off + ch, x_off:x_off + cw] = cropped
    return square


def preprocess_for_recognition(
    image: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
    mask: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Standardize hand image for the model.

    Pipeline: black background -> square crop -> resize -> normalize.
    Used for both dataset images and webcam crops so training matches inference.
    """
    if mask is not None:
        image = composite_on_black(image, mask)
    elif _is_plain_background(image):
        pass
    else:
        image = _auto_black_background(image)

    image = center_square_crop(image)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    if normalize:
        return resized.astype(np.float32) / 255.0
    return resized


def _is_plain_background(image: np.ndarray, threshold: int = 40) -> bool:
    """Detect ASL Alphabet style images (hand on dark/black background)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def _auto_black_background(image: np.ndarray) -> np.ndarray:
    """Isolate hand-like region and place on black background."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return composite_on_black(image, mask)
