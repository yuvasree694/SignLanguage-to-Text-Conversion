"""OpenCV-based hand detection and background removal."""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("asl_system")


class HandDetector:
    """Detect hand region using HSV skin segmentation and contour analysis."""

    def __init__(
        self,
        skin_lower: Tuple[int, int, int] = (0, 20, 70),
        skin_upper: Tuple[int, int, int] = (20, 255, 255),
        min_contour_area: int = 8000,
        blur_kernel: int = 5,
        max_bbox_area_ratio: float = 0.35,
        min_mask_fill_ratio: float = 0.40,
        reject_upper_frame_ratio: float = 0.30,
        min_hand_aspect_ratio: float = 0.45,
        max_hand_aspect_ratio: float = 2.2,
    ):
        self.skin_lower = np.array(skin_lower, dtype=np.uint8)
        self.skin_upper = np.array(skin_upper, dtype=np.uint8)
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.min_mask_fill_ratio = min_mask_fill_ratio
        self.reject_upper_frame_ratio = reject_upper_frame_ratio
        self.min_hand_aspect_ratio = min_hand_aspect_ratio
        self.max_hand_aspect_ratio = max_hand_aspect_ratio

    @classmethod
    def from_config(cls, prep_config: Dict[str, Any]) -> "HandDetector":
        """Build detector from preprocessing section of config."""
        return cls(
            skin_lower=tuple(prep_config["skin_lower_hsv"]),
            skin_upper=tuple(prep_config["skin_upper_hsv"]),
            min_contour_area=prep_config.get("min_contour_area", 8000),
            blur_kernel=prep_config.get("blur_kernel", 5),
            max_bbox_area_ratio=prep_config.get("max_bbox_area_ratio", 0.35),
            min_mask_fill_ratio=prep_config.get("min_mask_fill_ratio", 0.40),
            reject_upper_frame_ratio=prep_config.get("reject_upper_frame_ratio", 0.30),
            min_hand_aspect_ratio=prep_config.get("min_hand_aspect_ratio", 0.45),
            max_hand_aspect_ratio=prep_config.get("max_hand_aspect_ratio", 2.2),
        )

    def remove_background(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian blur and skin-color mask to isolate hand region."""
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result, mask

    def _is_valid_hand(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        contour_area: float,
    ) -> bool:
        """Reject face, background noise, and non-hand skin blobs."""
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        bbox_area = w * h

        if bbox_area / frame_area > self.max_bbox_area_ratio:
            return False

        center_y = y + h / 2
        if center_y < frame_h * self.reject_upper_frame_ratio:
            return False

        aspect = w / max(h, 1)
        if aspect < self.min_hand_aspect_ratio or aspect > self.max_hand_aspect_ratio:
            return False

        if contour_area < self.min_contour_area:
            return False

        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        roi_mask = mask[y:y2, x:x2]
        if roi_mask.size == 0:
            return False

        fill_ratio = np.count_nonzero(roi_mask) / roi_mask.size
        if fill_ratio < self.min_mask_fill_ratio:
            return False

        return True

    def detect_hand_region(
        self, frame: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], bool]:
        """
        Find largest valid hand contour and return cropped ROI.

        Returns:
            (cropped_hand, bbox, hand_detected)
        """
        if mask is None:
            _, mask = self.remove_background(frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, False

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:3]:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            padding = int(0.1 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            bbox = (x1, y1, x2 - x1, y2 - y1)

            if not self._is_valid_hand(frame, mask, bbox, area):
                continue

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            return cropped, bbox, True

        return None, None, False

    def preprocess_for_model(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """Full pipeline: background removal, hand crop, resize, normalize."""
        bg_removed, mask = self.remove_background(frame)
        hand_crop, _, detected = self.detect_hand_region(bg_removed, mask)

        if not detected or hand_crop is None:
            return None

        resized = cv2.resize(hand_crop, target_size, interpolation=cv2.INTER_AREA)
        if normalize:
            resized = resized.astype(np.float32) / 255.0
        return resized

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        label: str = "Hand",
    ) -> np.ndarray:
        """Draw hand bounding box on frame."""
        x, y, w, h = bbox
        output = frame.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        return output
