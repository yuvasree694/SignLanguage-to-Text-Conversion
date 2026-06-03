"""Real-time sign language recognition from webcam."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tensorflow import keras

from src.preprocessing.hand_detection import HandDetector
from src.preprocessing.image_pipeline import preprocess_for_recognition
from src.utils.model_info import get_model_input_size, load_class_labels
from src.utils.performance import PerformanceMonitor, PredictionSmoother

logger = logging.getLogger("asl_system")

NO_HAND_LABEL = "—"
DISPLAY_WAITING = "Show your hand"


class SentenceBuilder:
    """Build sentences from consecutive gesture detections."""

    def __init__(
        self,
        hold_frames: int = 12,
        separator: str = " ",
        confidence_threshold: float = 0.70,
    ):
        self.hold_frames = hold_frames
        self.separator = separator
        self.confidence_threshold = confidence_threshold
        self.sentence: List[str] = []
        self.current_gesture: Optional[str] = None
        self.hold_counter: int = 0
        self.last_added: Optional[str] = None

    def update(self, label: str, confidence: float) -> Optional[str]:
        if confidence < self.confidence_threshold:
            self.current_gesture = None
            self.hold_counter = 0
            return None

        if label == self.current_gesture:
            self.hold_counter += 1
        else:
            self.current_gesture = label
            self.hold_counter = 1

        if self.hold_counter >= self.hold_frames and label != self.last_added:
            self.sentence.append(label)
            self.last_added = label
            self.hold_counter = 0
            return label
        return None

    def get_sentence(self) -> str:
        return self.separator.join(self.sentence)

    def clear(self) -> None:
        self.sentence.clear()
        self.current_gesture = None
        self.hold_counter = 0
        self.last_added = None

    def backspace(self) -> None:
        if self.sentence:
            self.sentence.pop()
            self.last_added = self.sentence[-1] if self.sentence else None


class RealtimeRecognizer:
    """Real-time ASL gesture recognition from webcam feed."""

    def __init__(
        self,
        model: keras.Model,
        config: Dict[str, Any],
        class_labels: Optional[List[str]] = None,
        model_path: str = "",
    ):
        self.model = model
        self.config = config
        if class_labels is not None:
            self.class_labels = class_labels
        elif model_path:
            self.class_labels = load_class_labels(model, config, model_path)
        else:
            self.class_labels = config["classes"]["all"]

        self.hand_detector = HandDetector.from_config(config["preprocessing"])

        inf = config["inference"]
        self.confidence_threshold = inf["confidence_threshold"]
        self.min_probability_margin = inf.get("min_probability_margin", 0.12)
        self.smoother = PredictionSmoother(window_size=inf["prediction_smoothing"])
        self.sentence_builder = SentenceBuilder(
            hold_frames=inf["gesture_hold_frames"],
            separator=inf["sentence_separator"],
            confidence_threshold=self.confidence_threshold,
        )
        self.performance = PerformanceMonitor()

        # Use the loaded model's input size (e.g. 64x64 old vs 128x128 new)
        self.target_size = get_model_input_size(model)
        self.normalize = config["preprocessing"]["normalize"]
        logger.info("Model input size: %dx%d, classes: %d", self.target_size[0], self.target_size[1], len(self.class_labels))

    def _prediction_is_confident(self, probs: np.ndarray) -> Tuple[bool, float]:
        sorted_probs = np.sort(probs)[::-1]
        top_conf = float(sorted_probs[0])
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else top_conf
        return (
            top_conf >= self.confidence_threshold
            and margin >= self.min_probability_margin
        ), top_conf

    def predict_frame(
        self, frame: np.ndarray
    ) -> Tuple[str, float, Optional[np.ndarray], Optional[Tuple], bool]:
        start = time.perf_counter()

        bg_removed, mask = self.hand_detector.remove_background(frame)
        hand_crop, bbox, hand_detected = self.hand_detector.detect_hand_region(bg_removed, mask)

        if not hand_detected or hand_crop is None:
            latency = (time.perf_counter() - start) * 1000
            self.performance.record_inference(latency)
            return NO_HAND_LABEL, 0.0, None, None, False

        processed = preprocess_for_recognition(
            hand_crop,
            target_size=self.target_size,
            mask=None,
            normalize=False,
        )
        input_img = processed.astype(np.float32)
        if self.normalize:
            input_img /= 255.0

        batch = np.expand_dims(input_img, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.class_labels[idx]

        confident, confidence = self._prediction_is_confident(probs)
        if not confident:
            label = NO_HAND_LABEL
            confidence = 0.0

        preview = (processed if self.normalize else (processed / 255.0 * 255)).astype(np.uint8)
        latency = (time.perf_counter() - start) * 1000
        self.performance.record_inference(latency)

        return label, confidence, preview, bbox, True

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        label, confidence, hand_img, bbox, hand_detected = self.predict_frame(frame)

        display_label = DISPLAY_WAITING
        if hand_detected and label != NO_HAND_LABEL and confidence > 0:
            display_label = label

        result = {
            "raw_label": label,
            "raw_confidence": confidence,
            "display_label": display_label,
            "stable_label": None,
            "hand_detected": hand_detected,
            "sentence": self.sentence_builder.get_sentence(),
            "bbox": bbox if hand_detected and label != NO_HAND_LABEL else None,
            "hand_image": hand_img if hand_detected and label != NO_HAND_LABEL else None,
            "performance": self.performance.summary(),
        }

        if hand_detected and label != NO_HAND_LABEL and confidence >= self.confidence_threshold:
            stable = self.smoother.add(label, confidence)
            result["stable_label"] = stable
            if stable:
                result["display_label"] = stable
                added = self.sentence_builder.update(stable, confidence)
                result["word_added"] = added
                result["sentence"] = self.sentence_builder.get_sentence()
        else:
            self.smoother.reset()

        return result

    @staticmethod
    def load_model(model_path: str) -> keras.Model:
        return keras.models.load_model(model_path)
