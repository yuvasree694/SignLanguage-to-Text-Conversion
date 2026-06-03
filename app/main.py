"""
Real-time ASL Sign Language Recognition Application.

Usage:
    python app/main.py [--model PATH] [--camera INDEX]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.realtime import RealtimeRecognizer
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import setup_logger


def draw_ui(
    frame: np.ndarray,
    result: dict,
    class_labels: list,
) -> np.ndarray:
    """Draw prediction overlay on webcam frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    display_label = result.get("display_label", "Show your hand")
    confidence = result.get("raw_confidence", 0.0)
    sentence = result.get("sentence", "")
    hand_detected = result.get("hand_detected", False)

    if not hand_detected or display_label in ("—", "Show your hand"):
        sign_color = (180, 180, 180)
        conf_text = "Confidence: —"
    else:
        sign_color = (0, 255, 0)
        conf_text = f"Confidence: {confidence:.1%}"

    cv2.putText(display, f"Sign: {display_label}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, sign_color, 2)
    cv2.putText(display, conf_text, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display, f"Sentence: {sentence}", (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    bbox = result.get("bbox")
    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    perf = result.get("performance", {})
    fps_text = f"FPS: {perf.get('fps', 0):.1f} | Latency: {perf.get('avg_latency_ms', 0):.0f}ms"
    cv2.putText(display, fps_text, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    help_text = "Q: Quit | C: Clear | B: Backspace | SPACE: Add space"
    cv2.putText(display, help_text, (15, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    hand_img = result.get("hand_image")
    if hand_img is not None:
        preview = cv2.resize(hand_img, (100, 100))
        display[h - 110:h - 10, w - 110:w - 10] = preview
        cv2.rectangle(display, (w - 112, h - 112), (w - 8, h - 8), (255, 255, 255), 1)

    return display


def find_best_model(models_dir: Path) -> Path:
    """Find the most recent best_model.keras in models directory."""
    candidates = list(models_dir.rglob("best_model.keras"))
    if not candidates:
        candidates = list(models_dir.rglob("final_model.keras"))
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found in {models_dir}. Run: python scripts/train_model.py"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL Real-time Recognition")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--camera", type=int, default=None, help="Webcam index")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
    )

    model_path = args.model
    if model_path is None:
        model_path = str(find_best_model(resolve_path(config["paths"]["models"])))
    logger.info("Loading model: %s", model_path)

    try:
        model = RealtimeRecognizer.load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    recognizer = RealtimeRecognizer(model, config, model_path=model_path)
    cam_cfg = config["webcam"]
    camera_index = args.camera if args.camera is not None else cam_cfg["camera_index"]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", camera_index)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    logger.info("Starting real-time recognition. Press Q to quit.")
    window_name = "ASL Sign Language Recognition"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            result = recognizer.process_frame(frame)
            display = draw_ui(frame, result, config["classes"]["all"])
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                recognizer.sentence_builder.clear()
                logger.info("Sentence cleared")
            elif key == ord("b"):
                recognizer.sentence_builder.backspace()
            elif key == ord(" "):
                recognizer.sentence_builder.sentence.append("")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Runtime error: %s", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application closed")


if __name__ == "__main__":
    main()
