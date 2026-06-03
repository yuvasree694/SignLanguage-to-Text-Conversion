"""Evaluate a trained ASL model."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset import ASLDataset
from src.training.evaluator import ASLEvaluator
from src.training.trainer import ASLTrainer
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASL model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data", default=None, help="Processed test data directory")
    parser.add_argument("--output", default="outputs/evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger(log_file=config["logging"]["file"], level=config["logging"]["level"])

    data_dir = args.data or config["paths"]["processed_data"]
    dataset = ASLDataset(config)
    _, _, X_test, _, _, y_test = dataset.load_processed(data_dir)

    trainer = ASLTrainer(config)
    model = trainer.load_model(args.model)

    history_path = resolve_path(args.model).parent / "history.json"
    history = None
    if history_path.exists():
        import json
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

    evaluator = ASLEvaluator(config)
    metrics = evaluator.full_evaluation(model, X_test, y_test, history=history, output_dir=args.output)

    print("\n=== Evaluation Results ===")
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
