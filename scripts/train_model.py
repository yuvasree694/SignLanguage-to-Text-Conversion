"""Train ASL CNN model."""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Train ASL CNN model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="Processed data directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run architecture experiment by name")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger(log_file=config["logging"]["file"], level=config["logging"]["level"])

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    data_dir = args.data or config["paths"]["processed_data"]
    processed_path = resolve_path(data_dir)

    if not (processed_path / "X_train.npy").exists():
        print("ERROR: No processed data. Run:")
        print('  python scripts/import_asl_alphabet.py --source "PATH_TO_KAGGLE_DATA"')
        print("  python scripts/prepare_dataset.py")
        sys.exit(1)

    meta_path = processed_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("num_classes") != config["num_classes"]:
            print("ERROR: Class count changed. Re-run: python scripts/prepare_dataset.py")
            sys.exit(1)
        if not meta.get("real_data", True):
            print("WARNING: Old synthetic dataset — re-prepare with real ASL images!")

    dataset = ASLDataset(config)
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load_processed(data_dir)

    trainer = ASLTrainer(config)

    arch_override = None
    if args.experiment:
        from src.models.experiments import EXPERIMENT_CONFIGS
        for exp in EXPERIMENT_CONFIGS:
            if exp["name"] == args.experiment:
                arch_override = {**config["model"], **exp}
                break

    trainer.build_and_compile(architecture_override=arch_override)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        use_augmentation=not args.no_augmentation,
    )

    evaluator = ASLEvaluator(config)
    metrics = evaluator.full_evaluation(
        trainer.model, X_test, y_test,
        history=history.history,
    )

    print("\n=== Training Complete ===")
    print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision_macro']:.4f}")
    print(f"Recall:         {metrics['recall_macro']:.4f}")
    print(f"F1 Score:       {metrics['f1_macro']:.4f}")
    print(f"Train Accuracy: {max(history.history.get('accuracy', [0])):.4f}")


if __name__ == "__main__":
    main()
