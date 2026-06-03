"""Prepare ASL dataset: load, preprocess, split, and save."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset import ASLDataset
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASL dataset")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default=None, help="Raw data directory")
    parser.add_argument("--output", default=None, help="Processed output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger(log_file=config["logging"]["file"], level=config["logging"]["level"])

    raw_dir = args.input or config["paths"]["raw_data"]
    output_dir = args.output or config["paths"]["processed_data"]
    raw_path = resolve_path(raw_dir)

    if ASLDataset.is_synthetic_dataset(raw_dir):
        print("\n*** WARNING: Synthetic demo data detected! ***")
        print("Real hand gestures will NOT be recognized correctly.")
        print("Download ASL Alphabet from Kaggle and run:")
        print('  python scripts/import_asl_alphabet.py --source "PATH_TO_DATASET"\n')

    if not any(raw_path.iterdir()) if raw_path.exists() else True:
        print("ERROR: data/raw/ is empty.")
        print("1. Download: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("2. Extract zip")
        print('3. python scripts/import_asl_alphabet.py --source "EXTRACTED_FOLDER"')
        sys.exit(1)

    dataset = ASLDataset(config)
    X, y = dataset.load_from_directory(raw_dir)
    splits = dataset.split_dataset(X, y)
    dataset.save_processed(output_dir, *splits)
    print(f"Dataset prepared: {len(X)} images, {config['num_classes']} classes")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
