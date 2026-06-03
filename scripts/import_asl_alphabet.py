"""
Import Kaggle ASL Alphabet dataset into data/raw/.

1. Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Extract the zip
3. Run:
   python scripts/import_asl_alphabet.py --source "C:/path/to/asl_alphabet_train/asl_alphabet_train"
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path

SKIP = {"space", "nothing", "del", "delete", "nothing1", "nothing2", "nothing3", "nothing4"}


def find_class_root(source: Path) -> Path:
    """Find folder containing A, B, C subdirectories."""
    if (source / "A").is_dir():
        return source
    for child in source.iterdir():
        if child.is_dir() and (child / "A").is_dir():
            return child
    raise FileNotFoundError(
        f"Could not find A/, B/, C/ folders under {source}. "
        "Point --source to the extracted asl_alphabet_train folder."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kaggle ASL Alphabet (A-Z)")
    parser.add_argument("--source", required=True, help="Path to extracted Kaggle dataset")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    classes = set(config["classes"]["all"])
    source = Path(args.source).resolve()
    root = find_class_root(source)
    dest = resolve_path(config["paths"]["raw_data"])
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.upper()
        if label in SKIP or label not in classes:
            continue

        out_dir = dest / label
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(class_dir, out_dir)
        n = len(list(out_dir.glob("*.*")))
        copied += n
        print(f"  {label}: {n} images")

    print(f"\nImported {copied} images to {dest}")
    print("Next: python scripts/prepare_dataset.py")
    print("       python scripts/train_model.py --epochs 25")


if __name__ == "__main__":
    main()
