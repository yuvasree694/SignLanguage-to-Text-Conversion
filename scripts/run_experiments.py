"""Run architecture experiments."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.experiments import EXPERIMENT_CONFIGS, run_architecture_experiments
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN architecture experiments")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger(log_file=config["logging"]["file"], level=config["logging"]["level"])

    results = run_architecture_experiments(config)
    print("\n=== Architecture Experiments ===")
    for r in results:
        print(f"  {r['name']}: {r['params']:,} parameters")


if __name__ == "__main__":
    main()
