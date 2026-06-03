"""Architecture experimentation for systematic CNN optimization."""

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.models.cnn_model import build_cnn_model, compile_model
from src.utils.config_loader import resolve_path

logger = logging.getLogger("asl_system")


EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "baseline_2conv",
        "conv_layers": [
            {"filters": 32, "kernel_size": 3, "activation": "relu", "batch_norm": True},
            {"filters": 64, "kernel_size": 3, "activation": "relu", "batch_norm": True},
        ],
        "dense_units": [256],
        "dropout": 0.3,
    },
    {
        "name": "deep_4conv",
        "conv_layers": [
            {"filters": 32, "kernel_size": 3, "activation": "relu", "batch_norm": True},
            {"filters": 64, "kernel_size": 3, "activation": "relu", "batch_norm": True},
            {"filters": 128, "kernel_size": 3, "activation": "relu", "batch_norm": True},
            {"filters": 256, "kernel_size": 3, "activation": "relu", "batch_norm": True},
        ],
        "dense_units": [512, 256],
        "dropout": 0.5,
    },
    {
        "name": "wide_kernels",
        "conv_layers": [
            {"filters": 32, "kernel_size": 5, "activation": "relu", "batch_norm": True},
            {"filters": 64, "kernel_size": 5, "activation": "relu", "batch_norm": True},
            {"filters": 128, "kernel_size": 3, "activation": "relu", "batch_norm": True},
        ],
        "dense_units": [512],
        "dropout": 0.4,
    },
    {
        "name": "leaky_relu",
        "conv_layers": [
            {"filters": 32, "kernel_size": 3, "activation": "leaky_relu", "batch_norm": True},
            {"filters": 64, "kernel_size": 3, "activation": "leaky_relu", "batch_norm": True},
            {"filters": 128, "kernel_size": 3, "activation": "leaky_relu", "batch_norm": True},
        ],
        "dense_units": [256],
        "dropout": 0.5,
    },
    {
        "name": "high_dropout",
        "conv_layers": [
            {"filters": 32, "kernel_size": 3, "activation": "relu", "batch_norm": True},
            {"filters": 64, "kernel_size": 3, "activation": "relu", "batch_norm": True},
            {"filters": 128, "kernel_size": 3, "activation": "relu", "batch_norm": True},
        ],
        "dense_units": [512, 256],
        "dropout": 0.6,
    },
]

OPTIMIZER_EXPERIMENTS = [
    {"optimizer": "adam", "learning_rate": 0.001},
    {"optimizer": "adam", "learning_rate": 0.0005},
    {"optimizer": "sgd", "learning_rate": 0.01},
    {"optimizer": "rmsprop", "learning_rate": 0.001},
]


def run_architecture_experiments(
    config: Dict[str, Any],
    output_dir: str = "data/models/experiments",
) -> List[Dict[str, Any]]:
    """Build all experiment architectures and save metadata."""
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for exp in EXPERIMENT_CONFIGS:
        arch = copy.deepcopy(config["model"])
        arch.update(exp)
        model = build_cnn_model(config, architecture_override=arch)
        model = compile_model(model, config)

        exp_path = out / exp["name"]
        exp_path.mkdir(exist_ok=True)
        model.save(exp_path / "model.keras")

        meta = {
            "name": exp["name"],
            "params": model.count_params(),
            "architecture": exp,
            "timestamp": datetime.now().isoformat(),
        }
        with open(exp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        results.append(meta)
        logger.info("Experiment '%s': %d parameters", exp["name"], meta["params"])

    summary_path = out / "experiments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
