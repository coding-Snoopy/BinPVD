"""Utility functions for BinPVD experiments."""

import os
import random
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timestamp() -> str:
    """Return current timestamp string for checkpoint naming."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_config(config, path: str) -> None:
    """Save experiment config to JSON."""
    with open(path, "w") as f:
        json.dump(config.to_dict() if hasattr(config, "to_dict") else config, f, indent=2, default=str)


def compute_metrics(
    targets: List[int],
    preds: List[int],
) -> Dict[str, float]:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
    }


def setup_logging(output_dir: str, log_name: str = "train") -> logging.Logger:
    """Configure logging to file and console."""
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, f"{log_name}.log")

    logger = logging.getLogger("binpvd")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
