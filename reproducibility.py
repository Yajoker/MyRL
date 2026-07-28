"""Utilities for reproducible, seed-isolated training runs."""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def validate_random_seed(seed: int) -> int:
    """Return a non-negative integer seed or raise a clear configuration error."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("random_seed must be a non-negative integer")
    normalized = int(seed)
    if normalized < 0:
        raise ValueError("random_seed must be a non-negative integer")
    return normalized


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy and PyTorch before constructing the simulator or models.

    ``PYTHONHASHSEED`` only fully controls hash randomization when it is present
    before the interpreter starts.  It is still exported here so child
    processes, if introduced later, inherit the configured experiment seed.
    """

    normalized = validate_random_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(normalized)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return normalized


def seed_output_directory(root: Path, seed: int) -> Path:
    """Place one training seed under its own output directory."""

    normalized = validate_random_seed(seed)
    return Path(root) / f"seed_{normalized}"


def timestamped_seed_output_directory(
    root: Path,
    seed: int,
    started_at: datetime,
) -> Path:
    """Return an isolated seed directory carrying the run start timestamp."""

    normalized = validate_random_seed(seed)
    if not isinstance(started_at, datetime):
        raise TypeError("started_at must be a datetime")

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    return Path(root) / f"seed_{normalized}_{timestamp}"


__all__ = [
    "seed_everything",
    "seed_output_directory",
    "timestamped_seed_output_directory",
    "validate_random_seed",
]
