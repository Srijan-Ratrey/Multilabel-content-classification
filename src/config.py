"""Shared configuration: seed, label schema, paths, device.

Every entry point imports SEED and LABELS from here so there is exactly one
definition of each. LABELS order defines the column order of the (n, 6) label
matrix Y, and every saved probability matrix on disk must match it.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

# --- Reproducibility -------------------------------------------------------
SEED = 42

# --- Label schema ----------------------------------------------------------
# Column order for Y and for every *_probs.npy on disk. Do not reorder.
LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
N_LABELS = len(LABELS)

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PROBS = MODELS / "probs"
RESULTS = ROOT / "results"

TRAIN_CSV = DATA_RAW / "train.csv"
KAGGLE_TEST_CSV = DATA_RAW / "test.csv"
KAGGLE_TEST_LABELS_CSV = DATA_RAW / "test_labels.csv"
SPLITS_NPZ = DATA_PROCESSED / "splits.npz"

# --- Split proportions -----------------------------------------------------
# 70/15/15. Consequence worth stating up front: `threat` has only 478 positives
# in the whole dataset, so val and test each hold roughly 72 of them. That small
# sample is why the Phase 4 threshold grid stays coarse (19 points, not 91) and
# why `threat`'s tuned threshold is reported with a noise caveat.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

for _d in (DATA_PROCESSED, MODELS, PROBS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> int:
    """Seed every source of randomness we touch, and log it (spec requirement)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass  # torch is only needed for Phases 3/4.2

    logging.getLogger(__name__).info("seed set to %d", seed)
    return seed


def get_device() -> str:
    """Return the best available torch device string ('mps' on Apple Silicon)."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mlcc")
