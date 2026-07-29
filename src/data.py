"""Data loading, cleaning, label-matrix construction, and splitting.

The central object is ``Y``: an ``(n_samples, n_labels)`` binary matrix whose
column order is ``config.LABELS``. Everything downstream — baseline, transformer,
threshold calibration, error analysis — assumes exactly that shape and order.

Splits are computed once, persisted to ``data/processed/splits.npz``, and reloaded
by every later phase, so all reported numbers are comparable across models.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re

import numpy as np
import pandas as pd

from src.config import (
    KAGGLE_TEST_CSV,
    KAGGLE_TEST_LABELS_CSV,
    LABELS,
    RESULTS,
    SEED,
    SPLITS_NPZ,
    TEST_FRACTION,
    TRAIN_CSV,
    VAL_FRACTION,
    set_seed,
    setup_logging,
)

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Minimal cleaning: HTML-unescape, then collapse all whitespace runs.

    Deliberately minimal. ALL-CAPS, exclamation runs, and character repetition
    ("!!!!!", "sooooo") are genuine toxicity signal, so they are preserved.
    Lowercasing is left to the downstream consumers that actually want it:
    TfidfVectorizer lowercases by default, and distilbert-base-uncased is an
    uncased model. Stripping that signal here would silently remove it from both.
    """
    return _WHITESPACE.sub(" ", html.unescape(str(text))).strip()


def load_raw() -> pd.DataFrame:
    """Load Kaggle ``train.csv`` — the only fully-labeled file (159,571 rows).

    All three of our splits come from this file. See ``load_kaggle_test`` for why
    the competition's own test files cannot serve as our test set.
    """
    df = pd.read_csv(TRAIN_CSV)
    missing = [c for c in ["id", "comment_text", *LABELS] if c not in df.columns]
    if missing:
        raise ValueError(f"{TRAIN_CSV} is missing expected columns: {missing}")

    df["comment_text"] = df["comment_text"].map(clean_text)

    n_before = len(df)
    df = df[df["comment_text"].str.len() > 0].reset_index(drop=True)
    if len(df) < n_before:
        log.warning("dropped %d rows that were empty after cleaning", n_before - len(df))

    log.info("loaded %d labeled comments from %s", len(df), TRAIN_CSV.name)
    return df


def build_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the (n_samples, n_labels) binary label matrix in LABELS order."""
    Y = df[LABELS].to_numpy(dtype=np.int8)
    if not np.isin(Y, [0, 1]).all():
        raise ValueError(
            "train.csv labels must be strictly 0/1. Found other values — if these "
            "are -1, you are reading a competition test-label file, not train.csv."
        )
    log.info("label matrix %s | positives per label: %s", Y.shape, dict(zip(LABELS, Y.sum(0).tolist())))
    return Y


def _split_iterstrat(Y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Split with MultilabelStratifiedShuffleSplit (requires iterative-stratification)."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    X_dummy = np.zeros((len(Y), 1), dtype=np.int8)
    holdout_size = VAL_FRACTION + TEST_FRACTION

    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
    train_idx, holdout_idx = next(first.split(X_dummy, Y))

    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION / holdout_size, random_state=seed
    )
    rel_val, rel_test = next(second.split(X_dummy[holdout_idx], Y[holdout_idx]))
    return {
        "train": np.sort(train_idx),
        "val": np.sort(holdout_idx[rel_val]),
        "test": np.sort(holdout_idx[rel_test]),
    }


def _split_combination(Y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Split stratified on each row's exact label COMBINATION. **This is the default.**

    With 6 binary labels there are at most 64 distinct combinations, so stratifying on the
    combination is a *finer* partition than per-label stratification and preserves every
    label's positive rate by construction. Combinations too rare to divide three ways
    (fewer than 6 rows) are pooled into one bucket and split randomly.

    Why this is the default rather than iterative-stratification: it depends only on
    scikit-learn, so it produces byte-identical splits in environments where
    `iterative-stratification` is unavailable — notably Kaggle with internet off. The
    DistilBERT probabilities in `models/probs/` were produced against this split, so making
    it canonical is what lets the baseline and the transformer be compared like for like.
    """
    from sklearn.model_selection import train_test_split

    combo = (Y.astype(np.int64) * (2 ** np.arange(Y.shape[1]))).sum(axis=1)
    counts = pd.Series(combo).value_counts()
    rare = set(counts[counts < 6].index)
    if rare:
        combo = np.where(np.isin(combo, list(rare)), -1, combo)

    idx = np.arange(len(Y))
    holdout_size = VAL_FRACTION + TEST_FRACTION
    train_idx, holdout_idx = train_test_split(
        idx, test_size=holdout_size, random_state=seed, stratify=combo
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=TEST_FRACTION / holdout_size,
        random_state=seed,
        stratify=combo[holdout_idx],
    )
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def make_splits(Y: np.ndarray, seed: int = SEED, strategy: str = "combination") -> dict[str, np.ndarray]:
    """70/15/15 multi-label-stratified split, computed once and persisted.

    ``strategy="combination"`` (default) stratifies on the exact label combination and needs
    only scikit-learn; ``strategy="iterstrat"`` uses MultilabelStratifiedShuffleSplit. Both
    keep every label's positive rate stable across splits, which matters for `threat`
    (0.30% prevalence) where a plain random split would scatter its ~478 positives unevenly.

    The default is "combination" so that this module, `notebooks/train_model.ipynb`, and a
    Kaggle run all produce the *same* split — see `_split_combination` for why that matters.
    """
    splits = (_split_combination if strategy == "combination" else _split_iterstrat)(Y, seed)

    # No row may appear in two splits, and none may be lost.
    all_idx = np.concatenate(list(splits.values()))
    assert len(np.unique(all_idx)) == len(all_idx) == len(Y), "splits overlap or lose rows"

    np.savez_compressed(SPLITS_NPZ, **splits)
    log.info("saved splits (strategy=%s) to %s", strategy, SPLITS_NPZ)

    holdout_size = VAL_FRACTION + TEST_FRACTION
    summary = {
        "seed": seed,
        "strategy": strategy,
        "fractions": {"train": 1 - holdout_size, "val": VAL_FRACTION, "test": TEST_FRACTION},
        "sizes": {k: int(len(v)) for k, v in splits.items()},
        "positives_per_label": {
            k: dict(zip(LABELS, Y[v].sum(0).tolist())) for k, v in splits.items()
        },
        "positive_rate_per_label": {
            k: dict(zip(LABELS, Y[v].mean(0).round(5).tolist())) for k, v in splits.items()
        },
        "note": (
            "threat has ~72 positives in each of val and test. Per-label threshold "
            "calibration on that few positives is noisy; quantified in Phase 4."
        ),
        "reproducibility": (
            "strategy='combination' depends only on scikit-learn, so notebooks/train_model.ipynb "
            "reproduces this split exactly on Kaggle without iterative-stratification. The "
            "DistilBERT probabilities in models/probs/ were produced against it."
        ),
    }
    (RESULTS / "splits.json").write_text(json.dumps(summary, indent=2))
    return splits


def load_splits() -> dict[str, np.ndarray]:
    """Load persisted split indices. Fails loudly rather than silently re-splitting."""
    if not SPLITS_NPZ.exists():
        raise FileNotFoundError(
            f"{SPLITS_NPZ} not found. Run `python -m src.data` first — every phase must "
            "use the same splits for its numbers to be comparable."
        )
    with np.load(SPLITS_NPZ) as f:
        return {k: f[k] for k in ("train", "val", "test")}


def get_split_data(split: str) -> tuple[list[str], np.ndarray]:
    """Return ``(texts, Y)`` for one of 'train' | 'val' | 'test'."""
    df = load_raw()
    Y = build_label_matrix(df)
    idx = load_splits()[split]
    return df["comment_text"].to_numpy()[idx].tolist(), Y[idx]


def load_all_splits() -> dict[str, tuple[list[str], np.ndarray]]:
    """Load all three splits in one pass over the CSV."""
    df = load_raw()
    Y = build_label_matrix(df)
    texts = df["comment_text"].to_numpy()
    return {k: (texts[i].tolist(), Y[i]) for k, i in load_splits().items()}


def load_kaggle_test() -> tuple[list[str], np.ndarray]:
    """Load the competition test set, keeping ONLY rows with real ground truth.

    THE -1 TRAP: ``test_labels.csv`` has 153,164 rows, but 89,186 of them carry
    -1 in every label column. Those rows were withheld from the original
    competition's scoring and are NOT usable ground truth — -1 is "unknown", not
    "negative". Only 63,978 rows have real 0/1 labels. Evaluating against the
    unfiltered file treats 89,186 unknowns as negatives and produces meaningless
    metrics.

    The filter lives here, once, with assertions below, so no caller can bypass
    it. Note this is a secondary out-of-sample check only: the project's
    train/val/test splits all come from train.csv via ``make_splits``.
    """
    texts = pd.read_csv(KAGGLE_TEST_CSV)
    labels = pd.read_csv(KAGGLE_TEST_LABELS_CSV)
    df = texts.merge(labels, on="id", validate="one_to_one")

    n_total = len(df)
    scoreable = (df[LABELS] != -1).all(axis=1)
    partial = (df[LABELS] == -1).any(axis=1) & ~(df[LABELS] == -1).all(axis=1)
    assert not partial.any(), "unexpected row with a mix of -1 and real labels"

    df = df[scoreable].reset_index(drop=True)
    df["comment_text"] = df["comment_text"].map(clean_text)
    Y = df[LABELS].to_numpy(dtype=np.int8)

    # Hard guards: no -1 may survive, and the count must match the known figure.
    assert (Y >= 0).all(), "a -1 label survived filtering"
    assert np.isin(Y, [0, 1]).all(), "non-binary label survived filtering"
    assert len(df) == 63978, f"expected 63,978 scoreable rows, got {len(df)}"

    log.info(
        "kaggle test: kept %d scoreable rows, dropped %d unlabeled (-1) rows",
        len(df),
        n_total - len(df),
    )
    return df["comment_text"].tolist(), Y


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label matrix and persist splits.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--strategy",
        choices=["combination", "iterstrat"],
        default="combination",
        help="combination (default) matches notebooks/train_model.ipynb and Kaggle exactly",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    df = load_raw()
    Y = build_label_matrix(df)
    splits = make_splits(Y, seed=args.seed, strategy=args.strategy)

    for name, idx in splits.items():
        rates = dict(zip(LABELS, (100 * Y[idx].mean(0)).round(2).tolist()))
        log.info("%-5s n=%6d | positive rate %%: %s", name, len(idx), rates)


if __name__ == "__main__":
    main()
