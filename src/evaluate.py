"""Phase 2 — the evaluation harness. Every later phase reports through this.

**Why accuracy is disqualified.** 89.8% of Jigsaw comments carry no label at all,
and `threat` appears in 0.30% of them. A model that predicts nothing scores 89.8%
exact-match accuracy, 96.3% mean per-label accuracy, and 99.70% on `threat` —
while having macro-F1 of exactly 0.0 and zero utility. Accuracy measures how well
you predict the majority class, and here the majority class is "nothing".

So the headline metric is **macro-F1**, and the two aggregations are reported side
by side because their *divergence* is the diagnostic:

* **Micro-F1** pools every (example, label) decision into one contingency table.
  Frequent labels dominate it — `toxic` alone contributes 15,294 of the 35,098
  positive decisions, so a model can score well on micro-F1 while completely
  ignoring `threat`.
* **Macro-F1** computes F1 per label and averages the six equally. `threat`
  carries the same weight as `toxic`, so ignoring a rare label is punished in
  proportion to how many labels you ignore, not how many rows they cover.

A high micro / low macro gap therefore reads directly as "this model is ignoring
its rare labels" — the central claim this project is organized around.

Average precision (area under the PR curve) is also reported because it is
**threshold-free**: it summarizes the ranking quality that Phase 4 exploits when
it moves the operating point. A model can have excellent AP and terrible F1 at
0.5 — that combination is precisely a threshold problem, not a model problem.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from src.config import LABELS, N_LABELS, RESULTS, set_seed, setup_logging

log = logging.getLogger(__name__)

ACCURACY_CAVEAT = (
    "Shown only to demonstrate that it is misleading here: an all-zeros model "
    "achieves the accuracy figures in 'all_zeros_baseline' below at macro-F1 = 0.0. "
    "Accuracy rewards predicting the majority class, which in this dataset is 'no label'. "
    "Use macro_f1 as the headline metric."
)


def resolve_thresholds(thresholds: float | dict[str, float] | np.ndarray) -> np.ndarray:
    """Accept a scalar, a {label: t} mapping, or an array; return an (n_labels,) array."""
    if isinstance(thresholds, (int, float)):
        return np.full(N_LABELS, float(thresholds))
    if isinstance(thresholds, dict):
        missing = set(LABELS) - set(thresholds)
        if missing:
            raise ValueError(f"threshold mapping is missing labels: {sorted(missing)}")
        return np.array([float(thresholds[label]) for label in LABELS])
    arr = np.asarray(thresholds, dtype=float).ravel()
    if arr.shape != (N_LABELS,):
        raise ValueError(f"expected {N_LABELS} thresholds, got shape {arr.shape}")
    return arr


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: float | dict[str, float] | np.ndarray = 0.5,
    *,
    name: str = "model",
) -> dict:
    """Evaluate multi-label predictions.

    Args:
        y_true: (n_samples, n_labels) binary ground truth.
        y_prob: (n_samples, n_labels) predicted probabilities, columns in LABELS order.
        thresholds: scalar applied to all labels, or one threshold per label.
        name: label for logs and the results file.

    Returns a JSON-serializable dict. ``macro_f1`` is the headline.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if y_true.shape[1] != N_LABELS:
        raise ValueError(f"expected {N_LABELS} label columns, got {y_true.shape[1]}")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            "y_true must be strictly binary. A -1 here means unfiltered Kaggle "
            "test labels — see src.data.load_kaggle_test."
        )

    t = resolve_thresholds(thresholds)
    y_pred = (y_prob >= t[None, :]).astype(np.int8)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(N_LABELS)
    )

    # Average precision is threshold-free; it needs at least one positive to exist.
    ap = np.array([
        average_precision_score(y_true[:, i], y_prob[:, i]) if y_true[:, i].any() else float("nan")
        for i in range(N_LABELS)
    ])

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # --- the accuracy contrast (included only to disqualify accuracy) -------
    zeros = np.zeros_like(y_true)
    all_zeros = {
        "subset_accuracy": float((y_true.sum(axis=1) == 0).mean()),
        "mean_per_label_accuracy": float((y_true == zeros).mean()),
        "per_label_accuracy": {
            label: float((y_true[:, i] == 0).mean()) for i, label in enumerate(LABELS)
        },
        "macro_f1": float(f1_score(y_true, zeros, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, zeros, average="micro", zero_division=0)),
    }

    result = {
        "name": name,
        "n_samples": int(len(y_true)),
        # --- headline ------------------------------------------------------
        "macro_f1": macro_f1,
        # --- aggregation contrast -----------------------------------------
        "micro_f1": micro_f1,
        "micro_macro_gap": micro_f1 - macro_f1,
        # --- threshold-free ranking quality --------------------------------
        "macro_average_precision": float(np.nanmean(ap)),
        "micro_average_precision": float(average_precision_score(y_true, y_prob, average="micro")),
        # --- per label -----------------------------------------------------
        "per_label": {
            label: {
                "threshold": float(t[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "average_precision": float(ap[i]),
                "support": int(support[i]),
                "n_predicted_positive": int(y_pred[:, i].sum()),
                "positive_rate": float(y_true[:, i].mean()),
            }
            for i, label in enumerate(LABELS)
        },
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        # --- accuracy, quarantined ----------------------------------------
        "accuracy_do_not_use_as_headline": {
            "_caveat": ACCURACY_CAVEAT,
            "subset_accuracy_exact_match": float((y_pred == y_true).all(axis=1).mean()),
            "mean_per_label_accuracy": float((y_pred == y_true).mean()),
        },
        "all_zeros_baseline": all_zeros,
    }
    return result


def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str = "") -> Path:
    """Per-label precision-recall curves. The horizontal dashed line per label is
    its positive rate — the precision a random classifier would achieve, which
    makes visible how much harder the rare labels are.

    Note: this deliberately does NOT call ``matplotlib.use("Agg")``. Switching the
    backend is global and permanent for the process, so doing it here would break
    inline rendering for every subsequent cell in a notebook that imports this
    function. The CLI selects Agg itself in ``main()``, where it is safe.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for i, (label, ax) in enumerate(zip(LABELS, axes.ravel())):
        yt, yp = y_true[:, i], y_prob[:, i]
        if not yt.any():
            ax.set_title(f"{label} (no positives)")
            continue
        precision, recall, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        base = yt.mean()
        ax.plot(recall, precision, color="#C44E52", lw=1.8)
        ax.axhline(base, ls="--", lw=1, color="grey")
        ax.set_title(f"{label}\nAP={ap:.3f} | positives={int(yt.sum())} ({100*base:.2f}%)", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if i >= 3:
            ax.set_xlabel("recall")
        if i % 3 == 0:
            ax.set_ylabel("precision")
    fig.suptitle(title or "Per-label precision-recall curves (dashed = random-classifier precision)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    log.info("wrote %s", out_path)
    return out_path


def save_metrics(metrics: dict, out_path: Path) -> Path:
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out_path)
    return out_path


def log_report(metrics: dict) -> None:
    """Human-readable summary, headline first."""
    log.info("=== %s (n=%d) ===", metrics["name"], metrics["n_samples"])
    log.info("  MACRO-F1 (headline) : %.4f", metrics["macro_f1"])
    log.info("  micro-F1            : %.4f   (gap %+.4f)", metrics["micro_f1"], metrics["micro_macro_gap"])
    log.info("  macro AP            : %.4f   (threshold-free)", metrics["macro_average_precision"])
    log.info("  %-14s %5s %5s %5s %6s %8s %6s", "label", "P", "R", "F1", "AP", "support", "pred")
    for label, m in metrics["per_label"].items():
        log.info(
            "  %-14s %.3f %.3f %.3f  %.3f %8d %6d",
            label, m["precision"], m["recall"], m["f1"], m["average_precision"],
            m["support"], m["n_predicted_positive"],
        )
    az = metrics["all_zeros_baseline"]
    log.info(
        "  for contrast, an all-zeros model: exact-match acc %.4f, mean per-label acc %.4f, macro-F1 %.4f",
        az["subset_accuracy"], az["mean_per_label_accuracy"], az["macro_f1"],
    )


def compare_table(entries: list[tuple[str, dict]]) -> str:
    """Markdown results table across phases — used to build the README."""
    header = (
        "| model | macro-F1 | micro-F1 | macro-AP | " + " | ".join(f"F1 {l}" for l in LABELS) + " |\n"
        "|---|---|---|---|" + "---|" * N_LABELS + "\n"
    )
    rows = "".join(
        f"| {name} | **{m['macro_f1']:.4f}** | {m['micro_f1']:.4f} | {m['macro_average_precision']:.4f} | "
        + " | ".join(f"{m['per_label'][l]['f1']:.3f}" for l in LABELS)
        + " |\n"
        for name, m in entries
    )
    return header + rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 evaluation harness.")
    parser.add_argument("--probs", required=True, type=Path, help="(n, 6) .npy probability matrix")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--thresholds", default="0.5", help="scalar, or path to a thresholds JSON")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plots", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    set_seed()

    if args.plots:
        import matplotlib

        matplotlib.use("Agg")  # headless CLI: safe here, unlike inside plot_pr_curves

    from src.data import get_split_data

    _, y_true = get_split_data(args.split)
    y_prob = np.load(args.probs)

    try:
        thresholds: float | dict = float(args.thresholds)
    except ValueError:
        payload = json.loads(Path(args.thresholds).read_text())
        thresholds = payload.get("thresholds", payload)

    name = args.name or f"{args.probs.stem} @ {args.split}"
    metrics = evaluate(y_true, y_prob, thresholds, name=name)
    metrics["split"] = args.split
    metrics["probs_file"] = str(args.probs)
    log_report(metrics)

    if args.out:
        save_metrics(metrics, args.out)
    if args.plots:
        plot_pr_curves(y_true, y_prob, args.plots, title=f"Per-label PR curves — {name}")


if __name__ == "__main__":
    main()
