"""Phase 4.1 — per-label threshold calibration.

The model outputs probabilities. **You** choose the operating point, and 0.5 is an
arbitrary default that happens to be wrong for imbalanced labels.

Why 0.5 is wrong: a classifier trained on a label with 0.30% prevalence sees 333
negatives for every positive, so its calibrated posterior for a genuinely
threatening comment can sit at 0.2 and still be the model's way of saying "this is
~70x more likely to be a threat than a random comment". Thresholding that at 0.5
throws the detection away. The fix costs no retraining — only a different cut point
per label.

Methodology, which is the part that matters:

* Thresholds are selected on the **validation** split only.
* They are then **frozen** and applied to the **test** split, which is read exactly
  once, after selection. Tuning on test would report an optimistically biased number
  by letting the test labels choose the operating point.
* The grid is the spec's 19-point ``linspace(0.05, 0.95, 19)``. Deliberately coarse:
  `threat` has 71 validation positives, so a fine grid would fit noise in the third
  decimal place and generalize worse. The val-vs-test F1 gap reported per label
  quantifies exactly how much that small-sample noise costs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.config import LABELS, PROBS, RESULTS, set_seed, setup_logging
from src.data import load_all_splits
from src.evaluate import evaluate, log_report, save_metrics

log = logging.getLogger(__name__)

GRID = np.linspace(0.05, 0.95, 19)
DEFAULT_THRESHOLD = 0.5


def tune_thresholds(
    Y_val: np.ndarray, probs_val: np.ndarray, grid: np.ndarray = GRID
) -> tuple[dict[str, float], dict]:
    """Pick the F1-maximizing threshold per label on the validation set.

    Returns (thresholds, per-label diagnostics). Only validation data is touched.
    """
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}

    for i, label in enumerate(LABELS):
        f1s = np.array([
            f1_score(Y_val[:, i], probs_val[:, i] >= t, zero_division=0) for t in grid
        ])
        best = int(np.argmax(f1s))
        best_t = float(grid[best])
        f1_default = f1_score(Y_val[:, i], probs_val[:, i] >= DEFAULT_THRESHOLD, zero_division=0)

        thresholds[label] = best_t
        diagnostics[label] = {
            "chosen_threshold": best_t,
            "val_f1_at_chosen": float(f1s[best]),
            "val_f1_at_0.5": float(f1_default),
            "val_f1_gain": float(f1s[best] - f1_default),
            "val_positives": int(Y_val[:, i].sum()),
            "f1_curve": {f"{t:.2f}": float(f) for t, f in zip(grid, f1s)},
        }
        log.info(
            "  %-14s t=%.2f  val F1 %.4f -> %.4f (%+.4f) | %d val positives",
            label, best_t, f1_default, f1s[best], f1s[best] - f1_default, Y_val[:, i].sum(),
        )

    return thresholds, diagnostics


def plot_threshold_sweep(diagnostics: dict, thresholds: dict[str, float], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(LABELS)))
    for (label, d), c in zip(diagnostics.items(), colors):
        ts = np.array([float(t) for t in d["f1_curve"]])
        f1s = np.array(list(d["f1_curve"].values()))
        ax.plot(ts, f1s, marker="o", ms=3, color=c, label=f"{label} (t*={thresholds[label]:.2f})")
        ax.scatter([thresholds[label]], [d["val_f1_at_chosen"]], color=c, s=90, zorder=5,
                   edgecolor="black", linewidth=0.8)
    ax.axvline(DEFAULT_THRESHOLD, ls="--", color="grey", lw=1.2)
    ax.text(DEFAULT_THRESHOLD + 0.01, 0.02, "default 0.5", fontsize=9, color="grey")
    ax.set_xlabel("threshold")
    ax.set_ylabel("F1 on validation")
    ax.set_title("Per-label F1 vs threshold (validation) — the optimum is label-specific")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    log.info("wrote %s", out_path)
    return out_path


def calibrate(model_tag: str) -> dict:
    """Tune on val, freeze, evaluate on test. Writes thresholds + comparison JSON."""
    splits = load_all_splits()
    Y_val, Y_test = splits["val"][1], splits["test"][1]

    probs_val = np.load(PROBS / f"{model_tag}_val.npy")
    probs_test = np.load(PROBS / f"{model_tag}_test.npy")

    log.info("=== tuning thresholds on VALIDATION only (%s) ===", model_tag)
    thresholds, diagnostics = tune_thresholds(Y_val, probs_val)

    payload = {
        "model": model_tag,
        "grid": GRID.tolist(),
        "tuned_on": "validation split only — test was not consulted during selection",
        "thresholds": thresholds,
        "per_label": diagnostics,
    }
    save_metrics(payload, RESULTS / f"thresholds_{model_tag}.json")
    plot_threshold_sweep(diagnostics, thresholds, RESULTS / f"threshold_sweep_{model_tag}.png")

    # --- thresholds are now frozen; only here do we touch test ---------------
    before = evaluate(Y_test, probs_test, DEFAULT_THRESHOLD, name=f"{model_tag} @0.5 (test)")
    after = evaluate(Y_test, probs_test, thresholds, name=f"{model_tag} @tuned (test)")

    log.info("=== BEFORE: default 0.5 on test ===")
    log_report(before)
    log.info("=== AFTER: per-label tuned thresholds on test ===")
    log_report(after)

    comparison = {
        "model": model_tag,
        "headline": {
            "macro_f1_at_0.5": before["macro_f1"],
            "macro_f1_tuned": after["macro_f1"],
            "macro_f1_gain": after["macro_f1"] - before["macro_f1"],
            # None rather than a division by zero: a model that predicts nothing at 0.5
            # scores macro-F1 exactly 0.0, which is the failure mode this project is about.
            "relative_gain_percent": (
                100 * (after["macro_f1"] - before["macro_f1"]) / before["macro_f1"]
                if before["macro_f1"] > 0
                else None
            ),
        },
        "micro_f1_at_0.5": before["micro_f1"],
        "micro_f1_tuned": after["micro_f1"],
        "per_label": {
            label: {
                "threshold": thresholds[label],
                "test_f1_at_0.5": before["per_label"][label]["f1"],
                "test_f1_tuned": after["per_label"][label]["f1"],
                "test_f1_gain": after["per_label"][label]["f1"] - before["per_label"][label]["f1"],
                "val_f1_tuned": diagnostics[label]["val_f1_at_chosen"],
                # Generalization gap: how much of the validation gain survived on test.
                # Large negative values on rare labels = the threshold overfit the
                # small validation positive count.
                "val_to_test_gap": after["per_label"][label]["f1"] - diagnostics[label]["val_f1_at_chosen"],
                "val_positives": diagnostics[label]["val_positives"],
                "test_positives": before["per_label"][label]["support"],
                "recall_at_0.5": before["per_label"][label]["recall"],
                "recall_tuned": after["per_label"][label]["recall"],
                "precision_at_0.5": before["per_label"][label]["precision"],
                "precision_tuned": after["per_label"][label]["precision"],
            }
            for label in LABELS
        },
    }

    gaps = {l: comparison["per_label"][l]["val_to_test_gap"] for l in LABELS}
    noisiest = min(gaps, key=gaps.get)
    comparison["small_sample_caveat"] = (
        f"`{noisiest}` shows the largest validation-to-test F1 drop "
        f"({gaps[noisiest]:+.4f}) with only {diagnostics[noisiest]['val_positives']} validation "
        f"positives to tune against. Per-label thresholds for rare labels are estimated from "
        f"very few positives, so their tuned F1 on validation is an optimistic estimate. This is "
        f"a real limitation of the method, not a bug — reported rather than hidden."
    )

    save_metrics(comparison, RESULTS / f"threshold_comparison_{model_tag}.json")
    save_metrics(after, RESULTS / f"{model_tag}_tuned_metrics.json")

    rel = comparison["headline"]["relative_gain_percent"]
    log.info("=" * 72)
    log.info(
        "MACRO-F1  %.4f (@0.5)  ->  %.4f (tuned)   %+.4f  (%s)",
        before["macro_f1"], after["macro_f1"],
        after["macro_f1"] - before["macro_f1"],
        f"{rel:+.1f}%" if rel is not None else "from zero",
    )
    log.info("  %-14s %6s %8s %8s %8s", "label", "t", "F1@0.5", "F1 tuned", "gain")
    for label in LABELS:
        p = comparison["per_label"][label]
        log.info(
            "  %-14s %6.2f %8.4f %8.4f %+8.4f",
            label, p["threshold"], p["test_f1_at_0.5"], p["test_f1_tuned"], p["test_f1_gain"],
        )
    log.info(comparison["small_sample_caveat"])
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.1: per-label threshold calibration.")
    parser.add_argument(
        "--model",
        default="transformer",
        help="probability file tag in models/probs (e.g. transformer, baseline)",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed()
    calibrate(args.model)


if __name__ == "__main__":
    main()
