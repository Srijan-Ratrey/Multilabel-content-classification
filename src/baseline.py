"""Phase 1 — the dumb baseline: TF-IDF + One-vs-Rest logistic regression.

The point of this phase is the plumbing insight: **multi-label classification is
N independent binary problems**. That is literally what One-vs-Rest does — it
fits six separate logistic regressions, one per label, each answering "is this
label on?" with no knowledge of the other five. Everything about the label
co-occurrence structure found in Phase 0 is invisible to this model by
construction.

Two variants are trained, because they fail in *opposite* directions at the
default 0.5 threshold:

* ``--class-weight balanced`` (the spec's recipe) reweights each label's loss by
  inverse frequency. It over-predicts rare labels: high recall, poor precision.
* ``--class-weight none`` is the untouched maximum-likelihood fit. It behaves the
  way the spec anticipates — on `threat` (0.30% prevalence) it almost never
  crosses 0.5 at all, so precision/recall/F1 collapse toward zero.

Reporting both is what actually demonstrates the phenomenon: the rare-label
failure is a property of *the operating point*, not of the algorithm. Phase 4
fixes it by moving the threshold rather than by changing the model.

Metrics here are computed with sklearn directly. Phase 2 builds the real
evaluation harness and re-evaluates the probability matrices this phase saves.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier

from src.config import LABELS, MODELS, PROBS, RESULTS, SEED, set_seed, setup_logging
from src.data import load_all_splits

log = logging.getLogger(__name__)

MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
DEFAULT_THRESHOLD = 0.5


def build_pipeline(class_weight: str | None) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    """TF-IDF features + one independent logistic regression per label."""
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=SEED,
        ),
        n_jobs=-1,
    )
    return vectorizer, clf


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-label precision/recall/F1/support at a fixed threshold."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(LABELS))
    )
    return {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "n_predicted_positive": int(y_pred[:, i].sum()),
        }
        for i, label in enumerate(LABELS)
    }


def run(class_weight: str | None, tag: str) -> dict:
    set_seed()
    splits = load_all_splits()
    (train_texts, Y_train) = splits["train"]

    vectorizer, clf = build_pipeline(class_weight)

    t0 = time.perf_counter()
    X_train = vectorizer.fit_transform(train_texts)
    log.info("TF-IDF fit: %s in %.1fs", X_train.shape, time.perf_counter() - t0)

    t0 = time.perf_counter()
    clf.fit(X_train, Y_train)
    log.info("OvR LogReg fit (%d independent binary models) in %.1fs", len(LABELS), time.perf_counter() - t0)

    joblib.dump(vectorizer, MODELS / f"{tag}_tfidf.joblib")
    joblib.dump(clf, MODELS / f"{tag}_ovr.joblib")

    metrics = {
        "phase": 1,
        "model": f"tfidf+ovr-logreg ({'class_weight=balanced' if class_weight else 'unweighted'})",
        "class_weight": class_weight,
        "seed": SEED,
        "threshold": DEFAULT_THRESHOLD,
        "vectorizer": {"max_features": MAX_FEATURES, "ngram_range": list(NGRAM_RANGE), "vocab_size": len(vectorizer.vocabulary_)},
        "splits": {},
    }

    for split in ("val", "test"):
        texts, Y = splits[split]
        # predict_proba on an OvR wrapper returns P(label=1) per label, shape (n, 6).
        probs = clf.predict_proba(vectorizer.transform(texts))
        np.save(PROBS / f"{tag}_{split}.npy", probs.astype(np.float32))

        y_pred = (probs >= DEFAULT_THRESHOLD).astype(np.int8)
        metrics["splits"][split] = {
            "n": int(len(Y)),
            "per_label": per_label_metrics(Y, y_pred),
        }
        log.info("saved %s probabilities -> %s", split, PROBS / f"{tag}_{split}.npy")

    # The Phase 1 deliverable: state the rarest label's behaviour at 0.5 explicitly.
    rare = "threat"
    r = metrics["splits"]["test"]["per_label"][rare]
    metrics["rare_label_note"] = (
        f"At threshold {DEFAULT_THRESHOLD}, `{rare}` has {r['support']} true positives in test "
        f"and the model predicts positive {r['n_predicted_positive']} times "
        f"(precision {r['precision']:.3f}, recall {r['recall']:.3f}, F1 {r['f1']:.3f})."
    )

    out = RESULTS / f"{tag}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out)

    log.info("--- test per-label F1 @ %.2f ---", DEFAULT_THRESHOLD)
    for label, m in metrics["splits"]["test"]["per_label"].items():
        log.info(
            "  %-14s P=%.3f R=%.3f F1=%.3f | support=%4d predicted=%5d",
            label, m["precision"], m["recall"], m["f1"], m["support"], m["n_predicted_positive"],
        )
    log.info(metrics["rare_label_note"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 baseline: TF-IDF + OvR logistic regression.")
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="'balanced' is the spec recipe; 'none' exposes the rare-label collapse at 0.5",
    )
    args = parser.parse_args()

    setup_logging()
    class_weight = None if args.class_weight == "none" else "balanced"
    tag = "baseline" if class_weight else "baseline_unweighted"
    run(class_weight, tag)


if __name__ == "__main__":
    main()
