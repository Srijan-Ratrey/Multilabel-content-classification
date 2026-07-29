"""Run the fine-tuned model on your own text.

Usage:
    python -m src.predict "you are an idiot"
    python -m src.predict --interactive
    python -m src.predict --file comments.txt

The point of interest is the two verdict columns. The model emits six independent
probabilities; the labels it actually *fires* depend entirely on where you put the
cut point. This prints the decision at the default 0.5 alongside the decision at
the per-label thresholds calibrated in Phase 4, so the disagreement is visible on
real text rather than only in an aggregate metric.

`threat` is the clearest case: its tuned threshold is 0.20, so a comment the
default would wave through at p=0.35 is correctly flagged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from src.config import LABELS, MODELS, RESULTS, get_device, set_seed, setup_logging

log = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 256


def find_model_dir(explicit: Path | None = None) -> Path | str:
    """Locate the model: a Hub repo id if MODEL_ID is set, else a local directory.

    The MODEL_ID branch is what the deployed Space uses — the 256MB weights are gitignored,
    so they live in a Hugging Face model repo and `from_pretrained` resolves the id directly.
    Local search remains the fallback so `python -m src.predict` still works offline.
    """
    if explicit is None and (model_id := os.environ.get("MODEL_ID", "").strip()):
        log.info("MODEL_ID set -- loading from the Hugging Face Hub: %s", model_id)
        return model_id

    candidates = [explicit] if explicit else [
        MODELS / "distilbert-multilabel",
        Path("/kaggle/working/models/distilbert-multilabel"),
        Path("/kaggle/input") / "distilbert-multilabel",
    ]
    for base in candidates:
        if base is None or not base.exists():
            continue
        if (base / "config.json").exists():
            return base
        # An interrupted run can leave only checkpoint-N/ subdirectories.
        ckpts = sorted(
            (d for d in base.glob("checkpoint-*") if (d / "config.json").exists()),
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if ckpts:
            log.warning("no top-level model in %s; using latest checkpoint %s", base, ckpts[-1].name)
            return ckpts[-1]

    raise FileNotFoundError(
        "No saved model found. Train one with notebooks/train_model.ipynb or "
        "`python -m src.transformer`, or download the model directory from a Kaggle run's "
        "Output tab into models/distilbert-multilabel/, or pass --model-dir."
    )


def load_thresholds(path: Path | None = None) -> tuple[dict[str, float], bool]:
    """Load Phase 4 per-label thresholds. Returns (thresholds, were_they_calibrated)."""
    path = path or RESULTS / "thresholds_transformer.json"
    if path.exists():
        payload = json.loads(path.read_text())
        thresholds = payload.get("thresholds", payload)
        if all(label in thresholds for label in LABELS):
            # Round to 4dp: the search grid is np.linspace(0.05, 0.95, 19), whose values carry
            # float noise (0.55 arrives as 0.5499999999999999). Rounding is exact for a 0.05
            # grid and keeps the API's JSON readable.
            return {label: round(float(thresholds[label]), 4) for label in LABELS}, True
    log.warning("%s not found -- falling back to 0.5 for every label", path)
    return {label: 0.5 for label in LABELS}, False


@torch.no_grad()
def predict_probs(texts: list[str], model, tokenizer, device: str, max_length: int) -> np.ndarray:
    """Six independent sigmoid probabilities per text, shape (n, 6)."""
    model.eval()
    enc = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    # Sigmoid, NOT softmax: the six labels do not compete, so these need not sum to 1.
    return torch.sigmoid(model(**enc).logits).float().cpu().numpy()


def report(text: str, probs: np.ndarray, thresholds: dict[str, float], calibrated: bool) -> None:
    shown = text if len(text) <= 100 else text[:100] + "..."
    print(f'\n"{shown}"')
    print(f"  {'label':<14} {'prob':>6} {'thr':>6}  {'@0.5':<9} {'tuned':<9}")
    for i, label in enumerate(LABELS):
        p, t = float(probs[i]), thresholds[label]
        at_half = "FIRE" if p >= 0.5 else "-"
        at_tuned = "FIRE" if p >= t else "-"
        flag = "  <- differs" if (p >= 0.5) != (p >= t) else ""
        print(f"  {label:<14} {p:6.3f} {t:6.2f}  {at_half:<9} {at_tuned:<9}{flag}")

    fired = [l for i, l in enumerate(LABELS) if probs[i] >= thresholds[l]]
    kind = "tuned" if calibrated else "default 0.5"
    print(f"  => {', '.join(fired) if fired else 'no labels'}   ({kind} thresholds)")
    print(f"  (probabilities sum to {probs.sum():.2f} — unconstrained, because each label is independent)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify your own text with the fine-tuned model.")
    parser.add_argument("texts", nargs="*", help="one or more strings to classify")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--file", type=Path, help="classify each line of a text file")
    parser.add_argument("--interactive", action="store_true", help="type comments until Ctrl-D")
    args = parser.parse_args()

    setup_logging()
    set_seed()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = find_model_dir(args.model_dir)
    device = get_device()
    log.info("loading %s on %s", model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)

    thresholds, calibrated = load_thresholds(args.thresholds)
    log.info("thresholds: %s", {k: round(v, 2) for k, v in thresholds.items()})

    texts = list(args.texts)
    if args.file:
        texts += [l.strip() for l in args.file.read_text().splitlines() if l.strip()]

    if texts:
        probs = predict_probs(texts, model, tokenizer, device, args.max_length)
        for text, row in zip(texts, probs):
            report(text, row, thresholds, calibrated)

    if args.interactive or not texts:
        print("\nType a comment and press Enter (Ctrl-D or Ctrl-C to quit).")
        try:
            while True:
                line = input("\n> ").strip()
                if not line:
                    continue
                report(line, predict_probs([line], model, tokenizer, device, args.max_length)[0],
                       thresholds, calibrated)
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
