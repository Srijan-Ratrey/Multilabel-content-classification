"""Phase 3 — fine-tune DistilBERT for multi-label classification.

============================================================================
THE CORE MULTI-LABEL CONCEPT: SIGMOID + BCE, NOT SOFTMAX + CROSS-ENTROPY
============================================================================

The output layer produces ``num_labels`` logits, and each one is passed through
its **own independent sigmoid**. The loss is ``BCEWithLogitsLoss`` — binary
cross-entropy applied per label and then averaged.

It is *not* softmax with categorical cross-entropy, and the distinction is the
whole point of the phase:

* **Softmax couples the logits.** It exponentiates and divides by the sum over all
  classes, so the outputs are constrained to sum to 1. Raising one class's
  probability necessarily lowers every other's. That encodes "exactly one of these
  is correct" — a single-winner competition between labels.
* **Sigmoid treats each logit separately.** sigma(z_i) = 1 / (1 + e^-z_i) depends
  only on z_i. The six outputs are free to be independently high or low, so they
  can sum to 0.0 (a clean comment — 89.8% of this dataset) or to 6.0 (a comment
  that is toxic AND severe_toxic AND obscene AND threatening AND insulting AND
  identity-hateful — 31 rows in Jigsaw are exactly that).

Concretely: a comment labeled `toxic + obscene + insult` is a *single* training
example with three simultaneously-correct answers. Softmax cannot represent that
target — it would have to split probability mass between the three, driving each
toward 0.33 and never confidently predicting any of them, while also being unable
to represent the all-zero case at all. Sigmoid + BCE asks six independent
yes/no questions and lets all three fire at once.

This is the loss-level reason multi-label is a different problem from multi-class,
not just a different output shape.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from src.config import (
    LABELS,
    MODELS,
    N_LABELS,
    PROBS,
    RESULTS,
    SEED,
    get_device,
    set_seed,
    setup_logging,
)
from src.data import load_all_splits
from src.evaluate import evaluate, log_report, plot_pr_curves, save_metrics

log = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 2e-5
EPOCHS = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
OUTPUT_DIR = MODELS / "distilbert-multilabel"


class MultiLabelDataset(torch.utils.data.Dataset):
    """Pre-tokenized, **unpadded** dataset. Padding is applied per batch by the
    collator (see MultiLabelCollator) rather than to a fixed 256 tokens here.

    This matters a lot on this dataset: the median comment is 52 tokens but
    max_length is 256, so padding everything to 256 would spend roughly three
    quarters of the compute on padding tokens. Measured: 1.25 s/step padded to a
    fixed 256 vs ~0.35 s/step with dynamic padding plus length-grouped batching —
    a ~3.5x difference, or 2h25m vs ~45min for this run.

    Targets are float32 because BCEWithLogitsLoss compares against continuous
    targets, unlike CrossEntropyLoss which wants integer class indices — a small
    but telling API difference between multi-label and multi-class.
    """

    def __init__(self, encodings: dict, labels: np.ndarray | None):
        self.encodings = encodings
        self.labels = None if labels is None else labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}  # plain lists, variable length
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class MultiLabelCollator:
    """Pad each batch to its own longest sequence, and stack the (batch, 6) targets.

    Written explicitly rather than using DataCollatorWithPadding because the label
    field here is a fixed-width float vector per example, not a single class index,
    and it must not go through the tokenizer's padding logic.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        features = [dict(f) for f in features]
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        if labels is not None:
            batch["labels"] = torch.tensor(np.asarray(labels), dtype=torch.float32)
        return batch


class MultiLabelTrainer(Trainer):
    """Trainer with an explicit per-label binary cross-entropy loss.

    ``problem_type="multi_label_classification"`` on the config already makes
    HuggingFace use BCEWithLogitsLoss internally. The override is kept anyway
    because this loss *is* the lesson of the phase — it should be visible in the
    code rather than inferred from a config string.

    Also groups same-length examples into batches (see ``_get_train_sampler``).
    """

    def _get_train_sampler(self, train_dataset=None):
        """Batch examples of similar length together.

        With dynamic padding this is what actually buys the speedup: a random batch of
        32 almost always contains one long comment (P(any > 200 tokens) ~ 97%), so
        without grouping nearly every batch would still pad out to ~256 anyway.

        Implemented by overriding the sampler because the ``group_by_length``
        TrainingArguments flag was removed in transformers v5; LengthGroupedSampler
        itself still exists, and passing lengths explicitly is clearer than letting
        the Trainer infer them.
        """
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None

        lengths = [len(ids) for ids in dataset.encodings["input_ids"]]
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            lengths=lengths,
            generator=generator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, num_labels) raw scores, one per label

        # BCEWithLogitsLoss = sigmoid + binary cross-entropy, fused for numerical
        # stability. Applied elementwise: each of the 6 logits is scored against its
        # own 0/1 target independently. No normalization across the label dimension,
        # which is exactly what distinguishes this from softmax + cross-entropy.
        loss = nn.BCEWithLogitsLoss()(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


def tokenize(tokenizer, texts: list[str], max_length: int = MAX_LENGTH) -> dict:
    """Truncate to max_length but do NOT pad — the collator pads per batch."""
    return tokenizer(texts, truncation=True, padding=False, max_length=max_length)


def token_stats(tokenizer, texts: list[str]) -> dict:
    """Token-length distribution and the truncation cost at each candidate max_length.

    Worth measuring rather than assuming: choosing max_length is choosing how much
    of the input to silently discard, and Jigsaw's length distribution has a long tail.
    """
    lengths = np.array([len(tokenizer.encode(t, truncation=False)) for t in texts])
    stats = {
        "n_texts": int(len(lengths)),
        "percentiles": {f"p{p}": float(np.percentile(lengths, p)) for p in (50, 75, 90, 95, 99)},
        "mean": float(lengths.mean()),
        "max": int(lengths.max()),
        "truncation_cost": {
            str(limit): {
                "examples_truncated": int((lengths > limit).sum()),
                "fraction_truncated": float((lengths > limit).mean()),
                "tokens_lost_fraction": float(np.clip(lengths - limit, 0, None).sum() / lengths.sum()),
            }
            for limit in (128, 256, 512)
        },
    }
    return stats


def build_compute_metrics():
    """Per-epoch validation metrics through the Phase 2 harness, so model selection
    uses macro-F1 — the project's headline — rather than loss or accuracy."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        probs = torch.sigmoid(torch.as_tensor(logits)).numpy()
        m = evaluate(labels.astype(np.int8), probs, 0.5, name="val @0.5")
        return {
            "macro_f1": m["macro_f1"],
            "micro_f1": m["micro_f1"],
            "macro_ap": m["macro_average_precision"],
            "f1_threat": m["per_label"]["threat"]["f1"],
            "f1_identity_hate": m["per_label"]["identity_hate"]["f1"],
        }

    return compute_metrics


@torch.no_grad()
def predict_probs(model, dataset, collator, device: str, batch_size: int = EVAL_BATCH_SIZE) -> np.ndarray:
    """Run inference and return sigmoid probabilities, shape (n, num_labels).

    Examples are processed in length-sorted order so each batch pads to roughly its
    own length instead of to the longest example in the whole split, then the results
    are scattered back to the original row order. Same outputs, several times faster.
    """
    model.eval()
    lengths = np.array([len(dataset.encodings["input_ids"][i]) for i in range(len(dataset))])
    order = np.argsort(lengths, kind="stable")

    out = np.empty((len(dataset), N_LABELS), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = collator([dataset[int(i)] for i in idx])
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        # Sigmoid per logit — independent probabilities, NOT a distribution over labels.
        out[idx] = torch.sigmoid(logits).float().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: fine-tune DistilBERT (multi-label).")
    parser.add_argument("--token-stats", action="store_true", help="only measure token lengths, then exit")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--limit-train", type=int, default=None, help="debug: cap training rows")
    args = parser.parse_args()

    setup_logging()
    set_seed()
    device = get_device()
    log.info("device=%s | model=%s | max_length=%d", device, MODEL_NAME, args.max_length)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = load_all_splits()

    if args.token_stats:
        stats = {split: token_stats(tokenizer, texts) for split, (texts, _) in splits.items()}
        save_metrics(stats, RESULTS / "token_stats.json")
        for split, s in stats.items():
            log.info(
                "%-5s p50=%.0f p90=%.0f p95=%.0f p99=%.0f max=%d",
                split, s["percentiles"]["p50"], s["percentiles"]["p90"],
                s["percentiles"]["p95"], s["percentiles"]["p99"], s["max"],
            )
            for limit, c in s["truncation_cost"].items():
                log.info(
                    "      max_length=%-4s truncates %6d examples (%.2f%%), losing %.2f%% of all tokens",
                    limit, c["examples_truncated"], 100 * c["fraction_truncated"], 100 * c["tokens_lost_fraction"],
                )
        return

    train_texts, Y_train = splits["train"]
    if args.limit_train:
        train_texts, Y_train = train_texts[: args.limit_train], Y_train[: args.limit_train]

    t0 = time.perf_counter()
    datasets = {
        "train": MultiLabelDataset(tokenize(tokenizer, train_texts, args.max_length), Y_train),
        "val": MultiLabelDataset(tokenize(tokenizer, splits["val"][0], args.max_length), splits["val"][1]),
        "test": MultiLabelDataset(tokenize(tokenizer, splits["test"][0], args.max_length), splits["test"][1]),
    }
    log.info("tokenized %d train / %d val / %d test in %.1fs",
             len(datasets["train"]), len(datasets["val"]), len(datasets["test"]), time.perf_counter() - t0)

    # num_labels=6 with problem_type multi_label -> 6 independent sigmoid outputs.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=N_LABELS,
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)},
    )

    # warmup_ratio is deprecated in transformers v5, so derive the step count ourselves.
    steps_per_epoch = -(-len(datasets["train"]) // args.batch_size)  # ceil division
    warmup_steps = int(WARMUP_RATIO * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # select on the headline metric, not loss
        greater_is_better=True,
        logging_steps=100,
        seed=SEED,
        data_seed=SEED,
        # fp16/bf16 are left off: the MPS backend does not support HF Trainer's
        # mixed-precision path, so this runs fp32 on Apple Silicon.
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,  # MPS + forked workers is unreliable
        report_to=[],
    )

    collator = MultiLabelCollator(tokenizer)
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    train_minutes = (time.perf_counter() - t0) / 60
    log.info("training finished in %.1f min", train_minutes)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save probabilities for val and test — Phase 4 calibrates thresholds on val
    # and applies them to test, so both must be persisted.
    all_metrics = {
        "phase": 3,
        "model": MODEL_NAME,
        "loss": "BCEWithLogitsLoss (independent sigmoid per label, not softmax)",
        "hyperparameters": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
            "precision": "fp32 (MPS does not support Trainer mixed precision)",
            "seed": SEED,
        },
        "device": device,
        "train_minutes": train_minutes,
        "train_loss": float(train_result.training_loss),
        "n_train": len(datasets["train"]),
        "splits": {},
    }

    for split in ("val", "test"):
        probs = predict_probs(model, datasets[split], collator, device)
        np.save(PROBS / f"transformer_{split}.npy", probs.astype(np.float32))
        log.info("saved %s probabilities -> %s", split, PROBS / f"transformer_{split}.npy")

        m = evaluate(splits[split][1], probs, 0.5, name=f"DistilBERT @0.5 ({split})")
        m["split"] = split
        log_report(m)
        all_metrics["splits"][split] = m

    save_metrics(all_metrics, RESULTS / "transformer_metrics.json")

    import matplotlib

    matplotlib.use("Agg")
    plot_pr_curves(
        splits["test"][1],
        np.load(PROBS / "transformer_test.npy"),
        RESULTS / "pr_transformer.png",
        title="Per-label PR curves — DistilBERT, test split",
    )


if __name__ == "__main__":
    main()
