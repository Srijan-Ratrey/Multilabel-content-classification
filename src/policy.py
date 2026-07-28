"""Phase 4.2 — policy-conditioned classification (the differentiator).

Phase 3's model has six output heads. Head 3 means "threat" only because the
annotators put threats in column 3; the model never sees a definition of the word.
Two consequences follow, and both are exactly the problems real content-safety
systems have:

1. **Adding a policy means retraining.** A seventh head needs a new labeled corpus
   and a new fine-tune.
2. **Amending a policy means retraining.** If "threat" is narrowed to exclude
   quoted violence, the model cannot be told — the change has to be taught through
   thousands of relabeled examples.

This module builds the alternative. The policy definition becomes part of the
**input**:

    [CLS] <written policy definition> [SEP] <comment> [SEP]  ->  one sigmoid

The model is a cross-encoder over (policy, text) pairs with a **single** logit
answering "does this text violate *this* policy?". The same weights serve every
policy, and the policy is chosen at inference time by swapping the definition
string.

Two experiments:

* ``--variant all`` trains on all three written policies and is compared head-to-head
  with the fixed-label model on the same three labels, same test split, same harness.
* ``--variant heldout`` trains on `toxic` and `identity_hate` only, then is evaluated
  on `threat` — a policy whose definition it has **never seen**. A fixed six-head
  classifier cannot be evaluated this way at all: it has no mechanism for accepting a
  new policy. Any score above chance here is evidence the model learned to *read a
  rule and apply it* rather than to memorize a column index.

Note that ``truncation="only_second"`` is used deliberately: the policy definition
must survive intact, so only the comment is truncated. Truncating the definition
would corrupt the very thing being conditioned on.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.config import LABELS, MODELS, PROBS, RESULTS, SEED, get_device, set_seed, setup_logging
from src.data import load_all_splits
from src.evaluate import save_metrics
from src.policies import HELD_OUT_POLICY, POLICIES, definition, training_policies
from src.thresholds import GRID

log = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 2e-5
EPOCHS = 1
NEG_RATIO = 4  # negatives sampled per positive, per policy, for the training pairs


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, targets: np.ndarray | None):
        self.encodings = encodings
        self.targets = None if targets is None else targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor([self.targets[idx]])  # shape (1,) for the single logit
        return item


class PolicyTrainer(Trainer):
    """Single-logit binary cross-entropy — one yes/no question per (policy, text) pair."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.BCEWithLogitsLoss()(outputs.logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def sample_training_pairs(
    texts: list[str], Y: np.ndarray, policies: list[str], neg_ratio: int, seed: int
) -> tuple[list[str], list[str], np.ndarray]:
    """Build (definition, comment) training pairs.

    All positives are kept for each policy; negatives are subsampled at
    ``neg_ratio`` per positive. Without subsampling this would be 3 x 111,699 =
    335,097 pairs, dominated by the ~90% of comments that violate nothing. The
    resulting positive rate is much higher than the true base rate, which shifts the
    model's probability scale — handled downstream by tuning a threshold per policy
    on validation (where the base rate is left untouched).
    """
    rng = np.random.default_rng(seed)
    defs, comments, targets = [], [], []
    counts = {}

    for policy in policies:
        col = LABELS.index(policy)
        pos_idx = np.flatnonzero(Y[:, col] == 1)
        neg_pool = np.flatnonzero(Y[:, col] == 0)
        n_neg = min(len(neg_pool), neg_ratio * len(pos_idx))
        neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)

        idx = np.concatenate([pos_idx, neg_idx])
        d = definition(policy)
        defs.extend([d] * len(idx))
        comments.extend(texts[i] for i in idx)
        targets.extend(Y[idx, col].tolist())
        counts[policy] = {"positives": int(len(pos_idx)), "negatives": int(n_neg)}

    order = rng.permutation(len(defs))  # interleave policies so batches are mixed
    defs = [defs[i] for i in order]
    comments = [comments[i] for i in order]
    targets = np.asarray(targets, dtype=np.float32)[order]

    log.info("built %d training pairs: %s", len(defs), counts)
    return defs, comments, targets


def tokenize_pairs(tokenizer, defs: list[str], comments: list[str], max_length: int = MAX_LENGTH) -> dict:
    # only_second: never truncate the policy definition, only the comment.
    return tokenizer(
        defs, comments, truncation="only_second", padding="max_length", max_length=max_length
    )


@torch.no_grad()
def score_policy(model, tokenizer, policy: str, texts: list[str], device: str, max_length: int) -> np.ndarray:
    """P(violates `policy`) for every text, by conditioning on that policy's definition."""
    model.eval()
    d = definition(policy)
    enc = tokenize_pairs(tokenizer, [d] * len(texts), texts, max_length)
    ds = PairDataset(enc, None)
    loader = torch.utils.data.DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(torch.sigmoid(model(**batch).logits).float().cpu().numpy())
    return np.concatenate(out).ravel()


def tune_and_score(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> dict:
    """Tune a threshold on val, freeze it, then report test metrics. Same protocol as Phase 4.1."""
    f1s = [f1_score(y_val, p_val >= t, zero_division=0) for t in GRID]
    best_t = float(GRID[int(np.argmax(f1s))])

    pred = p_test >= best_t
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    return {
        "threshold": best_t,
        "val_f1_at_threshold": float(max(f1s)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(y_test, p_test)) if y_test.any() else float("nan"),
        "f1_at_0.5": float(f1_score(y_test, p_test >= 0.5, zero_division=0)),
        "test_positives": int(y_test.sum()),
        "n_predicted_positive": int(pred.sum()),
    }


def train_policy_model(
    train_policies: list[str], tag: str, epochs: float, max_length: int, neg_ratio: int
):
    setup_logging()
    set_seed()
    device = get_device()
    log.info("=== policy-conditioned model '%s' | training policies: %s ===", tag, train_policies)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = load_all_splits()
    train_texts, Y_train = splits["train"]

    defs, comments, targets = sample_training_pairs(
        train_texts, Y_train, train_policies, neg_ratio, SEED
    )
    train_ds = PairDataset(tokenize_pairs(tokenizer, defs, comments, max_length), targets)

    # num_labels=1: a single logit, because the question is binary given the policy.
    # The policy identity lives in the input text, not in the output dimension —
    # which is the entire point of the approach.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="multi_label_classification"
    )

    steps = -(-len(train_ds) // BATCH_SIZE)
    out_dir = MODELS / f"policy-{tag}"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=int(0.06 * steps * epochs),
        eval_strategy="no",  # no per-epoch eval: scoring happens per policy afterwards
        save_strategy="no",
        logging_steps=100,
        seed=SEED,
        data_seed=SEED,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = PolicyTrainer(
        model=model, args=training_args, train_dataset=train_ds, processing_class=tokenizer
    )
    t0 = time.perf_counter()
    result = trainer.train()
    minutes = (time.perf_counter() - t0) / 60
    log.info("trained '%s' in %.1f min (%d pairs, loss %.4f)", tag, minutes, len(train_ds), result.training_loss)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return model, tokenizer, splits, device, minutes, len(train_ds), float(result.training_loss)


def run_all_policies(args) -> None:
    """Experiment 1: train on every written policy, compare with the fixed-label model."""
    model, tokenizer, splits, device, minutes, n_pairs, loss = train_policy_model(
        list(POLICIES), "all", args.epochs, args.max_length, args.neg_ratio
    )

    Y_val, Y_test = splits["val"][1], splits["test"][1]
    per_policy = {}

    for policy in POLICIES:
        col = LABELS.index(policy)
        t0 = time.perf_counter()
        p_val = score_policy(model, tokenizer, policy, splits["val"][0], device, args.max_length)
        p_test = score_policy(model, tokenizer, policy, splits["test"][0], device, args.max_length)
        np.save(PROBS / f"policy_all_{policy}_val.npy", p_val.astype(np.float32))
        np.save(PROBS / f"policy_all_{policy}_test.npy", p_test.astype(np.float32))

        per_policy[policy] = tune_and_score(Y_val[:, col], p_val, Y_test[:, col], p_test)
        log.info(
            "  %-14s t=%.2f  test P=%.3f R=%.3f F1=%.4f AP=%.4f  (%.1f min)",
            policy, per_policy[policy]["threshold"], per_policy[policy]["precision"],
            per_policy[policy]["recall"], per_policy[policy]["f1"],
            per_policy[policy]["average_precision"], (time.perf_counter() - t0) / 60,
        )

    payload = {
        "phase": "4.2",
        "experiment": "policy-conditioned vs fixed-label",
        "model": f"{MODEL_NAME} cross-encoder over (policy_definition, comment) pairs, single sigmoid",
        "training_policies": list(POLICIES),
        "n_training_pairs": n_pairs,
        "negatives_per_positive": args.neg_ratio,
        "train_minutes": minutes,
        "train_loss": loss,
        "max_length": args.max_length,
        "policy_definitions": {p: POLICIES[p]["definition"] for p in POLICIES},
        "policy_conditioned": per_policy,
    }

    # Head-to-head against the fixed-label model, if Phase 4.1 has already run.
    fixed_path = RESULTS / "threshold_comparison_transformer.json"
    if fixed_path.exists():
        fixed = json.loads(fixed_path.read_text())["per_label"]
        comparison = {}
        for policy in POLICIES:
            pc, fx = per_policy[policy]["f1"], fixed[policy]["test_f1_tuned"]
            comparison[policy] = {
                "fixed_label_f1_tuned": fx,
                "policy_conditioned_f1_tuned": pc,
                "difference": pc - fx,
                "fixed_label_threshold": fixed[policy]["threshold"],
                "policy_conditioned_threshold": per_policy[policy]["threshold"],
            }
        payload["head_to_head"] = comparison
        mean_fixed = float(np.mean([c["fixed_label_f1_tuned"] for c in comparison.values()]))
        mean_pc = float(np.mean([c["policy_conditioned_f1_tuned"] for c in comparison.values()]))
        payload["head_to_head_summary"] = {
            "mean_f1_fixed_label": mean_fixed,
            "mean_f1_policy_conditioned": mean_pc,
            "difference": mean_pc - mean_fixed,
            "n_policies": len(comparison),
        }
        log.info("--- head to head (tuned F1 on test) ---")
        for policy, c in comparison.items():
            log.info(
                "  %-14s fixed-label %.4f | policy-conditioned %.4f | %+.4f",
                policy, c["fixed_label_f1_tuned"], c["policy_conditioned_f1_tuned"], c["difference"],
            )
        log.info("  mean over %d policies: %.4f vs %.4f (%+.4f)", len(comparison), mean_fixed, mean_pc, mean_pc - mean_fixed)
    else:
        log.warning("%s not found — run `python -m src.thresholds --model transformer` for the head-to-head", fixed_path)

    save_metrics(payload, RESULTS / "policy_conditioned.json")


def run_heldout(args) -> None:
    """Experiment 2: never train on the held-out policy, then ask about it anyway."""
    train_pols = training_policies()
    model, tokenizer, splits, device, minutes, n_pairs, loss = train_policy_model(
        train_pols, "heldout", args.epochs, args.max_length, args.neg_ratio
    )

    policy = HELD_OUT_POLICY
    col = LABELS.index(policy)
    Y_val, Y_test = splits["val"][1], splits["test"][1]

    p_val = score_policy(model, tokenizer, policy, splits["val"][0], device, args.max_length)
    p_test = score_policy(model, tokenizer, policy, splits["test"][0], device, args.max_length)
    np.save(PROBS / f"policy_heldout_{policy}_val.npy", p_val.astype(np.float32))
    np.save(PROBS / f"policy_heldout_{policy}_test.npy", p_test.astype(np.float32))

    zero_shot = tune_and_score(Y_val[:, col], p_val, Y_test[:, col], p_test)

    # Reference points that make the zero-shot number interpretable.
    base_rate = float(Y_test[:, col].mean())
    payload = {
        "phase": "4.2",
        "experiment": "held-out policy (zero-shot generalization to an unseen definition)",
        "held_out_policy": policy,
        "held_out_definition": definition(policy),
        "trained_only_on": train_pols,
        "n_training_pairs": n_pairs,
        "train_minutes": minutes,
        "train_loss": loss,
        "zero_shot": zero_shot,
        "reference_points": {
            "random_classifier_average_precision": base_rate,
            "test_positive_rate": base_rate,
            "note": (
                "average_precision above the test positive rate means the ranking carries real "
                "signal about a policy whose definition was never in training. A fixed six-head "
                "classifier has no way to be asked this question at all."
            ),
        },
    }

    # Supervised ceiling for the same policy, for honest context.
    for path, key in [
        (RESULTS / "threshold_comparison_transformer.json", "fixed_label_supervised_f1"),
        (RESULTS / "policy_conditioned.json", "policy_conditioned_supervised_f1"),
    ]:
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        if key.startswith("fixed"):
            payload["reference_points"][key] = blob["per_label"][policy]["test_f1_tuned"]
        else:
            payload["reference_points"][key] = blob["policy_conditioned"][policy]["f1"]

    save_metrics(payload, RESULTS / "policy_zeroshot.json")

    log.info("=" * 72)
    log.info("ZERO-SHOT on held-out policy '%s' (definition never seen in training):", policy)
    log.info(
        "  threshold %.2f | P=%.3f R=%.3f F1=%.4f | AP=%.4f vs %.4f for random",
        zero_shot["threshold"], zero_shot["precision"], zero_shot["recall"],
        zero_shot["f1"], zero_shot["average_precision"], base_rate,
    )
    for key in ("fixed_label_supervised_f1", "policy_conditioned_supervised_f1"):
        if key in payload["reference_points"]:
            log.info("  for reference, %s = %.4f", key, payload["reference_points"][key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.2: policy-conditioned classification.")
    parser.add_argument("--variant", choices=["all", "heldout"], default="all")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    args = parser.parse_args()

    setup_logging()
    if args.variant == "all":
        run_all_policies(args)
    else:
        run_heldout(args)


if __name__ == "__main__":
    main()
