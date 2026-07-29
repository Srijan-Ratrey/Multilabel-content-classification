"""Push the fine-tuned model to a Hugging Face model repo, for the Space to load at startup.

    hf auth login                                   # once; stores ~/.cache/huggingface/token
    python scripts/push_to_hub.py --repo <user>/distilbert-jigsaw-multilabel --dry-run
    python scripts/push_to_hub.py --repo <user>/distilbert-jigsaw-multilabel

Why the model lives on the Hub rather than in this git repo: the weights are 256MB and
`.gitignore` excludes `models/`. Keeping them on the Hub also means the Space image stays small
and the model can be replaced without a rebuild.

Make the repo **public** (the default here) and the deployed Space needs no token at all. Only
a private repo requires an `HF_TOKEN` secret on the Space.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Python puts this script's own directory on sys.path, not the repo root, so `src` would not
# be importable when invoked as `python scripts/push_to_hub.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS, RESULTS, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def build_model_card(repo: str, metrics_file: Path) -> str:
    """A model card carrying the real metrics, read from results/ rather than retyped."""
    lines = [
        "---",
        "license: mit",
        "language: en",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        "tags:",
        "  - multi-label-classification",
        "  - content-moderation",
        "  - distilbert",
        "---",
        "",
        "# DistilBERT — multi-label content / policy classification",
        "",
        "Fine-tuned `distilbert-base-uncased` predicting **all** applicable labels for a comment:",
        "`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.",
        "",
        "Six **independent** sigmoid outputs trained with `BCEWithLogitsLoss` — not softmax — so",
        "any combination of labels can fire and the probabilities do not sum to 1.",
        "",
    ]

    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        lines += [
            f"## Metrics (held-out test split, n={m['n_samples']:,})",
            "",
            f"**Macro-F1: {m['macro_f1']:.4f}**  ·  micro-F1: {m['micro_f1']:.4f}  ·  "
            f"macro average precision: {m['macro_average_precision']:.4f}",
            "",
            "| label | threshold | precision | recall | F1 | AP | support |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, d in m["per_label"].items():
            lines.append(
                f"| `{label}` | {d['threshold']:.2f} | {d['precision']:.3f} | {d['recall']:.3f} | "
                f"**{d['f1']:.3f}** | {d['average_precision']:.3f} | {d['support']:,} |"
            )
        az = m["all_zeros_baseline"]
        lines += [
            "",
            "### Why not accuracy",
            "",
            f"A model that predicts *nothing* scores **{az['subset_accuracy']:.2%} exact-match "
            f"accuracy** and **{az['mean_per_label_accuracy']:.2%} mean per-label accuracy** on this "
            f"data, at macro-F1 of exactly **{az['macro_f1']:.1f}**. Macro-F1 is the headline for "
            "that reason.",
        ]

    lines += [
        "",
        "## Thresholds",
        "",
        "The per-label thresholds in the table were calibrated on the **validation** split and",
        "applied unchanged to test. `thresholds_transformer.json` in this repo holds them.",
        "Using them instead of a flat 0.5 is worth **+0.021 macro-F1** with no retraining.",
        "",
        "## Limitations",
        "",
        "- Trained on 2018 English Wikipedia talk-page comments; other domains will degrade.",
        "- Rare labels are weak: F1 ~0.54 on `threat` and `severe_toxic`; `identity_hate` recall 0.535.",
        "- `threat`'s threshold rests on only 72 validation positives, so it is a noisy estimate.",
        "- Inputs truncated at 256 tokens (~6.9% of training comments were longer).",
        "- **A demo, not a moderation system.** Do not make consequential decisions about people with it.",
        "",
        "Code, full write-up, and a live demo:",
        "https://github.com/Srijan-Ratrey/Multilabel-content-classification",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the model to a Hugging Face model repo.")
    parser.add_argument("--repo", required=True, help="e.g. your-username/distilbert-jigsaw-multilabel")
    parser.add_argument("--model-dir", type=Path, default=MODELS / "distilbert-multilabel")
    parser.add_argument("--private", action="store_true",
                        help="private repo; the Space then needs an HF_TOKEN secret")
    parser.add_argument("--dry-run", action="store_true", help="show what would be uploaded, then stop")
    args = parser.parse_args()

    setup_logging()

    missing = [f for f in REQUIRED if not (args.model_dir / f).exists()]
    if missing:
        sys.exit(f"{args.model_dir} is missing {missing}. Train the model or download it first.")

    files = sorted(p for p in args.model_dir.iterdir() if p.is_file() and p.name != ".DS_Store")
    total = sum(p.stat().st_size for p in files)
    log.info("uploading %d files (%.0f MB) from %s -> %s",
             len(files), total / 1e6, args.model_dir, args.repo)
    for p in files:
        log.info("    %-24s %7.1f MB", p.name, p.stat().st_size / 1e6)

    thresholds = RESULTS / "thresholds_transformer.json"
    card = build_model_card(args.repo, RESULTS / "transformer_tuned_metrics.json")

    if args.dry_run:
        log.info("also uploading: thresholds_transformer.json, README.md (model card)")
        print("\n--- model card preview ---\n")
        print(card)
        log.info("dry run: nothing uploaded. Re-run without --dry-run to push.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception:
        sys.exit("Not authenticated. Run `hf auth login` (or set HF_TOKEN) and try again.")
    log.info("authenticated as %s", whoami.get("name", "?"))

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(args.model_dir),
        ignore_patterns=[".DS_Store", "*.pt", "optimizer*", "scheduler*", "rng_state*"],
    )
    if thresholds.exists():
        api.upload_file(path_or_fileobj=str(thresholds), path_in_repo="thresholds_transformer.json",
                        repo_id=args.repo)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.repo)

    log.info("done: https://huggingface.co/%s", args.repo)
    log.info("next: create a Docker Space and set the Space variable MODEL_ID=%s", args.repo)


if __name__ == "__main__":
    main()
