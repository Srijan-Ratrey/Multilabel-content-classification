"""Assemble and upload the Hugging Face Space (Gradio SDK, free tier).

    hf auth login
    python scripts/push_space.py --space <user>/multilabel-content-classification \
                                 --model-id <user>/distilbert-jigsaw-multilabel --dry-run
    python scripts/push_space.py --space <user>/multilabel-content-classification \
                                 --model-id <user>/distilbert-jigsaw-multilabel

A Space is its own git repo with a specific layout: `README.md` at the root carries the YAML
front matter that configures it, and `requirements.txt` at the root drives the build. This
script stages the right files under those names so the layout cannot be got wrong by hand.

Gradio SDK rather than Docker: Docker Spaces are a paid tier. The trade-off is that the API
becomes Gradio's two-step call protocol instead of a single REST POST — see spaces/README.md.
The model weights are NOT uploaded here; the Space pulls them from the model repo at startup
via the MODEL_ID variable, which this script sets for you.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

# (source relative to repo root, destination inside the Space repo)
FILES: list[tuple[str, str]] = [
    ("spaces/README.md", "README.md"),                # front matter configures the Space
    ("spaces/requirements.txt", "requirements.txt"),  # drives the Space build
    ("app.py", "app.py"),                             # app_file in the front matter
    ("src/__init__.py", "src/__init__.py"),
    ("src/config.py", "src/config.py"),
    ("src/predict.py", "src/predict.py"),
    ("results/thresholds_transformer.json", "results/thresholds_transformer.json"),
    ("results/transformer_tuned_metrics.json", "results/transformer_tuned_metrics.json"),
]


def stage(dest: Path) -> list[str]:
    staged = []
    for src_rel, dst_rel in FILES:
        src = ROOT / src_rel
        if not src.exists():
            sys.exit(f"missing required file: {src}")
        dst = dest / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        staged.append(f"{dst_rel:<45} {src.stat().st_size / 1024:8.1f} KB")
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the Gradio Space.")
    parser.add_argument("--space", required=True, help="e.g. your-username/multilabel-content-classification")
    parser.add_argument("--model-id", required=True, help="the model repo the Space should load")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        staged = stage(staging)

        log.info("staging %d files for Space %s", len(staged), args.space)
        for line in staged:
            log.info("    %s", line)
        log.info("MODEL_ID will be set to %s", args.model_id)
        log.info("note: the 256MB weights are NOT uploaded here -- the Space pulls them from that repo")

        if args.dry_run:
            log.info("dry run: nothing uploaded. Re-run without --dry-run to push.")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.whoami()
        except Exception:
            sys.exit("Not authenticated. Run `hf auth login` (or set HF_TOKEN) and try again.")

        api.create_repo(
            args.space, repo_type="space", space_sdk="gradio",
            private=args.private, exist_ok=True,
        )
        # Set before uploading, so the first build already has it and does not fail once.
        api.add_space_variable(args.space, "MODEL_ID", args.model_id)
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(staging))

        log.info("done: https://huggingface.co/spaces/%s", args.space)
        log.info("the first build takes several minutes (torch download); watch the Logs tab")


if __name__ == "__main__":
    main()
