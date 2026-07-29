"""Gradio UI for the multi-label policy classifier.

    python app.py                # http://127.0.0.1:7860
    python app.py --share        # public link
    python app.py --port 8080

The UI is built around this project's central claim: the model produces six
independent probabilities, and the *decision* is a separate choice you make on top
of them. So the threshold is a live control. Drag the uniform slider and labels
appear and disappear while the probabilities never move — which is the whole
argument for Phase 4 in one interaction.

Model loading, threshold loading, and inference are reused from src/predict.py
rather than reimplemented.
"""

from __future__ import annotations

import argparse
import json
import logging

import gradio as gr
import numpy as np

from src.config import LABELS, RESULTS, get_device, set_seed, setup_logging
from src.viz import CSS, bar_chart
from src.predict import find_model_dir, load_thresholds, predict_probs

log = logging.getLogger(__name__)

MAX_LENGTH = 256
# Bounds for the programmatic API (the UI is one text at a time). A public toxicity endpoint
# attracts abuse and scraping; 5,000 chars is the longest comment in the Jigsaw training data,
# so the cap is not a practical limitation.
MAX_TEXTS_PER_REQUEST = 32
MAX_CHARS_PER_TEXT = 5_000

def build_app(model, tokenizer, device: str, tuned: dict[str, float], calibrated: bool):
    def classify(text: str, mode: str, uniform: float):
        if not text or not text.strip():
            return "", None, ""
        probs = predict_probs([text], model, tokenizer, device, MAX_LENGTH)[0]

        use_tuned = mode.startswith("Calibrated")
        thresholds = tuned if use_tuned else {l: float(uniform) for l in LABELS}
        mode_label = (
            "per-label thresholds calibrated on validation (Phase 4)"
            if use_tuned
            else f"one uniform threshold of {uniform:.2f} for every label"
        )

        # Table view — the accessible equivalent of the chart above it.
        table = [
            [
                label,
                round(float(probs[i]), 4),
                round(float(thresholds[label]), 2),
                "yes" if probs[i] >= thresholds[label] else "no",
                "yes" if probs[i] >= 0.5 else "no",
                "differs" if (probs[i] >= thresholds[label]) != (probs[i] >= 0.5) else "",
            ]
            for i, label in enumerate(LABELS)
        ]

        differing = [r[0] for r in table if r[5]]
        note = (
            f"**{len(differing)} label(s) decided differently than the default 0.5:** "
            + ", ".join(f"`{d}`" for d in differing)
            if differing
            else "_Every label lands on the same side of both this threshold and the default 0.5._"
        )
        return bar_chart(probs, thresholds, mode_label), table, note

    # Gradio 6 moved `css` and `theme` from the Blocks constructor to launch(); passing them
    # here is accepted with a warning but the stylesheet does not reliably apply, which would
    # leave the chart unstyled. They are supplied in launch_app() instead.
    with gr.Blocks(title="Multi-label policy classifier") as demo:
        gr.Markdown(
            "# Multi-label content / policy classifier\n"
            "Fine-tuned DistilBERT predicting **all** applicable labels — a comment can be "
            "toxic *and* obscene *and* insulting at once. Test macro-F1 **0.6836** with the "
            "calibrated thresholds below.\n\n"
            "The model outputs six independent probabilities; **the decision is a separate "
            "choice.** Switch to *Uniform* and drag the slider — the probabilities never move, "
            "but the predicted labels change. That is what Phase 4 of this project is about."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Comment", placeholder="Type or paste a comment…", lines=4, autofocus=True
                )
                with gr.Row():
                    mode = gr.Radio(
                        [
                            "Calibrated per-label (recommended)",
                            "Uniform threshold",
                        ],
                        value="Calibrated per-label (recommended)",
                        label="Thresholds",
                        scale=2,
                    )
                    uniform = gr.Slider(
                        0.05, 0.95, value=0.5, step=0.05,
                        label="Uniform threshold (used only in Uniform mode)", scale=2,
                    )
                run = gr.Button("Classify", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown(
                    "**Calibrated thresholds**\n\n"
                    + "\n".join(f"- `{l}` — **{tuned[l]:.2f}**" for l in LABELS)
                    + (
                        "\n\nTuned per label on the validation split, never on test."
                        if calibrated
                        else "\n\n⚠️ Calibration file missing — showing 0.5 defaults."
                    )
                )

        chart = gr.HTML(label="Per-label probabilities")
        note = gr.Markdown()
        table = gr.Dataframe(
            headers=["label", "probability", "threshold", "fires?", "would fire @0.5", "note"],
            label="Table view",
            wrap=True,
        )

        gr.Examples(
            examples=[
                ["you are an idiot and everyone hates you"],
                ["I disagree with your edit, please cite a reliable source"],
                ["i will find you and kill you"],
                ["shut up you moron"],
                ["Thanks for fixing that typo, much appreciated!"],
            ],
            inputs=[text],
        )

        gr.Markdown(
            "---\n"
            "`threat` occurs in 0.30% of the training data, so its calibrated threshold is low "
            "(**0.20**) — a borderline threat gets flagged that 0.5 would wave through. `obscene` "
            "goes the other way at **0.65**. Accuracy is deliberately not reported: a model "
            "predicting nothing at all scores 89.8% exact-match accuracy on this data, at "
            "macro-F1 of exactly 0.0."
        )

        for trigger in (run.click, text.submit):
            trigger(classify, [text, mode, uniform], [chart, table, note])
        mode.change(classify, [text, mode, uniform], [chart, table, note])
        uniform.change(classify, [text, mode, uniform], [chart, table, note])

        # --- programmatic API -------------------------------------------------
        # gr.api exposes these as documented endpoints without any UI component, which is how
        # a free Gradio Space gets an API at all (mounting a FastAPI app would need the paid
        # Docker SDK). Contract: POST /gradio_api/call/<api_name> returns {"event_id": ...},
        # then GET /gradio_api/call/<api_name>/<event_id> streams the result. The
        # gradio_client library hides that two-step from Python callers.
        def predict_api(texts: list[str], thresholds: str = "tuned") -> list[dict]:
            """Classify comments. thresholds: 'tuned' (calibrated) or 'default' (0.5)."""
            if not isinstance(texts, list):
                raise gr.Error("`texts` must be a list of strings")
            if not 1 <= len(texts) <= MAX_TEXTS_PER_REQUEST:
                raise gr.Error(f"send between 1 and {MAX_TEXTS_PER_REQUEST} texts")
            for t in texts:
                if not isinstance(t, str):
                    raise gr.Error("every item in `texts` must be a string")
                if len(t) > MAX_CHARS_PER_TEXT:
                    raise gr.Error(f"each text must be under {MAX_CHARS_PER_TEXT} characters")
            if thresholds not in ("tuned", "default"):
                raise gr.Error("`thresholds` must be 'tuned' or 'default'")

            active = tuned if thresholds == "tuned" else {l: 0.5 for l in LABELS}
            probs = predict_probs(texts, model, tokenizer, device, MAX_LENGTH)
            return [
                {
                    "labels": [l for i, l in enumerate(LABELS) if row[i] >= active[l]],
                    "probabilities": {l: round(float(row[i]), 6) for i, l in enumerate(LABELS)},
                    "thresholds_applied": {l: float(active[l]) for l in LABELS},
                }
                for row in np.atleast_2d(probs)
            ]

        def info_api() -> dict:
            """Model card: labels, calibrated thresholds, metrics, and caveats."""
            payload = {
                "labels": LABELS,
                "thresholds": {l: float(tuned[l]) for l in LABELS},
                "calibrated": calibrated,
                "limits": {
                    "max_texts_per_request": MAX_TEXTS_PER_REQUEST,
                    "max_chars_per_text": MAX_CHARS_PER_TEXT,
                    "max_tokens": MAX_LENGTH,
                },
                "caveats": [
                    "Trained on 2018 English Wikipedia talk-page comments; other domains degrade.",
                    "Rare labels are weak: F1 ~0.54 on threat and severe_toxic.",
                    "identity_hate recall is 0.535 -- it misses about half of them.",
                    "A demo, not a moderation system.",
                ],
            }
            metrics_file = RESULTS / "transformer_tuned_metrics.json"
            if metrics_file.exists():
                m = json.loads(metrics_file.read_text())
                payload["metrics"] = {
                    "split": "held-out test",
                    "n_samples": m["n_samples"],
                    "macro_f1_headline": m["macro_f1"],
                    "micro_f1": m["micro_f1"],
                    "macro_average_precision": m["macro_average_precision"],
                    "per_label": m["per_label"],
                    "all_zeros_control": m["all_zeros_baseline"],
                }
            return payload

        gr.api(predict_api, api_name="predict",
               api_description="Classify comments and return probabilities plus thresholded labels.")
        gr.api(info_api, api_name="info",
               api_description="Model card: thresholds, metrics, and limitations.")

    return demo


def load_runtime(model_dir: str | None = None) -> dict:
    """Load model, tokenizer, device and thresholds once.

    The UI and the gr.api endpoints share this single load, so they can never end up serving
    different weights or different thresholds.
    """
    from pathlib import Path

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = find_model_dir(Path(model_dir) if model_dir else None)
    device = get_device()
    log.info("loading %s on %s", source, device)

    tokenizer = AutoTokenizer.from_pretrained(str(source))
    model = AutoModelForSequenceClassification.from_pretrained(str(source)).to(device)
    model.eval()

    tuned, calibrated = load_thresholds()
    log.info("thresholds: %s", {k: round(v, 2) for k, v in tuned.items()})

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "thresholds": tuned,
        "calibrated": calibrated,
        "source": str(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio UI for the policy classifier.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    setup_logging()
    set_seed()

    rt = load_runtime(args.model_dir)
    demo = build_app(rt["model"], rt["tokenizer"], rt["device"], rt["thresholds"], rt["calibrated"])
    demo.launch(css=CSS, theme=gr.themes.Soft(), server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
