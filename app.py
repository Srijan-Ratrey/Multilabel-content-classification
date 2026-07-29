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
import html as html_lib
import logging

import gradio as gr
import numpy as np

from src.config import LABELS, get_device, set_seed, setup_logging
from src.predict import find_model_dir, load_thresholds, predict_probs

log = logging.getLogger(__name__)

MAX_LENGTH = 256

# Colours are taken from the data-viz reference palette rather than invented, since a
# JS runtime was unavailable to run its validator. `critical` is a fixed status step
# (documented contrast 4.68 on the light surface, 3.62 on dark); the neutral is the
# secondary ink token (7.73 / 9.72). Both clear the 3:1 mark-vs-surface floor in both
# modes, verified numerically. State is additionally carried by the label name, the
# numeric value, and explicit "FLAGGED"/"below" text, so colour is never alone.
CSS = """
.viz-root {
  --surface-1: #fcfcfb; --surface-2: #f2f2ef;
  --text-primary: #0b0b0b; --text-secondary: #52514e;
  --mark-flagged: #d03b3b; --mark-below: #52514e; --rule: #d9d8d2;
  font-variant-numeric: tabular-nums;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-1: #1a1a19; --surface-2: #24241f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --mark-flagged: #d03b3b; --mark-below: #c3c2b7; --rule: #3a3a35;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-1: #1a1a19; --surface-2: #24241f;
  --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --mark-flagged: #d03b3b; --mark-below: #c3c2b7; --rule: #3a3a35;
}
.viz-root { background: var(--surface-1); padding: 14px 16px; border-radius: 10px; }
.viz-legend { display: flex; gap: 18px; align-items: center; margin-bottom: 12px;
  font-size: 12px; color: var(--text-secondary); flex-wrap: wrap; }
.viz-key { display: inline-flex; gap: 6px; align-items: center; }
.viz-swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.viz-row { display: grid; grid-template-columns: 112px 1fr 108px; gap: 10px;
  align-items: center; margin-bottom: 7px; }
.viz-name { font-size: 13px; color: var(--text-primary); text-align: right; }
/* the track is the 0..1 domain; 2px inset keeps a surface gap around the fill */
.viz-track { position: relative; height: 20px; background: var(--surface-2);
  border-radius: 4px; overflow: visible; }
.viz-fill { position: absolute; left: 0; top: 2px; bottom: 2px;
  border-radius: 0 4px 4px 0; }          /* 4px rounded data-end, anchored at the baseline */
.viz-thr { position: absolute; top: -3px; bottom: -3px; width: 2px;
  background: var(--text-secondary); }
.viz-thr::after { content: attr(data-t); position: absolute; top: -14px; left: -10px;
  font-size: 10px; color: var(--text-secondary); white-space: nowrap; }
.viz-val { font-size: 12px; color: var(--text-secondary); }
.viz-val b { color: var(--text-primary); font-size: 13px; }
.viz-verdict { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--rule);
  font-size: 13px; color: var(--text-primary); }
.viz-note { font-size: 11.5px; color: var(--text-secondary); margin-top: 6px; }
"""


def _bar_chart(probs: np.ndarray, thresholds: dict[str, float], mode_label: str) -> str:
    rows = []
    for i, label in enumerate(LABELS):
        p, t = float(probs[i]), float(thresholds[label])
        fired = p >= t
        colour = "var(--mark-flagged)" if fired else "var(--mark-below)"
        state = "FLAGGED" if fired else "below"
        rows.append(
            f'<div class="viz-row">'
            f'<div class="viz-name">{label}</div>'
            f'<div class="viz-track">'
            f'  <div class="viz-fill" style="width:{max(p, 0.004) * 100:.2f}%;background:{colour};"></div>'
            f'  <div class="viz-thr" style="left:{t * 100:.2f}%;" data-t="{t:.2f}"></div>'
            f"</div>"
            f'<div class="viz-val"><b>{p:.3f}</b> · {state}</div>'
            f"</div>"
        )

    fired = [l for i, l in enumerate(LABELS) if probs[i] >= thresholds[l]]
    verdict = ", ".join(fired) if fired else "no labels"
    return (
        f'<div class="viz-root">'
        f'<div class="viz-legend">'
        f'  <span class="viz-key"><span class="viz-swatch" style="background:var(--mark-flagged)"></span>'
        f"    at or above threshold (flagged)</span>"
        f'  <span class="viz-key"><span class="viz-swatch" style="background:var(--mark-below)"></span>'
        f"    below threshold</span>"
        f'  <span class="viz-key">| vertical tick = that label\'s threshold</span>'
        f"</div>"
        + "".join(rows)
        + f'<div class="viz-verdict"><b>Predicted:</b> {html_lib.escape(verdict)} '
        f'<span class="viz-note">({mode_label})</span></div>'
        f'<div class="viz-note">Probabilities sum to {probs.sum():.2f}. They are six independent '
        f"sigmoids, not a distribution — nothing constrains them to 1.0.</div>"
        f"</div>"
    )


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
        return _bar_chart(probs, thresholds, mode_label), table, note

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

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio UI for the policy classifier.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    setup_logging()
    set_seed()

    from pathlib import Path

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = find_model_dir(Path(args.model_dir) if args.model_dir else None)
    device = get_device()
    log.info("loading %s on %s", model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    tuned, calibrated = load_thresholds()
    log.info("thresholds: %s", {k: round(v, 2) for k, v in tuned.items()})

    demo = build_app(model, tokenizer, device, tuned, calibrated)
    demo.launch(css=CSS, theme=gr.themes.Soft(), server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
