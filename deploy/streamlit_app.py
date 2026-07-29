"""Streamlit Community Cloud deployment of the multi-label policy classifier.

Deployed from this repo with **Main file path** = `deploy/streamlit_app.py`. Community Cloud
searches the entrypoint's own directory for a requirements file before the repo root, so
`deploy/requirements.txt` is what gets installed — keeping jupyter, gradio, pandas and
scikit-learn out of the build.

The weights are not in this repo (256MB, gitignored); they are pulled from the public Hugging
Face model repo at startup, cached by @st.cache_resource so it happens once per container.

Chart, model loading, and thresholds are all reused from src/ rather than reimplemented, so
this UI cannot drift from the CLI or from the Gradio app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Community Cloud runs the entrypoint from the repo checkout, so the root (holding src/) must
# be importable — it is the parent of this file's directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import LABELS  # noqa: E402
from src.predict import load_thresholds, predict_probs  # noqa: E402
from src.viz import CSS, bar_chart  # noqa: E402

DEFAULT_MODEL_ID = "srijanratrey/distilbert-jigsaw-multilabel"
MAX_LENGTH = 256

st.set_page_config(page_title="Multi-label policy classifier", page_icon="🛡️", layout="wide")


def resolve_model_id() -> str:
    """Model repo id: Streamlit secret, then env var, then the public default."""
    try:
        if "MODEL_ID" in st.secrets:
            return str(st.secrets["MODEL_ID"])
    except Exception:
        pass  # no secrets.toml configured, which is fine — the default repo is public
    return os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)


@st.cache_resource(show_spinner="Loading the model (once per container)…")
def load_model(model_id: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)  # Community Cloud containers are small; more threads only contend
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    return model, tokenizer


MODEL_ID = resolve_model_id()
model, tokenizer = load_model(MODEL_ID)
tuned, calibrated = load_thresholds()

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

st.title("Multi-label content / policy classifier")
st.markdown(
    "Fine-tuned DistilBERT predicting **all** applicable labels — a comment can be toxic *and* "
    "obscene *and* insulting at once. Test **macro-F1 0.6836** with the calibrated thresholds.\n\n"
    "The model outputs six independent probabilities; **the decision is a separate choice.** "
    "Switch to *Uniform* below and drag the slider — the probabilities never move, but the "
    "predicted labels change. That is what this project is about."
)

with st.sidebar:
    st.subheader("Thresholds")
    mode = st.radio(
        "Mode",
        ["Calibrated per-label", "Uniform"],
        help="Calibrated thresholds were tuned on the validation split, never on test.",
    )
    uniform = st.slider("Uniform threshold", 0.05, 0.95, 0.50, 0.05, disabled=(mode != "Uniform"))

    st.divider()
    if calibrated:
        st.caption("Calibrated per label:")
        for label in LABELS:
            st.caption(f"`{label}` — **{tuned[label]:.2f}**")
    else:
        st.warning("Calibration file missing — using 0.5 for every label.")

    st.divider()
    st.caption(f"Model: `{MODEL_ID}`")
    st.caption(
        "`threat` occurs in 0.30% of the training data, so its calibrated threshold is 0.20 — "
        "a borderline threat gets flagged that 0.5 would wave through."
    )

examples = {
    "— pick an example —": "",
    "insulting": "you are an idiot and everyone hates you",
    "civil disagreement": "I disagree with your edit, please cite a reliable source",
    "threatening": "i will find you and kill you",
    "obscene + insulting": "shut up you moron",
    "friendly": "Thanks for fixing that typo, much appreciated!",
}
choice = st.selectbox("Examples", list(examples))
text = st.text_area("Comment", value=examples[choice], height=110, placeholder="Type or paste a comment…")

if not text.strip():
    st.info("Enter a comment above to classify it.")
    st.stop()

active = tuned if mode == "Calibrated per-label" else {l: float(uniform) for l in LABELS}
mode_label = (
    "per-label thresholds calibrated on validation"
    if mode == "Calibrated per-label"
    else f"one uniform threshold of {uniform:.2f} for every label"
)

probs = predict_probs([text], model, tokenizer, "cpu", MAX_LENGTH)[0]
st.markdown(bar_chart(probs, active, mode_label), unsafe_allow_html=True)

rows = [
    {
        "label": label,
        "probability": round(float(probs[i]), 4),
        "threshold": round(float(active[label]), 2),
        "fires?": "yes" if probs[i] >= active[label] else "no",
        "would fire @0.5": "yes" if probs[i] >= 0.5 else "no",
        "note": "differs" if (probs[i] >= active[label]) != (probs[i] >= 0.5) else "",
    }
    for i, label in enumerate(LABELS)
]
with st.expander("Table view", expanded=False):
    st.dataframe(rows, use_container_width=True, hide_index=True)

differing = [r["label"] for r in rows if r["note"]]
if differing:
    st.warning(
        f"**{len(differing)} label(s) decided differently than the default 0.5:** "
        + ", ".join(f"`{d}`" for d in differing)
    )
else:
    st.caption("Every label lands on the same side of both this threshold and the default 0.5.")

st.divider()
st.caption(
    "Accuracy is deliberately not reported: a model predicting nothing at all scores 89.8% "
    "exact-match accuracy on this data, at macro-F1 of exactly 0.0. **A demo, not a moderation "
    "system** — trained on 2018 Wikipedia comments, F1 is ~0.54 on `threat` and `severe_toxic`, "
    "and `identity_hate` recall is 0.535. "
    "[Source](https://github.com/Srijan-Ratrey/Multilabel-content-classification)"
)
