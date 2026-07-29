"""Production entry point: REST API with the Gradio UI mounted at the root.

    uvicorn serve:app --host 0.0.0.0 --port 7860

    GET  /            the Gradio UI
    GET  /health      liveness + which model and thresholds are loaded
    GET  /info        model card as JSON (metrics read from results/, never hardcoded)
    POST /predict     batch classification
    GET  /docs        OpenAPI documentation

The model is resolved by `MODEL_ID` (a Hugging Face repo id) when set, otherwise from a local
directory — see src.predict.find_model_dir. The UI and the API share one loaded model via
app.load_runtime(), so they can never disagree.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

import gradio as gr
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app import CSS, build_app, load_runtime
from src.config import LABELS, RESULTS, setup_logging
from src.predict import predict_probs

log = logging.getLogger(__name__)

MAX_LENGTH = 256
# A public toxicity demo attracts abuse and scraping, so the request surface is bounded.
# 5,000 chars is the longest comment in the Jigsaw training data, so it is not a limitation
# in practice. 32 sits at the point where batched throughput has already plateaued.
MAX_TEXTS_PER_REQUEST = 32
MAX_CHARS_PER_TEXT = 5_000

setup_logging()
RUNTIME = load_runtime()
log.info("serve.py ready | source=%s | device=%s", RUNTIME["source"], RUNTIME["device"])


# --------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=MAX_TEXTS_PER_REQUEST,
        description=f"1-{MAX_TEXTS_PER_REQUEST} comments to classify",
        examples=[["you are an idiot", "thanks for the fix!"]],
    )
    thresholds: Literal["tuned", "default"] | dict[str, float] = Field(
        "tuned",
        description=(
            "'tuned' = per-label thresholds calibrated on validation (recommended); "
            "'default' = 0.5 for every label; or an explicit {label: threshold} mapping."
        ),
    )

    @field_validator("texts")
    @classmethod
    def _check_lengths(cls, v: list[str]) -> list[str]:
        for i, t in enumerate(v):
            if len(t) > MAX_CHARS_PER_TEXT:
                raise ValueError(f"texts[{i}] is {len(t)} chars; limit is {MAX_CHARS_PER_TEXT}")
        return v

    @field_validator("thresholds")
    @classmethod
    def _check_thresholds(cls, v):
        if isinstance(v, dict):
            unknown = set(v) - set(LABELS)
            if unknown:
                raise ValueError(f"unknown labels: {sorted(unknown)}; valid labels are {LABELS}")
            for label, t in v.items():
                if not 0.0 <= float(t) <= 1.0:
                    raise ValueError(f"threshold for '{label}' must be in [0, 1], got {t}")
        return v


class Prediction(BaseModel):
    labels: list[str] = Field(..., description="labels at or above their threshold")
    probabilities: dict[str, float]
    thresholds_applied: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_source: str
    threshold_mode: str


# ------------------------------------------------------------------------ app
api = FastAPI(
    title="Multi-label content / policy classifier",
    version="1.0.0",
    description=(
        "Fine-tuned DistilBERT predicting all applicable policy labels for a comment "
        "(toxic, severe_toxic, obscene, threat, insult, identity_hate).\n\n"
        "Six **independent** sigmoids, so probabilities do not sum to 1 and any combination "
        "of labels can fire. Test macro-F1 **0.6836** with calibrated per-label thresholds.\n\n"
        "**This is a demo, not a moderation system.** It was trained on 2018 Wikipedia talk-page "
        "comments and is weakest exactly where it matters most: F1 is ~0.54 on `threat` and "
        "`severe_toxic`, and recall on `identity_hate` is 0.535. Do not use it to make "
        "consequential decisions about people. Request text is not logged or stored."
    ),
)


def _resolve(mode) -> tuple[dict[str, float], str]:
    if mode == "tuned":
        return RUNTIME["thresholds"], "tuned (calibrated on validation)"
    if mode == "default":
        return {l: 0.5 for l in LABELS}, "default 0.5"
    merged = {**RUNTIME["thresholds"], **{k: float(v) for k, v in mode.items()}}
    return merged, "custom"


@api.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "model_source": RUNTIME["source"],
        "device": RUNTIME["device"],
        "calibrated_thresholds_loaded": RUNTIME["calibrated"],
        "labels": LABELS,
    }


@api.get("/info", tags=["meta"])
def info() -> dict:
    """Model card. Metrics are read from results/ so they cannot drift from the repo."""
    payload = {
        "model": "distilbert-base-uncased fine-tuned for multi-label classification",
        "loss": "BCEWithLogitsLoss over 6 independent sigmoids (not softmax)",
        "labels": LABELS,
        "thresholds": RUNTIME["thresholds"],
        "model_source": RUNTIME["source"],
        "limits": {
            "max_texts_per_request": MAX_TEXTS_PER_REQUEST,
            "max_chars_per_text": MAX_CHARS_PER_TEXT,
            "max_tokens": MAX_LENGTH,
        },
        "caveats": [
            "Trained on 2018 English Wikipedia talk-page comments; other domains will degrade.",
            "Rare labels are weak: F1 ~0.54 on threat and severe_toxic.",
            "threat's threshold was tuned on only 72 validation positives, so it is noisy.",
            "Accuracy is not reported: an all-zeros model scores 89.8% exact-match at macro-F1 0.0.",
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
            "per_label": {
                l: {k: d[k] for k in ("threshold", "precision", "recall", "f1", "average_precision", "support")}
                for l, d in m["per_label"].items()
            },
            "all_zeros_control": m["all_zeros_baseline"],
        }
    return payload


@api.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    thresholds, mode = _resolve(req.thresholds)
    try:
        probs = predict_probs(
            req.texts, RUNTIME["model"], RUNTIME["tokenizer"], RUNTIME["device"], MAX_LENGTH
        )
    except Exception as exc:  # surface a real error rather than a silent empty result
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    return PredictResponse(
        predictions=[
            Prediction(
                labels=[l for i, l in enumerate(LABELS) if row[i] >= thresholds[l]],
                probabilities={l: round(float(row[i]), 6) for i, l in enumerate(LABELS)},
                thresholds_applied={l: float(thresholds[l]) for l in LABELS},
            )
            for row in np.atleast_2d(probs)
        ],
        model_source=RUNTIME["source"],
        threshold_mode=mode,
    )


demo = build_app(
    RUNTIME["model"], RUNTIME["tokenizer"], RUNTIME["device"],
    RUNTIME["thresholds"], RUNTIME["calibrated"],
)
# css/theme must be passed here: Gradio 6 moved them off the Blocks constructor.
app = gr.mount_gradio_app(api, demo, path="/", css=CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
