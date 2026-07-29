---
title: Multi-Label Content Policy Classifier
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# Multi-label content / policy classification

Fine-tuned DistilBERT that predicts **all** applicable policy labels for a comment — `toxic`,
`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` — rather than picking one.

**Test macro-F1: 0.6836** (held-out split, n=23,936) with per-label thresholds calibrated on
validation.

## What this demo is actually showing

The model emits six **independent** sigmoid probabilities, so a comment can be toxic *and*
obscene *and* insulting at once and the probabilities need not sum to 1. Softmax could not
represent that.

More importantly: **the model gives probabilities, but the decision is a separate choice.**
Switch the UI to *Uniform threshold* and drag the slider — the probabilities never move, yet the
predicted labels change. `threat` occurs in 0.30% of the training data, so its calibrated
threshold is **0.20**, not 0.5: a borderline threat gets flagged that the default would wave
through. Moving thresholds alone is worth **+0.021 macro-F1** here, and was worth +0.077 on the
TF-IDF baseline — with no retraining.

Accuracy is deliberately not reported. A model that predicts nothing at all scores **89.8%
exact-match accuracy** on this data, at macro-F1 of exactly **0.0**.

## API

Two endpoints, `predict` and `info`, documented under **Use via API** at the bottom of the page.

Easiest from Python:

```python
from gradio_client import Client

client = Client("srijanratrey/multilabel-content-classification")
print(client.predict(texts=["you are an idiot"], thresholds="tuned", api_name="/predict"))
```

Over plain HTTP it is a two-step call (Gradio's protocol): `POST /gradio_api/call/predict`
returns an `event_id`, then `GET /gradio_api/call/predict/{event_id}` streams the result.

```bash
EID=$(curl -s -X POST https://<space-host>/gradio_api/call/predict \
  -H 'Content-Type: application/json' \
  -d '{"data": [["you are an idiot"], "tuned"]}' | sed -n 's/.*"event_id":"\([^"]*\)".*/\1/p')
curl -N https://<space-host>/gradio_api/call/predict/$EID
```

`thresholds` is `"tuned"` (calibrated per label) or `"default"` (0.5 everywhere).
Limits: 32 texts per request, 5,000 characters each. Request text is not logged or stored.

## Limitations — read before using this for anything

- Trained on 2018 English Wikipedia talk-page comments. Other domains, platforms, and eras will
  degrade, likely a lot.
- **Rare labels are weak:** F1 ≈ 0.54 on both `threat` and `severe_toxic`; recall on
  `identity_hate` is 0.535, so it misses about half of them.
- `threat`'s threshold was tuned on only **72** validation positives, so it is a noisy estimate.
- Inputs are truncated at 256 tokens (~6.9% of the training comments were longer).
- This is a **demo, not a moderation system.** Do not use it to make consequential decisions
  about people.

Source and full write-up: https://github.com/Srijan-Ratrey/Multilabel-content-classification
