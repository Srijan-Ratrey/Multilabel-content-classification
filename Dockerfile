# Serving image for Hugging Face Spaces (Docker SDK) or any container host.
#
#   docker build -t policy-classifier .
#   docker run -p 7860:7860 -e MODEL_ID=<hf-user>/distilbert-jigsaw-multilabel policy-classifier
#
# The 256MB weights are NOT baked in: they are gitignored in this repo and live in a Hugging
# Face model repo, resolved at startup from MODEL_ID. That keeps the image small and lets the
# model be updated without a rebuild. To run fully self-contained instead, COPY a local
# models/distilbert-multilabel/ in and leave MODEL_ID unset.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HF caches default to /root; Spaces runs as a non-root user with $HOME=/home/user
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    # 2 vCPU on the free tier; more threads than cores adds contention, not speed
    OMP_NUM_THREADS=2 \
    PORT=7860

# Spaces requires a non-root user, and writable caches must belong to it.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user src/ ./src/
COPY --chown=user app.py serve.py ./
# Thresholds and metrics are read at runtime by /info and load_thresholds(); these are small
# JSON files tracked in git, not model weights.
COPY --chown=user results/thresholds_transformer.json results/transformer_tuned_metrics.json ./results/

USER user
EXPOSE 7860

# One worker on purpose: each worker would load its own ~600MB copy of the model, and a single
# process already sustains ~87 req/s on 2 CPU threads.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
