# Multi-Label Content / Policy Classification — Build Spec

This is a build brief for a learning + portfolio ML project. Work through the
phases **in order**. Each phase has a learning goal and concrete deliverables.
Do not skip ahead — later phases depend on the artifacts (metrics, thresholds,
saved models) produced by earlier ones. Commit after each phase.

---

## Build status

Progress tracker. "Code written" means the module exists and imports cleanly but has
**not** been run on the full data, so no metrics exist for it yet.

| Phase | Status | Evidence / what's missing |
|---|---|---|
| 0 — Setup & EDA | ✅ **done** | `results/eda.json`, `results/splits.json`, 4 EDA plots, `notebooks/01_eda.ipynb` · commit `4e9a460` |
| 1 — Dumb baseline | ✅ **done** | `results/baseline_metrics.json` (+ `_unweighted`), probs in `models/probs/` · commit `7876344` |
| 2 — Metric design | ✅ **done** | `src/evaluate.py`, `notebooks/02_metrics.ipynb`, `results/phase2_*.json`, `results/pr_*.png` · commit `69ef626` |
| 3 — Strong model (DistilBERT) | ⬜ **code written, not run** | `src/transformer.py` + `notebooks/train_model.ipynb` exist and are smoke-tested; **no** `results/transformer_metrics.json`, no `models/probs/transformer_*.npy` |
| 4.1 — Per-label thresholds | ◐ **partially done** | Run on the baseline: `results/thresholds_baseline.json`, `threshold_comparison_baseline.json` (macro-F1 0.5457 → 0.6227). **Not** run on the transformer — blocked by Phase 3 |
| 4.2 — Policy-conditioned | ⬜ **code written, not run** | `src/policy.py`, `src/policies.py` exist (3 written policy definitions, `threat` held out); **no** `results/policy_conditioned.json` or `policy_zeroshot.json` |
| 5 — Error analysis | ⬜ **not started** | Needs `src/error_analysis.py` + `notebooks/03_error_analysis.ipynb`; requires transformer probabilities from Phase 3 |
| 6 — Classifier chains | ⬜ **not started** | Needs `src/chains.py`. Reliability diagram deliberately skipped (chains-only was chosen) |
| Final — README | ◐ **in progress** | Engineering-doc README covering Phases 0–2 with verified numbers only; results table to be completed once Phases 3–6 run |

### Remaining work, in dependency order

1. **Phase 3 — train DistilBERT.** Run `notebooks/train_model.ipynb` (`QUICK_TEST = False`),
   or `python -m src.transformer`. ~45 min on an M4. This unblocks everything below.
   It also writes the Phase 4.1 transformer thresholds itself.
2. **Phase 4.1 on the transformer** — `python -m src.thresholds --model transformer`
   (skip if the notebook already produced `threshold_comparison_transformer.json`).
3. **Phase 4.2 — policy-conditioned.** `python -m src.policy --variant all`, then
   `python -m src.policy --variant heldout`. The head-to-head table needs step 2's output.
4. **Phase 5 — error analysis.** Write `src/error_analysis.py` +
   `notebooks/03_error_analysis.ipynb`: conflation matrix, true-vs-predicted co-occurrence,
   and ~15 hand-inspected failures with a cause hypothesis each.
5. **Phase 6 — classifier chains.** Write `src/chains.py`; compare against One-vs-Rest on
   both macro-F1 and co-occurrence fidelity (the Phase 5 metric).
6. **Final — complete the README** results table and findings from the above.

---

## Goal

Given a piece of text (comment/post/message), predict **all** applicable labels
(multi-label, not single-label). The whole project is organized around one
theme: **accuracy is the wrong metric here — the work is choosing the right
metric, calibrating per-label thresholds, and doing real error analysis.**

## Constraints & conventions

- Python 3.11+. Use a virtual environment.
- Keep dependencies minimal; pin versions in `requirements.txt`.
- Reproducibility: set a global seed (`SEED = 42`) everywhere (numpy, torch,
  sklearn splits). Log it.
- Structure the repo cleanly:
  ```
  .
  ├── README.md            # write this LAST, summarizing results
  ├── requirements.txt
  ├── data/                # raw + processed (gitignore raw)
  ├── src/
  │   ├── data.py          # loading, cleaning, label matrix
  │   ├── baseline.py      # TF-IDF + OvR LogReg
  │   ├── transformer.py   # fine-tuned model
  │   ├── evaluate.py      # metrics, threshold search, error analysis
  │   └── thresholds.py    # per-label threshold calibration
  ├── notebooks/           # exploration + error analysis
  ├── models/              # saved artifacts (gitignore large files)
  └── results/             # metrics JSON, plots
  ```
- Every metric you compute gets written to `results/` as JSON so phases can
  compare. Every plot saved to `results/`.
- Do NOT report accuracy as a headline metric. If you show it, show it only to
  demonstrate why it is misleading here.

## Dataset

Use the **Jigsaw Toxic Comment Classification** dataset (Kaggle:
`jigsaw-toxic-comment-classification-challenge`). ~160K Wikipedia comments,
6 binary labels: `toxic, severe_toxic, obscene, threat, insult, identity_hate`.
Labels co-occur and are heavily imbalanced.

If Jigsaw is unavailable, fall back to **GoEmotions** (27 labels + neutral) and
note the change in the README. Ask me before switching if you hit an access wall.

---

## Phase 0 — Setup & EDA

**Learning goal:** understand the multi-label + imbalance shape of the data
before modeling.

Tasks:
1. Set up venv, `requirements.txt`, repo structure above.
2. Load data in `src/data.py`. Build `Y` as an `(n_samples, n_labels)` binary
   matrix — this shape is central; everything downstream assumes it.
3. EDA notebook: per-label positive rate, label co-occurrence matrix (heatmap),
   distribution of number-of-labels-per-example, text length distribution.
4. Save an EDA summary to `results/eda.json` (per-label positive rates at least).

**Deliverable:** you can state which labels are rare (e.g. `threat` may be <1%)
and which co-occur. Commit.

---

## Phase 1 — Dumb baseline (learn the plumbing)

**Learning goal:** internalize that multi-label = N independent binary problems
(that's what One-vs-Rest is), and get first contact with imbalance.

Tasks in `src/baseline.py`:
1. `TfidfVectorizer(max_features=20000, ngram_range=(1,2))`.
2. `OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))`.
3. Train/val/test split (stratify as well as multi-label allows;
   use `iterative-stratification` if you add it, else random split + note it).
4. Predict probabilities (`predict_proba`) AND default-0.5 labels.
5. Save per-label F1, precision, recall to `results/baseline_metrics.json`.

**Deliverable:** a floor score per label. Note explicitly how the model behaves
on the rarest label at threshold 0.5 (it likely almost never predicts it).
This gap is your portfolio story later. Commit.

---

## Phase 2 — Metric design (MOST IMPORTANT — do not skip)

**Learning goal:** on multi-label + imbalanced data, accuracy is misleading
(all-zeros scores ~95%). Learn micro vs macro F1 and per-label PR curves.

Tasks in `src/evaluate.py`:
1. Implement a single `evaluate(y_true, y_prob, thresholds)` that reports:
   - **Micro-F1** (aggregated across labels; dominated by frequent labels)
   - **Macro-F1** (per-label F1 averaged equally; punishes ignoring rare labels)
     — this is the **headline metric**.
   - Per-label precision/recall/F1.
   - Exact-match / subset accuracy — include ONLY to show it's a poor fit.
2. Per-label PR curves saved as plots to `results/`.
3. Re-evaluate the Phase 1 baseline through this harness.

**Deliverable:** demonstrate the micro-high / macro-low divergence and explain
in the notebook what it means (model is ignoring rare classes). Commit.

---

## Phase 3 — Strong model (the upgrade)

**Learning goal:** the loss-level mechanics of multi-label — sigmoid (not
softmax) + binary cross-entropy per label.

Tasks in `src/transformer.py`:
1. Fine-tune `distilbert-base-uncased` (cheap; upgrade to `bert-base-uncased`
   if GPU allows). Use HuggingFace `transformers`.
2. **Critical:** output layer = `num_labels` independent **sigmoids**, loss =
   `BCEWithLogitsLoss`. NOT softmax / categorical cross-entropy — softmax forces
   a single class; sigmoid lets multiple labels fire independently. Add a code
   comment explaining exactly this, since it's the core multi-label concept.
3. Truncate/handle long inputs at model max length; note token-length stats.
4. Evaluate through the SAME `src/evaluate.py` harness at default 0.5 first.
5. Save probs on val + test to disk (needed for Phase 4).

**Deliverable:** transformer metrics vs baseline, same harness. Commit.

---

## Phase 4 — The hard part (where the value lives)

**Learning goal:** the model outputs probabilities; **you** choose the operating
point. Default 0.5 is wrong for imbalanced labels.

Tasks in `src/thresholds.py`:
1. **Per-label threshold calibration.** For each label, sweep thresholds on the
   **validation** set and pick the one maximizing that label's F1. Apply the
   chosen thresholds to the **test** set (never tune on test).
   ```python
   best_t = {}
   for i, label in enumerate(labels):
       ts = np.linspace(0.05, 0.95, 19)
       best_t[label] = max(ts, key=lambda t: f1_score(Y_val[:, i], probs_val[:, i] >= t))
   ```
2. Save `results/thresholds.json`. Re-evaluate test with tuned thresholds and
   compare macro-F1 before vs after. This is the single biggest lift; quantify it.
3. **Policy-conditioned variant (differentiator — implement at least a light
   version).** Write explicit written definitions for 2–3 of the labels and
   build a second classification path framed as "does this text violate policy X
   (defined as …)?" Compare fixed-label vs definition-conditioned behavior and
   discuss. This connects to real content-safety / policy-conditioned classifiers
   and is what makes the project memorable.

**Deliverable:** before/after macro-F1 table from thresholding, plus the
policy-conditioned comparison. Commit.

---

## Phase 5 — Error analysis (the senior signal)

**Learning goal:** metrics aren't the end — understand *what* it gets wrong.

Tasks (notebook + `results/`):
1. Per-label confusion patterns — which labels get conflated (expect
   `obscene`↔`insult` overlap; show it).
2. Predicted vs true label **co-occurrence** — does the model capture that
   `identity_hate` usually co-occurs with `toxic`? One-vs-Rest structurally
   cannot; note this as motivation for classifier chains.
3. Hand-inspect ~15 failures; for each, a one-line hypothesis for the cause
   (sarcasm, negation, rare vocabulary, label noise).

**Deliverable:** an error-analysis section with concrete examples + hypotheses.
Commit.

---

## Phase 6 (optional stretch)

- **Classifier chains** vs One-vs-Rest — model label dependencies; directly
  demonstrates the OvR limitation noted in Phase 5.
- **Calibration check** — reliability diagram: are output probabilities
  meaningful or just rankings?

---

## Final — README

Write `README.md` LAST. Include: the problem framing, the "accuracy is the wrong
metric" through-line, the baseline→transformer→thresholds progression as a
results table (macro-F1 at each stage), the policy-conditioned experiment, and
the error-analysis findings. Keep plots inline from `results/`.

## Acceptance criteria

- Runs end-to-end from a documented command sequence.
- Macro-F1 improves visibly across phases (baseline < transformer <
  transformer+tuned thresholds), shown in one table.
- Evaluation never uses accuracy as the headline; macro-F1 + per-label PR are
  primary.
- Thresholds tuned on validation, never on test.
- Error analysis contains real inspected examples, not just numbers.

## Notes for the agent

- Ask before installing anything heavy or downloading large data if the
  environment is constrained.
- If a dataset is inaccessible, stop and ask rather than silently substituting.
- Prefer small, testable increments; commit per phase with clear messages.