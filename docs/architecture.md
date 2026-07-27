# Architecture

Three services, one data path, no database.

Each diagram below is also exported to [`docs/diagrams/`](diagrams/) as an SVG by
`make diagrams`, so it exists as a file and not only as a fenced block.

---

## Container view

```mermaid
%%{init: {'htmlLabels': false, 'fontFamily': 'arial, helvetica, sans-serif', 'flowchart': {'htmlLabels': false, 'padding': 16, 'nodeSpacing': 60, 'rankSpacing': 70, 'useMaxWidth': true}}}%%
flowchart TB
    subgraph client["Client"]
        U["Reviewer / recruiter<br/>browser"]
    end
    subgraph web["ml-web · Next.js 14 · :3070"]
        P["/fintech/{fraud, credit-risk,<br/>churn}<br/>Zod forms + SHAP charts"]
        D["/dashboard · ISR 30s"]
    end
    subgraph api[" "]
        R["ml-api · FastAPI · :8070<br/>routes: /predict/* /explain/*<br/>/models /health<br/>src/serving/api.py"]
        REG["checkpoint registry<br/>src/serving/registry.py"]
        PRE["request → features<br/>src/serving/<br/>preprocessing.py"]
        PRD["predictors/{fraud, credit_risk,<br/>churn}.py"]
        EXP["SHAP + gradient<br/>src/explainability/"]
    end
    CK[("Artifacts (gitignored)<br/>checkpoints/&lt;problem&gt;/<br/>model.joblib<br/>features.joblib<br/>metadata.json")]
    ML[("ml-mlflow · :5070<br/>runs · params<br/>metrics<br/>model registry")]
    U --> P --> R
    U --> D --> ML
    R --> PRE --> PRD --> EXP
    REG --> CK
    PRD --> REG
    R -->|"training-time only"| ML
```

**Design decision this encodes.** The API discovers models by globbing
`checkpoints/*/metadata.json`. Adding a model requires no change to `api.py` and
no redeploy. A fresh clone with zero checkpoints reports `status: ok` with three
unavailable models without entering a crash loop.

**Why the browser only ever talks to Next.js.** `next.config.mjs` rewrites
`/api/*` to the FastAPI container server-side. One origin means no CORS
preflight, no CORS middleware in the backend, and no second hostname to configure
per environment.

**There is no application database.** The three `predict` endpoints are pure
functions of their request plus a checkpoint on disk; nothing is persisted. The
only durable state is MLflow's own SQLite store, which is a vendor schema this
project does not own. This document contains no entity-relationship diagram
because the project has no application data model to describe.

---

## The critical path: one prediction

```mermaid
%%{init: {'htmlLabels': false, 'fontFamily': 'arial, helvetica, sans-serif'}}%%
sequenceDiagram
    autonumber
    actor U as User
    participant W as Next.js
    participant A as FastAPI
    participant G as Registry
    participant F as Fraud features
    participant M as LightGBM
    participant S as SHAP
    U->>W: submit transaction form<br/>(Zod-validated)
    W->>A: POST /predict/fraud {json}
    A->>G: load("fraud")
    G-->>A: model + feature_columns<br/>+ category_dtypes + metadata
    Note over A,G: 503 with exact train command<br/>when checkpoint is missing
    A->>F: engineer_features(...)
    Note over F: same feature code<br/>used in training<br/>frequency maps fit<br/>on train
    F-->>A: feature frame
    A->>M: predict_proba(X)
    M-->>A: fraud probability
    A->>S: explain(model, X)
    S-->>A: per-feature SHAP<br/>contributions
    A-->>W: probability + risk band<br/>+ action + model version
    W-->>U: score + provenance<br/>+ SHAP bar chart
```

**Design decision this encodes.** `src/serving/registry.py` loads the checkpoint,
training and serving both call `src/features/fraud_features.py`, and
`src/explainability/shap_explainer.py` provides the explanation. Sharing the
implementation removes a second feature path that could diverge.
`tests/serving/test_skew.py` scores the same rows through both paths and demands
identical matrices.

Step 3 returns **503** when a checkpoint is missing, not a 500. The service is
healthy, but the model has not been trained. The response includes the exact
training command. HTTP 500 denotes an internal server error, which is not the
service state.

---

## Data pipeline

```mermaid
%%{init: {'htmlLabels': false, 'fontFamily': 'arial, helvetica, sans-serif', 'flowchart': {'htmlLabels': false, 'padding': 16, 'nodeSpacing': 60, 'rankSpacing': 70, 'useMaxWidth': true}}}%%
flowchart LR
    K1[("Kaggle competition<br/>ieee-fraud-detection<br/>590,540 × 394 · 3.5% fraud")]
    K2[("Kaggle dataset<br/>wordsforthewise/lending-club<br/>2.26M × 151 · CC0")]
    K3[("Kaggle dataset<br/>sakshigoyal7/<br/>credit-card-customers<br/>10,127 × 23")]
    OML[("OpenML 1597<br/>ULB fraud · NO ACCOUNT")]
    K1 & K2 & K3 & OML -->|"scripts/download_data.py<br/>rows recorded · file sha256<br/>compared when pinned"| C[("~/.cache/kagglehub/<br/>outside the repo")]
    C -->|"src/data/<br/>adapters/*"| P["canonical frame<br/>float32 · category · int8 label"]
    P -->|"src/data/split.py<br/>TIME-BASED"| T{{"train / val / test"}}
    T -->|"src/features/*<br/>fit on TRAIN only"| X["feature matrix"]
    X -->|"src/training/tabular.py<br/>configs/&lt;problem&gt;.yaml"| MDL["LightGBM + baselines<br/>prior · logreg"]
    MDL --> CK[("checkpoints/&lt;problem&gt;/")]
    MDL --> ML[("MLflow :5070")]
    MDL --> RES["reports/RESULTS.md<br/>reports/*_metrics.csv"]
    CK --> API["FastAPI :8070"] --> WEB["Next.js :3070"]
    SMP[("data/sample/*.csv<br/>500-row CI fixtures")] -.->|"CI only · writes NO checkpoint"| T
```

**Design decision this encodes.** The split happens **before** feature
engineering, not after. Frequency encodings and group aggregates are fitted on
the training rows only and carried forward as artifacts; fitting them on the full
frame leaks information about the test distribution and does not represent
deployment.

The dotted edge shows fixture isolation: the CI fixtures reach the trainer but
**cannot** reach MLflow, W&B, `checkpoints/` or `reports/`.
`TabularTrainer` makes the sample flag known before tracking is initialized;
dashboard history also requires `sample=false` plus matching problem/config
tags. There is no code path from a synthetic fixture to a published number.

For stratified cross-validation, the model edge has two distinct outputs:
out-of-fold predictions grade the model and feed performance figures, while a
separate estimator refit on all rows becomes the serving checkpoint. Keeping
those roles separate prevents an all-row refit from evaluating itself.

---

## Why `conftest.py` lives at the repository root

The root location establishes the import order before test collection:

```python
import lightgbm  # noqa: F401
import xgboost   # noqa: F401
```

LightGBM, XGBoost and PyTorch each vendor their own copy of `libomp`. On macOS,
importing PyTorch first and then one of the boosters aborts the process. The root
`conftest.py` runs before any test module is collected and forces the safe import
order; `tests/conftest.py` additionally sets `KMP_DUPLICATE_LIB_OK=TRUE` before
any import at all.

---

## Hardware notes

Development hardware: Apple Silicon, 4 performance + 6 efficiency cores, 32 GB
RAM, torch 2.13.0, MPS available.

- **LightGBM and XGBoost ship CPU-only wheels on macOS arm64.** No Metal backend
  exists for either. The three headline models are gradient-boosted trees, so
  MPS buys them nothing; `n_jobs` in the config is the only lever that matters.
- **MPS is available for the fraud autoencoder baseline.** This repository does
  not contain a committed MPS-versus-CPU benchmark, so it makes no speedup claim.
- `torch.get_num_threads()` defaults to **4**, not 10.
- **torch 2.13.0 MPS bug, reproduced twice on this machine:**
  `torch.nn.MultiheadAttention` hangs on MPS, and a CPU tensor loop deadlocked at
  0% CPU after a preceding MPS matmul *in the same process*. This repository has
  no attention layer. The test suite forces MPS off session-wide, and the
  serving path never touches MPS. Any future sequence model must run one process
  per device.
- In Docker, `infra/docker/api.Dockerfile` force-installs CPU PyTorch. MPS is
  macOS-only and cannot exist in a Linux container.
