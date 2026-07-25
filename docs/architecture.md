# Architecture

Three services, one data path, no database.

Each diagram below is also exported to [`docs/diagrams/`](diagrams/) as an SVG by
`make diagrams`, so it exists as a file and not only as a fenced block.

---

## Container view

```mermaid
flowchart TB
    subgraph client["Client"]
        U["Reviewer / recruiter<br/>browser"]
    end
    subgraph web["ml-web · Next.js 14 · :3070"]
        P["/fintech/{fraud,credit-risk,churn}<br/>Zod forms + SHAP charts"]
        D["/dashboard · ISR 30s"]
    end
    subgraph api["ml-api · FastAPI · :8070"]
        R["routes: /predict/* /explain/* /models /health<br/>src/serving/api.py"]
        REG["checkpoint registry<br/>src/serving/registry.py"]
        PRE["request → features<br/>src/serving/preprocessing.py"]
        PRD["predictors/{fraud,credit_risk,churn}.py"]
        EXP["SHAP + gradient<br/>src/explainability/"]
    end
    subgraph store["Artifacts (gitignored)"]
        CK[("checkpoints/&lt;problem&gt;/<br/>model.* · features.joblib<br/>metadata.json")]
    end
    subgraph track["ml-mlflow · :5070"]
        ML[("runs · params · metrics<br/>model registry")]
    end
    U --> P --> R
    U --> D --> ML
    R --> PRE --> PRD --> EXP
    REG --> CK
    PRD --> REG
    R -->|"training-time only"| ML
```

**Design decision this encodes.** The API discovers models by globbing
`checkpoints/*/metadata.json` rather than holding a hardcoded list, so adding a
model requires no change to `api.py` and no redeploy — and a fresh clone with
zero checkpoints reports `status: ok` with three unavailable models instead of
crash-looping.

**Why the browser only ever talks to Next.js.** `next.config.mjs` rewrites
`/api/*` to the FastAPI container server-side. One origin means no CORS
preflight, no CORS middleware in the backend, and no second hostname to configure
per environment.

**There is no application database.** The three `predict` endpoints are pure
functions of their request plus a checkpoint on disk; nothing is persisted. The
only durable state is MLflow's own SQLite store, which is a vendor schema this
project does not own and should not draw an ERD for. **That is why this document
contains no entity-relationship diagram** — an invented one would describe
nothing that exists.

---

## The critical path: one prediction

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant W as Next.js /fintech/fraud
    participant A as FastAPI /predict/fraud
    participant G as serving/registry.py
    participant F as features/fraud_features.py
    participant M as LightGBM checkpoint
    participant S as explainability/shap_explainer.py
    U->>W: submit transaction form (Zod-validated)
    W->>A: POST /predict/fraud {json}
    A->>G: load("fraud")
    G-->>A: model + feature_columns + category_dtypes + metadata
    Note over A,G: 503 with the exact train command<br/>when no checkpoint exists
    A->>F: engineer_features(payload, fit=False)
    Note over F: the SAME function training called,<br/>with the frequency maps fitted on train
    F-->>A: feature frame
    A->>M: predict_proba(X)
    M-->>A: fraud probability
    A->>S: explain(model, X)
    S-->>A: per-feature SHAP contributions
    A-->>W: {probability, risk_band, action, model_version}
    W-->>U: score + provenance + SHAP bar chart
```

**Design decision this encodes.** Step 5 calls the *same* `engineer_features`
that training called, with the frequency maps and category dtypes that training
fitted, loaded from the checkpoint. Training/serving skew is the most common
production ML bug and the only structural defence is refusing to have a second
implementation. `tests/serving/test_skew.py` scores the same rows through both
paths and demands identical matrices.

The second non-obvious choice is step 3's failure mode: a missing checkpoint is a
**503**, not a 500. The service is healthy; the model has not been trained. A 500
would send a reviewer reading tracebacks over a fresh clone behaving exactly as
documented.

---

## Data pipeline

```mermaid
flowchart LR
    K1[("Kaggle competition<br/>ieee-fraud-detection<br/>590,540 × 394 · 3.5% fraud")]
    K2[("Kaggle dataset<br/>wordsforthewise/lending-club<br/>2.26M × 151 · CC0")]
    K3[("Kaggle dataset<br/>sakshigoyal7/credit-card-customers<br/>10,127 × 23")]
    OML[("OpenML 1597<br/>ULB fraud · NO ACCOUNT")]
    K1 & K2 & K3 & OML -->|"scripts/download_data.py<br/>rows recorded · file sha256 compared when pinned"| C[("~/.cache/kagglehub/<br/>outside the repo")]
    C -->|"src/data/adapters/*"| P["canonical frame<br/>float32 · category · int8 label"]
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
frame leaks the test distribution and is worth roughly a point of AUC that
evaporates in production.

The dotted edge is the other load-bearing detail: the CI fixtures reach the
trainer but **cannot** reach MLflow, W&B, `checkpoints/` or `reports/`.
`TabularTrainer` makes the sample flag known before tracking is initialized;
dashboard history also requires `sample=false` plus matching problem/config
tags. There is no code path from a synthetic fixture to a published number.

For stratified cross-validation, the model edge has two distinct outputs:
out-of-fold predictions grade the model and feed performance figures, while a
separate estimator refit on all rows becomes the serving checkpoint. Keeping
those roles separate prevents an all-row refit from evaluating itself.

---

## Why `conftest.py` lives at the repository root

It is not decoration and it should not be moved:

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

Measured on the development machine — Apple Silicon, 4 performance + 6 efficiency
cores, 32 GB RAM, torch 2.13.0, MPS available.

- **LightGBM and XGBoost ship CPU-only wheels on macOS arm64.** No Metal backend
  exists for either. The three headline models are gradient-boosted trees, so
  MPS buys them nothing; `n_jobs` in the config is the only lever that matters.
- **MPS gives roughly 1.9–2.2× over CPU on dense matmul**, not 5–10×. The only
  model here that touches it is the fraud autoencoder baseline.
- `torch.get_num_threads()` defaults to **4**, not 10.
- **torch 2.13.0 MPS bug, reproduced twice on this machine:**
  `torch.nn.MultiheadAttention` hangs on MPS, and a CPU tensor loop deadlocked at
  0% CPU after a preceding MPS matmul *in the same process*. This repository has
  no attention layer so it does not trip the bug, but two rules follow from it:
  the test suite forces MPS off session-wide, and the serving path never touches
  MPS. Any future sequence model must run one process per device.
- In Docker, `infra/docker/api.Dockerfile` force-installs CPU PyTorch. MPS is
  macOS-only and cannot exist in a Linux container — see
  [ADR-0002](adr/0002-experiment-tracking.md) and the original Decision 3.
