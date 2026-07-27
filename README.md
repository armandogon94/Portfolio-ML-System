# Fintech ML System: Fraud, Credit Risk, and Attrition on Real Public Data

[![CI](https://github.com/armandogon94/Portfolio-ML-System/actions/workflows/ci.yml/badge.svg)](https://github.com/armandogon94/Portfolio-ML-System/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Three real-money fintech problems are covered: payment fraud, consumer credit
risk, and card attrition. Three accepted real-data training runs are tracked in
MLflow and served behind a FastAPI inference API with SHAP explanations.

- **Focus:** payment fraud · consumer credit risk · card attrition
- **Data:** [IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (590,540 × 394, 3.5% fraud) · [LendingClub](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (2.26M × 151; uploader tags CC0, upstream authority unverified) · [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) (10,127 × 23; uploader tags CC0, upstream authority unverified)
- **Stack:** Python 3.11 · LightGBM · PyTorch (MPS) · SHAP · MLflow · FastAPI · Next.js 14 · Docker
- **Output:** a results table where every filled cell traces to a CSV written by a training run. Three of five rows are measured: `fraud_ulb`, `credit_risk`, and `churn`. The `fraud` and `fraud_autoencoder` rows remain empty because both need the credential-blocked IEEE-CIS competition data.

📄 **[Read the full methodology & analysis →](reports/RESULTS.md)**

---

## Current status

**Three of five result rows are measured on real data: `fraud_ulb`,
`credit_risk`, and `churn`.**

The other two remain empty. IEEE-CIS is blocked because its competition
download needs a classic `kaggle.json` token rather than the working Kaggle OAuth
token; `fraud_autoencoder` needs the same data.

[`docs/PROGRESS.md`](docs/PROGRESS.md) lists exactly what is blocked, why, and the
commands to unblock it.

---

## Key Findings

- **The credential-free fraud result beat its logistic baseline on PR-AUC.**
  ULB/OpenML 1597 measured PR-AUC **0.8569 ± 0.0331** against
  **0.7300 ± 0.0279** for logistic regression, all out of fold.
- **Gradient boosting was barely better than logistic regression on
  LendingClub.** LightGBM measured **0.3935 PR-AUC** against **0.3720**, a
  **+0.0215** margin. The test partition is censored by terminal-status
  filtering at the 2018Q4 data cut, so this is not an unbiased estimate of
  forward default risk.
- **The churn score is not prospective.** Its **0.9735 ± 0.0078 PR-AUC**
  substantially detects an attrition that already happened because current
  status and trailing-activity features describe the same period.
## Results

```bash
uv run python scripts/train.py --model fraud_ulb   # no Kaggle account needed
uv run python scripts/evaluate.py --markdown
```

| Problem | Dataset (rows × cols, % positive) | Split | Model | PR-AUC ↑ | ROC-AUC ↑ | Baseline PR-AUC | Δ vs baseline | Source |
|---|---|---|---|---|---|---|---|---|
| Fraud | IEEE-CIS · 590,540 × 394 · 3.50% | time (`TransactionDT` 80/20) | LightGBM |  |  |  |  | `reports/fraud_metrics.csv` |
| Fraud (`fraud_ulb`, credential-free) | ULB / OpenML 1597 · 284,807 × 30 · 0.1727% | 5-fold stratified CV | LightGBM | **0.8569 ± 0.0331** | 0.9810 ± 0.0092 | 0.7300 ± 0.0279 | 0.1269 ± 0.0355 | [`reports/fraud_ulb_metrics.csv`](reports/fraud_ulb_metrics.csv) |
| Fraud (unsup. baseline) | IEEE-CIS · 590,540 × 394 · 3.50% | time (`TransactionDT` 80/20) | Autoencoder (MPS) |  |  |  |  | `reports/fraud_autoencoder_metrics.csv` |
| Credit risk | LendingClub · 1,345,310 terminal-status rows × 32 features · 19.96% default | time (`issue_d`, actual 71.6/8.8/19.6) | LightGBM | **0.3935** | 0.7160 | 0.3720 | 0.0215 | [`reports/credit_risk_metrics.csv`](reports/credit_risk_metrics.csv) |
| Churn | CC attrition · 10,127 × 23 · 16.07% | 5-fold stratified CV | LightGBM | **0.9735 ± 0.0078** | 0.9940 ± 0.0019 | 0.7800 ± 0.0217 | 0.1935 ± 0.0145 | [`reports/churn_metrics.csv`](reports/churn_metrics.csv) |

<sub>Three rows are measured; two remain empty. IEEE-CIS is blocked on a classic
`kaggle.json` competition token, and the autoencoder needs that dataset.
`scripts/evaluate.py` reads every filled cell from training-written
`reports/*_metrics.csv`; it computes nothing. Full config, caveats, and provenance:
[`reports/RESULTS.md`](reports/RESULTS.md).</sub>

<sub>For stratified cross-validation, the displayed and gated result is the CV
mean ± standard deviation. The serving checkpoint is refit on all rows, and the
curves/calibration/confusion matrix come from one persisted out-of-fold score per
row. The autoencoder uses raw reconstruction error for ranking metrics and its
training-set 95th-percentile threshold for hard decisions; it reports no Brier
score because that error is not a calibrated probability.</sub>

**Read PR-AUC first.** On ULB, ROC-AUC **0.9810 ± 0.0092** is inflated by the
**0.1727%** positive rate; the prior baseline gets **0.5000 ROC-AUC** but only
**0.0017 PR-AUC**. The informative comparison is **0.8569 ± 0.0331 PR-AUC**
against the logistic-regression baseline of **0.7300 ± 0.0279**.

**Do not read churn as forecasting.** `Attrition_Flag` is current status while
its strongest features summarise the same trailing activity window. With no
event timestamp, feature cutoff, or future outcome window, the dataset supports
detection of an attrition that already happened, not prospective prediction.

---

## Architecture

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

The API discovers models by globbing `checkpoints/*/metadata.json` rather than
holding a hardcoded list, so adding a model needs no change to `api.py` and no
redeploy; a fresh clone with zero checkpoints reports `status: ok` with three
unavailable models instead of crash-looping.
[SVG](docs/diagrams/c4-container.svg) · [full notes](docs/architecture.md)

## How a prediction happens

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

Step 5 calls the *same* `engineer_features` training called, with the frequency
maps and category dtypes training fitted, loaded from the checkpoint.
A shared implementation prevents training/serving skew.
`tests/serving/test_skew.py` scores the same rows through both paths and demands
identical matrices. [SVG](docs/diagrams/sequence-predict.svg)

## Data pipeline

```mermaid
flowchart LR
    K1[("Kaggle competition<br/>ieee-fraud-detection<br/>590,540 × 394 · 3.5% fraud")]
    K2[("Kaggle dataset<br/>wordsforthewise/lending-club<br/>2.26M × 151 · rights unresolved")]
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

The split happens **before** feature engineering. Frequency encodings and group
aggregates are fitted on the training rows only; fitting them on the full frame
leaks information about the test distribution into training. The dotted edge is
the other load-bearing detail: fixtures reach
the trainer but open no tracking run and cannot reach `checkpoints/` or
`reports/`.
[SVG](docs/diagrams/pipeline-dag.svg)

This project has no application database, so there is **no ERD**. The only
durable state is MLflow's SQLite store, which is a vendor schema this project
does not own. [`docs/architecture.md`](docs/architecture.md) documents that
boundary.

---

## Repository Structure

```bash
07-Portfolio-ML-System/
├── configs/                    # 4 dataset configs across the closed 3-problem scope.
│   │                           #   Each names a real source, split, seed, and sanity band.
│   ├── fraud.yaml              #   IEEE-CIS · time split on TransactionDT
│   ├── fraud_ulb.yaml          #   OpenML 1597 · credential-free · time split on Time
│   ├── credit_risk.yaml        #   LendingClub · time split on issue_d · 29-entry denylist
│   └── churn.yaml              #   Card attrition: 5-fold stratified CV
├── src/
│   ├── config.py               # Loads AND validates. Refuses a config with no real data.source.
│   ├── data/
│   │   ├── download.py         #   kaggle_dataset / kaggle_competition / openml. No fallback.
│   │   ├── split.py            #   Time-based by default; val sits between train and test.
│   │   └── adapters/           #   raw vendor columns -> canonical schema, one per dataset
│   ├── features/               # 4 dataset modules + schema.py. Training and serving share them.
│   ├── models/registry.py      # @register("lightgbm"). Unknown name -> KeyError listing valid ones.
│   ├── training/
│   │   ├── trainer.py          #   BaseTrainer: config, MLflow, W&B, model registry
│   │   ├── tabular.py          #   ONE config-driven trainer. Replaced 8 train_*.py files.
│   │   └── autoencoder_pipeline.py  # the unsupervised fraud baseline (the only MPS user)
│   ├── evaluation/             # PR-AUC primary; precision@k and recall@FPR for the ops framing
│   ├── explainability/         # SHAP (trees) + input gradients (autoencoder)
│   └── serving/                # registry · preprocessing · predictors/ · explain · handlers · api
│                               #   Every file under 150 lines. Was one 573-line monolith.
├── scripts/                    # download_data · train · evaluate · serve · make_figures
│                               #   make_fixtures · capture_screenshots · verify_fresh_clone.sh
├── data/
│   ├── README.md               # Provenance, licence, access gate, and the leakage traps per dataset
│   └── sample/                 # COMMITTED 500-row synthetic CI fixtures. Labels are pure noise.
├── tests/                      # Mirrors src/ package-for-package; local coverage gate is 80%.
│   ├── data/test_leakage_denylist.py   # blocks post-origination and target leakage
│   ├── serving/test_skew.py            # training features == serving features
│   ├── test_quality_gates.py           # sanity band, baseline improvement, monotonicity
│   └── e2e/test_train_to_serve.py      # fixtures -> train -> checkpoint -> HTTP, under 30s
├── web/                        # Next.js 14. Three routes + /dashboard. No placeholder copy.
├── infra/                      # docker/ multi-stage non-root images · compose/ 3-service stack
├── docs/                       # adr/ 0001-0005 · diagrams/ Mermaid + SVG · architecture.md
├── reports/                    # RESULTS.md + *_metrics.csv (written by training, never by hand)
├── conftest.py                 # AT ROOT ON PURPOSE: imports xgboost+lightgbm before torch (libomp)
└── uv.lock                     # COMMITTED. The Docker build fails without it.
```

---

## Quickstart

```bash
git clone https://github.com/armandogon94/Portfolio-ML-System.git
cd Portfolio-ML-System
cp .env.example .env

make setup           # uv sync --frozen --extra dev
make test            # test suite on committed fixtures; no credentials or network
make train-sample    # smoke-train all 4 configs (opens no tracker; writes no artifacts)
```

The fixture path needs no Kaggle account and makes no dataset-network calls. A
cold clone still needs network access for the clone and the first dependency
install; cached dependencies can be reused afterward.

To train on real data:

```bash
make data            # needs a Kaggle token; see data/README.md
make train
make evaluate        # reads reports/*_metrics.csv
make figures         # PR curves, calibration, SHAP -> reports/figures/
make screenshots-install  # one-time Chromium install
make screenshots     # needs the running stack and trained checkpoints
```

Credential-free real-data fraud path:

```bash
uv run python scripts/download_data.py --dataset ulb-creditcard
uv run python scripts/train.py --model fraud_ulb
uv run python scripts/evaluate.py --markdown
```

This fetches OpenML dataset 1597 through
`sklearn.datasets.fetch_openml(data_id=1597, as_frame=True)` and needs no Kaggle
account, token, or rules acceptance.

To run the stack:

```bash
make docker-up       # mlflow :5070 · api :8070 · web :3070
curl http://localhost:8070/health
open http://localhost:3070/dashboard
```

**Requires:** Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), Docker 24+, and pnpm 10
for the web app.

---

## Data & access

Four real datasets cover three business problems. The three Kaggle sources are
not anonymously downloadable; the OpenML source is:

| Dataset | Account | Extra gate | Redistributable |
|---|---|---|---|
| IEEE-CIS Fraud Detection | Free Kaggle | **Yes (one-click rules acceptance)** | **No** |
| LendingClub 2007-2018Q4 | Free Kaggle | No | Uploader tags CC0; upstream authority unverified. Do not redistribute rows. |
| Credit Card Customers | Free Kaggle | No | Uploader tags CC0; upstream authority unverified. Do not redistribute rows. |
| ULB Credit Card Fraud (OpenML 1597) | **None** | No | Unresolved: OpenML records only "Public"; the Kaggle mirror indicates ODbL-style terms. Treat as NOT cleared for redistribution. |

IEEE-CIS sits behind a Kaggle account *and* an acceptance of the competition
rules that cannot be scripted. On this machine,
`~/.kaggle/access_token` authenticates Kaggle **datasets** but
`kagglehub.competition_download('ieee-fraud-detection')` still returns 403; the
competition path needs a classic `~/.kaggle/kaggle.json` API token. The
ULB/OpenML path needs **no account at all**, so a reviewer with zero Kaggle
presence can reproduce a real-data fraud result with the commands above.

Because IEEE-CIS competition data is not redistributable, **no real row from any
of these datasets is committed here.** The only data in git is `data/sample/`:
500-row synthetic fixtures whose labels are drawn independently of the features,
used by CI and by nothing else.

Full provenance, per-dataset column notes and both leakage traps:
[`data/README.md`](data/README.md).

---

## Reproducibility

- **Seed 42**, declared once per config and threaded into the split,
  preprocessing and every estimator by `BaseTrainer._seed_everything`, including
  PyTorch weight initialisation, dropout and shuffling. Seeded MPS kernels are not
  guaranteed bit-deterministic.
- **`uv.lock` is committed.** `api.Dockerfile` runs
  `COPY pyproject.toml uv.lock ./` and `uv sync --frozen`.
  `scripts/verify_fresh_clone.sh` checks the quickstart from a fresh clone.
- **Every checkpoint carries its own provenance.**
  `checkpoints/<problem>/metadata.json` records the git SHA that trained it, the
  seed, the dataset source, the split config, the full feature column list, every
  metric, the checkpoint fit scope and row count, and the enforced leakage
  controls. Autoencoder checkpoints write the same fields and the same metrics
  CSV path as tabular checkpoints.
- **`/predict` returns `model_version`:** the short git SHA from that metadata is
  included in every response. A score with no provenance is unreviewable.
- **No published result metric is typed by hand.** `scripts/evaluate.py` reads
  `reports/*_metrics.csv`; it computes nothing. There is no code path from a
  fixture to a published table: sample mode opens no MLflow/W&B run and writes
  no checkpoint or metrics CSV, while dashboard history requires
  `sample=false` plus matching problem/config tags.

## Explainability

Two explainers, dispatched on the checkpoint's model type by
[`src/serving/explain.py`](src/serving/explain.py):

- **SHAP `TreeExplainer`** for LightGBM and XGBoost uses exact Tree SHAP on the
  request path and returns signed per-feature contributions.
- **Input-gradient attribution** for the autoencoder baseline, because
  `TreeExplainer` does not apply. The model is a symmetric autoencoder with six
  `nn.Linear` transforms and a 16-unit bottleneck; model-agnostic explanation is
  not used on the request path.

An unknown model type raises `NotImplementedError` → HTTP 501 rather than
returning an empty explanation that a UI would render as "no important features".

The `/explain/*` routes reuse the exact feature frame the score was computed
from, so the explanation always describes the number beside it.

## Testing

```bash
make test       # excludes the live-Kaggle canary
make test-all   # includes it (needs credentials)
make lint typecheck
make verify     # clone HEAD into a temp dir and run this README's quickstart
```

Checkpoint-dependent quality gates skip with an explicit training command when a
real checkpoint is absent. The fresh-clone verifier reports that skip separately.
The **88%** coverage badge is the rounded **88.24%** reported by:

```bash
./.venv/bin/pytest -p no:cacheprovider -m "not network" --cov=src --cov-report=term-missing
```

What the gates assert, beyond plumbing:

| Test | Assertion |
|---|---|
| [`tests/data/test_leakage_denylist.py`](tests/data/test_leakage_denylist.py) | No post-origination LendingClub column and neither `Naive_Bayes_Classifier_*` column reaches a model matrix |
| [`tests/training/test_split.py`](tests/training/test_split.py) | Equal timestamps stay in the earlier partition; train, validation, and test timestamp sets are disjoint |
| [`tests/serving/test_skew.py`](tests/serving/test_skew.py) | Serving features == training features, row for row, on a fixture batch |
| [`tests/test_quality_gates.py`](tests/test_quality_gates.py) | Expected-range smoke alarm, model beats its own baseline, monotonic direction |
| [`tests/e2e/test_train_to_serve.py`](tests/e2e/test_train_to_serve.py) | Fixtures → train → checkpoint → HTTP predict in under 30s, and asserts the fixture result is **near chance** |

`pyproject.toml` contains no `-m 'not network and not parity'` filter in
`addopts`. The network exclusion is defined in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Limitations & Known Caveats

- **Three of five rows are measured.** ULB/OpenML and churn have accepted
  out-of-fold results, and LendingClub has a held-out temporal result. IEEE-CIS
  remains credential-blocked, and its autoencoder needs the same data.
- **IEEE-CIS test labels do not exist.** The competition's `test_transaction.csv`
  is unlabelled, so evaluation is a temporal split *within* the training file.
  A random split would inflate AUC by putting the same card and device on both
  sides of the boundary.
- **Churn is not prospective.** `Attrition_Flag` and the trailing-activity
  features describe the same period. The source has no event timestamp, feature
  cutoff, or future outcome window, so the score substantially detects attrition
  that already happened.
- **The churn checkpoint fails one directional serving gate.** On the existing
  fixed probe, more inactivity lowers the score. The test remains enabled; the
  out-of-fold ranking result is published, but the serving score is not a
  monotonic retention policy.
- **The credit-risk thresholds are illustrative, not a credit policy.** A real
  approve/decline cut point comes from expected loss at a target approval rate,
  which needs a pricing model this repository does not contain.
- **Credit risk is post-pricing.** `int_rate`, `grade_ordinal`, and
  `sub_grade_ordinal` encode LendingClub's own origination-time risk pricing.
  The model predicts default given that pricing; it is not an independent
  approval or pricing model. The comparison without those fields has not been
  run.
- **MPS on the host, CPU in Docker.** `src/device.py` selects MPS on Apple
  Silicon, but `infra/docker/api.Dockerfile` force-installs CPU PyTorch. MPS is
  macOS-only and cannot exist in a Linux container. LightGBM and XGBoost ship
  CPU-only wheels on macOS arm64 regardless, so the three headline models gain
  nothing from MPS either way.
- **No cloud deployment.** Docker plus the committed screenshot script is the
  demo.
- **`docs/images/` is empty.** `scripts/capture_screenshots.py` is committed and
  regenerable, but screenshots were not generated in this results-publication
  pass.

## Tech decisions

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-gradio-to-nextjs.md) | Gradio → Next.js + shadcn/ui. Gradio could not provide per-model deep links or the required result-card layouts. |
| [0002](docs/adr/0002-experiment-tracking.md) | MLflow is primary for real runs; sample runs are deliberately untracked. W&B remains optional. |
| [0003](docs/adr/0003-real-data-over-synthetic.md) | **Real public data only.** Every generator is deleted, and the config loader enforces a real data source. |
| [0004](docs/adr/0004-narrow-to-fintech.md) | Ten industries → three fintech problems. Twenty-one catalogued models over four checkpoints did not match the implemented scope. |
| [0005](docs/adr/0005-assumptions-log.md) | Implementation assumptions, each reversible in one commit. |

## License

- Repository licence: MIT. See [LICENSE](LICENSE).
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Author

**Armando Gonzalez:** ex-software engineer at a fintech company, finishing an
M.S. in Data Science & AI at FIU.
[GitHub](https://github.com/armandogon94)
