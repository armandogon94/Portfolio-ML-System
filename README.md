# Fintech ML System: Fraud, Credit Risk, and Attrition on Real Public Data

[![CI](https://github.com/armandogon94/Portfolio-ML-System/actions/workflows/ci.yml/badge.svg)](https://github.com/armandogon94/Portfolio-ML-System/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Three real-money fintech problems are covered: payment fraud, consumer credit
risk, and card attrition. Three accepted real-data runs have committed aggregate
metrics and generated provenance records; their checkpoints and MLflow store stay
local, and FastAPI serves them when those artifacts are present.

- **Focus:** payment fraud · consumer credit risk · card attrition
- **Data:** [IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (competition-gated, not measured locally) · [LendingClub](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (1,345,310 terminal-status rows in the measured frame; uploader tags CC0, upstream authority unverified) · [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) (10,127 measured rows; uploader tags CC0, upstream authority unverified)
- **Stack:** Python 3.11 · LightGBM · PyTorch (MPS) · SHAP · MLflow · FastAPI · Next.js 14 · Docker
- **Output:** a results table where every measured cell traces to a CSV written by a training run. Three of five rows are measured: `fraud_ulb`, `credit_risk`, and `churn`. The `fraud` and `fraud_autoencoder` rows are not measured because both need the IEEE-CIS competition data.

📄 **[Read the full methodology & analysis →](reports/RESULTS.md)**

---

## Current status

**Three of five result rows are measured on real data: `fraud_ulb`,
`credit_risk`, and `churn`.**

The other two are not measured. Both require IEEE-CIS, whose competition
download needs a classic `kaggle.json` token that is not available here.

[`docs/PROGRESS.md`](docs/PROGRESS.md) lists exactly what is blocked, why, and the
commands to unblock it.

---

## Key Findings

- **The credential-free fraud result beat its logistic baseline on PR-AUC.**
  A chronological ULB/OpenML 1597 test measured PR-AUC
  **0.8073 ± 0.1357** against **0.7461 ± 0.1993** for logistic regression.
  Here and for credit risk, `±` is the standard deviation across five adjacent
  test-time blocks, not a confidence interval.
- **Gradient boosting was barely better than logistic regression on
  LendingClub.** LightGBM measured **0.3935 ± 0.0529 PR-AUC** against
  **0.3720 ± 0.0569**, a **+0.0215 ± 0.0062** margin across temporal blocks.
  The test partition is censored by terminal-status filtering at the 2018Q4
  data cut, so this is not an unbiased estimate of forward default risk.
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
| Fraud | IEEE-CIS · scale and class rate not measured locally | configured time split on `TransactionDT` | LightGBM | not measured | not measured | not measured | not measured | blocked; no result artifact |
| Fraud (`fraud_ulb`, credential-free) | ULB / OpenML 1597 · 284,807 rows · 29 features · 0.1727% | chronological 70/10/20 on `Time` | LightGBM | **0.8073 ± 0.1357** | 0.9828 ± 0.0194 | 0.7461 ± 0.1993 | 0.0612 ± 0.0781 | [`metrics`](reports/fraud_ulb_metrics.csv) · [`run provenance`](reports/fraud_ulb_run.json) |
| Fraud (unsup. baseline) | IEEE-CIS · scale and class rate not measured locally | configured time split on `TransactionDT` | Autoencoder (MPS) | not measured | not measured | not measured | not measured | blocked; no result artifact |
| Credit risk | LendingClub · 1,345,310 terminal-status rows · 32 features · 19.96% default | time on `issue_d` | LightGBM | **0.3935 ± 0.0529** | 0.7160 ± 0.0076 | 0.3720 ± 0.0569 | 0.0215 ± 0.0062 | [`metrics`](reports/credit_risk_metrics.csv) · [`run provenance`](reports/credit_risk_run.json) |
| Churn | CC attrition · 10,127 rows · 23 features · 16.07% | 5-fold stratified CV | LightGBM | **0.9735 ± 0.0078** | 0.9940 ± 0.0019 | 0.7800 ± 0.0217 | 0.1935 ± 0.0145 | [`metrics`](reports/churn_metrics.csv) · [`run provenance`](reports/churn_run.json) |

The `fraud` and `fraud_autoencoder` rows need the IEEE-CIS competition data, which requires Kaggle credentials that are not available here.

<sub>`scripts/evaluate.py` reads every measured cell from training-written
`reports/*_metrics.csv`; it computes nothing. Full config, caveats, and provenance:
[`reports/RESULTS.md`](reports/RESULTS.md).</sub>

<sub>For churn, the displayed and gated result is the stratified-CV mean ± fold
standard deviation; its serving checkpoint is refit on all rows. For the two
chronological runs, the point estimate uses the final held-out test partition and
`±` is the standard deviation across five adjacent, tie-safe test-time blocks.
That spread is descriptive, not a confidence interval. Figures use the same
persisted held-out rows. The autoencoder uses raw reconstruction error for ranking
metrics and reports no Brier score because the error is not a calibrated
probability.</sub>

<img src="reports/figures/calibration_curves.png" alt="Log-scale calibration plots with ten equal-count bins and 95% Wilson intervals. ULB fraud has a held-out Brier score of 0.00041 with 6 of 10 bins containing no observed positives; consumer credit risk has a held-out Brier score of 0.15507 with 0 of 10 bins containing no observed positives; card attrition has a mean fold Brier score of 0.02096 with 4 of 10 bins containing no observed positives.">

Observed event rates rise across the ordered score bins in all three measured
datasets. The whiskers are 95% Wilson intervals; zero-event bins show only their
measured upper limit down to a data-derived axis floor. This does not support
calibration claims for future data, a policy threshold, or causal interpretation.

**PR-AUC is the primary metric.** On the chronological ULB test, ROC-AUC
**0.9828 ± 0.0194** does not express positive predictive value at the source
**0.1727%** base rate. The operationally informative comparison is
**0.8073 ± 0.1357 PR-AUC** against the logistic-regression baseline of
**0.7461 ± 0.1993**.

<img src="reports/figures/precision_recall_curves.png" alt="Precision-recall curves from held-out predictions. Published PR-AUC values are 0.8073 ± 0.1357 versus 0.7461 ± 0.1993 on ULB fraud, 0.3935 ± 0.0529 versus 0.3720 ± 0.0569 on consumer credit risk, and 0.9735 ± 0.0078 versus 0.7800 ± 0.0217 on card attrition.">

The curves pool one held-out prediction per row. The chronological spreads come
from adjacent test-time blocks; the churn headline is the mean and standard
deviation of five folds. LightGBM's PR point estimate exceeds logistic
regression on all three measured sets, but the ULB delta spread includes zero.
This does not support a causal claim, a deployment threshold, statistical
significance, or prospective churn forecasting.

**The churn result measures detection, not forecasting.** `Attrition_Flag` is current status while
its strongest features summarise the same trailing activity window. With no
event timestamp, feature cutoff, or future outcome window, the dataset supports
detection of an attrition that already happened, not prospective prediction.

---

## Architecture

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

The API discovers models by globbing `checkpoints/*/metadata.json`. Adding a
model needs no route-code change, but its checkpoint still has to be mounted or
deployed. A fresh clone with zero
checkpoints reports `status: ok` with three unavailable models without entering
a crash loop.
[SVG](docs/diagrams/c4-container.svg) · [full notes](docs/architecture.md)

## How a prediction happens

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

`src/serving/registry.py` loads the checkpoint, training and serving both call
`src/features/fraud_features.py`, and explanations use
`src/explainability/shap_explainer.py`. This shared implementation prevents
training/serving skew.
`tests/serving/test_skew.py` scores the same rows through both paths and demands
identical matrices. [SVG](docs/diagrams/sequence-predict.svg)

## Data pipeline

```mermaid
%%{init: {'htmlLabels': false, 'fontFamily': 'arial, helvetica, sans-serif', 'flowchart': {'htmlLabels': false, 'padding': 16, 'nodeSpacing': 60, 'rankSpacing': 70, 'useMaxWidth': true}}}%%
flowchart LR
    K1[("Kaggle competition<br/>ieee-fraud-detection<br/>scale not measured locally")]
    K2[("Kaggle dataset<br/>wordsforthewise/lending-club<br/>source rows measured")]
    K3[("Kaggle dataset<br/>sakshigoyal7/<br/>credit-card-customers<br/>10,127 rows")]
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

The split happens **before** feature engineering. Frequency encodings and group
aggregates are fitted on the training rows only; fitting them on the full frame
leaks information about the test distribution into training. The dotted edge
shows fixture isolation: fixtures reach the trainer but open no tracking run and
cannot reach `checkpoints/` or `reports/`.
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
├── conftest.py                 # imports xgboost+lightgbm before torch to avoid libomp conflicts
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
make figures         # published PR and calibration figures -> reports/figures/
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
| LendingClub 2007-2018Q4 | Free Kaggle | No | Uploader tags CC0; upstream authority unverified. Rows are not cleared for redistribution. |
| Credit Card Customers | Free Kaggle | No | Uploader tags CC0; upstream authority unverified. Rows are not cleared for redistribution. |
| ULB Credit Card Fraud (OpenML 1597) | **None** | No | Unresolved: OpenML records only "Public"; the Kaggle mirror indicates ODbL-style terms. Rows are not cleared for redistribution. |

IEEE-CIS sits behind a Kaggle account *and* an acceptance of the competition
rules that cannot be scripted. `~/.kaggle/access_token` authenticates Kaggle
**datasets**, but
`kagglehub.competition_download('ieee-fraud-detection')` returns 403 with the
available credentials; the competition path needs a classic
`~/.kaggle/kaggle.json` API token. The ULB/OpenML path needs **no account at
all**, so a reviewer with zero Kaggle presence can reproduce a real-data fraud
result with the commands above.

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
- **Every published run has a committed record.**
  `reports/*_run.json` is generated from checkpoint metadata and cross-checks the
  aggregate metrics CSV byte source, seed, training commit, dataset identity,
  split, and warning state without publishing model or row-level data.
- **`/predict` returns `model_version`:** the short git SHA from that metadata is
  included in every response, linking the score to its training commit.
- **Published result cells are checked against generated records.**
  `scripts/evaluate.py` validates `reports/*_metrics.csv`, while the test suite
  cross-checks each measured CSV against its generated `reports/*_run.json`.
  Sample mode opens no MLflow/W&B run and writes no checkpoint, run record, or
  metrics CSV.

## Explainability

Two explainers, dispatched on the checkpoint's model type by
[`src/serving/explain.py`](src/serving/explain.py):

- **SHAP `TreeExplainer`** for LightGBM and XGBoost uses exact Tree SHAP on the
  request path and returns signed per-feature contributions.
- **Input-gradient attribution** for the autoencoder baseline, because
  `TreeExplainer` does not apply. The model is a symmetric autoencoder with six
  `nn.Linear` transforms and a 16-unit bottleneck; model-agnostic explanation is
  not used on the request path.

An unknown model type raises `NotImplementedError` → HTTP 501. Empty explanation
payloads do not reach the UI.

The `/explain/*` routes reuse the exact feature frame the score was computed
from, so the explanation always describes the number beside it.

## Testing

```bash
make test       # excludes the live-Kaggle canary
make test-all   # includes it (needs credentials)
make lint typecheck
make publication-check
make verify     # clone HEAD into a temp dir and run this README's quickstart
```

Checkpoint-dependent quality gates skip with an explicit training command when a
real checkpoint is absent. The fresh-clone verifier reports that skip separately.
The enforced coverage floor is defined in `pyproject.toml` and applied by both
local pytest configuration and CI; no hand-maintained coverage result is
published.

The gates assert:

| Test | Assertion |
|---|---|
| [`tests/data/test_leakage_denylist.py`](tests/data/test_leakage_denylist.py) | No post-origination LendingClub column and neither `Naive_Bayes_Classifier_*` column reaches a model matrix |
| [`tests/training/test_split.py`](tests/training/test_split.py) | Equal timestamps stay in the earlier partition; train, validation, and test timestamp sets are disjoint |
| [`tests/serving/test_skew.py`](tests/serving/test_skew.py) | Serving features == training features, row for row, on a fixture batch |
| [`tests/test_quality_gates.py`](tests/test_quality_gates.py) | Expected-range smoke alarm, model beats its own baseline, monotonic direction |
| [`tests/e2e/test_train_to_serve.py`](tests/e2e/test_train_to_serve.py) | Fixtures → train → checkpoint → HTTP predict in under 30s, and asserts the fixture result is **near chance** |
| [`scripts/check_publication.py`](scripts/check_publication.py) | Active result cells, split counts, figure-caption measurements, metric digests, and PNG bytes match generated evidence |

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
- **The churn score is not a monotonic retention policy.** The directional
  serving gate checks only the empirically monotone one-to-four-month segment;
  the source relationship reverses outside that segment. The out-of-fold
  ranking result does not support monotonicity over the full input range.
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
- **No screenshots are tracked.** `scripts/capture_screenshots.py` is committed
  and regenerable, but capture requires the live stack and a trained checkpoint.

## Architectural Decisions

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-gradio-to-nextjs.md) | Gradio → Next.js + shadcn/ui. Gradio could not provide per-model deep links or the required result-card layouts. |
| [0002](docs/adr/0002-experiment-tracking.md) | MLflow is primary for real runs; sample runs do not write tracking records. W&B remains optional. |
| [0003](docs/adr/0003-real-data-over-synthetic.md) | **Real public data only.** Every generator is deleted, and the config loader enforces a real data source. |
| [0004](docs/adr/0004-narrow-to-fintech.md) | Ten industries → three fintech problems. Twenty-one catalogued models over four checkpoints did not match the implemented scope. |
| [0005](docs/adr/0005-assumptions-log.md) | Implementation assumptions, each reversible in one commit. |

## License

- Repository licence: MIT. See [LICENSE](LICENSE).
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Author

**Armando Gonzalez, AI/ML Engineer, M.S. in Data Science and AI**
[GitHub](https://github.com/armandogon94)
