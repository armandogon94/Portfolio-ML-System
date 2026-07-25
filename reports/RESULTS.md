# Results & Methodology

> **Status: no result has been measured yet.**
>
> Every metric table below has empty cells, deliberately. The pipeline that
> would fill them is complete and tested; the datasets require a Kaggle account
> this build did not have. See [`docs/PROGRESS.md`](../docs/PROGRESS.md) for the
> exact commands.
>
> Filling these tables by hand would reproduce the failure this document exists
> to correct. Every number here will be read from `reports/*_metrics.csv`, which
> is written by `src/training/tabular.py` and by nothing else.

---

## Why the previous results were invalid

The version of this repository before the rebuild published four metrics:

| Problem | Metric | Score |
|---|---|---|
| Fraud Detection | AUC-ROC | 0.964 |
| Credit Risk | AUC-ROC | 0.888 |
| Real Estate Pricing | R² | 0.942 |
| Demand Forecasting | Avg MAE | 22.2 |

All four reproduced exactly from committed CSVs. None of them measured anything.

**Fraud.** `src/data/generate_fraud.py` generated two populations from two
distributions and took `is_fraud` as a *function parameter*:

```python
normal = _generate_transactions(rng, n_normal, is_fraud=False)
fraud  = _generate_transactions(rng, n_fraud,  is_fraud=True)

if is_fraud:  transaction_amount = rng.lognormal(mean=5.5, sigma=1.5, ...)
else:         transaction_amount = rng.lognormal(mean=3.5, sigma=1.0, ...)
```

The classifier's task was to separate `lognormal(5.5, 1.5)` from
`lognormal(3.5, 1.0)`. 0.964 is a measurement of how far apart those two
distributions were placed.

**Credit risk and churn.** Both generators computed a hand-written logistic
score from the features and drew the label from it:
`z = -3.0 + 0.02*(40-age) - 1.5*((credit_score-680)/100) + …`, then
`is_default = rng.random() < sigmoid(z)`. XGBoost recovering a logistic function
that the author wrote is a unit test for `model.fit()`, not a credit model.

**Housing.** Price was a polynomial of the features times
`rng.normal(1.0, 0.10)`. R² 0.942 is approximately the arithmetic ceiling of 10%
injected multiplicative noise — the model could not have scored much differently
whatever it did.

The old README disclosed that the data was "synthetic". That was true and
insufficient: readers hear "synthetic" as "simulated but structurally realistic".
What was actually true is that **the label was a closed-form function of the
features, written by the person reporting the score.** See
[ADR-0003](../docs/adr/0003-real-data-over-synthetic.md).

---

## Method

### Common to all three problems

**Splitting comes before feature engineering.** Frequency encodings and group
aggregates are fitted on the training rows only and carried into validation, test
and serving as artifacts stored in the checkpoint. Fitting them on the full frame
leaks the test distribution — a subtle, popular mistake worth roughly a point of
AUC that does not survive deployment.

**Baselines are mandatory.** Every run trains a prior-probability baseline and a
logistic-regression baseline on the same matrix, and reports
`test_pr_auc_delta` against the stronger of the two. A PR-AUC with no baseline
beside it is uninterpretable: 0.30 is excellent at 3.5% positives and
embarrassing at 40%.

**PR-AUC is the primary metric, not ROC-AUC.** At 3.5% positives (IEEE-CIS) or
0.17% (ULB), ROC-AUC is dominated by the enormous true-negative mass and a weak
model still scores above 0.8. Average precision moves when the top of the ranking
changes, which is the only part a review queue ever sees. ROC-AUC is reported as
a secondary number because it is what everyone expects to see.

**Two operational metrics are reported alongside both AUCs**, because they are
what a fintech reviewer actually asks about:

- `precision_at_1pct` — of the top 1% of transactions by score, the slice a human
  review team would work, what fraction are truly positive? The queue's hit rate.
- `recall_at_1pct_fpr` — at a 1% false-positive budget, what fraction of the
  positives do we catch? The loss-prevention number.

**A ceiling is enforced as well as a floor.** Each config declares
`expected.roc_auc_{min,max}`. A result above the ceiling is written into
`checkpoints/<problem>/metadata.json` as `suspected_leakage` and fails
`tests/test_quality_gates.py`. The failure this repository is correcting was a
good-looking number nobody interrogated.

---

## 1. Payment fraud — IEEE-CIS

**Dataset.** 590,540 transactions × 394 columns, ~3.5% fraudulent, contributed by
Vesta Corporation. Joined with a 20-column subset of `train_identity.csv`.
Provenance and licence: [`data/README.md`](../data/README.md).

**Split: time-based on `TransactionDT`.** First 80% of the time range trains,
last 20% tests, with a 10% validation fold carved chronologically from the tail
of train for early stopping.

This is not a stylistic preference. **A random split inflates the score by
several points**, because the same card, the same device and the same billing
address appear on both sides of a random boundary — the model memorises entities
rather than learning fraud, and the memorised entities are gone in production.
The competition's own `test_transaction.csv` is **unlabelled**, so a temporal
split within the training file is the only honest evaluation available.

**Feature engineering** (`src/features/fraud_features.py`):

| Feature | Rationale |
|---|---|
| `amt_decimal`, `amt_is_round` | The cents portion is signal. Card-testing bots produce round amounts; humans produce 149.99. |
| `tx_hour`, `tx_weekday`, `tx_is_night` | Card testing peaks overnight. Encoded explicitly rather than left for the trees to rediscover. |
| `D1..D15` → `D*_detrend` | The D columns drift with `TransactionDT`; subtracting the transaction day stops the model learning the calendar instead of the behaviour. |
| `card1_freq`, `addr1_freq`, `P_emaildomain_freq`, … | "How often has this card been seen" generalises past the split date; the raw id memorises. |
| `uid_amt_mean`, `uid_amt_ratio` | `card1 + addr1` is the closest thing to a stable account key. The ratio asks "how unusual is this amount *for this account*", which stays comparable across accounts with very different typical spend. |

**Expected result: ≈ 0.90 ROC-AUC**, written into `configs/fraud.yaml` before any
run. Kaggle winning *ensembles* reached 0.94–0.95 on the private leaderboard after
months of feature engineering and blending. A single honest LightGBM on a
temporal split lands near 0.90. **Anything ≥ 0.96 means a leak** — check for a
random split, `TransactionID` in the features, or identity columns joined after
the split. Investigate; do not celebrate.

### Results

| Model | PR-AUC ↑ | ROC-AUC ↑ | P@1% ↑ | R@1%FPR ↑ | Brier ↓ |
|---|---|---|---|---|---|
| LightGBM |  |  |  |  |  |
| Autoencoder (unsupervised, MPS) |  |  |  |  |  |
| Logistic regression (baseline) |  |  |  |  |  |
| Prior (baseline) |  |  | — | — |  |

*Not yet measured.* The autoencoder row is included because the gap between it
and the supervised model is the interesting number: it quantifies what
supervision buys over pure anomaly detection on this problem.

### Error analysis

*To be written from the trained model.* The questions this section will answer:
where does the model fail (new cards? a specific `ProductCD`? the last week of
the test window, where drift is largest?), and does performance decay
monotonically with distance from the training period.

---

## 2. Consumer credit risk — LendingClub

**Dataset.** Accepted loans 2007-2018Q4, ~2.26M rows × 151 columns, CC0.

**Target.** `loan_status` filtered to terminal outcomes only: `Fully Paid` → 0,
`Charged Off` → 1. Everything else — `Current`, `In Grace Period`, `Late (…)` —
is dropped, because those loans have not resolved. Labelling a `Current` loan as
non-default records a success that has not happened and biases the model toward
optimism on recent vintages, which are exactly the vintages with the most
`Current` rows.

**Split: time-based on `issue_d`.** Consumer credit shifts by vintage — 2015
borrowers are not 2018 borrowers, and the macro environment differs. Training on
earlier vintages and testing on later ones is the only split that resembles how
the model would be used.

### The leakage denylist — the most informative thing in this document

`configs/credit_risk.yaml` names ~28 columns the model may never see, enforced by
`tests/data/test_leakage_denylist.py` and additionally never read at all
(`src/data/adapters/lending_club.py` uses a `usecols` allowlist of ~30
origination-time fields).

The ones that matter:

`recoveries` · `collection_recovery_fee` · `total_rec_prncp` · `total_rec_int` ·
`total_pymnt` · `last_pymnt_amnt` · `out_prncp` · `debt_settlement_flag` ·
`settlement_*` · `last_fico_range_*` · `hardship_*`

Every one is recorded **after** origination. `recoveries` is non-zero only for a
loan that has already defaulted. A model that sees it reports ~0.99 ROC-AUC and
is completely worthless, because at decision time the value is always zero.

| Configuration | ROC-AUC | Interpretation |
|---|---|---|
| With denylisted columns included |  | *(to be measured — expect ~0.99)* |
| Denylist applied (the honest model) |  | *(to be measured — expect ~0.70)* |
| Δ |  |  |

That delta is the point. **A correct LendingClub model is not
impressive-looking**, and the ~0.30 gap between the two rows is the difference
between a portfolio number and a model.

**The business framing matters more than the AUC here.** At origination the
decision is not "classify this loan" but "approve at what rate". Expected loss at
a chosen approval threshold — `P(default) × loss_given_default` against interest
income — is the metric a lender optimises, and a ROC-AUC of 0.70 can be entirely
usable under it. The predictor's approve/review/decline thresholds are
illustrative, and `src/serving/predictors/credit_risk.py` says so in its response
body rather than implying a calibrated policy.

### Results

| Model | PR-AUC ↑ | ROC-AUC ↑ | P@1% ↑ | R@1%FPR ↑ | Brier ↓ |
|---|---|---|---|---|---|
| LightGBM |  |  |  |  |  |
| Logistic regression (FICO + DTI + term + grade) |  |  |  |  |  |
| Prior (baseline) |  |  | — | — |  |

*Not yet measured.* Expected band: 0.62 ≤ ROC-AUC ≤ 0.80. Above 0.80 means a
denylisted column got through.

---

## 3. Card attrition

**Dataset.** 10,127 rows × 23 columns, 16.07% attrited.

**n = 10,127 is small.** A single hold-out number on this dataset is noise
wearing a decimal point. Evaluation is **5-fold stratified cross-validation** and
every metric is reported as mean ± standard deviation. Reporting only the mean
would be half the story; the std is what tells you whether a 0.02 difference
between two configurations means anything.

### The Naive-Bayes leakage demonstration

The published CSV ships two columns:

```
Naive_Bayes_Classifier_Attrition_Flag_..._Months_Inactive_12_mon_1
Naive_Bayes_Classifier_Attrition_Flag_..._Months_Inactive_12_mon_2
```

These are pre-computed posterior probabilities of the target — the label,
laundered through a classifier — and the dataset's own author instructs users to
delete them. They are a well-known trap: a notebook that leaves them in reports
near-perfect AUC and has learned nothing.

| Configuration | ROC-AUC (5-fold mean ± std) |
|---|---|
| With both `Naive_Bayes_Classifier_*` columns |  |
| Without them (the honest model) |  |

*Not yet measured.* The adapter deliberately **keeps** both columns in the
returned frame so this comparison is possible; `configs/churn.yaml` is what stops
them reaching the model, and `tests/data/test_leakage_denylist.py` asserts the
config's 130-character strings match the adapter's constants exactly, so a typo
cannot silently disable the protection.

`CLIENTNUM` is denylisted for the same reason: it is a row identifier.

### Results

| Model | PR-AUC (mean ± std) | ROC-AUC (mean ± std) | P@1% | R@1%FPR |
|---|---|---|---|---|
| LightGBM |  |  |  |  |
| Logistic regression (baseline) |  |  |  |  |
| Prior (baseline) |  |  | — | — |

**Caveat, stated before the number rather than after it:** this dataset is small,
well separated and easy. A high score here reflects a property of the data, not
skill. It is not the flagship result and neither the README nor the API presents
it as one — `src/serving/predictors/churn.py` attaches the caveat to every
response body, and the UI renders it.

---

## Reproducing this document

```bash
uv run python scripts/download_data.py --dataset all   # needs a Kaggle token
uv run python scripts/train.py --model all
uv run python scripts/train.py --model fraud --autoencoder
uv run python scripts/evaluate.py --markdown           # emits the table bodies above
uv run python scripts/make_figures.py                  # PR curves, calibration, SHAP
```

> Seed 42. Produced by `make data && make train && make evaluate` on an M-series
> MacBook Pro (32 GB, 4 performance + 6 efficiency cores). Every row is read from
> `reports/*_metrics.csv`, which is written by the training run — **no number is
> typed by hand**, and `scripts/evaluate.py` computes nothing.

LightGBM and XGBoost ship CPU-only wheels on macOS arm64; there is no Metal
backend for either, so the three headline models are CPU-bound regardless of the
MPS availability the autoencoder enjoys.
