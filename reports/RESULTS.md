# Results & Methodology

> **Status: two of five rows are measured; three are not.**
>
> `fraud_ulb` and `churn` are accepted real-data, out-of-fold results.
> `fraud` (IEEE-CIS), `fraud_autoencoder`, and `credit_risk` remain empty and
> explicitly marked **not yet measured** below.
>
> IEEE-CIS is **BLOCKED**: with only the OAuth token in
> `~/.kaggle/access_token`,
> `kagglehub.competition_download('ieee-fraud-detection')` returns
> `403 ... Please make sure you are authenticated and have accepted the
> competition rules`. Kaggle datasets authenticate with that token; Kaggle
> competitions do not. IEEE-CIS needs a classic `~/.kaggle/kaggle.json` API
> token. LendingClub was not run; its 648 MB download was not attempted in this
> session. The autoencoder needs the blocked IEEE-CIS data.
>
> Every accepted metric below is read from `reports/*_metrics.csv`, written by
> training. Empty cells remain empty rather than becoming estimates.

---

## Accepted real-data results

**PR-AUC is the primary metric.** It measures ranking quality where the positive
class is rare; ROC-AUC is retained only as a familiar secondary diagnostic.

| Run | **PR-AUC ↑ (primary)** | ROC-AUC ↑ | Logistic-regression PR-AUC | Δ PR-AUC | P@1% ↑ | R@1%FPR ↑ | Status |
|---|---|---|---|---|---|---|---|
| `fraud` |  |  |  |  |  |  | **not yet measured — BLOCKED:** IEEE-CIS competition download needs a classic `kaggle.json` token |
| `fraud_ulb` | **0.8569 ± 0.0331** | 0.9810 ± 0.0092 | 0.7300 ± 0.0279 | 0.1269 ± 0.0355 | 0.1544 ± 0.0043 | 0.8964 ± 0.0219 | measured · [`fraud_ulb_metrics.csv`](fraud_ulb_metrics.csv) |
| `fraud_autoencoder` |  |  |  |  |  |  | **not yet measured — BLOCKED:** needs the same IEEE-CIS data |
| `credit_risk` |  |  |  |  |  |  | **not yet measured:** the 648 MB LendingClub download was not attempted in this session |
| `churn` | **0.9735 ± 0.0078** | 0.9940 ± 0.0019 | 0.7800 ± 0.0217 | 0.1935 ± 0.0145 | 1.0000 ± 0.0000 | 0.8869 ± 0.0317 | measured · [`churn_metrics.csv`](churn_metrics.csv) |

### `fraud_ulb`: PR-AUC is the informative result

At a **0.1727% positive rate**, ROC-AUC **0.9810 ± 0.0092** looks spectacular
and means very little: the prior baseline scores **0.5000 ROC-AUC** while
achieving only **0.0017 PR-AUC**. The informative comparison is the primary
metric, **PR-AUC 0.8569 ± 0.0331**, against the logistic-regression baseline of
**0.7300 ± 0.0279**.

| Run configuration | Recorded value |
|---|---|
| Dataset | OpenML 1597, ULB Credit Card Fraud |
| Rows / positive rate / features | 284,807 / 0.1727% / 29 |
| Split | Stratified 5-fold cross-validation; every reported score is out of fold |
| Model / seed | LightGBM / 42 |
| Training SHA | `05d32cae0d54dee97a70be575097182e9b5f8278` |
| Hardware | `macOS-26.5.2-arm64-arm-64bit`, arm64; CPU-only LightGBM wheel |
| Wall-clock | 67.8 s |
| Config / checkpoint | [`configs/fraud_ulb.yaml`](../configs/fraud_ulb.yaml) / `checkpoints/fraud_ulb/metadata.json` |

**Sanity-band adjudication.** The band was set by hand before the run at
PR-AUC **0.70–0.85**. The measured **0.8569** came in above it, so the checkpoint
correctly recorded `sanity_band_warning`; the warning was investigated rather
than ignored. The feature matrix is the 28 PCA components plus `Amount`, while
the target and source `Class` are denylisted. OpenML 1597 supplies no timestamp
or entity identifier that a stratified fold could straddle. The
logistic-regression baseline also reached **0.7300** instead of being left
behind, which is not the signature of a target leak. After that investigation,
`configs/fraud_ulb.yaml` was widened to **0.90** with an inline dated comment.
That was a post-run threshold change from **0.85** on **2026-07-25**, not a
pre-registered maximum, and the config and this report say so explicitly.

### `churn`: detection of an attrition that already happened

The **0.9735 ± 0.0078 PR-AUC** is not a prospective forecast. `Attrition_Flag`
is a *current* status while features such as `Total_Trans_Ct`,
`Total_Ct_Chng_Q4_Q1`, and `Months_Inactive_12_mon` summarise the same trailing
12 months. The dataset has no event timestamp, feature cutoff, or future outcome
window. A customer who has already left has collapsed activity in exactly the
window those features describe, so this score substantially reflects
**detecting an attrition that already happened**, not forecasting one. The
dataset cannot support prospective framing at all; that is a property of the
data, not a pipeline bug.

| Run configuration | Recorded value |
|---|---|
| Dataset | Kaggle `sakshigoyal7/credit-card-customers`, `BankChurners.csv` |
| Rows / positive rate / features | 10,127 / 16.07% / 23 |
| Split | Stratified 5-fold cross-validation; every reported score is out of fold |
| Model / seed | LightGBM / 42 |
| Training SHA | `05d32cae0d54dee97a70be575097182e9b5f8278` |
| Hardware | `macOS-26.5.2-arm64-arm-64bit`, arm64; CPU-only LightGBM wheel |
| Wall-clock | 11.1 s |
| Config / checkpoint | [`configs/churn.yaml`](../configs/churn.yaml) / `checkpoints/churn/metadata.json` |

The two `Naive_Bayes_Classifier_*` posterior columns are absent from the
23-feature checkpoint list. The configured denylist held.

**A quality gate fired, and the gate turned out to be wrong.** The first real
churn checkpoint failed `test_more_inactive_months_does_not_lower_churn_risk`,
which probed 0 against 6 inactive months. Before touching either side, the
empirical base rates were measured over all 10,127 rows:

| `Months_Inactive_12_mon` | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| attrition rate | 51.7% | 4.5% | 15.4% | 21.5% | 29.9% | 18.0% | 15.3% |
| n | 29 | 2,233 | 3,282 | 3,846 | 435 | 178 | 124 |

Attrition is **not** monotonic in inactivity in this dataset. The 0-month cell is
29 customers, and the rate falls again after 4 months. A model scoring 6 months
above 0 months would be contradicting its own training data, so the gate was
asserting a bug rather than catching one.

The gate now probes **1 against 4 months** — the segment where the relationship
is monotone increasing (4.5% → 29.9%) and every cell has hundreds to thousands of
rows behind it — and the checkpoint passes it. Widening it back to 0..6 needs
different data, not a different model. The test was corrected, not relaxed: it
still fails a model wired backwards over the range where "backwards" is defined.

This is also a caution about the serving surface: the score is genuinely
non-monotonic in inactivity at the extremes, so it ranks well but must not be
read as a retention *policy*.

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

**Chronological boundaries are tie-safe.** The row-count cut moves forward to the
next change in the split key, assigning a tied timestamp to the earlier partition.
Validation and test therefore contain only strictly later events. If moving a
boundary would empty a requested partition, training fails with an instruction to
change the proportions or use a finer-grained time key.

**Baselines are mandatory.** Every run trains a prior-probability baseline and a
logistic-regression baseline on the same matrix, and reports
`test_pr_auc_delta` (or its cross-validation mean) against the stronger of the
two. A PR-AUC with no baseline beside it is uninterpretable: 0.30 is excellent at
3.5% positives and embarrassing at 40%.

**No hyperparameter search has been run.** Main-model parameters were chosen by
hand; the baseline estimators use their library defaults except where a config
states otherwise. The resulting comparison is therefore **not** a fair-tuning
comparison. If a tuning budget is spent later, the same budget must be spent on
both the main model and its baselines.

**PR-AUC is the primary metric, not ROC-AUC.** At 3.5% positives (IEEE-CIS) or
0.1727% (ULB), ROC-AUC is dominated by the enormous true-negative mass and a weak
model still scores above 0.8. Average precision moves when the top of the ranking
changes, which is the only part a review queue ever sees. ROC-AUC is reported as
a secondary number because it is what everyone expects to see.

**Two operational metrics are reported alongside both AUCs**, because they are
what a fintech reviewer actually asks about:

- `precision_at_1pct` — of the top 1% of transactions by score, the slice a human
  review team would work, what fraction are truly positive? The queue's hit rate.
- `recall_at_1pct_fpr` — at a 1% false-positive budget, what fraction of the
  positives do we catch? The loss-prevention number.

**Each config declares an expected-range sanity band.** `sanity_band` is only a
smoke alarm: a result outside it is recorded as `sanity_band_warning` and must be
investigated, but the band cannot prove or disprove leakage. The actual leakage
defence is structural: chronological splitting, split-column exclusion, the
per-problem denylist, and the tests that assert those controls.

For stratified cross-validation, the headline and sanity-band metric is always
the **CV mean with its standard deviation**. Fold metrics never retain ambiguous
`test_*` names. The saved estimator is refit on all rows after evaluation, while
the PR, ROC, calibration, and confusion-matrix figures use persisted out-of-fold
predictions—each plotted row was scored by a model that did not train on it.

---

## 1. Payment fraud — IEEE-CIS

**Dataset.** 590,540 transactions × 394 columns, ~3.5% fraudulent, contributed by
Vesta Corporation. Joined with a 20-column subset of `train_identity.csv`.
Provenance and licence: [`data/README.md`](../data/README.md).

**Split: time-based on `TransactionDT`.** First 80% of the time range trains,
last 20% tests, with a 10% validation fold carved chronologically from the tail
of train for early stopping. Equal `TransactionDT` values stay on the earlier
side of each boundary, so one timestamp can never appear in two partitions.

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

**Point-in-time limitation.** The frequency counts and `uid_amt_mean` are fitted
on training rows only, so they do not see validation or test data. They are not,
however, computed as-of each training event: an early row can benefit from later
rows inside the same training window. This makes the offline estimate
**optimistic** relative to deployment, where only prior transactions would exist.
A full event-time feature-store rewrite is deliberately outside this baseline.

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
| Autoencoder (unsupervised, MPS) |  |  |  |  | not reported — uncalibrated |
| Logistic regression (baseline) |  |  |  |  |  |
| Prior (baseline) |  |  | — | — |  |

*Not yet measured.* The autoencoder row is included because the gap between it
and the supervised model is the interesting number: it quantifies what
supervision buys over pure anomaly detection on this problem.

The autoencoder reports ROC-AUC, PR-AUC, precision@1%, and recall@1%FPR on **raw
reconstruction error**. Raw error supplies an ordering but is not a probability,
so no Brier or calibration metric is reported. Precision, recall, and F1 use the
95th-percentile reconstruction-error threshold fitted on legitimate training
rows; the test set never fits a transform or threshold.

### Error analysis

*To be written from the trained model.* The questions this section will answer:
where does the model fail (new cards? a specific `ProductCD`? the last week of
the test window, where drift is largest?), and does performance decay
monotonically with distance from the training period.

---

## 1b. Payment fraud — ULB / OpenML 1597

OpenML returned **284,807 rows × 30 columns**: `V1`–`V28`, `Amount`, and
`Class`. It did **not** return `Time`, so this dataset cannot support a temporal
split. Row order is not substituted for an undocumented timestamp; evaluation
uses stratified 5-fold cross-validation.

| Model | **PR-AUC ↑ (primary)** | ROC-AUC ↑ | P@1% ↑ | R@1%FPR ↑ |
|---|---|---|---|---|
| LightGBM | **0.8569 ± 0.0331** | 0.9810 ± 0.0092 | 0.1544 ± 0.0043 | 0.8964 ± 0.0219 |
| Logistic regression (baseline) | 0.7300 ± 0.0279 |  |  |  |
| Prior (baseline) | 0.0017 | 0.5000 | — | — |

Every headline value is the mean ± standard deviation across held-out folds.
After cross-validation, the serving estimator was refit on all **284,807** rows;
it did not grade itself. The adjacent result caveat and sanity-band adjudication
above are part of this result, not footnotes.

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
| With denylisted columns included |  | *(not yet measured)* |
| Denylist applied (the honest model) |  | *(not yet measured)* |
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

*Not yet measured.* The 648 MB LendingClub download was not attempted in this
session.

---

## 3. Card attrition

**Dataset.** 10,127 rows × 23 columns, 16.07% attrited.

**n = 10,127 is small.** A single hold-out number on this dataset is noise
wearing a decimal point. Evaluation is **5-fold stratified cross-validation** and
every metric is reported as mean ± standard deviation. Reporting only the mean
would be half the story; the std is what tells you whether a 0.02 difference
between two configurations means anything. The checkpoint is a separate final
estimator refit on all 10,127 rows after cross-validation; performance figures
come from `reports/churn_oof_predictions.csv`, never from that in-sample refit.

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
| Without them (the honest model) | 0.9940 ± 0.0019 |

The leaky-column comparison itself is **not yet measured**. The honest model was
measured with both columns absent. The adapter deliberately **keeps** the source
columns in the returned frame so a future controlled comparison is possible;
`configs/churn.yaml` stops them reaching the model, and
`tests/data/test_leakage_denylist.py` asserts that a typo cannot silently disable
the protection.

`CLIENTNUM` is denylisted for the same reason: it is a row identifier.

### Results

| Model | PR-AUC (mean ± std) | ROC-AUC (mean ± std) | P@1% | R@1%FPR |
|---|---|---|---|---|
| LightGBM | **0.9735 ± 0.0078** | 0.9940 ± 0.0019 | 1.0000 ± 0.0000 | 0.8869 ± 0.0317 |
| Logistic regression (baseline) | 0.7800 ± 0.0217 |  |  |  |
| Prior (baseline) |  |  | — | — |

**Caveat, stated next to the number:** this is substantially detection of a
current attrition that already happened, not a prospective forecast. The target
and predictors describe the same trailing activity window; the dataset provides
no event timestamp, feature cutoff, or future outcome window from which to build
a prospective task.

---

## Reproducing this document

```bash
uv run python scripts/train.py --model fraud_ulb
uv run python scripts/train.py --model churn
./.venv/bin/python scripts/evaluate.py --markdown
```

> On 2026-07-25, only `fraud_ulb` and `churn` were run. Both checkpoint metadata
> files record training SHA `05d32cae0d54dee97a70be575097182e9b5f8278`,
> seed 42, and `macOS-26.5.2-arm64-arm-64bit` on arm64; LightGBM used its CPU-only
> macOS wheel. The current repository HEAD used to render this document is
> `e15842a`. `fraud`, `fraud_autoencoder`, and `credit_risk` were not run.
> `scripts/evaluate.py` computed nothing; it read the two training-written CSVs.

LightGBM and XGBoost ship CPU-only wheels on macOS arm64; there is no Metal
backend for either, so the three headline models are CPU-bound regardless of the
MPS availability the autoencoder enjoys.

For cross-validated models, figure captions state that curves and hard-decision
plots use out-of-fold predictions. The all-row refit is used for serving and
feature attribution, not to grade itself.
