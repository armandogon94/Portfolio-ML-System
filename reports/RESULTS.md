# Results & Methodology

> **Status: three of five rows are measured; two are not.**
>
> `fraud_ulb`, `credit_risk`, and `churn` are accepted real-data results.
> `fraud` (IEEE-CIS) and `fraud_autoencoder` remain empty and
> explicitly marked **not yet measured** below.
>
> IEEE-CIS is **BLOCKED**: with only the OAuth token in
> `~/.kaggle/access_token`,
> `kagglehub.competition_download('ieee-fraud-detection')` returns
> `403 ... Please make sure you are authenticated and have accepted the
> competition rules`. Kaggle datasets authenticate with that token; Kaggle
> competitions do not. IEEE-CIS needs a classic `~/.kaggle/kaggle.json` API
> token.
>
> The autoencoder needs the blocked IEEE-CIS data.
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
| `credit_risk` | **0.3935** | 0.7160 | 0.3720 | 0.0215 | 0.5807 | 0.0457 | measured · [`credit_risk_metrics.csv`](credit_risk_metrics.csv) |
| `churn` | **0.9735 ± 0.0078** | 0.9940 ± 0.0019 | 0.7800 ± 0.0217 | 0.1935 ± 0.0145 | 1.0000 ± 0.0000 | 0.8869 ± 0.0317 | measured · [`churn_metrics.csv`](churn_metrics.csv) |

<img src="figures/precision_recall_curves.png" alt="Precision-recall curves from out-of-fold predictions. The published five-fold mean PR-AUC is 0.8569 ± 0.0331 for LightGBM versus 0.7300 ± 0.0279 for logistic regression on ULB fraud, and 0.9735 ± 0.0078 versus 0.7800 ± 0.0217 on card attrition.">

The curves pool one held-out prediction per row, while the table reports the
mean ± standard deviation of the five fold scores. The credit-risk run has
metrics but no saved per-row predictions, so it is not plotted.

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
The retraction narrative deliberately quotes these values. The gate is that no
retracted number may appear in an **active** results table.

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
`rng.normal(1.0, 0.10)`. The target was therefore constructed directly from the
features by the same codebase that later reported R² 0.942.

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
leaks information about the test distribution and does not represent deployment.

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

<img src="figures/calibration_curves.png" alt="Log-scale calibration plots with ten equal-count bins and 95% Wilson intervals. ULB fraud has a Brier score of 0.00039 with 3 of 10 bins containing no observed positives; card attrition has a Brier score of 0.02096 with 4 of 10 bins containing no observed positives.">

The calibration panels retain all ten bins on log scales. Whiskers are 95%
Wilson intervals; bins with no observed positives show their measured upper
limit without a point marker. The plotted Brier scores are the fold means from
the corresponding metrics CSVs. Credit risk is absent because its temporal run
did not save per-row predictions.

---

## 1. Payment fraud — IEEE-CIS

**Dataset.** 590,540 transactions × 394 columns, ~3.5% fraudulent, contributed by
Vesta Corporation. Joined with a 20-column subset of `train_identity.csv`.
Provenance and licence: [`data/README.md`](../data/README.md).

**Split: time-based on `TransactionDT`.** First 80% of the time range trains,
last 20% tests, with a 10% validation fold carved chronologically from the tail
of train for early stopping. Equal `TransactionDT` values stay on the earlier
side of each boundary, so one timestamp can never appear in two partitions.

This is not a stylistic preference. A random split puts the same card, device,
and billing address on both sides of the boundary, so the model can memorise
entities that will not be available as shared identities in production.
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

**Expected result: ≈ 0.90 ROC-AUC**, a hand-entered config expectation rather
than a measurement. The configured sanity band is 0.85–0.96. A result outside
that range requires investigation of the split, `TransactionID`, and identity
features before publication.

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

**Dataset.** LendingClub accepted loans 2007-2018Q4,
`accepted_2007_to_2018Q4.csv.gz`: **2,260,701 raw rows × 151 columns** and
**392.6 MB**. The SHA-256 is
`55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`.
The uploader tags the file CC0, but the upstream authority is unverified, so
rows are not redistributed. After target filtering, **1,345,310
terminal-status rows** remain with an overall **0.1996 default rate**. The model
uses **32 origination-time features**.

**Target.** `loan_status` filtered to terminal outcomes only: `Fully Paid` → 0,
`Charged Off` → 1. Everything else — `Current`, `In Grace Period`, `Late (…)` —
is dropped, because those loans have not resolved. This removed **40.5%** of the
raw rows. Labelling a `Current` loan as non-default would record a success that
has not happened.

**Split: time-based on `issue_d`.** Consumer credit shifts by vintage — 2015
borrowers are not 2018 borrowers, and the macro environment differs. Training on
earlier vintages and testing on later ones is the only split that resembles how
the model would be used. The config requested 70/10/20. Because `issue_d` is
monthly and timestamp ties stay together, the boundaries moved to month edges
and produced an actual **71.6/8.8/19.6** split:

| Partition | n | Positives | Positive rate | Min `issue_d` | Max `issue_d` |
|---|---:|---:|---:|---|---|
| Train | 962,641 | 181,265 | 0.1883 | 2007-06-01 | 2016-04-01 |
| Validation | 118,689 | 29,840 | 0.2514 | 2016-05-01 | 2016-10-01 |
| Test | 263,980 | 57,454 | 0.2176 | 2016-11-01 | 2018-12-01 |

Zero `issue_d` values appear in more than one partition
(`split_key_leak: false`).

**Main interpretive limitation: terminal-status filtering censors the later
vintages.** The default rate rises from **0.1883** in train to **0.2176** in
test because filtering happens before the split. By the 2018Q4 data cut, the
latest vintages retain only loans that had already resolved, which
over-represents early charge-offs and fast payoffs. This survivorship/censoring
bias is a limitation of the target construction, not a modelling choice. The
held-out result must not be read as an unbiased estimate of forward default
risk.

### The leakage denylist — the most informative thing in this document

`configs/credit_risk.yaml` names 29 entries the model may never see, enforced by
`tests/data/test_leakage_denylist.py` and additionally never read at all
(`src/data/adapters/lending_club.py` uses a `usecols` allowlist of 30
origination-time fields, which the feature module turns into the 32 columns the
model actually sees).

The ones that matter:

`recoveries` · `collection_recovery_fee` · `total_rec_prncp` · `total_rec_int` ·
`total_pymnt` · `last_pymnt_amnt` · `out_prncp` · `debt_settlement_flag` ·
`settlement_*` · `last_fico_range_*` · `hardship_*`

Every one is recorded **after** origination. `recoveries` changes only after a
loan has defaulted. A model that sees it is completely worthless, because the
field is not an origination-time input.

| Configuration | ROC-AUC | Interpretation |
|---|---|---|
| With denylisted columns included |  | *(not yet measured)* |
| Denylist applied (the honest model) |  | *(not yet measured)* |
| Δ |  |  |

That delta is the point. **A correct LendingClub model is not
impressive-looking**; the controlled comparison remains empty until both rows
have been measured by the documented training path.

### Post-pricing risk model, not an approval or pricing model

`int_rate`, `grade_ordinal`, and `sub_grade_ordinal` are LendingClub's own risk
pricing, assigned at origination. Including them makes this a **post-pricing**
model: it estimates default risk given how LendingClub already priced the loan.
It is not an independent approval or pricing model. ROC-AUC would be lower
without those fields, but that comparison has not been run, so no number or
delta is reported for it.

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
| LightGBM | **0.3935** | 0.7160 | 0.5807 | 0.0457 | 0.1551 |
| Logistic regression (baseline) | 0.3720 | 0.6989 | 0.5307 | 0.0405 | 0.2122 |
| Prior (baseline) | 0.2176 | 0.5000 | — | — | 0.1711 |

LightGBM beats logistic regression by only **+0.0215 PR-AUC** (**0.3935** vs
**0.3720**) and **+0.017 ROC-AUC** (**0.7160** vs **0.6989**). On these
origination-time LendingClub features, gradient boosting is barely better than
logistic regression. That narrow margin is the finding.

The logistic baseline uses median imputation and standardisation for numeric
features, plus most-frequent imputation and one-hot encoding capped at 20
categories for categoricals. It was configured with `max_iter: 1000` and
`class_weight: balanced`.

| Run configuration | Recorded value |
|---|---|
| Dataset | Kaggle `wordsforthewise/lending-club`, `accepted_2007_to_2018Q4.csv.gz` |
| Raw / filtered rows | 2,260,701 / 1,345,310 terminal-status rows |
| Overall / test positive rate | 0.1996 / 0.2176 |
| Features | 32 origination-time columns |
| Split | Time on `issue_d`; requested 70/10/20, actual 71.6/8.8/19.6; no split-key overlap |
| Model / seed | LightGBM / 42 |
| Training SHA / time | `052adab726b6dd5a176cfdc737110b172198a749` / `2026-07-26T03:54:55Z` |
| Reproducibility | Trained twice at seed 42 on 2026-07-26. Every metric in `credit_risk_metrics.csv` was identical across both runs; the checkpoint on disk is the second. |
| Hardware | `macOS-26.5.2-arm64-arm-64bit`, arm64; CPU-only LightGBM wheel, no GPU or Metal backend |
| Fit / evaluation scope | 962,641 training-partition rows / 263,980 held-out test rows |
| Wall-clock / early stop | 49.2 s / iteration 513 of 1,500 configured |
| Hyperparameters | `objective=binary`; `metric=average_precision`; `learning_rate=0.05`; `num_leaves=31`; `min_child_samples=200`; `feature_fraction=0.8`; `bagging_fraction=0.8`; `bagging_freq=1`; `n_estimators=1500`; `early_stopping_rounds=100`; `n_jobs=8` |
| Sanity band | ROC-AUC 0.62–0.80; measured 0.7160, inside the band; `sanity_band_warning: null` |
| Config / checkpoint | [`configs/credit_risk.yaml`](../configs/credit_risk.yaml) / `checkpoints/credit_risk/metadata.json` |

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

### Rows no run has produced

No run has produced the `fraud` or `fraud_autoencoder` result tables yet. When
their shared IEEE-CIS data prerequisite is available, the commands will be:

```bash
uv run python scripts/download_data.py --dataset ieee-cis
uv run python scripts/train.py --model fraud
uv run python scripts/train.py --model fraud --autoencoder
```

### Rows produced by completed runs

```bash
uv run python scripts/train.py --model fraud_ulb
uv run python scripts/train.py --model churn
uv run python scripts/download_data.py --dataset lending-club
uv run python scripts/train.py --model credit_risk
uv run python scripts/describe_split.py --model credit_risk
uv run python scripts/evaluate.py --markdown
```

> On 2026-07-25, `fraud_ulb` and `churn` were run. Both checkpoint metadata files
> record training SHA `05d32cae0d54dee97a70be575097182e9b5f8278`,
> seed 42, and `macOS-26.5.2-arm64-arm-64bit` on arm64; LightGBM used its CPU-only
> macOS wheel.
>
> On 2026-07-26 UTC, `credit_risk` was trained with seed 42 at
> `052adab726b6dd5a176cfdc737110b172198a749` on
> `macOS-26.5.2-arm64-arm-64bit`, arm64. The LightGBM CPU-only run took 49.2 s
> wall clock and early-stopped at iteration 513 of 1,500 configured trees. The
> checkpoint was fitted on 962,641 training-partition rows and evaluated on the
> 263,980-row held-out test partition.
>
> `fraud` and `fraud_autoencoder` were not run. `scripts/evaluate.py` computed
> nothing; it read the three training-written CSVs.

LightGBM and XGBoost ship CPU-only wheels on macOS arm64; there is no Metal
backend for either, so the three headline models are CPU-bound regardless of the
MPS availability the autoencoder enjoys.

For cross-validated models, figure captions state that curves and hard-decision
plots use out-of-fold predictions. The all-row refit is used for serving and
feature attribution, not to grade itself.
