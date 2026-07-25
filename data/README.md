# Data

Three real, public datasets. No generator, no simulation, no synthetic
augmentation. See [ADR-0003](../docs/adr/0003-real-data-over-synthetic.md) for
why the previous synthetic pipeline was deleted rather than improved.

Downloads land in `~/.cache/kagglehub/`, outside the repository and outside the
Docker build context.

---

## What this repository redistributes

No rows from any of these datasets are committed. The only committed data files
are the PRNG-generated fixtures in `data/sample/`, produced by
`scripts/make_fixtures.py` with seed 42. Derived and processed artifacts are
gitignored.

---

## Access, stated honestly

| Dataset | Account needed | Extra gate | Redistributable |
|---|---|---|---|
| IEEE-CIS Fraud Detection | Free Kaggle account | **Yes** — one-click acceptance of the competition rules | **No** |
| LendingClub 2007-2018Q4 | Free Kaggle account | No | Uploader tags CC0; upstream authority unverified — do not redistribute rows |
| Credit Card Customers | Free Kaggle account | No | Uploader tags CC0; upstream authority unverified — do not redistribute rows |
| ULB Credit Card Fraud | **None** | No | Unresolved — OpenML records only "Public"; the Kaggle mirror indicates ODbL-style terms. Treat as NOT cleared for redistribution. |

**IEEE-CIS is public and free, but it is not anonymous-`curl`-able.** It sits
behind a Kaggle account *and* an explicit rules acceptance that cannot be
scripted. Anyone claiming otherwise has not tried it. `src/data/download.py`
turns the resulting 403 into the exact URL to visit.

**The ULB path needs no account at all.** It is fetched from OpenML through
`sklearn.datasets.fetch_openml(data_id=1597)`, so a reviewer with zero Kaggle
presence can still reproduce a real-data fraud result end to end:

```bash
uv run python scripts/download_data.py --dataset ulb-creditcard
```

---

## Getting the data

```bash
# What each dataset needs, without downloading anything:
uv run python scripts/download_data.py --check

# One-time Kaggle setup
#   https://www.kaggle.com/settings/account -> "Create New Token"
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# One-time IEEE-CIS rules acceptance (browser, cannot be scripted)
#   https://www.kaggle.com/competitions/ieee-fraud-detection/rules

uv run python scripts/download_data.py --dataset cc-churn       # ~2 MB
uv run python scripts/download_data.py --dataset ieee-cis       # ~118 MB zipped
uv run python scripts/download_data.py --dataset lending-club   # ~648 MB gzipped
```

Without credentials the script prints the remediation and exits non-zero. **There
is no synthetic fallback.** If the data cannot be obtained, nothing is trained
and no number is published.

---

## 1. Payment fraud — IEEE-CIS (Vesta Corporation)

| | |
|---|---|
| Source | <https://www.kaggle.com/competitions/ieee-fraud-detection/data> |
| Licence | Kaggle competition rules — **not redistributable** |
| Scale | `train_transaction.csv` 590,540 × 394; `train_identity.csv` ~144,000 × 41 |
| Positive rate | ~3.5% (20,663 fraudulent transactions) |
| Size | ~118 MB zipped, ~1.35 GB expanded |
| Adapter | [`src/data/adapters/ieee_cis.py`](../src/data/adapters/ieee_cis.py) |
| Target | `isFraud` — ships with the dataset, nothing here derives it |
| Split | Time-based on `TransactionDT`, first 80% train / last 20% test |
| sha256 | *not yet recorded — the download has not been run on this machine* |

Real e-commerce payments contributed by Vesta. Most columns are anonymised:
`card1`–`card6` are card attributes whose meanings are unpublished, `C1`–`C14` are
counting features, `D1`–`D15` are day-deltas, `V1`–`V339` are engineered features
Vesta does not describe. The adapter does not invent meanings for them.

**The competition's `test_transaction.csv` has no labels.** It exists to be scored
by the Kaggle leaderboard. A temporal split *within* the training file is
therefore the only honest evaluation available, and a random split would be
actively wrong: the same card and device appear on both sides of a random
boundary, and the resulting AUC does not survive deployment.

**Memory.** 590,540 × 393 in float32 is about 0.93 GB — comfortable in 32 GB. The
adapter downcasts every numeric column on read and converts strings to pandas
`category`; the naive float64-plus-object load is roughly four times larger.

### Columns the model sees

Not all 394. The adapter reads the transaction base, `C1`–`C14`, `D1`–`D15`,
`M1`–`M9` and `V1`–`V339`, joins a 20-column subset of identity, then
`src/features/fraud_features.py` engineers:

| Feature | Why |
|---|---|
| `amt_decimal`, `amt_is_round` | The cents portion is signal — card-testing bots produce round amounts, humans do not |
| `tx_hour`, `tx_weekday`, `tx_is_night` | Card testing peaks overnight |
| `D*_detrend` | `D` columns drift with `TransactionDT`; subtracting the transaction day stops the model learning the calendar |
| `card1_freq`, `addr1_freq`, … | "How often has this card been seen" generalises; the raw id memorises |
| `uid_amt_ratio` | This amount against the account's typical spend |

Frequency maps are fitted on the **training split only** and carried forward as
artifacts. Fitting them on the full frame leaks the test distribution.

---

## 2. Consumer credit risk — LendingClub 2007-2018Q4

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/wordsforthewise/lending-club> |
| Licence | Uploader tags CC0; upstream authority unverified — do not redistribute rows |
| File | `accepted_2007_to_2018Q4.csv.gz` |
| Scale | ~2,260,701 rows × 151 columns |
| Size | ~648 MB gzipped |
| Adapter | [`src/data/adapters/lending_club.py`](../src/data/adapters/lending_club.py) |
| Target | `is_default`, derived from `loan_status` |
| Split | Time-based on `issue_d` |
| Rows after filtering | *not yet recorded — the download has not been run* |
| sha256 | *not yet recorded* |

The upload originates from LendingClub itself. The uploader tags it CC0, but
the uploader's authority to apply CC0 to the upstream data is undocumented; do
not redistribute rows.

### Target construction

`loan_status` is filtered to **terminal outcomes only**:

```
"Fully Paid"  -> is_default = 0
"Charged Off" -> is_default = 1
```

Everything else — `Current`, `In Grace Period`, `Late (…)`, `Default`, `Issued` —
is **dropped**. Those loans have not resolved. Labelling a `Current` loan as
non-default records a success that has not happened yet and biases the model
toward optimism on recent vintages, which are exactly the ones with the most
`Current` rows.

### The leakage denylist

`configs/credit_risk.yaml` names ~28 columns the model must never see, and
`tests/data/test_leakage_denylist.py` fails if any of them reaches the feature
matrix. The important ones:

`recoveries` · `collection_recovery_fee` · `total_rec_prncp` · `total_rec_int` ·
`total_pymnt` · `last_pymnt_amnt` · `out_prncp` · `debt_settlement_flag` ·
`settlement_*` · `last_fico_range_*` · `hardship_*`

Every one is recorded **after** origination. `recoveries` is non-zero only for a
loan that has already defaulted — a model that sees it reports ~0.99 ROC-AUC and
is worthless, because at decision time the value is always zero. The adapter
additionally never *reads* these columns: `usecols` is an allowlist of ~30
origination-time fields, which is also what keeps a 648 MB file from becoming a
3 GB frame.

`reports/RESULTS.md` will carry the with-denylist / without-denylist delta once
the model has been trained. That contrast is the most informative single
comparison this repository can offer.

---

## 3. Card attrition — Credit Card Customers

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers> |
| Licence | Uploader tags CC0; upstream authority unverified — do not redistribute rows |
| File | `BankChurners.csv` |
| Scale | **10,127 rows × 23 columns** |
| Positive rate | 16.07% attrited |
| Size | < 2 MB |
| Adapter | [`src/data/adapters/credit_card_churn.py`](../src/data/adapters/credit_card_churn.py) |
| Target | `is_attrited`, mapped from `Attrition_Flag` |
| Split | **5-fold stratified cross-validation** |
| sha256 | *not yet recorded* |

The upload originates from `leaps.analyttica.com`. The uploader tags it CC0,
but the uploader's authority to apply CC0 to the upstream data is undocumented;
do not redistribute rows.

**n = 10,127 is small.** A single hold-out number on this dataset is noise wearing
a decimal point, so every metric is reported as a 5-fold mean ± standard
deviation. Reporting only the mean would be half the story.

### The two trap columns

The published CSV ships:

```
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
```

These are pre-computed posterior probabilities of the target — the label,
laundered through a classifier — and the dataset's own author instructs users to
delete them. The adapter **keeps them in the returned frame on purpose** so that
`reports/RESULTS.md` can report the AUC with and without as an explicit leakage
demonstration. `configs/churn.yaml` is what stops them reaching the model, and
`tests/data/test_leakage_denylist.py` asserts the config's 130-character strings
match the adapter's constants exactly, so a typo cannot silently disable the
protection.

`CLIENTNUM` is denylisted too: it is a row identifier.

---

## 4. Ungated fallback — ULB Credit Card Fraud

| | |
|---|---|
| Source | <https://www.openml.org/d/1597> (also Kaggle `mlg-ulb/creditcardfraud`) |
| Licence | Unresolved — OpenML records only "Public"; the Kaggle mirror indicates ODbL-style terms. Treat as NOT cleared for redistribution. |
| Scale | 284,807 transactions × 30 |
| Positive rate | **0.172%** (492 frauds) |
| Access | **No account. No token. No rules acceptance.** |

Features `V1`–`V28` are PCA components; only `Time` and `Amount` are
interpretable. That is why it is the fallback and not the centrepiece: it cannot
demonstrate feature engineering, categorical handling, or a SHAP plot whose axis
labels mean anything to a human. What it *can* do is let a reviewer with no
Kaggle account verify that the pipeline runs on real data.

At 0.172% positives, ROC-AUC on this dataset is close to meaningless. Average
precision is the only metric worth quoting.

---

## CI fixtures — `data/sample/`

| File | Rows | Shape of |
|---|---|---|
| `ieee_cis_sample.csv` | 500 | IEEE-CIS (95 columns: base + C/D/M + 40 V columns) |
| `lending_club_sample.csv` | 500 | LendingClub (a third are `Current`, so the terminal-status filter is exercised) |
| `churn_sample.csv` | 500 | Credit-card attrition (includes both `Naive_Bayes_Classifier_*` columns) |

> **These are synthetic and they are CI fixtures only. No number computed from
> them is ever published.**

They exist for one reason: IEEE-CIS competition data is not redistributable, so
committing 500 real rows would violate the competition rules. This is the only
synthetic data that survived ADR-0003, and it survives for a licensing reason
rather than a convenience one.

**The label in every fixture is drawn independently of the features** — a
Bernoulli draw with no relationship to any column. That is the exact opposite of
what the deleted `generate_fraud.py` did. A model trained on a fixture should
score near chance, and `tests/e2e/test_train_to_serve.py` asserts
`0.25 <= ROC-AUC <= 0.75`: a *good* score on the fixture is treated as a bug in
the fixture, not a result.

Two structural guards keep fixture numbers out of the metric path:

1. `TabularTrainer` with `sample=True` writes **no checkpoint and no metrics
   CSV**. There is no code path from a fixture to `reports/`.
2. `tests/test_quality_gates.py` **skips** when no real checkpoint exists rather
   than passing vacuously.

Regenerate them with:

```bash
uv run python scripts/make_fixtures.py
```

---

## Directory layout

```
data/
├── README.md          # this file
├── sample/            # COMMITTED — synthetic CI fixtures, ~400 KB total
│   ├── ieee_cis_sample.csv
│   ├── lending_club_sample.csv
│   └── churn_sample.csv
├── raw/               # GITIGNORED — nothing real is committed
└── processed/         # GITIGNORED — parquet intermediates
```
