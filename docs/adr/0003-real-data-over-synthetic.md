# ADR-0003: Real public data only. The synthetic generators are deleted.

## Status

**Accepted** — 2026-07-25. Supersedes the implicit "synthetic data is fine for a
portfolio" assumption baked into every earlier phase of this project.

This is the ADR the rest of the repository exists to serve. Read it before
0001, 0002 or 0004.

## Context

### The number

Until this change, `README.md` advertised:

| Problem | Algorithm | Key Metric | Score |
|---|---|---|---|
| Fraud Detection | PyTorch Autoencoder + Isolation Forest | AUC-ROC | **0.964** |

The number was not fabricated. It reproduces exactly from
`results/fraud_detection_metrics.csv` and from
`checkpoints/fraud_detection/metadata.json`. Every artefact agreed with it.

It was worse than fabricated. It was **real and meaningless.**

### The mechanism

`src/data/generate_fraud.py` did not generate transactions and then label them.
It generated two populations from two different distributions and passed the
label into the generator as an **input**:

```python
# src/data/generate_fraud.py:22-29 (deleted in this change)
n_fraud  = int(n_samples * 0.02)
n_normal = n_samples - n_fraud
normal = _generate_transactions(rng, n_normal, is_fraud=False)   # line 26
fraud  = _generate_transactions(rng, n_fraud,  is_fraud=True)    # line 29

# src/data/generate_fraud.py:37-56
def _generate_transactions(rng, n, is_fraud: bool) -> pd.DataFrame:
    if is_fraud:
        transaction_amount = rng.lognormal(mean=5.5, sigma=1.5, ...)
        distance_from_home = rng.exponential(80, n)
        is_online          = rng.binomial(1, 0.7, n)
    else:
        transaction_amount = rng.lognormal(mean=3.5, sigma=1.0, ...)
        distance_from_home = rng.exponential(10, n)
        is_online          = rng.binomial(1, 0.3, n)
```

`is_fraud` is a function parameter, not a derived outcome. The classifier's
entire task was to separate `lognormal(5.5, 1.5)` from `lognormal(3.5, 1.0)`.
0.964 is a measurement of how far apart the author put two random number
generators. It says nothing whatsoever about fraud.

The same defect ran through the whole repository:

| Generator | Label construction |
|---|---|
| `generate_credit_risk.py:50-63` | `z = -3.0 + 0.02*(40-age) - 1.5*((credit_score-680)/100) + ...`, then `is_default = rng.random() < sigmoid(z)` |
| `generate_customer_churn.py:71-86` | Same shape: a hand-written logistic score, then a Bernoulli draw |
| `generate_housing.py:52-70` | Polynomial price × `rng.normal(1.0, 0.10)` — the published R² 0.942 is roughly the arithmetic ceiling of 10% injected noise |

XGBoost recovering a logistic function that the author wrote is not a credit
model. It is a unit test for `model.fit()`.

### Why the old disclosure was not enough

`README.md` said, in one line near the bottom: *"All datasets are **synthetic**."*

That is true and it is not the point. Readers hear "synthetic" as "simulated but
structurally realistic" — the sense in which a synthetic-data vendor uses the
word. What was actually true is that **the label was a closed-form function of
the features, written by the person reporting the score.** That is the part that
voids the metric, and it was never stated.

Being technically accurate while leaving the reader with a false impression is
the failure mode here, and it is not fixed by a stronger disclaimer.

## Decision

**Every headline number in this repository is measured on real, public data that
a stranger can download with a documented command. There is no synthetic path,
no synthetic fallback, and no synthetic "augmentation modality".**

Concretely:

1. **All ten `src/data/generate_*.py` files are deleted**, along with
   `src/data/modality.py` and the `--modality {synthetic,stream,mixed}` flag. The
   three-modality system was scaffolding — it had never produced a single
   `results/modality_comparison_*.csv` or a single `*_stream` checkpoint.

2. **`src/config.py` refuses to load a config without a real `data.source`.**
   Not a warning: a `ConfigError` at load time. Kinds are limited to Kaggle
   competition, Kaggle dataset, or OpenML; dataset ids/slugs are required; the
   adapter module must exist; and generator/simulation markers are forbidden.
   A future contributor cannot reintroduce a generator without deleting this
   validation, which is a visible act rather than an accident.

3. **The four dataset paths across three problems are named, licensed and gated honestly:**

   | Problem | Dataset | Access |
   |---|---|---|
   | Fraud | IEEE-CIS (Vesta) — 590,540 × 394, ~3.5% fraud | Free Kaggle account **plus** one-click acceptance of the competition rules |
   | Credit risk | LendingClub 2007-2018Q4 — ~2.26M × 151, CC0 | Free Kaggle account |
   | Churn | Credit-card attrition — 10,127 × 23 | Free Kaggle account |
   | Ungated fallback | ULB credit-card fraud (OpenML 1597) | **No account at all** |

   IEEE-CIS is public and free but it is **not anonymous-`curl`-able**, and
   `data/README.md` says so in as many words. The ULB/OpenML path exists so a
   reviewer with zero Kaggle presence can still reproduce a real-data fraud
   result end to end through `configs/fraud_ulb.yaml`.

4. **`scripts/download_data.py` has no fallback branch.** When credentials are
   missing it prints the exact remediation and exits non-zero. Nothing is
   trained and no number is written.

5. **The one surviving synthetic artefact is `data/sample/*.csv`** — 500-row,
   schema-shaped CI fixtures whose labels are drawn *independently* of the
   features. They survive for a licensing reason, not a convenience one: IEEE-CIS
   competition data is not redistributable, so committing 500 real rows would
   violate the rules. Two structural guards keep them out of the metric path:
   `TabularTrainer` opens no tracking run and refuses to write a checkpoint or a
   metrics CSV when `--sample` is set. Dashboard history additionally requires
   `sample=false` plus matching problem/config tags.

6. **The retraction is published, not quietly deleted.** `README.md` opens with a
   correction section, above the fold, that states the mechanism — that the label
   was an input to the generator — rather than the softer and less useful
   "metrics were computed on synthetic data". A reader who saw the 0.964 claim
   deserves to know why it was wrong, not just that it is gone.

## Consequences

### The numbers get worse, and that is the deliverable

Expected honest results, written down in each config's `sanity_band` block *before*
any training run, so there is no room to rationalise a suspiciously good outcome
after the fact:

| Problem | Expected ROC-AUC | Interpretation |
|---|---|---|
| Fraud (IEEE-CIS, temporal split) | ≈ **0.90** | Kaggle winning *ensembles* reached 0.94–0.95 on the private leaderboard after months of feature engineering. A single honest LightGBM lands near 0.90. |
| Credit risk (LendingClub, denylist applied) | ≈ **0.70** | A correct consumer-credit model is not impressive-looking. Saying so is the point. |
| Churn (5-fold CV) | high | An easy, well-separated 10,127-row dataset. Reported with mean ± std and an explicit caveat. |

0.964 → ~0.90 is a *downgrade in the number and an upgrade in the claim.*

### The expected range is a smoke alarm, not a leakage test

`configs/<problem>.yaml` declares `sanity_band`, with an optional upper bound.
Crossing it records `sanity_band_warning` and requires investigation. It does not
prove leakage, and churn has no vacuous 1.0 upper bound. The enforceable leakage
controls are chronological splitting, exclusion of the configured split column,
the LendingClub/churn denylists, and the tests that assert those columns never
reach a feature matrix.

### Headline training remains impossible without credentials

The three headline datasets cannot train on a machine with no Kaggle token. That
is accepted. The OpenML 1597 fallback can:

```bash
uv run python scripts/download_data.py --dataset ulb-creditcard
uv run python scripts/train.py --model fraud_ulb
cat reports/fraud_ulb_metrics.csv
```

The alternative — a synthetic fallback so the demo always "works" — is exactly
the thing being deleted.

### What was kept

Nothing that worked was thrown away: `BaseTrainer`, the MLflow integration, the
SHAP and gradient explainers, `src/device.py`, the structured logging, the Docker
setup and the Next.js app all survive. The MPS autoencoder survives too, demoted
from headline model to *unsupervised baseline* — which is the honest role for it,
and which lets the results table show what supervision actually buys.

## Alternatives considered

**Keep the generators, label the metrics clearly as synthetic.** Rejected. The
previous README already did a weak version of this and it still misled. And a
results table of synthetic numbers is not evidence of anything; a reviewer cannot
distinguish a good model from a well-separated generator.

**Improve the generators so the label is not a closed-form function.** Rejected.
Building a genuinely realistic fraud simulator is harder than the modelling task
and would still produce a number nobody can check. Real public data with a real
provenance link is strictly more credible and strictly less work.

**Use only the ungated ULB dataset and skip Kaggle entirely.** Rejected as the
sole path, kept as a fallback. ULB is 284,807 PCA-transformed rows with no
categorical features and anonymised component names. It does provide `Time` as a
chronological split key, but it cannot demonstrate categorical handling or SHAP
explanations whose `V1`–`V28` axes mean anything to a human. It is the right
*credential-free escape hatch*, not the right centrepiece.

## References

- Retracted metric: `README.md` §"A correction, and why it's here"
- Enforcement: `src/config.py::_validate`, `src/training/tabular.py::_check_sanity_band`
- Gates: `tests/test_quality_gates.py`, `tests/data/test_leakage_denylist.py`
- Data provenance: [`data/README.md`](../../data/README.md)
- Narrowing rationale: [ADR-0004](0004-narrow-to-fintech.md)
