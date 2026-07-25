# ADR-0004: Ten industries become three fintech problems

## Status

**Accepted** — 2026-07-25.

## Context

The repository catalogued **twenty-one models across six industries**: real
estate, dental clinics, healthcare, fintech, logistics, and legal/immigration.

What was actually there:

- `configs/` held 10 YAML files. `checkpoints/` held 4 problems.
  `src/serving/api.py` exposed 21 routes. Six of those routes raised
  `FileNotFoundError` — a 500 — on every call, because no checkpoint existed.
- `web/lib/industries.ts` had 11 entries with `ready: true` and 10 with
  `ready: false`, including **eight literal `TODO(copy)` markers** in
  user-visible taglines.
- `README.md` advertised "a sortable table of all 20 cataloged models". Ten of
  those rows could never resolve.
- Three different model counts appeared in one repository: "all 20" in the
  README, "10 models" in `CLAUDE.md`, and a four-row metrics table.

The breadth was not evidence of range. It was evidence of a generator: heart
disease, H-1B approval, dental no-shows, delivery ETA, rental price, demand
forecasting and housing price share no domain, no data source and no reader.
Each was a `train_<problem>.py` copied from the last one.

## Decision

**Keep three fintech problems. Delete the other seven, and the industries that
existed only to hold them.**

| Kept | Deleted |
|---|---|
| `fraud` (IEEE-CIS) | `heart_disease`, `h1b_approval`, `dental_noshow` |
| `credit_risk` (LendingClub) | `delivery_eta`, `rental_price` |
| `churn` (card attrition) | `demand_forecasting`, `price_prediction` |

Deleted across `configs/`, `src/data/`, `src/features/`, `src/models/`,
`src/training/`, `tests/`, `web/app/`, `web/__tests__/` and `results/`.

The three survivors are not an arbitrary subset. They are **three real-money
decisions in one domain**, each with a different shape:

| Problem | Shape | What it demonstrates |
|---|---|---|
| Fraud | 590k rows, 3.5% positive, temporal | Severe imbalance, feature engineering on anonymised columns, PR-AUC over ROC-AUC |
| Credit risk | 2.26M rows, target derived from loan outcome | Leakage discipline — the with/without-denylist delta is the single best data-science paragraph available here |
| Churn | 10,127 rows | Correct statistics on small n: 5-fold CV with mean ± std, never a single hold-out number |

They share a vocabulary a fintech interviewer already has. Credit risk carries
a deliberately unimpressive hand-entered pre-run expectation of roughly 0.70
ROC-AUC, which is a stronger framing than promising three impressive-looking
numbers.

## Consequences

- **`src/serving/api.py` drops from 21 routes to 8** — three predict, three
  explain, `/models` and `/health`. Every route now has a model behind it, and a
  missing checkpoint returns 503 with the command that fixes it rather than a 500.
- **The web catalogue is fully resolved.** Three models, all `ready: true`, zero
  placeholder copy. `tests/lib/industries.test.ts` asserts no `TODO` string can
  reappear.
- **The model count is consistent everywhere** because it is derived:
  `Industry.modelCount` reads `models.length`, and the README quotes no count
  that is not backed by a file.
- **The deleted domains are in git history.** Nothing is lost; the commit that
  removed them is the record. They are not coming back — Phase B in the old plan
  proposed nine *more* industry models and directly contradicts this decision.
- **`src/evaluation/regression_metrics.py` and `timeseries_metrics.py` are
  deleted** along with `lstm_forecaster.py` and `price_model.py`. All three
  surviving problems are binary classification, so a regression metric module
  with no caller is just surface area.

## Alternatives considered

**Keep the breadth, finish the ten missing models.** Rejected. It is weeks of
work to produce ten more numbers with the same credibility problem, on ten
datasets nobody asked about, in service of a claim ("I can do any industry") that
no hiring manager is screening for.

**Keep the breadth, mark the unbuilt models honestly as roadmap.** Rejected. A
catalogue that is half roadmap reads as abandoned, and the honest label does not
fix the underlying tell: the six industries were never chosen, they were
enumerated.

**Narrow to one problem — fraud only.** Rejected as too far. Three problems in
one domain lets the results table show *contrast*: an imbalanced problem where
PR-AUC is the right metric, a hard problem with a pre-run expectation around
0.70, and an easy problem where a high score means nothing. One problem cannot
show that.

## References

- Data and provenance: [`data/README.md`](../../data/README.md)
- The reason the old metrics were void: [ADR-0003](0003-real-data-over-synthetic.md)
