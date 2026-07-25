# PROGRESS — rebuild to fintech + real data

Tracking file for the rebuild described in [`AGENT-BRIEF.md`](AGENT-BRIEF.md).
Branch: `rebuild/fintech-real-data`. Every slice below is committed separately.

**Status legend:** `[x]` done and committed · `[~]` partially done, see notes · `[ ]` not started ·
`[BLOCKED]` cannot proceed without something the owner must supply.

---

## Slices

- [x] **Slice 1 — Trust-signal repairs.** `uv.lock` un-ignored and committed, `LICENSE` (MIT) added,
      `.gitignore` hardened, broken git ref + stray `.coverage 2..7` removed, orphan stash dropped,
      `docker-compose.dev.bak.yml` deleted, README triage: **the 0.964 / 0.888 / 0.942 / 22.2 metrics
      are retracted**, `YOUR_USERNAME` and `:8000` removed, web port moved `3071 → 3070` per
      Appendix A1.
- [x] **Slice 2 — Delete the seven non-fintech domains.** Root de-clutter, infra relocation, ADR move.
- [x] **Slice 3 — Real data acquisition layer.** Downloader, adapters, sample fixtures, `data/README.md`.
      The *code path* is complete; the *download itself* is blocked (see below).
- [BLOCKED] **Slice 4 — Fraud on IEEE-CIS.** Needs the dataset. See "BLOCKED — needs owner".
- [BLOCKED] **Slice 5 — Credit risk on LendingClub.** Needs the dataset.
- [BLOCKED] **Slice 6 — Churn on credit-card attrition.** Needs the dataset.
- [x] **Slice 7 — Collapse the trainers, split the predictor.**
- [x] **Slice 8 — Tests that could actually fail.**
- [x] **Slice 9 — Docs, diagrams, ADRs, README rewrite.**
- [x] **Slice 10 — CI + fresh-clone verifier.**

---

## BLOCKED — needs owner

### B1. Kaggle credentials (blocks Slices 4, 5, 6 — i.e. every published metric)

No Kaggle API token is present on this machine and this session is not permitted to create accounts,
enter credentials, or accept competition terms. All three datasets sit behind a free Kaggle account;
IEEE-CIS additionally requires a one-click acceptance of the competition rules.

**Exact steps to unblock — run these yourself:**

```bash
# 1. Create a Kaggle API token (browser, one time)
#    https://www.kaggle.com/settings/account  →  "Create New Token"  →  downloads kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json

# 2. Accept the IEEE-CIS competition rules (browser, one time — required, cannot be scripted)
#    https://www.kaggle.com/competitions/ieee-fraud-detection/rules   →  "I Understand and Accept"

# 3. Download. Each dataset is independent; run only the ones you want.
cd "07-Portfolio-ML-System"
uv run python scripts/download_data.py --dataset cc-churn        # ~2 MB, fastest, no rules gate
uv run python scripts/download_data.py --dataset ieee-cis        # ~118 MB zipped / ~1.35 GB expanded
uv run python scripts/download_data.py --dataset lending-club    # ~648 MB gzipped  <-- large, see note
```

**Note on size:** `lending-club` is ~648 MB. This session's operating limit was 200 MB, which is a
second, independent reason Slice 5 did not run here.

**Credential-free alternative that works today:** the ULB credit-card fraud dataset is on OpenML and
needs no account at all.

```bash
uv run python scripts/download_data.py --dataset ulb-creditcard   # ~150 MB via sklearn/OpenML, zero credentials
```

`scripts/download_data.py` prints exactly this remediation when credentials are missing — it does not
fail silently and it never falls back to synthetic data.

### B2. Model training (blocks the results table)

Training was deliberately not run. Two reasons:

1. **No data** (B1).
2. **Compute budget.** This session was instructed not to run heavy compute. For the record, the
   measured hardware facts that shape any future run on this machine:
   - Apple Silicon, 10 cores (4P + 6E), 32 GB RAM, torch 2.13.0, MPS available.
   - LightGBM and XGBoost ship **CPU-only wheels on macOS arm64** — there is no Metal backend. The
     three headline models are gradient-boosted trees, so **MPS buys nothing for them**. Only the
     fraud autoencoder baseline uses MPS.
   - MPS gives ~1.9–2.2× over CPU on dense matmul, not 5–10×.
   - `torch.get_num_threads()` defaults to 4, not 10.
   - **torch 2.13.0 MPS bug, reproduced twice:** `torch.nn.MultiheadAttention` hangs on MPS, and a
     CPU transformer loop deadlocked at 0 % CPU after a preceding MPS matmul in the *same* process.
     Never mix MPS and CPU tensor workloads in one process — use a separate process per device.
     (Not currently hit by this repo, which has no attention layers, but it constrains any future
     sequence model.)

**To unblock after B1:**

```bash
uv run python scripts/train.py --model churn        # smallest, ~10 k rows — start here
uv run python scripts/train.py --model fraud        # IEEE-CIS, the centrepiece
uv run python scripts/train.py --model credit_risk
uv run python scripts/evaluate.py --model all
uv run python scripts/make_figures.py               # PR curves, calibration, SHAP → docs/images/
```

Then, and only then, fill the results table in `README.md` and `reports/RESULTS.md` **from
`reports/*_metrics.csv`** — do not type numbers by hand. `tests/test_quality_gates.py` asserts the
expected bands and will fail loudly on a leaked model:

| Problem | Expected honest ROC-AUC | Above this band means leakage |
|---|---|---|
| Fraud (IEEE-CIS, temporal split) | ≈ 0.90 | ≥ 0.96 |
| Credit risk (LendingClub, denylist applied) | ≈ 0.70 | ≥ 0.80 |
| Churn (5-fold CV) | high (easy dataset) | — report mean ± std, never one number |

### B3. Screenshots of a running UI (Appendix A2)

`scripts/capture_screenshots.py` is committed and driven by Playwright against seeded demo data
only. It was **not executed** in this session: the pages it captures render model predictions, and
with no trained checkpoints (B2) the captures would show empty states — i.e. they would be
screenshots of nothing, or worse, would need faked numbers to look good. That is the exact failure
this rebuild exists to correct.

**To unblock after B2:**

```bash
make docker-up                       # or: make serve  &&  make web-dev
uv run python scripts/capture_screenshots.py     # writes docs/images/*.png
```

The chart figures (PR curve, calibration, confusion matrix, SHAP summary) come from
`scripts/make_figures.py` and are likewise gated on real checkpoints.

---

## Verify-script status (Appendix A4)

`scripts/verify_fresh_clone.sh` is committed and executable. It runs in stages so partial
environments still produce signal. Recorded result of the last run is in the section below.

---

## Assumptions recorded (Appendix A5)

| # | Ambiguity | Decision | Recorded in |
|---|---|---|---|
| 1 | Brief §4 says web port `3071`; Appendix A1 says `3070` | Appendix A overrides → **3070** | `docs/adr/0005-assumptions-log.md` |
| 2 | Brief says squash-merge to `main`; run instructions say the per-slice commit graph is the artifact | **`git merge --no-ff`**, preserving every slice commit | `docs/adr/0005-assumptions-log.md` |
| 3 | Brief Slices 4–6 require training; owner note forbids it | Build everything except the run; record as BLOCKED | this file |
