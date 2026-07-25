# PROGRESS — rebuild to fintech + real data

This file is the standalone handoff and status tracker for the fintech rebuild.
Current branch: `main`. Baseline for the trust-audit repair: `05d32ca`.
The repair below is intentionally uncommitted because the owner explicitly
forbade commits; the older rebuild slices remain historical committed work.

**NEXT ACTION:** add model-level monotonic constraints for
`Months_Inactive_12_mon` and `inactive_share_of_tenure`, retrain `churn`, and
replace its published cells only if the new training-written CSV and checkpoint
pass the full non-network suite.

**Historical-slice legend:** `[x]` done and committed · `[~]` partially done,
see notes · `[BLOCKED]` cannot proceed without something the owner must supply.
The trust-audit section states its separate uncommitted status explicitly.

---

## Accepted real-data runs — 2026-07-25

Two of five evaluator rows are now measured. Both are stratified 5-fold
cross-validation results; the saved serving estimators were refit on all rows
only after one out-of-fold score had been written for every row.

| Run | Command | PR-AUC | ROC-AUC | Baseline PR-AUC | Δ | P@1% | R@1%FPR |
|---|---|---|---|---|---|---|---|
| `fraud_ulb` | `uv run python scripts/train.py --model fraud_ulb` | 0.8569 ± 0.0331 | 0.9810 ± 0.0092 | 0.7300 ± 0.0279 | 0.1269 ± 0.0355 | 0.1544 ± 0.0043 | 0.8964 ± 0.0219 |
| `churn` | `uv run python scripts/train.py --model churn` | 0.9735 ± 0.0078 | 0.9940 ± 0.0019 | 0.7800 ± 0.0217 | 0.1935 ± 0.0145 | 1.0000 ± 0.0000 | 0.8869 ± 0.0317 |

Reproduction and table rendering:

```bash
uv run python scripts/train.py --model fraud_ulb
uv run python scripts/train.py --model churn
./.venv/bin/python scripts/evaluate.py --markdown
```

Both checkpoint metadata files record training SHA
`05d32cae0d54dee97a70be575097182e9b5f8278`, seed 42, and
`macOS-26.5.2-arm64-arm-64bit` on arm64. The current publication HEAD is
`e15842a`. `fraud_ulb` used OpenML 1597: 284,807 rows, 0.1727% positive,
29 features, LightGBM, 67.8 s. `churn` used
`sakshigoyal7/credit-card-customers`: 10,127 rows, 16.07% positive, 23
features, LightGBM, 11.1 s.

The `fraud_ulb` PR-AUC sanity band was set by hand at 0.70–0.85 before the run.
The 0.8569 measurement tripped `sanity_band_warning`. Investigation found only
`V1`–`V28` plus `Amount` in the feature matrix, target/`Class` denylisting, no
timestamp or entity identifier for folds to straddle, and a logistic-regression
baseline that moved to 0.7300 with the model. The config maximum was then raised
from 0.85 to 0.90 on 2026-07-25 with a comment recording that this was a
post-run change.

The churn result is not prospective. `Attrition_Flag` is current status, and
the features summarise the same trailing activity window. With no event
timestamp, feature cutoff, or future outcome window, this is substantially
detection of attrition that already happened.

### Publication verification

- `scripts/evaluate.py --markdown` reproduced both rows exactly.
- Ruff check and format-check passed across 85 files.
- The non-network suite finished with **231 passed, 11 skipped, 1 deselected,
  1 failed**. The failure is
  `test_more_inactive_months_does_not_lower_churn_risk`: the measured churn
  checkpoint lowers its score on that fixed inactivity probe.
- The failing quality gate was not disabled, skipped, or relaxed. Fixing it
  requires a new model specification and a new measured run; post-processing
  this checkpoint would make the published metrics and serving artifact diverge.

---

## Methodology repair — 2026-07-25 (verified, uncommitted)

Baseline remains `05d32ca`; this work is layered over the existing uncommitted
trust-audit repair. The owner explicitly forbade commits and pushes.

- [x] Tie-safe chronological splits assign equal timestamps to the earlier
      partition and reject proportions that would empty a requested partition.
- [x] Split columns are excluded by the config-driven matrix builder for every
      problem.
- [x] Cross-validation reports/gates only the aggregate mean and standard
      deviation, persists one held-out OOF prediction per row, and refits the
      checkpoint estimator on all rows.
- [x] Autoencoder evaluation uses raw reconstruction error for ranking, the
      train-fitted 95th-percentile threshold for hard decisions, and no Brier
      score. Its checkpoint now has tabular-equivalent provenance/metrics and
      round-trips through `CheckpointRegistry`.
- [x] Python, NumPy, CUDA (when present), MPS (when exposed), and PyTorch CPU RNGs
      receive the config seed; MPS kernel nondeterminism remains documented.
- [x] `sanity_band` replaces the misleading leakage-ceiling terminology. It is
      documented as a smoke alarm; denylists, split-column exclusion, and their
      tests are the leakage defence. Churn's vacuous 1.0 maximum is gone.
- [x] Local artifacts are written before tracker calls; every tracker call is
      guarded; CLI cleanup runs in `finally`; successful local checkpoints are
      registered when MLflow is active.
- [x] Fraud's train-only aggregate features are documented as not point-in-time
      correct and therefore optimistically biased.
- [x] Results disclose that no automated tuning ran and that the comparison is
      not a fair-tuning comparison.

### RED / GREEN evidence

- Baseline focused suite: **55 passed, 18 skipped**.
- RED after methodology regressions: **22 failed** for the audited defects.
- Focused GREEN: **80 passed, 17 skipped**.
- Full non-network suite: **221 passed, 17 skipped, 1 deselected**.
- Coverage: **88% on `src`**, measured separately with the normal pytest
  coverage addopts.
- Ruff: check and format-check clean across **85 files**.
- All four configs load: `fraud`, `credit_risk`, `churn`, `fraud_ulb`.

At the end of this repair itself, no real-data training had run. The two
subsequent accepted runs and their OOF artifacts are recorded in the section
above. No commit and no push were made during this publication task.

---

## Trust-audit repair — 2026-07-25 (verified, uncommitted)

- [x] **Config enforcement is real.** Kinds are closed to Kaggle
      competition/dataset or OpenML; source ids are required; adapter modules
      must exist; ADR-0003 generator/simulation markers are rejected.
- [x] **Download integrity is truthful.** File digests are recorded and compared
      only when pinned; mismatches are fatal; unpinned digests print the exact
      `PROVENANCE["expected_sha256"]` assignment; primary CSV rows are streamed
      and compared non-fatally. No digest was invented.
- [x] **Fixture tracking is isolated.** Sample mode opens no MLflow/W&B run and
      writes no checkpoint/metrics CSV. Production runs carry `problem`,
      `sample=false`, and `config`; dashboard history requires all relevant tags.
- [x] **Constant scores fail the actual gate.** The reported metric set includes
      score standard deviation and the trainer rejects a zero-variance scorer as
      `DEGENERATE predictions`.
- [x] **OpenML 1597 is a complete second fraud config.**
      `configs/fraud_ulb.yaml` → `src.data.adapters.ulb_creditcard` →
      `src.features.ulb_features`; `Time` is the split key and never a feature.
      The 500-row repository fixture was generated with seed 42 and contains no
      real rows.
- [x] **False prose corrected.** SHA wording, MLflow/sample claims, evaluator
      output, credential-free commands, and uploader licence strings now match
      code.

### RED / GREEN evidence

| Slice | RED | GREEN |
|---|---|---|
| Config loader | 5 focused failures | 20 passed |
| Digest + row checks | 2 focused failures | 24 downloader/adapter tests passed |
| MLflow + dashboard | 2 Python + 3 web failures | 24 Python + 15 web tests passed |
| Constant predictor | 1 focused failure (only generic low-AUC warning) | 32 gate/metrics/trainer tests passed |
| OpenML ULB | 2 collection errors (adapter/features absent) | 78 focused tests passed |

### Current verification

- Python non-network suite: **205 passed, 18 skipped, 1 deselected**.
- Coverage: **85% on `src`**, measured with the normal pytest coverage addopts.
- Web: **98 passed** across 18 files; TypeScript clean.
- Ruff: check and format-check clean across 84 files.
- All four configs load: `fraud`, `credit_risk`, `churn`, `fraud_ulb`.

### Known limits / deliberately not done

- The live OpenML/Kaggle network path and real model training were not run
  during this repair phase. The later `fraud_ulb` and `churn` runs are recorded
  separately above.
- `docs/diagrams/pipeline-dag.svg` is a generated, single-line SVG containing the
  old `sha256 verified` label four times. It was not safely hand-edited and was
  not regenerated because the task explicitly said to regenerate nothing. The
  Mermaid source, README, and architecture document are corrected.
- No commit and no push were made.

---

## Slices

- [x] **Slice 1 — Trust-signal repairs.** `uv.lock` un-ignored and committed;
      `LICENSE` (MIT) added; `.gitignore` hardened; broken git ref
      `origin/HEAD 2`, seven stray `.coverage N` files and the orphan stash
      removed; `docker-compose.dev.bak.yml` deleted. **README triage: the 0.964 /
      0.888 / 0.942 / 22.2 metrics retracted**, `YOUR_USERNAME` and `:8000`
      removed, web port `3071 → 3070` per Appendix A1.
- [x] **Slice 2 — Delete the seven non-fintech domains.** Root de-cluttered
      (`AGENTS.md`, `PLAN.md`, `PORT-MAP.md`, `PORTS.md`, `CLAUDE.md`, `tasks/`
      gone; `SPEC.md` and `decision.md` untracked after migrating to ADRs).
      Dockerfiles → `infra/docker/`, compose → `infra/compose/`,
      `docs/decisions/ADR-001` → `docs/adr/0001-`. `results/` deleted entirely —
      every CSV in it held a retracted number.
- [x] **Slice 3 — Real data acquisition layer.** `src/data/download.py` with
      `kaggle_competition_cached()` (the old `stream.py` only had
      `dataset_download`, which cannot fetch IEEE-CIS) and `openml_cached()`;
      three adapters; `scripts/download_data.py`; `data/sample/*.csv` fixtures;
      `data/README.md`. **All ten generators deleted.** The code path is
      complete; the download itself is blocked — see B1.
- [BLOCKED] **Slice 4 — Fraud on IEEE-CIS.** The OAuth token works for Kaggle
      datasets but not this competition; a classic `kaggle.json` is required.
- [x] **Credential-free fraud path — ULB / OpenML 1597.** Real-data LightGBM run
      accepted with out-of-fold metrics and an all-row serving refit.
- [ ] **Slice 5 — Credit risk on LendingClub.** Not run. The 648 MB download was
      not attempted in this session.
- [x] **Slice 6 — Churn on card attrition.** Real-data LightGBM run accepted; the
      two Naive-Bayes posterior columns are absent from the checkpoint feature
      list.
- [x] **Slice 7 — Collapse the trainers, split the predictor.** Eight
      `train_*.py` → one `src/training/tabular.py`; the 573-line
      `predictor.py` → seven files, every one under 150 lines;
      `src/models/registry.py`; `src/features/schema.py` for categorical replay.
- [x] **Slice 8 — Tests that could actually fail.** `tests/` mirrors `src/`.
      **181 passing, 18 skipped, 85% coverage.** `-m 'not network and not parity'`
      removed from `addopts`. `mypy` added and clean.
- [x] **Slice 9 — Docs, diagrams, ADRs, README rewrite.** ADRs 0001–0005; three
      Mermaid diagrams inline and exported to SVG; `docs/architecture.md`;
      `docs/ports.example.md`; `data/README.md`; `reports/RESULTS.md`.
- [~] **Slice 10 — CI + fresh-clone verifier.** `.github/workflows/ci.yml` written
      (the directory had been empty across 84 commits) and
      `scripts/verify_fresh_clone.sh` written, run, and its findings fixed —
      stages 1-6 pass, stage 7 (Docker) did not complete here, see below.
      **CI has never run**, because pushing was forbidden. The badge will show
      "no status" until you push.

---

## BLOCKED and not run

### B1. IEEE-CIS — BLOCKED on the competition token

`~/.kaggle/access_token` exists and works for Kaggle **datasets**:
`kagglehub.dataset_download('sakshigoyal7/credit-card-customers')` succeeded.
The same OAuth token does not work for Kaggle **competitions**.

Exact command:

```bash
./.venv/bin/python -c "import kagglehub; kagglehub.competition_download('ieee-fraud-detection')"
```

Exact error:

```text
403 ... Please make sure you are authenticated and have accepted the competition rules
```

IEEE-CIS needs a classic Kaggle API token, not the OAuth
`access_token`:

```bash
# https://www.kaggle.com/settings/account -> "Create New Token"
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

uv run python scripts/download_data.py --dataset ieee-cis
uv run python scripts/train.py --model fraud
uv run python scripts/train.py --model fraud --autoencoder
```

The supervised `fraud` row and `fraud_autoencoder` row remain **not yet
measured** until that succeeds.

### B2. LendingClub — not run

`credit_risk` is not credential-blocked by the competition issue. Its 648 MB
dataset download was simply not attempted in this session, so the result row
remains empty.

```bash
uv run python scripts/download_data.py --dataset lending-club
uv run python scripts/train.py --model credit_risk
uv run python scripts/evaluate.py --markdown
```

### B3. Screenshots of a running UI (Appendix A2)

`scripts/capture_screenshots.py` is committed, Playwright-driven, and uses seeded
demo defaults only — never real or personal data. It was **not executed** during
this results-publication task.

**To capture the measured-model pages:**

```bash
make docker-up                                    # or: make serve  &&  make web-dev
uv run playwright install chromium                # one time
uv run python scripts/capture_screenshots.py      # writes docs/images/*.png
```

The chart figures (PR curve, ROC, calibration, confusion matrix, SHAP summary,
gain plot) come from `scripts/make_figures.py` and are gated on the same
checkpoints. Both scripts **exit non-zero rather than producing an empty
artefact**.

`README.md` states in its Limitations section that `docs/images/` is empty and
why, so the absence is disclosed rather than hidden.

---

## Verify-script status (Appendix A4)

`scripts/verify_fresh_clone.sh` is committed and executable. It clones the
committed HEAD into a throwaway directory — so only tracked files exist — and
runs the documented quickstart in seven stages. Run with `make verify`.

**Result of the last full run:**

| Stage | Result |
|---|---|
| 1. Clone committed HEAD | **PASS** |
| 2. Files the quickstart needs are tracked (`uv.lock`, `LICENSE`, `.env.example`, `pnpm-lock.yaml`, Dockerfile, compose, `data/sample/`) | **PASS** |
| 3. Hygiene — no `CLAUDE.md`/`AGENTS.md`/`PORT-MAP`/`.claude/`/`.bak` tracked; no `src/data/generate_*.py`; no `YOUR_USERNAME`; the retraction section present | **PASS** |
| 4. Every relative README link resolves inside the clone | **PASS** |
| 5. `uv sync --frozen` then `pytest -m "not network"` | **PASS** — 181 passed, 18 skipped, 1 deselected |
| 6. `pnpm install --frozen-lockfile` → typecheck → test → build | **PASS** — 98 web tests, build emits 7 routes |
| 7. `docker build` + `compose up --wait` + `GET /health` + `POST /predict/fraud` → 503 | **NOT COMPLETED** — see below |

### Stage 7 — what happened, honestly

Stage 7 did **not** produce a pass in this session. Two distinct things were found,
one fixed and one an environment constraint:

1. **Fixed.** The first run failed with
   `error getting credentials - exec: "docker-credential-osxkeychain": not found`.
   Docker Desktop installs that helper in
   `/Applications/Docker.app/Contents/Resources/bin` and adds the directory to
   PATH **only for login shells**, so `docker build` fails from any script or CI
   step even for anonymous public images. The verifier now prepends that
   directory when it exists, and its failure hint is derived from the build log
   instead of asserting a wrong cause. This changes nothing that any check
   asserts.

2. **Not resolved here.** With the PATH fixed, the build was queued behind other
   Docker builds on a machine sitting at **loadavg 26–29** from parallel work in
   other repositories. This session was instructed not to add heavy compute, so
   the build was left to run rather than being forced, and it had not finished
   when this file was written.

**To finish stage 7 yourself**, on a quiet machine:

```bash
make verify                          # all seven stages
SKIP_WEB=1 make verify               # stage 7 plus the cheap ones, faster
```

What stage 7 asserts, so you know what a pass means: the API image builds from
the clone (which is only possible because `uv.lock` is now tracked), the stack
comes up with `--wait`, `GET /health` answers within 60s, and
`POST /predict/fraud` returns **503** — not 500 and not 200. A 200 would mean a
model appeared from nowhere on a clone with no checkpoints.

### Things the verifier caught, and that were then fixed

This list is the useful part of running it:

1. **`infra/docker/api.Dockerfile` had `COPY checkpoints/ ./checkpoints/`.**
   `checkpoints/` is gitignored, so on a fresh clone the directory does not exist
   and the build fails outright. Replaced with a read-only bind mount in
   `infra/compose/base.yml` plus `RUN mkdir -p /app/checkpoints` so the
   registry's glob has a directory to find.
2. **`infra/compose/*.yml` still declared `context: .` and
   `dockerfile: Dockerfile.api`** after those files moved into `infra/`. Build
   context corrected to the repository root.
3. **The `Makefile` `docker-*` targets ran bare `docker compose`,** which finds
   no compose file now that it lives in `infra/compose/`. Every target now passes
   `-f infra/compose/base.yml`.
4. **`data/sample/` was not copied into the API image,** so the container could
   not run its own offline path.
5. **A TypeScript error in a test I had just written** (`TS2802`: spreading
   `HTMLOptionsCollection`). The local working tree had been typechecked *before*
   that test existed, so only the fresh-clone run found it.
6. **`SPEC.md` and `decision.md` had been silently re-added** by a `git add -A`
   after being untracked. Caught by counting tracked root entries.

None of these were fixed by weakening a check.

## Owner action items (small, but they are claims)

### C1. The CI badge shows "no status" until you push

`README.md` line 3 links to
`.../actions/workflows/ci.yml/badge.svg`. That URL currently 404s because the
workflow has never *run* — this session was forbidden from pushing, so GitHub has
no run to render. GitHub displays such a badge as a grey "no status", which is
accurate: there are no runs yet. It goes green on the first push of this branch.

Nothing needs changing. This is noted only so the grey badge is not mistaken for
a broken link.

### C2. Add your LinkedIn to the Author section

The README's Author section links to GitHub only. An earlier draft of this
rewrite contained a LinkedIn URL **guessed from the GitHub handle**, which is
precisely the kind of unverifiable claim this repository exists to eliminate, so
it was removed rather than left in.

Add the real one yourself:

```markdown
[LinkedIn](https://www.linkedin.com/in/<your-actual-handle>/) · [GitHub](https://github.com/armandogon94)
```

### C3. External link check (Appendix A4.3)

Every external URL in `README.md`, checked with `curl -L`:

| Status | URL |
|---|---|
| 200 | github.com/armandogon94 · the repo · the workflow page |
| 200 | all three shields.io badges |
| 200 | all three Kaggle dataset pages |
| 404 | the CI badge SVG — see C1, expected until first push |
| 000 | `localhost:8070/health`, `localhost:3070/dashboard` — these appear inside `curl`/`open` commands in the Quickstart, not as live links |

No claim in the README depends on a URL that is dead and presented as live.

---

## Assumptions recorded (Appendix A5)

Full text in [`docs/adr/0005-assumptions-log.md`](adr/0005-assumptions-log.md).

| # | Ambiguity | Decision |
|---|---|---|
| 1 | Brief §4 says web port `3071`; Appendix A1 and `PORT-ALLOCATION.md` say `3070` | **3070** — Appendix A overrides |
| 2 | Brief says squash-merge to `main`; run instructions say the per-slice commit graph is the artifact | **`git merge --no-ff`**, every slice commit preserved |
| 3 | Brief Slices 4–6 require training; the owner's note forbids downloads and heavy compute | Build the entire path, stop before the run, record as BLOCKED |

**Nothing was pushed.** The remote is untouched. Local commits only.
