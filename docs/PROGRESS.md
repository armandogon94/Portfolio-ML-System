# PROGRESS — rebuild to fintech + real data

This file is the standalone handoff and status tracker for the fintech rebuild.
Branch: `rebuild/fintech-real-data`. Every slice below is a separate commit.

**Legend:** `[x]` done and committed · `[~]` partially done, see notes ·
`[BLOCKED]` cannot proceed without something the owner must supply.

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
- [BLOCKED] **Slice 4 — Fraud on IEEE-CIS.** Everything except the run. See B1, B2.
- [BLOCKED] **Slice 5 — Credit risk on LendingClub.** See B1, B2.
- [BLOCKED] **Slice 6 — Churn on card attrition.** See B1, B2.
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

## BLOCKED — needs owner

### B1. Kaggle credentials (blocks Slices 4, 5, 6 — every published metric)

No Kaggle API token exists on this machine, and this session was not permitted to
create accounts, enter credentials, or accept competition terms. All three
datasets sit behind a free Kaggle account; IEEE-CIS additionally requires a
one-click acceptance of the competition rules that cannot be scripted.

**Exact steps to unblock — run these yourself:**

```bash
# 1. Create a Kaggle API token (browser, one time)
#    https://www.kaggle.com/settings/account  ->  "Create New Token"
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Accept the IEEE-CIS competition rules (browser, one time, cannot be scripted)
#    https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#    -> "I Understand and Accept"

# 3. Download. Each dataset is independent; run only the ones you want.
cd 07-Portfolio-ML-System
uv run python scripts/download_data.py --check                 # what each needs, no download
uv run python scripts/download_data.py --dataset cc-churn      # ~2 MB, fastest, no rules gate
uv run python scripts/download_data.py --dataset ieee-cis      # ~118 MB zipped / ~1.35 GB expanded
uv run python scripts/download_data.py --dataset lending-club  # ~648 MB gzipped  <-- large, see note
```

**Size note.** `lending-club` is ~648 MB. This session's operating limit was
200 MB, which is a second, independent reason Slice 5 did not run here.

**Credential-free alternative that works today.** The ULB credit-card fraud
dataset is on OpenML and needs no account at all:

```bash
uv run python scripts/download_data.py --dataset ulb-creditcard   # ~150 MB, zero credentials
```

`scripts/download_data.py` prints exactly this remediation when credentials are
missing. Verified by running it with a scrubbed environment — it exits non-zero
and never falls back to synthetic data.

### B2. Model training (blocks the results table and every figure)

Training was deliberately not run. Two independent reasons:

1. **No data** (B1).
2. **Compute budget.** This session was instructed not to run heavy compute.

Measured hardware facts that shape any future run on this machine — cite these
rather than re-measuring:

- Apple Silicon, 10 cores (4 performance + 6 efficiency), 32 GB RAM, torch
  2.13.0, MPS available.
- **LightGBM and XGBoost ship CPU-only wheels on macOS arm64.** No Metal backend
  exists for either. The three headline models are gradient-boosted trees, so
  **MPS buys them nothing**; `n_jobs` (set to 8 in the configs — leaving two
  cores for the OS) is the only lever. Only the fraud autoencoder uses MPS.
- MPS gives ~1.9–2.2× over CPU on dense matmul, not 5–10×.
- `torch.get_num_threads()` defaults to 4, not 10.
- **torch 2.13.0 MPS bug, reproduced twice:** `torch.nn.MultiheadAttention` hangs
  on MPS, and a CPU tensor loop deadlocked at 0% CPU after a preceding MPS matmul
  in the *same* process. Never mix MPS and CPU tensor workloads in one process —
  use a separate process per device. This repository has no attention layer so it
  does not trip the bug, but two rules already follow from it: the test suite
  forces MPS off session-wide (`tests/conftest.py`), and the serving path never
  touches MPS (`src/serving/explain.py`).

**To unblock after B1:**

```bash
uv run python scripts/train.py --model churn        # smallest, ~10k rows — start here
uv run python scripts/train.py --model fraud        # IEEE-CIS, the centrepiece
uv run python scripts/train.py --model fraud --autoencoder   # unsupervised baseline
uv run python scripts/train.py --model credit_risk
uv run python scripts/evaluate.py                   # renders reports/*_metrics.csv
uv run python scripts/evaluate.py --markdown        # emits the README table body
uv run python scripts/make_figures.py               # PR curves, calibration, SHAP
```

Then fill the results tables in `README.md` and `reports/RESULTS.md`
**from `scripts/evaluate.py --markdown`** — do not type numbers by hand.

`tests/test_quality_gates.py` asserts the expected bands and will fail loudly on
a leaked model:

| Problem | Expected honest ROC-AUC | Above this means leakage |
|---|---|---|
| Fraud (IEEE-CIS, temporal split) | ≈ 0.90 | ≥ 0.96 |
| Credit risk (LendingClub, denylist applied) | ≈ 0.70 | ≥ 0.80 |
| Churn (5-fold CV) | high — easy dataset | report mean ± std, never one fold |

The trainer also writes a `suspected_leakage` string into
`checkpoints/<problem>/metadata.json` whenever the ceiling is exceeded, and
`scripts/evaluate.py` prints it in red.

### B3. Screenshots of a running UI (Appendix A2)

`scripts/capture_screenshots.py` is committed, Playwright-driven, and uses seeded
demo defaults only — never real or personal data. It was **not executed**: the
pages it captures render model predictions, and with no trained checkpoints (B2)
every capture would show the `UntrainedNotice` empty state. Screenshots of
nothing are not worth committing, and faking them is the exact failure this
rebuild corrects.

**To unblock after B2:**

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
