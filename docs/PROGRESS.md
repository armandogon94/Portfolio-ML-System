# PROGRESS: fintech real-data rebuild

HEAD: `6cc1361` on `main`
NEXT ACTION: **Create the two verified local commits.** The source workspace is
writable, but this session cannot create `.git/index.lock`. Run:

```bash
git add .github/workflows/ci.yml scripts/verify_fresh_clone.sh web/lib/industries.ts
git -c user.name="Armando Gonzalez" -c user.email="armandogon94@gmail.com" \
  commit -m "fix(ci): reject retired README metric"

git add README.md THIRD_PARTY_NOTICES.md data/README.md docs/PROGRESS.md \
  docs/adr/0001-gradio-to-nextjs.md docs/adr/0002-experiment-tracking.md \
  docs/adr/0003-real-data-over-synthetic.md docs/adr/0004-narrow-to-fintech.md \
  docs/adr/0005-assumptions-log.md docs/ports.example.md pyproject.toml \
  reports/RESULTS.md scripts/make_figures.py web/README.md
git -c user.name="Armando Gonzalez" -c user.email="armandogon94@gmail.com" \
  commit -m "docs: remove dangling image claims and em dashes"
```

Do not push.

## CI green repair: VERIFIED, COMMIT BLOCKED

- The removed README correction heading produced the focused RED result:
  `grep -q "A correction, and why it's here" README.md` exited 1.
- README contains no `0.964`; the replacement negative assertion and the
  existing ADR-link assertion both pass.
- The workflow comment and fresh-clone hygiene check now describe and enforce
  the current contract. The docs gate also found a retired placeholder token in
  a TypeScript comment; the comment was corrected and the unchanged gate passes.
- The limitation text now says that no screenshots are tracked and points to
  the regenerable capture script. `scripts/make_figures.py` no longer creates an
  empty image directory. The remaining tracked path reference is the capture
  script's runtime output message.
- The exact tracked-Markdown count moved from 138 to 0:
  `git ls-files '*.md' | xargs grep -c "$(printf '\u2014')"`.
- Exit 0: Ruff check, Ruff format check, mypy, non-network pytest with coverage,
  README link check, README claims check, Mermaid/SVG count, synthetic-generator
  check, static diagram text inspection of all committed SVGs, generated results
  Markdown, shell syntax, web lint, web typecheck, web tests, and web build.
- Pytest result: 262 passed, 5 skipped, 1 deselected, 88.26% coverage.
- Web result: 18 files and 98 tests passed; the production build completed.
- Exit 0: `uv lock --check --offline` and the pnpm frozen lockfile-only check.
- Not run by task rule: Docker builds, full fresh-clone verification, live
  services, screenshot capture, training, and browser-based diagram rendering.
- Exact dependency installs were not repeated because the sandbox blocks package
  network access and writes to the existing user caches.
- BLOCKED: `git add` cannot create `.git/index.lock` in this session and exits
  with `Operation not permitted`. The required path-scoped commands are above.

## Diagram text containment slice: BLOCKED

- [x] Copied `scripts/check_diagram_text.py` from the tested owner-provided
  path.
- [x] Added the required top-level `htmlLabels: false` directive to every
  Mermaid source and every Markdown Mermaid block.
- [x] Shortened and wrapped the C4, prediction sequence, and training pipeline
  labels. The single-node C4 subgraphs were folded into their nodes.
- [x] Added `make diagrams-check` and the Python 3.11 CI gate. Removed the
  obsolete SVG label hardening step and script.
- [x] Re-ran `.venv/bin/python scripts/evaluate.py --markdown`: exit 0 with
  the accepted values unchanged.
- [x] Re-ran `.venv/bin/pytest -m "not network"`: 262 passed, 5 skipped,
  1 deselected, and 88.26% coverage.
- [x] Re-ran Ruff and `git diff --check`: both exited 0.
- BLOCKED Regeneration and rendered inspection. The owner-provided
  Mermaid CLI 11.16.0 reached Chromium startup, then macOS denied its Mach port
  rendezvous with `Permission denied`. The checker could not render any of the
  12 source instances in this sandbox.
- KNOWN ISSUE The current SVG exports are stale. Direct inspection with the
  checker reports `foreignObject` labels in the C4 and pipeline SVGs, plus
  overflowing sequence labels. No diagram has a PASS result yet.
- COMMIT No commit exists for this incomplete slice. Committing source files
  while their generated exports are stale would violate the repository
  contract.

## Deferred next action: IEEE-CIS unblock

Install a classic
`~/.kaggle/kaggle.json` API token from
<https://www.kaggle.com/settings/account>, set mode `600`, and re-run the two
competition-download checks below. The OAuth token in
`~/.kaggle/access_token` was re-confirmed on 2026-07-25 to return 403 for both
`kagglehub.competition_download` and the `kaggle competitions download` CLI,
although it still works for Kaggle datasets.
SUCCESS: `uv run python scripts/download_data.py --dataset ieee-cis` completes;
then run the two blocked fraud training commands without changing their rows in
the results documents beforehand.

**`./scripts/verify_fresh_clone.sh` PASSES: all seven stages, 2026-07-25 at
`60ee15f`.** This is the first time stage 7 has ever completed. Verbatim tail:

```
==> [7/7] docker compose up + /health
    PASS: docker build (API image)
    PASS: GET /health
    PASS: POST /predict/fraud -> 503 on an untrained clone (correct)
    PASS: teardown
PASS: every fresh-clone stage ran and the documented quickstart reproduced.
```

Stage 7 had never passed because every container healthcheck fetched
`http://localhost:<port>`, and those images resolve `localhost` to `::1` only
while the servers bind IPv4 `0.0.0.0`. The healthchecks now use `127.0.0.1`.

The verifier clones committed `HEAD`, so it verifies the last commit rather than
the working tree.

Status legend: `[x]` complete · `[~]` partial · `[ ]` not started · `BLOCKED`
needs an external input. The credit-risk result batch is committed locally on
`main`. **Nothing has been pushed**: the public remote still serves the
pre-retraction tree, and pushing remains an owner decision.

## Accepted measured results

The only accepted result rows are generated by these commands:

```bash
uv run python scripts/train.py --model fraud_ulb
uv run python scripts/train.py --model churn
uv run python scripts/download_data.py --dataset lending-club
uv run python scripts/train.py --model credit_risk
uv run python scripts/describe_split.py --model credit_risk
uv run python scripts/evaluate.py --markdown
```

| Run | PR-AUC | ROC-AUC | Baseline PR-AUC | Δ |
|---|---|---|---|---|
| `fraud_ulb` | 0.8569 ± 0.0331 | 0.9810 ± 0.0092 | 0.7300 ± 0.0279 | 0.1269 ± 0.0355 |
| `credit_risk` | 0.3935 | 0.7160 | 0.3720 | 0.0215 |
| `churn` | 0.9735 ± 0.0078 | 0.9940 ± 0.0019 | 0.7800 ± 0.0217 | 0.1935 ± 0.0145 |

Sources: `reports/fraud_ulb_metrics.csv`,
`reports/credit_risk_metrics.csv`, `reports/churn_metrics.csv`, and the
checkpoint metadata written by the three training commands. The fraud and churn
metadata name training commit
`05d32cae0d54dee97a70be575097182e9b5f8278`; credit risk names
`052adab726b6dd5a176cfdc737110b172198a749`. All three use seed 42.

The `fraud` and `fraud_autoencoder` rows remain empty. No run has produced
their tables. Their expected-performance bands are hand-entered config values
and are not presented as measurements.

## Historical slices and commit map

- [x] **Slice 1: trust-signal repairs.** Commit `1def9ad` tracked `uv.lock`,
  added the licence, and retracted the invalid synthetic results.
- [x] **Slice 2: narrow to fintech.** Commits `75afe98`, `f50c1af`, and
  `d3d9951` removed the non-fintech domains, narrowed the web routes, and
  removed superseded process files.
- [x] **Slice 3: real-data acquisition.** Commit `75afe98` contains the data
  layer, configs, adapters, training refactor, serving split, and test rebuild.
  Commit `e15842a` added and validated the credential-free ULB/OpenML path.
- BLOCKED **Slice 4: IEEE-CIS fraud result.** The implementation is in
  `75afe98`; no result commit exists because the competition download is still
  blocked.
- [x] **Slice 5: LendingClub credit risk result.** The implementation is in
  `75afe98`; the real run completed on 2026-07-26 UTC with training SHA
  `052adab726b6dd5a176cfdc737110b172198a749`. It measured **0.3935 PR-AUC**,
  **0.7160 ROC-AUC**, a **0.3720 logistic-regression PR-AUC**, and a
  **+0.0215 PR-AUC** margin. The run used:

  ```bash
  uv run python scripts/download_data.py --dataset lending-club
  uv run python scripts/train.py --model credit_risk
  uv run python scripts/describe_split.py --model credit_risk
  uv run python scripts/evaluate.py --markdown
  ```

  `scripts/describe_split.py` is the new non-training audit command for
  partition counts, positive rates, time bounds, and split-key overlap. The
  result CSV, script, test, and publication edits are committed locally and
  unpushed.
- [x] **Slice 6: card attrition result.** Pipeline work is in `75afe98` and
  `e15842a`; the accepted measured row was published in `4584c9b`.
- [x] **Slice 7: config-driven trainer and split serving modules.** Commit
  `75afe98`.
- [x] **Slice 8: tests that can fail.** Core rebuild in `75afe98`,
  constant-predictor repair in `57b4921`, methodology gates in `e15842a`.
  The current port, screenshot, coverage, and web-test gate fixes are
  uncommitted in this batch.
- [x] **Slice 9: documentation, diagrams, ADRs, and results report.** Commit
  `f80fec0`; measured result publication in `4584c9b`.
- [~] **Slice 10: CI and fresh-clone verification.** Initial verifier/CI in
  `f80fec0`, prior verifier repair in `57b4921`. This batch fixes the
  unconditional PASS, adds fixture smoke training and web README link checks,
  and wires the Python matrix; a full verifier run against the repaired script
  requires an owner-created commit first.

## BLOCKED and not run

### IEEE-CIS competition download: BLOCKED

`~/.kaggle/access_token` authenticates Kaggle dataset downloads but the
competition downloads remain blocked. Both paths were re-confirmed on
2026-07-25:

```bash
./.venv/bin/python -c "import kagglehub; kagglehub.competition_download('ieee-fraud-detection')"
kaggle competitions download -c ieee-fraud-detection
```

Observed error:

```text
403 ... Please make sure you are authenticated and have accepted the competition rules
```

Install a classic API token from <https://www.kaggle.com/settings/account>, then
run:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

uv run python scripts/download_data.py --dataset ieee-cis
uv run python scripts/train.py --model fraud
uv run python scripts/train.py --model fraud --autoencoder
```

### Screenshot success gate: BLOCKED on a runnable measured-model UI

The screenshot command must install its browser, launch a browser, write real
PNG files, and exit zero. Missing Playwright now produces zero captures and a
non-zero command result instead of a false “screenshot written” message.

```bash
make screenshots-install
make docker-up
make screenshots
```

No screenshots are tracked. Do not mark the screenshot gate complete until the
committed capture script has produced the dashboard, measured-model prediction,
and MLflow views and they have been visually inspected.

## Current audit repair

- [x] Unsupported magnitudes removed: MPS speedup, full-frame encoding AUC
  effect, random-split inflation magnitude, remembered leaderboard range, and
  machine load average.
- [x] Source counts corrected: the credit-risk denylist has 29 entries and the
  autoencoder has six `nn.Linear` transforms with a 16-unit bottleneck.
- [x] Ports 80 and 443 added to the native launcher refusal set, with a
  parameterized test covering every documented reserved port.
- [x] Web tests no longer pass when the test directory is empty; local coverage
  now enforces the same threshold as CI; the CI placeholder-copy check can fail.
- [x] The fresh-clone verifier records every skipped stage. Default skips end
  `INCOMPLETE` with non-zero status; `--allow-skips` ends `PARTIAL PASS`.
- [x] The verifier runs `make train-sample`, checks both onboarding READMEs, and
  states that checkpoint-dependent quality gates skip on a fresh clone.
- [x] The broken `docker-test` target and stale production compose file were
  removed rather than left as non-working commands.
- [x] Screenshot dependencies and the Chromium install target were added; figure
  generation already exits non-zero with the exact train command when no
  checkpoint exists.
- [x] Docker dependencies are version-pinned, the CI Python matrix selects its
  declared interpreter, and `uv` is documented as a prerequisite.
- [x] Python/Kaggle credential behavior is documented honestly: Python
  entrypoints read process environment variables and Kaggle files, not `.env`.
- [x] The Kaggle OAuth precheck gap was fixed on 2026-07-25 in `e15842a`.
  `ensure_kaggle_env()` accepts env vars, `kaggle.json`, or
  `~/.kaggle/access_token`; the three fake-home tests remain in
  `tests/data/test_download.py` and are re-verified in this batch.
- [x] The fake no-run provenance footer is absent. Because real runs were later
  completed, `reports/RESULTS.md` retains their past-tense provenance and
  clearly separates the rows no run has produced.

Source-audit and focused-test commands for the counts above:

```bash
./.venv/bin/python - <<'PY'
import torch.nn as nn
import yaml

from src.models.autoencoder import FraudAutoencoder

with open("configs/credit_risk.yaml") as stream:
    config = yaml.safe_load(stream)
model = FraudAutoencoder(input_dim=1)
print("denylist entries:", len(config["data"]["denylist"]))
print("linear transforms:", sum(isinstance(layer, nn.Linear) for layer in model.modules()))
print("bottleneck units:", model.encoder[-3].out_features)
PY
./.venv/bin/pytest -p no:cacheprovider --no-cov -m "not network" \
  tests/test_serve.py tests/test_capture_screenshots.py tests/data/test_download.py -q
```

## Decisions made and why

- The obsolete production compose file and broken `docker-test` target were
  removed: a documented path that deterministically fails is worse than no
  path.
- Playwright is a locked development dependency with an explicit Chromium
  install target so screenshot success means a browser actually ran.
- The verifier uses `INCOMPLETE` by default for skipped stages; only an explicit
  `--allow-skips` request may produce `PARTIAL PASS`. `PASS` is reserved for a
  run in which every stage executed.
- Historical architectural decisions remain in `docs/adr/`. The amendment in
  `docs/adr/0001-gradio-to-nextjs.md` corrects stale archaeology without
  rewriting the original decision.

## Known issues and verification limits

- The ULB credential-free path gap is resolved in `e15842a` (2026-07-25);
  this batch re-verifies the credential-source tests rather than claiming a
  second implementation.
- The verifier's former unconditional PASS is resolved in this uncommitted
  batch. The RED command and observed false-success behavior are recorded
  below.
- The current batch is intentionally uncommitted. The fresh-clone verifier
  cannot exercise these exact files until the owner commits them; this is a
  property of its committed-HEAD isolation gate, not a reason to weaken it.
- The full fresh-clone Docker stage has not been rerun for this uncommitted
  batch. `bash -n` and the explicit false-PASS simulation cover the shell logic;
  the complete verifier run remains a later owner-review check.
- The screenshot success gate remains open as documented above.
- Only the IEEE-CIS `fraud` and `fraud_autoencoder` result rows remain empty.
- The requested single-file pytest command executes all three new assertions but
  the repository-wide `--cov=src` configuration then exits non-zero because one
  focused file cannot meet the global 80% coverage gate. The focused convention
  below uses `PYTEST_ADDOPTS=--no-cov`; the coverage threshold was not weakened.

## Credit-risk publication verification

```bash
MPLCONFIGDIR=/tmp/codex-matplotlib-cache PYTEST_ADDOPTS=--no-cov \
  .venv/bin/python -m pytest tests/training/test_describe_split.py -q
.venv/bin/python scripts/evaluate.py --markdown
```

```text
...                                                                      [100%]
| credit_risk | 0.3935 | 0.7160 | 0.3720 | 0.0215 | 0.5807 | 0.0457 |
```

`git diff --check` passed. The final stale-claim grep returned only the two
blocked IEEE-CIS rows and the deliberately empty controlled comparisons.

Whole-suite state after the run and the two fixes below:

```text
256 passed, 5 skipped, 12 warnings in 27.52s
Required test coverage of 80.0% reached. Total coverage: 88.26%
```

`ruff check`, `ruff format --check` and `mypy src` are all clean.

### Two bugs the first real credit-risk checkpoint exposed

Both were pre-existing and both were invisible until a `credit_risk` checkpoint
existed on disk. Neither test was weakened to make it pass.

1. **The credit-risk serving path raised on every request.**
   `tests/test_quality_gates.py::test_a_higher_fico_score_does_not_raise_default_risk`
   skips itself when there is no checkpoint, so it had never run. With one, it
   failed, not on the assertion, but with
   `AttributeError: Can only use .dt accessor with datetimelike values` from
   `src/features/credit_risk_features.py`. The adapter parses `issue_d` and
   `earliest_cr_line` to `datetime64`, but `src/serving/preprocessing.py` builds
   its one-row frame from the request payload, where any unsupplied key is a
   float `NaN`, so the shared feature module met `float64` where it assumed
   dates. Fixed by coercing both columns with `pd.to_datetime(errors="coerce")`
   inside the shared module: a no-op on the training frame, and `NaT` (hence a
   `NaN` feature, which LightGBM reads as "unknown") on a partial request. The
   gate now passes on its merits. The model does score higher FICO as lower
   default risk.

2. **`KAGGLEHUB_TOKEN_PATH` was frozen at import time.**
   `tests/data/test_download.py::test_real_kaggle_canary` failed in a full-suite
   run but passed alone. `src/data/kaggle_credentials.py` bound the OAuth token
   path as a module-level constant, and the module is first imported inside a
   test that redirects `Path.home()` to a `tmp_path`, pinning the constant to a
   temporary directory for the rest of the process, so every later credential
   check reported "no token" against a home that never existed. Replaced with
   `kagglehub_token_path()`, resolved per call, matching what
   `load_kaggle_creds` already did.

### Reproducibility of the credit-risk number

`credit_risk` was trained twice at seed 42 on 2026-07-26, the second time after
fix 1 above touched a training code path. Every metric in
`reports/credit_risk_metrics.csv` was byte-identical across both runs; only
`trained_at` differs. The checkpoint on disk is the second run.

## RED/GREEN evidence for the prior audit batch

RED commands:

```bash
PATH=/usr/bin:/bin SKIP_WEB=1 SKIP_DOCKER=1 ./scripts/verify_fresh_clone.sh
./.venv/bin/pytest -p no:cacheprovider --no-cov tests/test_serve.py tests/test_capture_screenshots.py -q
```

The first command exited zero while all three substantive stages were skipped
and printed an unconditional PASS. The focused pytest command failed for the
missing reserved-port constant, both privileged ports reaching uvicorn, and a
missing Playwright import being counted as one capture.

Focused GREEN:

```text
...........................                                              [100%]
```

Command:

```bash
./.venv/bin/pytest -p no:cacheprovider --no-cov -m "not network" \
  tests/test_serve.py tests/test_capture_screenshots.py tests/data/test_download.py -q
```

Final non-network suite:

```bash
./.venv/bin/pytest -p no:cacheprovider --no-cov -m "not network" 2>&1 | tail -5
```

```text
  <repo>/.venv/lib/python3.11/site-packages/mlflow/tracking/_tracking_service/utils.py:184: FutureWarning: The filesystem tracking backend (e.g., './mlruns') is deprecated as of February 2026. Consider transitioning to a database backend (e.g., 'sqlite:///mlflow.db') to take advantage of the latest MLflow features. See https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store for migration guidance.
    return FileStore(store_uri, store_uri)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
246 passed, 11 skipped, 1 deselected, 13 warnings in 35.66s
```

Final lint/format gate:

```bash
./.venv/bin/ruff check . && ./.venv/bin/ruff format --check .
```

```text
All checks passed!
87 files already formatted
```

`bash -n scripts/verify_fresh_clone.sh` and
`cd web && pnpm exec tsc --noEmit` both exited zero with no output. The required
retraction grep returned only values inside explicitly historical/retraction
narratives; its verbatim output belongs in the final audit handoff rather than
being copied here and recursively becoming a new grep result.
