# PROGRESS: adversarial publication repair

Base HEAD: `9ea52c8` on `main`

Working-tree status: intentionally uncommitted. The task explicitly forbids
`git add`, `git commit`, `git checkout`, branch creation, tags, and pushes.
There is therefore no commit SHA for this repair batch.

NEXT ACTION: Armando reviews the complete working-tree diff. After he commits it,
run `./scripts/verify_fresh_clone.sh` from an environment with Docker Desktop and
package-network access. Do not weaken or skip a stage.

## Current outcome

- [x] Published claim ledger audited against metrics CSVs, run records, source
  integrity, split reports, row-level prediction evidence, and figure bytes.
- [x] Invalid ULB stratified result retracted and replaced with a chronological
  result derived from the source's documented `Time` field.
- [x] Every accepted summary cell generated from a training-written metrics CSV.
- [x] Every accepted run has a public record with the base commit, exact training
  source-tree digest, dirty-worktree flag, seed, split, source integrity, dataset
  summary, metric digest, and evaluation scope.
- [x] Both published figures regenerated from held-out predictions and bound to
  their inputs and output PNG hashes in `reports/figures/manifest.json`.
- [x] Result tables and figure captions now show fold or temporal-block spread.
- [x] Publication, hygiene, baseline, provenance, and fresh-clone gates repaired.
- [x] Python and web suites pass in the working tree and in a projected clone
  containing only tracked and intended new files.
- [~] Browser-based Mermaid rerendering could not start Chromium in this sandbox.
  The three committed SVGs pass the static text-containment checker.
- [~] Docker verification could not run because no Docker daemon is reachable.

## Findings and repairs, worst first

### Critical: the fraud evaluation used the wrong split

OpenML metadata names `Time` as dataset 1597's row-id attribute, and the cached
ARFF contains it. Scikit-learn omits row-id attributes from
`fetch_openml(...).frame`. The prior adapter mistook that client behavior for a
source limitation and used stratified five-fold cross-validation.

Repair:

- `src/data/download.py` restores only the documented row-id field from the exact
  cached ARFF and refuses to discard it silently.
- `src/data/adapters/ulb_creditcard.py` requires `Time` and verifies the OpenML
  source MD5.
- `configs/fraud_ulb.yaml` now uses a tie-safe chronological 70/10/20 split.
- `reports/fraud_ulb_split.json` records all partition counts, rates, boundaries,
  and the absence of split-key overlap.
- The old stratified result is explicitly retracted in `reports/RESULTS.md`.

Focused RED: the adapter/config test showed no `Time` and a stratified split.
Focused GREEN: the restored source loaded 284,807 rows with 31 canonical columns,
and the split audit reproduced byte-for-byte on two runs.

### High: published claims were not mechanically closed

The metrics CSVs existed, but the active tables, prose-only caption values,
figure sample counts, and PNG bytes were not all checked against a generated
artifact. A stale or hand-edited claim could pass CI.

Repair:

- Training writes a validated `reports/<run>_run.json`.
- `scripts/check_publication.py` checks summary cells, dataset summaries, both
  temporal split tables, metric digests, training-source digests, figure input
  evidence, finding titles, caption-only Brier values, zero-event-bin counts,
  and output PNG hashes.
- `make publication-check` and CI run that checker.
- Mutation tests prove a changed summary cell and a mismatched metrics digest
  fail.

### High: several quality gates could pass without guarding the publication

Proved RED before repair:

- The aggregate metric gates preferred ignored checkpoints, so a fresh clone
  skipped published results.
- `fraud_ulb` was outside the original problem parametrization.
- The typography scan inspected only tracked files, so an intended untracked
  publication file escaped.
- Figure tests asserted hard-coded counts and did not require finding titles or
  a generated evidence manifest.

Repair:

- Aggregate gates read public run records first and cover every published run.
- Checkpoint-only serving probes retain explicit skips when a model is absent.
- Hygiene scans tracked plus nonignored untracked files.
- Figure tests derive counts from rows, require finding titles, and validate the
  manifest.
- Unit tests no longer depend on ignored local prediction CSVs.

### High: temporal point estimates had no uncertainty

Credit risk published bare test values. The corrected ULB run also needed a
temporal variation view.

Repair:

- Chronological runs report the point estimate on the complete final test plus
  the standard deviation across five adjacent, tie-safe test-time blocks.
- This spread is labeled as descriptive, not a confidence interval. It does not
  cover refitting, model selection, or alternative cut dates.
- Churn retains its five-fold mean and fold standard deviation.
- The ULB point-estimate delta spread includes zero, and the documents say so.

### Medium: the documented sample quickstart crashed

RED:

```bash
MPLCONFIGDIR=/tmp/portfolio-ml-mpl-cache \
UV_CACHE_DIR=/tmp/portfolio-ml-uv-cache UV_OFFLINE=1 make train-sample
```

The sparse 100-row fraud fixture could not form five two-class temporal blocks,
so publication uncertainty raised a `ValueError`. Sample mode is prohibited from
publishing metrics or predictions.

Repair: sample runs skip publication-only temporal spread while the real-data
path still requires all five blocks. A regression test fails if sample mode calls
that calculation. The documented command now exits zero for all four configs and
writes no checkpoint, metrics CSV, run record, or prediction artifact.

### Medium: a warm Kaggle cache still required the network

The LendingClub source existed in kagglehub's versioned cache, but
`dataset_download()` contacted the API before returning it. With network blocked,
the local training command failed.

Repair: the download layer resolves the newest complete cached version first.
The adapter's pinned source digest and row-count checks still reject stale or
wrong bytes. A regression test proves the cached path makes no network call.

### Medium: Wilson intervals could exclude an exact boundary by rounding

A calibration test with an all-positive bin produced a tiny negative error-bar
length because floating-point rounding put the computed upper endpoint just
below the observed proportion of one.

Repair: Wilson endpoints are clamped to contain the observed proportion. Boundary
tests cover zero and one.

## Accepted measured results

Regeneration commands:

```bash
uv run python scripts/train.py --model fraud_ulb
uv run python scripts/describe_split.py --model fraud_ulb \
  --out reports/fraud_ulb_split.json
uv run python scripts/train.py --model credit_risk
uv run python scripts/describe_split.py --model credit_risk \
  --out reports/credit_risk_split.json
uv run python scripts/train.py --model churn
uv run python scripts/make_figures.py --published-only
uv run python scripts/evaluate.py --markdown
uv run python scripts/check_publication.py
```

| Run | PR-AUC | ROC-AUC | Logistic PR-AUC | Delta |
|---|---:|---:|---:|---:|
| `fraud_ulb` | 0.8073 ± 0.1357 | 0.9828 ± 0.0194 | 0.7461 ± 0.1993 | 0.0612 ± 0.0781 |
| `credit_risk` | 0.3935 ± 0.0529 | 0.7160 ± 0.0076 | 0.3720 ± 0.0569 | 0.0215 ± 0.0062 |
| `churn` | 0.9735 ± 0.0078 | 0.9940 ± 0.0019 | 0.7800 ± 0.0217 | 0.1935 ± 0.0145 |

For `fraud_ulb` and `credit_risk`, `±` is the standard deviation across five
adjacent temporal test blocks. For `churn`, it is the fold standard deviation.
The values come from `reports/*_metrics.csv`, and the source-tree digest recorded
by all three final runs is
`eb2b5342e08e7106b6dc6bb199ec66776ba7c5a41d6f8123c196d9f9f58f3531`.

The IEEE-CIS `fraud` and `fraud_autoencoder` rows remain not measured. No result
is inferred or copied into their empty cells.

## Verification

Working-tree commands that pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest -m "not network" --cov=src --cov-report=term-missing \
  --cov-fail-under=80
uv run python scripts/check_publication.py
uv run python scripts/check_diagram_text.py \
  docs/diagrams/c4-container.svg \
  docs/diagrams/pipeline-dag.svg \
  docs/diagrams/sequence-predict.svg
cd web
pnpm lint
pnpm typecheck
pnpm test
pnpm build
```

The final working-tree Python run passed 292 tests, skipped 5
checkpoint-dependent cases, deselected the one network canary, and measured
86.86% source coverage against the 80% floor. Web verification passed 18 test
files and 98 tests, and the production build completed.

A projected clone was assembled from:

```bash
git ls-files -z --cached --others --exclude-standard
```

This is a read-only Git query, not a Git write. In that projected tree:

- the non-network Python suite passed without checkpoints or ignored prediction
  files: 282 passed, 15 skipped, 1 deselected, and 87.07% source coverage;
- all four sample configs trained without publishing artifacts;
- web lint, typecheck, 98 tests, and the production build passed.

Dependency installation itself was not repeated offline because the local caches
lacked the locked Playwright, hatchling, and Base UI tarballs. Existing locked
environments were reused only for executing the projected-tree code.

## Public-safety audit

- No API-key, access-token, or private-key signature was found in public files.
- No personal email was found beyond third-party package metadata; the permitted
  owner identity remains in the public profile and license.
- No absolute author filesystem path appears in public files.
- No restricted or derived real-data row, checkpoint, row-level prediction file,
  or MLflow store is tracked.
- Only the four generated synthetic CI fixtures are under `data/sample/`.
- LendingClub and churn uploader license tags are described as unverified, and
  the repository does not redistribute their rows.
- ULB redistribution remains unresolved and no row is redistributed.
- All README-relative links resolve; the CI badge names the existing workflow.
- No literal or encoded U+2014 character is present in the publishable tree.

## BLOCKED and known limits

### IEEE-CIS data

The competition download still needs an accepted competition account and a
classic `~/.kaggle/kaggle.json` token. Exact unblock:

```bash
uv run python scripts/download_data.py --dataset ieee-cis
uv run python scripts/train.py --model fraud
uv run python scripts/train.py --model fraud --autoencoder
```

### Docker

`docker info` reports no reachable daemon in this environment. The Docker build,
compose health check on port 8070, and untrained 503 response were not rerun.
They remain required in the post-commit fresh-clone verifier.

### Browser-rendered Mermaid gate

The locked Mermaid CLI 11.16.0 is cached, but Chromium launch is denied by this
sandbox before any diagram renders. All three committed SVGs pass the static
geometry checker. Browser rerendering remains required outside the sandbox.

### Actual fresh-clone script

The task forbids every Git command that writes, so
`scripts/verify_fresh_clone.sh` was not run because its first stage executes
`git clone`. The projected-tree verification above exercised the repaired files
without violating that rule. After the owner commits, run the actual verifier
with network access and Docker available.

## Decisions

- Restore the documented OpenML row-id attribute instead of using row order or a
  random split.
- Retract the invalid result rather than preserving its larger score.
- Do not invent an expected range for the corrected ULB run after seeing it.
- Persist temporal test predictions locally for audits, but keep them ignored
  because real row-level derivatives are not redistributable.
- Commit only aggregate metrics, generated public run records, split reports,
  figure hashes, and publication figures.
- Treat source-tree bytes as run provenance when policy requires training from
  an uncommitted worktree.
- Keep sample smoke training isolated from every publication artifact.
