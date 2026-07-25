# ADR-0005: Assumptions taken during the autonomous rebuild

## Status

**Accepted** — 2026-07-25. Written by the agent that performed the rebuild, per
the instruction to record any ambiguity resolved without asking.

## Context

The rebuild ran unsupervised against a written brief. Three points were genuinely
ambiguous — the source documents disagreed with each other. Each was resolved by
picking the option most consistent with the governing document, and each is
recorded here so the owner can reverse it cheaply.

## Assumptions

### A1. Web port is 3070, not 3071

**Conflict.** The brief's target-layout section specified `WEB_PORT=3071`; the
brief's Appendix A and `PORT-ALLOCATION.md` both assign `3070` to this repository.

**Resolved:** `3070`. Appendix A states it overrides anything above it in the
brief, and `PORT-ALLOCATION.md` is described as authoritative. Both 3070 and 3071
sit inside this repository's allocated `3070–3079` block, so either would have
been collision-free; 3070 is simply the one the authoritative documents name.

Changed in `.env.example`, `infra/compose/base.yml`, `infra/compose/dev.yml`,
`infra/docker/web.Dockerfile`, `web/package.json`, `Makefile` and both READMEs.

**To reverse:** change `WEB_PORT` and the `-p` flags in `web/package.json`. The
container-internal port and the host port are kept identical on purpose, so there
is exactly one number to change.

### A2. Slice commits are preserved; the branch is merged with `--no-ff`

**Conflict.** The brief's final slice said "squash-merge to `main`". The run
instructions said "commit after EVERY slice — small commits are the point, the
commit graph is a portfolio artifact."

**Resolved:** keep every slice commit and merge with `git merge --no-ff`. A
squash would destroy exactly the artefact the run instructions call valuable, and
the two statements are otherwise compatible: the branch still lands on `main` as
one reviewable unit.

**Nothing was pushed.** The remote is untouched; the owner reviews before
anything ships.

### A3. Slices 4–6 build everything except the training run

**Conflict.** Brief Slices 4, 5 and 6 are "train the model and report the
number". The owner's priority note for this run forbade downloading datasets and
running heavy compute, and forbade using credentials.

**Resolved:** implement the entire path — adapters, splits, features, trainer,
metrics, quality gates, figures and screenshot scripts — and stop at the point
where data or compute is required. The results table ships with **empty cells and
an explicit "not yet measured" note**, never a placeholder number.

The remaining steps, in the exact order to run them, are in
[`docs/PROGRESS.md`](../PROGRESS.md) under "BLOCKED — needs owner".

This is the assumption most worth checking: it means the repository is complete
as *engineering* and incomplete as a *results document*. The README says so
plainly rather than implying otherwise.

## Consequences

Every one of these is reversible in a single commit. None of them changes the
substance of ADR-0003, which is the decision that matters.
