# ADR-0002: MLflow as the primary tracker, W&B optional

## Status

**Accepted** — 2026-04-11, migrated from `decision.md` Decision 1 on 2026-07-25.

## Context

The training code used Weights & Biases with a graceful fallback when no API key
was present. `docker-compose.yml` declared an MLflow service that **no code
connected to** — it was dead weight in the compose file.

Both are industry-standard. The question was whether to pick one.

## Options

| Option | Pros | Cons |
|---|---|---|
| A) Replace W&B with MLflow | Single tool, self-hosted | Discards a working integration |
| B) Keep W&B only | No new code | The MLflow service stays dead; no model registry |
| C) MLflow primary, W&B optional | Self-hosted with zero config; gives a model registry | Two tracking UIs to keep coherent |

## Decision

**C — MLflow is primary and always on; W&B is optional and off by default.**

MLflow needs no account, no API key and no network: the default tracking URI is
a local `mlruns/` directory, and `docker compose` raises a server on the repo's
assigned port. That matters more than it sounds — a reviewer cloning this
repository gets working experiment tracking without signing up for anything.

W&B stays wired but inert. `BaseTrainer._init_wandb` treats the literal
placeholder string shipped in `.env.example` as "unset", so a user who copies the
example file does not get a confusing auth failure for a feature they did not ask
for.

The deciding capability is the **model registry**. `BaseTrainer.register_model`
creates a registered model and a version per run, so a checkpoint has a lifecycle
(versioned, promotable) rather than just a file on disk. The W&B free tier does
not offer that.

## Consequences

- Tracking failures never fail a training run. Every MLflow and W&B call in
  `BaseTrainer` is wrapped: losing a registry entry is not a reason to lose a
  trained model. The warning is logged and the run continues.
- All three problems log into **one** experiment (`fintech-ml-system`, set by
  `training.mlflow_experiment` in each config) so their runs are directly
  comparable in the UI. `web/lib/dashboard.ts` hardcodes that same name — if it
  changes in the configs, it must change there too.
- MLflow's SQLite store records artifact URIs as absolute paths. Moving the repo
  directory therefore breaks artifact links in *existing* runs. New runs are
  unaffected. This is a known MLflow limitation, not something to work around.
- MLflow runs on port **5070** on the host. Not 5000: macOS AirPlay Receiver
  binds 5000, and a container that binds it appears to start and is then
  unreachable. See [`docs/ports.example.md`](../ports.example.md).

## References

- `src/training/trainer.py` — `_init_mlflow`, `_init_wandb`, `register_model`
- `infra/compose/base.yml` — the `mlflow` service
