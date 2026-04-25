/**
 * Typed MLflow REST client.
 *
 * Hits MLflow's official REST API directly (no Python SDK needed) so the
 * Next.js dashboard (task A.9.9) can fetch run history from Server
 * Components / Route Handlers. Every response is Zod-validated at the
 * boundary — a shape change on MLflow's side surfaces as a typed error
 * rather than a `cannot read property of undefined` at render time.
 *
 * Endpoints used:
 *   GET  /api/2.0/mlflow/experiments/get-by-name?experiment_name=<name>
 *   POST /api/2.0/mlflow/runs/search
 *
 * Base URL resolves from `MLFLOW_TRACKING_URI` (env) → falls back to
 * `http://mlflow:5000` (compose network DNS) so the default works inside
 * the docker-compose stack with no extra config.
 */
import { z } from "zod";

// ─── Errors ────────────────────────────────────────────────────────────────

/** Thrown for any non-2xx MLflow response or shape-validation failure. */
export class MlflowError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(`MLflow ${status}`);
    this.name = "MlflowError";
  }
}

// ─── Schemas (internal) ────────────────────────────────────────────────────

const ExperimentSchema = z.object({
  experiment_id: z.string(),
  name: z.string(),
});

const MetricSchema = z.object({
  key: z.string(),
  value: z.number(),
  timestamp: z.number(),
  step: z.number(),
});

const RunSchema = z.object({
  info: z.object({
    run_id: z.string(),
    end_time: z.number(), // millis since epoch
    status: z.string(),
  }),
  data: z
    .object({
      metrics: z.array(MetricSchema).optional(),
    })
    .optional(),
});

const GetExperimentResponseSchema = z.object({
  experiment: ExperimentSchema,
});

const SearchRunsResponseSchema = z.object({
  runs: z.array(RunSchema).optional(),
});

/** MLflow returns a JSON error body with `error_code` + `message` on 4xx/5xx. */
const MlflowErrorBodySchema = z.object({
  error_code: z.string().optional(),
  message: z.string().optional(),
});

// ─── Public types ──────────────────────────────────────────────────────────

export type RunHistoryPoint = { endTime: number; metric: number };

// ─── Internals ─────────────────────────────────────────────────────────────

const DEFAULT_BASE_URL = "http://mlflow:5000";

function resolveBaseUrl(override?: string): string {
  if (override) return override.replace(/\/$/, "");
  const fromEnv = process.env.MLFLOW_TRACKING_URI;
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  return DEFAULT_BASE_URL;
}

async function parseOrThrow<T>(res: Response, schema: z.ZodSchema<T>): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new MlflowError(res.status, body);
  }
  const json = await res.json();
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    throw new MlflowError(res.status, json);
  }
  return parsed.data;
}

// ─── Public API ────────────────────────────────────────────────────────────

/**
 * Fetch the most recent runs for `experimentName` and return one point per run
 * containing `endTime` (ms since epoch) and `metric` (the value of `metricKey`).
 *
 * Behavior:
 * - Returns an empty array if the experiment doesn't exist yet — this is a
 *   normal state for models that haven't been trained, and we don't want a
 *   missing experiment to crash the dashboard.
 * - Skips runs that don't contain `metricKey` (e.g. failed runs, or runs from
 *   a different model variant logged into the same experiment).
 * - Sorts ascending by `endTime` so the caller can feed it directly into a
 *   left-to-right sparkline. (MLflow returns DESC; we flip on the way out.)
 *
 * Throws `MlflowError` on any non-2xx response or shape mismatch other than
 * the well-known "experiment doesn't exist" path.
 */
export async function getRunHistory(
  experimentName: string,
  metricKey: string,
  options?: { limit?: number; baseUrl?: string },
): Promise<RunHistoryPoint[]> {
  const baseUrl = resolveBaseUrl(options?.baseUrl);
  const limit = options?.limit ?? 10;

  // Step 1: resolve experiment_id from name. Treat RESOURCE_DOES_NOT_EXIST as
  // "no history yet" rather than an error — untrained models shouldn't break
  // the dashboard.
  const expUrl = `${baseUrl}/api/2.0/mlflow/experiments/get-by-name?experiment_name=${encodeURIComponent(
    experimentName,
  )}`;
  const expRes = await fetch(expUrl);

  if (!expRes.ok) {
    const body = await expRes.json().catch(() => null);
    const parsedErr = MlflowErrorBodySchema.safeParse(body);
    if (parsedErr.success && parsedErr.data.error_code === "RESOURCE_DOES_NOT_EXIST") {
      return [];
    }
    throw new MlflowError(expRes.status, body);
  }

  const expJson = await expRes.json();
  const expParsed = GetExperimentResponseSchema.safeParse(expJson);
  if (!expParsed.success) {
    throw new MlflowError(expRes.status, expJson);
  }
  const experimentId = expParsed.data.experiment.experiment_id;

  // Step 2: search runs for that experiment, ordered newest-first.
  const searchRes = await fetch(`${baseUrl}/api/2.0/mlflow/runs/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      experiment_ids: [experimentId],
      max_results: limit,
      order_by: ["attributes.end_time DESC"],
    }),
  });

  const searchData = await parseOrThrow(searchRes, SearchRunsResponseSchema);
  const runs = searchData.runs ?? [];

  // Step 3: project to {endTime, metric} and drop runs that don't have the metric.
  const points: RunHistoryPoint[] = [];
  for (const run of runs) {
    const metric = run.data?.metrics?.find((m) => m.key === metricKey);
    if (metric === undefined) continue;
    points.push({ endTime: run.info.end_time, metric: metric.value });
  }

  // Sort oldest-first so a sparkline reads left-to-right.
  points.sort((a, b) => a.endTime - b.endTime);
  return points;
}
