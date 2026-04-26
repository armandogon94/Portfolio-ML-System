/**
 * Dashboard data shape + server-side join (A.9.9).
 *
 * `DashboardRow` is the public contract consumed by DashboardTable
 * and IndustrySummaryTile. `getDashboardRows()` produces the rows
 * by joining INDUSTRIES (industries.ts) × FastAPI `/models`
 * metadata × MLflow run history.
 *
 * Architecture decision (deviation from SPEC §A.9.9):
 * The original spec planned Next.js Route Handlers under
 * `web/app/api/models/route.ts` and `/api/mlflow-history/route.ts`.
 * That can't work because `next.config.mjs` already wholesale
 * rewrites `/api/*` → FastAPI — route handlers under that path
 * would never receive requests. Instead this function fetches
 * directly from `${INTERNAL_API_URL}/models` server-side and calls
 * `getRunHistory` from `lib/mlflow.ts` directly. Net effect: same
 * data, fewer files, no rewrite-rule special cases.
 */

import { INDUSTRIES } from "@/lib/industries";
import { getRunHistory } from "@/lib/mlflow";
import type { ModelStatus } from "@/components/ModelStatusBadge";

export type DashboardKeyMetric = {
  /** Raw MLflow metric key (e.g., "test_auc_roc"). */
  name: string;
  /** Display label (e.g., "AUC-ROC", "RMSE"). */
  label: string;
  /** Latest value, or null when no checkpoint exists. */
  value: number | null;
  /** Optional unit suffix — e.g., "%" for percentages. Omit for unitless. */
  unit?: string;
  /** Used by sort + recommendation logic; mirrors scripts/train.py RECOMMENDATION_KEY. */
  higherIsBetter: boolean;
};

export type DashboardRow = {
  /** Industry slug from industries.ts (e.g., "fintech"). */
  industrySlug: string;
  /** Display name of the industry (e.g., "Fintech"). */
  industryTitle: string;
  /** Model slug under the industry (e.g., "credit-risk"). */
  modelSlug: string;
  /** Display name of the model. */
  modelTitle: string;
  /** One-line description from industries.ts. */
  modelDescription: string;
  /** Lifecycle state — drives the badge color + sort key. */
  status: ModelStatus;
  /** Link to the model demo page when ready, else null. */
  href: string | null;
  /** Key metric (latest run) plus rendering metadata. Null when no checkpoint. */
  keyMetric: DashboardKeyMetric | null;
  /** ISO timestamp of last training run, or null. */
  lastTrained: string | null;
  /** Sparkline series — oldest first. Empty array when no MLflow history. */
  history: number[];
};

// ─── Mappings: industry/model slug → backend "problem" name + key metric ───
// Mirrors scripts/train.py (TRAINERS, CONFIG_NAMES, RECOMMENDATION_KEY) so
// the dashboard surfaces the same key metric the trainer ranks modalities by.

type ProblemSpec = {
  /** Backend checkpoint directory + /models key. */
  problem: string;
  /** Key metric for ranking + sparkline. */
  metricName: string;
  /** Display label for the metric. */
  metricLabel: string;
  /** Higher metric is better (e.g., AUC, R²) vs lower (e.g., RMSE, MAE). */
  higherIsBetter: boolean;
};

const PROBLEM_BY_INDUSTRY_MODEL: Record<string, ProblemSpec> = {
  "fintech/credit-risk": {
    problem: "credit_risk",
    metricName: "test_auc_roc",
    metricLabel: "AUC-ROC",
    higherIsBetter: true,
  },
  "fintech/fraud": {
    problem: "fraud_detection",
    metricName: "test_autoencoder_auc_roc",
    metricLabel: "AUC-ROC",
    higherIsBetter: true,
  },
  "fintech/churn": {
    problem: "customer_churn",
    metricName: "test_auc_roc",
    metricLabel: "AUC-ROC",
    higherIsBetter: true,
  },
  "real-estate/price": {
    problem: "price_prediction",
    metricName: "test_r2",
    metricLabel: "R²",
    higherIsBetter: true,
  },
  "real-estate/rental-price": {
    problem: "rental_price",
    metricName: "test_r2",
    metricLabel: "R²",
    higherIsBetter: true,
  },
  "dental/no-show": {
    problem: "dental_noshow",
    metricName: "test_auc_roc",
    metricLabel: "AUC-ROC",
    higherIsBetter: true,
  },
  "healthcare/heart-disease": {
    problem: "heart_disease",
    metricName: "test_auc_roc",
    metricLabel: "AUC-ROC",
    higherIsBetter: true,
  },
  "logistics/demand": {
    problem: "demand_forecasting",
    metricName: "test_avg_mae",
    metricLabel: "MAE",
    higherIsBetter: false,
  },
  "logistics/eta": {
    problem: "delivery_eta",
    metricName: "test_rmse",
    metricLabel: "RMSE",
    higherIsBetter: false,
  },
  "legal/h1b-approval": {
    problem: "h1b_approval",
    metricName: "test_auc_roc",
    metricLabel: "AUC-ROC",
    higherIsBetter: true,
  },
};

// Shape of one entry in the FastAPI /models response. Keys at the
// top level are problem names; the value is a copy of metadata.json.
type ModelInfo = {
  problem?: string;
  metrics?: Record<string, number>;
  timestamp?: string;
  [k: string]: unknown;
};
type ModelInfoMap = Record<string, ModelInfo>;

async function fetchModelInfo(): Promise<ModelInfoMap> {
  const baseUrl = process.env.INTERNAL_API_URL ?? "http://localhost:8070";
  try {
    const res = await fetch(`${baseUrl}/models`, {
      // ISR: 30s — same cadence as the /dashboard page revalidate.
      next: { revalidate: 30 },
    });
    if (!res.ok) return {};
    const json = (await res.json()) as ModelInfoMap;
    return json ?? {};
  } catch {
    // FastAPI down → empty map → every row falls back to not_built.
    // Don't throw; the dashboard should still render an empty-state
    // table instead of crashing the whole route.
    return {};
  }
}

async function safeGetRunHistory(experimentName: string, metricKey: string): Promise<number[]> {
  // The dashboard sparklines are decorative — never let an MLflow
  // outage take down the whole page. Fall back to empty history.
  try {
    const points = await getRunHistory(experimentName, metricKey, { limit: 10 });
    return points.map((p) => p.metric);
  } catch {
    return [];
  }
}

/**
 * Build the full DashboardRow[] for the dashboard page.
 *
 * Cross-joins industries.ts × FastAPI /models × MLflow history and
 * resolves each (industry, model) pair to either a ready row (with
 * metric + history) or a not_built row (everything else).
 *
 * Robust to backend outages: if FastAPI or MLflow is unreachable,
 * the returned rows fall back to safe defaults so the page renders
 * an "empty" dashboard instead of a 500.
 */
export async function getDashboardRows(): Promise<DashboardRow[]> {
  const modelInfo = await fetchModelInfo();

  // First pass: build base rows synchronously.
  const baseRows = INDUSTRIES.flatMap((ind) =>
    ind.models.map((m) => {
      const key = `${ind.slug}/${m.slug}`;
      const spec = PROBLEM_BY_INDUSTRY_MODEL[key];
      const info = spec ? modelInfo[spec.problem] : undefined;
      const isReady = Boolean(info?.metrics) && m.ready;

      return {
        industrySlug: ind.slug,
        industryTitle: ind.title,
        modelSlug: m.slug,
        modelTitle: m.title,
        modelDescription: m.description,
        status: isReady ? ("ready" as ModelStatus) : ("not_built" as ModelStatus),
        href: isReady ? `/${ind.slug}/${m.slug}` : null,
        keyMetric:
          isReady && spec
            ? {
                name: spec.metricName,
                label: spec.metricLabel,
                value: info?.metrics?.[spec.metricName] ?? null,
                higherIsBetter: spec.higherIsBetter,
              }
            : null,
        lastTrained: isReady ? (info?.timestamp ?? null) : null,
        // History fetched in the second pass.
        history: [] as number[],
      } satisfies DashboardRow;
    }),
  );

  // Second pass: fetch MLflow histories for ready rows in parallel.
  // Untrained rows skip the call entirely — saves on round-trips.
  const histories = await Promise.all(
    baseRows.map((row) => {
      const key = `${row.industrySlug}/${row.modelSlug}`;
      const spec = PROBLEM_BY_INDUSTRY_MODEL[key];
      if (row.status !== "ready" || !spec) return Promise.resolve([] as number[]);
      // Experiment name follows the trainer's convention: <problem>.
      // (BaseTrainer._init_mlflow uses the problem name as experiment.)
      return safeGetRunHistory(spec.problem, spec.metricName);
    }),
  );

  return baseRows.map((row, idx) => ({ ...row, history: histories[idx] }));
}
