/**
 * Dashboard data shape — typed once, consumed by DashboardTable,
 * IndustrySummaryTile, and the /dashboard page.
 *
 * The fetch + join logic that produces DashboardRow[] (industries.ts
 * × `/models` × MLflow history) ships in A.9.9. A.9.8 only defines
 * the type so the consumer components can be tested standalone with
 * a hand-crafted fixture.
 */

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
