/**
 * Typed FastAPI client. Six endpoints, one shared error type.
 *
 * All calls target relative `/api/*` paths which Next.js rewrites (see
 * `next.config.mjs`) to the FastAPI server at `INTERNAL_API_URL`. That keeps the
 * browser on a single origin, so there are no CORS preflights and no backend
 * middleware to maintain. Response bodies are validated with Zod at the boundary,
 * so a backend shape change surfaces here rather than as `undefined` in a chart.
 *
 * The 503 case is first-class. A fresh clone has no trained checkpoints, and the
 * API says so explicitly with the command that fixes it. The UI surfaces that
 * message verbatim instead of rendering a generic failure.
 */
import { z } from "zod";

import type { ChurnInput, CreditRiskInput, FraudInput } from "@/lib/schemas";

const API_BASE = "/api";

/** Thrown for any non-2xx response. Carries the status and the parsed body. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(`API ${status}`);
    this.name = "ApiError";
  }
}

/** True when the failure is "this model has not been trained yet". */
export function isUntrainedModelError(error: unknown): error is ApiError {
  return error instanceof ApiError && error.status === 503;
}

/** Human-readable message for any error thrown by this module. */
export function apiErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const body = error.body as { detail?: unknown } | null;
    if (body && typeof body.detail === "string") return body.detail;
    return `API ${error.status}`;
  }
  return error instanceof Error ? error.message : String(error);
}

async function post<T>(path: string, body: unknown, schema: z.ZodSchema<T>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => null);
    throw new ApiError(res.status, errorBody);
  }
  return schema.parse(await res.json());
}

// ─── Shared response shapes ────────────────────────────────────────────────

export const ExplanationSchema = z.object({
  feature_importances: z.record(z.string(), z.number()),
  top_features: z.array(z.object({ feature: z.string(), importance: z.number() })),
  explanation_type: z.string(),
});
export type Explanation = z.infer<typeof ExplanationSchema>;

/** Provenance every prediction carries: which commit trained the model, on what. */
const ProvenanceFields = {
  model_version: z.string(),
  trained_on: z.string(),
};

// ─── Payment fraud ─────────────────────────────────────────────────────────

export const FraudRiskBandSchema = z.enum(["LOW", "MEDIUM", "HIGH", "CRITICAL"]);
export type FraudRiskBand = z.infer<typeof FraudRiskBandSchema>;

export const FraudActionSchema = z.enum([
  "AUTO_APPROVE",
  "MONITOR",
  "MANUAL_REVIEW",
  "DECLINE",
]);
export type FraudAction = z.infer<typeof FraudActionSchema>;

export const FraudPredictionSchema = z.object({
  fraud_probability: z.number().min(0).max(1),
  risk_band: FraudRiskBandSchema,
  recommended_action: FraudActionSchema,
  ...ProvenanceFields,
});
export type FraudPrediction = z.infer<typeof FraudPredictionSchema>;

export const predictFraud = (input: FraudInput) =>
  post("/predict/fraud", input, FraudPredictionSchema);

export const explainFraud = (input: FraudInput) =>
  post("/explain/fraud", input, ExplanationSchema);

// ─── Consumer credit risk ──────────────────────────────────────────────────

export const CreditDecisionSchema = z.enum(["APPROVE", "REVIEW", "DECLINE"]);
export type CreditDecision = z.infer<typeof CreditDecisionSchema>;

export const CreditRiskPredictionSchema = z.object({
  default_probability: z.number().min(0).max(1),
  decision: CreditDecisionSchema,
  // The API states in-band that these thresholds are illustrative, not policy.
  threshold_basis: z.string(),
  ...ProvenanceFields,
});
export type CreditRiskPrediction = z.infer<typeof CreditRiskPredictionSchema>;

export const predictCreditRisk = (input: CreditRiskInput) =>
  post("/predict/credit-risk", input, CreditRiskPredictionSchema);

export const explainCreditRisk = (input: CreditRiskInput) =>
  post("/explain/credit-risk", input, ExplanationSchema);

// ─── Card attrition ────────────────────────────────────────────────────────

export const RetentionActionSchema = z.enum([
  "NO_ACTION",
  "PROACTIVE_CHECKIN",
  "URGENT_OUTREACH",
]);
export type RetentionAction = z.infer<typeof RetentionActionSchema>;

export const ChurnPredictionSchema = z.object({
  attrition_probability: z.number().min(0).max(1),
  retention_action: RetentionActionSchema,
  // n = 10,127 and the dataset is easy. The API ships that caveat with the score.
  caveat: z.string(),
  ...ProvenanceFields,
});
export type ChurnPrediction = z.infer<typeof ChurnPredictionSchema>;

export const predictChurn = (input: ChurnInput) =>
  post("/predict/churn", input, ChurnPredictionSchema);

export const explainChurn = (input: ChurnInput) =>
  post("/explain/churn", input, ExplanationSchema);
