/**
 * Typed FastAPI client.
 *
 * All calls target relative `/api/*` paths which Next.js rewrites
 * (see next.config.mjs) to the FastAPI server (`INTERNAL_API_URL`).
 * That keeps the browser on a single origin (the Next.js app), so
 * there are no CORS preflights and no backend-side middleware to
 * maintain. Response bodies are validated with Zod at the boundary —
 * type-safe even if FastAPI returns an unexpected shape.
 *
 * Industry sections are grouped so Phase A.3–A.8 agents can append to
 * their own industry without touching others.
 */
import { z } from "zod";

import type { CreditRiskInput, DeliveryEtaInput } from "@/lib/schemas";

// ─── Shared infrastructure ─────────────────────────────────────────────────

const API_BASE = "/api";

/** Thrown for any non-2xx HTTP response. Carries status + parsed body. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(`API ${status}`);
    this.name = "ApiError";
  }
}

async function post<T>(
  path: string,
  body: unknown,
  responseSchema: z.ZodSchema<T>,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => null);
    throw new ApiError(res.status, errBody);
  }
  const json = await res.json();
  return responseSchema.parse(json);
}

// ─── Industry: Fintech ─────────────────────────────────────────────────────

export const RecommendationSchema = z.enum(["APPROVE", "REVIEW", "DECLINE"]);
export type Recommendation = z.infer<typeof RecommendationSchema>;

export const CreditRiskPredictionSchema = z.object({
  risk_score: z.number().min(0).max(1),
  recommendation: RecommendationSchema,
  confidence: z.number().min(0).max(1),
  default_probability: z.number().min(0).max(1),
});
export type CreditRiskPrediction = z.infer<typeof CreditRiskPredictionSchema>;

export const ExplanationSchema = z.object({
  feature_importances: z.record(z.string(), z.number()),
  top_features: z.array(
    z.object({
      feature: z.string(),
      importance: z.number(),
    }),
  ),
  explanation_type: z.string(),
});
export type Explanation = z.infer<typeof ExplanationSchema>;

export const predictCreditRisk = (input: CreditRiskInput) =>
  post("/predict/credit-risk", input, CreditRiskPredictionSchema);

export const explainCreditRisk = (input: CreditRiskInput) =>
  post("/explain/credit-risk", input, ExplanationSchema);

// ─── Industry: Real Estate (A.3 appends here) ──────────────────────────────
// ─── Industry: Dental (A.4 appends here) ───────────────────────────────────
// ─── Industry: Healthcare (A.5 appends here) ───────────────────────────────
// ─── Industry: Logistics ───────────────────────────────────────────────────

export const DeliveryEtaPredictionSchema = z.object({
  eta_hours: z.number(),
  confidence_interval: z.tuple([z.number(), z.number()]),
});
export type DeliveryEtaPrediction = z.infer<typeof DeliveryEtaPredictionSchema>;

export const predictDeliveryEta = (input: DeliveryEtaInput) =>
  post("/predict/eta", input, DeliveryEtaPredictionSchema);

export const explainDeliveryEta = (input: DeliveryEtaInput) =>
  post("/explain/eta", input, ExplanationSchema);
// ─── Industry: Legal/Immigration (A.8 appends here) ────────────────────────
