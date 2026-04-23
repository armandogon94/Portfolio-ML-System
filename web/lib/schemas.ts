/**
 * Zod input schemas — one per FastAPI endpoint.
 *
 * Grouped by industry so Phase A.3–A.8 agents can append to their own
 * industry section without touching others. Each schema mirrors the
 * corresponding Pydantic BaseModel in `src/serving/api.py`. Keep field
 * names and types in sync; a Zod mismatch here = a 422 from FastAPI.
 */
import { z } from "zod";

// ─── Industry: Fintech ─────────────────────────────────────────────────────
// Mirrors LoanApplication in src/serving/api.py.
export const CreditRiskInputSchema = z.object({
  age: z.number().int().min(18).max(100),
  annual_income: z.number().nonnegative(),
  credit_score: z.number().int().min(300).max(850),
  num_open_accounts: z.number().int().nonnegative(),
  payment_history_pct: z.number().min(0).max(100),
  debt_to_income_ratio: z.number().min(0).max(5),
  employment_years: z.number().nonnegative(),
  loan_amount: z.number().nonnegative(),
});
export type CreditRiskInput = z.infer<typeof CreditRiskInputSchema>;

// Default values used for form initialization + the demo tile on the
// credit-risk page. Match the Pydantic defaults so a "submit without
// editing" works out of the box.
export const CREDIT_RISK_DEFAULTS: CreditRiskInput = {
  age: 35,
  annual_income: 65000,
  credit_score: 700,
  num_open_accounts: 3,
  payment_history_pct: 85,
  debt_to_income_ratio: 0.3,
  employment_years: 8,
  loan_amount: 25000,
};

// ─── Industry: Real Estate (A.3 appends here) ──────────────────────────────
// ─── Industry: Dental (A.4 appends here) ───────────────────────────────────
// ─── Industry: Healthcare (A.5 appends here) ───────────────────────────────
// ─── Industry: Logistics ───────────────────────────────────────────────────
// Mirrors DeliveryRequest in src/serving/api.py — keep field names and
// types in sync or FastAPI will 422.
export const DeliveryEtaInputSchema = z.object({
  distance_km: z.number().min(1).max(2000),
  package_weight_kg: z.number().min(0.1).max(50),
  traffic_congestion: z.number().int().min(1).max(5),
  weather_severity: z.number().int().min(0).max(4),
  time_of_day: z.number().int().min(0).max(23),
  day_of_week: z.number().int().min(0).max(6),
  carrier_priority: z.number().int().min(1).max(3),
  origin_destination_tier: z.number().int().min(1).max(4),
});
export type DeliveryEtaInput = z.infer<typeof DeliveryEtaInputSchema>;
export const DELIVERY_ETA_DEFAULTS: DeliveryEtaInput = {
  distance_km: 50,
  package_weight_kg: 2,
  traffic_congestion: 3,
  weather_severity: 1,
  time_of_day: 10,
  day_of_week: 2,
  carrier_priority: 2,
  origin_destination_tier: 2,
};
// ─── Industry: Legal/Immigration (A.8 appends here) ────────────────────────
