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
// ─── Industry: Healthcare ───
// Mirrors PatientVitals in src/serving/api.py.
export const HeartDiseaseInputSchema = z.object({
  age: z.number().int().min(25).max(85),
  sex: z.number().int().min(0).max(1),
  chest_pain_type: z.number().int().min(1).max(4),
  resting_bp: z.number().int().min(80).max(200),
  cholesterol: z.number().int().min(100).max(400),
  max_heart_rate: z.number().int().min(60).max(220),
  exercise_angina: z.number().int().min(0).max(1),
  oldpeak: z.number().min(0).max(6),
});
export type HeartDiseaseInput = z.infer<typeof HeartDiseaseInputSchema>;
export const HEART_DISEASE_DEFAULTS: HeartDiseaseInput = {
  age: 55, sex: 1, chest_pain_type: 3, resting_bp: 130, cholesterol: 240,
  max_heart_rate: 150, exercise_angina: 0, oldpeak: 1.0,
};
// ─── Industry: Logistics (A.7 appends here) ────────────────────────────────
// ─── Industry: Legal/Immigration (A.8 appends here) ────────────────────────
