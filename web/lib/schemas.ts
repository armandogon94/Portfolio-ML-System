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
// ─── Industry: Logistics (A.7 appends here) ────────────────────────────────
// ─── Industry: Legal/Immigration ───────────────────────────────────────────
// Mirrors H1BApplication in src/serving/api.py.
export const H1BApprovalInputSchema = z.object({
  prevailing_wage: z.number().min(40000).max(300000),
  soc_code_level: z.number().int().min(1).max(4),
  employer_size_tier: z.number().int().min(1).max(5),
  job_level: z.number().int().min(1).max(4),
  education_level: z.number().int().min(1).max(5),
  experience_years: z.number().int().min(0).max(30),
  country_of_citizenship_tier: z.number().int().min(1).max(5),
  employer_prior_approval_rate: z.number().min(0).max(1),
});
export type H1BApprovalInput = z.infer<typeof H1BApprovalInputSchema>;
export const H1B_APPROVAL_DEFAULTS: H1BApprovalInput = {
  prevailing_wage: 120000,
  soc_code_level: 3,
  employer_size_tier: 3,
  job_level: 2,
  education_level: 2,
  experience_years: 5,
  country_of_citizenship_tier: 2,
  employer_prior_approval_rate: 0.75,
};
