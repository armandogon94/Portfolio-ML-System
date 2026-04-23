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
// Mirrors RentalListing in src/serving/api.py.
export const RentalPriceInputSchema = z.object({
  bedrooms: z.number().int().min(0).max(6),
  bathrooms: z.number().int().min(1).max(4),
  square_feet: z.number().int().min(300).max(5000),
  property_type: z.number().int().min(1).max(4),
  location_tier: z.number().int().min(1).max(5),
  distance_to_downtown_km: z.number().min(0).max(50),
  amenity_score: z.number().int().min(0).max(10),
  peer_nightly_rate: z.number().nonnegative(),
});
export type RentalPriceInput = z.infer<typeof RentalPriceInputSchema>;

export const RENTAL_PRICE_DEFAULTS: RentalPriceInput = {
  bedrooms: 2,
  bathrooms: 1,
  square_feet: 900,
  property_type: 2,
  location_tier: 3,
  distance_to_downtown_km: 5,
  amenity_score: 6,
  peer_nightly_rate: 150,
};

// ─── Industry: Dental (A.4 appends here) ───────────────────────────────────
// Mirrors DentalAppointment in src/serving/api.py.
export const DentalNoShowInputSchema = z.object({
  age: z.number().int().min(18).max(90),
  prior_no_shows: z.number().int().min(0).max(10),
  days_until_appointment: z.number().int().min(0).max(60),
  appointment_hour: z.number().int().min(8).max(18),
  distance_km: z.number().min(0).max(50),
  insurance_type: z.number().int().min(1).max(4),
  procedure_complexity: z.number().int().min(1).max(5),
  prior_appointments: z.number().int().min(0).max(20),
});
export type DentalNoShowInput = z.infer<typeof DentalNoShowInputSchema>;

export const DENTAL_NOSHOW_DEFAULTS: DentalNoShowInput = {
  age: 35,
  prior_no_shows: 1,
  days_until_appointment: 14,
  appointment_hour: 10,
  distance_km: 8,
  insurance_type: 1,
  procedure_complexity: 2,
  prior_appointments: 5,
};

// ─── Industry: Healthcare (A.5 appends here) ───────────────────────────────
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
