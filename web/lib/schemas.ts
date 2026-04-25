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

// Mirrors BankCustomer in src/serving/api.py (A.6).
export const CustomerChurnInputSchema = z.object({
  tenure_months: z.number().int().min(0).max(120),
  balance: z.number().nonnegative(),
  num_products: z.number().int().min(1).max(6),
  has_credit_card: z.number().int().min(0).max(1),
  is_active_member: z.number().int().min(0).max(1),
  estimated_salary: z.number().nonnegative(),
  age: z.number().int().min(18).max(92),
  geography_tier: z.number().int().min(1).max(3),
});
export type CustomerChurnInput = z.infer<typeof CustomerChurnInputSchema>;
export const CUSTOMER_CHURN_DEFAULTS: CustomerChurnInput = {
  tenure_months: 48,
  balance: 50000,
  num_products: 2,
  has_credit_card: 1,
  is_active_member: 1,
  estimated_salary: 75000,
  age: 40,
  geography_tier: 2,
};

// Mirrors Transaction in src/serving/api.py (A.9.4 — Gradio fraud-tab port).
// merchant_category is a string keyed off MERCHANT_CATEGORIES in
// src/data/generate_fraud.py; the FastAPI Pydantic accepts the raw string.
export const FRAUD_MERCHANT_CATEGORIES = [
  "grocery",
  "restaurant",
  "gas_station",
  "online_retail",
  "electronics",
  "clothing",
  "travel",
  "entertainment",
  "healthcare",
  "utilities",
  "education",
  "home_improvement",
  "automotive",
  "subscription",
  "atm_withdrawal",
] as const;
export const FraudInputSchema = z.object({
  transaction_amount: z.number().nonnegative(),
  merchant_category: z.enum(FRAUD_MERCHANT_CATEGORIES),
  hour_of_day: z.number().int().min(0).max(23),
  day_of_week: z.number().int().min(0).max(6),
  distance_from_home: z.number().nonnegative(),
  is_online: z.number().int().min(0).max(1),
  card_age_days: z.number().int().min(0),
  num_transactions_last_hour: z.number().int().min(0),
  amount_vs_avg_ratio: z.number().nonnegative(),
});
export type FraudInput = z.infer<typeof FraudInputSchema>;
export const FRAUD_DEFAULTS: FraudInput = {
  transaction_amount: 150,
  merchant_category: "online_retail",
  hour_of_day: 14,
  day_of_week: 2,
  distance_from_home: 15,
  is_online: 1,
  card_age_days: 365,
  num_transactions_last_hour: 1,
  amount_vs_avg_ratio: 3,
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

// Mirrors Property in src/serving/api.py (A.9.5 — Gradio price-tab port).
export const PricePredictionInputSchema = z.object({
  square_feet: z.number().int().min(300).max(10000),
  bedrooms: z.number().int().min(0).max(8),
  bathrooms: z.number().int().min(1).max(6),
  year_built: z.number().int().min(1900).max(2030),
  lot_size_sqft: z.number().int().min(0).max(50000),
  garage_spaces: z.number().int().min(0).max(4),
  has_pool: z.number().int().min(0).max(1),
  neighborhood_tier: z.number().int().min(1).max(5),
  proximity_to_city_center: z.number().min(0).max(50),
});
export type PricePredictionInput = z.infer<typeof PricePredictionInputSchema>;
export const PRICE_PREDICTION_DEFAULTS: PricePredictionInput = {
  square_feet: 1800,
  bedrooms: 3,
  bathrooms: 2,
  year_built: 2000,
  lot_size_sqft: 8000,
  garage_spaces: 2,
  has_pool: 0,
  neighborhood_tier: 3,
  proximity_to_city_center: 10,
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
  age: 55,
  sex: 1,
  chest_pain_type: 3,
  resting_bp: 130,
  cholesterol: 240,
  max_heart_rate: 150,
  exercise_angina: 0,
  oldpeak: 1.0,
};

// ─── Industry: Logistics (A.7 appends here) ────────────────────────────────
// Mirrors DeliveryRequest in src/serving/api.py.
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
