/**
 * Zod input schemas: one per FastAPI endpoint.
 *
 * Every field name is a **real column from the real dataset**, mirroring
 * `src/serving/schemas.py` exactly. A mismatch here is a 422 from FastAPI, so
 * these are the contract, not a convenience layer.
 *
 * They are deliberate subsets: IEEE-CIS has 394 columns and a 394-input form is
 * not a demo. Unsupplied columns arrive at the model as NaN, which LightGBM
 * handles natively as "unknown"; see `src/serving/preprocessing.py`.
 *
 * Defaults are seeded demo values, never real customer data.
 */
import { z } from "zod";

// ─── Payment fraud (IEEE-CIS) ──────────────────────────────────────────────

export const FraudInputSchema = z.object({
  // Seconds from the dataset's reference point, NOT a Unix timestamp.
  TransactionDT: z.number().int().min(0),
  TransactionAmt: z.number().positive(),
  ProductCD: z.enum(["W", "C", "R", "H", "S"]),
  card1: z.number(),
  card2: z.number(),
  card3: z.number(),
  card4: z.enum(["visa", "mastercard", "discover", "american express"]),
  card5: z.number(),
  card6: z.enum(["debit", "credit"]),
  addr1: z.number(),
  addr2: z.number(),
  dist1: z.number(),
  P_emaildomain: z.string().min(1),
  C1: z.number().min(0),
  C13: z.number().min(0),
  C14: z.number().min(0),
  D1: z.number(),
  D15: z.number(),
});
export type FraudInput = z.infer<typeof FraudInputSchema>;

export const FRAUD_DEFAULTS: FraudInput = {
  TransactionDT: 8_640_000,
  TransactionAmt: 149.99,
  ProductCD: "W",
  card1: 13926,
  card2: 361,
  card3: 150,
  card4: "visa",
  card5: 226,
  card6: "debit",
  addr1: 315,
  addr2: 87,
  dist1: 19,
  P_emaildomain: "gmail.com",
  C1: 1,
  C13: 1,
  C14: 1,
  D1: 14,
  D15: 0,
};

// ─── Consumer credit risk (LendingClub) ────────────────────────────────────
// Every field is available at origination. No post-origination field appears
// here. That is the leak configs/credit_risk.yaml's denylist exists to prevent.

export const CreditRiskInputSchema = z.object({
  loan_amnt: z.number().positive(),
  funded_amnt: z.number().positive(),
  term: z.enum([" 36 months", " 60 months"]),
  int_rate: z.number().min(0).max(50),
  installment: z.number().positive(),
  grade: z.enum(["A", "B", "C", "D", "E", "F", "G"]),
  sub_grade: z.string().regex(/^[A-G][1-5]$/, "Format: a grade letter then 1-5, e.g. C1"),
  emp_length: z.string().min(1),
  home_ownership: z.enum(["RENT", "OWN", "MORTGAGE"]),
  annual_inc: z.number().positive(),
  verification_status: z.enum(["Verified", "Source Verified", "Not Verified"]),
  purpose: z.string().min(1),
  addr_state: z.string().length(2),
  dti: z.number().min(0),
  delinq_2yrs: z.number().min(0),
  fico_range_low: z.number().min(300).max(850),
  fico_range_high: z.number().min(300).max(850),
  inq_last_6mths: z.number().min(0),
  open_acc: z.number().min(0),
  pub_rec: z.number().min(0),
  revol_bal: z.number().min(0),
  revol_util: z.number().min(0),
  total_acc: z.number().min(0),
  application_type: z.enum(["Individual", "Joint App"]),
  mort_acc: z.number().min(0),
  pub_rec_bankruptcies: z.number().min(0),
});
export type CreditRiskInput = z.infer<typeof CreditRiskInputSchema>;

export const CREDIT_RISK_DEFAULTS: CreditRiskInput = {
  loan_amnt: 15000,
  funded_amnt: 15000,
  term: " 36 months",
  int_rate: 13.56,
  installment: 509.66,
  grade: "C",
  sub_grade: "C1",
  emp_length: "5 years",
  home_ownership: "MORTGAGE",
  annual_inc: 72000,
  verification_status: "Source Verified",
  purpose: "debt_consolidation",
  addr_state: "CA",
  dti: 18.24,
  delinq_2yrs: 0,
  fico_range_low: 695,
  fico_range_high: 699,
  inq_last_6mths: 1,
  open_acc: 11,
  pub_rec: 0,
  revol_bal: 14300,
  revol_util: 52.4,
  total_acc: 24,
  application_type: "Individual",
  mort_acc: 1,
  pub_rec_bankruptcies: 0,
};

// ─── Card attrition ────────────────────────────────────────────────────────
// The two Naive_Bayes_Classifier_* columns from the published CSV are absent on
// purpose: they are the target laundered through a classifier and are denylisted
// in configs/churn.yaml.

export const ChurnInputSchema = z.object({
  Customer_Age: z.number().min(18).max(120),
  Gender: z.enum(["M", "F"]),
  Dependent_count: z.number().min(0),
  Education_Level: z.enum([
    "Uneducated",
    "High School",
    "College",
    "Graduate",
    "Post-Graduate",
    "Doctorate",
    "Unknown",
  ]),
  Marital_Status: z.enum(["Married", "Single", "Divorced", "Unknown"]),
  Income_Category: z.enum([
    "Less than $40K",
    "$40K - $60K",
    "$60K - $80K",
    "$80K - $120K",
    "$120K +",
    "Unknown",
  ]),
  Card_Category: z.enum(["Blue", "Silver", "Gold", "Platinum"]),
  Months_on_book: z.number().min(0),
  Total_Relationship_Count: z.number().min(0),
  Months_Inactive_12_mon: z.number().min(0).max(12),
  Contacts_Count_12_mon: z.number().min(0),
  Credit_Limit: z.number().positive(),
  Total_Revolving_Bal: z.number().min(0),
  Avg_Open_To_Buy: z.number().min(0),
  Total_Amt_Chng_Q4_Q1: z.number().min(0),
  Total_Trans_Amt: z.number().min(0),
  Total_Trans_Ct: z.number().min(0),
  Total_Ct_Chng_Q4_Q1: z.number().min(0),
  Avg_Utilization_Ratio: z.number().min(0).max(1),
});
export type ChurnInput = z.infer<typeof ChurnInputSchema>;

export const CHURN_DEFAULTS: ChurnInput = {
  Customer_Age: 45,
  Gender: "M",
  Dependent_count: 2,
  Education_Level: "Graduate",
  Marital_Status: "Married",
  Income_Category: "$60K - $80K",
  Card_Category: "Blue",
  Months_on_book: 36,
  Total_Relationship_Count: 4,
  Months_Inactive_12_mon: 2,
  Contacts_Count_12_mon: 3,
  Credit_Limit: 12000,
  Total_Revolving_Bal: 1200,
  Avg_Open_To_Buy: 10800,
  Total_Amt_Chng_Q4_Q1: 0.75,
  Total_Trans_Amt: 4400,
  Total_Trans_Ct: 67,
  Total_Ct_Chng_Q4_Q1: 0.68,
  Avg_Utilization_Ratio: 0.1,
};
