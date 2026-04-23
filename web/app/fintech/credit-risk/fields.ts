/**
 * Field configuration for the credit-risk form.
 *
 * Kept co-located with the page so A.3–A.8 agents can follow the
 * same pattern per industry (fields.ts sibling of page.tsx). Types
 * line up with CreditRiskInput from lib/schemas.ts.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { CreditRiskInput } from "@/lib/schemas";

export const CREDIT_RISK_FIELDS: FieldConfig<CreditRiskInput>[] = [
  { name: "age", label: "Age", type: "number", min: 18, max: 100 },
  {
    name: "annual_income",
    label: "Annual Income ($)",
    type: "number",
    min: 0,
  },
  {
    name: "credit_score",
    label: "Credit Score",
    type: "number",
    min: 300,
    max: 850,
  },
  {
    name: "num_open_accounts",
    label: "Open Credit Accounts",
    type: "number",
    min: 0,
  },
  {
    name: "payment_history_pct",
    label: "Payment History (%)",
    type: "number",
    min: 0,
    max: 100,
    description: "Percentage of on-time payments over the last 24 months.",
  },
  {
    name: "debt_to_income_ratio",
    label: "Debt-to-Income Ratio",
    type: "slider",
    min: 0,
    max: 2,
    step: 0.01,
  },
  {
    name: "employment_years",
    label: "Years Employed",
    type: "number",
    min: 0,
  },
  {
    name: "loan_amount",
    label: "Loan Amount Requested ($)",
    type: "number",
    min: 0,
  },
];
