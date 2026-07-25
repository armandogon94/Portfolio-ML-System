/**
 * Form fields for the consumer-credit-risk model.
 *
 * Every field is LendingClub data available AT ORIGINATION. There is deliberately
 * no input for `recoveries`, `total_pymnt`, `last_pymnt_amnt` or any other
 * post-origination column: those are denylisted in configs/credit_risk.yaml, and
 * a model that sees them reports ~0.99 ROC-AUC while being worthless.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { CreditRiskInput } from "@/lib/schemas";

export const CREDIT_RISK_FIELDS: FieldConfig<CreditRiskInput>[] = [
  { name: "loan_amnt", label: "Loan amount ($)", type: "number", min: 500 },
  { name: "funded_amnt", label: "Funded amount ($)", type: "number", min: 500 },
  {
    name: "term",
    label: "Term",
    type: "select",
    options: [" 36 months", " 60 months"],
    description: "The leading space is in the source data, not a typo.",
  },
  { name: "int_rate", label: "Interest rate (%)", type: "number", min: 0, step: 0.01 },
  { name: "installment", label: "Monthly installment ($)", type: "number", min: 1, step: 0.01 },
  {
    name: "grade",
    label: "Grade",
    type: "select",
    options: ["A", "B", "C", "D", "E", "F", "G"],
    description: "Ordinal, not nominal — A is best.",
  },
  { name: "sub_grade", label: "Sub-grade", type: "text", description: "Grade letter + 1-5, e.g. C1." },
  {
    name: "emp_length",
    label: "Employment length",
    type: "select",
    options: [
      "< 1 year",
      "1 year",
      "2 years",
      "3 years",
      "4 years",
      "5 years",
      "6 years",
      "7 years",
      "8 years",
      "9 years",
      "10+ years",
    ],
  },
  {
    name: "home_ownership",
    label: "Home ownership",
    type: "select",
    options: ["RENT", "OWN", "MORTGAGE"],
  },
  { name: "annual_inc", label: "Annual income ($)", type: "number", min: 1 },
  {
    name: "verification_status",
    label: "Income verification",
    type: "select",
    options: ["Verified", "Source Verified", "Not Verified"],
  },
  {
    name: "purpose",
    label: "Loan purpose",
    type: "select",
    options: [
      "debt_consolidation",
      "credit_card",
      "home_improvement",
      "major_purchase",
      "medical",
      "small_business",
      "car",
      "other",
    ],
  },
  { name: "addr_state", label: "State", type: "text", description: "Two-letter code, e.g. CA." },
  {
    name: "dti",
    label: "Debt-to-income",
    type: "number",
    min: 0,
    step: 0.01,
    description: "Monthly debt payments over monthly income, as reported.",
  },
  { name: "delinq_2yrs", label: "Delinquencies (2 yrs)", type: "number", min: 0 },
  { name: "fico_range_low", label: "FICO range low", type: "number", min: 300, max: 850 },
  { name: "fico_range_high", label: "FICO range high", type: "number", min: 300, max: 850 },
  { name: "inq_last_6mths", label: "Credit inquiries (6 mo)", type: "number", min: 0 },
  { name: "open_acc", label: "Open credit lines", type: "number", min: 0 },
  { name: "pub_rec", label: "Derogatory public records", type: "number", min: 0 },
  { name: "revol_bal", label: "Revolving balance ($)", type: "number", min: 0 },
  { name: "revol_util", label: "Revolving utilisation (%)", type: "number", min: 0, step: 0.1 },
  { name: "total_acc", label: "Total credit lines", type: "number", min: 0 },
  {
    name: "application_type",
    label: "Application type",
    type: "select",
    options: ["Individual", "Joint App"],
  },
  { name: "mort_acc", label: "Mortgage accounts", type: "number", min: 0 },
  { name: "pub_rec_bankruptcies", label: "Bankruptcies", type: "number", min: 0 },
];
