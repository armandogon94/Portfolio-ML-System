/**
 * Field configuration for the H-1B approval form.
 *
 * Follows the credit-risk fields.ts shape so ModelForm can render the
 * whole form generically. Tier fields (employer size, country, etc.)
 * are plain numbers with min/max — a proper dropdown lands later.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { H1BApprovalInput } from "@/lib/schemas";

export const H1B_APPROVAL_FIELDS: FieldConfig<H1BApprovalInput>[] = [
  {
    name: "prevailing_wage",
    label: "Prevailing Wage ($)",
    type: "number",
    min: 40000,
    max: 300000,
    description: "DOL-certified wage for the SOC code + area.",
  },
  {
    name: "soc_code_level",
    label: "SOC Code Level (1–4)",
    type: "number",
    min: 1,
    max: 4,
    description: "4 = senior specialty occupation.",
  },
  {
    name: "employer_size_tier",
    label: "Employer Size Tier (1–5)",
    type: "number",
    min: 1,
    max: 5,
    description: "1 = <50 employees, 5 = Fortune 500.",
  },
  {
    name: "job_level",
    label: "Job Level (1–4)",
    type: "number",
    min: 1,
    max: 4,
  },
  {
    name: "education_level",
    label: "Education Level (1–5)",
    type: "number",
    min: 1,
    max: 5,
    description: "1 = Bachelor's, 3 = PhD, 5 = PhD + post-doc.",
  },
  {
    name: "experience_years",
    label: "Years of Experience",
    type: "number",
    min: 0,
    max: 30,
  },
  {
    name: "country_of_citizenship_tier",
    label: "Country Tier (1–5)",
    type: "number",
    min: 1,
    max: 5,
    description: "5 = high-volume backlog country.",
  },
  {
    name: "employer_prior_approval_rate",
    label: "Employer Prior Approval Rate",
    type: "slider",
    min: 0,
    max: 1,
    step: 0.01,
    description: "Historical approval rate for this sponsor's filings.",
  },
];
