/**
 * Field configuration for the customer-churn form.
 *
 * Mirrors CustomerChurnInput in lib/schemas.ts. Kept co-located with the
 * page so the agent-per-industry pattern established by credit-risk
 * carries over cleanly.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { CustomerChurnInput } from "@/lib/schemas";

export const CUSTOMER_CHURN_FIELDS: FieldConfig<CustomerChurnInput>[] = [
  {
    name: "tenure_months",
    label: "Tenure (months)",
    type: "number",
    min: 0,
    max: 120,
    description: "Months the customer has banked with us.",
  },
  {
    name: "balance",
    label: "Account Balance ($)",
    type: "number",
    min: 0,
  },
  {
    name: "num_products",
    label: "Number of Products",
    type: "number",
    min: 1,
    max: 6,
    description: "Savings, credit card, mortgage, investment, etc.",
  },
  {
    name: "has_credit_card",
    label: "Has Credit Card",
    type: "number",
    min: 0,
    max: 1,
    description: "1 = yes, 0 = no.",
  },
  {
    name: "is_active_member",
    label: "Active Member",
    type: "number",
    min: 0,
    max: 1,
    description: "1 = active in last 90 days.",
  },
  {
    name: "estimated_salary",
    label: "Estimated Salary ($)",
    type: "number",
    min: 0,
  },
  {
    name: "age",
    label: "Age",
    type: "number",
    min: 18,
    max: 92,
  },
  {
    name: "geography_tier",
    label: "Geography Tier",
    type: "number",
    min: 1,
    max: 3,
    description: "1 = core, 2 = established, 3 = expanding market.",
  },
];
