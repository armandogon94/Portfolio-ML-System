/**
 * Field configuration for the fraud-detection form (A.9.4).
 *
 * Sibling of page.tsx per the per-industry pattern. Types line up
 * with FraudInput from lib/schemas.ts. The merchant_category field
 * is the first categorical select rendered by ModelForm — its
 * options come from the same list used to generate synthetic fraud
 * data in src/data/generate_fraud.py so the model sees them
 * verbatim.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { FraudInput } from "@/lib/schemas";
import { FRAUD_MERCHANT_CATEGORIES } from "@/lib/schemas";

export const FRAUD_FIELDS: FieldConfig<FraudInput>[] = [
  {
    name: "transaction_amount",
    label: "Transaction Amount ($)",
    type: "number",
    min: 0,
    step: 0.01,
  },
  {
    name: "merchant_category",
    label: "Merchant Category",
    type: "select",
    options: [...FRAUD_MERCHANT_CATEGORIES],
    description: "Categorical input fed into the autoencoder via label encoding.",
  },
  {
    name: "hour_of_day",
    label: "Hour of Day (0–23)",
    type: "number",
    min: 0,
    max: 23,
  },
  {
    name: "day_of_week",
    label: "Day of Week (0=Mon, 6=Sun)",
    type: "number",
    min: 0,
    max: 6,
  },
  {
    name: "distance_from_home",
    label: "Distance From Home (km)",
    type: "number",
    min: 0,
  },
  {
    name: "is_online",
    label: "Online Transaction (1=yes, 0=no)",
    type: "number",
    min: 0,
    max: 1,
  },
  {
    name: "card_age_days",
    label: "Card Age (days)",
    type: "number",
    min: 0,
  },
  {
    name: "num_transactions_last_hour",
    label: "Transactions in Last Hour",
    type: "number",
    min: 0,
  },
  {
    name: "amount_vs_avg_ratio",
    label: "Amount / Avg-Transaction Ratio",
    type: "slider",
    min: 0,
    max: 10,
    step: 0.1,
    description: "Spike vs the cardholder's typical spend — a strong fraud signal.",
  },
];
