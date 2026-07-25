/**
 * Form fields for the payment-fraud model.
 *
 * These are IEEE-CIS's own column names, kept verbatim so a reviewer can look
 * any of them up in the competition's data description. Most are anonymised by
 * Vesta — `card1` is a card identifier whose meaning is not published, `C1`/`C13`
 * are counting features, `D1`/`D15` are day-deltas. The descriptions say so
 * rather than inventing a friendlier meaning.
 *
 * This is a deliberate 18-field subset of 394. Everything not supplied reaches
 * the model as NaN, which LightGBM handles natively as "unknown".
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { FraudInput } from "@/lib/schemas";

export const FRAUD_FIELDS: FieldConfig<FraudInput>[] = [
  {
    name: "TransactionAmt",
    label: "Transaction amount ($)",
    type: "number",
    min: 0.01,
    step: 0.01,
    description: "The cents portion is itself a signal — bots produce round amounts.",
  },
  {
    name: "TransactionDT",
    label: "Transaction time offset (s)",
    type: "number",
    min: 0,
    description: "Seconds from the dataset's reference point. Not a Unix timestamp.",
  },
  {
    name: "ProductCD",
    label: "Product code",
    type: "select",
    options: ["W", "C", "R", "H", "S"],
    description: "Vesta's anonymised product category.",
  },
  {
    name: "card1",
    label: "card1",
    type: "number",
    description: "Anonymised card identifier. Used as a frequency encoding, never raw.",
  },
  { name: "card2", label: "card2", type: "number", description: "Anonymised card attribute." },
  { name: "card3", label: "card3", type: "number", description: "Anonymised card attribute." },
  {
    name: "card4",
    label: "Card network",
    type: "select",
    options: ["visa", "mastercard", "discover", "american express"],
  },
  { name: "card5", label: "card5", type: "number", description: "Anonymised card attribute." },
  { name: "card6", label: "Card type", type: "select", options: ["debit", "credit"] },
  { name: "addr1", label: "addr1", type: "number", description: "Anonymised billing region." },
  { name: "addr2", label: "addr2", type: "number", description: "Anonymised billing country." },
  { name: "dist1", label: "dist1", type: "number", description: "Anonymised distance measure." },
  {
    name: "P_emaildomain",
    label: "Purchaser email domain",
    type: "text",
    description: "e.g. gmail.com. anonymous.com is a distinct, meaningful value.",
  },
  { name: "C1", label: "C1", type: "number", min: 0, description: "Counting feature." },
  { name: "C13", label: "C13", type: "number", min: 0, description: "Counting feature." },
  { name: "C14", label: "C14", type: "number", min: 0, description: "Counting feature." },
  {
    name: "D1",
    label: "D1",
    type: "number",
    description: "Day-delta feature. De-trended against transaction day before use.",
  },
  { name: "D15", label: "D15", type: "number", description: "Day-delta feature." },
];
