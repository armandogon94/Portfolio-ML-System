/**
 * Form fields for the card-attrition model.
 *
 * Labels are the human reading of the dataset's real column names — the `name`
 * values must stay byte-identical to `ChurnInput` in lib/schemas.ts and to
 * `CardholderRequest` in src/serving/schemas.py, or FastAPI returns 422.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { ChurnInput } from "@/lib/schemas";

export const CHURN_FIELDS: FieldConfig<ChurnInput>[] = [
  { name: "Customer_Age", label: "Customer age", type: "number", min: 18, max: 120 },
  { name: "Gender", label: "Gender", type: "select", options: ["M", "F"] },
  { name: "Dependent_count", label: "Dependents", type: "number", min: 0, max: 10 },
  {
    name: "Education_Level",
    label: "Education level",
    type: "select",
    options: [
      "Uneducated",
      "High School",
      "College",
      "Graduate",
      "Post-Graduate",
      "Doctorate",
      "Unknown",
    ],
  },
  {
    name: "Marital_Status",
    label: "Marital status",
    type: "select",
    options: ["Married", "Single", "Divorced", "Unknown"],
  },
  {
    name: "Income_Category",
    label: "Income category",
    type: "select",
    options: [
      "Less than $40K",
      "$40K - $60K",
      "$60K - $80K",
      "$80K - $120K",
      "$120K +",
      "Unknown",
    ],
  },
  {
    name: "Card_Category",
    label: "Card tier",
    type: "select",
    options: ["Blue", "Silver", "Gold", "Platinum"],
  },
  {
    name: "Months_on_book",
    label: "Months on book",
    type: "number",
    min: 0,
    description: "How long this customer has held the card.",
  },
  {
    name: "Total_Relationship_Count",
    label: "Products held",
    type: "number",
    min: 0,
    max: 10,
  },
  {
    name: "Months_Inactive_12_mon",
    label: "Inactive months (last 12)",
    type: "number",
    min: 0,
    max: 12,
    description: "Dormancy is the single strongest attrition signal in this dataset.",
  },
  {
    name: "Contacts_Count_12_mon",
    label: "Service contacts (last 12)",
    type: "number",
    min: 0,
    description: "Many contacts against few products is the classic pre-churn shape.",
  },
  { name: "Credit_Limit", label: "Credit limit ($)", type: "number", min: 1 },
  { name: "Total_Revolving_Bal", label: "Revolving balance ($)", type: "number", min: 0 },
  { name: "Avg_Open_To_Buy", label: "Average open to buy ($)", type: "number", min: 0 },
  {
    name: "Total_Amt_Chng_Q4_Q1",
    label: "Spend change Q4 vs Q1",
    type: "number",
    min: 0,
    step: 0.01,
  },
  { name: "Total_Trans_Amt", label: "Total transaction amount ($)", type: "number", min: 0 },
  { name: "Total_Trans_Ct", label: "Total transaction count", type: "number", min: 0 },
  {
    name: "Total_Ct_Chng_Q4_Q1",
    label: "Count change Q4 vs Q1",
    type: "number",
    min: 0,
    step: 0.01,
  },
  {
    name: "Avg_Utilization_Ratio",
    label: "Average utilisation",
    type: "number",
    min: 0,
    max: 1,
    step: 0.01,
  },
];
