/**
 * Field configuration for the heart-disease form.
 *
 * Mirrors PatientVitals in src/serving/api.py and HeartDiseaseInput in
 * lib/schemas.ts. All 8 Cleveland-style vitals are numeric; oldpeak is
 * a slider (small continuous range), everything else is a number input.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { HeartDiseaseInput } from "@/lib/schemas";

export const HEART_DISEASE_FIELDS: FieldConfig<HeartDiseaseInput>[] = [
  { name: "age", label: "Age", type: "number", min: 25, max: 85 },
  {
    name: "sex",
    label: "Sex (0 = female, 1 = male)",
    type: "number",
    min: 0,
    max: 1,
  },
  {
    name: "chest_pain_type",
    label: "Chest Pain Type (1–4, 4 = asymptomatic)",
    type: "number",
    min: 1,
    max: 4,
    description:
      "1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic (highest risk).",
  },
  {
    name: "resting_bp",
    label: "Resting Blood Pressure (mmHg)",
    type: "number",
    min: 80,
    max: 200,
  },
  {
    name: "cholesterol",
    label: "Cholesterol (mg/dL)",
    type: "number",
    min: 100,
    max: 400,
  },
  {
    name: "max_heart_rate",
    label: "Max Heart Rate (bpm)",
    type: "number",
    min: 60,
    max: 220,
  },
  {
    name: "exercise_angina",
    label: "Exercise-Induced Angina (0 = no, 1 = yes)",
    type: "number",
    min: 0,
    max: 1,
  },
  {
    name: "oldpeak",
    label: "Oldpeak (ST depression)",
    type: "slider",
    min: 0,
    max: 6,
    step: 0.1,
  },
];
