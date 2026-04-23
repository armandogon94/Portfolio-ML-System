/**
 * Field configuration for the dental patient no-show form.
 *
 * Schema bounds (min/max) mirror DentalNoShowInputSchema in lib/schemas.ts
 * so a valid form cannot produce a 422 from FastAPI.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { DentalNoShowInput } from "@/lib/schemas";

export const DENTAL_NOSHOW_FIELDS: FieldConfig<DentalNoShowInput>[] = [
  { name: "age", label: "Patient Age", type: "number", min: 18, max: 90 },
  {
    name: "prior_no_shows",
    label: "Prior No-Shows",
    type: "number",
    min: 0,
    max: 10,
    description: "How many appointments the patient has previously missed.",
  },
  {
    name: "days_until_appointment",
    label: "Days Until Appointment",
    type: "number",
    min: 0,
    max: 60,
  },
  {
    name: "appointment_hour",
    label: "Appointment Hour (24h)",
    type: "number",
    min: 8,
    max: 18,
    description: "Clinic operating hours are 8:00-18:00.",
  },
  {
    name: "distance_km",
    label: "Distance to Clinic (km)",
    type: "slider",
    min: 0,
    max: 50,
    step: 0.5,
  },
  {
    name: "insurance_type",
    label: "Insurance Type",
    type: "number",
    min: 1,
    max: 4,
    description: "1 = private, 2 = medicaid, 3 = uninsured, 4 = medicare.",
  },
  {
    name: "procedure_complexity",
    label: "Procedure Complexity",
    type: "number",
    min: 1,
    max: 5,
    description: "1 = routine cleaning, 5 = major surgery.",
  },
  {
    name: "prior_appointments",
    label: "Prior Appointments (total)",
    type: "number",
    min: 0,
    max: 20,
  },
];
