/**
 * Field configuration for the delivery-ETA form.
 *
 * Same pattern as fintech/credit-risk/fields.ts — one FieldConfig per
 * DeliveryEtaInput property. Sliders on the ordinal scales give the
 * demo a more tactile feel; raw numeric inputs for distance / weight.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { DeliveryEtaInput } from "@/lib/schemas";

export const DELIVERY_ETA_FIELDS: FieldConfig<DeliveryEtaInput>[] = [
  {
    name: "distance_km",
    label: "Distance (km)",
    type: "number",
    min: 1,
    max: 2000,
  },
  {
    name: "package_weight_kg",
    label: "Package Weight (kg)",
    type: "number",
    min: 0.1,
    max: 50,
    step: 0.1,
  },
  {
    name: "traffic_congestion",
    label: "Traffic Congestion (1 = light, 5 = peak)",
    type: "slider",
    min: 1,
    max: 5,
    step: 1,
  },
  {
    name: "weather_severity",
    label: "Weather Severity (0 = clear, 4 = storm)",
    type: "slider",
    min: 0,
    max: 4,
    step: 1,
  },
  {
    name: "time_of_day",
    label: "Pickup Hour (0–23)",
    type: "number",
    min: 0,
    max: 23,
    step: 1,
  },
  {
    name: "day_of_week",
    label: "Day of Week (Mon = 0, Sun = 6)",
    type: "number",
    min: 0,
    max: 6,
    step: 1,
  },
  {
    name: "carrier_priority",
    label: "Carrier Priority (1 = standard, 3 = express)",
    type: "slider",
    min: 1,
    max: 3,
    step: 1,
  },
  {
    name: "origin_destination_tier",
    label: "Origin↔Destination Tier (1 = hub↔hub, 4 = remote↔remote)",
    type: "slider",
    min: 1,
    max: 4,
    step: 1,
  },
];
