/**
 * Field configuration for the price-prediction form (A.9.5).
 *
 * Sibling of page.tsx per the per-industry pattern. Types line up
 * with PricePredictionInput from lib/schemas.ts. Mirrors the Gradio
 * price-tab UX: square_feet, bedrooms, bathrooms, year_built,
 * lot_size_sqft, garage_spaces, has_pool (binary), neighborhood_tier
 * (1–5 ordinal), and proximity_to_city_center.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { PricePredictionInput } from "@/lib/schemas";

export const PRICE_PREDICTION_FIELDS: FieldConfig<PricePredictionInput>[] = [
  {
    name: "square_feet",
    label: "Square Feet",
    type: "number",
    min: 300,
    max: 10000,
  },
  { name: "bedrooms", label: "Bedrooms", type: "number", min: 0, max: 8 },
  { name: "bathrooms", label: "Bathrooms", type: "number", min: 1, max: 6 },
  {
    name: "year_built",
    label: "Year Built",
    type: "number",
    min: 1900,
    max: 2030,
  },
  {
    name: "lot_size_sqft",
    label: "Lot Size (sqft)",
    type: "number",
    min: 0,
    max: 50000,
  },
  {
    name: "garage_spaces",
    label: "Garage Spaces",
    type: "number",
    min: 0,
    max: 4,
  },
  {
    name: "has_pool",
    label: "Has Pool (1=yes, 0=no)",
    type: "number",
    min: 0,
    max: 1,
  },
  {
    name: "neighborhood_tier",
    label: "Neighborhood Tier (1–5, higher is more desirable)",
    type: "number",
    min: 1,
    max: 5,
  },
  {
    name: "proximity_to_city_center",
    label: "Proximity to City Center (miles)",
    type: "slider",
    min: 0,
    max: 50,
    step: 0.5,
    description: "Closer to downtown is generally a positive price signal.",
  },
];
