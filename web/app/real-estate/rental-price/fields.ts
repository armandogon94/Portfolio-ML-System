/**
 * Field configuration for the rental-price form (Real Estate / A.3).
 *
 * Types line up with RentalPriceInput from lib/schemas.ts. All fields are
 * number inputs except location_tier which is a 1-5 slider — giving the user
 * a clear "desirability dial" rather than asking them to type a magic number.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { RentalPriceInput } from "@/lib/schemas";

export const RENTAL_PRICE_FIELDS: FieldConfig<RentalPriceInput>[] = [
  { name: "bedrooms", label: "Bedrooms", type: "number", min: 0, max: 6, step: 1 },
  { name: "bathrooms", label: "Bathrooms", type: "number", min: 1, max: 4, step: 1 },
  {
    name: "square_feet",
    label: "Square Feet",
    type: "number",
    min: 300,
    max: 5000,
    step: 10,
  },
  {
    name: "property_type",
    label: "Property Type",
    type: "number",
    min: 1,
    max: 4,
    step: 1,
    description: "1 = studio, 2 = apartment, 3 = house, 4 = condo.",
  },
  {
    name: "location_tier",
    label: "Location Tier",
    type: "slider",
    min: 1,
    max: 5,
    step: 1,
    description: "1 = outskirts, 5 = most desirable area.",
  },
  {
    name: "distance_to_downtown_km",
    label: "Distance to Downtown (km)",
    type: "number",
    min: 0,
    max: 50,
    step: 0.1,
  },
  {
    name: "amenity_score",
    label: "Amenity Score",
    type: "number",
    min: 0,
    max: 10,
    step: 1,
    description: "0-10 scale — pool, gym, parking, laundry, etc.",
  },
  {
    name: "peer_nightly_rate",
    label: "Peer Nightly Rate ($)",
    type: "number",
    min: 0,
    step: 1,
    description: "Median nightly rate of nearby comparable listings.",
  },
];
