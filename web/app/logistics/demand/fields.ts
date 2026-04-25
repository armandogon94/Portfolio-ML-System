/**
 * Field configuration for the demand-forecast form (A.9.6).
 *
 * Single-field form: a product-category select. The 5 options come
 * from PRODUCT_CATEGORIES in src/data/generate_timeseries.py — same
 * list the LSTM's per-product scalers were trained against, so the
 * server-side `if product not in scalers` guard never fires.
 */
import type { FieldConfig } from "@/components/ModelForm";
import type { DemandRequest } from "@/lib/schemas";
import { DEMAND_PRODUCT_CATEGORIES } from "@/lib/schemas";

export const DEMAND_REQUEST_FIELDS: FieldConfig<DemandRequest>[] = [
  {
    name: "product",
    label: "Product Category",
    type: "select",
    options: [...DEMAND_PRODUCT_CATEGORIES],
    description: "Select a product category to forecast the next 7 days of demand.",
  },
];
