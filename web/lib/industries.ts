/**
 * Industry registry — the single source of truth for landing tiles,
 * sidebar nav, and dashboard listings.
 *
 * Each entry describes one industry vertical and the ML models Phase A.3+
 * will ship underneath. As industry slices land (A.3–A.8), the agents will
 * append model entries to their industry's `models` list so the stub
 * industry-index pages can enumerate what's available.
 *
 * Icons are lucide-react components. Keep them consistent with the tile
 * art; swap in a better icon only if the current one misleads the viewer.
 * All taglines are marked TODO(copy) — placeholders that Armando can edit
 * without touching any component code.
 */
import {
  DollarSign,
  Gavel,
  HeartPulse,
  Home,
  Smile,
  Truck,
  type LucideIcon,
} from "lucide-react";

export type IndustryModel = {
  /** Route slug relative to the industry index (e.g., "credit-risk"). */
  slug: string;
  /** Display name on the industry page. */
  title: string;
  /** One-line description of what this model does. */
  description: string;
  /** Whether the model page has been built yet. */
  ready: boolean;
};

export type Industry = {
  /** URL slug; also the app/ directory name. */
  slug: string;
  /** Display name. */
  title: string;
  /** TODO(copy) — short tagline shown on the landing tile. */
  tagline: string;
  /** Route href (always "/" + slug). */
  href: string;
  /** lucide-react icon component. */
  icon: LucideIcon;
  /** Total models planned for this industry (incl. not-yet-shipped). */
  modelCount: number;
  /** Model list — A.3–A.8 flip `ready: true` and point slug at their page. */
  models: IndustryModel[];
};

export const INDUSTRIES: Industry[] = [
  {
    slug: "real-estate",
    title: "Real Estate",
    tagline: "Price prediction, rental estimates, days-on-market", // TODO(copy)
    href: "/real-estate",
    icon: Home,
    modelCount: 3,
    models: [
      { slug: "price", title: "Price Prediction", description: "Synthetic home-price regressor (LightGBM).", ready: false },
      { slug: "rental-price", title: "Rental Price", description: "Airbnb-trained nightly rate estimator.", ready: false },
      { slug: "days-on-market", title: "Days on Market", description: "How long a listing will take to sell.", ready: false },
    ],
  },
  {
    slug: "dental",
    title: "Dental Clinics",
    tagline: "Cavity detection from X-rays, no-show prediction, treatment plans", // TODO(copy)
    href: "/dental",
    icon: Smile,
    modelCount: 3,
    models: [
      { slug: "caries", title: "Cavity Detection", description: "EfficientNet-B0 X-ray classifier (MPS).", ready: false },
      { slug: "no-show", title: "Patient No-Show", description: "XGBoost risk of missed appointment.", ready: false },
      { slug: "treatment-plan", title: "Treatment Plan", description: "Recommender for next-procedure.", ready: false },
    ],
  },
  {
    slug: "healthcare",
    title: "Healthcare",
    tagline: "Readmission risk, heart disease, diabetes, length-of-stay", // TODO(copy)
    href: "/healthcare",
    icon: HeartPulse,
    modelCount: 4,
    models: [
      { slug: "readmission", title: "30-Day Readmission", description: "UCI Diabetes 130 risk model.", ready: false },
      { slug: "heart-disease", title: "Heart Disease Risk", description: "Cleveland dataset LightGBM.", ready: false },
      { slug: "diabetes-onset", title: "Diabetes Onset", description: "PIMA-style XGBoost classifier.", ready: false },
      { slug: "length-of-stay", title: "Length of Stay", description: "LSTM over synthetic vitals windows.", ready: false },
    ],
  },
  {
    slug: "fintech",
    title: "Fintech",
    tagline: "Credit risk, fraud detection, loan approval, customer churn", // TODO(copy)
    href: "/fintech",
    icon: DollarSign,
    modelCount: 4,
    models: [
      { slug: "credit-risk", title: "Credit Risk Scoring", description: "XGBoost loan-default classifier.", ready: true },
      { slug: "fraud", title: "Fraud Detection", description: "PyTorch autoencoder + isolation forest.", ready: false },
      { slug: "loan-approval", title: "Loan Approval", description: "LightGBM eligibility classifier.", ready: false },
      { slug: "churn", title: "Customer Churn", description: "Bank churn XGBoost classifier.", ready: false },
    ],
  },
  {
    slug: "logistics",
    title: "Logistics",
    tagline: "Demand forecasting, delivery ETA, shipment damage risk", // TODO(copy)
    href: "/logistics",
    icon: Truck,
    modelCount: 3,
    models: [
      { slug: "demand", title: "Demand Forecasting", description: "PyTorch LSTM 7-day forecast.", ready: false },
      { slug: "eta", title: "Delivery ETA", description: "Feed-forward NN on Amazon deliveries.", ready: false },
      { slug: "damage-risk", title: "Shipment Damage Risk", description: "XGBoost risk classifier.", ready: false },
    ],
  },
  {
    slug: "legal",
    title: "Legal / Immigration",
    tagline: "H-1B approval, case duration, legal document classification", // TODO(copy)
    href: "/legal",
    icon: Gavel,
    modelCount: 3,
    models: [
      { slug: "h1b-approval", title: "H-1B Approval", description: "XGBoost on LCA disclosure data.", ready: false },
      { slug: "case-duration", title: "Case Duration", description: "LightGBM regressor on USCIS data.", ready: false },
      { slug: "doc-classification", title: "Legal Doc Classifier", description: "DistilBERT fine-tune (MPS).", ready: false },
    ],
  },
];

/** Lookup by slug — used by industry-index pages to enumerate their models. */
export function getIndustry(slug: string): Industry | undefined {
  return INDUSTRIES.find((i) => i.slug === slug);
}
