/**
 * Industry registry: the single source of truth for landing tiles, nav and the
 * dashboard.
 *
 * One industry. Three models. Every entry is `ready: true` and every tagline is
 * real copy.
 *
 * The previous version listed six industries and twenty-one models, ten of which
 * were permanent placeholders carrying unfinished taglines. A dashboard
 * whose rows never resolve reads as abandonment, and six unrelated verticals read
 * as generated breadth. See `docs/adr/0004-narrow-to-fintech.md`.
 */
import { DollarSign, type LucideIcon } from "lucide-react";

export type IndustryModel = {
  /** Route slug relative to the industry index (e.g. "credit-risk"). */
  slug: string;
  /** Display name on the industry page. */
  title: string;
  /** One line on what this model does and what it is trained on. */
  description: string;
  /**
   * Whether the model page exists. Every entry here is true: an unbuilt model
   * does not get a catalogue row. Whether a *checkpoint* exists is a separate,
   * runtime question answered by the FastAPI `/models` endpoint.
   */
  ready: boolean;
};

export type Industry = {
  /** URL slug; also the app/ directory name. */
  slug: string;
  /** Display name. */
  title: string;
  /** Short tagline shown on the landing tile. */
  tagline: string;
  /** Route href (always "/" + slug). */
  href: string;
  /** lucide-react icon component. */
  icon: LucideIcon;
  /** Number of models. Derived from `models.length`, never hand-maintained. */
  modelCount: number;
  models: IndustryModel[];
};

const FINTECH_MODELS: IndustryModel[] = [
  {
    slug: "fraud",
    title: "Payment Fraud",
    description:
      "LightGBM on IEEE-CIS (Vesta) real e-commerce payments, evaluated on a time-based split.",
    ready: true,
  },
  {
    slug: "credit-risk",
    title: "Consumer Credit Risk",
    description:
      "LightGBM on LendingClub 2007-2018Q4. Terminal loan outcomes only, post-origination fields denylisted.",
    ready: true,
  },
  {
    slug: "churn",
    title: "Card Attrition",
    description:
      "LightGBM on 10,127 real bank cardholders, scored by 5-fold stratified cross-validation.",
    ready: true,
  },
];

export const INDUSTRIES: Industry[] = [
  {
    slug: "fintech",
    title: "Fintech",
    tagline: "Payment fraud, consumer credit risk, and card attrition",
    href: "/fintech",
    icon: DollarSign,
    modelCount: FINTECH_MODELS.length,
    models: FINTECH_MODELS,
  },
];

/** Look up an industry by slug. Returns undefined for an unknown slug. */
export function getIndustry(slug: string): Industry | undefined {
  return INDUSTRIES.find((industry) => industry.slug === slug);
}

/** Flat list of every (industry, model) pair. Used by nav and the dashboard. */
export const ALL_MODELS = INDUSTRIES.flatMap((industry) =>
  industry.models.map((model) => ({
    ...model,
    industrySlug: industry.slug,
    industryTitle: industry.title,
    href: `/${industry.slug}/${model.slug}`,
  })),
);
