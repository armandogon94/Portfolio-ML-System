/**
 * PredictionResult — the shared right-column result card for all three models.
 *
 * One component, three problems. Each page supplies a decision label, a
 * probability and its rows; the semantics of the decision differ but the visual
 * language does not, so a reviewer scanning the three pages sees one product.
 *
 * Two things this card always shows, and the reason it exists as a shared
 * component rather than three bespoke ones:
 *
 * `model_version` — the git SHA of the commit that trained the checkpoint. A
 *   score with no provenance is unreviewable, and this repository exists because
 *   an unreviewable score was published.
 * `caveat` — free text the API attaches to a score (small n, illustrative
 *   thresholds). It is rendered, not swallowed.
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/** Semantic tone. Green = benign, amber = look at it, red = act. */
export type Tone = "good" | "warn" | "bad";

export const TONE_CLASSES: Record<Tone, string> = {
  good:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
  warn:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  bad:
    "bg-red-100 text-red-800 border-red-300 " +
    "dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
};

export function formatPercent(value: number): string {
  // Decimals are kept even when they round to .00 so the column width is
  // stable between renders and does not jitter as the user resubmits.
  return `${(value * 100).toFixed(2)}%`;
}

export type PredictionResultProps = {
  /** e.g. "Decision", "Recommended action". */
  decisionLabel: string;
  /** e.g. "APPROVE", "MANUAL_REVIEW". */
  decision: string;
  decisionTone: Tone;
  /** e.g. "Default probability". */
  probabilityLabel: string;
  probability: number;
  /** Short git SHA of the commit that trained the checkpoint. */
  modelVersion: string;
  /** Dataset slug the checkpoint was trained on. */
  trainedOn: string;
  /** Any caveat the API attached to this score. Rendered verbatim. */
  caveat?: string;
};

export function PredictionResult({
  decisionLabel,
  decision,
  decisionTone,
  probabilityLabel,
  probability,
  modelVersion,
  trainedOn,
  caveat,
}: PredictionResultProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Model output, decision, and provenance</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{decisionLabel}</span>
          <span
            data-testid="decision"
            className={cn(
              "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold",
              TONE_CLASSES[decisionTone],
            )}
          >
            {decision}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{probabilityLabel}</span>
          <span data-testid="probability" className="font-mono text-sm tabular-nums">
            {formatPercent(probability)}
          </span>
        </div>

        <div className="flex items-center justify-between border-t pt-3">
          <span className="text-xs text-muted-foreground">Model version</span>
          <span data-testid="model-version" className="font-mono text-xs text-muted-foreground">
            {modelVersion}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Trained on</span>
          <span data-testid="trained-on" className="font-mono text-xs text-muted-foreground">
            {trainedOn}
          </span>
        </div>

        {caveat && (
          <p data-testid="caveat" className="border-t pt-3 text-xs text-muted-foreground">
            {caveat}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
