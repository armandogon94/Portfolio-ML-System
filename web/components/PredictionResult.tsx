/**
 * PredictionResult — right-column result card for any classification model.
 *
 * Currently typed for credit-risk (APPROVE / REVIEW / DECLINE); future
 * industry models can either reuse this shape or render their own card.
 * Color coding is semantic — green for approve/good, amber for review,
 * red for decline/high-risk — so the user doesn't need to read numbers
 * to get a sense of the outcome.
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { CreditRiskPrediction, Recommendation } from "@/lib/api";

/** Tailwind classes per recommendation tier. Exported for reuse / testing. */
export const RECOMMENDATION_CLASSES: Record<Recommendation, string> = {
  APPROVE:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
  REVIEW:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  DECLINE:
    "bg-red-100 text-red-800 border-red-300 " +
    "dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
};

function formatPercent(value: number): string {
  // Two-decimal percentage — e.g., 0.12345 -> "12.35%". We keep the
  // decimals even when they'd round to .00 so the output width is
  // stable column-to-column.
  return `${(value * 100).toFixed(2)}%`;
}

export function PredictionResult({ result }: { result: CreditRiskPrediction }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Model output and recommendation</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Recommendation</span>
          <span
            data-testid="recommendation"
            className={cn(
              "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold",
              RECOMMENDATION_CLASSES[result.recommendation],
            )}
          >
            {result.recommendation}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Risk score</span>
          <span data-testid="risk-score" className="font-mono text-sm tabular-nums">
            {formatPercent(result.risk_score)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Confidence</span>
          <span data-testid="confidence" className="font-mono text-sm tabular-nums">
            {formatPercent(result.confidence)}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
