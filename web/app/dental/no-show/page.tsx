"use client";

/**
 * Dental patient no-show model demo page.
 *
 * Pattern mirrors /fintech/credit-risk: a ModelForm on the left, a
 * result card + ExplainabilityChart on the right. The result card is
 * inlined here (NoShowResult) rather than reusing PredictionResult
 * because the schema differs — probability_no_show + risk_band
 * instead of risk_score + recommendation.
 *
 * On submit we fire predict + explain in parallel via Promise.allSettled
 * so a single user action surfaces both outputs; errors become toasts.
 */

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

import { ModelForm } from "@/components/ModelForm";
import { ExplainabilityChart } from "@/components/ExplainabilityChart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  predictDentalNoShow,
  explainDentalNoShow,
  ApiError,
  type DentalNoShowPrediction,
  type Explanation,
  type RiskBand,
} from "@/lib/api";
import {
  DentalNoShowInputSchema,
  DENTAL_NOSHOW_DEFAULTS,
  type DentalNoShowInput,
} from "@/lib/schemas";
import { DENTAL_NOSHOW_FIELDS } from "./fields";

function apiErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    const body = err.body as { detail?: unknown } | null;
    if (body && typeof body.detail === "string") {
      return `API ${err.status}: ${body.detail}`;
    }
    return `API ${err.status}`;
  }
  return err instanceof Error ? err.message : String(err);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

// Tailwind classes per risk band — HIGH_RISK red, MODERATE amber,
// LIKELY_TO_SHOW green. Matches the semantic palette used by
// PredictionResult.tsx so the visual language is consistent across
// industries.
const RISK_BAND_CLASSES: Record<RiskBand, string> = {
  HIGH_RISK:
    "bg-red-100 text-red-800 border-red-300 " +
    "dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
  MODERATE:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  LIKELY_TO_SHOW:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
};

function NoShowResult({ result }: { result: DentalNoShowPrediction }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Probability this patient misses their appointment</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Risk band</span>
          <span
            data-testid="risk-band"
            className={cn(
              "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold",
              RISK_BAND_CLASSES[result.risk_band],
            )}
          >
            {result.risk_band}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">No-show probability</span>
          <span
            data-testid="probability-no-show"
            className="font-mono text-sm tabular-nums"
          >
            {formatPercent(result.probability_no_show)}
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

export default function DentalNoShowPage() {
  const form = useForm<DentalNoShowInput>({
    resolver: zodResolver(DentalNoShowInputSchema),
    defaultValues: DENTAL_NOSHOW_DEFAULTS,
  });

  const predict = useMutation<DentalNoShowPrediction, unknown, DentalNoShowInput>({
    mutationFn: predictDentalNoShow,
    onError: (err) => toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, DentalNoShowInput>({
    mutationFn: explainDentalNoShow,
    onError: (err) => toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: DentalNoShowInput) => {
    await Promise.allSettled([
      predict.mutateAsync(values),
      explain.mutateAsync(values),
    ]);
  };

  return (
    <main className="container mx-auto grid max-w-6xl gap-6 px-4 py-8 md:grid-cols-2">
      <section aria-labelledby="form-heading" className="space-y-4">
        <header>
          <h1 id="form-heading" className="text-2xl font-semibold tracking-tight">
            Patient No-Show Prediction
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter an appointment to see the modeled probability the patient
            misses it, plus the features driving that risk.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={DENTAL_NOSHOW_FIELDS}
              onSubmit={onSubmit}
              submitLabel={isPending ? "Scoring..." : "Submit"}
              isSubmitting={isPending}
            />
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="result-heading" className="space-y-4">
        <header>
          <h2 id="result-heading" className="text-2xl font-semibold tracking-tight">
            Result
          </h2>
          <p className="text-sm text-muted-foreground">
            Risk band plus SHAP-based feature attribution.
          </p>
        </header>

        {isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Scoring appointment…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && <NoShowResult result={predict.data} />}
        {!isPending && explain.data && (
          <ExplainabilityChart importances={explain.data.feature_importances} />
        )}

        {!isPending && !predict.data && !explain.data && (
          <Card>
            <CardContent className="py-10 text-sm text-muted-foreground">
              Submit the form to see a risk band and the SHAP-based
              explanation alongside it.
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
