"use client";

/**
 * Rental-price model demo page (Real Estate / A.3).
 *
 * Regression output — no classification badge. The right column renders:
 *   1. Predicted nightly rate as "$X.XX/night"
 *   2. 90% confidence interval (low / high) when provided
 *   3. ExplainabilityChart from the SHAP endpoint
 *
 * Shape parallels the fintech/credit-risk page so readers can compare the
 * two vertical slices side-by-side.
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
import {
  predictRentalPrice,
  explainRentalPrice,
  ApiError,
  type RentalPricePrediction,
  type Explanation,
} from "@/lib/api";
import {
  RentalPriceInputSchema,
  RENTAL_PRICE_DEFAULTS,
  type RentalPriceInput,
} from "@/lib/schemas";
import { RENTAL_PRICE_FIELDS } from "./fields";

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

function formatUsd(value: number): string {
  return `$${value.toFixed(2)}`;
}

function RentalPriceResultCard({ result }: { result: RentalPricePrediction }) {
  const [low, high] = result.confidence_interval ?? [null, null];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Estimated nightly rate for this listing</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-baseline justify-between">
          <span className="text-sm text-muted-foreground">Nightly rate</span>
          <span data-testid="predicted-rate" className="font-mono text-2xl font-semibold tabular-nums">
            {formatUsd(result.predicted_rate)}
            <span className="ml-1 text-sm font-normal text-muted-foreground">/night</span>
          </span>
        </div>
        {low !== null && high !== null && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Confidence interval</span>
            <span data-testid="confidence-interval" className="font-mono text-sm tabular-nums">
              {formatUsd(low)} – {formatUsd(high)}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function RentalPricePage() {
  const form = useForm<RentalPriceInput>({
    resolver: zodResolver(RentalPriceInputSchema),
    defaultValues: RENTAL_PRICE_DEFAULTS,
  });

  const predict = useMutation<RentalPricePrediction, unknown, RentalPriceInput>({
    mutationFn: predictRentalPrice,
    onError: (err) => toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, RentalPriceInput>({
    mutationFn: explainRentalPrice,
    onError: (err) => toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: RentalPriceInput) => {
    // Parallel dispatch — we don't want the explanation call to wait on the
    // prediction response. allSettled keeps one failure from aborting the
    // other's toast/rendering path.
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
            Rental Price Prediction
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter a listing&apos;s features to see the model&apos;s estimated nightly rate and the
            features driving the prediction.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={RENTAL_PRICE_FIELDS}
              onSubmit={onSubmit}
              submitLabel={isPending ? "Predicting..." : "Submit"}
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
            Predicted nightly rate and SHAP-based feature attribution.
          </p>
        </header>

        {isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Predicting rate…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && <RentalPriceResultCard result={predict.data} />}
        {!isPending && explain.data && (
          <ExplainabilityChart importances={explain.data.feature_importances} />
        )}

        {!isPending && !predict.data && !explain.data && (
          <Card>
            <CardContent className="py-10 text-sm text-muted-foreground">
              Submit the form to see a predicted nightly rate alongside the
              SHAP-based explanation.
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
