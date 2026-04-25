"use client";

/**
 * Price-prediction model demo page (A.9.5 — port of the Gradio price tab).
 *
 * Synthetic-data LightGBM regressor. Output is a single dollar amount
 * plus a hardcoded ±10% range from the predictor — same shape as the
 * rental-price page (A.3), but values use thousands-grouped USD
 * formatting since residential prices land in the $100k–$1M+ range.
 *
 * On submit we dispatch /predict/price and /explain/price in parallel
 * via Promise.allSettled — one failure doesn't abort the other path.
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
  predictPrice,
  explainPrice,
  ApiError,
  type PricePrediction,
  type Explanation,
} from "@/lib/api";
import {
  PricePredictionInputSchema,
  PRICE_PREDICTION_DEFAULTS,
  type PricePredictionInput,
} from "@/lib/schemas";
import { PRICE_PREDICTION_FIELDS } from "./fields";

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

const USD_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function formatUsd(value: number): string {
  return USD_FORMATTER.format(value);
}

function PriceResultCard({ result }: { result: PricePrediction }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Estimated home price (LightGBM regressor)</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-baseline justify-between">
          <span className="text-sm text-muted-foreground">Predicted price</span>
          <span data-testid="predicted-price" className="font-mono text-2xl font-semibold tabular-nums">
            {formatUsd(result.predicted_price)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">±10% range</span>
          <span data-testid="price-range" className="font-mono text-sm tabular-nums">
            {formatUsd(result.price_range_low)} – {formatUsd(result.price_range_high)}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export default function PricePage() {
  const form = useForm<PricePredictionInput>({
    resolver: zodResolver(PricePredictionInputSchema),
    defaultValues: PRICE_PREDICTION_DEFAULTS,
  });

  const predict = useMutation<PricePrediction, unknown, PricePredictionInput>({
    mutationFn: predictPrice,
    onError: (err) => toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, PricePredictionInput>({
    mutationFn: explainPrice,
    onError: (err) => toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: PricePredictionInput) => {
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
            Price Prediction
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter a property&apos;s features to see the model&apos;s estimated
            sale price and the features driving the prediction.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={PRICE_PREDICTION_FIELDS}
              onSubmit={onSubmit}
              submitLabel={isPending ? "Estimating..." : "Submit"}
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
            Predicted price and SHAP-based feature attribution.
          </p>
        </header>

        {isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Estimating price…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && <PriceResultCard result={predict.data} />}
        {!isPending && explain.data && (
          <ExplainabilityChart importances={explain.data.feature_importances} />
        )}

        {!isPending && !predict.data && !explain.data && (
          <Card>
            <CardContent className="py-10 text-sm text-muted-foreground">
              Submit the form to see a predicted price alongside the
              SHAP-based explanation.
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
