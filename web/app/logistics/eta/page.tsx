"use client";

/**
 * Delivery-ETA model demo page (Phase A.7 — Logistics).
 *
 * Same composition as /fintech/credit-risk: ModelForm on the left,
 * prediction + SHAP explanation on the right, both mutations fired in
 * parallel. The result card is inlined here (instead of reusing
 * PredictionResult, which is typed for the credit-risk response) so
 * we can render the ETA in hours with a ±20% confidence band.
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
  predictDeliveryEta,
  explainDeliveryEta,
  ApiError,
  type DeliveryEtaPrediction,
  type Explanation,
} from "@/lib/api";
import {
  DeliveryEtaInputSchema,
  DELIVERY_ETA_DEFAULTS,
  type DeliveryEtaInput,
} from "@/lib/schemas";
import { DELIVERY_ETA_FIELDS } from "./fields";

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

/** Format a duration in hours as "N.N hours". One decimal keeps the
 * number readable without losing meaningful resolution (the model
 * predicts in hours, not sub-minutes). */
function formatHours(hours: number): string {
  return `${hours.toFixed(1)} hours`;
}

function DeliveryEtaResult({ result }: { result: DeliveryEtaPrediction }) {
  const [lo, hi] = result.confidence_interval;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Estimated delivery time and ±20% band</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">ETA</span>
          <span
            data-testid="eta-hours"
            className="font-mono text-sm tabular-nums"
          >
            {formatHours(result.eta_hours)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            Confidence interval
          </span>
          <span
            data-testid="confidence-interval"
            className="font-mono text-sm tabular-nums"
          >
            ({lo.toFixed(1)} – {hi.toFixed(1)})
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export default function DeliveryEtaPage() {
  const form = useForm<DeliveryEtaInput>({
    resolver: zodResolver(DeliveryEtaInputSchema),
    defaultValues: DELIVERY_ETA_DEFAULTS,
  });

  const predict = useMutation<DeliveryEtaPrediction, unknown, DeliveryEtaInput>({
    mutationFn: predictDeliveryEta,
    onError: (err) =>
      toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, DeliveryEtaInput>({
    mutationFn: explainDeliveryEta,
    onError: (err) =>
      toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: DeliveryEtaInput) => {
    // Parallel dispatch — mirrors the credit-risk page. Promise.allSettled
    // keeps one failure from aborting the other's toast/rendering path.
    await Promise.allSettled([
      predict.mutateAsync(values),
      explain.mutateAsync(values),
    ]);
  };

  return (
    <main className="container mx-auto grid max-w-6xl gap-6 px-4 py-8 md:grid-cols-2">
      <section aria-labelledby="form-heading" className="space-y-4">
        <header>
          <h1
            id="form-heading"
            className="text-2xl font-semibold tracking-tight"
          >
            Delivery ETA Prediction
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter shipment details to estimate delivery time in hours along
            with the drivers behind the prediction.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={DELIVERY_ETA_FIELDS}
              onSubmit={onSubmit}
              submitLabel={isPending ? "Predicting..." : "Submit"}
              isSubmitting={isPending}
            />
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="result-heading" className="space-y-4">
        <header>
          <h2
            id="result-heading"
            className="text-2xl font-semibold tracking-tight"
          >
            Result
          </h2>
          <p className="text-sm text-muted-foreground">
            Prediction and SHAP-based feature attribution.
          </p>
        </header>

        {isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Estimating delivery time…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && (
          <DeliveryEtaResult result={predict.data} />
        )}
        {!isPending && explain.data && (
          <ExplainabilityChart
            importances={explain.data.feature_importances}
          />
        )}

        {!isPending && !predict.data && !explain.data && (
          <Card>
            <CardContent className="py-10 text-sm text-muted-foreground">
              Submit the form to see a prediction and the SHAP-based
              explanation alongside it.
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
