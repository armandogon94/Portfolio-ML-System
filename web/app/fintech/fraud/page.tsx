"use client";

/**
 * Fraud-detection model demo page (A.9.4 — port of the Gradio fraud tab).
 *
 * Left column: ModelForm with the 9 transaction features, one of
 * which (merchant_category) is a categorical select.
 * Right column: inline FraudResult card + ExplainabilityChart.
 *
 * Result card renders the 4-tier risk_level (LOW / MEDIUM / HIGH /
 * CRITICAL) instead of credit-risk's 3-tier APPROVE/REVIEW/DECLINE,
 * so we render it locally rather than overloading PredictionResult.
 *
 * On submit we dispatch /predict/fraud and /explain/fraud in parallel
 * via Promise.allSettled — same pattern as every other industry page.
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
  predictFraud,
  explainFraud,
  ApiError,
  type FraudPrediction,
  type FraudRiskLevel,
  type Explanation,
} from "@/lib/api";
import {
  FraudInputSchema,
  FRAUD_DEFAULTS,
  type FraudInput,
} from "@/lib/schemas";
import { FRAUD_FIELDS } from "./fields";

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

// 4-tier semantic palette — green → amber → orange → red. Matches the
// credit-risk visual language (green = good) so users scanning the
// fintech industry index see consistent color semantics.
const RISK_LEVEL_CLASSES: Record<FraudRiskLevel, string> = {
  LOW:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
  MEDIUM:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  HIGH:
    "bg-orange-100 text-orange-800 border-orange-300 " +
    "dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-700",
  CRITICAL:
    "bg-red-100 text-red-800 border-red-300 " +
    "dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function FraudResult({ result }: { result: FraudPrediction }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>
          Anomaly score, autoencoder reconstruction error, and isolation-forest cross-check
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Risk level</span>
          <span
            data-testid="risk-level"
            className={cn(
              "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold",
              RISK_LEVEL_CLASSES[result.risk_level],
            )}
          >
            {result.risk_level}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Fraud probability</span>
          <span data-testid="fraud-probability" className="font-mono text-sm tabular-nums">
            {formatPercent(result.fraud_probability)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Reconstruction error</span>
          <span className="font-mono text-sm tabular-nums">
            {result.reconstruction_error.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Anomaly threshold</span>
          <span className="font-mono text-sm tabular-nums">
            {result.anomaly_threshold.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Isolation Forest</span>
          <span className="font-mono text-sm tabular-nums">
            {result.is_anomaly_isolation_forest ? "anomalous" : "normal"}
            {" · "}
            {result.isolation_forest_score.toFixed(3)}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export default function FraudPage() {
  const form = useForm<FraudInput>({
    resolver: zodResolver(FraudInputSchema),
    defaultValues: FRAUD_DEFAULTS,
  });

  const predict = useMutation<FraudPrediction, unknown, FraudInput>({
    mutationFn: predictFraud,
    onError: (err) => toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, FraudInput>({
    mutationFn: explainFraud,
    onError: (err) => toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: FraudInput) => {
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
            Fraud Detection
          </h1>
          <p className="text-sm text-muted-foreground">
            Score a transaction with the autoencoder anomaly detector — high
            reconstruction error means the transaction looks unlike anything
            the model saw in training.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={FRAUD_FIELDS}
              onSubmit={onSubmit}
              submitLabel={isPending ? "Analyzing..." : "Submit"}
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
            Anomaly score plus gradient-based feature attribution.
          </p>
        </header>

        {isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Analyzing transaction…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && <FraudResult result={predict.data} />}
        {!isPending && explain.data && (
          <ExplainabilityChart importances={explain.data.feature_importances} />
        )}

        {!isPending && !predict.data && !explain.data && (
          <Card>
            <CardContent className="py-10 text-sm text-muted-foreground">
              Submit the form to see the autoencoder&apos;s anomaly score and
              the gradient-based feature attribution.
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
