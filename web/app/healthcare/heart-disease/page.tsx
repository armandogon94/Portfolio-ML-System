"use client";

/**
 * Heart-disease risk model demo page.
 *
 * Structure follows the credit-risk page: ModelForm on the left,
 * prediction + SHAP explanation on the right. PredictionResult is
 * specific to credit-risk (APPROVE/REVIEW/DECLINE), so heart-disease
 * renders its own inline result card with HIGH/ELEVATED/LOW banding.
 * Colors are semantic — red=HIGH, amber=ELEVATED, green=LOW — so the
 * viewer doesn't need to read probabilities to get a sense of risk.
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
  predictHeartDisease,
  explainHeartDisease,
  ApiError,
  type HeartDiseasePrediction,
  type Explanation,
} from "@/lib/api";
import {
  HeartDiseaseInputSchema,
  HEART_DISEASE_DEFAULTS,
  type HeartDiseaseInput,
} from "@/lib/schemas";
import { HEART_DISEASE_FIELDS } from "./fields";

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

const RISK_BAND_CLASSES: Record<HeartDiseasePrediction["risk_band"], string> = {
  HIGH:
    "bg-red-100 text-red-800 border-red-300 " +
    "dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
  ELEVATED:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  LOW:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function HeartDiseaseResultCard({ result }: { result: HeartDiseasePrediction }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Heart disease probability and risk band</CardDescription>
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
          <span className="text-sm text-muted-foreground">
            Probability of disease
          </span>
          <span
            data-testid="probability-disease"
            className="font-mono text-sm tabular-nums"
          >
            {formatPercent(result.probability_disease)}
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

export default function HeartDiseasePage() {
  const form = useForm<HeartDiseaseInput>({
    resolver: zodResolver(HeartDiseaseInputSchema),
    defaultValues: HEART_DISEASE_DEFAULTS,
  });

  const predict = useMutation<HeartDiseasePrediction, unknown, HeartDiseaseInput>({
    mutationFn: predictHeartDisease,
    onError: (err) =>
      toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, HeartDiseaseInput>({
    mutationFn: explainHeartDisease,
    onError: (err) =>
      toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: HeartDiseaseInput) => {
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
            Heart Disease Risk
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter patient vitals to see the model&apos;s predicted disease
            probability and which features most drove the score.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={HEART_DISEASE_FIELDS}
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
            Prediction and SHAP-based feature attribution.
          </p>
        </header>

        {isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Scoring vitals…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && <HeartDiseaseResultCard result={predict.data} />}
        {!isPending && explain.data && (
          <ExplainabilityChart importances={explain.data.feature_importances} />
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
