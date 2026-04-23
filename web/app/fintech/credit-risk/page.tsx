"use client";

/**
 * Credit-risk model demo page.
 *
 * Left column: ModelForm fed by the CreditRiskInputSchema.
 * Right column: PredictionResult + ExplainabilityChart, each
 * painting once its matching mutation resolves.
 *
 * On submit we fire both mutations in parallel via Promise.all so
 * a single user action surfaces both the prediction and the
 * explanation. Errors surface as Sonner toasts; the page stays
 * functional if the user retries.
 */

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

import { ModelForm } from "@/components/ModelForm";
import { PredictionResult } from "@/components/PredictionResult";
import { ExplainabilityChart } from "@/components/ExplainabilityChart";
import { Card, CardContent } from "@/components/ui/card";
import {
  predictCreditRisk,
  explainCreditRisk,
  ApiError,
  type CreditRiskPrediction,
  type Explanation,
} from "@/lib/api";
import {
  CreditRiskInputSchema,
  CREDIT_RISK_DEFAULTS,
  type CreditRiskInput,
} from "@/lib/schemas";
import { CREDIT_RISK_FIELDS } from "./fields";

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

export default function CreditRiskPage() {
  const form = useForm<CreditRiskInput>({
    resolver: zodResolver(CreditRiskInputSchema),
    defaultValues: CREDIT_RISK_DEFAULTS,
  });

  const predict = useMutation<CreditRiskPrediction, unknown, CreditRiskInput>({
    mutationFn: predictCreditRisk,
    onError: (err) => toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, CreditRiskInput>({
    mutationFn: explainCreditRisk,
    onError: (err) => toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: CreditRiskInput) => {
    // Parallel dispatch — both endpoints take <100ms locally; we don't
    // want to serialize them. Promise.allSettled keeps one failure
    // from aborting the other's toast/rendering path.
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
            Credit Risk Scoring
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter a loan application to see the model&apos;s risk score and the
            features driving the decision.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={CREDIT_RISK_FIELDS}
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
              Scoring application…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && <PredictionResult result={predict.data} />}
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
