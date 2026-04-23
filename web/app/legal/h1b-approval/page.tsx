"use client";

/**
 * H-1B approval model demo page.
 *
 * Layout and submit flow mirror /fintech/credit-risk: ModelForm on the
 * left, prediction + SHAP explanation on the right, both mutations fired
 * in parallel via Promise.allSettled so a single user action surfaces
 * both results (and one failure doesn't block the other from painting).
 *
 * The recommendation tier enum differs from credit-risk
 * (APPROVE_LIKELY / REVIEW / DECLINE_LIKELY), so this page renders its
 * own result card rather than reusing PredictionResult.
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
  predictH1bApproval,
  explainH1bApproval,
  ApiError,
  type H1BApprovalPrediction,
  type H1BRecommendation,
  type Explanation,
} from "@/lib/api";
import {
  H1BApprovalInputSchema,
  H1B_APPROVAL_DEFAULTS,
  type H1BApprovalInput,
} from "@/lib/schemas";
import { H1B_APPROVAL_FIELDS } from "./fields";

/** Tailwind classes per H-1B recommendation tier. Green = approve, amber =
 * review, red = decline. Colors match the credit-risk card's visual
 * language so users on both pages get the same semantic cues. */
const H1B_RECOMMENDATION_CLASSES: Record<H1BRecommendation, string> = {
  APPROVE_LIKELY:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
  REVIEW:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  DECLINE_LIKELY:
    "bg-red-100 text-red-800 border-red-300 " +
    "dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

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

function H1BPredictionResult({ result }: { result: H1BApprovalPrediction }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Approval probability and recommendation</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Recommendation</span>
          <span
            data-testid="recommendation"
            className={cn(
              "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold",
              H1B_RECOMMENDATION_CLASSES[result.recommendation],
            )}
          >
            {result.recommendation}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            Approval probability
          </span>
          <span
            data-testid="probability-approval"
            className="font-mono text-sm tabular-nums"
          >
            {formatPercent(result.probability_approval)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Confidence</span>
          <span
            data-testid="confidence"
            className="font-mono text-sm tabular-nums"
          >
            {formatPercent(result.confidence)}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export default function H1BApprovalPage() {
  const form = useForm<H1BApprovalInput>({
    resolver: zodResolver(H1BApprovalInputSchema),
    defaultValues: H1B_APPROVAL_DEFAULTS,
  });

  const predict = useMutation<H1BApprovalPrediction, unknown, H1BApprovalInput>({
    mutationFn: predictH1bApproval,
    onError: (err) =>
      toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, H1BApprovalInput>({
    mutationFn: explainH1bApproval,
    onError: (err) =>
      toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: H1BApprovalInput) => {
    // Parallel dispatch so a single click surfaces both prediction and
    // SHAP explanation. allSettled keeps one failure from aborting the
    // other's toast/render path — matches credit-risk page behavior.
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
            H-1B Visa Approval
          </h1>
          <p className="text-sm text-muted-foreground">
            Enter an LCA-inspired petition to see the model&apos;s approval
            probability and the features driving the decision.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={H1B_APPROVAL_FIELDS}
              onSubmit={onSubmit}
              submitLabel={isPending ? "Scoring..." : "Submit"}
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
              Scoring petition…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && (
          <H1BPredictionResult result={predict.data} />
        )}
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
