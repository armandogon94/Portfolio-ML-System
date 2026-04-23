"use client";

/**
 * Customer-churn model demo page.
 *
 * Left column: ModelForm fed by the CustomerChurnInputSchema.
 * Right column: inline ChurnResult card + ExplainabilityChart, each
 * painting once its matching mutation resolves.
 *
 * Shape mirrors the credit-risk page intentionally — the two fintech
 * demos should feel interchangeable from the user's perspective. The
 * result card diverges because churn outputs a 3-tier retention
 * recommendation instead of credit-risk's approve/review/decline, so
 * we render it locally rather than overloading PredictionResult.
 *
 * On submit we dispatch both mutations in parallel via
 * Promise.allSettled so one failure doesn't abort the other's
 * toast/rendering path.
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
  predictCustomerChurn,
  explainCustomerChurn,
  ApiError,
  type CustomerChurnPrediction,
  type Explanation,
  type RetentionAction,
} from "@/lib/api";
import {
  CustomerChurnInputSchema,
  CUSTOMER_CHURN_DEFAULTS,
  type CustomerChurnInput,
} from "@/lib/schemas";
import { CUSTOMER_CHURN_FIELDS } from "./fields";

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

// Semantic color per retention tier — red = immediate action, amber =
// soft-touch, muted = no action needed. Matches credit-risk's visual
// language so users can scan fintech pages side by side.
const RETENTION_CLASSES: Record<RetentionAction, string> = {
  URGENT_OUTREACH:
    "bg-red-100 text-red-800 border-red-300 " +
    "dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
  PROACTIVE_CHECKIN:
    "bg-amber-100 text-amber-800 border-amber-300 " +
    "dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700",
  NO_ACTION:
    "bg-green-100 text-green-800 border-green-300 " +
    "dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function ChurnResult({ result }: { result: CustomerChurnPrediction }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>Churn probability and retention action</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Recommendation</span>
          <span
            data-testid="recommendation"
            className={cn(
              "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold",
              RETENTION_CLASSES[result.retention_recommendation],
            )}
          >
            {result.retention_recommendation}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Churn probability</span>
          <span data-testid="probability-churn" className="font-mono text-sm tabular-nums">
            {formatPercent(result.probability_churn)}
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

export default function CustomerChurnPage() {
  const form = useForm<CustomerChurnInput>({
    resolver: zodResolver(CustomerChurnInputSchema),
    defaultValues: CUSTOMER_CHURN_DEFAULTS,
  });

  const predict = useMutation<CustomerChurnPrediction, unknown, CustomerChurnInput>({
    mutationFn: predictCustomerChurn,
    onError: (err) => toast.error("Prediction failed", { description: apiErrorMessage(err) }),
  });
  const explain = useMutation<Explanation, unknown, CustomerChurnInput>({
    mutationFn: explainCustomerChurn,
    onError: (err) => toast.error("Explanation failed", { description: apiErrorMessage(err) }),
  });

  const isPending = predict.isPending || explain.isPending;

  const onSubmit = async (values: CustomerChurnInput) => {
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
            Customer Churn Prediction
          </h1>
          <p className="text-sm text-muted-foreground">
            Score a bank customer to see churn probability and the recommended
            retention action.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={CUSTOMER_CHURN_FIELDS}
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
              Scoring customer…
            </CardContent>
          </Card>
        )}

        {!isPending && predict.data && <ChurnResult result={predict.data} />}
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
