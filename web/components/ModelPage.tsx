"use client";

/**
 * ModelPage — the shared layout every model demo page renders.
 *
 * The three fintech pages were previously ~200 lines each of near-identical
 * form/mutation/result plumbing that had already drifted (two spelled the error
 * handler differently, one rendered its own result card). One component removes
 * that drift and makes the three demos genuinely interchangeable to a reviewer.
 *
 * Behaviour worth knowing:
 *
 * - Predict and explain fire in parallel via `Promise.allSettled`, so a failing
 *   explanation still shows the score and vice versa.
 * - A 503 (no checkpoint trained yet) renders `UntrainedNotice` with the API's
 *   own remediation text instead of a generic error toast. On a fresh clone that
 *   is the expected state, not a failure.
 */

import type { FieldValues } from "react-hook-form";
import { useForm, type DefaultValues, type Resolver } from "react-hook-form";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

import { ExplainabilityChart } from "@/components/ExplainabilityChart";
import { ModelForm, type FieldConfig } from "@/components/ModelForm";
import { PredictionResult, type Tone } from "@/components/PredictionResult";
import { UntrainedNotice } from "@/components/UntrainedNotice";
import { Card, CardContent } from "@/components/ui/card";
import { ApiError, apiErrorMessage, isUntrainedModelError, type Explanation } from "@/lib/api";

/** What a page extracts from its own prediction shape for the shared card. */
export type ResultView = {
  decisionLabel: string;
  decision: string;
  decisionTone: Tone;
  probabilityLabel: string;
  probability: number;
  modelVersion: string;
  trainedOn: string;
  caveat?: string;
};

export type ModelPageProps<TInput extends FieldValues, TPrediction> = {
  title: string;
  intro: string;
  /**
   * `zodResolver(SomeSchema)` from the calling page.
   *
   * The resolver is passed in already constructed rather than the schema
   * itself: `zodResolver` cannot infer through this component's generic
   * parameter, and resolving that at the concrete call site is honest typing
   * where a cast here would be a lie.
   */
  resolver: Resolver<TInput>;
  defaults: TInput;
  fields: FieldConfig<TInput>[];
  predict: (input: TInput) => Promise<TPrediction>;
  explain: (input: TInput) => Promise<Explanation>;
  toResultView: (prediction: TPrediction) => ResultView;
};

export function ModelPage<TInput extends FieldValues, TPrediction>({
  title,
  intro,
  resolver,
  defaults,
  fields,
  predict,
  explain,
  toResultView,
}: ModelPageProps<TInput, TPrediction>) {
  const form = useForm<TInput>({
    resolver,
    defaultValues: defaults as DefaultValues<TInput>,
  });

  const notifyUnlessUntrained = (label: string) => (error: unknown) => {
    // A 503 is rendered inline by UntrainedNotice; a toast on top would be noise.
    if (isUntrainedModelError(error)) return;
    toast.error(label, { description: apiErrorMessage(error) });
  };

  const prediction = useMutation<TPrediction, unknown, TInput>({
    mutationFn: predict,
    onError: notifyUnlessUntrained("Prediction failed"),
  });
  const explanation = useMutation<Explanation, unknown, TInput>({
    mutationFn: explain,
    onError: notifyUnlessUntrained("Explanation failed"),
  });

  const isPending = prediction.isPending || explanation.isPending;
  const untrained = isUntrainedModelError(prediction.error)
    ? (prediction.error as ApiError)
    : null;

  const onSubmit = async (values: TInput) => {
    await Promise.allSettled([
      prediction.mutateAsync(values),
      explanation.mutateAsync(values),
    ]);
  };

  return (
    <main className="container mx-auto grid max-w-6xl gap-6 px-4 py-8 md:grid-cols-2">
      <section aria-labelledby="form-heading" className="space-y-4">
        <header>
          <h1 id="form-heading" className="text-2xl font-semibold tracking-tight">
            {title}
          </h1>
          <p className="text-sm text-muted-foreground">{intro}</p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={fields}
              onSubmit={onSubmit}
              submitLabel={isPending ? "Scoring..." : "Score"}
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
            Prediction, provenance, and per-feature SHAP attribution.
          </p>
        </header>

        {isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Scoring…
            </CardContent>
          </Card>
        )}

        {!isPending && untrained && (
          <UntrainedNotice detail={apiErrorMessage(untrained)} />
        )}

        {!isPending && !untrained && prediction.data && (
          <PredictionResult {...toResultView(prediction.data)} />
        )}

        {!isPending && !untrained && explanation.data && (
          <ExplainabilityChart importances={explanation.data.feature_importances} />
        )}

        {!isPending && !untrained && !prediction.data && (
          <Card>
            <CardContent className="py-10 text-sm text-muted-foreground">
              Submit the form to see a score, the commit that trained the model,
              and the SHAP attribution behind the decision.
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
