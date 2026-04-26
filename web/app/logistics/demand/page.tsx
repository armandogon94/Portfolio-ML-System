"use client";

/**
 * Demand-forecast model demo page (A.9.6 — port of the Gradio demand tab).
 *
 * Single-input form (product category) wired into the LSTM 7-day
 * forecaster. Output is an array of 7 daily demand predictions
 * rendered as a Recharts LineChart instead of the result-card +
 * ExplainabilityChart layout other industry pages use. There's no
 * /explain/demand endpoint — the gradient explainer isn't wired
 * for the LSTM yet, so the right column shows just the line chart
 * plus a small summary line with the average predicted demand.
 *
 * Recharts replaces Gradio's Plotly chart — documented as
 * acceptable visual regression in SPEC §Q-A.9-1.
 */

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ModelForm } from "@/components/ModelForm";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { predictDemand, ApiError, type DemandForecast } from "@/lib/api";
import { DemandRequestSchema, DEMAND_REQUEST_DEFAULTS, type DemandRequest } from "@/lib/schemas";
import { DEMAND_REQUEST_FIELDS } from "./fields";

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

const FORECAST_LINE_COLOR = "#3b82f6"; // blue-500

function ForecastChart({ result }: { result: DemandForecast }) {
  // Map predictions to {day, demand} so Recharts can label the X
  // axis as "Day 1..7" (relative offsets, not calendar dates — the
  // server doesn't return a date series for the forecast horizon).
  const data = result.predictions.map((demand, idx) => ({
    day: `Day ${idx + 1}`,
    demand: Number(demand.toFixed(2)),
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {result.product.charAt(0).toUpperCase() + result.product.slice(1)} forecast
        </CardTitle>
        <CardDescription>
          Avg predicted demand over {result.forecast_days} days:{" "}
          <span data-testid="avg-demand" className="font-mono font-semibold">
            {result.avg_predicted_demand.toFixed(1)}
          </span>{" "}
          units/day
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div data-testid="forecast-chart" className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="day" className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip
                contentStyle={{
                  background: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "0.375rem",
                  fontSize: "0.75rem",
                }}
              />
              <Line
                type="monotone"
                dataKey="demand"
                stroke={FORECAST_LINE_COLOR}
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

export default function DemandPage() {
  const form = useForm<DemandRequest>({
    resolver: zodResolver(DemandRequestSchema),
    defaultValues: DEMAND_REQUEST_DEFAULTS,
  });

  const predict = useMutation<DemandForecast, unknown, DemandRequest>({
    mutationFn: predictDemand,
    onError: (err) => toast.error("Forecast failed", { description: apiErrorMessage(err) }),
  });

  const onSubmit = async (values: DemandRequest) => {
    await predict.mutateAsync(values).catch(() => {
      // onError already surfaced a toast — swallow so React Query's
      // unhandled-rejection guard doesn't log a second time.
    });
  };

  return (
    <main className="container mx-auto grid max-w-6xl gap-6 px-4 py-8 md:grid-cols-2">
      <section aria-labelledby="form-heading" className="space-y-4">
        <header>
          <h1 id="form-heading" className="text-2xl font-semibold tracking-tight">
            Demand Forecasting
          </h1>
          <p className="text-sm text-muted-foreground">
            Pick a product category to see the LSTM&apos;s 7-day demand forecast, seeded from the
            last 30 days of historical sales.
          </p>
        </header>
        <Card>
          <CardContent className="pt-6">
            <ModelForm
              form={form}
              fields={DEMAND_REQUEST_FIELDS}
              onSubmit={onSubmit}
              submitLabel={predict.isPending ? "Forecasting..." : "Submit"}
              isSubmitting={predict.isPending}
            />
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="result-heading" className="space-y-4">
        <header>
          <h2 id="result-heading" className="text-2xl font-semibold tracking-tight">
            Forecast
          </h2>
          <p className="text-sm text-muted-foreground">
            7-day demand projection from the LSTM forecaster.
          </p>
        </header>

        {predict.isPending && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating forecast…
            </CardContent>
          </Card>
        )}

        {!predict.isPending && predict.data && <ForecastChart result={predict.data} />}

        {!predict.isPending && !predict.data && (
          <Card>
            <CardContent className="py-10 text-sm text-muted-foreground">
              Submit a product category to see its 7-day demand forecast.
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
