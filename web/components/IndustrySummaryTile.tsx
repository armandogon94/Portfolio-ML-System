/**
 * IndustrySummaryTile — at-a-glance card per industry on the dashboard.
 *
 * Shows the industry's icon + title (linked to the index page),
 * how many of its planned models are ready (e.g., "3/4 ready"),
 * and the average key-metric across the ready ones.
 *
 * Pure data-in: receives the Industry registry entry plus the
 * subset of dashboard rows belonging to that industry. Filtering
 * the rows is done by the caller (the /dashboard page in A.9.9).
 */

import Link from "next/link";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Industry } from "@/lib/industries";
import type { DashboardRow } from "@/lib/dashboard";

type Props = {
  industry: Industry;
  rows: DashboardRow[];
};

function avgReadyMetric(rows: DashboardRow[]): { value: number; sample: DashboardRow["keyMetric"] } | null {
  // Average across rows that are both ready AND have a numeric metric
  // value. Mixing different metrics (AUC vs RMSE) would be nonsense,
  // but within an industry the per-model metrics tend to share a
  // family (classifiers → AUC, regressors → R²) — close enough for
  // a tile-level summary. We still return one of the matching
  // keyMetric objects so the tile can label correctly.
  const ready = rows.filter(
    (r) => r.status === "ready" && r.keyMetric && r.keyMetric.value != null,
  );
  if (ready.length === 0) return null;
  const sum = ready.reduce((acc, r) => acc + (r.keyMetric!.value ?? 0), 0);
  return {
    value: sum / ready.length,
    sample: ready[0].keyMetric,
  };
}

function formatMetric(value: number, sample: DashboardRow["keyMetric"]): string {
  if (sample?.unit === "%" || (value >= 0 && value <= 1)) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(2) + (sample?.unit ? ` ${sample.unit}` : "");
}

export function IndustrySummaryTile({ industry, rows }: Props) {
  const Icon = industry.icon;
  const readyCount = rows.filter((r) => r.status === "ready").length;
  const total = industry.modelCount;
  const avg = avgReadyMetric(rows);

  return (
    <Card>
      <CardHeader className="pb-3">
        <Link
          href={industry.href}
          className="inline-flex items-center gap-2 hover:underline"
          aria-label={industry.title}
        >
          <Icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          <CardTitle className="text-base">{industry.title}</CardTitle>
        </Link>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-baseline justify-between text-sm">
          <span className="text-muted-foreground">Models</span>
          <span data-testid="ready-count" className="font-mono tabular-nums">
            {readyCount}/{total} ready
          </span>
        </div>
        {avg ? (
          <div className="flex items-baseline justify-between text-sm">
            <span className="text-muted-foreground">
              Avg {avg.sample?.label ?? "metric"}
            </span>
            <span data-testid="avg-metric" className="font-mono tabular-nums">
              {formatMetric(avg.value, avg.sample ?? null)}
            </span>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No ready models yet.</p>
        )}
      </CardContent>
    </Card>
  );
}
