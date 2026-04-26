"use client";

/**
 * DashboardTable — sortable, filterable table of every model in the
 * 20-entry catalog. Pure data-in: receives a DashboardRow[] from the
 * /dashboard Server Component and owns only the local sort + filter
 * state. No data fetching here.
 *
 * Columns:
 *   1. Industry           (text, sortable alpha)
 *   2. Model              (link if ready, plain text otherwise)
 *   3. Status             (ModelStatusBadge)
 *   4. Key metric         (numeric + label + unit, sortable desc)
 *   5. Trend              (MetricSparkline of last N runs)
 *   6. Last trained       (locale date, sortable desc)
 *
 * Sort: clicking a header toggles between asc / desc. "Status" and
 * "industry" sort lexically; "metric" and "last trained" sort
 * numerically. `null` values always sort last regardless of order.
 *
 * Filter: a single industry-slug select narrows the rows to one
 * vertical. INDUSTRIES is read from lib/industries.ts so the option
 * list stays in lockstep with the registry.
 */

import * as React from "react";
import Link from "next/link";

import { ModelStatusBadge } from "@/components/ModelStatusBadge";
import { MetricSparkline } from "@/components/MetricSparkline";
import { INDUSTRIES } from "@/lib/industries";
import type { DashboardRow } from "@/lib/dashboard";

type SortKey = "industry" | "model" | "status" | "metric" | "lastTrained";
type SortDir = "asc" | "desc";

// Compare with nullish-last semantics: null values always sort to
// the end of the list regardless of direction. We multiply only the
// non-null branch by `sign`, so flipping asc↔desc never bubbles "no
// data" rows to the top.
function compareNullable<T>(
  a: T | null | undefined,
  b: T | null | undefined,
  cmp: (x: T, y: T) => number,
  sign: number,
): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1; // a after
  if (b == null) return -1; // b after
  return sign * cmp(a, b);
}

function sortRows(rows: DashboardRow[], key: SortKey, dir: SortDir): DashboardRow[] {
  const sign = dir === "asc" ? 1 : -1;
  const sorted = [...rows].sort((a, b) => {
    switch (key) {
      case "industry":
        return sign * a.industryTitle.localeCompare(b.industryTitle);
      case "model":
        return sign * a.modelTitle.localeCompare(b.modelTitle);
      case "status":
        // Status order: ready (0) → training (1) → not_built (2)
        // so asc puts ready first.
        return sign * (statusRank(a.status) - statusRank(b.status));
      case "metric":
        return compareNullable(a.keyMetric?.value, b.keyMetric?.value, (x, y) => x - y, sign);
      case "lastTrained":
        return compareNullable(
          a.lastTrained,
          b.lastTrained,
          (x, y) => Date.parse(x) - Date.parse(y),
          sign,
        );
    }
  });
  return sorted;
}

function statusRank(s: DashboardRow["status"]): number {
  return s === "ready" ? 0 : s === "training" ? 1 : 2;
}

function formatMetric(metric: DashboardRow["keyMetric"]): string {
  if (!metric || metric.value == null) return "—";
  // Heuristic: AUC, R², probability-like scores live in [0, 1] — show
  // as percentage with one decimal. Larger magnitudes (RMSE, MAE,
  // dollar predictions) get 4 significant figures.
  if (metric.unit === "%" || (metric.value >= 0 && metric.value <= 1)) {
    return `${(metric.value * 100).toFixed(1)}%`;
  }
  return metric.value.toFixed(2) + (metric.unit ? ` ${metric.unit}` : "");
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  // ISO → YYYY-MM-DD for table density. The full timestamp is
  // available on hover via the title attribute.
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 10);
}

type SortableHeaderProps = {
  label: string;
  sortKey: SortKey;
  current: SortKey;
  dir: SortDir;
  onClick: (k: SortKey) => void;
  ariaLabel: string;
};

function SortableHeader({ label, sortKey, current, dir, onClick, ariaLabel }: SortableHeaderProps) {
  const active = current === sortKey;
  return (
    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
      <button
        type="button"
        onClick={() => onClick(sortKey)}
        aria-label={ariaLabel}
        className="inline-flex items-center gap-1 hover:text-foreground"
      >
        {label}
        {active && <span className="text-xs">{dir === "asc" ? "▲" : "▼"}</span>}
      </button>
    </th>
  );
}

export function DashboardTable({ rows }: { rows: DashboardRow[] }) {
  const [industryFilter, setIndustryFilter] = React.useState<string>("all");
  const [sortKey, setSortKey] = React.useState<SortKey>("status");
  const [sortDir, setSortDir] = React.useState<SortDir>("asc");

  const filtered = React.useMemo(
    () => (industryFilter === "all" ? rows : rows.filter((r) => r.industrySlug === industryFilter)),
    [rows, industryFilter],
  );
  const sorted = React.useMemo(
    () => sortRows(filtered, sortKey, sortDir),
    [filtered, sortKey, sortDir],
  );

  const handleSortClick = (k: SortKey) => {
    if (k === sortKey) {
      // Toggle direction on repeat click.
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(k);
      // Numeric/temporal columns default to descending (best/newest first);
      // text columns default to ascending (A → Z).
      setSortDir(k === "metric" || k === "lastTrained" ? "desc" : "asc");
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <label htmlFor="industry-filter" className="text-sm text-muted-foreground">
          Industry filter
        </label>
        <select
          id="industry-filter"
          aria-label="Industry filter"
          value={industryFilter}
          onChange={(e) => setIndustryFilter(e.target.value)}
          className={
            "h-8 rounded-md border border-input bg-transparent px-2 text-sm " +
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          }
        >
          <option value="all">All industries</option>
          {INDUSTRIES.map((i) => (
            <option key={i.slug} value={i.slug}>
              {i.title}
            </option>
          ))}
        </select>
      </div>

      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40">
            <tr>
              <SortableHeader
                label="Industry"
                sortKey="industry"
                current={sortKey}
                dir={sortDir}
                onClick={handleSortClick}
                ariaLabel="Sort by industry"
              />
              <SortableHeader
                label="Model"
                sortKey="model"
                current={sortKey}
                dir={sortDir}
                onClick={handleSortClick}
                ariaLabel="Sort by model"
              />
              <SortableHeader
                label="Status"
                sortKey="status"
                current={sortKey}
                dir={sortDir}
                onClick={handleSortClick}
                ariaLabel="Sort by status"
              />
              <SortableHeader
                label="Metric"
                sortKey="metric"
                current={sortKey}
                dir={sortDir}
                onClick={handleSortClick}
                ariaLabel="Sort by metric"
              />
              <th
                scope="col"
                className="px-3 py-2 text-left text-xs font-medium text-muted-foreground"
              >
                Trend
              </th>
              <SortableHeader
                label="Last trained"
                sortKey="lastTrained"
                current={sortKey}
                dir={sortDir}
                onClick={handleSortClick}
                ariaLabel="Sort by last trained date"
              />
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-6 text-center text-sm text-muted-foreground">
                  No models to display.
                </td>
              </tr>
            ) : (
              sorted.map((r) => (
                <tr key={`${r.industrySlug}/${r.modelSlug}`} className="border-t">
                  <td className="px-3 py-2 align-top">{r.industryTitle}</td>
                  <td className="px-3 py-2 align-top">
                    {r.href ? (
                      <Link
                        href={r.href}
                        data-testid="model-name"
                        className="font-medium text-primary hover:underline"
                      >
                        {r.modelTitle}
                      </Link>
                    ) : (
                      <span data-testid="model-name" className="font-medium text-foreground">
                        {r.modelTitle}
                      </span>
                    )}
                    <p className="text-xs text-muted-foreground">{r.modelDescription}</p>
                  </td>
                  <td className="px-3 py-2 align-top">
                    <ModelStatusBadge status={r.status} />
                  </td>
                  <td className="px-3 py-2 align-top">
                    <div className="font-mono text-sm tabular-nums">
                      {formatMetric(r.keyMetric)}
                    </div>
                    {r.keyMetric && (
                      <div className="text-xs text-muted-foreground">{r.keyMetric.label}</div>
                    )}
                  </td>
                  <td className="px-3 py-2 align-top">
                    <MetricSparkline data={r.history} />
                  </td>
                  <td
                    className="px-3 py-2 align-top text-xs text-muted-foreground"
                    title={r.lastTrained ?? undefined}
                  >
                    {formatDate(r.lastTrained)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
