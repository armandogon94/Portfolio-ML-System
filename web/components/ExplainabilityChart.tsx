"use client";

/**
 * ExplainabilityChart — horizontal bar chart of feature importances
 * from a SHAP or gradient-based explainer. Bars colored by sign:
 * positive (raises predicted value) in red-ish, negative (reduces
 * predicted value) in green-ish. Sorted by |importance| descending
 * so the most impactful features appear on top.
 */

import { Bar, BarChart, Cell, ReferenceLine, XAxis, YAxis } from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

/** Sort an importances map into a ranked array (desc by |value|). */
export function sortByAbsImportance(
  importances: Record<string, number>,
  topN?: number,
): { feature: string; importance: number }[] {
  const ranked = Object.entries(importances)
    .map(([feature, importance]) => ({ feature, importance }))
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
  return topN === undefined ? ranked : ranked.slice(0, topN);
}

// Sign → Tailwind-ish hex. Kept explicit (not CSS vars) because
// Recharts paints SVG attributes directly, not via CSS classes.
const POSITIVE_COLOR = "#ef4444"; // red-500: raises risk / value
const NEGATIVE_COLOR = "#10b981"; // emerald-500: lowers risk / value

type Props = {
  importances: Record<string, number>;
  /** Default 10. Pass Infinity / omit for all. */
  topN?: number;
  /** Optional chart title and subtitle overrides. */
  title?: string;
  description?: string;
};

export function ExplainabilityChart({
  importances,
  topN = 10,
  title = "Explainability",
  description = "Top features driving this prediction (sorted by impact)",
}: Props) {
  const data = sortByAbsImportance(importances, topN);

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No features to display.
          </p>
        ) : (
          // Fixed dimensions (vs ResponsiveContainer) — jsdom has no layout,
          // and fixed width/height makes the chart testable under Vitest.
          // 340px is a reasonable default for a card body; callers can
          // wrap in a container if they need responsive behavior.
          <BarChart
            width={340}
            height={Math.max(200, data.length * 28)}
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
          >
            <XAxis type="number" hide />
            <YAxis
              type="category"
              dataKey="feature"
              width={140}
              tick={{ fontSize: 12 }}
            />
            <ReferenceLine x={0} stroke="currentColor" strokeOpacity={0.3} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {data.map((entry) => (
                <Cell
                  key={entry.feature}
                  fill={entry.importance >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR}
                />
              ))}
            </Bar>
          </BarChart>
        )}
      </CardContent>
    </Card>
  );
}
