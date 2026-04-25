"use client";

/**
 * MetricSparkline — minimal Recharts LineChart styled as an inline
 * sparkline (no axes, no grid, no tooltip). Used by the dashboard's
 * per-model "metric history" column to surface how the key metric
 * has trended across the most recent training runs.
 *
 * Pure presentational Client Component — accepts pre-fetched data
 * and renders. No fetching or transformation inside; the dashboard
 * lib/dashboard.ts joins MLflow history into the row shape and
 * passes the numeric series in.
 */

import { Line, LineChart } from "recharts";

const DEFAULT_COLOR = "#3b82f6"; // blue-500
const DEFAULT_WIDTH = 120;
const DEFAULT_HEIGHT = 30;

type Props = {
  /** Ordered series — oldest first, newest last (left → right). */
  data: number[];
  /** Stroke color. Defaults to Tailwind blue-500. */
  color?: string;
  /** Pixel width. Defaults to 120 (table-cell-friendly). */
  width?: number;
  /** Pixel height. Defaults to 30 (single-line height). */
  height?: number;
};

export function MetricSparkline({
  data,
  color = DEFAULT_COLOR,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
}: Props) {
  if (data.length === 0) {
    // Don't draw an empty axis-less chart — a "No history" pill is
    // both more honest and avoids a Recharts warning about an
    // empty data prop.
    return (
      <span
        className="inline-flex items-center text-xs text-muted-foreground"
        style={{ width, height, lineHeight: `${height}px` }}
      >
        No history
      </span>
    );
  }

  // Recharts wants an array of objects; key doesn't matter since we
  // hide the axes. Use the index as a synthetic x-value.
  const series = data.map((value, idx) => ({ idx, value }));

  // Fixed width/height (no ResponsiveContainer) — jsdom tests can't
  // measure parent dimensions, and dashboard sparklines live in a
  // table cell with predictable size anyway.
  return (
    <div aria-hidden="true">
      <LineChart
        data={series}
        width={width}
        height={height}
        margin={{ top: 2, right: 2, bottom: 2, left: 2 }}
      >
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </div>
  );
}
