/** A.9.7: MetricSparkline component tests.
 *
 * The sparkline is a tiny inline LineChart (no axes, no tooltip)
 * used by the dashboard's per-row "metric history" cell. Pure
 * presentational Client Component: accepts data, optional color,
 * optional height/width. No data fetching inside.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { MetricSparkline } from "@/components/MetricSparkline";

describe("MetricSparkline", () => {
  it("renders an SVG sparkline when given a non-empty data array", () => {
    const { container } = render(<MetricSparkline data={[1, 2, 3, 4, 5]} />);
    // Recharts renders to <svg>; we just need to confirm one is present.
    expect(container.querySelector("svg")).toBeInTheDocument();
    // Sparkline should NOT render axes or a legend.
    expect(container.querySelector(".recharts-cartesian-axis")).not.toBeInTheDocument();
    expect(container.querySelector(".recharts-legend-wrapper")).not.toBeInTheDocument();
  });

  it("renders 'No history' empty state when given an empty array", () => {
    render(<MetricSparkline data={[]} />);
    expect(screen.getByText(/no history/i)).toBeInTheDocument();
  });

  it("still renders a chart for a single data point (no early-return for length=1)", () => {
    const { container } = render(<MetricSparkline data={[0.85]} />);
    // A single point should still produce a chart (Recharts handles len=1).
    expect(container.querySelector("svg")).toBeInTheDocument();
    expect(screen.queryByText(/no history/i)).not.toBeInTheDocument();
  });

  it("accepts a custom color via the `color` prop and uses it on the line stroke", () => {
    const { container } = render(<MetricSparkline data={[1, 2, 3]} color="#ff00ff" />);
    const linePath = container.querySelector(".recharts-line-curve");
    expect(linePath).toHaveAttribute("stroke", "#ff00ff");
  });
});
