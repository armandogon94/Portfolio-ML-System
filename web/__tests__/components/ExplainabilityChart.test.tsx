/** A.2.6: ExplainabilityChart component tests. */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { ExplainabilityChart, sortByAbsImportance } from "@/components/ExplainabilityChart";

describe("sortByAbsImportance helper", () => {
  it("sorts descending by absolute importance", () => {
    const input = {
      a: 0.1,
      b: -0.5,
      c: 0.3,
      d: -0.2,
    };
    const result = sortByAbsImportance(input);
    expect(result.map((r) => r.feature)).toEqual(["b", "c", "d", "a"]);
  });

  it("takes top N when provided", () => {
    const input = { a: 0.1, b: -0.5, c: 0.3, d: -0.2, e: 0.4 };
    const result = sortByAbsImportance(input, 2);
    expect(result).toHaveLength(2);
    expect(result.map((r) => r.feature)).toEqual(["b", "e"]);
  });

  it("preserves sign on importance", () => {
    const input = { a: -0.3, b: 0.5 };
    const result = sortByAbsImportance(input);
    expect(result.find((r) => r.feature === "a")?.importance).toBe(-0.3);
    expect(result.find((r) => r.feature === "b")?.importance).toBe(0.5);
  });

  it("returns empty array for empty input", () => {
    expect(sortByAbsImportance({})).toEqual([]);
  });
});

describe("ExplainabilityChart component", () => {
  const importances = {
    credit_score: -0.42,
    debt_to_income: 0.31,
    annual_income: -0.18,
    employment_years: -0.07,
  };

  it("renders the feature names as labels", () => {
    render(<ExplainabilityChart importances={importances} />);
    // Recharts can render axis tick text multiple times in jsdom; we
    // just assert presence via getAllByText.
    expect(screen.getAllByText("credit_score").length).toBeGreaterThan(0);
    expect(screen.getAllByText("debt_to_income").length).toBeGreaterThan(0);
  });

  it("respects topN by limiting rendered bars", () => {
    render(<ExplainabilityChart importances={importances} topN={2} />);
    expect(screen.getAllByText("credit_score").length).toBeGreaterThan(0);
    expect(screen.getAllByText("debt_to_income").length).toBeGreaterThan(0);
    expect(screen.queryAllByText("employment_years")).toHaveLength(0);
  });

  it("renders gracefully with empty importances", () => {
    const { container } = render(<ExplainabilityChart importances={{}} />);
    // Empty state shows a placeholder rather than an empty chart.
    expect(container.textContent?.toLowerCase()).toContain("no features");
  });
});
