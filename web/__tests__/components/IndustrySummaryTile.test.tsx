/** A.9.8 — IndustrySummaryTile component tests. */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { IndustrySummaryTile } from "@/components/IndustrySummaryTile";
import { getIndustry } from "@/lib/industries";
import type { DashboardRow } from "@/lib/dashboard";

const FINTECH = getIndustry("fintech")!;

function row(modelSlug: string, status: DashboardRow["status"], metric: number | null): DashboardRow {
  return {
    industrySlug: "fintech",
    industryTitle: "Fintech",
    modelSlug,
    modelTitle: modelSlug,
    modelDescription: "",
    status,
    href: status === "ready" ? `/fintech/${modelSlug}` : null,
    keyMetric:
      metric === null
        ? null
        : { name: "test_auc_roc", label: "AUC-ROC", value: metric, higherIsBetter: true },
    lastTrained: status === "ready" ? "2026-04-23T00:00:00" : null,
    history: [],
  };
}

describe("IndustrySummaryTile", () => {
  it("renders the industry title and a link to its index page", () => {
    const tile = render(
      <IndustrySummaryTile industry={FINTECH} rows={[row("credit-risk", "ready", 0.85)]} />,
    );
    // shadcn CardTitle renders as <div> — assert text presence instead of heading role.
    expect(tile.getByText("Fintech")).toBeInTheDocument();
    expect(tile.getByRole("link", { name: /fintech/i })).toHaveAttribute("href", "/fintech");
  });

  it("shows ready-count over total in the format 'N/M ready'", () => {
    const rows = [
      row("credit-risk", "ready", 0.85),
      row("fraud", "ready", 0.91),
      row("loan-approval", "not_built", null),
      row("churn", "ready", 0.79),
    ];
    render(<IndustrySummaryTile industry={FINTECH} rows={rows} />);
    // FINTECH has modelCount=4 total
    expect(screen.getByTestId("ready-count")).toHaveTextContent("3/4 ready");
  });

  it("shows the average key-metric across ready rows when ≥1 ready", () => {
    const rows = [
      row("credit-risk", "ready", 0.80),
      row("fraud", "ready", 0.90),
    ];
    render(<IndustrySummaryTile industry={FINTECH} rows={rows} />);
    // (0.80 + 0.90) / 2 = 0.85 → "85.0%"
    expect(screen.getByTestId("avg-metric")).toHaveTextContent(/85\.0%|0\.850|0\.85/);
  });

  it("renders a 'No ready models' note when zero rows are ready", () => {
    const rows = [
      row("credit-risk", "not_built", null),
      row("fraud", "not_built", null),
    ];
    render(<IndustrySummaryTile industry={FINTECH} rows={rows} />);
    expect(screen.getByText(/no ready models/i)).toBeInTheDocument();
  });
});
